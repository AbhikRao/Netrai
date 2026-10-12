#!/usr/bin/env python3
"""Two-stage locked evaluation on fresh test cohorts (round 2: Messidor-2/DDR; round 3: ODIR/EyeQ).

``freeze``: label-free. Computes every predeclared variant's logits on the test
images, then writes a lock binding the image bytes, predictions, model files,
decision policies, code, protocol and the label file *hash* (never its values).

``evaluate``: verifies every lock binding, reads the labels once, and reports
model-only referable sensitivity/specificity at each variant's frozen policy,
threshold-free AUC and sensitivity at 85% specificity, QWK, per-grade recall,
exam-cluster bootstrap intervals and paired differences versus the reference.

Logits already present in the output directory (e.g. computed on another device)
are reused only after a CPU spot check of their first images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from export_adapted_variant import adapted_model
from models.netrai_model import NetrAIModel
from utils.artifacts import atomic_json
from utils.metrics import quadratic_weighted_kappa
from utils.preprocessing import preprocess_for_model

ROOT = Path(__file__).resolve().parents[1]
MESSIDOR = ROOT / 'data/messidor2'
LABELS = MESSIDOR / 'adjudicated_labels_download/messidor_data.csv'
DDR = ROOT / 'data/DDR/extracted/DDR-dataset/DR_grading'
DDR_LABELS = DDR / 'test.txt'
ODIR = ROOT / 'data/odir/extracted/ODIR-5K/ODIR-5K'
ODIR_LABELS = ODIR / 'data.xlsx'
EYEQ = ROOT / 'data/eyeq'
COHORTS = {
    'messidor2': {'images': MESSIDOR / 'images', 'labels': LABELS, 'cluster': 'exam (left/right pair)'},
    'ddr_test': {'images': DDR / 'test', 'labels': DDR_LABELS, 'cluster': 'image (no patient ids released)'},
}
ROUND3_COHORTS = {
    'odir': {'images': ODIR / 'Training Images', 'labels': ODIR_LABELS, 'cluster': 'patient (ODIR ID)'},
    'eyeq_remainder': {'images': EYEQ / 'images/test', 'labels': EYEQ / 'data/Label_EyeQ_test.csv',
                       'cluster': 'patient (EyePACS filename prefix)'},
}
ROUNDS = {
    2: {'cohorts': COHORTS, 'labels': [LABELS, DDR_LABELS],
        'protocol': ROOT / '.project-notes/docs/MODEL_IMPROVEMENT_PLAN_v2.md',
        'tests': {'messidor2': 'primary: 1748 images / 874 exams, Krause 2018 adjudicated grades',
                  'ddr_test': 'secondary: DDR DR_grading test split, 4105 images, grade 5 = ungradable'},
        'exposure': 'Messidor-2 adjudicated and DDR test labels are exposed after this evaluation.'},
    3: {'cohorts': ROUND3_COHORTS, 'labels': [ODIR_LABELS, EYEQ / 'data/Label_EyeQ_test.csv'],
        'protocol': ROOT / '.project-notes/docs/MODEL_IMPROVEMENT_PLAN_v3.md',
        'tests': {'odir': 'primary: ODIR-5K training release (Kaggle mirror), 6784 images / 3392 patients after '
                          'DDR-overlap exclusion; keyword-mapped ICDR grades',
                  'eyeq_remainder': 'secondary: EyeQ/EyePACS test remainder, predeclared 4000-image patient subset'},
        'exposure': 'ODIR-5K keyword labels and the EyeQ test-remainder DR grades are exposed after this evaluation.'},
}
SPOT_CHECK_IMAGES, SPOT_CHECK_TOLERANCE = 8, 5e-3
QUALITY_TERMS = ('low image quality', 'lens dust', 'optic disk photographically invisible', 'image offset',
                 'no fundus image', 'image quality')


def odir_grade(keywords):
    """Predeclared ODIR keyword -> ICDR grade mapping (None = not usable for the DR endpoint)."""
    import re
    terms = [t.strip().lower() for t in re.split('[,，;]', str(keywords)) if t.strip()]
    if any(any(q in t for q in QUALITY_TERMS) for t in terms):
        return 'ungradable'
    grades = []
    for t in terms:
        if 'hypertensive' in t:
            continue
        if re.search(r'severe\s*non[\s-]*proliferative', t):
            grades.append(3)
        elif re.search(r'moderate\s*non[\s-]*proliferative', t):
            grades.append(2)
        elif re.search(r'mild\s*non[\s-]*proliferative', t):
            grades.append(1)
        elif re.search(r'proliferative', t) and 'non' not in t:
            grades.append(4)
        elif 'diabetic retinopathy' in t:
            grades.append('unspecified')
    if 'unspecified' in grades:
        return 'unspecified'
    if grades:
        return max(grades)
    if any(t == 'normal fundus' for t in terms):
        return 0
    return 'other_disease_only'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(cohort='messidor2'):
    if cohort == 'odir':
        excluded = set(pd.read_csv(ROOT / 'results/model_improvement/odir_excluded_images.csv').image)
        images = sorted(p.name for p in (ODIR / 'Training Images').glob('*.jpg') if p.name not in excluded)
        return pd.DataFrame({'image_id': images, 'exam_id': ['odir_' + i.split('_')[0] for i in images]})
    if cohort == 'eyeq_remainder':
        labels = pd.read_csv(EYEQ / 'data/Label_EyeQ_test.csv', usecols=['image', 'quality'])
        labels = labels.loc[labels.quality.isin([0, 1])]
        from extract_backbone_cache import cohort as cached
        used = set(cached('eyeq_test', 3000, 2026).image_id)  # round-2 selection subset
        overlap = ROOT / 'results/verification_2026-09-24/perceptual_overlap/visually_confirmed_pairs.csv'
        used |= set(pd.read_csv(overlap).image_b)
        labels = labels.loc[~labels.image.isin(used)]
        labels['patient'] = labels.image.str.split('_').str[0]
        patients = labels.patient.drop_duplicates().sample(frac=1, random_state=20261012)
        counts, chosen, total = labels.groupby('patient').size(), [], 0
        for patient in patients:  # predeclared 4,000-image patient subset bounds compute
            if total >= 4000:
                break
            chosen.append(patient); total += int(counts[patient])
        labels = labels.loc[labels.patient.isin(chosen)].sort_values('image')
        names = []
        for image in labels.image:
            stem = Path(image).stem
            found = next(p for p in (EYEQ / f'images/test/{image}', EYEQ / f'images/test/{stem}.png',
                                     EYEQ / f'images/test/{stem}.jpg') if p.is_file())
            names.append(found.name)
        return pd.DataFrame({'image_id': names, 'label_image': labels.image.values,
                             'exam_id': ('eyeq_' + labels.patient).values})
    if cohort == 'ddr_test':
        images = sorted(p.name for p in DDR.joinpath('test').iterdir())
        return pd.DataFrame({'image_id': images, 'exam_id': images})
    pairs = pd.read_csv(MESSIDOR / 'messidor-2.csv', sep=';')
    pairs.columns = [c.strip() for c in pairs.columns]
    exam = {}
    for index, row in pairs.iterrows():
        exam[row['left'].strip()] = exam[row['right'].strip()] = f'exam_{index:04d}'
    images = sorted(p.name for p in (MESSIDOR / 'images').iterdir())
    if set(images) != set(exam):
        raise ValueError('Messidor-2 images and exam pairs disagree')
    return pd.DataFrame({'image_id': images, 'exam_id': [exam[i] for i in images]})


def load_model(spec):
    if spec['kind'] == 'legacy':
        return adapted_model(None)
    if spec['kind'] == 'rung':
        return adapted_model(ROOT / spec['checkpoint'])
    model = NetrAIModel('efficientnet_b4', 5, pretrained=False)
    state = torch.load(ROOT / spec['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model_state_dict', state))
    return model.eval()


def logits_for(spec, frame, batch=8, folder=MESSIDOR / 'images'):
    size = int(spec.get('size', 512))
    model = load_model(spec)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch):
            arrays = []
            for name in frame.image_id[start:start + batch]:
                data = np.frombuffer((folder / name).read_bytes(), np.uint8)
                rgb = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                arrays.append(preprocess_for_model(rgb, size=size).transpose(2, 0, 1))
            grade, referral = model(torch.from_numpy(np.stack(arrays)).float())
            outputs.append(torch.cat([grade, referral[:, None]], 1).numpy())
            if start % 200 == 0:
                print(f"{spec['id']} {folder.parent.name}/{folder.name}: {start}/{len(frame)}", flush=True)
    return np.concatenate(outputs).astype(np.float64)


def check_matlab_variant(config):
    """The exported MATLAB primary must carry exactly the locked primary policy."""
    spec = next(v for v in config['variants'] if v['id'] == config['primary'])
    policy = json.loads((ROOT / spec['policy']).read_text())
    variant = ROOT / config['matlab_primary_variant']
    matlab = json.loads((variant / 'netrai_config.json').read_text())
    if not (np.allclose(matlab['grade_thresholds'], policy['thresholds'], rtol=0, atol=1e-12)
            and abs(matlab['referable_head_threshold'] - policy['head_threshold']) < 1e-12):
        raise ValueError('MATLAB primary variant thresholds differ from the locked policy')
    if int(matlab['input_size']) != int(spec.get('size', 512)):
        raise ValueError('MATLAB primary variant input size differs from the locked variant')
    return [variant / 'netrai_fold0.onnx', variant / 'netrai_config.json', variant / 'netrai_bundle.json']


def freeze(args):
    config = json.loads(args.variants.read_text())
    matlab_files = check_matlab_variant(config) if config.get('matlab_primary_variant') else []
    lock_path = args.output / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    setup = ROUNDS[args.round]
    cache, inventories, spot_checks = {}, [], {}
    for cohort, info in setup['cohorts'].items():
        frame = inventory(cohort)
        frame['sha256'] = [sha(info['images'] / i) for i in frame.image_id]
        inv = args.output / f'inventory_{cohort}.csv'
        frame.to_csv(inv, index=False)
        inventories.append(inv)
        for spec in config['variants']:
            if spec['kind'] in ('ensemble', 'policy_of'):
                continue
            path = args.output / f"logits_{cohort}_{spec['id']}.npy"
            if not path.exists():
                np.save(path, logits_for(spec, frame, folder=info['images']))
            else:  # precomputed elsewhere: shape and CPU spot check before binding
                saved = np.load(path)
                if saved.shape != (len(frame), 6):
                    raise ValueError(f'{path.name}: shape {saved.shape} != ({len(frame)}, 6)')
                rows = np.unique(np.linspace(0, len(frame) - 1, SPOT_CHECK_IMAGES).round().astype(int))
                head = frame.iloc[rows].reset_index(drop=True)  # spread over the file, includes the last row
                error = float(np.abs(logits_for(spec, head, folder=info['images']) - saved[rows]).max())
                if error > SPOT_CHECK_TOLERANCE:
                    raise ValueError(f'{path.name}: CPU spot check differs by {error}')
                spot_checks[path.name] = error
            cache[(cohort, spec['id'])] = path
        for spec in config['variants']:
            if spec['kind'] == 'ensemble':
                path = args.output / f"logits_{cohort}_{spec['id']}.npy"
                np.save(path, np.mean([np.load(cache[(cohort, m)]) for m in spec['members']], axis=0))
                cache[(cohort, spec['id'])] = path
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in
                [args.variants, *inventories, Path(__file__), args.protocol,
                 ROOT / 'python/utils/preprocessing.py', ROOT / 'python/models/netrai_model.py',
                 ROOT / 'python/export_adapted_variant.py', ROOT / 'python/utils/metrics.py',
                 ROOT / 'python/weights/fold0_thresholds.npy', *setup['labels'], *matlab_files, *cache.values()]
                + [ROOT / s['checkpoint'] for s in config['variants'] if s.get('checkpoint')]
                + [ROOT / s['policy'] for s in config['variants'] if s.get('policy')]}
    atomic_json(lock_path, {'schema_version': 2 if args.round == 3 else 1, 'round': args.round,
                            'locked_utc': datetime.now(timezone.utc).isoformat(), 'tests': setup['tests'],
                            'precomputed_logits_cpu_spot_check_max_abs': spot_checks,
                            'logits_provenance': config.get('logits_provenance'),
                            'labels_read': False, 'primary_variant': config['primary'],
                            'reference_variant': config['reference'], 'bootstrap_seed': config['bootstrap_seed'],
                            'bootstrap_replicates': config['bootstrap_replicates'], 'bindings': bindings})
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def thresholds_for(spec):
    if spec.get('policy'):
        policy = json.loads((ROOT / spec['policy']).read_text())
        return np.asarray(policy['thresholds'], float), float(policy.get('head_threshold', 0.5))
    return np.load(ROOT / 'python/weights/fold0_thresholds.npy').astype(float), 0.5


def metrics(logits, grade, thresholds, head_threshold):
    shifted = logits[:, :5] - logits[:, :5].max(1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    score = probability @ np.arange(5)
    predicted = (score[:, None] >= thresholds[None]).sum(1)
    truth, positive = grade >= 2, predicted >= 2
    order = np.argsort(-score)
    tpr = np.cumsum(truth[order]) / truth.sum()
    fpr = np.cumsum(~truth[order]) / (~truth).sum()
    ranks = pd.Series(score).rank().to_numpy()
    auc = (ranks[truth].sum() - truth.sum() * (truth.sum() + 1) / 2) / (truth.sum() * (~truth).sum())
    head = 1 / (1 + np.exp(-logits[:, 5])) >= head_threshold
    return {'images': int(len(grade)), 'referable': int(truth.sum()),
            'tp': int((truth & positive).sum()), 'fn': int((truth & ~positive).sum()),
            'tn': int((~truth & ~positive).sum()), 'fp': int((~truth & positive).sum()),
            'sensitivity': float(positive[truth].mean()), 'specificity': float((~positive[~truth]).mean()),
            'auc_expected_grade_score': float(auc),
            'sensitivity_at_85_specificity': float(tpr[(1 - fpr) >= 0.85].max()),
            'qwk': float(quadratic_weighted_kappa(grade, predicted)),
            'per_grade_recall': [float(np.mean(predicted[grade == g] == g)) for g in range(5)],
            'per_grade_support': [int(np.sum(grade == g)) for g in range(5)],
            'head_disagreement_rate': float(np.mean(head != positive))}, positive


def cohort_labels(cohort, frame):
    """First and only read of a reserved label file, after lock verification."""
    if cohort == 'odir':
        table = pd.read_excel(ODIR_LABELS)
        per_eye = {}
        for row in table.itertuples(index=False):
            per_eye[str(row[3]).strip()] = odir_grade(row[5])  # Left-Fundus, Left-Diagnostic Keywords
            per_eye[str(row[4]).strip()] = odir_grade(row[6])  # Right-Fundus, Right-Diagnostic Keywords
        mapped = frame.image_id.map(per_eye)
        if mapped.isna().any():
            raise ValueError('ODIR label coverage incomplete')
        gradable = mapped.map(lambda g: isinstance(g, (int, np.integer))).to_numpy()
        grade = np.array([g if isinstance(g, (int, np.integer)) else -1 for g in mapped], dtype=float)
        frame = frame.assign(odir_status=mapped.astype(str))
        return frame, gradable, grade
    if cohort == 'eyeq_remainder':
        labels = pd.read_csv(EYEQ / 'data/Label_EyeQ_test.csv', usecols=['image', 'DR_grade'])
        frame = frame.merge(labels.rename(columns={'image': 'label_image'}), on='label_image', how='left',
                            validate='one_to_one')
        if frame.DR_grade.isna().any():
            raise ValueError('EyeQ label coverage incomplete')
        return frame, np.ones(len(frame), bool), frame.DR_grade.to_numpy(float)
    if cohort == 'messidor2':
        labels = pd.read_csv(LABELS)
        # The release lists 'IM000012.jpg' for files named 'IM000012.JPG': join case-insensitively.
        labels['key'] = labels.image_id.str.strip().str.lower()
        frame = frame.assign(key=frame.image_id.str.lower()).merge(
            labels.drop(columns='image_id'), on='key', how='left', validate='one_to_one')
        if frame.adjudicated_gradable.isna().any():
            raise ValueError('Messidor-2 label coverage incomplete')
        gradable = (frame.adjudicated_gradable == 1).to_numpy()
        grade = frame.adjudicated_dr_grade.to_numpy()
    else:
        labels = pd.read_csv(DDR_LABELS, sep=' ', header=None, names=['image_id', 'grade'])
        frame = frame.merge(labels, on='image_id', how='left', validate='one_to_one')
        if frame.grade.isna().any():
            raise ValueError('DDR label coverage incomplete')
        grade = frame.grade.to_numpy()
        gradable = grade <= 4  # DDR grade 5 = ungradable
    return frame, gradable, grade


def evaluate_cohort(cohort, args, config, lock, cohorts=COHORTS):
    frame = pd.read_csv(args.output / f'inventory_{cohort}.csv')
    frame, gradable, grade_all = cohort_labels(cohort, frame)
    grade = grade_all[gradable].astype(int)
    clusters = frame.exam_id.to_numpy()[gradable]
    unique, index = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(lock['bootstrap_seed'])
    draws = rng.integers(len(unique), size=(lock['bootstrap_replicates'], len(unique)))
    weights = np.stack([np.bincount(d, minlength=len(unique)) for d in draws])[:, index]
    truth = grade >= 2
    results, positives = {}, {}
    for spec in config['variants']:
        source_id = spec.get('logits_from', spec['id'])  # secondary operating point shares logits
        logits = np.load(args.output / f"logits_{cohort}_{source_id}.npy")[gradable]
        thresholds, head = thresholds_for(spec)
        result, positive = metrics(logits, grade, thresholds, head)
        sens = (weights * (truth & positive)).sum(1) / (weights * truth).sum(1)
        spec_b = (weights * (~truth & ~positive)).sum(1) / (weights * ~truth).sum(1)
        result['sensitivity_ci95'] = np.percentile(sens, [2.5, 97.5]).tolist()
        result['specificity_ci95'] = np.percentile(spec_b, [2.5, 97.5]).tolist()
        results[spec['id']], positives[spec['id']] = result, positive
    reference = lock['reference_variant']
    comparisons = {}
    for name, positive in positives.items():
        if name == reference:
            continue
        base = positives[reference]
        d_sens = ((weights * (truth & positive)).sum(1) - (weights * (truth & base)).sum(1)) / (weights * truth).sum(1)
        d_spec = ((weights * (~truth & ~positive)).sum(1) - (weights * (~truth & ~base)).sum(1)) / (weights * ~truth).sum(1)
        comparisons[f'{name}_vs_{reference}'] = {
            'sensitivity_difference': float(positive[truth].mean() - base[truth].mean()),
            'sensitivity_difference_ci95': np.percentile(d_sens, [2.5, 97.5]).tolist(),
            'specificity_difference': float((~positive[~truth]).mean() - (~base[~truth]).mean()),
            'specificity_difference_ci95': np.percentile(d_spec, [2.5, 97.5]).tolist()}
    population = {'images': int(len(frame)), 'gradable': int(gradable.sum()),
                  'ungradable': int((~gradable).sum()), 'clusters_gradable': int(len(unique))}
    if 'odir_status' in frame:  # 'ungradable' above = every eye outside the DR endpoint
        population['not_in_dr_endpoint_by_status'] = {str(k): int(v) for k, v in
                                                     frame.odir_status[~gradable].value_counts().items()}
    return {'cluster_unit': cohorts[cohort]['cluster'], 'population': population,
            'results': results, 'comparisons': comparisons}


def evaluate(args):
    lock = json.loads((args.output / 'lock.json').read_text())
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    report_path = args.output / 'report.json'
    if report_path.exists():
        raise FileExistsError('This lock has already been evaluated')
    config = json.loads(args.variants.read_text())
    setup = ROUNDS[lock.get('round', 2)]
    cohorts = {cohort: evaluate_cohort(cohort, args, config, lock, setup['cohorts']) for cohort in setup['cohorts']}
    report = {'schema_version': 2, 'lock_sha256': sha(args.output / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(),
              'primary_variant': lock['primary_variant'], 'reference_variant': lock['reference_variant'],
              'cohorts': cohorts,
              'exposure': setup['exposure']}
    atomic_json(report_path, report)
    for cohort, block in cohorts.items():
        for name, result in block['results'].items():
            print(cohort, name, {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v])
                                 for k, v in result.items() if k in ('sensitivity', 'sensitivity_ci95', 'specificity',
                                 'specificity_ci95', 'auc_expected_grade_score', 'sensitivity_at_85_specificity', 'qwk')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'evaluate'))
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--round', type=int, choices=sorted(ROUNDS), default=2,
                        help='freeze only; evaluate follows the round recorded in the lock')
    parser.add_argument('--protocol', type=Path, help='default: the round plan')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    args.protocol = args.protocol or ROUNDS[args.round]['protocol']
    args.variants, args.output, args.protocol = args.variants.resolve(), args.output.resolve(), args.protocol.resolve()
    (freeze if args.stage == 'freeze' else evaluate)(args)


if __name__ == '__main__':
    main()
