#!/usr/bin/env python3
"""Two-stage locked evaluation on a fresh test cohort (round 2: Messidor-2).

``freeze``: label-free. Computes every predeclared variant's logits on the test
images, then writes a lock binding the image bytes, predictions, model files,
decision policies, code, protocol and the label file *hash* (never its values).

``evaluate``: verifies every lock binding, reads the labels once, and reports
model-only referable sensitivity/specificity at each variant's frozen policy,
threshold-free AUC and sensitivity at 85% specificity, QWK, per-grade recall,
exam-cluster bootstrap intervals and paired differences versus the reference.
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory():
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


def logits_for(spec, frame, batch=8):
    model = load_model(spec)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch):
            arrays = []
            for name in frame.image_id[start:start + batch]:
                data = np.frombuffer((MESSIDOR / 'images' / name).read_bytes(), np.uint8)
                rgb = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                arrays.append(preprocess_for_model(rgb).transpose(2, 0, 1))
            grade, referral = model(torch.from_numpy(np.stack(arrays)).float())
            outputs.append(torch.cat([grade, referral[:, None]], 1).numpy())
            if start % 200 == 0:
                print(f"{spec['id']}: {start}/{len(frame)}", flush=True)
    return np.concatenate(outputs).astype(np.float64)


def freeze(args):
    config = json.loads(args.variants.read_text())
    lock_path = args.output / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    frame = inventory()
    frame['sha256'] = [sha(MESSIDOR / 'images' / i) for i in frame.image_id]
    frame.to_csv(args.output / 'inventory.csv', index=False)
    torch.set_num_threads(args.threads)
    cache = {}
    for spec in config['variants']:
        if spec['kind'] == 'ensemble':
            continue
        path = args.output / f"logits_{spec['id']}.npy"
        if not path.exists():
            np.save(path, logits_for(spec, frame))
        cache[spec['id']] = path
    for spec in config['variants']:
        if spec['kind'] == 'ensemble':
            path = args.output / f"logits_{spec['id']}.npy"
            np.save(path, np.mean([np.load(cache[m]) for m in spec['members']], axis=0))
            cache[spec['id']] = path
    bindings = {str(p.relative_to(ROOT)): sha(p) for p in
                [args.variants, args.output / 'inventory.csv', Path(__file__), args.protocol,
                 ROOT / 'python/utils/preprocessing.py', ROOT / 'python/models/netrai_model.py',
                 ROOT / 'python/export_adapted_variant.py', LABELS, *cache.values()]
                + [ROOT / s['checkpoint'] for s in config['variants'] if s.get('checkpoint')]
                + [ROOT / s['policy'] for s in config['variants'] if s.get('policy')]}
    atomic_json(lock_path, {'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
                            'test': 'Messidor-2 (1748 images, 874 exams), Krause 2018 adjudicated grades',
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


def evaluate(args):
    lock = json.loads((args.output / 'lock.json').read_text())
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    report_path = args.output / 'report.json'
    if report_path.exists():
        raise FileExistsError('This lock has already been evaluated')
    config = json.loads(args.variants.read_text())
    frame = pd.read_csv(args.output / 'inventory.csv')
    labels = pd.read_csv(LABELS)  # first and only read of the reserved labels
    labels['image_id'] = labels.image_id.str.strip()
    frame = frame.merge(labels, on='image_id', how='left', validate='one_to_one')
    if frame.adjudicated_gradable.isna().any():
        raise ValueError('Label coverage incomplete')
    gradable = (frame.adjudicated_gradable == 1).to_numpy()
    grade = frame.adjudicated_dr_grade.to_numpy()[gradable].astype(int)
    exams = frame.exam_id.to_numpy()[gradable]
    unique, index = np.unique(exams, return_inverse=True)
    rng = np.random.default_rng(lock['bootstrap_seed'])
    draws = rng.integers(len(unique), size=(lock['bootstrap_replicates'], len(unique)))
    weights = np.stack([np.bincount(d, minlength=len(unique)) for d in draws])[:, index]
    truth = grade >= 2
    results, positives = {}, {}
    for spec in config['variants']:
        logits = np.load(args.output / f"logits_{spec['id']}.npy")[gradable]
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
    report = {'schema_version': 1, 'lock_sha256': sha(args.output / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(),
              'population': {'images': len(frame), 'gradable': int(gradable.sum()),
                             'ungradable': int((~gradable).sum()), 'exams_gradable': int(len(unique))},
              'primary_variant': lock['primary_variant'], 'reference_variant': reference,
              'results': results, 'comparisons': comparisons,
              'exposure': 'Messidor-2 adjudicated labels are exposed after this evaluation.'}
    atomic_json(report_path, report)
    for name, result in results.items():
        print(name, {k: result[k] for k in ('sensitivity', 'sensitivity_ci95', 'specificity', 'specificity_ci95',
                                            'auc_expected_grade_score', 'sensitivity_at_85_specificity', 'qwk')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'evaluate'))
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / '.project-notes/docs/MODEL_IMPROVEMENT_PLAN_v2.md')
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    (freeze if args.stage == 'freeze' else evaluate)(args)


if __name__ == '__main__':
    main()
