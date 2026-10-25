#!/usr/bin/env python3
"""Locked, single-use confirmation test of frozen grading models on BRSET 1.0.2 (plan v5 Addendum 4).

Same two-stage pattern as ``locked_mbrset.py`` (``logits`` / ``quality`` / ``freeze`` / ``evaluate``). Differences:
  * file names carry no patient id: bootstrap clusters come from ``patients.csv`` (image_id, patient_id), written before the
    lock from those two label columns only and bound into the lock;
  * primary population: images with a ``DR_ICDR`` grade and ``quality`` == 'Adequate' (the four factor columns never exclude);
  * secondary: any-DR AUC; specificity among non-referable eyes with / without another labelled disease; per-camera results;
    quality AUC (``quality`` == 'Inadequate' and 'any factor unsatisfactory') - ranking only.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exposure
import locked_mbrset as lm
from locked_evaluation import ROOT, logits_for, sha, thresholds_for
from locked_rfmid import logits_on, temperature_for
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m
from utils.artifacts import atomic_json

COHORT = 'brset'
DATA = ROOT / 'data/brset/fundus_photos'
IDENTITY = ROOT / 'results/model_improvement/brset_identity'
PATIENTS = IDENTITY / 'patients.csv'
EXCLUSIONS = IDENTITY / 'exclusions_v1.txt'
PROTOCOL = ROOT / '.project-notes/docs/MODEL_IMPROVEMENT_PLAN_v5.md'
X9_PARAMETERS = lm.X9_PARAMETERS
CODE = ['python/locked_brset.py', 'python/locked_mbrset.py', 'python/locked_evaluation.py', 'python/locked_rfmid.py', 'python/exposure.py',
        'python/utils/preprocessing.py', 'python/models/netrai_model.py', 'python/export_adapted_variant.py',
        'python/netrai_eval/metrics.py', 'python/netrai_eval/bootstrap.py', 'python/train_quality_x9.py',
        'python/models/quality_model.py', 'python/modules/quality.py']
LEGACY_TEMPERATURE = lm.LEGACY_TEMPERATURE
SPOT_CHECK_IMAGES, SPOT_CHECK_TOLERANCE = lm.SPOT_CHECK_IMAGES, lm.SPOT_CHECK_TOLERANCE
CHUNK = lm.CHUNK
FACTORS = ('focus', 'illumination', 'image_field', 'artifacts')
# 'other disease' = any labelled non-DR condition (DR itself, macular edema and haemorrhage are DR-related and left out)
OTHER_DISEASE = ('scar', 'nevus', 'amd', 'vascular_occlusion', 'hypertensive_retinopathy', 'drusen', 'retinal_detachment',
                 'myopic_fundus', 'increased_cup_disc', 'other')
POSITIVE = {'1', '1.0', 'yes', 'true'}


def inventory(data: Path = DATA, exclusions: Path | None = EXCLUSIONS, patients: Path | None = PATIENTS) -> pd.DataFrame:
    """Every image (file name, patient id, cluster = patient, excluded flag). Patient ids come from ``patients.csv``."""
    files = sorted(p.name for p in data.iterdir() if p.suffix.lower() == '.jpg')
    frame = pd.DataFrame({'image_id': files})
    if patients is not None:
        table = pd.read_csv(patients, dtype=str)
        mapping = dict(zip(table.image_id, table.patient_id))
        missing = sorted(set(files) - set(mapping))
        if missing:
            raise ValueError(f'Images without a patient id: {missing[:5]}')
        frame['patient'] = [mapping[f] for f in files]
    else:
        frame['patient'] = ''
    frame['cluster'] = frame.patient
    excluded = set()
    if exclusions is not None and Path(exclusions).is_file():
        excluded = {line.strip() for line in Path(exclusions).read_text().splitlines() if line.strip() and not line.startswith('#')}
    unknown = excluded - set(files)
    if unknown:
        raise ValueError(f'Exclusions name unknown images: {sorted(unknown)[:5]}')
    frame['excluded'] = frame.image_id.isin(excluded)
    return frame


def label_file() -> Path:
    return exposure._resolved(exposure.entry(COHORT)['label_paths'][0])


def _positive(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin(POSITIVE).to_numpy()


def read_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the label CSV by image id. Callers must have passed ``assert_label_read_allowed`` first."""
    table = pd.read_csv(label_file(), dtype=str, keep_default_na=False)
    table.columns = [c.strip() for c in table.columns]
    table['file'] = table.image_id.str.strip() + '.jpg'
    if table.file.duplicated().any():
        raise ValueError('The label file repeats an image')
    quality = table.quality.str.strip()
    if not set(quality) <= {'Adequate', 'Inadequate'}:
        raise ValueError(f'Unexpected quality spellings: {sorted(set(quality) - {"Adequate", "Inadequate"})}')
    icdr = table.DR_ICDR.str.strip().str.replace(r'\.0$', '', regex=True)
    if not set(icdr) <= {'', '0', '1', '2', '3', '4'}:
        raise ValueError(f'Unexpected DR_ICDR spellings: {sorted(set(icdr) - {"", "0", "1", "2", "3", "4"})}')
    table = table.assign(_quality=quality, _icdr=icdr).set_index('file')
    out = frame.copy()
    joined = table.reindex(out.image_id)
    out['has_row'] = joined._quality.notna().to_numpy()
    out['gradable'] = (joined._quality == 'Adequate').to_numpy()
    out['graded'] = (joined._icdr.fillna('') != '').to_numpy()
    out['icdr'] = [int(v) if isinstance(v, str) and v != '' else -1 for v in joined._icdr]
    out['camera'] = joined.camera.fillna('').str.strip().to_numpy()
    for column in OTHER_DISEASE:
        out[f'od_{column}'] = _positive(joined[column]) if column in joined else False
    out['other_disease'] = out[[f'od_{c}' for c in OTHER_DISEASE]].any(axis=1)
    unsat = np.zeros(len(out), bool)
    spellings = {}
    for factor in FACTORS:
        values = joined[factor].fillna('').astype(str).str.strip() if factor in joined else pd.Series([''] * len(out))
        spellings[factor] = sorted(set(values))
        unsat |= values.str.lower().str.startswith('unsat').to_numpy()
    out['any_factor_unsatisfactory'] = unsat
    return out, spellings


def other_disease_stats(grade: np.ndarray, logits: np.ndarray, thresholds: np.ndarray, other: np.ndarray) -> dict:
    positive = m.decode_thresholds(m.expected_grade_score(logits), thresholds) >= 2
    neg = grade < 2
    with_other, without = neg & other, neg & ~other
    return {'specificity_with_other_disease': float((~positive[with_other]).mean()) if with_other.any() else float('nan'),
            'specificity_without_other_disease': float((~positive[without]).mean()) if without.any() else float('nan')}


def logits_command(args) -> None:
    torch.set_num_threads(args.threads)
    config = json.loads(args.variants.read_text())
    frame = inventory(args.data, None, None)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    frame[['image_id']].to_csv(args.out / 'logits_order.csv', index=False)
    for spec in config['variants']:
        if spec['id'] not in args.ids or spec['kind'] in ('ensemble', 'policy_of'):
            continue
        target = args.out / f"logits_{spec['id']}.npy"
        if target.exists():
            continue
        parts = args.out / f"parts_{spec['id']}"
        parts.mkdir(exist_ok=True)
        for start in range(0, len(frame), CHUNK):
            part = parts / f'{start:06d}.npy'
            if not part.exists():
                chunk = frame.iloc[start:start + CHUNK].reset_index(drop=True)
                np.save(part, logits_on(spec, chunk, args.data, device))
                print(f"{spec['id']}: {min(start + CHUNK, len(frame))}/{len(frame)}", flush=True)
        np.save(target, np.concatenate([np.load(parts / f'{s:06d}.npy') for s in range(0, len(frame), CHUNK)]))
        shutil.rmtree(parts)


def quality_command(args) -> None:
    """Quality scores on every image, refit exactly as in the Challenge-2 lock; the refit threshold must match it."""
    import cv2
    import train_quality_x9 as x9
    from modules.quality import assess_image_quality
    frame = inventory(args.data, None, None)
    locked = json.loads(X9_PARAMETERS.read_text())
    train = x9.load_split('training')
    xt = np.load(x9.ROOT / 'results/model_improvement/experiments/X9/embeddings_dev.npz')['train']
    qmodel = x9.load_candidate()
    predict, info = x9.fit_pipeline(xt, train, qmodel, float(locked['temperature']))
    if abs(info['operating_threshold_5pct_false_reject_oof'] - locked['operating_threshold_5pct_false_reject_oof']) > 1e-9:
        raise ValueError('The X9 refit does not reproduce the locked operating threshold')
    paths = [args.data / f for f in frame.image_id]
    scores = predict(x9.embed(paths, qmodel))
    iqs = np.array([assess_image_quality(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))[0] for p in paths])
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / 'quality_scores.npz', image_id=frame.image_id.to_numpy(), eyeq=scores['eyeq_reject_logit'],
             final=scores['final'], iqs=iqs, **{f'head_{k}': v for k, v in scores['heads'].items()})


def freeze(args) -> None:
    config = json.loads(args.variants.read_text())
    lock_path = args.output / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    exposure.assert_fresh_allowed(COHORT)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = inventory(args.data, args.exclusions, args.patients)
    frame['sha256'] = [sha(args.data / i) for i in frame.image_id]
    inv = args.output / 'inventory_brset.csv'
    frame.to_csv(inv, index=False)
    cache, spot = {}, {}
    for spec in config['variants']:
        if spec['kind'] in ('ensemble', 'policy_of'):
            continue
        path = args.output / f"logits_{spec['id']}.npy"
        pre = args.precomputed / f"logits_{spec['id']}.npy" if args.precomputed else None
        if not path.exists() and pre is not None and pre.exists():
            order = pd.read_csv(args.precomputed / 'logits_order.csv').image_id.tolist()
            if order != frame.image_id.tolist():
                raise ValueError(f'{pre.name}: row order differs from the inventory')
            saved = np.load(pre)
            if saved.shape != (len(frame), 6) or not np.isfinite(saved).all():
                raise ValueError(f'{pre.name}: shape {saved.shape} or non-finite values')
            rows = np.unique(np.linspace(0, len(frame) - 1, SPOT_CHECK_IMAGES).round().astype(int))
            head = frame.iloc[rows].reset_index(drop=True)
            error = float(np.abs(logits_for(spec, head, folder=args.data) - saved[rows]).max())
            if error > SPOT_CHECK_TOLERANCE:
                raise ValueError(f'{pre.name}: CPU spot check differs by {error}')
            shutil.copy2(pre, path)
            spot[spec['id']] = error
        if not path.exists():
            np.save(path, logits_for(spec, frame, folder=args.data))
        cache[spec['id']] = path
    for spec in config['variants']:
        if spec['kind'] == 'ensemble':
            path = args.output / f"logits_{spec['id']}.npy"
            np.save(path, np.mean([np.load(cache[mid]) for mid in spec['members']], axis=0))
            cache[spec['id']] = path
    quality = args.output / 'quality_scores.npz'
    source = args.precomputed / 'quality_scores.npz' if args.precomputed else None
    if source is None or not source.exists():
        raise FileNotFoundError('quality_scores.npz is required (run the quality command first)')
    if list(np.load(source, allow_pickle=True)['image_id']) != frame.image_id.tolist():
        raise ValueError('quality scores: row order differs from the inventory')
    shutil.copy2(source, quality)
    files = [args.variants, inv, args.protocol, LEGACY_TEMPERATURE, X9_PARAMETERS, quality, label_file(), args.patients,
             *[ROOT / c for c in CODE]]
    files += [p for p in (args.exclusions,) if p is not None and Path(p).is_file()]
    files += list(cache.values())
    files += [ROOT / s['checkpoint'] for s in config['variants'] if s.get('checkpoint')]
    files += [ROOT / s['policy'] for s in config['variants'] if s.get('policy')]
    bindings = {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in files}  # label file: hash only
    atomic_json(lock_path, {
        'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
        'tests': {COHORT: f'{len(frame)} images, {int(frame.excluded.sum())} identity-excluded; referable-DR endpoint (DR_ICDR >= 2)'},
        'fresh': True, 'labels_read': False, 'primary_variant': config['primary'], 'references': config['references'],
        'bootstrap_seed': config['bootstrap_seed'], 'bootstrap_replicates': config['bootstrap_replicates'],
        'logits_provenance': config.get('logits_provenance'), 'selection_note': config.get('selection_note'),
        'precomputed_logits_cpu_spot_check_max_abs': spot,
        'quality_operating_threshold': json.loads(X9_PARAMETERS.read_text())['operating_threshold_5pct_false_reject_oof'],
        'bindings': bindings})
    exposure.record_locked(COHORT, lock_path)
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def evaluate(args) -> None:
    lock = json.loads((args.output / 'lock.json').read_text())
    exposure.assert_label_read_allowed(COHORT, args.output)
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    config = json.loads(args.variants.read_text())
    frame = pd.read_csv(args.output / 'inventory_brset.csv', dtype={'patient': str, 'cluster': str})
    everything, spellings = read_labels(frame)
    excluded = everything.excluded.to_numpy(bool)
    primary = everything.has_row.to_numpy() & everything.gradable.to_numpy() & everything.graded.to_numpy() & ~excluded
    d = everything[primary].reset_index(drop=True)
    grade, clusters = d.icdr.to_numpy(int), d.cluster.to_numpy()
    other, camera = d.other_disease.to_numpy(bool), d.camera.to_numpy()
    seed, reps = lock['bootstrap_seed'], lock['bootstrap_replicates']
    logits = {s['id']: np.load(args.output / f"logits_{s.get('logits_from', s['id'])}.npy")[primary] for s in config['variants']}
    thresholds = {s['id']: thresholds_for(s)[0] for s in config['variants']}
    results, summaries, any_dr, other_stats = {}, {}, {}, {}
    for name in logits:
        results[name] = bs.bootstrap(lambda rows, n=name: lm.referable_stats(grade[rows], logits[n][rows], thresholds[n]), clusters, reps, seed)
        summaries[name] = lm.grade_summary(grade, logits[name], thresholds[name])
        spec = next(s for s in config['variants'] if s['id'] == name)
        score = lm.any_dr_score(logits[name], temperature_for(spec))
        any_dr[name] = bs.bootstrap(lambda rows, sc=score: {'auc_any_dr': m.auc(grade[rows] >= 1, sc[rows])}, clusters, reps, seed)['auc_any_dr']
        other_stats[name] = bs.bootstrap(lambda rows, n=name: other_disease_stats(grade[rows], logits[n][rows], thresholds[n], other[rows]), clusters, reps, seed)
    comparisons = {}
    for reference in lock['references']:
        for name in logits:
            if name == reference:
                continue
            comparisons[f'{name}_vs_{reference}'] = bs.paired_bootstrap(
                lambda rows, n=name: lm.referable_stats(grade[rows], logits[n][rows], thresholds[n]),
                lambda rows, r=reference: lm.referable_stats(grade[rows], logits[r][rows], thresholds[r]), clusters, reps, seed)
    primary_name = lock['primary_variant']
    promotion = lm.verdict(results[primary_name], comparisons[f'{primary_name}_vs_A_legacy'])
    per_camera = {}
    for cam in sorted(set(camera)):
        sel = np.flatnonzero(camera == cam)
        if len(sel) < 50 or (grade[sel] >= 2).sum() < 5:
            per_camera[cam] = {'images': int(len(sel)), 'note': 'too few referable cases for estimates'}
            continue
        per_camera[cam] = {'images': int(len(sel)), 'referable': int((grade[sel] >= 2).sum()), 'variants': {
            n: bs.bootstrap(lambda rows, n=n, sel=sel: lm.referable_stats(grade[sel][rows], logits[n][sel][rows], thresholds[n]),
                            clusters[sel], reps, seed) for n in (primary_name, *lock['references'])}}
    z = np.load(args.output / 'quality_scores.npz', allow_pickle=True)
    qsel = everything.has_row.to_numpy() & ~excluded
    qscores = {'stacked': z['final'][qsel], 'eyeq_candidate': z['eyeq'][qsel], 'heuristic_iqs': -z['iqs'][qsel]}
    cand_threshold = float(np.log(0.375 / (1 - 0.375)))
    quality = {}
    for target_name, target in (('inadequate', ~everything.gradable.to_numpy()[qsel]),
                                ('any_factor_unsatisfactory', everything.any_factor_unsatisfactory.to_numpy()[qsel])):
        qclusters = everything.cluster.to_numpy()[qsel]
        block = {name: {'auc': bs.bootstrap(lambda rows, sc=sc: {'auc': m.auc(target[rows], sc[rows])}, qclusters, reps, seed)['auc']}
                 for name, sc in qscores.items()}
        for name, thr in (('stacked', lock['quality_operating_threshold']), ('eyeq_candidate', cand_threshold)):
            called = qscores[name] >= thr
            block[name]['operating_point'] = {'threshold': thr, 'reject_recall': float(called[target].mean()) if target.any() else float('nan'),
                                              'false_reject_rate': float(called[~target].mean())}
        quality[target_name] = {'positives': int(target.sum()), 'images': int(qsel.sum()), 'results': block}
    report = {'schema_version': 1, 'tier': 'TF (confirmation of a variant selected on mBRSET)', 'lock_sha256': sha(args.output / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'primary_variant': primary_name, 'references': lock['references'],
              'population': {'images_in_lock': int(len(everything)), 'identity_excluded': int(excluded.sum()),
                             'without_label_row': list(everything.image_id[~everything.has_row]),
                             'inadequate_quality': int((everything.has_row & ~everything.gradable & ~everything.excluded).sum()),
                             'adequate_without_grade': list(everything.image_id[everything.gradable & ~everything.graded & ~everything.excluded]),
                             'primary_images': int(len(d)), 'referable': int((grade >= 2).sum()), 'any_dr': int((grade >= 1).sum()),
                             'patients': int(len(np.unique(clusters))), 'grade_support': np.bincount(grade, minlength=5).tolist(),
                             'cameras': {c: int((camera == c).sum()) for c in sorted(set(camera))},
                             'non_referable_with_other_disease': int(((grade < 2) & other).sum()),
                             'factor_column_spellings': spellings},
              'results': results, 'decision_summaries': summaries, 'any_dr_auc': any_dr, 'other_disease_specificity': other_stats,
              'comparisons': comparisons, 'promotion_verdict': promotion, 'per_camera': per_camera, 'quality': quality,
              'limitations': ['Three Sao Paulo centres, 2010-2020, two desktop cameras; not a smartphone test (that is mBRSET).',
                              'The primary variant was selected on mBRSET (exposed): this is a confirmation, not a fresh model selection.',
                              'Images with Inadequate quality are outside the DR metrics (counted).',
                              'Clusters are patients (patient_id column).']}
    report_path = args.output / 'report.json'
    atomic_json(report_path, report)
    exposure.record_exposure(COHORT, args.output / 'lock.json', report_path)
    for name, block in results.items():
        print(name, {k: round(v['estimate'], 4) for k, v in block.items()})
    print('promotion verdict:', promotion['passed'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'evaluate'):
        s = sub.add_parser(name)
        s.add_argument('--variants', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--data', type=Path, default=DATA)
        s.add_argument('--exclusions', type=Path, default=EXCLUSIONS)
        s.add_argument('--patients', type=Path, default=PATIENTS)
        s.add_argument('--protocol', type=Path, default=PROTOCOL)
        if name == 'freeze':
            s.add_argument('--precomputed', type=Path, help='folder with logits_<id>.npy, logits_order.csv and quality_scores.npz')
    g = sub.add_parser('logits')
    g.add_argument('--variants', type=Path, required=True)
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--data', type=Path, default=DATA)
    g.add_argument('--ids', nargs='+', required=True)
    g.add_argument('--device', default='cpu')
    g.add_argument('--threads', type=int, default=6)
    q = sub.add_parser('quality')
    q.add_argument('--out', type=Path, required=True)
    q.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args(argv)
    {'freeze': freeze, 'evaluate': evaluate, 'logits': logits_command, 'quality': quality_command}[args.command](args)


if __name__ == '__main__':
    main()
