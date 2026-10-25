#!/usr/bin/env python3
"""Locked, single-use evaluation of frozen grading models on mBRSET (plan v5 + Addenda 2 and 3).

``logits``   label-free: logits of the listed variants on ALL images (CPU, resumable in chunks), keyed by inventory order.
``quality``  label-free: X9 stacked quality score, EyeQ candidate logit and heuristic IQS on ALL images, refit exactly as in
             the Challenge-2 lock (the operating threshold must equal its locked ``parameters.json``).
``freeze``   inventories every image (file names only; identity-excluded images flagged, not dropped), binds image hashes,
             clusters, exclusions, models/policies/code, logits, quality scores and the label file hash into ``lock.json``
             with ``labels_read: false`` and records the cohort as *locked*. No label value is read.
``evaluate`` verifies every binding, opens the label CSV once through ``exposure.assert_label_read_allowed`` and reports:
  primary    referable DR (ICDR >= 2) on gradable (``final_quality == 'yes'``), graded, non-excluded images: sensitivity and
             specificity at each frozen policy, AUC and sensitivity at 85% specificity of the expected-grade score, QWK and
             per-grade recall, patient-cluster bootstrap; paired differences vs every reference; the promotion verdict;
  secondary  any-DR AUC of the temperature-scaled 1 - p(grade 0); quality AUC for ``final_quality == 'no'``.
A second evaluation is refused.
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
from locked_evaluation import ROOT, check_matlab_variant, logits_for, sha, thresholds_for
from locked_rfmid import logits_on, temperature_for
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m
from utils.artifacts import atomic_json

COHORT = 'mbrset'
DATA = ROOT / 'data/mbrset/images'
IDENTITY = ROOT / 'results/model_improvement/mbrset_identity'
EXCLUSIONS = IDENTITY / 'exclusions_v1.txt'
# O8 (2026-10-23): clusters = patient file-name prefix only. Within mBRSET no same-patient image pair reaches vessel
# correlation 0.6 and the strongest cross-patient pairs (0.71-0.73) are visibly different retinas, so 0.6 links here are
# low-information chains (one component would hold 920 images), not identity. Pass --clusters to merge anyway.
CLUSTERS = None
PROTOCOL = ROOT / '.project-notes/docs/MODEL_IMPROVEMENT_PLAN_v5.md'
X9_PARAMETERS = ROOT / 'results/model_improvement/x9_lock/parameters.json'
CODE = ['python/locked_mbrset.py', 'python/locked_evaluation.py', 'python/locked_rfmid.py', 'python/exposure.py',
        'python/utils/preprocessing.py', 'python/models/netrai_model.py', 'python/export_adapted_variant.py',
        'python/netrai_eval/metrics.py', 'python/netrai_eval/bootstrap.py', 'python/train_quality_x9.py',
        'python/models/quality_model.py', 'python/modules/quality.py']
LEGACY_TEMPERATURE = ROOT / 'python/weights/calibration.json'
SPOT_CHECK_IMAGES, SPOT_CHECK_TOLERANCE = 8, 5e-3
CHUNK = 256
TARGET_SENSITIVITY, TARGET_SPECIFICITY = 0.90, 0.85


def inventory(data: Path = DATA, exclusions: Path | None = EXCLUSIONS, clusters: Path | None = CLUSTERS) -> pd.DataFrame:
    """Every image (file name, patient prefix, cluster, excluded flag). File names only; no label file is opened.

    Clusters: patients (file-name prefix) merged through the within-test vessel clusters (``clusters.csv`` from
    ``vessel_clusters.py``, columns ``image`` and ``cluster``) by single linkage.
    """
    files = sorted(p.name for p in data.iterdir() if p.suffix.lower() == '.jpg')
    frame = pd.DataFrame({'image_id': files, 'patient': [f.split('.')[0] for f in files]})
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for patient in frame.patient:
        find('p' + patient)
    if clusters is not None and Path(clusters).is_file():
        table = pd.read_csv(clusters)
        table['file'] = table.image.astype(str).str.replace(r'.*/', '', regex=True)
        mapping = dict(zip(table.file, table.cluster))
        missing = sorted(set(files) - set(mapping))
        if missing:
            raise ValueError(f'Images without a vessel cluster: {missing[:5]}')
        first = {}
        for name, patient in zip(frame.image_id, frame.patient):
            c = mapping[name]
            if c in first:
                union('p' + patient, 'p' + first[c])
            else:
                first[c] = patient
    frame['cluster'] = [find('p' + p) for p in frame.patient]
    excluded = set()
    if exclusions is not None and Path(exclusions).is_file():
        excluded = {line.strip() for line in Path(exclusions).read_text().splitlines() if line.strip() and not line.startswith('#')}
    unknown = excluded - set(files)
    if unknown:
        raise ValueError(f'Exclusions name unknown images: {sorted(unknown)[:5]}')
    frame['excluded'] = frame.image_id.isin(excluded)
    return frame


def label_file() -> Path:
    """The label CSV, taken from the exposure registry (this module never names the file itself)."""
    return exposure._resolved(exposure.entry(COHORT)['label_paths'][0])


def read_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Join the label CSV by file name. Callers must have passed ``assert_label_read_allowed`` first.

    Returns ``has_row``, ``gradable`` (final_quality == 'yes'), ``graded`` (final_icdr 0-4) and ``icdr`` (-1 if none).
    """
    table = pd.read_csv(label_file(), dtype=str, keep_default_na=False)
    table.columns = [c.strip() for c in table.columns]
    table['file'] = table.file.str.strip()
    if table.file.duplicated().any():
        raise ValueError('The label file repeats an image')
    quality = table.final_quality.str.strip().str.lower()
    if not set(quality) <= {'yes', 'no'}:
        raise ValueError(f'Unexpected final_quality spellings: {sorted(set(quality) - {"yes", "no"})}')
    icdr_text = table.final_icdr.str.strip().str.replace(r'\.0$', '', regex=True)
    if not set(icdr_text) <= {'', '0', '1', '2', '3', '4'}:
        raise ValueError(f'Unexpected final_icdr spellings: {sorted(set(icdr_text) - {"", "0", "1", "2", "3", "4"})}')
    key = dict(zip(table.file, zip(quality, icdr_text)))
    out = frame.copy()
    rows = [key.get(f) for f in out.image_id]
    out['has_row'] = [r is not None for r in rows]
    out['gradable'] = [r is not None and r[0] == 'yes' for r in rows]
    out['graded'] = [r is not None and r[1] != '' for r in rows]
    out['icdr'] = [int(r[1]) if r is not None and r[1] != '' else -1 for r in rows]
    return out


def referable_stats(grade: np.ndarray, logits: np.ndarray, thresholds: np.ndarray) -> dict:
    """Primary statistics at a frozen policy (expected grade of the untempered softmax, as in every earlier lock)."""
    truth = grade >= 2
    score = m.expected_grade_score(logits)
    positive = m.decode_thresholds(score, thresholds) >= 2
    rates = m.binary_rates(truth, positive)
    return {'sensitivity': rates['sensitivity'], 'specificity': rates['specificity'], 'auc': m.auc(truth, score),
            'sensitivity_at_85_specificity': m.sensitivity_at_specificity(truth, score, 0.85)}


def grade_summary(grade: np.ndarray, logits: np.ndarray, thresholds: np.ndarray) -> dict:
    predicted = m.decode_thresholds(m.expected_grade_score(logits), thresholds)
    rates = m.binary_rates(grade >= 2, predicted >= 2)
    matrix = m.confusion_matrix(grade, predicted)
    return {**{k: rates[k] for k in ('tp', 'fn', 'tn', 'fp', 'ppv', 'npv')}, 'qwk': m.quadratic_weighted_kappa(grade, predicted),
            'per_grade_recall': m.per_grade_recall(matrix).tolist(), 'per_grade_support': matrix.sum(1).tolist(),
            'confusion_matrix': matrix.tolist()}


def any_dr_score(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits[:, :5] / temperature
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return 1 - p[:, 0]


def verdict(result: dict, comparison: dict) -> dict:
    """Plan v5 promotion rule (point estimates > targets and paired sensitivity gain over A_legacy, CI excluding 0)."""
    sens, spec = result['sensitivity']['estimate'], result['specificity']['estimate']
    gain = comparison['sensitivity']
    passed = sens > TARGET_SENSITIVITY and spec > TARGET_SPECIFICITY and gain['difference_ci95'][0] > 0
    return {'rule': 'sensitivity > 0.90 and specificity > 0.85 (point estimates) and paired sensitivity gain over A_legacy with 95% CI > 0',
            'sensitivity': sens, 'specificity': spec, 'sensitivity_gain_vs_A_legacy': gain['difference'],
            'sensitivity_gain_ci95': gain['difference_ci95'], 'passed': bool(passed),
            'note': 'A pass makes the primary eligible for promotion; the deployed default changes only by owner decision.'}


def logits_command(args) -> None:
    """Logits of the listed variants on every image, saved in chunks so an interrupted CPU run resumes."""
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
    matlab_files = check_matlab_variant(config) if config.get('matlab_primary_variant') else []
    args.output.mkdir(parents=True, exist_ok=True)
    frame = inventory(args.data, args.exclusions, args.clusters)
    frame['sha256'] = [sha(args.data / i) for i in frame.image_id]
    inv = args.output / 'inventory_mbrset.csv'
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
            if saved.shape != (len(frame), 6):
                raise ValueError(f'{pre.name}: shape {saved.shape} != ({len(frame)}, 6)')
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
    files = [args.variants, inv, args.protocol, LEGACY_TEMPERATURE, X9_PARAMETERS, quality, label_file(),
             *[ROOT / c for c in CODE], *matlab_files, *cache.values()]
    files += [p for p in (args.clusters, args.exclusions) if p is not None and Path(p).is_file()]
    files += [ROOT / s['checkpoint'] for s in config['variants'] if s.get('checkpoint')]
    files += [ROOT / s['policy'] for s in config['variants'] if s.get('policy')]
    bindings = {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in files}  # label file: hash only
    atomic_json(lock_path, {
        'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
        'tests': {COHORT: f'{len(frame)} images, {int(frame.excluded.sum())} identity-excluded; referable-DR endpoint (ICDR >= 2)'},
        'fresh': True, 'labels_read': False, 'primary_variant': config['primary'], 'references': config['references'],
        'bootstrap_seed': config['bootstrap_seed'], 'bootstrap_replicates': config['bootstrap_replicates'],
        'logits_provenance': config.get('logits_provenance'), 'precomputed_logits_cpu_spot_check_max_abs': spot,
        'quality_operating_threshold': json.loads(X9_PARAMETERS.read_text())['operating_threshold_5pct_false_reject_oof'],
        'bindings': bindings})
    exposure.record_locked(COHORT, lock_path)
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def evaluate(args) -> None:
    lock = json.loads((args.output / 'lock.json').read_text())
    exposure.assert_label_read_allowed(COHORT, args.output)  # raises without a lock / after a report
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    config = json.loads(args.variants.read_text())
    frame = pd.read_csv(args.output / 'inventory_mbrset.csv', dtype={'patient': str, 'cluster': str})
    everything = read_labels(frame)
    excluded = everything.excluded.to_numpy(bool)
    primary = everything.has_row.to_numpy() & everything.gradable.to_numpy() & everything.graded.to_numpy() & ~excluded
    d = everything[primary].reset_index(drop=True)
    grade, clusters = d.icdr.to_numpy(int), d.cluster.to_numpy()
    seed, reps = lock['bootstrap_seed'], lock['bootstrap_replicates']
    logits = {}
    for spec in config['variants']:
        logits[spec['id']] = np.load(args.output / f"logits_{spec.get('logits_from', spec['id'])}.npy")[primary]
    thresholds = {s['id']: thresholds_for(s)[0] for s in config['variants']}
    results, summaries, any_dr = {}, {}, {}
    for name in logits:
        results[name] = bs.bootstrap(lambda rows, n=name: referable_stats(grade[rows], logits[n][rows], thresholds[n]), clusters, reps, seed)
        summaries[name] = grade_summary(grade, logits[name], thresholds[name])
        spec = next(s for s in config['variants'] if s['id'] == name)
        score = any_dr_score(logits[name], temperature_for(spec))
        any_dr[name] = bs.bootstrap(lambda rows, sc=score: {'auc_any_dr': m.auc(grade[rows] >= 1, sc[rows])}, clusters, reps, seed)['auc_any_dr']
    comparisons = {}
    for reference in lock['references']:
        for name in logits:
            if name == reference:
                continue
            comparisons[f'{name}_vs_{reference}'] = bs.paired_bootstrap(
                lambda rows, n=name: referable_stats(grade[rows], logits[n][rows], thresholds[n]),
                lambda rows, r=reference: referable_stats(grade[rows], logits[r][rows], thresholds[r]), clusters, reps, seed)
    primary_name = lock['primary_variant']
    promotion = verdict(results[primary_name], comparisons[f'{primary_name}_vs_A_legacy'])
    # quality secondary: every non-excluded image with a label row, graded or not
    z = np.load(args.output / 'quality_scores.npz', allow_pickle=True)
    qsel = everything.has_row.to_numpy() & ~excluded
    ungradable = ~everything.gradable.to_numpy()[qsel]
    qclusters = everything.cluster.to_numpy()[qsel]
    qscores = {'stacked': z['final'][qsel], 'eyeq_candidate': z['eyeq'][qsel], 'heuristic_iqs': -z['iqs'][qsel]}
    cand_threshold = float(np.log(0.375 / (1 - 0.375)))

    def point(score, thr):
        called = score >= thr
        return {'reject_recall': float(called[ungradable].mean()) if ungradable.any() else float('nan'),
                'false_reject_rate': float(called[~ungradable].mean())}

    quality = {name: {'auc_ungradable': bs.bootstrap(lambda rows, sc=sc: {'auc': m.auc(ungradable[rows], sc[rows])}, qclusters, reps, seed)['auc']}
               for name, sc in qscores.items()}
    quality['stacked']['operating_point_frozen'] = point(qscores['stacked'], lock['quality_operating_threshold'])
    quality['eyeq_candidate']['operating_point_candidate_0.375'] = point(qscores['eyeq_candidate'], cand_threshold)
    report = {'schema_version': 1, 'tier': 'TF', 'lock_sha256': sha(args.output / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'primary_variant': primary_name, 'references': lock['references'],
              'population': {'images_in_lock': int(len(everything)), 'identity_excluded': int(excluded.sum()),
                             'without_label_row': list(everything.image_id[~everything.has_row]),
                             'ungradable': int((everything.has_row & ~everything.gradable & ~everything.excluded).sum()),
                             'gradable_without_grade': list(everything.image_id[everything.gradable & ~everything.graded & ~everything.excluded]),
                             'primary_images': int(len(d)), 'referable': int((grade >= 2).sum()), 'any_dr': int((grade >= 1).sum()),
                             'patients_clusters': int(len(np.unique(clusters))), 'grade_support': np.bincount(grade, minlength=5).tolist()},
              'results': results, 'decision_summaries': summaries, 'any_dr_auc': any_dr, 'comparisons': comparisons,
              'promotion_verdict': promotion,
              'quality': {'images': int(qsel.sum()), 'ungradable': int(ungradable.sum()), 'results': quality},
              'limitations': ['Single country (Brazil), single handheld camera (Phelcom Eyer); results describe this device and population.',
                              'Ungradable images are outside the DR metrics; screening yield must count them as recaptures/referrals.',
                              'Clusters are patients (file-name prefix) merged through within-test vessel matches.']}
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
        s.add_argument('--clusters', type=Path, default=CLUSTERS)
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
