#!/usr/bin/env python3
"""Locked, single-use evaluation of the X9 quality score on DeepDRiD Challenge-2 (pre-registered in X9_PREREG.md).

``freeze``   scores the evaluation images (label-free): frozen-encoder embeddings, the EyeQ candidate head, the DeepDRiD heads
             and the stacked score (all fitted on DeepDRiD TRAINING only, exactly as ``train_quality_x9.fit_pipeline``), plus
             the heuristic scalar IQS; binds images, code, weights, parameters and scores into ``lock.json`` with
             ``labels_read: false`` and records the cohort as *locked*. No label is read.
``evaluate`` verifies the bindings, opens the label workbook once through ``exposure.assert_label_read_allowed`` and reports:
             primary  AUC for ungradable (Overall quality = 0) of the stacked score vs the EyeQ candidate and the heuristic IQS
                      (paired patient-cluster bootstrap) and the reject recall / false-reject rate at the frozen operating point;
             secondary AUC of the factor heads for heavy artifact / poor clarity / poor field and Spearman correlations.
Primary subset: all evaluation images except patients 420 and 454 (repeat of a validation patient's retina).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import exposure
import train_quality_x9 as x9
from modules.quality import assess_image_quality
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m
from utils.artifacts import atomic_json, sha256_file

ROOT = x9.ROOT
COHORT = 'deepdrid_challenge2_quality'
EVAL_IMAGES = x9.DRID / 'Online-Challenge1&2-Evaluation' / 'Images'
EXCLUDED_PATIENTS = {420, 454}
DEV_CACHE = ROOT / 'results/model_improvement/experiments/X9/embeddings_dev.npz'
PROTOCOL = ROOT / '.project-notes/docs/X9_PREREG.md'
CODE = ['python/locked_quality_eval.py', 'python/train_quality_x9.py', 'python/models/quality_model.py', 'python/modules/quality.py',
        'python/exposure.py', 'python/netrai_eval/metrics.py', 'python/netrai_eval/bootstrap.py']
CANDIDATE_REJECT_THRESHOLD = 0.375


def inventory(images: Path = EVAL_IMAGES) -> pd.DataFrame:
    rows = []
    for folder in sorted(p for p in images.iterdir() if p.is_dir()):
        for path in sorted(folder.glob('*.jpg')):
            rows.append({'image_id': path.stem, 'patient_id': int(folder.name), 'path': str(path)})
    frame = pd.DataFrame(rows)
    frame['primary'] = ~frame.patient_id.isin(EXCLUDED_PATIENTS)
    return frame


def freeze(args) -> None:
    lock_path = args.output / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    exposure.assert_fresh_allowed(COHORT)
    args.output.mkdir(parents=True, exist_ok=True)
    frame = inventory(args.images)
    frame['sha256'] = [sha256_file(p) for p in frame.path]
    inv = args.output / 'inventory.csv'
    frame.to_csv(inv, index=False)
    train = x9.load_split('training')
    xt = np.load(args.dev_cache)['train']
    cfg = json.loads(x9.CANDIDATE_CFG.read_text())
    qmodel = x9.load_candidate()
    predict, info = x9.fit_pipeline(xt, train, qmodel, float(cfg['confidence_temperature']))
    embeddings = x9.embed([Path(p) for p in frame.path], qmodel)
    scores = predict(embeddings)
    iqs = np.array([assess_image_quality(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))[0] for p in frame.path])
    np.savez(args.output / 'scores.npz', eyeq=scores['eyeq_reject_logit'], final=scores['final'], iqs=iqs,
             **{f'head_{k}': v for k, v in scores['heads'].items()})
    atomic_json(args.output / 'parameters.json', {**info, 'candidate_reject_threshold': CANDIDATE_REJECT_THRESHOLD})
    files = [inv, args.output / 'scores.npz', args.output / 'parameters.json', args.dev_cache, args.protocol, x9.CANDIDATE, x9.CANDIDATE_CFG,
             *[ROOT / c for c in CODE]]
    bindings = {str(Path(p).resolve().relative_to(ROOT)): sha256_file(p) for p in files}
    atomic_json(lock_path, {'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
                            'tests': {COHORT: f'{len(frame)} images, primary subset {int(frame.primary.sum())}'}, 'fresh': True,
                            'labels_read': False, 'bootstrap_seed': args.seed, 'bootstrap_replicates': args.replicates,
                            'operating_threshold': info['operating_threshold_5pct_false_reject_oof'], 'bindings': bindings})
    exposure.record_locked(COHORT, lock_path)
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def read_workbook() -> pd.DataFrame:
    """Open the reserved workbook (path from the registry). Callers must have passed assert_label_read_allowed."""
    path = exposure._resolved(exposure.entry(COHORT)['label_paths'][0])
    table = pd.read_excel(path)
    table.columns = [str(c).strip() for c in table.columns]
    return table


def spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def evaluate(args) -> None:
    lock = json.loads((args.output / 'lock.json').read_text())
    exposure.assert_label_read_allowed(COHORT, args.output)
    for name, digest in lock['bindings'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f'Locked artifact changed: {name}')
    frame = pd.read_csv(args.output / 'inventory.csv')
    z = np.load(args.output / 'scores.npz')
    table = read_workbook()
    table['image_id'] = table['image_id'].astype(str)
    merged = frame.merge(table, on='image_id', how='left', indicator=True)
    labelled = (merged['_merge'] == 'both').to_numpy() & merged.primary.to_numpy()
    unlabelled = list(merged.image_id[merged.primary & (merged['_merge'] != 'both')])
    sel = np.flatnonzero(labelled)
    d = merged.iloc[sel].reset_index(drop=True)
    positive = (d['Overall quality'].astype(float) == 0).to_numpy()
    clusters = d.patient_id.to_numpy()
    s = {'stacked': z['final'][sel], 'eyeq_candidate': z['eyeq'][sel], 'heuristic_iqs': -z['iqs'][sel]}
    seed, reps = lock['bootstrap_seed'], lock['bootstrap_replicates']
    threshold = lock['operating_threshold']
    cand_threshold = float(np.log(CANDIDATE_REJECT_THRESHOLD / (1 - CANDIDATE_REJECT_THRESHOLD)))

    def point(score, thr):
        called = score >= thr
        return {'reject_recall': float(called[positive].mean()), 'false_reject_rate': float(called[~positive].mean())}

    results = {}
    for name, score in s.items():
        results[name] = {'auc_ungradable': bs.bootstrap(lambda rows, sc=score: {'auc': m.auc(positive[rows], sc[rows])}, clusters, reps, seed)['auc']}
    results['stacked']['operating_point_frozen'] = point(s['stacked'], threshold)
    results['eyeq_candidate']['operating_point_candidate_0.375'] = point(s['eyeq_candidate'], cand_threshold)
    comparisons = {}
    for other in ('eyeq_candidate', 'heuristic_iqs'):
        comparisons[f'stacked_minus_{other}'] = bs.paired_bootstrap(
            lambda rows: {'auc': m.auc(positive[rows], s['stacked'][rows])}, lambda rows, o=other: {'auc': m.auc(positive[rows], s[o][rows])},
            clusters, reps, seed)['auc']
    secondary = {}
    for head, column, rule, sign in (('artifact', 'Artifact', lambda v: v >= 6, 1), ('clarity', 'Clarity', lambda v: v <= 4, -1),
                                      ('field', 'Field definition', lambda v: v <= 4, -1)):
        y = rule(d[column].astype(float)).to_numpy()
        h = z[f'head_{head}'][sel]
        secondary[head] = {'auc': m.auc(y, h) if y.any() and (~y).any() else float('nan'), 'positives': int(y.sum()),
                           'spearman_with_severity': spearman(h, sign * d[column].astype(float).to_numpy())}
    report = {'schema_version': 1, 'tier': 'TF', 'lock_sha256': sha256_file(args.output / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(),
              'population': {'primary_images': int(len(d)), 'ungradable': int(positive.sum()), 'patients': int(len(np.unique(clusters))),
                             'unlabelled_primary_images': unlabelled},
              'results': results, 'comparisons': comparisons, 'secondary_factor_heads': secondary,
              'limitations': ['Ungradable prevalence is high in DeepDRiD (about half): metrics are not transferable to screening populations.',
                              'The EyeQ-trained candidate and DeepDRiD factor heads share one frozen ImageNet encoder at 224 px.',
                              'Challenge-2 images come from the same release as the DeepDRiD training/validation images used to fit the heads (same camera family).']}
    atomic_json(args.output / 'report.json', report)
    exposure.record_exposure(COHORT, args.output / 'lock.json', args.output / 'report.json')
    for name, block in results.items():
        print(name, {k: (round(v['estimate'], 4) if isinstance(v, dict) and 'estimate' in v else v) for k, v in block.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'evaluate'):
        s = sub.add_parser(name)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--images', type=Path, default=EVAL_IMAGES)
        s.add_argument('--dev-cache', type=Path, default=DEV_CACHE)
        s.add_argument('--protocol', type=Path, default=PROTOCOL)
        s.add_argument('--seed', type=int, default=20261107)
        s.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args(argv)
    (freeze if args.command == 'freeze' else evaluate)(args)


if __name__ == '__main__':
    main()
