#!/usr/bin/env python3
"""Post-hoc paired threshold-free comparison on an already evaluated lock.

For each cohort of a lock whose ``report.json`` exists (labels already exposed),
reports the AUC of the expected-grade score and sensitivity at 85% specificity for
each listed pair, with cluster-bootstrap 95% intervals of the paired difference.
Threshold-free, so it does not depend on any decision policy; it is analysis of
exposed data, not a new acceptance test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

import locked_evaluation as le
from utils.artifacts import atomic_json


def score(logits):
    shifted = logits[:, :5] - logits[:, :5].max(1, keepdims=True)
    probability = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    return probability @ np.arange(5)


def sens_at_85(truth, value):
    fpr, tpr, _ = roc_curve(truth, value)
    return float(tpr[(1 - fpr) >= 0.85].max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--pair', nargs=2, action='append', required=True, metavar=('A', 'B'))
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.lock / 'lock.json').read_text())
    if not (args.lock / 'report.json').is_file():
        raise RuntimeError('Run the locked evaluation first; this analysis never reads unexposed labels')
    cohorts = le.ROUNDS[lock.get('round', 2)]['cohorts']
    out = {'schema_version': 1, 'status': 'post_hoc_threshold_free_analysis_of_exposed_labels',
           'lock_sha256': le.sha(args.lock / 'lock.json'), 'replicates': args.replicates, 'cohorts': {}}
    for cohort in cohorts:
        frame = pd.read_csv(args.lock / f'inventory_{cohort}.csv')
        frame, gradable, grade = le.cohort_labels(cohort, frame)
        truth = grade[gradable] >= 2
        clusters = frame.exam_id.to_numpy()[gradable]
        unique, index = np.unique(clusters, return_inverse=True)
        members = [np.flatnonzero(index == k) for k in range(len(unique))]
        rng = np.random.default_rng(lock['bootstrap_seed'])
        samples = [np.concatenate([members[k] for k in rng.integers(len(unique), size=len(unique))])
                   for _ in range(args.replicates)]

        def values(name):
            return score(np.load(args.lock / f'logits_{cohort}_{name}.npy')[gradable])

        block = {}
        for a, b in args.pair:
            va, vb = values(a), values(b)
            point = {'auc': (roc_auc_score(truth, va), roc_auc_score(truth, vb)),
                     'sens_at_85_spec': (sens_at_85(truth, va), sens_at_85(truth, vb))}
            diffs = {k: [] for k in point}
            for s in samples:
                t = truth[s]
                diffs['auc'].append(roc_auc_score(t, va[s]) - roc_auc_score(t, vb[s]))
                diffs['sens_at_85_spec'].append(sens_at_85(t, va[s]) - sens_at_85(t, vb[s]))
            block[f'{a}_vs_{b}'] = {k: {a: float(point[k][0]), b: float(point[k][1]),
                                        'difference': float(point[k][0] - point[k][1]),
                                        'difference_ci95': np.percentile(diffs[k], [2.5, 97.5]).tolist()}
                                    for k in point}
        out['cohorts'][cohort] = {'cluster_unit': cohorts[cohort]['cluster'], 'gradable': int(gradable.sum()),
                                  'referable': int(truth.sum()), 'pairs': block}
        for name, v in block.items():
            print(cohort, name, {k: (round(x['difference'], 4), [round(y, 4) for y in x['difference_ci95']])
                                 for k, x in v.items()})
    atomic_json(args.output, out)


if __name__ == '__main__':
    main()
