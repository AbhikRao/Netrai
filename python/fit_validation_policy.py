#!/usr/bin/env python3
"""Fit the predeclared round-2 decision policy on pooled validation outputs.

Grade-2 boundary and referral-head threshold use the source-balanced rule at the
target sensitivity (no single validation source dominates); other ordinal
boundaries maximise QWK; temperature is refit. Accepts one or more training runs
(several runs = ensemble of mean logits). Never reads fresh-test data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_grading_adapter import evaluate, fit_head_threshold, fit_policy, scores
from utils.artifacts import atomic_json, sha256_file
from utils.calibration import fit_temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--target-sensitivity', type=float, default=0.92)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = [pd.read_csv(run / 'validation_rows.csv') for run in args.runs]
    if any(not r.image_id.equals(rows[0].image_id) for r in rows):
        raise ValueError('Validation rows differ between runs')
    logits = np.mean([np.load(run / 'best_validation_outputs.npy') for run in args.runs], axis=0)
    grade, source = rows[0].grade.to_numpy(), rows[0].source.to_numpy()
    thresholds = fit_policy(scores(logits), grade, args.target_sensitivity, source, 'balanced')
    head = fit_head_threshold(logits, grade, args.target_sensitivity, source, 'balanced')
    temperature, _ = fit_temperature(logits[:, :5], grade)
    per_source = {name: evaluate(logits[source == name], grade[source == name], thresholds, head)
                  for name in np.unique(source)}
    atomic_json(args.output, {
        'schema_version': 1, 'status': 'round2_validation_policy_not_acceptance',
        'runs': [str(r) for r in args.runs],
        'validation_outputs_sha256': [sha256_file(r / 'best_validation_outputs.npy') for r in args.runs],
        'rule': 'source-balanced grade-2 boundary and head threshold at target sensitivity; QWK for others',
        'target_sensitivity': args.target_sensitivity, 'thresholds': thresholds, 'head_threshold': head,
        'temperature': float(temperature), 'validation_per_source': per_source})
    print(json.dumps({'thresholds': thresholds, 'head_threshold': head,
                      'per_source': {k: (round(v['sensitivity'], 3), round(v['specificity'], 3), round(v['auc'], 3))
                                     for k, v in per_source.items()}}, indent=1))


if __name__ == '__main__':
    main()
