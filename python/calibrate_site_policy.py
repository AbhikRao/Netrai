#!/usr/bin/env python3
"""Fit a site-calibrated decision policy for one adaptation-ladder rung.

The grade-2 boundary and referral-head threshold are fitted on a named target-site
calibration cohort (default: the full DeepDRiD validation partition) with the
pooled lowest-FPR rule at the declared sensitivity target; the other ordinal
boundaries maximize QWK; temperature is refit on the same cohort. A patient
half-split cross-validation estimates held-out behaviour of this rule. The
policy is written as a hashed JSON consumed by ``export_adapted_variant.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_grading_adapter import CACHE, evaluate, fit_head_threshold, fit_policy, scores
from utils.artifacts import atomic_json, sha256_file
from utils.calibration import fit_temperature
from utils.validation import ordinal_thresholds


def cross_validate(logits, labels, groups, target, seeds=20):
    rows = []
    for seed in range(seeds):
        patients = np.unique(groups)
        np.random.default_rng(seed).shuffle(patients)
        first = np.isin(groups, patients[:len(patients) // 2])
        for fit, held in ((first, ~first), (~first, first)):
            thresholds = fit_policy(scores(logits[fit]), labels[fit], target)
            head = fit_head_threshold(logits[fit], labels[fit], target)
            result = evaluate(logits[held], labels[held], thresholds, head)
            rows.append((result['sensitivity'], result['specificity'], result['head_disagreement_rate']))
    values = np.asarray(rows)
    return {'splits': len(rows), 'mean_sensitivity': float(values[:, 0].mean()),
            'mean_specificity': float(values[:, 1].mean()),
            'mean_head_disagreement': float(values[:, 2].mean()),
            'fraction_meeting_both_targets': float(np.mean((values[:, 0] > 0.90) & (values[:, 1] > 0.85)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('rung', choices=('A', 'B', 'C', 'D'))
    parser.add_argument('--cohort', default='deepdrid_validation')
    parser.add_argument('--target-sensitivity', type=float, default=0.92)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    outputs_path = args.run / f'rung_{args.rung}_outputs.npz'
    data = np.load(outputs_path)
    logits, labels = data[args.cohort], data[f'{args.cohort}_grade']
    groups = pd.read_csv(CACHE / args.cohort / 'rows.csv', dtype=str).group_id.to_numpy()
    if len(groups) != len(labels):
        raise ValueError('Cohort rows and saved outputs are misaligned')
    thresholds = ordinal_thresholds(fit_policy(scores(logits), labels, args.target_sensitivity)).tolist()
    head = fit_head_threshold(logits, labels, args.target_sensitivity)
    temperature, _ = fit_temperature(logits[:, :5], labels)
    policy = {
        'schema_version': 1, 'status': 'site_calibrated_development_policy_not_acceptance',
        'run': args.run.name, 'rung': args.rung, 'outputs_sha256': sha256_file(outputs_path),
        'calibration_cohort': args.cohort, 'calibration_images': int(len(labels)),
        'calibration_patients': int(len(np.unique(groups))),
        'rule': 'pooled lowest-FPR grade-2 boundary and head threshold at target sensitivity; QWK for other boundaries',
        'target_sensitivity': args.target_sensitivity,
        'thresholds': thresholds, 'head_threshold': head, 'temperature': float(temperature),
        'in_sample': evaluate(logits, labels, thresholds, head),
        'patient_half_split_cross_validation': cross_validate(logits, labels, groups, args.target_sensitivity),
    }
    output = args.output or args.run / f'site_policy_{args.rung}.json'
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, policy)
    print(json.dumps({k: policy[k] for k in ('thresholds', 'head_threshold', 'temperature',
                                              'patient_half_split_cross_validation')}, indent=2))


if __name__ == '__main__':
    main()
