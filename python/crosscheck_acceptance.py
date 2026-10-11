#!/usr/bin/env python3
"""Independently recompute locked DeepDRiD acceptance metrics from per-image CSVs.

Reads the post-lock labelled manifest and each variant's per-image output, then
recomputes primary-population (identity-eligible, gradable) model-only metrics,
patient-cluster bootstrap intervals and quality-gated counts, and compares them
with the MATLAB acceptance report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.artifacts import atomic_json
from utils.metrics import quadratic_weighted_kappa


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('acceptance', type=Path, help='results/model_improvement/acceptance')
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    labelled = pd.read_csv(args.acceptance / 'labelled/evaluation_labelled.csv', dtype={'patient_id': str, 'id_code': str})
    report = json.loads((args.acceptance / 'evaluation/acceptance_report.json').read_text())
    variants = report['variants'] if isinstance(report['variants'], list) else [report['variants']]
    primary = labelled.acceptance_identity_eligible.astype(bool) & (labelled.true_grade <= 4)
    out = {'primary_images': int(primary.sum()), 'primary_patients': int(labelled.loc[primary, 'patient_id'].nunique()),
           'all_records': int(len(labelled)), 'variants': {}}
    rng = np.random.default_rng(args.seed)
    patients = labelled.loc[primary, 'patient_id'].to_numpy()
    unique = np.unique(patients)
    draws = rng.integers(len(unique), size=(args.replicates, len(unique)))
    index = np.searchsorted(unique, patients)
    for variant in variants:
        rows = pd.read_csv(args.acceptance / f"evaluation/per_image_{variant['id']}.csv",
                           dtype={'patient_id': str, 'id_code': str}).set_index('id_code').loc[labelled.id_code]
        ok = primary.to_numpy() & (rows.model_status.to_numpy() == 'success')
        truth = labelled.true_grade.to_numpy()[ok] >= 2
        predicted = rows.predicted_grade.to_numpy()[ok] >= 2
        grades_true = labelled.true_grade.to_numpy()[ok].astype(int)
        grades_pred = rows.predicted_grade.to_numpy()[ok].astype(int)
        weights = np.stack([np.bincount(d, minlength=len(unique)) for d in draws])[:, index[ok[primary.to_numpy()]]]
        sens = (weights * (truth & predicted)).sum(1) / (weights * truth).sum(1)
        spec = (weights * (~truth & ~predicted)).sum(1) / (weights * ~truth).sum(1)
        action = rows.quality_triage_action.to_numpy()[primary.to_numpy()]
        referable_all = labelled.true_grade.to_numpy()[primary.to_numpy()] >= 2
        mine = {'images': int(ok.sum()), 'tp': int((truth & predicted).sum()), 'fn': int((truth & ~predicted).sum()),
                'tn': int((~truth & ~predicted).sum()), 'fp': int((~truth & predicted).sum()),
                'sensitivity': float(predicted[truth].mean()), 'specificity': float((~predicted[~truth]).mean()),
                'sensitivity_ci95_patient_cluster': interval(sens), 'specificity_ci95_patient_cluster': interval(spec),
                'auc_expected_grade_score': float(roc_auc_score(truth, rows.continuous_score.to_numpy()[ok])),
                'qwk': float(quadratic_weighted_kappa(grades_true, grades_pred)),
                'per_grade_recall': [float(np.mean(grades_pred[grades_true == g] == g)) for g in range(5)],
                'per_grade_support': [int(np.sum(grades_true == g)) for g in range(5)],
                'quality_gated': {'referable_referred': int(np.sum(referable_all & (action == 'refer'))),
                                  'referable_auto_cleared': int(np.sum(referable_all & (action == 'auto_clear'))),
                                  'referable_unresolved': int(np.sum(referable_all & np.isin(action, ['recapture', 'uncertain_human_review', 'error']))),
                                  'negative_auto_cleared': int(np.sum(~referable_all & (action == 'auto_clear')))}}
        matlab = variant['primary']['model_only']
        mine['matches_matlab_counts'] = all(mine[k] == matlab[k] for k in ('tp', 'fn', 'tn', 'fp'))
        out['variants'][variant['id']] = mine
        print(variant['id'], json.dumps({k: mine[k] for k in ('tp', 'fn', 'tn', 'fp', 'sensitivity', 'specificity',
                                                               'sensitivity_ci95_patient_cluster', 'specificity_ci95_patient_cluster',
                                                               'auc_expected_grade_score', 'qwk', 'matches_matlab_counts')}))
    atomic_json(args.acceptance / 'python_crosscheck.json', out)


if __name__ == '__main__':
    main()
