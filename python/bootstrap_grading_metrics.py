#!/usr/bin/env python3
"""Add patient-level bootstrap confidence intervals to a grading validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from utils.metrics import quadratic_weighted_kappa


ROOT = Path(__file__).resolve().parents[1]


def diagnostic_metrics(frame: pd.DataFrame) -> dict[str, float]:
    truth_grade = frame['true_grade'].to_numpy(dtype=int)
    prediction_grade = frame['predicted_grade'].to_numpy(dtype=int)
    truth = truth_grade >= 2
    prediction = prediction_grade >= 2
    tp = int(np.sum(truth & prediction))
    tn = int(np.sum(~truth & ~prediction))
    fp = int(np.sum(~truth & prediction))
    fn = int(np.sum(truth & ~prediction))
    result = {
        'qwk': float(quadratic_weighted_kappa(truth_grade, prediction_grade)),
        'accuracy': float(np.mean(truth_grade == prediction_grade)),
        'referable_sensitivity': tp / (tp + fn) if tp + fn else float('nan'),
        'referable_specificity': tn / (tn + fp) if tn + fp else float('nan'),
    }
    scores = frame['referable_probability'].to_numpy(dtype=float)
    result['referable_auc'] = (
        float(roc_auc_score(truth, scores)) if np.unique(truth).size == 2
        else float('nan'))
    return result


def stratified_bootstrap(
    frame: pd.DataFrame, replicates: int = 2000, seed: int = 2026,
) -> dict:
    """Resample images within true grades and return percentile intervals."""
    if replicates < 100:
        raise ValueError('Use at least 100 bootstrap replicates')
    frame = frame[
        (frame['status'] == 'success')
        & frame['true_grade'].notna()
        & frame['predicted_grade'].notna()
    ].copy()
    if frame.empty:
        raise ValueError('No successfully evaluated rows found')
    groups = [group.reset_index(drop=True) for _, group in frame.groupby('true_grade')]
    rng = np.random.default_rng(seed)
    distributions: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled = pd.concat([
            group.iloc[rng.integers(0, len(group), size=len(group))]
            for group in groups
        ], ignore_index=True)
        for name, value in diagnostic_metrics(sampled).items():
            if np.isfinite(value):
                distributions.setdefault(name, []).append(value)

    point = diagnostic_metrics(frame)
    metrics = {}
    for name, value in point.items():
        samples = np.asarray(distributions.get(name, []), dtype=float)
        metrics[name] = {
            'point_estimate': value,
            'ci95_low': float(np.percentile(samples, 2.5)) if samples.size else None,
            'ci95_high': float(np.percentile(samples, 97.5)) if samples.size else None,
        }
    per_grade = {}
    for grade, group in frame.groupby('true_grade'):
        correct = (group['predicted_grade'].to_numpy(dtype=int) == int(grade))
        boot = np.empty(replicates, dtype=float)
        for index in range(replicates):
            boot[index] = correct[
                rng.integers(0, correct.size, size=correct.size)].mean()
        per_grade[str(int(grade))] = {
            'support': int(correct.size),
            'recall': float(correct.mean()),
            'recall_ci95_low': float(np.percentile(boot, 2.5)),
            'recall_ci95_high': float(np.percentile(boot, 97.5)),
        }
    return {
        'schema_version': 1,
        'sampling_unit': 'image (one IDRiD eye image per row)',
        'method': 'true-grade-stratified percentile bootstrap',
        'replicates': replicates,
        'seed': seed,
        'images': int(len(frame)),
        'metrics': metrics,
        'per_grade': per_grade,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('per_image_csv', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replicates', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.per_image_csv.with_name('bootstrap_metrics.json')
    evidence = stratified_bootstrap(
        pd.read_csv(args.per_image_csv), args.replicates, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(evidence, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
