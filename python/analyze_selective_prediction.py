#!/usr/bin/env python3
"""Build coverage-risk evidence for confidence/disagreement abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.metrics import quadratic_weighted_kappa


ROOT = Path(__file__).resolve().parents[1]


def curve(frame: pd.DataFrame, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    required = {
        'status', 'true_grade', 'predicted_grade', 'confidence',
        'head_disagreement',
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    evaluated = frame[frame['status'] == 'success'].copy()
    if evaluated.empty:
        raise ValueError('No successful predictions found')
    if thresholds is None:
        thresholds = np.round(np.arange(0.0, 1.0, 0.05), 2)

    truth_grade = evaluated['true_grade'].to_numpy(dtype=int)
    pred_grade = evaluated['predicted_grade'].to_numpy(dtype=int)
    confidence = evaluated['confidence'].to_numpy(dtype=float)
    disagreement = evaluated['head_disagreement'].astype(bool).to_numpy()
    rows = []
    for threshold in thresholds:
        retained = (~disagreement) & (confidence >= threshold)
        reviewed = ~retained
        truth = truth_grade[retained] >= 2
        prediction = pred_grade[retained] >= 2
        tp = int(np.sum(truth & prediction))
        tn = int(np.sum(~truth & ~prediction))
        fp = int(np.sum(~truth & prediction))
        fn = int(np.sum(truth & ~prediction))
        rows.append({
            'minimum_calibrated_grade_confidence': float(threshold),
            'total_images': int(len(evaluated)),
            'auto_decision_images': int(retained.sum()),
            'human_review_images': int(reviewed.sum()),
            'coverage': float(retained.mean()),
            'human_review_workload': float(reviewed.mean()),
            'selective_grade_error_rate': (
                float(np.mean(truth_grade[retained] != pred_grade[retained]))
                if retained.any() else None),
            'selective_qwk': (
                float(quadratic_weighted_kappa(
                    truth_grade[retained], pred_grade[retained]))
                if retained.sum() > 1 else None),
            'referable_sensitivity': tp / (tp + fn) if tp + fn else None,
            'referable_specificity': tn / (tn + fp) if tn + fp else None,
            'auto_decision_false_negatives': fn,
            'auto_decision_false_positives': fp,
            'disagreement_reviews': int(disagreement.sum()),
            'low_confidence_reviews': int((~disagreement & (confidence < threshold)).sum()),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--per-image-csv', type=Path,
        default=ROOT / 'results/verification_2026-09-18/fold0_dual_head/per_image_results.csv')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/verification_2026-09-18/selective_prediction')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = curve(pd.read_csv(args.per_image_csv))
    args.output.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output / 'coverage_risk_curve.csv', index=False)
    payload = {
        'schema_version': 1,
        'dataset': 'APTOS Fold-0 internal holdout',
        'policy': (
            'Review every dual-head disagreement and every prediction below '
            'the selected calibrated grade-confidence threshold.'
        ),
        'curve': results.where(pd.notna(results), None).to_dict(orient='records'),
        'limitations': [
            'This is internal post-hoc evidence, not an externally selected threshold.',
            'Reported workload assumes every abstention receives human review.',
        ],
    }
    (args.output / 'coverage_risk_curve.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
