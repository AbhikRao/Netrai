#!/usr/bin/env python3
"""Fit and verify NetrAI scalar temperature scaling on a reproducible split.

The current checked-in Fold-0 validation CSV stores softmax probabilities.
Taking log(probability) recovers logits up to a sample-wise additive constant,
which has no effect on softmax temperature scaling. If explicit logit columns
are present, they are used directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.calibration import (
    calibration_metrics,
    fit_temperature,
    reliability_bins,
    softmax,
    temperature_scale_logits,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'results' / 'verification_2026-09-17' / 'fold0_model' / 'per_image_results.csv'
DEFAULT_OUTPUT = ROOT / 'results' / 'verification_2026-09-17' / 'calibration'
DEFAULT_ARTIFACT = ROOT / 'python' / 'weights' / 'calibration.json'


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_logits(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
    logit_columns = [f'logit_grade_{index}' for index in range(5)]
    probability_columns = [f'prob_grade_{index}' for index in range(5)]
    if all(column in frame for column in logit_columns):
        return frame[logit_columns].to_numpy(dtype=np.float64), 'stored_grade_logits'
    if not all(column in frame for column in probability_columns):
        raise ValueError('input CSV needs logit_grade_0..4 or prob_grade_0..4 columns')
    probabilities = frame[probability_columns].to_numpy(dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return np.log(np.clip(probabilities, 1e-12, 1.0)), 'log_softmax_probabilities_equivalent_up_to_additive_constant'


def plot_reliability(raw_rows: list[dict], calibrated_rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for rows, label, color in (
            (raw_rows, 'Raw', '#607D8B'), (calibrated_rows, 'Temperature-scaled', '#1565C0')):
        valid = [row for row in rows if row['count'] > 0]
        axis.plot(
            [row['mean_confidence'] for row in valid], [row['accuracy'] for row in valid],
            marker='o', linewidth=2, label=label, color=color)
    axis.plot([0, 1], [0, 1], '--', color='black', linewidth=1, label='Perfect calibration')
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean confidence', ylabel='Observed accuracy',
             title='NetrAI post-hoc reliability (disjoint evaluation half)')
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(args: argparse.Namespace) -> None:
    source = args.input.resolve()
    frame = pd.read_csv(source)
    if 'status' in frame:
        frame = frame.loc[frame['status'] == 'success'].copy()
    required = {'id_code', 'true_grade', 'predicted_grade'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'input CSV is missing columns: {sorted(missing)}')
    if frame[list(required)].isna().any().any():
        raise ValueError('calibration rows must have ids, labels, and deployed predictions')

    labels = frame['true_grade'].to_numpy(dtype=int)
    deployed_predictions = frame['predicted_grade'].to_numpy(dtype=int)
    logits, input_representation = read_logits(frame)
    indices = np.arange(len(frame))
    calibration_indices, evaluation_indices = train_test_split(
        indices, test_size=args.evaluation_fraction, random_state=args.seed,
        stratify=labels)

    temperature, optimizer = fit_temperature(logits[calibration_indices], labels[calibration_indices])
    raw_probabilities = softmax(logits)
    calibrated_probabilities = temperature_scale_logits(logits, temperature)

    def comparison(selected: np.ndarray, policy_predictions: np.ndarray | None = None) -> dict:
        predictions = None if policy_predictions is None else policy_predictions[selected]
        return {
            'raw': calibration_metrics(
                raw_probabilities[selected], labels[selected], predictions, n_bins=args.bins),
            'calibrated': calibration_metrics(
                calibrated_probabilities[selected], labels[selected], predictions, n_bins=args.bins),
        }

    metrics = {
        'calibration_split_argmax': comparison(calibration_indices),
        'evaluation_split_argmax': comparison(evaluation_indices),
        'evaluation_split_deployed_ordinal_policy': comparison(
            evaluation_indices, deployed_predictions),
    }
    raw_bins = reliability_bins(
        raw_probabilities[evaluation_indices], labels[evaluation_indices],
        deployed_predictions[evaluation_indices], n_bins=args.bins)
    calibrated_bins = reliability_bins(
        calibrated_probabilities[evaluation_indices], labels[evaluation_indices],
        deployed_predictions[evaluation_indices], n_bins=args.bins)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_frame = frame[['id_code', 'true_grade', 'predicted_grade']].copy()
    split_frame['split'] = 'calibration'
    split_frame.loc[split_frame.index[evaluation_indices], 'split'] = 'evaluation'
    split_frame.to_csv(output / 'split_manifest.csv', index=False)
    rows = []
    for method, bin_rows in (('raw', raw_bins), ('calibrated', calibrated_bins)):
        for row in bin_rows:
            rows.append({'method': method, **row})
    pd.DataFrame(rows).to_csv(output / 'reliability_bins.csv', index=False)
    plot_reliability(raw_bins, calibrated_bins, output / 'reliability_diagram.png')

    artifact = {
        'schema_version': 1,
        'method': 'scalar_temperature_scaling',
        'temperature': temperature,
        'applies_to': 'five_class_grade_logits_for_confidence_only',
        'changes_grade_decision': False,
        'created_on': date.today().isoformat(),
        'source': {
            'file': str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            'sha256': sha256_file(source),
            'dataset': 'APTOS 2019 reproducible Fold 0',
            'input_representation': input_representation,
            'total_samples': int(len(frame)),
        },
        'split': {
            'seed': args.seed,
            'stratified_by': 'true_grade',
            'calibration_samples': int(len(calibration_indices)),
            'evaluation_samples': int(len(evaluation_indices)),
            'evaluation_fraction': args.evaluation_fraction,
            'calibration_class_counts': np.bincount(
                labels[calibration_indices], minlength=5).astype(int).tolist(),
            'evaluation_class_counts': np.bincount(
                labels[evaluation_indices], minlength=5).astype(int).tolist(),
        },
        'optimizer': optimizer,
        'metrics': metrics,
        'limitation': (
            'Post-hoc calibration/evaluation halves come from Fold 0, which was also used for '
            'checkpoint selection and ordinal-threshold optimization. This is not independent '
            'external or prospective clinical validation.'),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, allow_nan=False) + '\n'
    args.artifact.write_text(serialized, encoding='utf-8')
    (output / 'calibration_metrics.json').write_text(serialized, encoding='utf-8')
    print(serialized, end='')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--evaluation-fraction', type=float, default=0.5)
    parser.add_argument('--bins', type=int, default=15)
    args = parser.parse_args()
    if not 0.2 <= args.evaluation_fraction <= 0.8:
        parser.error('--evaluation-fraction must be between 0.2 and 0.8')
    if args.bins < 2:
        parser.error('--bins must be at least 2')
    return args


if __name__ == '__main__':
    main(parse_args())
