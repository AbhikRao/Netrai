#!/usr/bin/env python3
"""Compare interpretable M1 baselines on the fixed EyeQ train/test split.

This is a model-selection diagnostic, not a deployment trainer. It determines
whether the existing IQS/focus/illumination/FOV feature set contains enough
signal to justify replacing the threshold policy. No candidate is exported or
deployed automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['iqs', 'focus', 'illumination', 'fov']
CLASS_NAMES = ['good', 'usable', 'reject']


def metrics(labels: np.ndarray, predictions: np.ndarray) -> tuple[dict, np.ndarray]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return {
        'accuracy': float(np.mean(labels == predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(
            labels, predictions, labels=[0, 1, 2], average='macro', zero_division=0)),
        'per_class_recall': {
            name: float(matrix[index, index] / max(matrix[index].sum(), 1))
            for index, name in enumerate(CLASS_NAMES)
        },
        'per_class_precision': {
            name: float(matrix[index, index] / max(matrix[:, index].sum(), 1))
            for index, name in enumerate(CLASS_NAMES)
        },
    }, matrix


def build_models(seed: int) -> dict:
    return {
        'balanced_logistic_regression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)),
        'balanced_random_forest': RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight='balanced_subsample',
            n_jobs=-1, random_state=seed),
        'balanced_extra_trees': ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=20, class_weight='balanced',
            n_jobs=-1, random_state=seed),
        'balanced_hist_gradient_boosting': HistGradientBoostingClassifier(
            max_iter=200, l2_regularization=1.0, class_weight='balanced',
            random_state=seed),
    }


def run(args: argparse.Namespace) -> dict:
    train = pd.read_csv(args.train_features.resolve())
    test = pd.read_csv(args.test_features.resolve())
    required = set(FEATURES + ['quality', 'status'])
    for name, frame in (('train', train), ('test', test)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f'{name} feature CSV is missing columns: {missing}')
        if not frame['status'].eq('success').all():
            raise ValueError(f'{name} feature CSV contains failed/incomplete rows')
    x_train = train[FEATURES].to_numpy(float)
    y_train = train['quality'].to_numpy(int)
    x_test = test[FEATURES].to_numpy(float)
    y_test = test['quality'].to_numpy(int)

    results = {}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, model in build_models(args.seed).items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        summary, matrix = metrics(y_test, predictions)
        results[name] = summary
        pd.DataFrame(
            matrix,
            index=[f'true_{item}' for item in CLASS_NAMES],
            columns=[f'pred_{item}' for item in CLASS_NAMES],
        ).to_csv(output / f'{name}_confusion_matrix.csv', index_label='quality')

    ranked = sorted(results, key=lambda name: results[name]['macro_f1'], reverse=True)
    artifact = {
        'schema_version': 1,
        'status': 'feature_baseline_ablation_not_deployed',
        'features': FEATURES,
        'train_images': int(len(train)),
        'held_out_test_images': int(len(test)),
        'seed': int(args.seed),
        'results': results,
        'ranking_by_macro_f1': ranked,
        'best_candidate': ranked[0],
        'deployment_decision': (
            'Do not deploy automatically. A candidate must materially improve the '
            'held-out endpoint and have a reproducible MATLAB implementation.'
        ),
        'limitations': [
            'The candidates use only four handcrafted scores, not retinal image pixels.',
            'EyeQ labels are image-level and no patient identifier is available for patient-grouped splitting.',
            'This ablation does not validate portable rural camera domains.',
        ],
    }
    with (output / 'quality_baseline_ablation.json').open('w', encoding='utf-8') as stream:
        json.dump(artifact, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(artifact, indent=2, allow_nan=False))
    return artifact


def parse_args() -> argparse.Namespace:
    default = ROOT / 'results' / 'verification_2026-09-20' / 'eyeq_quality_calibration'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--train-features', type=Path,
        default=default / 'eyeq_train_quality_features.csv')
    parser.add_argument(
        '--test-features', type=Path,
        default=default / 'eyeq_test_quality_features.csv')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results' / 'verification_2026-09-20' / 'eyeq_quality_ablation')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())
