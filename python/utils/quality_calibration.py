"""Ordered EyeQ threshold calibration for NetrAI's scalar image-quality score."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score


EYEQ_GOOD = 0
EYEQ_USABLE = 1
EYEQ_REJECT = 2


def load_iqs_thresholds(
        path: str | Path, *, allow_evidence_only: bool = False,
        ) -> tuple[float, float, dict]:
    """Load a versioned EyeQ artifact without silently deploying weak evidence."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f'Quality-threshold artifact not found: {artifact_path}')
    with artifact_path.open(encoding='utf-8') as handle:
        artifact = json.load(handle)
    required = {
        'schema_version', 'status', 'deployment_status', 'reject_threshold',
        'good_threshold', 'train_metrics', 'test_metrics',
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f'Quality-threshold artifact is missing fields: {missing}')
    deployment_status = str(artifact['deployment_status'])
    if deployment_status != 'deployed' and not allow_evidence_only:
        raise ValueError(
            'Quality-threshold artifact is evidence-only and must not be '
            f'deployed (deployment_status={deployment_status!r})')
    reject = float(artifact['reject_threshold'])
    good = float(artifact['good_threshold'])
    if (not np.isfinite([reject, good]).all() or reject < 0 or good > 1
            or reject >= good):
        raise ValueError('Quality thresholds must satisfy 0 <= reject < good <= 1')
    return reject, good, artifact


def apply_iqs_thresholds(
        scores: np.ndarray, reject_threshold: float,
        good_threshold: float) -> np.ndarray:
    """Map IQS to EyeQ labels: 0 Good, 1 Usable, 2 Reject."""
    values = np.asarray(scores, dtype=np.float64)
    reject = float(reject_threshold)
    good = float(good_threshold)
    if not np.isfinite(values).all() or not np.isfinite([reject, good]).all():
        raise ValueError('scores and thresholds must be finite')
    if reject >= good:
        raise ValueError('reject_threshold must be lower than good_threshold')
    predictions = np.full(values.shape, EYEQ_USABLE, dtype=int)
    predictions[values < reject] = EYEQ_REJECT
    predictions[values >= good] = EYEQ_GOOD
    return predictions


def fit_iqs_thresholds(
        scores: np.ndarray, labels: np.ndarray, seed: int = 42) -> tuple[float, float, dict]:
    """Fit two ordered IQS cut points by maximizing three-class macro-F1."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    if values.shape != targets.shape or len(values) < 6:
        raise ValueError('scores and labels need matching lengths with at least six rows')
    if not np.isfinite(values).all() or not set(np.unique(targets)).issubset({0, 1, 2}):
        raise ValueError('invalid scores or EyeQ labels')
    if np.unique(targets).size != 3:
        raise ValueError('all three EyeQ quality classes are required')
    lower = float(values.min()) - 1e-6
    upper = float(values.max()) + 1e-6

    def objective(parameters: np.ndarray) -> float:
        reject, good = sorted(float(value) for value in parameters)
        if good - reject < 1e-6:
            return 1.0
        predictions = apply_iqs_thresholds(values, reject, good)
        return 1.0 - float(f1_score(
            targets, predictions, labels=[0, 1, 2], average='macro', zero_division=0))

    result = differential_evolution(
        objective, bounds=[(lower, upper), (lower, upper)], seed=seed,
        popsize=20, tol=1e-9, polish=True, updating='immediate', workers=1)
    if not result.success:
        raise RuntimeError(f'IQS threshold optimization failed: {result.message}')
    reject_threshold, good_threshold = sorted(float(value) for value in result.x)
    return reject_threshold, good_threshold, {
        'training_macro_f1': 1.0 - float(result.fun),
        'optimizer_evaluations': int(result.nfev),
        'seed': int(seed),
    }
