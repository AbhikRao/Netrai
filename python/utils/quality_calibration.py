"""Ordered EyeQ threshold calibration for NetrAI's scalar image-quality score."""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import f1_score


EYEQ_GOOD = 0
EYEQ_USABLE = 1
EYEQ_REJECT = 2


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
