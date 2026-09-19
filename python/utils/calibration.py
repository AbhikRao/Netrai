"""Reproducible probability calibration utilities for NetrAI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray) -> np.ndarray:
    """Return a numerically stable row-wise softmax."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('logits must be a finite 1-D or 2-D class array')
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def temperature_scale_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling and return calibrated probabilities."""
    temperature = validate_temperature(temperature)
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def temperature_scale_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale probabilities via equivalent log-probability logits."""
    values = np.asarray(probabilities, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError('probabilities must be a 1-D or 2-D class array')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('probabilities must be finite and non-negative')
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError('each probability row must have positive mass')
    normalized = values / row_sums
    calibrated = temperature_scale_logits(np.log(np.clip(normalized, 1e-12, 1.0)), temperature)
    return calibrated[0] if one_dimensional else calibrated


def validate_temperature(temperature: float) -> float:
    """Validate and normalize a scalar temperature."""
    value = float(temperature)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('temperature must be a finite positive scalar')
    return value


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values, targets = _validate_probabilities_and_labels(probabilities, labels)
    selected = values[np.arange(len(targets)), targets]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values, targets = _validate_probabilities_and_labels(probabilities, labels)
    one_hot = np.eye(values.shape[1], dtype=np.float64)[targets]
    return float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))


def expected_calibration_error(
        probabilities: np.ndarray, labels: np.ndarray, predictions: np.ndarray | None = None,
        n_bins: int = 15) -> float:
    """Top-label ECE for argmax or a supplied deployed decision policy."""
    values, targets = _validate_probabilities_and_labels(probabilities, labels)
    if n_bins < 2:
        raise ValueError('n_bins must be at least 2')
    if predictions is None:
        decisions = values.argmax(axis=1)
    else:
        decisions = np.asarray(predictions, dtype=int).reshape(-1)
        if decisions.shape != targets.shape or np.any((decisions < 0) | (decisions >= values.shape[1])):
            raise ValueError('predictions must contain one valid class index per label')
    confidences = values[np.arange(len(targets)), decisions]
    correct = decisions == targets
    error = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidences >= lower) & (confidences <= upper) if index == 0 else (
            (confidences > lower) & (confidences <= upper))
        if in_bin.any():
            error += float(in_bin.mean()) * abs(
                float(correct[in_bin].mean()) - float(confidences[in_bin].mean()))
    return float(error)


def calibration_metrics(
        probabilities: np.ndarray, labels: np.ndarray,
        predictions: np.ndarray | None = None, n_bins: int = 15) -> dict[str, float]:
    """Return accuracy, mean confidence, NLL, Brier, and top-label ECE."""
    values, targets = _validate_probabilities_and_labels(probabilities, labels)
    decisions = values.argmax(axis=1) if predictions is None else np.asarray(predictions, dtype=int)
    confidences = values[np.arange(len(targets)), decisions]
    return {
        'accuracy': float(np.mean(decisions == targets)),
        'mean_confidence': float(np.mean(confidences)),
        'nll': multiclass_nll(values, targets),
        'brier': multiclass_brier(values, targets),
        'ece': expected_calibration_error(values, targets, decisions, n_bins=n_bins),
    }


def reliability_bins(
        probabilities: np.ndarray, labels: np.ndarray,
        predictions: np.ndarray | None = None, n_bins: int = 15) -> list[dict[str, float | int]]:
    """Return auditable bin-level top-label reliability statistics."""
    values, targets = _validate_probabilities_and_labels(probabilities, labels)
    decisions = values.argmax(axis=1) if predictions is None else np.asarray(predictions, dtype=int)
    confidences = values[np.arange(len(targets)), decisions]
    correct = decisions == targets
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = (confidences >= lower) & (confidences <= upper) if index == 0 else (
            (confidences > lower) & (confidences <= upper))
        rows.append({
            'bin': index,
            'lower': float(lower),
            'upper': float(upper),
            'count': int(in_bin.sum()),
            'mean_confidence': float(confidences[in_bin].mean()) if in_bin.any() else float('nan'),
            'accuracy': float(correct[in_bin].mean()) if in_bin.any() else float('nan'),
        })
    return rows


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, dict[str, float]]:
    """Fit one positive temperature by minimizing multiclass NLL."""
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    if values.ndim != 2 or values.shape[0] != len(targets):
        raise ValueError('logits and labels must have matching sample counts')
    if not np.isfinite(values).all() or np.any((targets < 0) | (targets >= values.shape[1])):
        raise ValueError('logits/labels contain invalid values')

    def objective(log_temperature: float) -> float:
        return multiclass_nll(temperature_scale_logits(values, np.exp(log_temperature)), targets)

    result = minimize_scalar(
        objective, bounds=(np.log(0.05), np.log(10.0)), method='bounded',
        options={'xatol': 1e-10})
    if not result.success:
        raise RuntimeError(f'temperature optimization failed: {result.message}')
    temperature = validate_temperature(np.exp(result.x))
    return temperature, {
        'optimizer_objective_nll': float(result.fun),
        'optimizer_iterations': int(result.nfev),
    }


def load_calibration_temperature(path: str | Path) -> float:
    """Load and strictly validate a versioned NetrAI calibration artifact."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f'calibration artifact not found: {artifact_path}')
    try:
        payload = json.loads(artifact_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid calibration artifact {artifact_path}: {exc}') from exc
    if payload.get('schema_version') != 1 or payload.get('method') != 'scalar_temperature_scaling':
        raise ValueError(f'unsupported calibration artifact schema/method: {artifact_path}')
    return validate_temperature(payload.get('temperature'))


def _validate_probabilities_and_labels(
        probabilities: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=int).reshape(-1)
    if values.ndim != 2 or values.shape[0] != len(targets) or values.shape[1] < 2:
        raise ValueError('probabilities and labels must have matching sample counts')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('probabilities must be finite and non-negative')
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0) or np.any((targets < 0) | (targets >= values.shape[1])):
        raise ValueError('probabilities/labels contain invalid values')
    return values / row_sums, targets
