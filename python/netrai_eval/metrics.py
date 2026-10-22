"""Metric functions shared by all experiments. Pure numpy/sklearn; no I/O.

Conventions: grades are integers 0-4; referable = grade >= 2; the referral score is the
expected grade ``sum_k k*P(k)``. Undefined rates are NaN (never silently 0).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

GRADES = 5


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def grade_probabilities(logits: np.ndarray) -> np.ndarray:
    """Softmax of the five grade logits (the sixth output is the referral head)."""
    return softmax(np.asarray(logits)[:, :GRADES])


def expected_grade_score(logits: np.ndarray) -> np.ndarray:
    return grade_probabilities(logits) @ np.arange(GRADES)


def referral_head_probability(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float)[:, 5]))


def decode_thresholds(score: np.ndarray, thresholds) -> np.ndarray:
    """Grade = number of ordinal thresholds the score reaches (the deployed decoding)."""
    return (np.asarray(score)[:, None] >= np.asarray(thresholds, dtype=float)[None]).sum(axis=1)


def _rate(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else float('nan')


def binary_rates(truth: np.ndarray, positive: np.ndarray) -> dict:
    truth, positive = np.asarray(truth, bool), np.asarray(positive, bool)
    tp, fn = int((truth & positive).sum()), int((truth & ~positive).sum())
    tn, fp = int((~truth & ~positive).sum()), int((~truth & positive).sum())
    return {'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': _rate(tp, tp + fn), 'specificity': _rate(tn, tn + fp),
            'ppv': _rate(tp, tp + fp), 'npv': _rate(tn, tn + fn)}


def auc(truth: np.ndarray, score: np.ndarray) -> float:
    truth = np.asarray(truth, bool)
    if truth.all() or not truth.any():
        return float('nan')
    return float(roc_auc_score(truth, score))


def pr_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Average precision of the referable class."""
    truth = np.asarray(truth, bool)
    if not truth.any():
        return float('nan')
    return float(average_precision_score(truth, score))


def sensitivity_at_specificity(truth: np.ndarray, score: np.ndarray, specificity: float) -> float:
    """Highest sensitivity among thresholds whose specificity is >= the target (as in the locked code)."""
    truth = np.asarray(truth, bool)
    if truth.all() or not truth.any():
        return float('nan')
    fpr, tpr, _ = roc_curve(truth, score)
    allowed = (1 - fpr) >= specificity
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def specificity_at_sensitivity(truth: np.ndarray, score: np.ndarray, sensitivity: float) -> float:
    """Highest specificity among thresholds whose sensitivity is >= the target."""
    truth = np.asarray(truth, bool)
    if truth.all() or not truth.any():
        return float('nan')
    fpr, tpr, _ = roc_curve(truth, score)
    allowed = tpr >= sensitivity
    return float((1 - fpr[allowed]).max()) if allowed.any() else 0.0


def confusion_matrix(true: np.ndarray, predicted: np.ndarray, classes: int = GRADES) -> np.ndarray:
    """Rows = true grade, columns = predicted grade."""
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (np.asarray(true, int), np.asarray(predicted, int)), 1)
    return matrix


def per_grade_recall(matrix: np.ndarray) -> np.ndarray:
    support = matrix.sum(axis=1)
    return np.where(support > 0, np.diag(matrix) / np.maximum(support, 1), np.nan)


def per_grade_precision(matrix: np.ndarray) -> np.ndarray:
    predicted = matrix.sum(axis=0)
    return np.where(predicted > 0, np.diag(matrix) / np.maximum(predicted, 1), np.nan)


def per_grade_f1(matrix: np.ndarray) -> np.ndarray:
    recall, precision = per_grade_recall(matrix), per_grade_precision(matrix)
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = 2 * recall * precision / (recall + precision)
    f1 = np.where((recall + precision) == 0, 0.0, f1)  # present but never correct -> 0
    return np.where(np.isnan(recall) & np.isnan(precision), np.nan, f1)


def macro_f1(matrix: np.ndarray) -> float:
    """Mean F1 over grades that occur in the truth or the predictions (absent grades are skipped)."""
    f1 = per_grade_f1(matrix)
    return float(np.nanmean(f1)) if np.any(~np.isnan(f1)) else float('nan')


def quadratic_weighted_kappa(true: np.ndarray, predicted: np.ndarray, classes: int = GRADES) -> float:
    """QWK over the fixed 0..classes-1 range (the standard definition).

    The older ``utils.metrics.quadratic_weighted_kappa`` normalises by the grades observed in the
    batch; the two agree whenever all grades occur in the truth or the predictions.
    """
    matrix = confusion_matrix(true, predicted, classes).astype(float)
    total = matrix.sum()
    if total == 0:
        return float('nan')
    index = np.arange(classes)
    weights = (index[:, None] - index[None, :]) ** 2 / (classes - 1) ** 2
    expected = np.outer(matrix.sum(axis=1), matrix.sum(axis=0)) / total
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float('nan')
    return float(1.0 - (weights * matrix).sum() / denominator)


def mean_absolute_grade_error(true: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(true, int) - np.asarray(predicted, int))))


def ece(probability: np.ndarray, outcome: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error of a probability against a binary outcome (equal-width bins)."""
    probability, outcome = np.asarray(probability, float), np.asarray(outcome, bool)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(outcome[mask].mean() - probability[mask].mean())
    return float(total)


def brier(probability: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((np.asarray(probability, float) - np.asarray(outcome, float)) ** 2))


def multiclass_ece(probabilities: np.ndarray, true: np.ndarray, bins: int = 15) -> float:
    """Top-label (confidence) ECE of the five-grade distribution."""
    probabilities = np.asarray(probabilities, float)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(true, int)
    return ece(confidence, correct, bins)
