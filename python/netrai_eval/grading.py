"""Grading and referral evaluation with cluster bootstrap intervals."""
from __future__ import annotations

import numpy as np

from . import bootstrap as bs
from . import metrics as m


def _scalars(truth, score, predicted, head_probability, head_positive):
    """The statistics given intervals; kept small so bootstrapping stays fast."""
    referable = truth >= 2
    positive = predicted >= 2
    matrix = m.confusion_matrix(truth, predicted)
    recall = m.per_grade_recall(matrix)
    values = {
        'auc': m.auc(referable, score),
        'sensitivity_at_85_specificity': m.sensitivity_at_specificity(referable, score, 0.85),
        'sensitivity_at_90_specificity': m.sensitivity_at_specificity(referable, score, 0.90),
        'specificity_at_90_sensitivity': m.specificity_at_sensitivity(referable, score, 0.90),
        'sensitivity': m.binary_rates(referable, positive)['sensitivity'],
        'specificity': m.binary_rates(referable, positive)['specificity'],
        'qwk': m.quadratic_weighted_kappa(truth, predicted),
        'macro_f1': m.macro_f1(matrix),
        'exact_accuracy': float(np.mean(truth == predicted)),
    }
    for g in range(m.GRADES):
        values[f'recall_grade_{g}'] = float(recall[g])
    if head_positive is not None:
        rates = m.binary_rates(referable, head_positive)
        values['head_sensitivity'], values['head_specificity'] = rates['sensitivity'], rates['specificity']
    return values


def evaluate_grading(logits, grades, clusters, *, thresholds, head_threshold=None, replicates=1000, seed=20261020,
                     intervals=True) -> dict:
    """Full grading/referral report for one model on one cohort.

    ``logits`` has five grade logits plus the referral logit; ``thresholds`` are the four ordinal
    boundaries on the expected-grade score. Returns point estimates (and bootstrap intervals for the
    headline scalars) plus matrices and calibration. Evidence-tier stamping is done by ``report``.
    """
    logits, grades = np.asarray(logits, float), np.asarray(grades, int)
    if logits.shape != (len(grades), 6):
        raise ValueError(f'logits must be (N, 6); got {logits.shape} for {len(grades)} labels')
    clusters = np.asarray(clusters)
    score = m.expected_grade_score(logits)
    predicted = m.decode_thresholds(score, thresholds)
    head_probability = m.referral_head_probability(logits)
    head_positive = None if head_threshold is None else head_probability >= head_threshold
    referable, positive = grades >= 2, predicted >= 2
    matrix = m.confusion_matrix(grades, predicted)
    probabilities = m.grade_probabilities(logits)
    grade_head_referral = probabilities[:, 2:].sum(axis=1)
    report = {
        'n_images': int(len(grades)), 'n_clusters': int(len(np.unique(clusters))),
        'referable_prevalence': float(referable.mean()),
        'ranking': {'auc': m.auc(referable, score), 'pr_auc': m.pr_auc(referable, score),
                    'sensitivity_at_specificity': {str(t): m.sensitivity_at_specificity(referable, score, t) for t in (0.80, 0.85, 0.90, 0.95)},
                    'specificity_at_sensitivity': {str(t): m.specificity_at_sensitivity(referable, score, t) for t in (0.85, 0.90, 0.95)}},
        'referral_grade_policy': {**m.binary_rates(referable, positive), 'thresholds': [float(t) for t in thresholds]},
        'grading': {'confusion_matrix': matrix.tolist(), 'support': matrix.sum(axis=1).tolist(),
                    'recall': m.per_grade_recall(matrix).tolist(), 'precision': m.per_grade_precision(matrix).tolist(),
                    'f1': m.per_grade_f1(matrix).tolist(), 'macro_f1': m.macro_f1(matrix),
                    'qwk': m.quadratic_weighted_kappa(grades, predicted),
                    'mean_absolute_grade_error': m.mean_absolute_grade_error(grades, predicted),
                    'exact_accuracy': float(np.mean(grades == predicted)),
                    'off_by_two_or_more': float(np.mean(np.abs(grades - predicted) >= 2))},
        'calibration': {'ece_referral_head': m.ece(head_probability, referable),
                        'brier_referral_head': m.brier(head_probability, referable),
                        'ece_grade_head_referral': m.ece(grade_head_referral, referable),
                        'brier_grade_head_referral': m.brier(grade_head_referral, referable),
                        'top_label_ece': m.multiclass_ece(probabilities, grades)},
    }
    if head_positive is not None:
        report['referral_head'] = {**m.binary_rates(referable, head_positive), 'threshold': float(head_threshold),
                                   'disagreement_with_grade_policy': float(np.mean(head_positive != positive))}
    if intervals:
        report['intervals'] = bs.bootstrap(
            lambda rows: _scalars(grades[rows], score[rows], predicted[rows], head_probability[rows],
                                  None if head_positive is None else head_positive[rows]),
            clusters, replicates, seed)
    return report


def decision_scalars(truth, predicted, referral=None):
    """Statistics of one decision rule used by paired comparisons."""
    truth, predicted = np.asarray(truth, int), np.asarray(predicted, int)
    matrix = m.confusion_matrix(truth, predicted)
    recall, precision = m.per_grade_recall(matrix), m.per_grade_precision(matrix)
    positive = predicted >= 2 if referral is None else np.asarray(referral, bool)
    rates = m.binary_rates(truth >= 2, positive)
    values = {'qwk': m.quadratic_weighted_kappa(truth, predicted), 'macro_f1': m.macro_f1(matrix),
              'exact_accuracy': float(np.mean(truth == predicted)),
              'referral_sensitivity': rates['sensitivity'], 'referral_specificity': rates['specificity']}
    for g in range(m.GRADES):
        values[f'recall_grade_{g}'] = float(recall[g])
        values[f'precision_grade_{g}'] = float(precision[g])
    return values


def compare_grade_decisions(truth, predicted_a, predicted_b, clusters, *, referral_a=None, referral_b=None,
                            replicates=1000, seed=20261020) -> dict:
    """Paired class-wise trade-offs of decision rule A versus B on the same images.

    Reports, with paired cluster-bootstrap intervals, the change in every per-grade recall and
    precision, macro-F1, QWK and referral sensitivity/specificity, the two confusion matrices, and
    whether the referral decisions are identical (``referral_identical``).
    """
    truth, a, b = np.asarray(truth, int), np.asarray(predicted_a, int), np.asarray(predicted_b, int)
    ref_a = a >= 2 if referral_a is None else np.asarray(referral_a, bool)
    ref_b = b >= 2 if referral_b is None else np.asarray(referral_b, bool)
    out = {'n_images': int(len(truth)), 'referral_identical': bool(np.array_equal(ref_a, ref_b)),
           'referral_decisions_changed': int(np.sum(ref_a != ref_b)),
           'confusion_matrix_a': m.confusion_matrix(truth, a).tolist(),
           'confusion_matrix_b': m.confusion_matrix(truth, b).tolist()}
    out['paired'] = bs.paired_bootstrap(
        lambda rows: decision_scalars(truth[rows], a[rows], ref_a[rows]),
        lambda rows: decision_scalars(truth[rows], b[rows], ref_b[rows]), clusters, replicates, seed)
    return out
