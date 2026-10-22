"""Tests for the unified evaluation harness (metrics, cluster bootstrap, grading, tiered reports)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

import locked_evaluation as le
from netrai_eval import bootstrap as bs
from netrai_eval import grading, metrics as m, report
from utils.metrics import quadratic_weighted_kappa as legacy_qwk

THRESHOLDS = np.array([0.5, 1.5, 2.5, 3.5])


def noisy_cohort(n=600, seed=0, clusters=True):
    rng = np.random.default_rng(seed)
    grades = rng.choice(5, size=n, p=[0.45, 0.15, 0.2, 0.1, 0.1])
    logits = np.full((n, 6), 0.0)
    logits[np.arange(n), grades] = 3.0
    logits += rng.normal(0, 1.5, (n, 6))
    logits[:, 5] = (grades >= 2) * 2.0 - 1.0 + rng.normal(0, 1.5, n)
    ids = np.repeat(np.arange(n // 2), 2) if clusters else np.arange(n)
    return logits, grades, ids


class MetricTests(unittest.TestCase):
    def test_ranking_metrics_match_sklearn(self):
        logits, grades, _ = noisy_cohort()
        score, referable = m.expected_grade_score(logits), grades >= 2
        self.assertAlmostEqual(m.auc(referable, score), roc_auc_score(referable, score), places=12)
        self.assertAlmostEqual(m.pr_auc(referable, score), average_precision_score(referable, score), places=12)

    def test_agrees_with_the_locked_evaluation_code(self):
        logits, grades, _ = noisy_cohort(seed=3)
        locked, _ = le.metrics(logits, grades, THRESHOLDS, 0.5)
        score = m.expected_grade_score(logits)
        predicted = m.decode_thresholds(score, THRESHOLDS)
        rates = m.binary_rates(grades >= 2, predicted >= 2)
        matrix = m.confusion_matrix(grades, predicted)
        self.assertAlmostEqual(rates['sensitivity'], locked['sensitivity'], places=12)
        self.assertAlmostEqual(rates['specificity'], locked['specificity'], places=12)
        self.assertAlmostEqual(m.auc(grades >= 2, score), locked['auc_expected_grade_score'], places=12)
        self.assertAlmostEqual(m.sensitivity_at_specificity(grades >= 2, score, 0.85),
                               locked['sensitivity_at_85_specificity'], places=12)
        self.assertAlmostEqual(m.quadratic_weighted_kappa(grades, predicted), locked['qwk'], places=12)
        np.testing.assert_allclose(m.per_grade_recall(matrix), locked['per_grade_recall'], atol=1e-12)
        self.assertEqual(matrix.sum(axis=1).tolist(), locked['per_grade_support'])

    def test_sensitivity_and_specificity_at_targets_on_a_hand_checked_example(self):
        truth = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], bool)
        score = np.array([0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])
        # thresholds above 0.5 keep all 5 negatives correct: sensitivity 4/5 at specificity 5/5? one positive (0.45) is lost
        self.assertAlmostEqual(m.sensitivity_at_specificity(truth, score, 1.0), 0.8)
        self.assertAlmostEqual(m.sensitivity_at_specificity(truth, score, 0.8), 1.0)
        self.assertAlmostEqual(m.specificity_at_sensitivity(truth, score, 1.0), 0.8)
        self.assertAlmostEqual(m.specificity_at_sensitivity(truth, score, 0.8), 1.0)

    def test_undefined_cases_are_nan_not_zero(self):
        self.assertTrue(np.isnan(m.auc(np.ones(5, bool), np.arange(5.0))))
        self.assertTrue(np.isnan(m.sensitivity_at_specificity(np.zeros(5, bool), np.arange(5.0), 0.9)))
        self.assertTrue(np.isnan(m.binary_rates(np.zeros(4, bool), np.zeros(4, bool))['sensitivity']))

    def test_confusion_recall_precision_f1(self):
        truth = np.array([0, 0, 0, 1, 1, 2, 3, 4, 4, 4])
        pred = np.array([0, 0, 1, 1, 0, 2, 4, 4, 4, 3])
        matrix = m.confusion_matrix(truth, pred)
        self.assertEqual(matrix.sum(), 10)
        np.testing.assert_allclose(m.per_grade_recall(matrix), [2 / 3, 1 / 2, 1.0, 0.0, 2 / 3])
        np.testing.assert_allclose(m.per_grade_precision(matrix), [2 / 3, 1 / 2, 1.0, 0.0, 2 / 3])
        self.assertAlmostEqual(m.macro_f1(matrix), np.mean([2 / 3, 1 / 2, 1.0, 0.0, 2 / 3]))

    def test_absent_grades_are_skipped_in_macro_f1(self):
        matrix = m.confusion_matrix(np.array([0, 0, 2]), np.array([0, 0, 2]))
        self.assertAlmostEqual(m.macro_f1(matrix), 1.0)

    def test_qwk_matches_sklearn_and_the_legacy_function_when_all_grades_occur(self):
        logits, grades, _ = noisy_cohort(seed=4)
        pred = m.decode_thresholds(m.expected_grade_score(logits), THRESHOLDS)
        value = m.quadratic_weighted_kappa(grades, pred)
        self.assertAlmostEqual(value, cohen_kappa_score(grades, pred, weights='quadratic', labels=range(5)), places=12)
        self.assertAlmostEqual(value, legacy_qwk(grades, pred), places=12)

    def test_qwk_uses_the_fixed_range_when_grades_are_missing(self):
        truth, pred = np.array([0, 1, 2, 2, 1, 0]), np.array([0, 1, 2, 1, 1, 0])
        self.assertAlmostEqual(m.quadratic_weighted_kappa(truth, pred),
                               cohen_kappa_score(truth, pred, weights='quadratic', labels=range(5)), places=12)
        self.assertEqual(m.quadratic_weighted_kappa(truth, truth), 1.0)

    def test_ece_and_brier_known_values(self):
        outcome = np.array([1, 0] * 50, bool)
        self.assertAlmostEqual(m.ece(np.full(100, 0.9), outcome), 0.4, places=12)
        self.assertAlmostEqual(m.ece(np.full(100, 0.5), outcome), 0.0, places=12)
        self.assertAlmostEqual(m.brier(np.full(100, 0.5), outcome), 0.25)

    def test_ece_of_a_calibrated_model_is_small(self):
        rng = np.random.default_rng(1)
        probability = rng.random(20000)
        outcome = rng.random(20000) < probability
        self.assertLess(m.ece(probability, outcome), 0.02)


class BootstrapTests(unittest.TestCase):
    def test_cluster_members_group_rows(self):
        members = bs.cluster_members(['a', 'b', 'a', 'c', 'b', 'a'])
        self.assertEqual(sorted(sorted(g.tolist()) for g in members), [[0, 2, 5], [1, 4], [3]])

    def test_resampling_keeps_clusters_whole(self):
        clusters = np.repeat(np.arange(20), 3)
        members = bs.cluster_members(clusters)
        rows = bs.resample(members, np.random.default_rng(0))
        counts = np.bincount(clusters[rows], minlength=20)
        self.assertTrue(np.all(counts % 3 == 0))  # each cluster contributes all 3 rows, a whole number of times

    def test_seeded_determinism_and_interval_contains_estimate(self):
        logits, grades, ids = noisy_cohort(n=400)
        score = m.expected_grade_score(logits)
        fn = lambda rows: {'auc': m.auc(grades[rows] >= 2, score[rows])}
        first, second = bs.bootstrap(fn, ids, 200, seed=9), bs.bootstrap(fn, ids, 200, seed=9)
        self.assertEqual(first, second)
        low, high = first['auc']['ci95']
        self.assertLessEqual(low, first['auc']['estimate'])
        self.assertGreaterEqual(high, first['auc']['estimate'])
        self.assertNotEqual(bs.bootstrap(fn, ids, 200, seed=10), first)

    def test_interval_narrows_with_more_data(self):
        def width(n):
            logits, grades, ids = noisy_cohort(n=n, seed=2)
            score = m.expected_grade_score(logits)
            out = bs.bootstrap(lambda r: {'auc': m.auc(grades[r] >= 2, score[r])}, ids, 200, seed=1)
            return out['auc']['ci95'][1] - out['auc']['ci95'][0]
        self.assertLess(width(2400), width(300))

    def test_paired_identical_models_have_zero_difference(self):
        logits, grades, ids = noisy_cohort(n=300)
        score = m.expected_grade_score(logits)
        fn = lambda rows: {'auc': m.auc(grades[rows] >= 2, score[rows])}
        result = bs.paired_bootstrap(fn, fn, ids, 100, seed=3)
        self.assertEqual(result['auc']['difference'], 0.0)
        self.assertEqual(result['auc']['difference_ci95'], [0.0, 0.0])

    def test_paired_detects_a_real_difference(self):
        logits, grades, ids = noisy_cohort(n=800, seed=5)
        good = m.expected_grade_score(logits)
        bad = good + np.random.default_rng(7).normal(0, 2.0, len(good))
        result = bs.paired_bootstrap(lambda r: {'auc': m.auc(grades[r] >= 2, good[r])},
                                     lambda r: {'auc': m.auc(grades[r] >= 2, bad[r])}, ids, 200, seed=3)
        self.assertGreater(result['auc']['difference'], 0.05)
        self.assertGreater(result['auc']['difference_ci95'][0], 0.0)

    def test_mismatched_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            bs.paired_bootstrap(lambda r: {'a': 1.0}, lambda r: {'b': 1.0}, [0, 1], 5)


class GradingTests(unittest.TestCase):
    def test_full_report_structure_and_consistency(self):
        logits, grades, ids = noisy_cohort()
        result = grading.evaluate_grading(logits, grades, ids, thresholds=THRESHOLDS, head_threshold=0.5,
                                          replicates=50, seed=1)
        self.assertEqual(result['n_images'], 600)
        self.assertEqual(result['n_clusters'], 300)
        locked, _ = le.metrics(logits, grades, THRESHOLDS, 0.5)
        self.assertAlmostEqual(result['referral_grade_policy']['sensitivity'], locked['sensitivity'])
        self.assertAlmostEqual(result['referral_head']['disagreement_with_grade_policy'],
                               locked['head_disagreement_rate'])
        self.assertEqual(np.array(result['grading']['confusion_matrix']).sum(), 600)
        for key in ('auc', 'qwk', 'macro_f1', 'recall_grade_1', 'head_sensitivity', 'sensitivity_at_85_specificity'):
            self.assertIn(key, result['intervals'])
            low, high = result['intervals'][key]['ci95']
            self.assertLessEqual(low, high)
        self.assertGreaterEqual(result['calibration']['ece_referral_head'], 0.0)

    def test_bad_logit_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            grading.evaluate_grading(np.zeros((10, 5)), np.zeros(10, int), np.arange(10), thresholds=THRESHOLDS)

    def test_identical_decisions_have_no_tradeoff(self):
        logits, grades, ids = noisy_cohort(n=400)
        pred = m.decode_thresholds(m.expected_grade_score(logits), THRESHOLDS)
        result = grading.compare_grade_decisions(grades, pred, pred, ids, replicates=30, seed=2)
        self.assertTrue(result['referral_identical'])
        self.assertEqual(result['referral_decisions_changed'], 0)
        for stats in result['paired'].values():
            self.assertEqual(stats['difference'], 0.0)

    def test_swapping_grades_within_the_same_referral_side_preserves_referral(self):
        logits, grades, ids = noisy_cohort(n=800, seed=8)
        a = m.decode_thresholds(m.expected_grade_score(logits), THRESHOLDS)
        b = a.copy()
        b[a == 0] = np.where(np.random.default_rng(0).random((a == 0).sum()) < 0.3, 1, 0)  # some 0 -> 1
        result = grading.compare_grade_decisions(grades, a, b, ids, replicates=60, seed=2)
        self.assertTrue(result['referral_identical'])
        paired = result['paired']
        self.assertGreater(paired['recall_grade_0']['difference'], 0.0)  # a keeps more grade-0 eyes than b
        self.assertEqual(paired['referral_sensitivity']['difference'], 0.0)
        self.assertEqual(paired['referral_specificity']['difference'], 0.0)

    def test_changed_referral_decisions_are_flagged(self):
        logits, grades, ids = noisy_cohort(n=400, seed=9)
        a = m.decode_thresholds(m.expected_grade_score(logits), THRESHOLDS)
        b = a.copy()
        b[(a == 1)] = 2  # promote grade-1 predictions into the referable range
        result = grading.compare_grade_decisions(grades, a, b, ids, replicates=20, seed=2)
        self.assertFalse(result['referral_identical'])
        self.assertEqual(result['referral_decisions_changed'], int((a == 1).sum()))


class ReportTests(unittest.TestCase):
    def make(self, **overrides):
        args = dict(tier='TD', cohort='c', model='m', metrics={'auc': float('nan')}, n_images=10, n_clusters=5,
                    limitations=['development data'])
        args.update(overrides)
        return report.build_report(**args)

    def test_every_tier_has_a_statement(self):
        self.assertEqual(set(report.TIERS), {'T0', 'TD', 'TL', 'TX', 'TF'})
        for tier in ('T0', 'TD', 'TL', 'TX'):
            self.assertEqual(self.make(tier=tier)['tier_statement'], report.TIERS[tier])

    def test_unknown_tier_rejected(self):
        with self.assertRaises(ValueError):
            self.make(tier='T9')

    def test_fresh_tier_requires_a_lock_and_other_tiers_forbid_one(self):
        with self.assertRaises(ValueError):
            self.make(tier='TF')
        self.assertEqual(self.make(tier='TF', lock_sha256='abc')['lock_sha256'], 'abc')
        with self.assertRaises(ValueError):
            self.make(tier='TX', lock_sha256='abc')

    def test_limitations_are_mandatory(self):
        with self.assertRaises(ValueError):
            self.make(limitations=[])

    def test_tampered_statement_is_detected(self):
        bad = self.make()
        bad['tier_statement'] = 'Fresh independent test.'
        with self.assertRaises(ValueError):
            report.validate_report(bad)

    def test_write_is_immutable_and_nan_becomes_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'r.json'
            report.write_report(path, self.make())
            self.assertIsNone(json.loads(path.read_text())['metrics']['auc'])
            self.assertEqual(report.read_report(path)['tier'], 'TD')
            with self.assertRaises(FileExistsError):
                report.write_report(path, self.make())


if __name__ == '__main__':
    unittest.main()
