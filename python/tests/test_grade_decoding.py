"""Tests for the X1 grade decoders (referral-preserving, cross-fitted)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import grade_decoding as gd
from netrai_eval import metrics as m


def synthetic(n=1500, seed=0, suppress_grade1=0.25, sources=3):
    """Class probabilities from a noisy model that systematically under-predicts grade 1."""
    rng = np.random.default_rng(seed)
    truth = rng.choice(5, size=n, p=[0.45, 0.2, 0.18, 0.1, 0.07])
    logits = rng.normal(0, 1.0, (n, 5))
    logits[np.arange(n), truth] += 2.2
    # neighbouring grades are confusable (ordinal structure)
    for shift in (-1, 1):
        neighbour = np.clip(truth + shift, 0, 4)
        logits[np.arange(n), neighbour] += 1.0
    logits[:, 1] += np.log(suppress_grade1)
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    source = np.array([f's{i % sources}' for i in range(n)])
    clusters = np.array([f'{source[i]}_c{i // 2}' for i in range(n)])  # two rows per cluster, same source
    return p, truth, source, clusters


class RestrictedArgmaxTests(unittest.TestCase):
    def test_sides_are_respected_and_referral_is_the_mask(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(5), 500)
        referable = rng.random(500) < 0.4
        pred = gd.restricted_argmax(p, referable)
        self.assertTrue(np.array_equal(pred >= 2, referable))
        self.assertTrue(set(pred[~referable]) <= {0, 1})
        self.assertTrue(set(pred[referable]) <= {2, 3, 4})
        self.assertTrue(np.array_equal(pred[~referable], p[~referable][:, :2].argmax(1)))
        self.assertTrue(np.array_equal(pred[referable], p[referable][:, 2:].argmax(1) + 2))

    def test_bias_moves_predictions_toward_the_biased_class(self):
        rng = np.random.default_rng(2)
        p = rng.dirichlet(np.ones(5), 400)
        referable = rng.random(400) < 0.5
        plain = gd.restricted_argmax(p, referable)
        biased = gd.restricted_argmax(p, referable, np.array([0, 10.0, 0, 0, 0]))
        self.assertTrue(np.all(biased[~referable] == 1))
        self.assertTrue(np.array_equal(biased[referable], plain[referable]))
        self.assertTrue(np.array_equal(gd.restricted_argmax(p, referable, np.zeros(5)), plain))


class FitTests(unittest.TestCase):
    def setUp(self):
        self.p, self.truth, self.sources, self.clusters = synthetic()
        self.score = gd.expected_score(self.p)
        self.t_star = float(np.quantile(self.score, 0.55))
        self.referable = self.score >= self.t_star

    def test_bias_fit_recovers_the_suppressed_grade_and_improves_the_objective(self):
        zero = gd.restricted_argmax(self.p, self.referable)
        bias = gd.fit_bias(self.p, self.truth, self.sources, self.referable)
        fitted = gd.restricted_argmax(self.p, self.referable, bias)
        self.assertGreater(bias[1], 0.0)  # grade 1 was under-predicted
        self.assertEqual(bias[0], 0.0)
        self.assertEqual(bias[2], 0.0)
        self.assertGreaterEqual(gd.balanced_macro_f1(self.truth, fitted, self.sources),
                                gd.balanced_macro_f1(self.truth, zero, self.sources))
        recall = lambda pred: m.per_grade_recall(m.confusion_matrix(self.truth, pred))[1]
        self.assertGreater(recall(fitted), recall(zero))
        self.assertTrue(np.array_equal(fitted >= 2, self.referable))

    def test_bias_values_come_from_the_grid(self):
        bias = gd.fit_bias(self.p, self.truth, self.sources, self.referable)
        for value in bias:
            self.assertIn(round(float(value), 6), [round(float(g), 6) for g in gd.BIAS_GRID])

    def test_score_threshold_fit_keeps_the_referral_boundary_and_does_not_hurt_the_objective(self):
        start = np.array([self.t_star / 2, self.t_star, self.t_star + (4 - self.t_star) / 3,
                          self.t_star + 2 * (4 - self.t_star) / 3])
        fitted = gd.fit_score_thresholds(self.score, self.truth, self.sources, self.t_star)
        self.assertEqual(fitted[1], self.t_star)
        self.assertTrue(np.all(np.diff(fitted) > 0))
        self.assertGreater(fitted[0], 0.0)
        self.assertLess(fitted[3], 4.0)
        before = gd.balanced_macro_f1(self.truth, gd.decode_thresholds(self.score, start), self.sources)
        after = gd.balanced_macro_f1(self.truth, gd.decode_thresholds(self.score, fitted), self.sources)
        self.assertGreaterEqual(after, before)
        self.assertTrue(np.array_equal(gd.decode_thresholds(self.score, fitted) >= 2, self.referable))

    def test_qwk_threshold_fit_keeps_the_referral_boundary(self):
        fitted = gd.fit_qwk_thresholds(self.score, self.truth, self.t_star)
        self.assertEqual(fitted[1], self.t_star)
        self.assertTrue(np.all(np.diff(fitted) > 0))
        start = np.array([self.t_star / 2, self.t_star, self.t_star + (4 - self.t_star) / 3,
                          self.t_star + 2 * (4 - self.t_star) / 3])
        self.assertGreaterEqual(m.quadratic_weighted_kappa(self.truth, gd.decode_thresholds(self.score, fitted)),
                                m.quadratic_weighted_kappa(self.truth, gd.decode_thresholds(self.score, start)))

    def test_bad_referral_boundary_is_rejected(self):
        for t in (0.0, 4.0, -1.0):
            with self.assertRaises(ValueError):
                gd.fit_score_thresholds(self.score, self.truth, self.sources, t)
            with self.assertRaises(ValueError):
                gd.fit_qwk_thresholds(self.score, self.truth, t)


class BalancedObjectiveTests(unittest.TestCase):
    def test_single_source_equals_plain_macro_f1(self):
        truth = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        pred = np.array([0, 1, 2, 3, 3, 0, 0, 2])
        sources = np.array(['a'] * 8)
        self.assertAlmostEqual(gd.balanced_macro_f1(truth, pred, sources),
                               m.macro_f1(m.confusion_matrix(truth, pred)))

    def test_sources_are_averaged_not_pooled(self):
        truth = np.array([0, 1] * 50 + [0, 1])
        good = truth.copy()
        pred = good.copy()
        pred[-2:] = [1, 0]  # the tiny source is completely wrong
        sources = np.array(['big'] * 100 + ['tiny'] * 2)
        per_source = [m.macro_f1(m.confusion_matrix(truth[sources == s], pred[sources == s])) for s in ('big', 'tiny')]
        self.assertAlmostEqual(gd.balanced_macro_f1(truth, pred, sources), float(np.mean(per_source)))


class FoldTests(unittest.TestCase):
    def test_clusters_stay_whole_and_sources_cover_every_fold(self):
        _, _, sources, clusters = synthetic(n=900)
        folds = gd.make_folds(clusters, sources, 5, seed=3)
        frame = pd.DataFrame({'c': clusters, 's': sources, 'f': folds})
        self.assertTrue((frame.groupby('c').f.nunique() == 1).all())
        self.assertEqual(set(folds), set(range(5)))
        for source in np.unique(sources):
            self.assertEqual(set(frame[frame.s == source].f), set(range(5)))

    def test_deterministic_and_seed_dependent(self):
        _, _, sources, clusters = synthetic(n=600)
        a, b = gd.make_folds(clusters, sources, 5, 1), gd.make_folds(clusters, sources, 5, 1)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, gd.make_folds(clusters, sources, 5, 2)))

    def test_a_cluster_spanning_two_sources_is_rejected(self):
        with self.assertRaises(ValueError):
            gd.make_folds(np.array(['x', 'x', 'y', 'y']), np.array(['a', 'b', 'a', 'a']), 2, 0)


class CrossfitTests(unittest.TestCase):
    def setUp(self):
        self.p, self.truth, self.sources, self.clusters = synthetic(n=1200, seed=4)
        score = gd.expected_score(self.p)
        t_star = float(np.quantile(score, 0.55))
        self.policy = [t_star / 2, t_star, t_star + 0.4, t_star + 0.9]

    def test_every_row_is_decoded_and_referral_never_changes(self):
        result = gd.crossfit(self.p, self.truth, self.sources, self.clusters, self.policy, k=4, seed=1)
        referable = result['referable']
        for name, predicted in result['predictions'].items():
            self.assertTrue(np.array_equal(predicted >= 2, referable), name)
            self.assertTrue(((predicted >= 0) & (predicted <= 4)).all(), name)

    def test_fit_and_evaluation_rows_are_disjoint_and_cover_everything(self):
        seen = []
        gd.crossfit(self.p, self.truth, self.sources, self.clusters, self.policy, k=4, seed=1,
                    fitter=lambda fit, ev: seen.append((set(fit.tolist()), set(ev.tolist()))))
        self.assertEqual(len(seen), 4)
        everyone = set(range(len(self.truth)))
        for fit, ev in seen:
            self.assertFalse(fit & ev)
            self.assertEqual(fit | ev, everyone)
        evaluated = [row for _, ev in seen for row in ev]
        self.assertEqual(sorted(evaluated), list(range(len(self.truth))))
        # clusters never straddle fit and eval
        for fit, ev in seen:
            self.assertFalse({self.clusters[i] for i in fit} & {self.clusters[i] for i in ev})

    def test_decoder_A_is_the_shipped_policy_and_B_needs_no_fitting(self):
        result = gd.crossfit(self.p, self.truth, self.sources, self.clusters, self.policy, k=3, seed=1)
        score = gd.expected_score(self.p)
        self.assertTrue(np.array_equal(result['predictions']['A'], gd.decode_thresholds(score, self.policy)))
        self.assertTrue(np.array_equal(result['predictions']['B'], gd.restricted_argmax(self.p, result['referable'])))

    def test_fitted_decoders_recover_grade_one_on_held_out_folds(self):
        result = gd.crossfit(self.p, self.truth, self.sources, self.clusters, self.policy, k=5, seed=2)
        recall = lambda name: m.per_grade_recall(m.confusion_matrix(self.truth, result['predictions'][name]))[1]
        self.assertGreater(recall('C'), recall('A_cv') + 0.05)
        self.assertGreater(recall('E'), recall('A_cv') + 0.0)


class AssessmentTests(unittest.TestCase):
    def tables(self, recall1, macro, qwk, r0, hi):
        return {'recall': [r0, recall1, hi, hi, hi], 'macro_f1': macro, 'qwk': qwk, 'support': [100, 60, 40, 20, 10]}

    def per_source(self, candidate_grade1):
        base = {'support': [100, 60, 40, 20, 10], 'recall': [0.9, 0.1, 0.6, 0.6, 0.6]}
        return {s: {'A_cv': base, 'decoder': {'recall': [0.9, candidate_grade1, .6, .6, .6], 'support': [100, 60, 40, 20, 10]}}
                for s in ('a', 'b', 'c', 'd')}

    def paired(self, gain, ci_low, qwk_delta):
        return {'macro_f1': {'difference': gain, 'difference_ci95': [ci_low, gain + 0.02]},
                'qwk': {'difference': qwk_delta, 'difference_ci95': [qwk_delta - 0.01, qwk_delta + 0.01]}}

    def test_all_criteria_met(self):
        out = gd.assess(self.tables(0.35, 0.5, 0.7, 0.88, 0.6), self.tables(0.1, 0.45, 0.7, 0.9, 0.6),
                        self.paired(0.04, 0.01, 0.0), self.per_source(0.4), 0)
        self.assertTrue(out['all_met'], out['checks'])

    def test_each_criterion_can_fail_independently(self):
        base = self.tables(0.1, 0.45, 0.7, 0.9, 0.6)
        good = (self.tables(0.35, 0.5, 0.7, 0.88, 0.6), base, self.paired(0.04, 0.01, 0.0), self.per_source(0.4), 0)
        cases = {
            '1_referral_identical': (good[0], good[1], good[2], good[3], 2),
            '2a_grade1_recall_pooled_ge_0.30': (self.tables(0.2, 0.5, 0.7, 0.88, 0.6), base, good[2], good[3], 0),
            '2b_grade1_recall_ge_0.30_in_>=3_eligible_sources': (good[0], base, good[2], self.per_source(0.2), 0),
            '3_macro_f1_gain_ge_0.03_ci_excludes_0': (good[0], base, self.paired(0.04, -0.01, 0.0), good[3], 0),
            '4_qwk_drop_le_0.01': (good[0], base, self.paired(0.04, 0.01, -0.05), good[3], 0),
            '5_grade0_recall_drop_le_0.05': (self.tables(0.35, 0.5, 0.7, 0.8, 0.6), base, good[2], good[3], 0),
            '6_referable_grade_recall_drops_le_0.05': (self.tables(0.35, 0.5, 0.7, 0.88, 0.5), base, good[2], good[3], 0),
        }
        for failing, args in cases.items():
            with self.subTest(failing=failing):
                out = gd.assess(*args)
                self.assertFalse(out['checks'][failing])
                self.assertFalse(out['all_met'])

    def test_small_sources_are_not_eligible_for_the_per_source_criterion(self):
        sources = self.per_source(0.4)
        sources['a']['A_cv']['support'] = [100, 5, 40, 20, 10]
        out = gd.assess(self.tables(0.35, 0.5, 0.7, 0.88, 0.6), self.tables(0.1, 0.45, 0.7, 0.9, 0.6),
                        self.paired(0.04, 0.01, 0.0), sources, 0)
        self.assertNotIn('a', out['eligible_sources'])


class EndToEndTests(unittest.TestCase):
    def test_evaluate_model_runs_on_saved_run_files(self):
        p, truth, sources, clusters = synthetic(n=900, seed=7)
        outputs = np.concatenate([np.log(p), np.zeros((len(p), 1))], axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = base / 'run'
            run.mkdir()
            pd.DataFrame({'image_id': [f'i{k}' for k in range(len(truth))], 'grade': truth, 'source': sources,
                          'group_id': clusters}).to_csv(run / 'validation_rows.csv', index=False)
            np.save(run / 'best_validation_outputs.npy', outputs)
            score = gd.expected_score(p)
            t_star = float(np.quantile(score, 0.55))
            (base / 'policy.json').write_text(json.dumps({'thresholds': [t_star / 2, t_star, t_star + 0.4, t_star + 0.9]}))
            with mock.patch.object(gd, 'MI', base):
                result = gd.evaluate_model({'id': 'toy', 'runs': ['run'], 'policy': 'policy.json'},
                                           repeats=2, folds=4, replicates=20, seed=5)
        self.assertEqual(result['n_images'], 900)
        self.assertEqual(len(result['repeats']), 2)
        primary = result['repeats'][0]
        self.assertEqual(set(primary['pooled']), {'A', 'A_cv', 'B', 'C', 'E'})
        self.assertEqual(set(primary['assessment']), {'B', 'C', 'E'})
        for name in ('B', 'C', 'E'):
            self.assertEqual(primary['paired_vs_A_cv'][name]['referral_decisions_changed'], 0)
        self.assertEqual(set(primary['per_source']), {'s0', 's1', 's2'})
        self.assertNotIn('assessment', result['repeats'][1])


if __name__ == '__main__':
    unittest.main()
