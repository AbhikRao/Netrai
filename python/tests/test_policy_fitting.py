"""Tests for decision-policy fitting (thresholds, referral head, selection value, CLI guards)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import fit_validation_policy as fvp
from train_grading_adapter import (balanced_threshold, fit_head_threshold, fit_policy, midpoint_below,
                                   referral_threshold, scores, selection_value, sens_at_spec,
                                   youden_balanced_threshold)


def two_source_data(seed=0):
    """A large source with well separated scores and a small, noisier, shifted one."""
    rng = np.random.default_rng(seed)
    parts = []
    for name, size, shift, noise in (('big', 600, 0.0, 0.5), ('small', 120, 0.4, 0.7)):
        grade = rng.choice(5, size=size, p=[0.5, 0.1, 0.2, 0.1, 0.1])
        score = np.clip(grade + shift + rng.normal(0, noise, size), 0.0, 4.0)
        parts.append(pd.DataFrame({'score': score, 'grade': grade, 'source': name}))
    frame = pd.concat(parts, ignore_index=True)
    return frame.score.to_numpy(), frame.grade.to_numpy(), frame.source.to_numpy()


def source_average(score, labels, sources, threshold):
    names = [n for n in np.unique(sources)]
    sens = np.mean([np.mean(score[(sources == n) & (labels >= 2)] >= threshold) for n in names])
    spec = np.mean([np.mean(score[(sources == n) & (labels < 2)] < threshold) for n in names])
    return sens, spec


class MidpointTests(unittest.TestCase):
    def test_midpoint_sits_between_neighbouring_samples(self):
        values = np.array([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(midpoint_below(values, 2.0), 1.5)
        self.assertEqual(midpoint_below(values, 3.0), 2.5)

    def test_no_sample_equals_the_boundary(self):
        score, _, _ = two_source_data()
        for cut in np.unique(score)[1::50]:
            self.assertNotIn(midpoint_below(score, cut), set(score.tolist()))

    def test_smallest_value_falls_back_below_the_cut(self):
        values = np.array([1.0, 2.0])
        boundary = midpoint_below(values, 1.0)
        self.assertLess(boundary, 1.0)
        self.assertGreater(boundary, 1.0 - 1e-3)

    def test_decisions_on_fitting_data_are_unchanged_by_the_midpoint(self):
        values = np.array([0.1, 0.4, 0.9, 1.7])
        cut = 0.9
        boundary = midpoint_below(values, cut)
        self.assertEqual((values >= boundary).tolist(), (values >= cut).tolist())


class ThresholdRuleTests(unittest.TestCase):
    def setUp(self):
        self.score, self.grade, self.source = two_source_data()

    def test_referral_threshold_reaches_target_sensitivity(self):
        for target in (0.80, 0.90, 0.95):
            threshold = referral_threshold(self.score, self.grade, target)
            sens = np.mean(self.score[self.grade >= 2] >= threshold)
            self.assertGreaterEqual(sens, target - 1e-12)

    def test_balanced_threshold_reaches_source_averaged_target(self):
        threshold = balanced_threshold(self.score, self.grade, self.source, 0.90)
        sens, _ = source_average(self.score, self.grade, self.source, threshold)
        self.assertGreaterEqual(sens, 0.90 - 1e-12)
        self.assertNotIn(threshold, set(self.score.tolist()))

    def test_balanced_rule_is_not_dominated_by_the_large_source(self):
        pooled = referral_threshold(self.score, self.grade, 0.90)
        balanced = balanced_threshold(self.score, self.grade, self.source, 0.90)
        pooled_avg = source_average(self.score, self.grade, self.source, pooled)[0]
        balanced_avg = source_average(self.score, self.grade, self.source, balanced)[0]
        self.assertGreaterEqual(balanced_avg, 0.90 - 1e-12)
        # The pooled rule only guarantees the pooled sensitivity; the balanced rule guarantees the average.
        self.assertGreaterEqual(balanced_avg, min(pooled_avg, 0.90) - 1e-12)

    def test_youden_threshold_maximises_the_source_averaged_index(self):
        chosen = youden_balanced_threshold(self.score, self.grade, self.source)
        sens, spec = source_average(self.score, self.grade, self.source, chosen)
        best = max(sum(source_average(self.score, self.grade, self.source, cut)) - 1
                   for cut in np.unique(self.score))
        self.assertAlmostEqual(sens + spec - 1, best, places=9)

    def test_head_threshold_is_a_probability_and_meets_the_target(self):
        rng = np.random.default_rng(1)
        logits = np.zeros((len(self.grade), 6))
        logits[:, 5] = (self.grade >= 2) * 2.0 - 1.0 + rng.normal(0, 1.0, len(self.grade))
        for rule in ('pooled', 'balanced', 'youden-balanced'):
            threshold = fit_head_threshold(logits, self.grade, 0.90, self.source, rule)
            self.assertGreater(threshold, 0.0)
            self.assertLess(threshold, 1.0)
        pooled = fit_head_threshold(logits, self.grade, 0.90)
        probability = 1 / (1 + np.exp(-logits[:, 5]))
        self.assertGreaterEqual(np.mean(probability[self.grade >= 2] >= pooled), 0.90 - 1e-12)


class FitPolicyTests(unittest.TestCase):
    def setUp(self):
        self.score, self.grade, self.source = two_source_data(3)

    def test_policy_thresholds_are_increasing_and_inside_the_score_range(self):
        for rule in ('pooled', 'balanced', 'youden-balanced', 'worst-source'):
            thresholds = fit_policy(self.score, self.grade, 0.90, self.source, rule)
            self.assertEqual(len(thresholds), 4, rule)
            self.assertTrue(np.all(np.diff(thresholds) > 0), rule)
            self.assertGreaterEqual(thresholds[0], 0.0)
            self.assertLessEqual(thresholds[-1], 4.0)

    def test_grade_two_boundary_is_the_rule_selected_referral_boundary(self):
        balanced = fit_policy(self.score, self.grade, 0.90, self.source, 'balanced')
        self.assertEqual(balanced[1], balanced_threshold(self.score, self.grade, self.source, 0.90))
        pooled = fit_policy(self.score, self.grade, 0.90, self.source, 'pooled')
        self.assertEqual(pooled[1], referral_threshold(self.score, self.grade, 0.90))

    def test_worst_source_rule_is_at_least_as_conservative_as_pooled(self):
        worst = fit_policy(self.score, self.grade, 0.90, self.source, 'worst-source')
        pooled = fit_policy(self.score, self.grade, 0.90, self.source, 'pooled')
        self.assertLessEqual(worst[1], pooled[1] + 1e-9)


class SelectionValueTests(unittest.TestCase):
    def test_perfect_separation_scores_one(self):
        grade = np.array([0] * 50 + [3] * 50)
        score = grade.astype(float)
        self.assertEqual(sens_at_spec(score, grade, 0.85), 1.0)
        self.assertEqual(selection_value(score, grade), 1.0)

    def test_chance_ranking_is_far_from_one(self):
        rng = np.random.default_rng(2)
        grade = rng.integers(0, 5, 2000)
        self.assertLess(selection_value(rng.random(2000), grade), 0.4)

    def test_scores_are_the_expected_grade(self):
        logits = np.full((2, 6), -30.0)
        logits[0, 4] = 30.0
        logits[1, 1] = 30.0
        np.testing.assert_allclose(scores(logits), [4.0, 1.0], atol=1e-6)


class ValidationPolicyCliTests(unittest.TestCase):
    def make_run(self, base, name, rows, seed=0):
        run = base / name
        run.mkdir()
        rows.to_csv(run / 'validation_rows.csv', index=False)
        rng = np.random.default_rng(seed)
        logits = rng.normal(0, 1, (len(rows), 6))
        logits[np.arange(len(rows)), rows.grade.to_numpy()] += 4.0
        logits[:, 5] = (rows.grade.to_numpy() >= 2) * 4.0 - 2.0 + rng.normal(0, 1, len(rows))
        np.save(run / 'best_validation_outputs.npy', logits)
        return run

    def rows(self, ids):
        rng = np.random.default_rng(4)
        grade = rng.choice(5, size=len(ids), p=[0.4, 0.15, 0.25, 0.1, 0.1])
        return pd.DataFrame({'image_id': ids, 'grade': grade, 'source': ['a', 'b'] * (len(ids) // 2),
                             'group_id': ids})

    def run_cli(self, *argv):
        with mock.patch.object(sys, 'argv', ['fit_validation_policy.py', *map(str, argv)]), \
                contextlib.redirect_stdout(io.StringIO()):
            fvp.main()

    def test_happy_path_writes_a_four_threshold_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = self.make_run(base, 'run1', self.rows([f'i{k}' for k in range(400)]))
            out = base / 'policy.json'
            self.run_cli(run, '--output', out)
            policy = json.loads(out.read_text())
            self.assertEqual(len(policy['thresholds']), 4)
            self.assertTrue(0 < policy['head_threshold'] < 1)
            self.assertEqual(policy['rule'], 'balanced')
            self.assertEqual(policy['target_sensitivity'], 0.90)

    def test_refuses_runs_with_different_validation_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = self.make_run(base, 'run1', self.rows([f'i{k}' for k in range(200)]))
            second = self.make_run(base, 'run2', self.rows([f'j{k}' for k in range(200)]))
            with self.assertRaisesRegex(ValueError, 'differ'):
                self.run_cli(first, second, '--output', base / 'policy.json')
            self.assertFalse((base / 'policy.json').exists())

    def test_refuses_to_overwrite_an_existing_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = self.make_run(base, 'run1', self.rows([f'i{k}' for k in range(200)]))
            out = base / 'policy.json'
            out.write_text('{}')
            with self.assertRaises(FileExistsError):
                self.run_cli(run, '--output', out)
            self.assertEqual(out.read_text(), '{}')

    def test_ensemble_of_identical_runs_equals_the_single_run_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rows = self.rows([f'i{k}' for k in range(400)])
            first = self.make_run(base, 'run1', rows)
            second = base / 'run2'
            second.mkdir()
            rows.to_csv(second / 'validation_rows.csv', index=False)
            np.save(second / 'best_validation_outputs.npy', np.load(first / 'best_validation_outputs.npy'))
            self.run_cli(first, '--output', base / 'single.json')
            self.run_cli(first, second, '--output', base / 'ensemble.json')
            single = json.loads((base / 'single.json').read_text())
            ensemble = json.loads((base / 'ensemble.json').read_text())
            self.assertEqual(single['thresholds'], ensemble['thresholds'])
            self.assertEqual(single['head_threshold'], ensemble['head_threshold'])


if __name__ == '__main__':
    unittest.main()
