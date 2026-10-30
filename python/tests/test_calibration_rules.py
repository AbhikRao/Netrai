"""Tests for the site-calibration rules: R1 reproduces the pre-registered procedure, the keep-frozen rules and the selection rule."""
import unittest

import numpy as np

import calibration_rules as cr
import locked_fresh2 as lf


def synthetic(n=1200, prevalence=0.3, separation=2.0, seed=0):
    rng = np.random.default_rng(seed)
    referable = rng.random(n) < prevalence
    score = np.where(referable, rng.normal(1.0 + separation / 2, 0.5, n), rng.normal(1.0 - separation / 2, 0.5, n))
    clusters = np.repeat(np.arange(n // 2), 2)[:n].astype(str)
    return score, referable, clusters


class CalibrationRuleTests(unittest.TestCase):
    def test_r1_reproduces_the_locked_procedure_on_the_same_draws(self):
        score, referable, clusters = synthetic()
        mine = cr.rule_draws(score, referable, clusters, frozen=1.3, size=200, draws=40, seed=7, rules=('R1',))['R1']
        locked = lf.calibration_draws(score, referable, clusters, 1.3, 200, 40, 7)
        for key in ('p_both_targets', 'sensitivity_mean', 'specificity_mean', 'sensitivity_p10', 'specificity_p10', 'frozen_sensitivity'):
            self.assertAlmostEqual(mine[key], locked[key], places=12, msg=key)

    def test_keep_frozen_rules_keep_a_passing_cut_and_refit_a_failing_one(self):
        rng = np.random.default_rng(1)
        referable = np.r_[np.ones(50, bool), np.zeros(150, bool)]
        good = np.where(referable, rng.uniform(1.5, 3, 200), rng.uniform(0, 0.9, 200))      # frozen 1.0 separates perfectly
        self.assertEqual(cr.fit_cut('R3', good, referable, 1.0), 1.0)
        self.assertNotEqual(cr.fit_cut('R1', good, referable, 1.0), 1.0)                     # R1 always moves the cut
        bad = np.where(referable, rng.uniform(1.5, 3, 200), rng.uniform(1.1, 1.4, 200))      # frozen 1.0 refers every healthy eye
        self.assertGreater(cr.fit_cut('R3', bad, referable, 1.0), 1.0)
        few = np.r_[np.ones(3, bool), np.zeros(100, bool)]
        self.assertEqual(cr.fit_cut('R1', np.linspace(0, 3, 103), few, 1.0), 1.0)          # too few referable: frozen kept

    def test_selection_rule(self):
        site = lambda p, s: {'p_both_targets': p, 'sensitivity_mean': s}
        per_site = {'a': {'R1': site(0.50, 0.91), 'R2': site(0.40, 0.95), 'R3': site(0.70, 0.93), 'R4': site(0.60, 0.95)},
                    'b': {'R1': site(0.50, 0.91), 'R2': site(0.40, 0.95), 'R3': site(0.69, 0.93), 'R4': site(0.62, 0.96)}}
        self.assertEqual(cr.select_rule(per_site)['selected'], 'R3')
        per_site['b']['R3'] = site(0.69, 0.88)                                               # violates the sensitivity floor
        self.assertEqual(cr.select_rule(per_site)['selected'], 'R4')
        tie = {'a': {'R1': site(0.70, 0.91), 'R3': site(0.71, 0.92)}}
        self.assertEqual(cr.select_rule(tie)['selected'], 'R1')                              # within 0.02: preference order


class SecondaryVerdictTests(unittest.TestCase):
    def test_verdict(self):
        import fresh2_secondary as fs
        s = lambda p, sens: {'p_both_targets': p, 'sensitivity_mean': sens}
        self.assertTrue(fs.verdict(s(0.8, 0.93), s(0.7, 0.91))['confirmed'])
        self.assertFalse(fs.verdict(s(0.6, 0.93), s(0.7, 0.91))['confirmed'])
        self.assertFalse(fs.verdict(s(0.8, 0.89), s(0.7, 0.91))['confirmed'])


if __name__ == '__main__':
    unittest.main()
