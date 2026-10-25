"""Tests for the per-site calibration tool: boundary fitting, monotone thresholds, guards, policy file."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import calibrate_site as cs

BASE = {'thresholds': [1.0, 1.2, 2.0, 2.8], 'head_threshold': 0.2, 'temperature': 0.8}


def logits_for_scores(scores):
    """Logits whose expected grade equals ``scores`` approximately: two-class mixture between floor and ceil grade."""
    out = np.full((len(scores), 6), -12.0)
    for i, s in enumerate(scores):
        lo = int(np.floor(min(s, 3.999)))
        frac = s - lo
        out[i, lo] = np.log(max(1 - frac, 1e-6))
        out[i, lo + 1] = np.log(max(frac, 1e-6))
    return out


class CalibrateSiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        rng = np.random.default_rng(0)
        n_pos, n_neg = 80, 220
        score = np.concatenate([rng.normal(2.2, 0.5, n_pos), rng.normal(0.4, 0.4, n_neg)]).clip(0, 3.9)
        grade = np.concatenate([np.full(n_pos, 2), np.zeros(n_neg, int)])
        self.score, self.grade = score, grade
        np.save(self.dir / 'logits.npy', logits_for_scores(score))
        pd.DataFrame({'grade': grade, 'patient': np.arange(len(grade)) // 2}).to_csv(self.dir / 'labels.csv', index=False)
        (self.dir / 'base.json').write_text(json.dumps(BASE))
        (self.dir / 'netrai_bundle.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [
            {'role': 'model', 'path': 'm.onnx', 'sha256': 'a' * 64}, {'role': 'config', 'path': 'c.json', 'sha256': 'b' * 64}]}))

    def args(self, **kw):
        base = dict(logits=self.dir / 'logits.npy', labels=self.dir / 'labels.csv', base_policy=self.dir / 'base.json',
                    output=self.dir / 'site.json', site='test', target_sensitivity=0.92, bundle=None, variant='E_768')
        base.update(kw)
        return SimpleNamespace(**base)

    def test_fit_boundary_reaches_target_sensitivity(self):
        referable = self.grade >= 2
        cut = cs.fit_boundary(cs.expected_grade(logits_for_scores(self.score)), referable, 0.92)
        sens = (cs.expected_grade(logits_for_scores(self.score))[referable] >= cut).mean()
        self.assertGreaterEqual(sens, 0.92)
        self.assertLess(sens, 0.97)

    def test_monotone_thresholds_keep_the_referral_boundary_exact(self):
        self.assertEqual(cs.monotone_thresholds([1.0, 1.2, 2.0, 2.8], 1.5), [1.0, 1.5, 2.0, 2.8])
        self.assertEqual(cs.monotone_thresholds([1.0, 1.2, 2.0, 2.8], 0.7), [0.7, 0.7, 2.0, 2.8])
        self.assertEqual(cs.monotone_thresholds([1.0, 1.2, 2.0, 2.8], 2.4), [1.0, 2.4, 2.4, 2.8])

    def test_fit_writes_policy_with_provenance_and_no_validation_claim(self):
        policy = cs.fit(self.args())
        saved = json.loads((self.dir / 'site.json').read_text())
        self.assertEqual(saved['status'], 'site_calibrated')
        self.assertEqual(saved['site']['referable'], 80)
        self.assertEqual(saved['site']['patients'], 150)
        self.assertIn('NONE', saved['site']['validation'])
        self.assertEqual(saved['head_threshold'], 0.2)
        self.assertTrue(np.all(np.diff(saved['thresholds']) >= 0) or saved['thresholds'][0] <= saved['thresholds'][1])
        self.assertEqual(policy['thresholds'], saved['thresholds'])
        self.assertEqual((saved['model_variant'], saved['model_sha256'], saved['config_sha256']), ('E_768', 'a' * 64, 'b' * 64))

    def test_policy_without_a_bundle_cannot_be_created(self):
        (self.dir / 'netrai_bundle.json').unlink()
        with self.assertRaises(FileNotFoundError):
            cs.fit(self.args())
        (self.dir / 'bundle2.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [{'role': 'model', 'path': 'm', 'sha256': 'c' * 64}]}))
        with self.assertRaises(ValueError):
            cs.fit(self.args(bundle=self.dir / 'bundle2.json'))

    def test_too_few_referable_cases_is_refused_and_small_samples_warn(self):
        pd.DataFrame({'grade': [0] * 100 + [2] * 3}).to_csv(self.dir / 'tiny.csv', index=False)
        np.save(self.dir / 'tiny.npy', logits_for_scores(np.r_[np.full(100, .3), np.full(3, 2.2)]))
        with self.assertRaises(ValueError):
            cs.fit(self.args(logits=self.dir / 'tiny.npy', labels=self.dir / 'tiny.csv'))
        pd.DataFrame({'grade': [0] * 100 + [2] * 10}).to_csv(self.dir / 'small.csv', index=False)
        np.save(self.dir / 'small.npy', logits_for_scores(np.r_[np.full(100, .3), np.full(10, 2.2)]))
        policy = cs.fit(self.args(logits=self.dir / 'small.npy', labels=self.dir / 'small.csv'))
        self.assertTrue(any('referable' in w for w in policy['site']['warnings']))

    def test_matlab_style_base_config_is_accepted(self):
        (self.dir / 'matlab.json').write_text(json.dumps({'grade_thresholds': [1.0, 1.2, 2.0, 2.8], 'referable_head_threshold': 0.2}))
        policy = cs.fit(self.args(base_policy=self.dir / 'matlab.json'))
        self.assertNotIn('grade_thresholds', policy)
        self.assertEqual(len(policy['thresholds']), 4)

    def test_length_mismatch_and_bad_grades_are_refused(self):
        pd.DataFrame({'grade': [0, 1]}).to_csv(self.dir / 'short.csv', index=False)
        with self.assertRaises(ValueError):
            cs.fit(self.args(labels=self.dir / 'short.csv'))
        pd.DataFrame({'grade': [7] * len(self.grade)}).to_csv(self.dir / 'bad.csv', index=False)
        with self.assertRaises(ValueError):
            cs.fit(self.args(labels=self.dir / 'bad.csv'))


if __name__ == '__main__':
    unittest.main()
