"""Tests for lesion-feature summaries and the leave-one-site-out fusion study (synthetic data)."""
import unittest

import numpy as np
import pandas as pd

import fusion_eval as fe
import lesion_features as lf


class SummaryTests(unittest.TestCase):
    def test_summarise_counts_pixels_and_components(self):
        probs = np.zeros((40, 40, 4), np.float32)
        probs[5:10, 5:10, 0] = 0.95
        probs[20:23, 20:23, 0] = 0.95
        valid = np.ones((40, 40), bool)
        s = lf.summarise(probs, valid)
        self.assertEqual(s['hist'][0].sum(), 1600)
        self.assertEqual(s['hist'][0][95], 25 + 9)
        self.assertEqual(s['components'][0][2], 2)
        self.assertAlmostEqual(float(s['max'][0]), 0.95, places=5)
        self.assertEqual(s['fov_pixels'], 1600)

    def test_summarise_ignores_pixels_outside_field(self):
        probs = np.full((20, 20, 4), 0.99, np.float32)
        valid = np.zeros((20, 20), bool)
        valid[:10] = True
        s = lf.summarise(probs, valid)
        self.assertEqual(s['hist'][1].sum(), 200)

    def test_file_name_mapping(self):
        self.assertEqual(lf.file_name('messidor_IM000012_JPG', 'messidor'), 'IM000012.JPG')
        self.assertEqual(lf.file_name('messidor_20051020_43808_0100_PP_png', 'messidor'), '20051020_43808_0100_PP.png')
        self.assertEqual(lf.file_name('odir_0_right', 'odir'), '0_right.jpg')
        self.assertEqual(lf.file_name('fdr26_228QQUAD01', 'fdr26'), '228QQUAD01.jpg')
        self.assertEqual(lf.file_name('deepdrid_100_l1', 'deepdrid'), '100_l1.jpg')


class FusionTests(unittest.TestCase):
    def make(self, signal=True, seed=0):
        rng = np.random.default_rng(seed)
        rows, grading, lesion = [], [], []
        for site in ('a', 'b', 'c', 'd'):
            n = 300
            grade = rng.integers(0, 5, n)
            positive = grade >= 2
            g = grade + rng.normal(0, 1.6, n)            # noisy grading score
            les = rng.normal(0, 1, (n, 12))
            if signal:
                les[:, 0] += positive * 1.5               # lesion feature carries independent information
            rows.append(pd.DataFrame({'image_id': [f'{site}{i}' for i in range(n)], 'source': site, 'grade': grade,
                                      'group_id': [f'{site}{i}' for i in range(n)]}))
            grading.append(g)
            lesion.append(les)
        return pd.concat(rows, ignore_index=True), np.concatenate(grading), np.vstack(lesion)

    def test_informative_lesion_features_improve_every_site(self):
        frame, g, l = self.make(True)
        r = fe.leave_one_site_out(frame, g, l, replicates=100)
        self.assertTrue(r['rule']['success'])
        for site in r['sites'].values():
            self.assertGreater(site['fused_minus_grading']['auc']['estimate'], 0.01)

    def test_noise_features_do_not_pass_the_rule(self):
        frame, g, l = self.make(False)
        r = fe.leave_one_site_out(frame, g, l, replicates=100)
        self.assertFalse(r['rule']['success'])

    def test_target_site_never_enters_its_own_fit(self):
        frame, g, l = self.make(True)
        # make site 'a' lesion features constant noise-free signal; fitting must not use 'a' to score 'a'
        calls = []
        original = fe.fit_score
        def spy(x_train, y_train, x_test, c=1.0):
            calls.append((len(x_train), len(x_test)))
            return original(x_train, y_train, x_test, c)
        fe.fit_score = spy
        try:
            fe.leave_one_site_out(frame, g, l, replicates=10)
        finally:
            fe.fit_score = original
        self.assertTrue(all(a == 900 and b == 300 for a, b in calls))

    def test_within_site_standardisation_removes_site_scale_and_offset(self):
        frame, g, l = self.make(True)
        g2 = g.copy()
        mask = (frame.source == 'a').to_numpy()
        g2[mask] = g2[mask] * 5 + 10  # one site's model has a different score scale
        z = fe.within_site_standardise(frame, g2)
        for site in 'abcd':
            part = z[(frame.source == site).to_numpy()]
            self.assertAlmostEqual(float(part.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(part.std()), 1.0, places=6)
        raw = fe.leave_one_site_out(frame, g2, l, replicates=30)
        zs = fe.leave_one_site_out(frame, g2, l, replicates=30, within_site=True)
        # the fused model on the mis-scaled site is repaired by within-site standardisation
        self.assertGreater(zs['sites']['a']['fused']['auc'], raw['sites']['a']['fused']['auc'] - 1e-9)

    def test_lesion_vector_uses_threshold_bin(self):
        hist = np.zeros((4, lf.BINS), np.int64)
        hist[:, 90] = 10
        item = {'hist': hist, 'max': np.full(4, 0.9, np.float32), 'components': np.ones((4, 4), np.int32), 'fov_pixels': 100}
        high = fe.lesion_vector(item, {k: 0.95 for k in fe.LESIONS}, 0.95)
        low = fe.lesion_vector(item, {k: 0.5 for k in fe.LESIONS}, 0.5)
        self.assertLess(high[0], low[0])
        self.assertEqual(len(high), 12)


if __name__ == '__main__':
    unittest.main()
