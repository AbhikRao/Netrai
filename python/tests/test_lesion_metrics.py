"""Tests for the histogram-based lesion metrics."""
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import lesion_metrics as lm


class HistogramMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.mask = rng.random((200, 200)) < 0.05
        self.prob = np.clip(self.mask * 0.35 + rng.random((200, 200)) * 0.6, 0, 1)

    def test_aupr_and_auc_match_scikit_learn_up_to_binning(self):
        hist = lm.histograms(self.prob, self.mask)
        self.assertAlmostEqual(lm.aupr(hist), average_precision_score(self.mask.ravel(), self.prob.ravel()), delta=0.01)
        self.assertAlmostEqual(lm.roc_auc(hist), roc_auc_score(self.mask.ravel(), self.prob.ravel()), delta=0.002)

    def test_perfect_and_inverted_predictions(self):
        perfect = lm.histograms(self.mask.astype(float), self.mask)
        self.assertAlmostEqual(lm.aupr(perfect), 1.0, places=6)
        self.assertAlmostEqual(lm.roc_auc(perfect), 1.0, places=6)
        self.assertAlmostEqual(lm.dice_at(perfect, 0.5), 1.0, places=6)
        inverted = lm.histograms(1.0 - self.mask.astype(float), self.mask)
        self.assertAlmostEqual(lm.roc_auc(inverted), 0.0, places=6)
        self.assertLess(lm.aupr(inverted), 0.1)

    def test_dice_and_iou_at_a_threshold_match_the_direct_computation(self):
        hist = lm.histograms(self.prob, self.mask)
        pred = self.prob >= 0.5
        tp, fp, fn = (pred & self.mask).sum(), (pred & ~self.mask).sum(), (~pred & self.mask).sum()
        self.assertAlmostEqual(lm.dice_at(hist, 0.5), 2 * tp / (2 * tp + fp + fn), places=3)
        self.assertAlmostEqual(lm.iou_at(hist, 0.5), tp / (tp + fp + fn), places=3)

    def test_best_threshold_has_the_highest_dice(self):
        hist = lm.histograms(self.prob, self.mask)
        threshold, best = lm.best_dice_threshold(hist)
        for t in np.linspace(0.05, 0.95, 19):
            self.assertLessEqual(lm.dice_at(hist, t), best + 1e-12)
        self.assertAlmostEqual(lm.dice_at(hist, threshold), best)

    def test_empty_masks_give_nan_not_zero(self):
        hist = lm.histograms(self.prob, np.zeros_like(self.mask))
        self.assertTrue(np.isnan(lm.aupr(hist)))
        self.assertTrue(np.isnan(lm.roc_auc(hist)))

    def test_valid_region_excludes_pixels(self):
        valid = np.zeros_like(self.mask)
        valid[:100] = True
        hist = lm.histograms(self.prob, self.mask, valid)
        self.assertEqual(hist.sum(), 100 * 200)

    def test_histograms_add_across_images(self):
        a, b = lm.histograms(self.prob, self.mask), lm.histograms(self.prob[::-1], self.mask[::-1])
        self.assertAlmostEqual(lm.aupr(a + b), lm.aupr(a), delta=0.02)
        self.assertEqual((a + b).sum(), a.sum() + b.sum())


class LesionLevelTests(unittest.TestCase):
    def test_detection_and_false_positives(self):
        mask = np.zeros((60, 60))
        mask[5:8, 5:8] = 1      # lesion 1 (found)
        mask[30:33, 30:33] = 1  # lesion 2 (missed)
        prob = np.zeros((60, 60))
        prob[6:8, 6:8] = 0.9    # overlaps lesion 1
        prob[50:52, 50:52] = 0.9  # false positive far from any lesion
        result = lm.lesion_level(prob, mask, 0.5)
        self.assertEqual(result, {'gt_lesions': 2, 'detected': 1, 'false_positives': 1})
        self.assertEqual(lm.lesion_level(prob, mask, 0.95)['detected'], 0)


class LesionSetTests(unittest.TestCase):
    def build(self, images=12, seed=1):
        rng = np.random.default_rng(seed)
        s = lm.LesionSet()
        for _ in range(images):
            masks = (rng.random((40, 40, 4)) < 0.03).astype(np.uint8) * 255
            probs = np.clip((masks > 0) * 0.4 + rng.random((40, 40, 4)) * 0.6, 0, 1)
            s.add(probs, masks)
        return s

    def test_summary_has_every_lesion_and_uses_a_given_threshold(self):
        s = self.build()
        summary = s.summary()
        self.assertEqual(set(summary), set(lm.LESIONS))
        self.assertIn('dice_best_on_this_set', summary['MA'])
        fixed = s.summary({name: 0.6 for name in lm.LESIONS})
        self.assertEqual(fixed['EX']['threshold'], 0.6)
        self.assertNotIn('dice_best_on_this_set', fixed['EX'])
        self.assertGreater(fixed['EX']['positive_pixels'], 0)

    def test_bootstrap_is_seeded_and_brackets_the_estimate(self):
        s = self.build()
        a = s.bootstrap(40, seed=3, thresholds={n: 0.6 for n in lm.LESIONS})
        b = s.bootstrap(40, seed=3, thresholds={n: 0.6 for n in lm.LESIONS})
        self.assertEqual(a, b)
        estimate = s.summary()['HE']['aupr']
        low, high = a['HE']['aupr_ci95']
        self.assertLessEqual(low, high)
        self.assertLess(abs(estimate - 0.5 * (low + high)), 0.1)

    def test_image_level_presence_is_recorded(self):
        s = self.build(images=3)
        self.assertEqual(len(s.max_prob), 3)
        self.assertEqual(s.has_lesion[0].shape, (4,))


if __name__ == '__main__':
    unittest.main()
