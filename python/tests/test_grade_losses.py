"""Tests for the grade-head objectives."""
import unittest

import torch
from torch.nn import functional as F

from models import grade_losses as gl


def logits_for(probabilities):
    return torch.log(torch.tensor(probabilities, dtype=torch.float32))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(32, 5)
        self.targets = torch.randint(0, 5, (32,))

    def test_ce_variants_match_torch(self):
        self.assertAlmostEqual(float(gl.build_grade_loss('ce')(self.logits, self.targets)),
                               float(F.cross_entropy(self.logits, self.targets, label_smoothing=0.1)), places=6)
        self.assertAlmostEqual(float(gl.build_grade_loss('ce_nosmooth')(self.logits, self.targets)),
                               float(F.cross_entropy(self.logits, self.targets)), places=6)

    def test_soft_labels_are_distributions_peaked_on_the_target(self):
        labels = gl.soft_labels(torch.tensor([0, 2, 4]), sigma=0.7)
        torch.testing.assert_close(labels.sum(dim=1), torch.ones(3))
        self.assertEqual(labels.argmax(dim=1).tolist(), [0, 2, 4])
        self.assertAlmostEqual(float(labels[1, 1]), float(labels[1, 3]), places=6)  # symmetric around grade 2
        self.assertGreater(float(labels[1, 1]), float(labels[1, 0]))

    def test_emd_is_zero_for_a_correct_one_hot_prediction(self):
        logits = torch.full((3, 5), -30.0)
        logits[torch.arange(3), torch.tensor([0, 2, 4])] = 30.0
        self.assertAlmostEqual(float(gl.squared_emd(logits, torch.tensor([0, 2, 4]))), 0.0, places=6)

    def test_emd_grows_with_the_grade_distance(self):
        target = torch.zeros(1, dtype=torch.long)
        values = []
        for wrong in (1, 2, 3, 4):
            logits = torch.full((1, 5), -30.0)
            logits[0, wrong] = 30.0
            values.append(float(gl.squared_emd(logits, target)))
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[0])
        # cross-entropy cannot tell the distance, EMD can
        ce = [float(F.cross_entropy(torch.tensor([[(-30.0 if k != w else 30.0) for k in range(5)]]), target))
              for w in (1, 4)]
        self.assertAlmostEqual(ce[0], ce[1], places=3)

    def test_ordinal_label_smoothing_prefers_near_misses(self):
        near = logits_for([[0.1, 0.8, 0.05, 0.03, 0.02]])
        far = logits_for([[0.1, 0.02, 0.03, 0.05, 0.8]])
        target = torch.tensor([0])
        self.assertLess(float(gl.ordinal_label_smoothing(near, target)), float(gl.ordinal_label_smoothing(far, target)))

    def test_gce_approaches_cross_entropy_as_q_shrinks(self):
        value = float(gl.generalized_cross_entropy(self.logits, self.targets, q=1e-4))
        self.assertAlmostEqual(value, float(F.cross_entropy(self.logits, self.targets)), places=3)

    def test_gce_bounds_the_penalty_of_a_mislabelled_example(self):
        confident_wrong = torch.tensor([[-30.0, 30.0, -30.0, -30.0, -30.0]])
        target = torch.tensor([0])
        self.assertLessEqual(float(gl.generalized_cross_entropy(confident_wrong, target, 0.7)), 1 / 0.7 + 1e-6)
        self.assertGreater(float(F.cross_entropy(confident_wrong, target)), 50.0)

    def test_every_loss_has_a_finite_gradient(self):
        for name in gl.LOSSES:
            logits = self.logits.clone().requires_grad_(True)
            gl.build_grade_loss(name)(logits, self.targets).backward()
            self.assertTrue(torch.isfinite(logits.grad).all(), name)
            self.assertGreater(float(logits.grad.abs().sum()), 0.0, name)

    def test_unknown_names_and_bad_q_are_rejected(self):
        with self.assertRaises(ValueError):
            gl.build_grade_loss('corn')
        with self.assertRaises(ValueError):
            gl.generalized_cross_entropy(self.logits, self.targets, q=0.0)


if __name__ == '__main__':
    unittest.main()
