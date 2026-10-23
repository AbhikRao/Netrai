"""Tests for the lesion segmentation networks and loss (CPU, tiny inputs, no pretrained downloads)."""
import unittest

import torch

from models import lesion_unet as lu


class NetworkTests(unittest.TestCase):
    def test_both_architectures_return_one_logit_map_per_lesion_at_input_resolution(self):
        x = torch.randn(2, 3, 64, 96)
        for arch in ('unet', 'fpn'):
            with self.subTest(arch=arch):
                model = lu.LesionNet(arch, 'efficientnet_b0', pretrained=False).eval()
                with torch.no_grad():
                    out = model(x)
                self.assertEqual(tuple(out.shape), (2, 4, 64, 96))
                self.assertTrue(torch.isfinite(out).all())

    def test_input_size_must_be_divisible_by_32(self):
        model = lu.LesionNet('unet', 'efficientnet_b0', pretrained=False).eval()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 3, 50, 64))

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            lu.LesionNet('segformer', 'efficientnet_b0', pretrained=False)

    def test_gradients_reach_encoder_and_head(self):
        model = lu.LesionNet('unet', 'efficientnet_b0', pretrained=False)
        out = model(torch.randn(2, 3, 64, 64))
        out.mean().backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(next(model.encoder.parameters()).grad.abs().sum()), 0)

    def test_only_onnx_friendly_operations_are_used(self):
        model = lu.LesionNet('unet', 'efficientnet_b0', pretrained=False)
        kinds = {type(m).__name__ for m in model.decoder.modules()}
        self.assertFalse(any('Transpose' in k for k in kinds))


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.targets = (torch.rand(2, 4, 16, 16) < 0.05).float()

    def test_perfect_logits_give_a_near_zero_loss_and_wrong_logits_a_large_one(self):
        perfect = (self.targets * 2 - 1) * 12
        wrong = -perfect
        self.assertLess(float(lu.lesion_loss(perfect, self.targets)), 0.05)
        self.assertGreater(float(lu.lesion_loss(wrong, self.targets)), 1.0)

    def test_unannotated_channels_do_not_contribute(self):
        logits = torch.randn(2, 4, 16, 16)
        annotated = torch.ones(2, 4, dtype=torch.bool)
        annotated[:, 3] = False
        changed = logits.clone()
        changed[:, 3] += 50.0  # wildly wrong prediction in the unannotated channel
        self.assertAlmostEqual(float(lu.lesion_loss(logits, self.targets, annotated)),
                               float(lu.lesion_loss(changed, self.targets, annotated)), places=5)
        self.assertNotAlmostEqual(float(lu.lesion_loss(logits, self.targets)),
                                  float(lu.lesion_loss(changed, self.targets)), places=3)

    def test_positive_weight_increases_the_penalty_for_missed_lesions(self):
        logits = torch.full_like(self.targets, -4.0)
        base = float(lu.lesion_loss(logits, self.targets, dice_weight=0.0))
        weighted = float(lu.lesion_loss(logits, self.targets, pos_weight=torch.full((4,), 20.0), dice_weight=0.0))
        self.assertGreater(weighted, base)

    def test_empty_targets_and_all_unannotated_are_finite(self):
        logits = torch.randn(1, 4, 8, 8, requires_grad=True)
        loss = lu.lesion_loss(logits, torch.zeros(1, 4, 8, 8))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        none = lu.lesion_loss(torch.randn(1, 4, 8, 8), torch.zeros(1, 4, 8, 8), torch.zeros(1, 4, dtype=torch.bool))
        self.assertTrue(torch.isfinite(none))


if __name__ == '__main__':
    unittest.main()
