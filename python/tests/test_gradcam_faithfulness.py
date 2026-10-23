"""Tests for the Grad-CAM faithfulness metrics (synthetic cams, masks and a toy model)."""
import unittest

import numpy as np
import torch

import evaluate_gradcam_faithfulness as gf


def disc(size, cy, cx, r):
    yy, xx = np.mgrid[:size, :size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.size = 128
        self.fov = disc(self.size, 64, 64, 60)
        self.lesion = disc(self.size, 40, 40, 6) & self.fov

    def test_energy_ratio_is_one_for_uniform_and_high_for_lesion_centred_cam(self):
        uniform = np.ones((self.size, self.size), np.float32)
        self.assertAlmostEqual(gf.energy_ratio(uniform, self.lesion, self.fov), 1.0, places=6)
        focused = self.lesion.astype(np.float32) + 0.01
        self.assertGreater(gf.energy_ratio(focused, self.lesion, self.fov), 10)
        away = disc(self.size, 90, 90, 6).astype(np.float32) + 0.01
        self.assertLess(gf.energy_ratio(away, self.lesion, self.fov), 1.0)

    def test_pointing_game_hits_near_lesion_and_misses_far(self):
        near = np.zeros((self.size, self.size), np.float32); near[44, 44] = 1.0
        far = np.zeros((self.size, self.size), np.float32); far[100, 60] = 1.0
        hit, chance = gf.pointing_hit(near, self.lesion, self.fov, self.size)
        self.assertTrue(hit)
        self.assertTrue(0 < chance < 0.2)
        self.assertFalse(gf.pointing_hit(far, self.lesion, self.fov, self.size)[0])

    def test_pointing_ignores_cam_maxima_outside_the_field(self):
        cam = np.zeros((self.size, self.size), np.float32); cam[0, 0] = 5.0; cam[40, 40] = 1.0
        self.assertTrue(gf.pointing_hit(cam, self.lesion, self.fov, self.size)[0])

    def test_lesion_mapping_keeps_tiny_lesions(self):
        masks = np.zeros((600, 800, 4), np.uint8); masks[300, 400, 2] = 255
        mapped = gf.lesion_in_model_space(masks, np.array([False, False, True, False]), 128)
        self.assertTrue(mapped.any())
        self.assertFalse(gf.lesion_in_model_space(masks, np.array([False, False, False, False]), 128).any())

    def test_deletion_masks_cam_pixels_first_and_curve_auc_orders_methods(self):
        class Toy(torch.nn.Module):  # referral = mean of the central patch of the first channel
            def forward(self, x):
                return x.new_zeros(len(x), 5), 10 * x[:, 0, 40:50, 40:50].mean(dim=(1, 2)) - 1.0

        model = Toy().eval()
        tensor = torch.ones(1, 3, self.size, self.size)
        cam = np.zeros((self.size, self.size), np.float32); cam[40:50, 40:50] = 1.0   # cam on the informative patch
        out = gf.deletion_curves(model, tensor, cam, self.fov, np.random.default_rng(0))
        self.assertLess(gf.curve_auc(out['base'], out['cam']), gf.curve_auc(out['base'], out['random']))
        self.assertGreater(out['base'], 0.9)

    def test_summarise_reports_chance_adjusted_pointing(self):
        import pandas as pd
        frame = pd.DataFrame({'image': list('abcd'), 'has_lesion': [True, True, True, False],
                              'energy_ratio': [2.0, 1.0, 3.0, np.nan], 'pointing_hit': [True, False, True, False],
                              'pointing_chance': [0.1, 0.1, 0.1, np.nan], 'referral_p': [0.9, 0.8, 0.2, np.nan],
                              'del_auc_cam': [0.5, 0.6, 0.7, np.nan], 'del_auc_random': [0.8, 0.8, 0.8, np.nan],
                              'lesion_fraction': [0.01] * 4})
        s = gf.summarise(frame)
        self.assertEqual(s['images_with_lesions'], 3)
        self.assertAlmostEqual(s['pointing_game']['hit_minus_chance'], 2 / 3 - 0.1)
        self.assertEqual(s['deletion']['images_called_referable'], 2)
        self.assertLess(s['deletion']['cam_minus_random'], 0)


if __name__ == '__main__':
    unittest.main()
