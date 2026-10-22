"""Tests for the opt-in learned lesion module (mask summary, overlay, thresholds, panel)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from modules import lesions_learned as ll


class LearnedLesionTests(unittest.TestCase):
    def test_summarise_counts_components_and_area_inside_the_field(self):
        probs = np.zeros((60, 60, 4), np.float32)
        probs[5:10, 5:10, 0] = 0.9          # EX blob inside the field
        probs[20:23, 20:23, 0] = 0.9        # second EX blob
        probs[30:34, 30:34, 1] = 0.95       # HE
        probs[0, 0, 2] = 0.99               # MA pixel outside the field must be ignored
        valid = np.zeros((60, 60), bool); valid[2:58, 2:58] = True
        s = ll.summarise_masks(probs, {'EX': 0.5, 'HE': 0.5, 'MA': 0.5, 'SE': 0.5}, valid)
        self.assertEqual(s['count'], {'EX': 2, 'HE': 1, 'MA': 0, 'SE': 0})
        self.assertAlmostEqual(s['area_fraction']['EX'], (25 + 9) / valid.sum())

    def test_thresholds_use_the_ma_lesion_operating_point(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.json'
            path.write_text(json.dumps({'dice_thresholds': {'EX': .68, 'HE': .9, 'MA': .899, 'SE': .898}, 'ma_lesion_threshold': .95}))
            t = ll.load_thresholds(path)
        self.assertEqual(t['MA'], .95)
        self.assertEqual(t['EX'], .68)

    def test_overlay_colours_only_masked_pixels_and_keeps_shape(self):
        image = np.full((20, 20, 3), 100, np.uint8)
        masks = {n: np.zeros((20, 20), bool) for n in ll.LESIONS}
        masks['HE'][5:8, 5:8] = True
        out = ll.overlay(image, masks)
        self.assertEqual(out.shape, image.shape)
        self.assertTrue((out[0, 0] == 100).all())
        self.assertGreater(int(out[6, 6, 0]), 100)   # red channel pulled up by the HE colour

    def test_panel_renders_a_png(self):
        image = np.full((40, 40, 3), 90, np.uint8)
        masks = {n: np.zeros((40, 40), bool) for n in ll.LESIONS}
        summary = {'overlay': ll.overlay(image, masks), 'count': {n: 0 for n in ll.LESIONS}, 'area_fraction': {n: 0.0 for n in ll.LESIONS}}
        with tempfile.TemporaryDirectory() as d:
            ll.save_panel(summary, Path(d) / 'p.png')
            self.assertGreater((Path(d) / 'p.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
