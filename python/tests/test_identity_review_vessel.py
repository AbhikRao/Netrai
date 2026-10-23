"""Tests for the label-free vessel-pattern identity screen, on synthetic fundus-like images."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd

import identity_review_vessel as ivr


def synthetic_fundus(seed, size=320):
    """Round field of view with a random branching 'vessel' pattern on a noisy background."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size, 3), np.uint8)
    yy, xx = np.mgrid[:size, :size]
    inside = (yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.47) ** 2
    background = rng.normal(0, 3, (size, size))
    canvas = np.zeros((size, size), np.uint8)
    for _ in range(14):  # random polylines
        points = rng.integers(20, size - 20, (4, 2)).astype(np.int32)
        cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, 255, int(rng.integers(2, 5)))
    green = 120 + background - 0.25 * cv2.GaussianBlur(canvas, (0, 0), 1.2).astype(np.float32)
    for channel, factor in enumerate((1.4, 1.0, 0.6)):
        image[:, :, channel] = np.clip(green * factor, 0, 255).astype(np.uint8)
    image[~inside] = 0
    return image


def write(path, image):
    cv2.imwrite(str(path), image)
    return path


class VesselVectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def corr(self, a, b, size=128):
        return float(ivr.vessel_vector(a, size).ravel() @ ivr.vessel_vector(b, size).ravel())

    def test_image_matches_itself(self):
        path = write(self.dir / 'a.png', synthetic_fundus(1))
        self.assertGreater(self.corr(path, path), 0.99)

    def test_reencoded_and_resized_copy_still_matches(self):
        image = synthetic_fundus(2)
        original = write(self.dir / 'a.png', image)
        jpeg = self.dir / 'copy.jpg'
        cv2.imwrite(str(jpeg), cv2.resize(image, (480, 480), interpolation=cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, 70])
        self.assertGreater(self.corr(original, jpeg), 0.8)

    def test_different_retina_does_not_match(self):
        a = write(self.dir / 'a.png', synthetic_fundus(3))
        b = write(self.dir / 'b.png', synthetic_fundus(4))
        self.assertLess(self.corr(a, b), 0.4)

    def test_mirror_is_found_by_the_mirrored_query(self):
        image = synthetic_fundus(5)
        a = write(self.dir / 'a.png', image)
        b = write(self.dir / 'b.png', image[:, ::-1].copy())
        plain = self.corr(a, b)
        vector = ivr.vessel_vector(a, 128).reshape(128, 128)[:, ::-1].ravel()
        mirrored = float(vector @ ivr.vessel_vector(b, 128).ravel())
        self.assertGreater(mirrored, 0.95)
        self.assertGreater(mirrored, plain)

    def test_unreadable_image_raises(self):
        broken = self.dir / 'broken.png'
        broken.write_bytes(b'not an image')
        with self.assertRaises(ValueError):
            ivr.vessel_vector(broken, 64)

    def test_vectors_are_unit_norm(self):
        path = write(self.dir / 'a.png', synthetic_fundus(6))
        self.assertAlmostEqual(float(np.linalg.norm(ivr.vessel_vector(path, 64))), 1.0, places=3)


class ScreenCliTests(unittest.TestCase):
    def test_screen_flags_a_planted_copy_and_not_unrelated_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'test').mkdir()
            (base / 'reference').mkdir()
            planted = synthetic_fundus(10)
            write(base / 'test' / 't_copy.png', planted)
            write(base / 'test' / 't_new.png', synthetic_fundus(11))
            write(base / 'reference' / 'r_same.png', planted)
            for k in range(12, 20):
                write(base / 'reference' / f'r_{k}.png', synthetic_fundus(k))
            argv = ['identity_review_vessel.py', '--test', str(base / 'test'), '--reference', str(base / 'reference'),
                    '--output', str(base / 'out'), '--workers', '1', '--candidates', '5']
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                ivr.main()
            pairs = pd.read_csv(base / 'out' / 'vessel_pairs.csv')
            best = pairs.groupby('test').corr128.max()
            by_name = {Path(k).name: v for k, v in best.items()}
            self.assertGreater(by_name['t_copy.png'], 0.9)
            self.assertLess(by_name['t_new.png'], 0.5)
            top = pairs[pairs.test.str.endswith('t_copy.png')].iloc[0]
            self.assertTrue(top.reference.endswith('r_same.png'))
            summary = json.loads((base / 'out' / 'vessel_summary.json').read_text())
            self.assertEqual(summary['test_images'], 2)
            self.assertEqual(summary['reference_images'], 9)


if __name__ == '__main__':
    unittest.main()
