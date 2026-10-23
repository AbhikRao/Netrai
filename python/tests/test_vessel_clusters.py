"""Tests for vessel-correlation clustering and near-copy listing."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pandas as pd

import vessel_clusters as vc
from identity_review_vessel import vectors
from tests.test_identity_review_vessel import synthetic_fundus, write


class ClusterLabelTests(unittest.TestCase):
    def test_connected_components_with_single_linkage(self):
        sim = np.eye(6)
        for a, b in ((0, 1), (1, 2), (4, 5)):  # 0-1-2 chain, 4-5 pair, 3 alone
            sim[a, b] = sim[b, a] = 0.9
        labels = vc.cluster_labels(sim, 0.6)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[1], labels[2])
        self.assertEqual(labels[4], labels[5])
        self.assertEqual(len(set(labels)), 3)
        self.assertNotEqual(labels[3], labels[0])

    def test_threshold_is_inclusive_and_diagonal_ignored(self):
        sim = np.eye(2)
        sim[0, 1] = sim[1, 0] = 0.6
        self.assertEqual(len(set(vc.cluster_labels(sim, 0.6))), 1)
        self.assertEqual(len(set(vc.cluster_labels(sim, 0.61))), 2)


class ImageTests(unittest.TestCase):
    def test_same_retina_in_different_exposures_clusters_and_others_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'a').mkdir()
            image = synthetic_fundus(1)
            write(base / 'a' / 'r1.png', image)
            write(base / 'a' / 'r1_dark.png', np.clip(image.astype(np.float32) * 0.6, 0, 255).astype(np.uint8))
            write(base / 'a' / 'r1_mirror.png', image[:, ::-1].copy())
            write(base / 'a' / 'other1.png', synthetic_fundus(2))
            write(base / 'a' / 'other2.png', synthetic_fundus(3))
            argv = ['vessel_clusters.py', '--images', str(base / 'a'), '--output', str(base / 'out'), '--workers', '1']
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                vc.main()
            frame = pd.read_csv(base / 'out' / 'clusters.csv')
            by_name = {Path(r.image).name: r.cluster for r in frame.itertuples()}
            self.assertEqual(by_name['r1.png'], by_name['r1_dark.png'])
            self.assertEqual(by_name['r1.png'], by_name['r1_mirror.png'])
            self.assertNotEqual(by_name['other1.png'], by_name['r1.png'])
            self.assertNotEqual(by_name['other1.png'], by_name['other2.png'])
            with self.assertRaises(FileExistsError), mock.patch.object(sys, 'argv', argv):
                vc.main()

    def test_near_copies_lists_test_images_with_a_reference_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ('test', 'ref'):
                (base / name).mkdir()
            copy = synthetic_fundus(10)
            write(base / 'test' / 'has_copy.png', copy)
            write(base / 'test' / 'fresh.png', synthetic_fundus(11))
            write(base / 'ref' / 'ref_copy.png', cv2.resize(copy, (256, 256)))
            write(base / 'ref' / 'ref_other.png', synthetic_fundus(12))
            argv = ['vessel_clusters.py', '--images', str(base / 'test'), '--against', str(base / 'ref'),
                    '--output', str(base / 'out'), '--workers', '1']
            with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                vc.main()
            table = pd.read_csv(base / 'out' / 'near_copies_against_reference.csv')
            best = {Path(r.image).name: (r.max_correlation, Path(r.nearest_reference).name) for r in table.itertuples()}
            self.assertGreater(best['has_copy.png'][0], 0.8)
            self.assertEqual(best['has_copy.png'][1], 'ref_copy.png')
            self.assertLess(best['fresh.png'][0], 0.5)


if __name__ == '__main__':
    unittest.main()
