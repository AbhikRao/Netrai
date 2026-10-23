"""Tests for the heuristic lesion baseline runner (synthetic images; slow detectors run once)."""
import unittest
from pathlib import Path

import numpy as np

import lesion_heuristic_baseline as hb
from tests.test_lesion_data import DataTestCase, fundus
import lesion_data as ld
import pandas as pd


class CountTests(unittest.TestCase):
    def test_pixel_counts_respect_the_valid_region(self):
        pred = np.zeros((10, 10), bool); truth = np.zeros((10, 10), bool); valid = np.ones((10, 10), bool)
        pred[:5], truth[3:8] = True, True
        self.assertEqual(hb.pixel_counts(pred, truth, valid), (20, 30, 30))
        valid[:, 5:] = False
        self.assertEqual(hb.pixel_counts(pred, truth, valid), (10, 15, 15))

    def test_ratio_is_nan_for_empty_denominators(self):
        self.assertTrue(np.isnan(hb.ratio(1, 0)))
        self.assertEqual(hb.ratio(1, 4), 0.25)

    def test_only_train_and_valid_can_be_scored(self):
        with self.assertRaises(ValueError):
            hb.evaluate_split(Path('.'), 'test')


class RunTests(DataTestCase):
    def test_heuristics_run_on_an_exported_split_and_report_every_scored_lesion(self):
        clusters = [(str(r.image_path), i) for i, r in enumerate(ld.ddr_records('train') + ld.ddr_records('valid') + ld.idrid_train_records())]
        csv = self.base / 'clusters.csv'
        pd.DataFrame(clusters, columns=['image', 'cluster']).to_csv(csv, index=False)
        out = self.base / 'export'
        ld.export_dataset(out, long_side=256, valid_fraction=0.3, seed=2, clusters_csv=csv, workers=1)
        masks = hb.heuristic_masks(fundus(256, 256, 5))
        self.assertEqual(set(masks), {'ma', 'hem', 'exudate'})
        for mask in masks.values():
            self.assertEqual(mask.shape, (256, 256))
            self.assertEqual(mask.dtype, bool)
        result = hb.evaluate_split(out, 'valid', replicates=10)
        self.assertEqual(set(result), {'MA', 'HE', 'EX', 'EX+SE'})
        for entry in result.values():
            self.assertGreater(entry['images'], 0)
            self.assertIn('dice', entry['pixel'])
            self.assertEqual(entry['lesion_level']['gt_lesions'] >= entry['lesion_level']['detected'], True)


if __name__ == '__main__':
    unittest.main()
