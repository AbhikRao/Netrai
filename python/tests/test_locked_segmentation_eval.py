"""Tests for the locked lesion-segmentation evaluation: guards, bindings and a tiny end-to-end run on synthetic data."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pandas as pd
import torch

import exposure
import lesion_data as ld
import locked_segmentation_eval as lse
from models.lesion_unet import LesionNet

SIZE = 128


def fundus(seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    inside = (yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2 < (SIZE * 0.46) ** 2
    image = np.zeros((SIZE, SIZE, 3), np.uint8)
    image[inside] = rng.integers(90, 170, 3)
    return image


class LockedSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.ddr = base / 'ddr'
        (self.ddr / 'test' / 'image').mkdir(parents=True)
        for lesion in ld.LESIONS:
            (self.ddr / 'test' / 'label' / lesion).mkdir(parents=True)
        self.ids = [f'img{i}' for i in range(4)]
        for i, stem in enumerate(self.ids):
            cv2.imwrite(str(self.ddr / 'test' / 'image' / f'{stem}.jpg'), fundus(i))
            for lesion in ld.LESIONS:
                mask = np.zeros((SIZE, SIZE), np.uint8)
                mask[50 + i:60, 40:70] = 255
                cv2.imwrite(str(self.ddr / 'test' / 'label' / lesion / f'{stem}.tif'), mask)
        self.registry = base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            lse.COHORT: {'kind': 'segmentation', 'status': 'reserved', 'label_paths': [str(self.ddr / 'test' / 'label')]}}}))
        torch.manual_seed(0)
        model = LesionNet('unet', 'efficientnet_b0', pretrained=False)
        self.weights = base / 'best.pt'
        torch.save({'model_state_dict': model.state_dict(), 'config': {'arch': 'unet', 'encoder': 'efficientnet_b0'}}, self.weights)
        self.thresholds = base / 'thresholds.json'
        self.thresholds.write_text(json.dumps({'dice_thresholds': {k: 0.5 for k in ld.LESIONS}, 'ma_lesion_threshold': 0.5}))
        self.exclusions = base / 'exclusions.csv'
        pd.DataFrame({'image_id': self.ids, 'decision': ['keep', 'keep', 'keep', 'exclude']}).to_csv(self.exclusions, index=False)
        for patch in (mock.patch.object(exposure, 'REGISTRY_PATH', self.registry), mock.patch.object(ld, 'DDR_ROOT', self.ddr),
                      mock.patch('lesion_heuristic_baseline.heuristic_masks',
                                 side_effect=lambda rgb: {k: np.zeros(rgb.shape[:2], bool) for k in ('ma', 'hem', 'exudate')})):
            patch.start()
            self.addCleanup(patch.stop)
        self.lock_dir = base / 'lock'

    def freeze_args(self, **kw):
        base = dict(weights=self.weights, thresholds=self.thresholds, exclusions=self.exclusions, output=self.lock_dir,
                    long_side=SIZE, tile=SIZE, overlap=32, enhance='none', replicates=20, seed=1)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_freeze_does_not_read_masks_and_registers_lock(self):
        with mock.patch.object(ld, 'read_masks', side_effect=AssertionError('labels read at freeze')):
            lse.freeze(self.freeze_args())
        lock = json.loads((self.lock_dir / 'lock.json').read_text())
        self.assertIs(lock['labels_read'], False)
        self.assertEqual(lock['tests'][lse.COHORT]['images_primary'], 3)
        self.assertEqual(exposure.status(lse.COHORT), 'locked')

    def test_freeze_refuses_overwrite_and_inventory_mismatch_and_exposed(self):
        lse.freeze(self.freeze_args())
        with self.assertRaises(FileExistsError):
            lse.freeze(self.freeze_args())
        pd.DataFrame({'image_id': self.ids[:3], 'decision': ['keep'] * 3}).to_csv(self.exclusions, index=False)
        with self.assertRaises(ValueError):
            lse.freeze(self.freeze_args(output=self.lock_dir.parent / 'lock2'))
        exposure._update(lse.COHORT, status='exposed')
        pd.DataFrame({'image_id': self.ids, 'decision': ['keep'] * 4}).to_csv(self.exclusions, index=False)
        with self.assertRaises(exposure.ExposureError):
            lse.freeze(self.freeze_args(output=self.lock_dir.parent / 'lock3'))

    def test_labels_are_unreachable_without_a_lock(self):
        with self.assertRaises(exposure.ExposureError):
            ld.ddr_records('test', self.lock_dir)
        record = ld.Record('ddr_img0', 'ddr', self.ddr / 'test/image/img0.jpg',
                           {k: self.ddr / 'test/label' / k / 'img0.tif' for k in ld.LESIONS})
        with self.assertRaises(exposure.ExposureError):
            ld.read_masks(record, (SIZE, SIZE))

    def test_tampered_binding_is_detected(self):
        lse.freeze(self.freeze_args())
        self.thresholds.write_text(json.dumps({'dice_thresholds': {k: 0.4 for k in ld.LESIONS}, 'ma_lesion_threshold': 0.5}))
        with self.assertRaises(ValueError):
            lse.evaluate(SimpleNamespace(lock=self.lock_dir))

    def test_single_evaluation_then_exposed_and_second_run_refused(self):
        lse.freeze(self.freeze_args())
        lse.evaluate(SimpleNamespace(lock=self.lock_dir))
        report = json.loads((self.lock_dir / 'report.json').read_text())
        self.assertEqual(report['tier'], 'TF')
        self.assertEqual(report['primary']['images'], 3)
        self.assertEqual(report['secondary_all_images_contains_near_copies']['images'], 4)
        self.assertIn('MA', report['primary']['lesions'])
        self.assertEqual(exposure.status(lse.COHORT), 'exposed')
        with self.assertRaises(exposure.ExposureError):
            lse.evaluate(SimpleNamespace(lock=self.lock_dir))

    def test_heuristic_failure_does_not_abort_and_is_excluded_from_comparisons(self):
        calls = {'n': 0}

        def flaky(rgb):
            calls['n'] += 1
            if calls['n'] == 2:
                raise RuntimeError('heuristic exploded')
            return {k: np.zeros(rgb.shape[:2], bool) for k in ('ma', 'hem', 'exudate')}

        with mock.patch('lesion_heuristic_baseline.heuristic_masks', side_effect=flaky):
            lse.freeze(self.freeze_args())
            lse.evaluate(SimpleNamespace(lock=self.lock_dir))
        report = json.loads((self.lock_dir / 'report.json').read_text())
        self.assertEqual(report['secondary_all_images_contains_near_copies']['heuristic_failures'], 1)
        ex = report['secondary_all_images_contains_near_copies']['lesions']['EX']['vs_heuristic']
        self.assertEqual(ex['images_annotated'], 3)  # the failed image is left out of the paired comparison

    def test_score_stats_matches_direct_computation(self):
        rng = np.random.default_rng(0)
        thresholds = {k: 0.5 for k in ld.LESIONS}
        stats, direct = [], np.zeros(3)
        for _ in range(4):
            probs = rng.random((40, 40, 4)).astype(np.float32)
            masks = (rng.random((40, 40, 4)) > 0.7).astype(np.uint8) * 255
            valid = np.ones((40, 40), bool)
            heur = {k: rng.random((40, 40)) > 0.5 for k in ('ma', 'hem', 'exudate')}
            stats.append(lse.image_stats(probs, masks, np.ones(4, bool), valid, heur, thresholds, 0.5))
            direct += lse.counts(probs[..., 0] >= 0.5, masks[..., 0] > 0, valid)
        out = lse.score_stats(stats, thresholds, 20, 0)
        self.assertAlmostEqual(out['lesions']['EX']['vs_heuristic']['learned_dice'], lse.dice_from_counts(*direct))

    def test_paired_bootstrap_detects_a_clear_gain(self):
        good = np.tile([90.0, 5.0, 5.0], (30, 1))
        bad = np.tile([30.0, 40.0, 70.0], (30, 1))
        result = lse.paired_dice_bootstrap(good, bad, 200, 0)
        self.assertGreater(result['ci95'][0], 0)
        same = lse.paired_dice_bootstrap(good, good, 200, 0)
        self.assertEqual(same['difference'], 0.0)

    def test_native_probabilities_match_native_size(self):
        model = LesionNet('unet', 'efficientnet_b0', pretrained=False).eval()
        rgb = fundus(7)
        probs = lse.native_probabilities(model, rgb, torch.device('cpu'), 64, 64, 16, 'none')
        self.assertEqual(probs.shape, (SIZE, SIZE, 4))
        self.assertTrue(((probs >= 0) & (probs <= 1)).all())


if __name__ == '__main__':
    unittest.main()
