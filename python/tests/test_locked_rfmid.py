"""Tests for the locked RFMiD 2.0 evaluation: guards, endpoints and a synthetic end-to-end run."""
import json
import tempfile
import zlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pandas as pd

import exposure
import locked_rfmid as lr

ROOT = lr.ROOT


class LockedRfmidTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'results')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.data = base / 'data'
        rng = np.random.default_rng(0)
        self.rows = []
        n = 0
        for split, label in (('Training_set', 'RFMiD_2_Training_labels.csv'), ('Validation_set', 'RFMiD_2_Validation_labels.csv'), ('Test_set', 'RFMiD_2_Testing_labels.csv')):
            (self.data / split).mkdir(parents=True)
            records = []
            for i in range(12):
                ident = n + 1
                name = f'{ident}.jpg'
                cv2.imwrite(str(self.data / split / name), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
                dr = int(rng.random() < 0.5)
                wnl = int(dr == 0 and rng.random() < 0.5)
                records.append({'ID': ident, 'DR': dr, 'WNL': wnl, 'OTHER': 1})
                self.rows.append((split, name, ident, dr, wnl))
                n += 1
            pd.DataFrame(records).to_csv(self.data / split / label, index=False, encoding='latin-1')
        self.exclusions = base / 'exclusions.txt'
        self.exclusions.write_text('3.jpg\n')
        clusters = pd.DataFrame({'image': [f'/x/{r[1]}' for r in self.rows], 'cluster': [r[2] // 2 for r in self.rows], 'cluster_size': 2})
        self.clusters = base / 'clusters.csv'
        clusters.to_csv(self.clusters, index=False)
        self.protocol = base / 'protocol.md'
        self.protocol.write_text('prereg')
        self.variants = base / 'variants.json'
        self.variants.write_text(json.dumps({'primary': 'M', 'reference': 'A_legacy', 'bootstrap_seed': 5, 'bootstrap_replicates': 30,
                                             'variants': [{'id': 'A_legacy', 'kind': 'legacy'}, {'id': 'M', 'kind': 'legacy'}]}))
        self.registry = base / 'registry.json'
        label_paths = [str((self.data / s / f).resolve()) for s, f in (('Training_set', 'RFMiD_2_Training_labels.csv'), ('Validation_set', 'RFMiD_2_Validation_labels.csv'), ('Test_set', 'RFMiD_2_Testing_labels.csv'))]
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'rfmid2': {'kind': 'grading(any-DR)', 'status': 'reserved', 'label_paths': label_paths}}}))
        self.dr = {f'{r[0]}/{r[1]}': r[3] for r in self.rows}

        def fake_logits(spec, frame, folder=None, **kw):
            out = np.stack([np.random.default_rng(zlib.crc32(f'{spec["id"]}|{i}'.encode())).normal(0, 1, 6) for i in frame.image_id])
            truth = np.array([self.dr[i] for i in frame.image_id])
            out[:, 0] -= 2.0 * truth * (1 if spec['id'] == 'M' else 0.2)  # model M separates any-DR better
            return out

        for patch in (mock.patch.object(exposure, 'REGISTRY_PATH', self.registry), mock.patch.object(lr, 'logits_for', side_effect=fake_logits)):
            patch.start()
            self.addCleanup(patch.stop)
        self.lock = base / 'lock'

    def args(self, **kw):
        base = dict(variants=self.variants, output=self.lock, data=self.data, exclusions=self.exclusions, clusters=self.clusters, protocol=self.protocol, precomputed=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_inventory_excludes_listed_image_and_assigns_clusters(self):
        frame = lr.inventory(self.data, self.exclusions, self.clusters)
        self.assertEqual(len(frame), 35)
        self.assertNotIn('3.jpg', set(frame.file))
        self.assertTrue(frame.image_id.str.contains('/').all())

    def test_freeze_does_not_open_labels_and_locks(self):
        with mock.patch.object(lr, 'read_labels', side_effect=AssertionError('labels read at freeze')):
            lr.freeze(self.args())
        lock = json.loads((self.lock / 'lock.json').read_text())
        self.assertIs(lock['labels_read'], False)
        self.assertEqual(exposure.status('rfmid2'), 'locked')
        with self.assertRaises(FileExistsError):
            lr.freeze(self.args())

    def test_evaluate_requires_lock_then_runs_once_and_marks_exposed(self):
        with self.assertRaises(Exception):
            lr.evaluate(self.args())
        lr.freeze(self.args())
        lr.evaluate(self.args())
        report = json.loads((self.lock / 'report.json').read_text())
        self.assertEqual(report['tier'], 'TF')
        self.assertEqual(report['population']['images'], 35)
        auc_m = report['results']['M']['auc_any_dr']['estimate']
        auc_a = report['results']['A_legacy']['auc_any_dr']['estimate']
        self.assertGreater(auc_m, auc_a)
        self.assertIn('M_vs_A_legacy', report['comparisons'])
        self.assertEqual(exposure.status('rfmid2'), 'exposed')
        with self.assertRaises(exposure.ExposureError):
            lr.evaluate(self.args())

    def test_tampered_binding_is_detected(self):
        lr.freeze(self.args())
        self.protocol.write_text('changed')
        with self.assertRaises(ValueError):
            lr.evaluate(self.args())

    def test_precomputed_logits_are_spot_checked_and_bound(self):
        pre = Path(self.temp.name) / 'pre'
        pre.mkdir()
        frame = lr.inventory(self.data, self.exclusions, self.clusters)
        good = lr.logits_for({'id': 'M', 'kind': 'legacy'}, frame)
        np.save(pre / 'logits_M.npy', good)
        lr.freeze(self.args(precomputed=pre))
        lock = json.loads((self.lock / 'lock.json').read_text())
        self.assertIn('M', lock['precomputed_logits_cpu_spot_check_max_abs'])
        self.assertLess(lock['precomputed_logits_cpu_spot_check_max_abs']['M'], 1e-9)

    def test_bad_precomputed_logits_are_rejected(self):
        pre = Path(self.temp.name) / 'pre2'
        pre.mkdir()
        np.save(pre / 'logits_M.npy', np.zeros((35, 6)) + 5.0)  # wrong values: spot check must fail
        with self.assertRaises(ValueError):
            lr.freeze(self.args(precomputed=pre))
        np.save(pre / 'logits_M.npy', np.zeros((3, 6)))  # wrong shape
        with self.assertRaises(ValueError):
            lr.freeze(self.args(precomputed=pre, output=self.lock.parent / 'lock_b'))

    def test_images_without_a_label_row_are_left_out_and_reported(self):
        # drop the label row of one image: the evaluation must not fail or guess, and must report it
        frame = lr.inventory(self.data, self.exclusions, self.clusters)
        target = frame.image_id.iloc[0]
        split, name = target.split('/')
        path = next(Path(self.registry.parent / 'data' / split).glob('*labels.csv'))
        table = pd.read_csv(path, encoding='latin-1')
        table = table[table.ID != int(name.split('.')[0])]
        table.to_csv(path, index=False, encoding='latin-1')
        lr.freeze(self.args())
        lr.evaluate(self.args())
        report = json.loads((self.lock / 'report.json').read_text())
        self.assertEqual(report['population']['unlabelled_in_release'], [target])
        self.assertEqual(report['population']['images'], 34)
        self.assertEqual(report['population']['images_in_lock'], 35)

    def test_same_id_in_two_splits_joins_by_folder(self):
        frame = lr.inventory(self.data, self.exclusions, self.clusters)
        labels = lr.read_labels(frame, self.data)
        self.assertTrue(labels.has_label.all())

    def test_endpoint_stats_definitions(self):
        dr = np.array([1, 1, 0, 0, 0, 1])
        wnl = np.array([0, 0, 1, 1, 0, 0])
        scores = {'any_dr': np.array([.9, .8, .1, .2, .7, .3]), 'referral_head': np.array([.9, .7, .2, .1, .6, .4]),
                  'expected_grade': np.array([2.0, 1.5, 0.2, 0.3, 1.2, 0.5])}
        s = lr.endpoint_stats(dr, wnl, scores, np.array([1.0, 2, 3, 4]))
        self.assertAlmostEqual(s['sensitivity_grade_ge1'], 2 / 3)       # 3 DR eyes, two at score >= 1.0
        self.assertAlmostEqual(s['specificity_wnl'], 1.0)               # both WNL eyes below 1.0
        self.assertAlmostEqual(s['specificity_other_disease'], 0.0)     # the other-disease negative is called positive
        self.assertAlmostEqual(s['specificity_grade_ge1'], 2 / 3)

    def test_scores_use_temperature(self):
        logits = np.array([[2.0, 0, 0, 0, 0, 0.0]])
        hot = lr.scores_for(logits, 2.0)['any_dr'][0]
        cold = lr.scores_for(logits, 0.5)['any_dr'][0]
        self.assertGreater(hot, cold)


if __name__ == '__main__':
    unittest.main()
