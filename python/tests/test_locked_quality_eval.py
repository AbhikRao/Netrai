"""Tests for the locked DeepDRiD Challenge-2 quality evaluation (synthetic images, labels and models)."""
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pandas as pd

import exposure
import locked_quality_eval as lq

ROOT = lq.ROOT


class LockedQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'results')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.images = base / 'Images'
        rng = np.random.default_rng(0)
        self.truth = {}
        rows = []
        for patient in (401, 402, 420, 454):
            (self.images / str(patient)).mkdir(parents=True)
            for eye in ('l1', 'r1'):
                image_id = f'{patient}_{eye}'
                cv2.imwrite(str(self.images / str(patient) / f'{image_id}.jpg'), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
                bad = int(rng.random() < 0.5)
                self.truth[image_id] = bad
                rows.append({'image_id': image_id, 'Overall quality': 0 if bad else 1, 'Artifact': 6 if bad else 0,
                             'Clarity': 4 if bad else 8, 'Field definition': 4 if bad else 8})
        self.workbook = base / 'labels.xlsx'
        self.rows = rows
        pd.DataFrame(rows).to_excel(self.workbook, index=False)
        self.registry = base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            lq.COHORT: {'kind': 'quality', 'status': 'reserved', 'label_paths': [str(self.workbook.resolve())]}}}))
        self.protocol = base / 'protocol.md'
        self.protocol.write_text('prereg')
        self.cache = base / 'dev.npz'
        np.savez(self.cache, train=np.zeros((3, 576)), valid=np.zeros((3, 576)))

        def fake_embed(paths, model, batch=32):
            return np.stack([np.full(576, float(self.truth[Path(p).stem])) for p in paths])  # embedding carries the truth

        def fake_fit(xt, train, qmodel, temperature):
            def predict(features):
                signal = features[:, 0]
                return {'eyeq_reject_logit': signal * 0.5, 'heads': {k: signal * 1.0 for k in ('ungradable', 'artifact', 'clarity', 'field')},
                        'final': signal * 3.0}
            return predict, {'operating_threshold_5pct_false_reject_oof': 0.5, 'regularisation_C': {}, 'temperature': 1.0}

        patches = [mock.patch.object(exposure, 'REGISTRY_PATH', self.registry), mock.patch.object(lq.x9, 'embed', side_effect=fake_embed),
                   mock.patch.object(lq.x9, 'fit_pipeline', side_effect=fake_fit), mock.patch.object(lq.x9, 'load_candidate', return_value=object()),
                   mock.patch.object(lq.x9, 'load_split', return_value=pd.DataFrame({'patient_id': [1]})),
                   mock.patch.object(lq, 'assess_image_quality', side_effect=lambda rgb: (float(rgb.mean() / 255), {}))]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.lock = base / 'lock'

    def args(self, **kw):
        base = dict(output=self.lock, images=self.images, dev_cache=self.cache, protocol=self.protocol, seed=3, replicates=30)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_inventory_marks_the_excluded_patients(self):
        frame = lq.inventory(self.images)
        self.assertEqual(len(frame), 8)
        self.assertEqual(int(frame.primary.sum()), 4)
        self.assertFalse(frame[frame.patient_id.isin([420, 454])].primary.any())

    def test_freeze_reads_no_labels_and_locks(self):
        with mock.patch.object(lq, 'read_workbook', side_effect=AssertionError('labels read at freeze')):
            lq.freeze(self.args())
        lock = json.loads((self.lock / 'lock.json').read_text())
        self.assertIs(lock['labels_read'], False)
        self.assertEqual(exposure.status(lq.COHORT), 'locked')
        self.assertEqual(lock['operating_threshold'], 0.5)
        with self.assertRaises(FileExistsError):
            lq.freeze(self.args())

    def test_evaluate_once_reports_primary_subset_and_marks_exposed(self):
        lq.freeze(self.args())
        lq.evaluate(self.args())
        report = json.loads((self.lock / 'report.json').read_text())
        self.assertEqual(report['population']['primary_images'], 4)
        self.assertEqual(report['population']['unlabelled_primary_images'], [])
        auc = report['results']['stacked']['auc_ungradable']['estimate']
        self.assertGreater(auc, 0.99 if len({self.truth[i] for i in self.truth}) > 1 else 0)
        self.assertIn('stacked_minus_heuristic_iqs', report['comparisons'])
        self.assertEqual(exposure.status(lq.COHORT), 'exposed')
        with self.assertRaises(exposure.ExposureError):
            lq.evaluate(self.args())

    def test_missing_label_rows_are_reported_not_fatal(self):
        pd.DataFrame(self.rows[:-1]).to_excel(self.workbook, index=False)  # drop the last image's row (patient 454, excluded anyway)
        pd.DataFrame([r for r in self.rows if r['image_id'] != '401_l1']).to_excel(self.workbook, index=False)
        lq.freeze(self.args())
        lq.evaluate(self.args())
        report = json.loads((self.lock / 'report.json').read_text())
        self.assertEqual(report['population']['unlabelled_primary_images'], ['401_l1'])
        self.assertEqual(report['population']['primary_images'], 3)

    def test_tampered_binding_is_detected(self):
        lq.freeze(self.args())
        self.protocol.write_text('changed')
        with self.assertRaises(ValueError):
            lq.evaluate(self.args())


if __name__ == '__main__':
    unittest.main()
