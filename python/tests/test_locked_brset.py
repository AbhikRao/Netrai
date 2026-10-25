"""Tests for the locked BRSET confirmation: guards, patient clusters, label parsing on a fake same-format file, single use."""
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
import locked_brset as lb

ROOT = lb.ROOT
HEADER = ['image_id', 'patient_id', 'camera', 'patient_age', 'comorbidities', 'diabetes_time_y', 'insulin', 'patient_sex', 'exam_eye',
          'diabetes', 'nationality', 'optic_disc', 'vessels', 'macula', 'DR_SDRG', 'DR_ICDR', 'focus', 'illumination', 'image_field',
          'artifacts', 'diabetic_retinopathy', 'macular_edema', 'scar', 'nevus', 'amd', 'vascular_occlusion', 'hypertensive_retinopathy',
          'drusen', 'hemorrhage', 'retinal_detachment', 'myopic_fundus', 'increased_cup_disc', 'other', 'quality']


class LockedBrsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'results')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.data = base / 'fundus_photos'
        self.data.mkdir()
        rng = np.random.default_rng(0)
        records, patients, self.grade = [], [], {}
        n = 0
        for patient in range(1, 61):
            for k in (1, 2):
                n += 1
                ident = f'img{n:05d}'
                name = ident + '.jpg'
                cv2.imwrite(str(self.data / name), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
                g = int(rng.choice([0, 0, 0, 1, 2, 3, 4]))
                row = {h: '0' for h in HEADER}
                row.update({'image_id': ident, 'patient_id': str(patient), 'camera': 'Canon CR' if patient % 2 else 'NIKON NF5050',
                            'DR_ICDR': str(g), 'DR_SDRG': str(g), 'quality': 'Inadequate' if patient % 17 == 0 else 'Adequate',
                            'focus': 'Satisfatory' if patient % 11 else 'Unsatisfactory', 'illumination': 'Satisfatory', 'image_field': 'Complete',
                            'artifacts': 'Does not have', 'amd': '1' if (g == 0 and patient % 5 == 0) else '0'})
                records.append(row)
                patients.append((name, str(patient)))
                self.grade[name] = g
        self.labels = base / 'label_fake.csv'
        pd.DataFrame(records, columns=HEADER).to_csv(self.labels, index=False)
        self.patients = base / 'patients.csv'
        pd.DataFrame(patients, columns=['image_id', 'patient_id']).to_csv(self.patients, index=False)
        self.exclusions = base / 'exclusions.txt'
        self.exclusions.write_text('# review\nimg00003.jpg\n')
        self.protocol = base / 'protocol.md'
        self.protocol.write_text('prereg')
        self.variants = base / 'variants.json'
        self.variants.write_text(json.dumps({
            'primary': 'M', 'references': ['A_legacy', 'R'], 'bootstrap_seed': 5, 'bootstrap_replicates': 30,
            'variants': [{'id': 'A_legacy', 'kind': 'legacy'}, {'id': 'M', 'kind': 'legacy'}, {'id': 'R', 'kind': 'legacy'},
                         {'id': 'E', 'kind': 'ensemble', 'members': ['M', 'R']}]}))
        self.registry = base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'brset': {'kind': 'grading', 'status': 'reserved', 'label_paths': [str(self.labels.resolve())]}}}))

        def fake_logits(spec, frame, folder=None, **kw):
            out = np.stack([np.random.default_rng(zlib.crc32(f'{spec["id"]}|{i}'.encode())).normal(0, 1, 6) for i in frame.image_id])
            g = np.array([self.grade[i] for i in frame.image_id])
            out[np.arange(len(g)), g] += 4.0 if spec['id'] == 'M' else 1.0
            return out

        self.x9 = base / 'x9_parameters.json'  # the real file is a git-ignored local artifact; CI has none
        self.x9.write_text(json.dumps({'operating_threshold_5pct_false_reject_oof': 1.25, 'temperature': 0.7}))
        for patch in (mock.patch.object(exposure, 'REGISTRY_PATH', self.registry), mock.patch.object(lb, 'X9_PARAMETERS', self.x9), mock.patch.object(lb, 'logits_for', side_effect=fake_logits)):
            patch.start()
            self.addCleanup(patch.stop)
        self.pre = base / 'pre'
        self.pre.mkdir()
        frame = lb.inventory(self.data, None, None)
        frame[['image_id']].to_csv(self.pre / 'logits_order.csv', index=False)
        for vid in ('A_legacy', 'M', 'R'):
            np.save(self.pre / f'logits_{vid}.npy', fake_logits({'id': vid}, frame))
        np.savez(self.pre / 'quality_scores.npz', image_id=frame.image_id.to_numpy(), eyeq=rng.normal(size=len(frame)), final=rng.normal(size=len(frame)),
                 iqs=rng.random(len(frame)))
        self.lock = base / 'lock'

    def args(self, **kw):
        base = dict(variants=self.variants, output=self.lock, data=self.data, exclusions=self.exclusions, patients=self.patients,
                    protocol=self.protocol, precomputed=self.pre)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_inventory_uses_patient_ids_and_flags_exclusions(self):
        frame = lb.inventory(self.data, self.exclusions, self.patients)
        self.assertEqual(len(frame), 120)
        self.assertEqual(frame.excluded.sum(), 1)
        self.assertEqual(frame.cluster.nunique(), 60)
        self.assertEqual(frame.set_index('image_id').loc['img00001.jpg', 'cluster'], frame.set_index('image_id').loc['img00002.jpg', 'cluster'])

    def test_missing_patient_or_unknown_exclusion_is_refused(self):
        pd.read_csv(self.patients).iloc[1:].to_csv(self.patients, index=False)
        with self.assertRaises(ValueError):
            lb.inventory(self.data, self.exclusions, self.patients)
        self.exclusions.write_text('img99999.jpg\n')
        with self.assertRaises(ValueError):
            lb.inventory(self.data, self.exclusions, None)

    def test_label_parser_on_fake_same_format_file(self):
        out, spellings = lb.read_labels(lb.inventory(self.data, self.exclusions, self.patients))
        out = out.set_index('image_id')
        self.assertTrue(out.has_row.all())
        self.assertFalse(out.loc['img00067.jpg', 'gradable'])          # patient 34 -> Inadequate
        self.assertTrue(out.loc['img00001.jpg', 'gradable'])
        self.assertEqual(out.loc['img00001.jpg', 'icdr'], self.grade['img00001.jpg'])
        self.assertEqual(out.loc['img00001.jpg', 'camera'], 'Canon CR')
        self.assertTrue(out.other_disease.any())
        self.assertTrue(out.any_factor_unsatisfactory.any())
        self.assertIn('Unsatisfactory', spellings['focus'])

    def test_unexpected_spelling_is_refused(self):
        table = pd.read_csv(self.labels, dtype=str)
        table.loc[0, 'quality'] = 'Good'
        table.to_csv(self.labels, index=False)
        with self.assertRaises(ValueError):
            lb.read_labels(lb.inventory(self.data, None, self.patients))

    def test_freeze_does_not_open_labels_and_locks(self):
        with mock.patch.object(lb, 'read_labels', side_effect=AssertionError('labels read at freeze')):
            lb.freeze(self.args())
        lock = json.loads((self.lock / 'lock.json').read_text())
        self.assertIs(lock['labels_read'], False)
        self.assertEqual(exposure.status('brset'), 'locked')
        self.assertTrue(any(k.endswith('patients.csv') for k in lock['bindings']))
        self.assertTrue(any(k.endswith('label_fake.csv') for k in lock['bindings']))
        with self.assertRaises(FileExistsError):
            lb.freeze(self.args())

    def test_bad_precomputed_logits_are_rejected(self):
        np.save(self.pre / 'logits_M.npy', np.zeros((120, 6)) + 5.0)
        with self.assertRaises(ValueError):
            lb.freeze(self.args())

    def test_evaluate_requires_lock_runs_once_and_marks_exposed(self):
        with self.assertRaises(Exception):
            lb.evaluate(self.args())
        lb.freeze(self.args())
        lb.evaluate(self.args())
        report = json.loads((self.lock / 'report.json').read_text())
        pop = report['population']
        self.assertEqual(pop['identity_excluded'], 1)
        self.assertEqual(pop['inadequate_quality'], 6)                 # patients 17, 34, 51 x 2 eyes
        self.assertEqual(pop['primary_images'], 120 - 1 - 6)
        self.assertGreater(report['results']['M']['auc']['estimate'], report['results']['A_legacy']['auc']['estimate'])
        for key in ('M_vs_A_legacy', 'M_vs_R', 'R_vs_A_legacy', 'E_vs_R'):
            self.assertIn(key, report['comparisons'])
        self.assertIn('passed', report['promotion_verdict'])
        self.assertIn('Canon CR', report['per_camera'])
        self.assertIn('specificity_with_other_disease', report['other_disease_specificity']['M'])
        self.assertIn('inadequate', report['quality'])
        self.assertEqual(exposure.status('brset'), 'exposed')
        with self.assertRaises(exposure.ExposureError):
            lb.evaluate(self.args())

    def test_tampered_binding_is_detected(self):
        lb.freeze(self.args())
        self.patients.write_text(self.patients.read_text() + '\n')
        with self.assertRaises(ValueError):
            lb.evaluate(self.args())

    def test_other_disease_stats(self):
        grade = np.array([0, 0, 0, 0, 3])
        logits = np.full((5, 6), -10.0)
        logits[np.arange(5), [0, 3, 0, 0, 3]] = 10.0                   # predicted grades 0,3,0,0,3
        s = lb.other_disease_stats(grade, logits, np.array([0.5, 1.5, 2.5, 3.5]), np.array([False, True, True, False, False]))
        self.assertAlmostEqual(s['specificity_with_other_disease'], 0.5)
        self.assertAlmostEqual(s['specificity_without_other_disease'], 1.0)

    def test_module_never_names_the_label_file(self):
        self.assertNotIn('label_brset', (ROOT / 'python/locked_brset.py').read_text())


if __name__ == '__main__':
    unittest.main()
