"""Tests for the locked mBRSET evaluation: guards, label parsing on a fake same-format file, endpoints, single use."""
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
import locked_mbrset as lm

ROOT = lm.ROOT
HEADER = ('patient,age,sex,dm_time,insulin,insulin_time,oraltreatment_dm,systemic_hypertension,insurance,educational_level,'
          'alcohol_consumption,smoking,obesity,vascular_disease,acute_myocardial_infarction,nephropathy,neuropathy,diabetic_foot,'
          'file,laterality,final_artifacts,final_quality,final_icdr,final_edema').split(',')


class LockedMbrsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'results')
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.data = base / 'images'
        self.data.mkdir()
        rng = np.random.default_rng(0)
        records, self.grade = [], {}
        for patient in range(1, 41):
            for k in (1, 2):
                name = f'{patient}.{k}.jpg'
                cv2.imwrite(str(self.data / name), rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
                g = int(rng.choice([0, 0, 0, 1, 2, 3, 4]))
                quality = 'no' if patient % 13 == 0 else 'yes'
                icdr = '' if (patient == 7 and k == 2) else str(g)
                self.grade[name] = g if icdr else -1
                row = {h: '' for h in HEADER}
                row.update({'patient': str(patient), 'file': name, 'laterality': 'left' if k == 1 else 'right',
                            'final_quality': quality, 'final_icdr': icdr, 'final_artifacts': 'no', 'final_edema': 'no'})
                records.append(row)
        self.labels = base / 'labels_fake.csv'
        pd.DataFrame(records, columns=HEADER).to_csv(self.labels, index=False)
        self.exclusions = base / 'exclusions.txt'
        self.exclusions.write_text('# identity review\n3.1.jpg\n')
        files = sorted(self.grade)
        clusters = pd.DataFrame({'image': [f'/x/{f}' for f in files], 'cluster': range(len(files))})
        clusters.loc[clusters.image == '/x/5.1.jpg', 'cluster'] = int(clusters.loc[clusters.image == '/x/6.2.jpg', 'cluster'].iloc[0])
        self.clusters = base / 'clusters.csv'
        clusters.to_csv(self.clusters, index=False)
        self.protocol = base / 'protocol.md'
        self.protocol.write_text('prereg')
        self.variants = base / 'variants.json'
        self.variants.write_text(json.dumps({
            'primary': 'M', 'references': ['A_legacy', 'R'], 'bootstrap_seed': 5, 'bootstrap_replicates': 30,
            'variants': [{'id': 'A_legacy', 'kind': 'legacy'}, {'id': 'M', 'kind': 'legacy'}, {'id': 'R', 'kind': 'legacy'},
                         {'id': 'M2', 'kind': 'policy_of', 'logits_from': 'M'}, {'id': 'E', 'kind': 'ensemble', 'members': ['M', 'R']}]}))
        self.registry = base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'mbrset': {'kind': 'grading', 'status': 'reserved', 'label_paths': [str(self.labels.resolve())]}}}))

        def fake_logits(spec, frame, folder=None, **kw):
            out = np.stack([np.random.default_rng(zlib.crc32(f'{spec["id"]}|{i}'.encode())).normal(0, 1, 6) for i in frame.image_id])
            g = np.array([max(self.grade[i], 0) for i in frame.image_id])
            out[np.arange(len(g)), g] += 4.0 if spec['id'] == 'M' else 1.0  # M ranks grades better
            return out

        self.fake_logits = fake_logits
        self.x9 = base / 'x9_parameters.json'  # the real file is a git-ignored local artifact; CI has none
        self.x9.write_text(json.dumps({'operating_threshold_5pct_false_reject_oof': 1.25, 'temperature': 0.7}))
        for patch in (mock.patch.object(exposure, 'REGISTRY_PATH', self.registry), mock.patch.object(lm, 'X9_PARAMETERS', self.x9), mock.patch.object(lm, 'logits_for', side_effect=fake_logits)):
            patch.start()
            self.addCleanup(patch.stop)
        self.pre = base / 'pre'
        self.pre.mkdir()
        frame = lm.inventory(self.data, None, None)
        frame[['image_id']].to_csv(self.pre / 'logits_order.csv', index=False)
        for vid in ('A_legacy', 'M', 'R'):
            np.save(self.pre / f'logits_{vid}.npy', fake_logits({'id': vid}, frame))
        n = len(frame)
        np.savez(self.pre / 'quality_scores.npz', image_id=frame.image_id.to_numpy(), eyeq=rng.normal(size=n),
                 final=np.array([3.0 if f.split('.')[0] in ('13', '26', '39') else 0.0 for f in frame.image_id]), iqs=rng.random(n))
        self.lock = base / 'lock'

    def args(self, **kw):
        base = dict(variants=self.variants, output=self.lock, data=self.data, exclusions=self.exclusions, clusters=self.clusters,
                    protocol=self.protocol, precomputed=self.pre)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_inventory_flags_exclusions_and_merges_clusters(self):
        frame = lm.inventory(self.data, self.exclusions, self.clusters)
        self.assertEqual(len(frame), 80)
        self.assertEqual(frame.excluded.sum(), 1)
        self.assertTrue(frame.loc[frame.image_id == '3.1.jpg', 'excluded'].iloc[0])
        cluster = dict(zip(frame.image_id, frame.cluster))
        self.assertEqual(cluster['1.1.jpg'], cluster['1.2.jpg'])  # same patient
        self.assertEqual(cluster['5.2.jpg'], cluster['6.1.jpg'])  # merged through a vessel match
        self.assertNotEqual(cluster['1.1.jpg'], cluster['2.1.jpg'])

    def test_unknown_exclusion_is_refused(self):
        self.exclusions.write_text('999.1.jpg\n')
        with self.assertRaises(ValueError):
            lm.inventory(self.data, self.exclusions, self.clusters)

    def test_label_parser_on_fake_same_format_file(self):
        out = lm.read_labels(lm.inventory(self.data, self.exclusions, self.clusters)).set_index('image_id')
        self.assertTrue(out.has_row.all())
        self.assertFalse(out.loc['13.1.jpg', 'gradable'])   # final_quality 'no'
        self.assertTrue(out.loc['1.1.jpg', 'gradable'])
        self.assertFalse(out.loc['7.2.jpg', 'graded'])      # empty final_icdr is never guessed
        self.assertEqual(out.loc['7.2.jpg', 'icdr'], -1)
        self.assertEqual(out.loc['1.1.jpg', 'icdr'], self.grade['1.1.jpg'])

    def test_unexpected_spelling_is_refused(self):
        table = pd.read_csv(self.labels, dtype=str, keep_default_na=False)
        table.loc[0, 'final_quality'] = 'maybe'
        table.to_csv(self.labels, index=False)
        with self.assertRaises(ValueError):
            lm.read_labels(lm.inventory(self.data, None, None))

    def test_freeze_does_not_open_labels_and_locks(self):
        with mock.patch.object(lm, 'read_labels', side_effect=AssertionError('labels read at freeze')), \
                mock.patch.object(pd, 'read_csv', wraps=pd.read_csv) as reader:
            lm.freeze(self.args())
            self.assertFalse(any(str(self.labels) in str(c) for c in reader.call_args_list))
        lock = json.loads((self.lock / 'lock.json').read_text())
        self.assertIs(lock['labels_read'], False)
        self.assertEqual(exposure.status('mbrset'), 'locked')
        self.assertIn('M', lock['precomputed_logits_cpu_spot_check_max_abs'])
        self.assertTrue(any(k.endswith('labels_fake.csv') for k in lock['bindings']))  # hash bound
        with self.assertRaises(FileExistsError):
            lm.freeze(self.args())

    def test_bad_precomputed_logits_are_rejected(self):
        np.save(self.pre / 'logits_M.npy', np.zeros((80, 6)) + 5.0)
        with self.assertRaises(ValueError):
            lm.freeze(self.args())

    def test_evaluate_requires_lock_runs_once_and_marks_exposed(self):
        with self.assertRaises(Exception):
            lm.evaluate(self.args())
        lm.freeze(self.args())
        lm.evaluate(self.args())
        report = json.loads((self.lock / 'report.json').read_text())
        pop = report['population']
        self.assertEqual(report['tier'], 'TF')
        self.assertEqual(pop['identity_excluded'], 1)
        self.assertEqual(pop['ungradable'], 6)                       # patients 13, 26, 39 x 2 eyes
        self.assertEqual(pop['gradable_without_grade'], ['7.2.jpg'])
        self.assertEqual(pop['primary_images'], 80 - 1 - 6 - 1)
        self.assertGreater(report['results']['M']['auc']['estimate'], report['results']['A_legacy']['auc']['estimate'])
        for key in ('M_vs_A_legacy', 'M_vs_R', 'R_vs_A_legacy', 'E_vs_R', 'M2_vs_A_legacy'):
            self.assertIn(key, report['comparisons'])
        self.assertIn('passed', report['promotion_verdict'])
        self.assertAlmostEqual(report['quality']['results']['stacked']['auc_ungradable']['estimate'], 1.0)
        self.assertEqual(exposure.status('mbrset'), 'exposed')
        with self.assertRaises(exposure.ExposureError):
            lm.evaluate(self.args())

    def test_tampered_binding_is_detected(self):
        lm.freeze(self.args())
        self.protocol.write_text('changed')
        with self.assertRaises(ValueError):
            lm.evaluate(self.args())

    def test_referable_stats_definitions(self):
        grade = np.array([0, 1, 2, 3, 4, 0])
        logits = np.full((6, 6), -10.0)
        logits[np.arange(6), [0, 2, 2, 1, 4, 0]] = 10.0  # predicted grades 0,2,2,1,4,0
        s = lm.referable_stats(grade, logits, np.array([0.5, 1.5, 2.5, 3.5]))
        self.assertAlmostEqual(s['sensitivity'], 2 / 3)   # grades 2,3,4 -> called 2,1,4
        self.assertAlmostEqual(s['specificity'], 2 / 3)   # grades 0,1,0 -> called 0,2,0

    def test_verdict_rule(self):
        res = {'sensitivity': {'estimate': 0.92}, 'specificity': {'estimate': 0.86}}
        good = {'sensitivity': {'difference': 0.1, 'difference_ci95': [0.02, 0.18]}}
        bad = {'sensitivity': {'difference': 0.1, 'difference_ci95': [-0.01, 0.18]}}
        self.assertTrue(lm.verdict(res, good)['passed'])
        self.assertFalse(lm.verdict(res, bad)['passed'])
        self.assertFalse(lm.verdict({'sensitivity': {'estimate': 0.95}, 'specificity': {'estimate': 0.85}}, good)['passed'])

    def test_module_never_names_the_label_file(self):
        source = (ROOT / 'python/locked_mbrset.py').read_text()
        self.assertNotIn('labels_mbrset', source)


if __name__ == '__main__':
    unittest.main()
