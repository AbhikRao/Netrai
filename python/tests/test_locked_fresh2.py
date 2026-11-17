"""Tests for the round-6 fresh-set locks (Paraguay, SUSTech-SYSU): blind extraction, sealed label parsing on fake same-format files,
duplicate/category-5 rules, the calibration-procedure endpoint and single use."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pandas as pd

import blind_extract
import exposure
import locked_fresh2 as lf

ROOT = lf.ROOT


def jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    ok, data = cv2.imencode('.jpg', rng.integers(0, 255, (16, 16, 3), dtype=np.uint8))
    return data.tobytes()


class FreshTwoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'results')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.grade = {}
        # --- fake Paraguay archive: category folders, one byte-identical copy in the same folder, one in a conflicting folder
        folders = list(lf.PARAGUAY_FOLDERS)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            for i in range(60):
                folder = folders[i % len(folders)]
                zf.writestr(f'{folder}/{i}.jpg', jpeg(i))
            zf.writestr(f'{folders[0]}/dup_same.jpg', jpeg(0))         # copy of image 0, same category
            zf.writestr(f'{folders[5]}/dup_conflict.jpg', jpeg(1))     # copy of image 1 (category 2) in a referable category
            zf.writestr('Annotations of the classifications.xlsx', b'not-a-real-workbook')
        (self.base / 'py.zip').write_bytes(buf.getvalue())
        self.py = self.base / 'paraguay_dr'
        blind_extract.blind_extract(self.base / 'py.zip', self.py, self.py / 'sealed', r'(?i)\.(jpe?g)$', 'py_')
        # --- fake SUSTech: originalImages + drLabels + c5 reclassification
        buf = io.BytesIO()
        rows, c5 = [], []
        with zipfile.ZipFile(buf, 'w') as zf:
            for i in range(80):
                zf.writestr(f'originalImages/{i + 1}.jpg', jpeg(100 + i))
                g = [0, 0, 0, 1, 2, 3, 4, 5][i % 8]
                rows.append([f'{i + 1}.jpg', i % 2, g, g, g])
                if g == 5:
                    c5.append([f'{i + 1}.jpg', 3, 3, 3])
            out = io.StringIO(); w = csv.writer(out)
            w.writerow(['Fundus_images', 'left_versus_right_eye(left_0_right_1)', lf.SUSTECH_ICDR, 'DR_grade(American_Academy_of_Ophthalmology)', 'DR_grade(Scottish_DR_grading_protocol)'])
            w.writerows(rows); zf.writestr('originalImages/drLabels.csv', out.getvalue())
            out = io.StringIO(); w = csv.writer(out)
            w.writerow(['Fundus_images', lf.SUSTECH_ICDR, 'DR_grade(American_Academy_of_Ophthalmology)', 'DR_grade(Scottish_DR_grading_protocol)'])
            w.writerows(c5); zf.writestr('originalImages/c5_DR_reclassified.csv', out.getvalue())
            zf.writestr('exudatesLabels/1.xml', '<x/>')
        (self.base / 'ss.zip').write_bytes(buf.getvalue())
        self.ss = self.base / 'sustech_sysu'
        blind_extract.blind_extract(self.base / 'ss.zip', self.ss, self.ss / 'sealed', r'(?i)^originalImages/.*\.jpe?g$', 'ss_')
        self.registry = self.base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'paraguay_hc': {'kind': 'grading', 'status': 'reserved', 'label_paths': [str((self.py / 'sealed').resolve())]},
            'sustech_sysu': {'kind': 'grading', 'status': 'reserved', 'label_paths': [str((self.ss / 'sealed').resolve())]}}}))
        self.variants = self.base / 'variants.json'
        self.variants.write_text(json.dumps({'primary': 'M', 'reference': 'A_legacy', 'bootstrap_seed': 5, 'bootstrap_replicates': 30,
                                             'variants': [{'id': 'A_legacy', 'kind': 'legacy'}, {'id': 'M', 'kind': 'legacy'}]}))
        self.protocol = self.base / 'protocol.md'
        self.protocol.write_text('prereg')

        def fake_logits(spec, frame, folder=None, **kw):
            return np.stack([np.random.default_rng(zlib.crc32(f'{spec["id"]}|{i}'.encode())).normal(0, 1, 6) for i in frame.image_id])

        self.fake_logits = fake_logits
        patches = [mock.patch.object(exposure, 'REGISTRY_PATH', self.registry), mock.patch.object(lf, 'logits_for', side_effect=fake_logits),
                   mock.patch.object(lf, 'data_dir', side_effect=lambda c: {'paraguay_hc': self.py, 'sustech_sysu': self.ss}[c]),
                   mock.patch.object(lf, 'CODE', ['python/locked_fresh2.py'])]
        for p in patches:
            p.start(); self.addCleanup(p.stop)

    def precompute(self, cohort):
        pre = self.base / f'pre_{cohort}'
        pre.mkdir()
        frame = lf.inventory(cohort)
        frame[['image_id']].to_csv(pre / 'logits_order.csv', index=False)
        for vid in ('A_legacy', 'M'):
            np.save(pre / f'logits_{vid}.npy', self.fake_logits({'id': vid}, frame))
        return pre

    def args(self, cohort, **kw):
        base = dict(cohort=cohort, variants=self.variants, output=self.base / f'lock_{cohort}', protocol=self.protocol, precomputed=self.precompute(cohort),
                    exclusions=None, clusters=None, images=None, adabn_logits=None, adabn_list=None, near_identical=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_blind_extraction_hides_labels_and_counts_duplicates(self):
        names = sorted(p.name for p in (self.py / 'images').iterdir())
        self.assertEqual(len(names), 60)                                   # 62 archive images, 2 byte-identical copies
        self.assertTrue(all(n.startswith('py_') and len(n) == len('py_') + 16 + 4 for n in names))
        mapping = pd.read_csv(self.py / 'sealed/mapping.csv')
        self.assertEqual(len(mapping), 62)
        self.assertTrue((self.py / 'sealed/Annotations of the classifications.xlsx').is_file())
        self.assertFalse((self.ss / 'images').joinpath('1.xml').exists())
        self.assertTrue((self.ss / 'sealed/exudatesLabels/1.xml').is_file())

    def test_paraguay_labels_duplicate_rules(self):
        frame = lf.inventory('paraguay_hc')
        out, notes = lf.read_labels('paraguay_hc', frame)
        self.assertEqual(len(notes['images_with_conflicting_categories']), 1)
        self.assertEqual(notes['duplicate_copies_same_category'], 1)
        self.assertEqual(int((~out.primary).sum()), 1)                    # only the conflicting image leaves the primary population
        self.assertTrue(set(out.grade[out.primary]) <= {0, 1, 2, 3, 4})

    def test_sustech_labels_category5_secondary(self):
        frame = lf.inventory('sustech_sysu')
        out, notes = lf.read_labels('sustech_sysu', frame)
        self.assertEqual(notes['category5_images'], 10)
        self.assertEqual(int(out.secondary.sum()), 10)
        self.assertEqual(int(out.primary.sum()), 70)
        self.assertTrue((out.grade[out.secondary] == 3).all())             # re-classified ICDR for laser-treated eyes
        self.assertEqual(notes['images_without_label_row'], 0)

    def test_freeze_does_not_open_labels_then_evaluate_once(self):
        for cohort in ('paraguay_hc', 'sustech_sysu'):
            args = self.args(cohort)
            with mock.patch.object(lf, 'read_labels', side_effect=AssertionError('labels read at freeze')):
                lf.freeze(args)
            lock = json.loads((args.output / 'lock.json').read_text())
            self.assertIs(lock['labels_read'], False)
            self.assertEqual(exposure.status(cohort), 'locked')
            self.assertTrue(any('sealed' in k for k in lock['bindings']))
            lf.evaluate(args)
            report = json.loads((args.output / 'report.json').read_text())
            self.assertEqual(report['tier'], 'TF')
            self.assertIn('M_vs_A_legacy', report['comparisons'])
            self.assertEqual(set(report['site_calibration_procedure']), {'100', '200'} if report['population']['primary_images'] > 100 else {'100', '200'})
            self.assertEqual(exposure.status(cohort), 'exposed')
            with self.assertRaises(exposure.ExposureError):
                lf.evaluate(args)

    def test_tampered_sealed_file_is_detected(self):
        args = self.args('sustech_sysu')
        lf.freeze(args)
        path = self.ss / 'sealed/originalImages/drLabels.csv'
        path.write_text(path.read_text() + '\n')
        with self.assertRaises(ValueError):
            lf.evaluate(args)

    def test_near_identical_conflicts_leave_the_population(self):
        frame = lf.inventory('sustech_sysu')
        out, _ = lf.read_labels('sustech_sysu', frame)
        primary = out[out.primary]
        a = primary[primary.grade == 0].image_id.iloc[0]; b = primary[primary.grade == 2].image_id.iloc[0]; c = primary[primary.grade == 0].image_id.iloc[1]
        pairs = pd.DataFrame({'a': [a, a], 'b': [b, c], 'corr': [0.99, 0.97]})
        fixed, conflicted = lf.exclude_conflicting_near_copies(out, pairs)
        self.assertEqual(conflicted, sorted([a, b]))                       # a/c agree, so c stays
        self.assertFalse(fixed.set_index('image_id').loc[[a, b], 'primary'].any())
        self.assertTrue(fixed.set_index('image_id').loc[c, 'primary'])

    def test_calibration_draws_reach_target_on_separable_scores(self):
        rng = np.random.default_rng(0)
        y = np.r_[np.ones(300, bool), np.zeros(700, bool)]
        score = np.where(y, rng.normal(2.5, 0.4, 1000), rng.normal(0.5, 0.4, 1000))
        out = lf.calibration_draws(score, y, np.arange(1000).astype(str), frozen=3.5, size=200, draws=50, seed=1)
        self.assertLess(out['frozen_sensitivity'], 0.2)                    # badly placed frozen cut
        self.assertGreater(out['sensitivity_mean'], 0.88)
        self.assertGreaterEqual(out['p_both_targets'], 0.7)


if __name__ == '__main__':
    unittest.main()
