"""Tests for the exposure registry and its guards."""
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import exposure
import locked_evaluation as le

ROOT = exposure.ROOT
DDR = 'data/DDR/extracted/DDR-dataset'
DRID = 'data/DRiD/DeepDRiD-master/regular_fundus_images/Online-Challenge1&2-Evaluation'


class RealRegistryTests(unittest.TestCase):
    def test_required_cohorts_and_statuses(self):
        cohorts = exposure.load_registry()['cohorts']
        for name in ('messidor2', 'ddr_test', 'odir', 'eyeq_remainder', 'eyeq_test', 'fdr26',
                     'deepdrid_validation', 'deepdrid_grading_evaluation', 'idrid_test'):
            self.assertEqual(cohorts[name]['status'], 'exposed', name)
        self.assertEqual(cohorts['aptos']['status'], 'training')
        self.assertEqual(cohorts['ddr_lesion_seg_test']['status'], 'exposed')
        for name in ('ddr_lesion_seg_test', 'deepdrid_challenge2_quality', 'rfmid2'):
            # a reserved cohort legitimately moves reserved -> locked -> exposed; it must never be training/planned
            self.assertIn(cohorts[name]['status'], ('reserved', 'locked', 'exposed'), name)
            self.assertTrue(cohorts[name]['label_paths'])

    def test_every_exposed_locked_round_is_registered_with_its_report(self):
        cohorts = exposure.load_registry()['cohorts']
        for name in ('messidor2', 'ddr_test', 'odir', 'eyeq_remainder', 'fdr26'):
            self.assertTrue(cohorts[name].get('lock'), name)
            self.assertTrue(cohorts[name].get('report'), name)

    def test_exposed_cohorts_cannot_be_locked_as_fresh(self):
        for name in ('fdr26', 'odir', 'messidor2', 'aptos', 'ddr_lesion_seg_test', 'rfmid2'):  # both were evaluated once (2026-10-22)
            with self.assertRaises(exposure.ExposureError):
                exposure.assert_fresh_allowed(name)
            exposure.assert_fresh_allowed(name, development=True)  # allowed when declared
        exposure.assert_fresh_allowed('rfmid1')  # planned: fine

    def test_unknown_cohort_is_refused(self):
        with self.assertRaises(exposure.ExposureError):
            exposure.assert_fresh_allowed('not_registered')

    def test_reserved_label_paths_are_detected(self):
        # dynamic: every label path of a cohort that is currently reserved or locked must be detected (after the October 2026
        # evaluations every grading/segmentation/quality test is exposed, so the list may legitimately be empty)
        cohorts = exposure.load_registry()['cohorts']
        protected = [p for item in cohorts.values() if item['status'] in ('reserved', 'locked') for p in item.get('label_paths', [])]
        for path in protected:
            with self.subTest(path=path):
                self.assertIsNotNone(exposure.is_reserved_label_path(ROOT / path))
                self.assertIsNotNone(exposure.is_reserved_label_path(path))
                with self.assertRaises(exposure.ExposureError):
                    exposure.assert_not_reserved_path(path)

    def test_exposed_challenge2_labels_are_no_longer_protected(self):
        self.assertEqual(exposure.status('deepdrid_challenge2_quality'), 'exposed')
        self.assertIsNone(exposure.is_reserved_label_path(f'{DRID}/Challenge2_labels.xlsx'))

    def test_exposed_rfmid2_labels_are_no_longer_protected(self):
        self.assertEqual(exposure.status('rfmid2'), 'exposed')
        self.assertIsNone(exposure.is_reserved_label_path('data/rfmid2/extracted/Test_set/RFMiD_2_Testing_labels.csv'))

    def test_exposed_lesion_test_labels_are_no_longer_protected(self):
        # the DDR lesion test was evaluated once under lock v2 and is exposed: its labels are ordinary development data now
        self.assertEqual(exposure.status('ddr_lesion_seg_test'), 'exposed')
        for path in (f'{DDR}/lesion_segmentation/test/label/MA', f'{DDR}/lesion_detection/test'):
            self.assertIsNone(exposure.is_reserved_label_path(path))

    def test_images_and_exposed_files_are_not_reserved(self):
        allowed = [f'{DDR}/lesion_segmentation/test/image',
                   f'{DDR}/lesion_segmentation/test/image/007-1789-100.jpg',
                   f'{DDR}/lesion_segmentation/train/label/MA',
                   f'{DDR}/lesion_segmentation/valid',
                   f'{DDR}/lesion_detection/train',
                   f'{DRID}/Challenge1_labels.xlsx',
                   f'{DRID}/Images']
        for path in allowed:
            with self.subTest(path=path):
                self.assertIsNone(exposure.is_reserved_label_path(path))

    def test_no_project_code_reads_reserved_labels(self):
        """Static guard: only the exposure module and tests may mention the reserved label locations."""
        markers = ['lesion_segmentation/test/label', 'lesion_detection/test', 'Challenge2_labels',
                   'RFMiD_2_Training_labels', 'RFMiD_2_Validation_labels', 'RFMiD_2_Testing_labels']
        allowed = {'exposure.py'}
        offenders = []
        for path in sorted((ROOT / 'python').rglob('*.py')):
            relative = path.relative_to(ROOT / 'python')
            if relative.parts[0] in ('tests', '.venv') or relative.name in allowed:
                continue
            text = path.read_text(encoding='utf-8', errors='ignore')
            for marker in markers:
                if marker in text:
                    offenders.append(f'{relative}: {marker}')
        self.assertEqual(offenders, [], 'code references reserved label locations; route it through exposure.py')


class TemporaryRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.registry = self.base / 'registry.json'
        self.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
            'res': {'kind': 'grading', 'status': 'reserved', 'label_paths': ['data/x/labels']},
            'used': {'kind': 'grading', 'status': 'exposed'}}}))
        patch = mock.patch.object(exposure, 'REGISTRY_PATH', self.registry)
        patch.start()
        self.addCleanup(patch.stop)

    def lock(self, **overrides):
        directory = self.base / 'lock'
        directory.mkdir(exist_ok=True)
        content = {'tests': {'res': 'x'}, 'labels_read': False}
        content.update(overrides)
        (directory / 'lock.json').write_text(json.dumps(content))
        return directory

    def test_label_read_needs_a_lock(self):
        with self.assertRaisesRegex(exposure.ExposureError, 'No lock'):
            exposure.assert_label_read_allowed('res', self.base / 'nothing')

    def test_label_read_needs_the_cohort_in_the_lock(self):
        with self.assertRaisesRegex(exposure.ExposureError, 'does not list'):
            exposure.assert_label_read_allowed('res', self.lock(tests={'other': 'x'}))

    def test_label_read_refused_when_lock_says_labels_were_read(self):
        with self.assertRaisesRegex(exposure.ExposureError, 'labels_read'):
            exposure.assert_label_read_allowed('res', self.lock(labels_read=True))

    def test_label_read_refused_after_evaluation(self):
        directory = self.lock()
        (directory / 'report.json').write_text('{}')
        with self.assertRaisesRegex(exposure.ExposureError, 'already evaluated'):
            exposure.assert_label_read_allowed('res', directory)

    def test_label_read_allowed_with_a_valid_unevaluated_lock(self):
        exposure.assert_label_read_allowed('res', self.lock())

    def test_record_locked_then_exposed_updates_the_registry(self):
        directory = self.lock()
        exposure.record_locked('res', directory / 'lock.json')
        self.assertEqual(exposure.status('res'), 'locked')
        # a locked cohort's labels are still protected
        self.assertIsNotNone(exposure.is_reserved_label_path(ROOT / 'data/x/labels/file.csv'))
        exposure.record_exposure('res', directory / 'lock.json', directory / 'report.json')
        self.assertEqual(exposure.status('res'), 'exposed')
        self.assertIsNone(exposure.is_reserved_label_path(ROOT / 'data/x/labels/file.csv'))
        self.assertIn('report', exposure.entry('res'))
        with self.assertRaises(exposure.ExposureError):
            exposure.assert_fresh_allowed('res')

    def test_record_exposure_is_idempotent_for_exposed_cohorts(self):
        before = json.loads(self.registry.read_text())
        exposure.record_exposure('used', self.base / 'l.json', self.base / 'r.json')
        self.assertEqual(json.loads(self.registry.read_text()), before)

    def test_exposed_labels_may_be_read_for_development(self):
        directory = self.lock(tests={'used': 'x'})
        with self.assertRaisesRegex(exposure.ExposureError, 'already exposed'):
            exposure.assert_label_read_allowed('used', directory)


class FreezeIntegrationTests(unittest.TestCase):
    """The lock/evaluate path honours the registry (uses the synthetic cohort of the guard tests)."""

    def test_freeze_of_an_exposed_cohort_is_refused_unless_development(self):
        from tests.test_locked_evaluation import FreezeEvaluateGuardTests as base
        case = base('test_freeze_records_no_labels_and_binds_artifacts')
        case.setUp()
        try:
            case.registry.write_text(json.dumps({'schema_version': 1, 'updated': '2026-01-01', 'cohorts': {
                'synth': {'kind': 'grading', 'status': 'exposed'}}}))
            with self.assertRaises(exposure.ExposureError):
                le.freeze(case.args())
            self.assertFalse((case.output / 'lock.json').exists())
            le.freeze(case.args(development=True))
            lock = json.loads((case.output / 'lock.json').read_text())
            self.assertIs(lock['fresh'], False)
            self.assertEqual(json.loads(case.registry.read_text())['cohorts']['synth']['status'], 'exposed')
        finally:
            case.doCleanups()

    def test_freeze_then_evaluate_moves_a_reserved_cohort_to_locked_then_exposed(self):
        from tests.test_locked_evaluation import FreezeEvaluateGuardTests as base
        case = base('test_freeze_records_no_labels_and_binds_artifacts')
        case.setUp()
        try:
            le.freeze(case.args())
            self.assertEqual(json.loads(case.registry.read_text())['cohorts']['synth']['status'], 'locked')
            self.assertIs(json.loads((case.output / 'lock.json').read_text())['fresh'], True)
            le.evaluate(case.args())
            item = json.loads(case.registry.read_text())['cohorts']['synth']
            self.assertEqual(item['status'], 'exposed')
            self.assertTrue(item['report'].endswith('report.json'))
        finally:
            case.doCleanups()


if __name__ == '__main__':
    unittest.main()
