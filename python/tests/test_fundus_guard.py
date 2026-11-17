"""The non-fundus input guard (modules/fundus_guard.py): parameters bound by SHA-256 and shared byte for byte with MATLAB, a fundus
photograph passes, a portrait is not graded, and the scores equal the cross-runtime fixture that matlab/tests/testCheckFundusPhoto.m
also checks."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1]
ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

from modules import fundus_guard as fg  # noqa: E402

PORTRAIT = Path(__import__('matplotlib').get_data_path()) / 'sample_data/grace_hopper.jpg'
MATLAB = shutil.which('matlab')     # MATLAB's own sample image, when MATLAB is installed on this machine
PEPPERS = Path(MATLAB).resolve().parents[1] / 'toolbox/matlab/matlab_images/png/peppers.png' if MATLAB else None
FIXTURE = ROOT / 'matlab/tests/fixtures/fundus_guard_scores.json'
SAMPLE = ROOT / 'data/patient_sample.jpg'       # a local fundus photograph; datasets and images are not part of the repository
needs_sample = unittest.skipUnless(SAMPLE.is_file(), 'the local sample fundus photograph is not in this checkout')


def rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class FundusGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard = fg.FundusGuard()
        cls.fixture = json.loads(FIXTURE.read_text())

    @needs_sample
    def test_fundus_photograph_passes(self):
        self.assertTrue(self.guard.check(rgb(SAMPLE))['is_fundus'])

    def test_a_portrait_does_not_pass(self):
        portrait = self.guard.check(rgb(PORTRAIT))
        self.assertFalse(portrait['is_fundus'])
        self.assertEqual(portrait['threshold'], self.guard.threshold)

    @needs_sample
    def test_fundus_score_equals_the_cross_runtime_fixture(self):
        # MATLAB and this machine agree to 3e-4 on the sample score; other CPUs and BLAS builds may differ a little more
        self.assertAlmostEqual(self.guard.score(rgb(SAMPLE)), self.fixture['patient_sample'], delta=1e-3)

    def test_non_fundus_scores_equal_the_cross_runtime_fixture(self):
        self.assertAlmostEqual(self.guard.score(rgb(PORTRAIT)), self.fixture['grace_hopper'], delta=1e-4 * self.fixture['grace_hopper'])
        if PEPPERS is not None and PEPPERS.is_file():
            self.assertAlmostEqual(self.guard.score(rgb(PEPPERS)), self.fixture['peppers'], delta=1e-4 * self.fixture['peppers'])

    def test_runtimes_share_byte_identical_parameters(self):
        for name in ('fundus_guard.json', 'fundus_guard.f32'):
            self.assertEqual((ROOT / 'python/weights' / name).read_bytes(), (ROOT / 'matlab/models' / name).read_bytes(), name)

    def test_tampered_parameters_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copy(fg.PARAMS, folder)
            blob = bytearray((fg.WEIGHTS / 'fundus_guard.f32').read_bytes())
            blob[4 * 600] ^= 1
            Path(folder, 'fundus_guard.f32').write_bytes(bytes(blob))
            with self.assertRaises(fg.GuardError):
                fg.FundusGuard(Path(folder) / 'fundus_guard.json')

    def test_pipeline_rejects_or_flags_a_portrait(self):
        import inference
        with tempfile.TemporaryDirectory() as folder:
            result = inference.run_pipeline(str(PORTRAIT), output_dir=folder, variant='legacy')
            saved = json.loads(Path(folder, 'result.json').read_text())
        self.assertFalse(result['fundus_guard']['is_fundus'])
        if self.guard.action == 'reject':
            self.assertEqual((result['status'], result['rejection_reason']), ('rejected', 'not_fundus'))
            self.assertEqual(result['triage_action'], 'recapture')
            self.assertNotIn('grade', result)
            self.assertEqual(saved['rejection_reason'], 'not_fundus')
        else:                                   # the shipped guard flags: graded, with a visible warning in result and report
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['input_warning'], fg.FEEDBACK)
            self.assertEqual(saved['input_warning'], fg.FEEDBACK)

    @needs_sample
    def test_fundus_photograph_carries_no_warning(self):
        import inference
        with tempfile.TemporaryDirectory() as folder:
            result = inference.run_pipeline(str(SAMPLE), output_dir=folder, variant='legacy')
        self.assertTrue(result['fundus_guard']['is_fundus'])
        self.assertNotIn('input_warning', result)

    def test_poor_quality_rejection_keeps_the_same_schema(self):
        import inference
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'black.png'
            cv2.imwrite(str(path), np.zeros((256, 256, 3), np.uint8))
            result = inference.run_pipeline(str(path), output_dir=folder, variant='legacy')
        self.assertEqual((result['status'], result['rejection_reason'], result['fundus_guard']), ('rejected', 'image_quality', None))


if __name__ == '__main__':
    unittest.main()
