"""Tests for the per-site calibration tool: boundary fitting, monotone thresholds, guards, policy file, calibration/verification identities."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import calibrate_site as cs

BASE = {'thresholds': [1.0, 1.2, 2.0, 2.8], 'head_threshold': 0.2, 'temperature': 0.8}
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / 'matlab/tests/fixtures/site_policy'
E768 = ROOT / 'matlab/variant_R3_E_768'


def fake_digests(prefix, n):
    return [hashlib.sha256(f'{prefix}-{i}'.encode()).hexdigest() for i in range(n)]


def write_image_list(logits_path, digests, model='a' * 64, names=None):
    """The image list ``score`` writes next to its logits (here for synthetic logits)."""
    names = names or [f'{d[:8]}.jpg' for d in digests]
    Path(logits_path).with_suffix('.images.json').write_text(json.dumps(
        {'schema_version': 1, 'model_sha256': model, 'config_sha256': 'b' * 64, 'images': list(names), 'sha256': list(digests)}))


def logits_for_scores(scores):
    """Logits whose expected grade equals ``scores`` approximately: two-class mixture between floor and ceil grade."""
    out = np.full((len(scores), 6), -12.0)
    for i, s in enumerate(scores):
        lo = int(np.floor(min(s, 3.999)))
        frac = s - lo
        out[i, lo] = np.log(max(1 - frac, 1e-6))
        out[i, lo + 1] = np.log(max(frac, 1e-6))
    return out


class CalibrateSiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        rng = np.random.default_rng(0)
        n_pos, n_neg = 80, 220
        score = np.concatenate([rng.normal(2.2, 0.5, n_pos), rng.normal(0.4, 0.4, n_neg)]).clip(0, 3.9)
        grade = np.concatenate([np.full(n_pos, 2), np.zeros(n_neg, int)])
        self.score, self.grade = score, grade
        np.save(self.dir / 'logits.npy', logits_for_scores(score))
        write_image_list(self.dir / 'logits.npy', fake_digests('cal', len(score)))
        pd.DataFrame({'grade': grade, 'patient': np.arange(len(grade)) // 2}).to_csv(self.dir / 'labels.csv', index=False)
        (self.dir / 'base.json').write_text(json.dumps(BASE))
        (self.dir / 'netrai_bundle.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [
            {'role': 'model', 'path': 'm.onnx', 'sha256': 'a' * 64}, {'role': 'config', 'path': 'c.json', 'sha256': 'b' * 64}]}))

    def args(self, **kw):
        base = dict(logits=self.dir / 'logits.npy', labels=self.dir / 'labels.csv', base_policy=self.dir / 'base.json',
                    output=self.dir / 'site.json', site='test', target_sensitivity=0.92, bundle=None, variant='E_768')
        base.update(kw)
        return SimpleNamespace(**base)

    def test_fit_boundary_reaches_target_sensitivity(self):
        referable = self.grade >= 2
        cut = cs.fit_boundary(cs.expected_grade(logits_for_scores(self.score)), referable, 0.92)
        sens = (cs.expected_grade(logits_for_scores(self.score))[referable] >= cut).mean()
        self.assertGreaterEqual(sens, 0.92)
        self.assertLess(sens, 0.97)

    def test_monotone_thresholds_keep_the_referral_boundary_exact_and_strictly_increasing(self):
        self.assertEqual(cs.monotone_thresholds([1.0, 1.2, 2.0, 2.8], 1.5), [1.0, 1.5, 2.0, 2.8])
        for cut in (0.7, 2.4, 2.9, 0.0):
            t = cs.monotone_thresholds([1.0, 1.2, 2.0, 2.8], cut)
            self.assertEqual(t[1], cut)
            self.assertTrue(all(b > a for a, b in zip(t, t[1:])), t)

    def test_fit_writes_policy_with_provenance_and_no_validation_claim(self):
        policy = cs.fit(self.args())
        saved = json.loads((self.dir / 'site.json').read_text())
        self.assertEqual(saved['status'], 'site_calibrated')
        self.assertEqual(saved['site']['referable'], 80)
        self.assertEqual(saved['site']['patients'], 150)
        self.assertIn('NONE', saved['site']['validation'])
        self.assertEqual(saved['head_threshold'], 0.2)
        self.assertTrue(np.all(np.diff(saved['thresholds']) >= 0) or saved['thresholds'][0] <= saved['thresholds'][1])
        self.assertEqual(policy['thresholds'], saved['thresholds'])
        self.assertEqual((saved['model_variant'], saved['model_sha256'], saved['config_sha256']), ('E_768', 'a' * 64, 'b' * 64))

    def test_policy_without_a_bundle_cannot_be_created(self):
        (self.dir / 'netrai_bundle.json').unlink()
        with self.assertRaises(FileNotFoundError):
            cs.fit(self.args())
        (self.dir / 'bundle2.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [{'role': 'model', 'path': 'm', 'sha256': 'c' * 64}]}))
        with self.assertRaises(ValueError):
            cs.fit(self.args(bundle=self.dir / 'bundle2.json'))

    def test_too_few_referable_cases_is_refused_and_small_samples_warn(self):
        pd.DataFrame({'grade': [0] * 100 + [2] * 3}).to_csv(self.dir / 'tiny.csv', index=False)
        np.save(self.dir / 'tiny.npy', logits_for_scores(np.r_[np.full(100, .3), np.full(3, 2.2)]))
        with self.assertRaises(ValueError):
            cs.fit(self.args(logits=self.dir / 'tiny.npy', labels=self.dir / 'tiny.csv'))
        pd.DataFrame({'grade': [0] * 100 + [2] * 10}).to_csv(self.dir / 'small.csv', index=False)
        np.save(self.dir / 'small.npy', logits_for_scores(np.r_[np.full(100, .3), np.full(10, 2.2)]))
        write_image_list(self.dir / 'small.npy', fake_digests('small', 110))
        policy = cs.fit(self.args(logits=self.dir / 'small.npy', labels=self.dir / 'small.csv'))
        self.assertTrue(any('referable' in w for w in policy['site']['warnings']))

    def test_matlab_style_base_config_is_accepted(self):
        (self.dir / 'matlab.json').write_text(json.dumps({'grade_thresholds': [1.0, 1.2, 2.0, 2.8], 'referable_head_threshold': 0.2}))
        policy = cs.fit(self.args(base_policy=self.dir / 'matlab.json'))
        self.assertNotIn('grade_thresholds', policy)
        self.assertEqual(len(policy['thresholds']), 4)

    def test_length_mismatch_and_bad_grades_are_refused(self):
        pd.DataFrame({'grade': [0, 1]}).to_csv(self.dir / 'short.csv', index=False)
        with self.assertRaises(ValueError):
            cs.fit(self.args(labels=self.dir / 'short.csv'))
        pd.DataFrame({'grade': [7] * len(self.grade)}).to_csv(self.dir / 'bad.csv', index=False)
        with self.assertRaises(ValueError):
            cs.fit(self.args(labels=self.dir / 'bad.csv'))

    def test_blank_or_fractional_grades_are_refused(self):
        for bad in (np.nan, 2.5):
            grades = self.grade.astype(float)
            grades[3] = bad
            pd.DataFrame({'grade': grades}).to_csv(self.dir / 'holes.csv', index=False)
            with self.assertRaises(ValueError):
                cs.fit(self.args(labels=self.dir / 'holes.csv'))


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        (self.dir / 'base.json').write_text(json.dumps(BASE))
        (self.dir / 'netrai_bundle.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [
            {'role': 'model', 'path': 'm.onnx', 'sha256': 'a' * 64}, {'role': 'config', 'path': 'c.json', 'sha256': 'b' * 64}]}))
        rng = np.random.default_rng(3)
        for name, seed in (('cal', 1), ('chk', 2)):
            r = np.random.default_rng(seed)
            score = np.r_[r.normal(2.4, 0.4, 80), r.normal(0.4, 0.3, 220)].clip(0, 3.9)
            np.save(self.dir / f'{name}.npy', logits_for_scores(score))
            write_image_list(self.dir / f'{name}.npy', fake_digests(name, 300))
            pd.DataFrame({'grade': [2] * 80 + [0] * 220, 'patient': [f'{name}{i // 2}' for i in range(300)]}).to_csv(self.dir / f'{name}.csv', index=False)
        self.policy = cs.fit(SimpleNamespace(logits=self.dir / 'cal.npy', labels=self.dir / 'cal.csv', base_policy=self.dir / 'base.json',
                                             output=self.dir / 'site.json', site='clinic', target_sensitivity=0.95, bundle=None, variant='E_768'))

    def args(self, **kw):
        base = dict(policy=self.dir / 'site.json', logits=self.dir / 'chk.npy', labels=self.dir / 'chk.csv', output=self.dir / 'check.json',
                    calibration_labels=self.dir / 'cal.csv', base_policy=self.dir / 'base.json', bundle=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_wilson_interval(self):
        low, high = cs.wilson(9, 10)
        self.assertAlmostEqual(low, 0.5958, places=3)
        self.assertAlmostEqual(high, 0.9821, places=3)

    def test_verify_on_a_separate_sample(self):
        result = cs.verify(self.args())
        self.assertEqual(result['referable'], 80)
        self.assertTrue(result['meets_targets_point'])
        self.assertLessEqual(result['sensitivity_ci95'][0], result['sensitivity'])
        self.assertTrue(json.loads((self.dir / 'check.json').read_text())['meets_targets_point'])

    def test_verify_refuses_the_calibration_sample_another_model_and_shared_patients(self):
        with self.assertRaises(ValueError):
            cs.verify(self.args(logits=self.dir / 'cal.npy', labels=self.dir / 'cal.csv'))
        (self.dir / 'other_bundle.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [
            {'role': 'model', 'path': 'm', 'sha256': 'c' * 64}, {'role': 'config', 'path': 'c', 'sha256': 'b' * 64}]}))
        with self.assertRaises(ValueError):
            cs.verify(self.args(bundle=self.dir / 'other_bundle.json'))
        shared = pd.read_csv(self.dir / 'chk.csv'); shared.loc[0, 'patient'] = 'cal0'; shared.to_csv(self.dir / 'chk_shared.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'patients'):              # refused by default, without --calibration-labels
            cs.verify(self.args(labels=self.dir / 'chk_shared.csv', calibration_labels=None))

    def test_patients_are_recorded_hashed_and_required_for_verification(self):
        self.assertEqual(self.policy['calibration_patient_sha256'], sorted(cs.text_sha256(f'cal{i}') for i in range(150)))
        self.assertNotIn('cal17', (self.dir / 'site.json').read_text())
        pd.read_csv(self.dir / 'chk.csv').drop(columns='patient').to_csv(self.dir / 'chk_nopatient.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'patient column'):
            cs.verify(self.args(labels=self.dir / 'chk_nopatient.csv', calibration_labels=None))


class ImageListTests(unittest.TestCase):
    """``score`` writes the image list next to the logits; ``fit`` records it and ``verify`` refuses shared images or another model's logits."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        (self.dir / 'base.json').write_text(json.dumps(BASE))
        (self.dir / 'netrai_bundle.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [
            {'role': 'model', 'path': 'm.onnx', 'sha256': 'a' * 64}, {'role': 'config', 'path': 'c.json', 'sha256': 'b' * 64}]}))
        for name, seed in (('cal', 1), ('chk', 2)):
            r = np.random.default_rng(seed)
            np.save(self.dir / f'{name}.npy', logits_for_scores(np.r_[r.normal(2.4, 0.4, 80), r.normal(0.4, 0.3, 220)].clip(0, 3.9)))
            pd.DataFrame({'grade': [2] * 80 + [0] * 220}).to_csv(self.dir / f'{name}.csv', index=False)
            self.write_list(name, [f'{name}{i:03d}'.ljust(64, '0') for i in range(300)])
        self.policy = cs.fit(SimpleNamespace(logits=self.dir / 'cal.npy', labels=self.dir / 'cal.csv', base_policy=self.dir / 'base.json',
                                             output=self.dir / 'site.json', site='clinic', target_sensitivity=0.95, bundle=None, variant='E_768'))

    def write_list(self, name, digests, model='a' * 64):
        (self.dir / f'{name}.images.json').write_text(json.dumps({'schema_version': 1, 'model_sha256': model, 'config_sha256': 'b' * 64,
                                                                  'images': [f'{d[:6]}.jpg' for d in digests], 'sha256': digests}))

    def args(self, **kw):
        base = dict(policy=self.dir / 'site.json', logits=self.dir / 'chk.npy', labels=self.dir / 'chk.csv', output=self.dir / 'check.json',
                    calibration_labels=None, base_policy=self.dir / 'base.json', bundle=None)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_fit_records_the_calibration_images(self):
        self.assertEqual(len(self.policy['calibration_image_sha256']), 300)
        self.assertEqual(self.policy['calibration_image_sha256'], sorted(self.policy['calibration_image_sha256']))
        self.assertEqual(cs.verify(self.args())['warnings'], [])

    def test_verify_refuses_a_shared_image(self):
        digests = json.loads((self.dir / 'chk.images.json').read_text())['sha256']
        digests[5] = self.policy['calibration_image_sha256'][0]
        self.write_list('chk', digests)
        with self.assertRaisesRegex(ValueError, 'calibration sample'):
            cs.verify(self.args())

    def test_logits_of_another_model_or_wrong_length_are_refused(self):
        self.write_list('chk', [f'x{i:03d}'.ljust(64, '0') for i in range(300)], model='c' * 64)
        with self.assertRaisesRegex(ValueError, 'different model'):
            cs.verify(self.args())
        self.write_list('chk', ['y' * 64])
        with self.assertRaises(ValueError):
            cs.verify(self.args())

    def test_reordered_calibration_sample_is_refused(self):
        # the review reproduction: calibration logits and labels in reverse order, offered as a "separate" sample
        np.save(self.dir / 'rev.npy', np.load(self.dir / 'cal.npy')[::-1])
        pd.read_csv(self.dir / 'cal.csv').iloc[::-1].to_csv(self.dir / 'rev.csv', index=False)
        meta = json.loads((self.dir / 'cal.images.json').read_text())
        self.write_list('rev', meta['sha256'][::-1])
        with self.assertRaisesRegex(ValueError, 'calibration sample'):
            cs.verify(self.args(logits=self.dir / 'rev.npy', labels=self.dir / 'rev.csv'))
        (self.dir / 'rev.images.json').unlink()                                # and without its image list
        with self.assertRaisesRegex(ValueError, 'no image identities'):
            cs.verify(self.args(logits=self.dir / 'rev.npy', labels=self.dir / 'rev.csv'))

    def test_partially_overlapping_sample_is_refused(self):
        digests = json.loads((self.dir / 'chk.images.json').read_text())['sha256']
        digests[100:110] = self.policy['calibration_image_sha256'][:10]
        self.write_list('chk', digests)
        with self.assertRaisesRegex(ValueError, '10 images were in the calibration sample'):
            cs.verify(self.args())

    def test_policy_without_recorded_images_is_refused(self):
        policy = json.loads((self.dir / 'site.json').read_text())
        del policy['calibration_image_sha256']
        (self.dir / 'old.json').write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, 'does not record its calibration images'):
            cs.verify(self.args(policy=self.dir / 'old.json'))

    def test_a_sample_without_image_identities_cannot_be_fitted(self):
        (self.dir / 'cal.images.json').unlink()
        with self.assertRaisesRegex(ValueError, 'no image identities'):
            cs.fit(SimpleNamespace(logits=self.dir / 'cal.npy', labels=self.dir / 'cal.csv', base_policy=self.dir / 'base.json',
                                   output=self.dir / 'x.json', site='clinic', target_sensitivity=0.95, bundle=None, variant='E_768'))

    def test_image_files_can_identify_the_sample_and_must_match_the_scored_rows(self):
        folder = self.dir / 'imgs'
        folder.mkdir()
        for i in range(300):
            (folder / f'c{i}.jpg').write_bytes(b'calibration image %d' % i)
            (folder / f'v{i}.jpg').write_bytes(b'verification image %d' % i)
        for name, prefix in (('cal', 'c'), ('chk', 'v')):
            labels = pd.read_csv(self.dir / f'{name}.csv').assign(image=[f'{prefix}{i}.jpg' for i in range(300)])
            labels.to_csv(self.dir / f'{name}_img.csv', index=False)
            np.save(self.dir / f'{name}_img.npy', np.load(self.dir / f'{name}.npy'))
        policy = cs.fit(SimpleNamespace(logits=self.dir / 'cal_img.npy', labels=self.dir / 'cal_img.csv', base_policy=self.dir / 'base.json',
                                        output=self.dir / 'site_img.json', site='clinic', target_sensitivity=0.95, bundle=None, variant='E_768',
                                        images_root=folder))
        self.assertIn(cs.file_sha256(folder / 'c7.jpg'), policy['calibration_image_sha256'])
        result = cs.verify(self.args(policy=self.dir / 'site_img.json', logits=self.dir / 'chk_img.npy', labels=self.dir / 'chk_img.csv', images_root=folder))
        self.assertEqual(result['referable'], 80)
        mixed = pd.read_csv(self.dir / 'chk_img.csv'); mixed.loc[3, 'image'] = 'c3.jpg'; mixed.to_csv(self.dir / 'chk_mixed.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'calibration sample'):
            cs.verify(self.args(policy=self.dir / 'site_img.json', logits=self.dir / 'chk_img.npy', labels=self.dir / 'chk_mixed.csv', images_root=folder))
        self.write_list('chk_img', fake_digests('v', 300))                    # an image list whose names differ from the labels
        with self.assertRaisesRegex(ValueError, 'row by row'):
            cs.verify(self.args(policy=self.dir / 'site_img.json', logits=self.dir / 'chk_img.npy', labels=self.dir / 'chk_img.csv'))


def fixture_inputs(folder: Path) -> SimpleNamespace:
    """The calibration sample behind matlab/tests/fixtures/site_policy/policy_from_python.json (rows 0-1 are the two fixture images)."""
    np.save(folder / 'cal.npy', logits_for_scores(np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])))
    names = ['cal_1.png', 'cal_2.png'] + [f'synthetic_{i:02d}.png' for i in range(3, 13)]
    digests = [cs.file_sha256(FIXTURE / n) for n in names[:2]] + [hashlib.sha256(n.encode()).hexdigest() for n in names[2:]]
    identity = cs.model_identity(E768 / 'netrai_config.json', None)
    (folder / 'cal.images.json').write_text(json.dumps({'schema_version': 1, **identity, 'images': names, 'sha256': digests}))
    pd.DataFrame({'grade': [0] * 6 + [2] * 6, 'patient': [f'P{i:02d}' for i in range(1, 13)], 'image': names}).to_csv(folder / 'cal.csv', index=False)
    return SimpleNamespace(logits=folder / 'cal.npy', labels=folder / 'cal.csv', base_policy=E768 / 'netrai_config.json', output=folder / 'policy.json',
                           site='cross-language fixture', target_sensitivity=0.95, bundle=None, variant='E_768', images_root=None)


class CrossLanguageTests(unittest.TestCase):
    """Python and MATLAB write and enforce the same identity fields (MATLAB side: matlab/tests/testVerifySitePolicyProvenance.m)."""

    def test_committed_fixture_is_what_fit_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            policy = cs.fit(fixture_inputs(Path(temp)))
        committed = json.loads((FIXTURE / 'policy_from_python.json').read_text())
        for key in ('status', 'model_variant', 'model_sha256', 'config_sha256', 'calibration_image_sha256', 'calibration_patient_sha256'):
            self.assertEqual(policy[key], committed[key], key)
        np.testing.assert_allclose(policy['thresholds'], committed['thresholds'], atol=1e-12)
        self.assertIn(cs.file_sha256(FIXTURE / 'cal_1.png'), committed['calibration_image_sha256'])
        self.assertNotIn(cs.file_sha256(FIXTURE / 'new_1.png'), committed['calibration_image_sha256'])
        self.assertIn(cs.text_sha256('P03'), committed['calibration_patient_sha256'])

    def test_text_hash_is_the_one_matlab_uses(self):
        self.assertEqual(cs.text_sha256('P03'), 'c994ecef8f751a780b28bac70bd0b73948560afbdc833e24df9c06d8b22bcfaf')  # = textSHA256("P03")

    def test_policy_written_by_matlab_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            (temp / 'netrai_bundle.json').write_text(json.dumps({'schema_version': 2, 'artifacts': [
                {'role': 'model', 'path': 'm.onnx', 'sha256': 'a' * 64}, {'role': 'config', 'path': 'c.json', 'sha256': 'b' * 64}]}))
            (temp / 'base.json').write_text(json.dumps(BASE))
            shared = hashlib.sha256(b'image in both samples').hexdigest()
            (temp / 'matlab_policy.json').write_text(json.dumps({     # the fields calibrateSite.m writes (no logits_sha256)
                'status': 'site_calibrated', 'model_variant': 'E_768', 'model_sha256': 'a' * 64, 'config_sha256': 'b' * 64,
                'thresholds': [1.0, 1.5, 2.0, 2.8], 'site': {'name': 'matlab site'}, 'calibration_image_sha256': [shared],
                'calibration_patient_sha256': [cs.text_sha256('P1')]}))
            r = np.random.default_rng(5)
            np.save(temp / 'chk.npy', logits_for_scores(np.r_[r.normal(2.4, 0.4, 50), r.normal(0.4, 0.3, 150)].clip(0, 3.9)))
            pd.DataFrame({'grade': [2] * 50 + [0] * 150, 'patient': [f'Q{i}' for i in range(200)]}).to_csv(temp / 'chk.csv', index=False)
            args = SimpleNamespace(policy=temp / 'matlab_policy.json', logits=temp / 'chk.npy', labels=temp / 'chk.csv', output=temp / 'o.json',
                                   calibration_labels=None, base_policy=temp / 'base.json', bundle=None)
            write_image_list(temp / 'chk.npy', [shared] + fake_digests('q', 199))
            with self.assertRaisesRegex(ValueError, 'calibration sample'):
                cs.verify(args)
            write_image_list(temp / 'chk.npy', fake_digests('q', 200))
            self.assertTrue(cs.verify(args)['meets_targets_point'])
            labels = pd.read_csv(temp / 'chk.csv'); labels.loc[9, 'patient'] = 'P1'; labels.to_csv(temp / 'chk.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'patients'):
                cs.verify(args)


@unittest.skipUnless(__import__('modules.variant_e768', fromlist=['weights_present']).weights_present(), 'E_768 weights not installed')
class ScoreTests(unittest.TestCase):
    def test_score_matches_the_screening_runtime_and_lists_the_image(self):
        import cv2
        from modules.variant_e768 import E768Runtime
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            pd.DataFrame({'image': ['patient_sample.jpg']}).to_csv(temp / 'images.csv', index=False)
            cs.score(SimpleNamespace(images=temp / 'images.csv', root=cs.Path(__file__).resolve().parents[2] / 'data', out=temp / 'logits.npy'))
            logits = np.load(temp / 'logits.npy')
            meta = json.loads((temp / 'logits.images.json').read_text())
        image = cv2.cvtColor(cv2.imread(str(Path(__file__).resolve().parents[2] / 'data/patient_sample.jpg')), cv2.COLOR_BGR2RGB)
        runtime = E768Runtime()
        np.testing.assert_allclose(logits, runtime.predict([image])['logits'], atol=1e-9)
        self.assertEqual((meta['images'], meta['model_sha256']), (['patient_sample.jpg'], runtime.identity['model_sha256']))
        self.assertEqual(meta['sha256'], [cs.file_sha256(Path(__file__).resolve().parents[2] / 'data/patient_sample.jpg')])


if __name__ == '__main__':
    unittest.main()
