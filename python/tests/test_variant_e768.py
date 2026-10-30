"""Tests for the Python E_768 variant: site-policy binding, variant resolution and (when the local weights exist) a real-image parity check."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2

import inference
from modules import variant_e768 as v

ROOT = v.ROOT
IDENTITY = {'model_sha256': 'a' * 64, 'config_sha256': 'b' * 64}
CONFIG = {'grade_thresholds': [1.0, 1.2, 2.0, 2.8], 'referable_head_threshold': 0.2}


def policy(**kw):
    base = {'status': 'site_calibrated', 'thresholds': [1.0, 1.5, 2.0, 2.8], 'model_sha256': 'a' * 64, 'config_sha256': 'b' * 64,
            'head_threshold': 0.3, 'site': {'name': 'clinic A', 'referable': 55}}
    base.update(kw)
    return base


class SitePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)

    def write(self, data):
        path = self.dir / 'p.json'
        path.write_text(json.dumps(data))
        return path

    def test_no_policy_keeps_frozen_thresholds_and_flags_uncalibrated(self):
        config, status = v.apply_site_policy(CONFIG, IDENTITY, None)
        self.assertEqual(config['grade_thresholds'], [1.0, 1.2, 2.0, 2.8])
        self.assertTrue(status['calibration_status'].startswith('UNCALIBRATED'))

    def test_bound_policy_replaces_thresholds_and_records_provenance(self):
        config, status = v.apply_site_policy(CONFIG, IDENTITY, self.write(policy()))
        self.assertEqual(config['grade_thresholds'], [1.0, 1.5, 2.0, 2.8])
        self.assertEqual(config['referable_head_threshold'], 0.3)
        self.assertEqual(status['site_name'], 'clinic A')
        self.assertEqual(len(status['site_policy_sha256']), 64)
        self.assertEqual(CONFIG['grade_thresholds'], [1.0, 1.2, 2.0, 2.8])  # input not mutated

    def test_refuses_other_model_missing_identity_and_malformed_policies(self):
        bad = [policy(model_sha256='c' * 64), policy(config_sha256='d' * 64), policy(thresholds=[1, 2, 3]), policy(thresholds=[1, 3, 2, 4]), policy(thresholds=[1.0, 1.5, 1.5, 2.8]),
               policy(status='frozen'), policy(head_threshold=1.5)]
        for item in bad:
            with self.assertRaises(v.VariantError, msg=str(item)):
                v.apply_site_policy(CONFIG, IDENTITY, self.write(item))
        missing = policy()
        del missing['model_sha256']
        with self.assertRaises(v.VariantError):
            v.apply_site_policy(CONFIG, IDENTITY, self.write(missing))
        with self.assertRaises(v.VariantError):
            v.apply_site_policy(CONFIG, IDENTITY, self.dir / 'nope.json')


class ResolveVariantTests(unittest.TestCase):
    def test_legacy_never_takes_a_site_policy(self):
        self.assertEqual(inference.resolve_variant('legacy'), ('legacy', None))
        with self.assertRaises(ValueError):
            inference.resolve_variant('legacy', 'p.json')

    def test_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            inference.resolve_variant('E_999')

    @unittest.skipIf(v.weights_present(), 'E_768 weights are installed locally')
    def test_auto_falls_back_to_legacy_without_weights_and_explicit_e768_fails_clearly(self):
        self.assertEqual(inference.resolve_variant('auto'), ('legacy', None))
        with self.assertRaises(ValueError):
            inference.resolve_variant('auto', 'p.json')
        with self.assertRaises(v.VariantError):
            inference.resolve_variant('E_768')


class MemberManifestTests(unittest.TestCase):
    def test_tampered_member_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'torch').mkdir()
            (d / 'torch' / 'm.pt').write_bytes(b'not the released weights')
            (d / 'torch_members.json').write_text(json.dumps({'members': {'s17': {'file': 'm.pt', 'sha256': '0' * 64}}}))
            self.assertTrue(v.members_present(d))
            with self.assertRaises(v.VariantError):
                v.load_members(d)
            self.assertFalse(v.members_present(d / 'missing'))


@unittest.skipUnless(v.weights_present() and v.members_present(), 'E_768 ONNX and PyTorch members are not installed')
class MemberParityTests(unittest.TestCase):
    def test_pytorch_members_reproduce_the_onnx_graph_and_give_a_map(self):
        import numpy as np
        import torch
        from utils.preprocessing import preprocess_for_model
        image = cv2.cvtColor(cv2.imread(str(ROOT / 'data/patient_sample.jpg')), cv2.COLOR_BGR2RGB)
        runtime = v.E768Runtime()
        onnx = runtime.predict([image])['logits'][0]
        ensemble = v.load_members()
        with torch.no_grad():
            g, r = ensemble(torch.from_numpy(preprocess_for_model(image, size=768).transpose(2, 0, 1)[None]).float())
        self.assertLess(float(np.abs(np.r_[g[0].numpy(), r.numpy()] - onnx).max()), 1e-4)
        cam = runtime.gradcam(image)
        self.assertEqual(cam.shape, (768, 768))
        self.assertTrue(np.isfinite(cam).all())


@unittest.skipUnless(v.weights_present(), 'E_768 weights are not installed')
class RuntimeParityTests(unittest.TestCase):
    def test_real_image_matches_the_matlab_result_and_site_policy_changes_the_decision(self):
        image = cv2.cvtColor(cv2.imread(str(ROOT / 'data/patient_sample.jpg')), cv2.COLOR_BGR2RGB)
        runtime = v.E768Runtime()
        out = runtime.predict([image])
        self.assertEqual(int(out['grades'][0]), 2)                      # MATLAB main_pipeline: grade 2, score 1.100
        self.assertAlmostEqual(float(out['continuous_scores'][0]), 1.100, delta=0.01)
        self.assertEqual(out['logits'].shape, (1, 6))
        self.assertTrue(runtime.status['calibration_status'].startswith('UNCALIBRATED'))


if __name__ == '__main__':
    unittest.main()
