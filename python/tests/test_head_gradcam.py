"""Head-only Grad-CAM (modules.explainability.head_gradcam) gives pytorch_grad_cam's GradCAM map at backbone.bn2, for one model and
for a mean ensemble, and the legacy pipeline's map equals the MATLAB pipeline's map (committed fixture)."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

PYTHON_DIR = Path(__file__).resolve().parents[1]
ROOT = PYTHON_DIR.parent
sys.path.insert(0, str(PYTHON_DIR))

from models.netrai_model import NetrAIModel  # noqa: E402
from modules.explainability import generate_gradcam, gradcam_target, head_gradcam  # noqa: E402
from modules.variant_e768 import MeanEnsemble  # noqa: E402


class BothHeads(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, inputs):
        logits, referral = self.network(inputs)
        return torch.cat([logits, referral[:, None]], dim=1)


def reference_map(model, x, target):
    from pytorch_grad_cam import GradCAM
    layers = [member.backbone.bn2 for member in getattr(model, 'members', [model])]
    with GradCAM(model=BothHeads(model).eval(), target_layers=layers) as cam:
        return cam(input_tensor=x, targets=[target])[0]


class HeadGradCAMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(5)
        cls.members = [NetrAIModel('efficientnet_b0', 5, pretrained=False).eval() for _ in range(2)]
        cls.x = torch.rand(1, 3, 128, 128)

    def test_single_model_equals_pytorch_grad_cam(self):
        for target in ('referral', 'ordinal', 3):
            cam = head_gradcam(self.members[0], self.x, gradcam_target(target))
            self.assertEqual((cam.shape, cam.dtype), ((128, 128), np.float32))
            self.assertGreater(float(cam.max()), 0.5)                       # a non-degenerate map
            np.testing.assert_allclose(cam, reference_map(self.members[0], self.x, gradcam_target(target)), atol=1e-6)

    def test_mean_ensemble_equals_pytorch_grad_cam(self):
        ensemble = MeanEnsemble(self.members).eval()
        cam = head_gradcam(ensemble, self.x, gradcam_target('referral'))
        np.testing.assert_allclose(cam, reference_map(ensemble, self.x, gradcam_target('referral')), atol=1e-6)
        single = [head_gradcam(m, self.x, gradcam_target('referral')) for m in self.members]
        self.assertLess(float(np.corrcoef(single[0].ravel(), single[1].ravel())[0, 1]), 0.99)   # members explain differently

    def test_model_is_left_without_gradients(self):
        model = NetrAIModel('efficientnet_b0', 5, pretrained=False).eval()
        head_gradcam(model, self.x, gradcam_target('referral'))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_unknown_target_gives_no_map(self):
        image = np.full((64, 64, 3), 120, np.uint8)
        with self.assertWarns(UserWarning):
            self.assertIsNone(generate_gradcam(self.members[0], image, target_class='lesion'))


class LegacyCrossRuntimeTests(unittest.TestCase):
    def test_legacy_map_equals_the_matlab_map(self):
        import inference
        model, _, _ = inference._get_model()
        image = cv2.cvtColor(cv2.imread(str(ROOT / 'data/patient_sample.jpg')), cv2.COLOR_BGR2RGB)
        cam = generate_gradcam(model, image, target_class='referral')
        self.assertEqual(cam.shape, (512, 512))
        # 8 x 8 block means of the MATLAB pipeline's map (generateGradCAM -> referralGradCAM on the legacy ONNX graph)
        matlab = np.fromfile(ROOT / 'matlab/tests/fixtures/legacy_gradcam_patient_sample_64.f32', dtype='<f4').reshape(64, 64)
        python = cam.reshape(64, 8, 64, 8).mean(axis=(1, 3))
        self.assertGreater(float(np.corrcoef(matlab.ravel(), python.ravel())[0, 1]), 0.999)
        self.assertLess(float(np.abs(matlab - python).max()), 0.01)


if __name__ == '__main__':
    unittest.main()
