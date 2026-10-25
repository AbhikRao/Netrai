"""E_768 research variant for the Python pipeline: ONNX runtime inference, site policy and calibration status.

Mirrors the MATLAB path (``main_pipeline`` Variant/SitePolicy): the two-member E_768 ONNX graph (mean of two 768 px models, six outputs: five grade
logits + one referral logit) is verified against the SHA-256 in ``matlab/variant_R3_E_768/netrai_bundle.json`` before use. The referral cut-off is
the frozen round-3 policy until a per-site policy (``calibrate_site.py``) is loaded; a site policy is bound to the model and config it was fitted for.
No Grad-CAM is available for the ONNX graph (no gradients): the report says so instead of substituting a map.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from utils.preprocessing import preprocess_for_model
from utils.safety import assign_triage

ROOT = Path(__file__).resolve().parents[2]
VARIANT_DIR = ROOT / 'matlab/variant_R3_E_768'
UNCALIBRATED = 'UNCALIBRATED: frozen thresholds; run python/calibrate_site.py on a labelled sample of this site'
SITE_CALIBRATED = 'site-calibrated (not independently validated)'


class VariantError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def weights_present(variant_dir: Path = VARIANT_DIR) -> bool:
    return (variant_dir / 'netrai_fold0.onnx').is_file() and (variant_dir / 'netrai_config.json').is_file()


def load_identity(variant_dir: Path = VARIANT_DIR) -> dict:
    """Verified model/config identity from the committed bundle (the ONNX hash is checked against the file)."""
    bundle = json.loads((variant_dir / 'netrai_bundle.json').read_text())
    roles = {a['role']: a['sha256'] for a in bundle['artifacts']}
    model, config = variant_dir / 'netrai_fold0.onnx', variant_dir / 'netrai_config.json'
    if _sha256(model) != roles['model']:
        raise VariantError('The E_768 ONNX file does not match the SHA-256 in netrai_bundle.json; refusing to use it.')
    if _sha256(config) != roles['config']:
        raise VariantError('The E_768 config does not match the SHA-256 in netrai_bundle.json.')
    return {'model_sha256': roles['model'], 'config_sha256': roles['config']}


def apply_site_policy(config: dict, identity: dict, policy_path: str | Path | None) -> tuple[dict, dict]:
    """Return (config with the site boundaries, status). Policies of another model/config or without identity are refused."""
    status = {'calibration_status': UNCALIBRATED, 'site_policy_path': '', 'site_policy_sha256': '', 'site_name': '', 'site_referable_cases': None}
    config = dict(config)
    if not policy_path:
        return config, status
    path = Path(policy_path)
    if not path.is_file():
        raise VariantError(f'Site policy not found: {path}')
    policy = json.loads(path.read_text())
    thresholds = policy.get('thresholds')
    if (not isinstance(thresholds, list) or len(thresholds) != 4 or not np.isfinite(thresholds).all()
            or any(b < a for a, b in zip(thresholds, thresholds[1:]))):
        raise VariantError('A site policy needs four finite, non-decreasing ordinal thresholds.')
    if policy.get('status') != 'site_calibrated':
        raise VariantError('The policy file is not a site-calibrated policy (status != site_calibrated).')
    if policy.get('model_sha256') is None or policy.get('config_sha256') is None:
        raise VariantError('The site policy does not name the model it was fitted for; re-create it with calibrate_site.py fit.')
    if policy['model_sha256'].lower() != identity['model_sha256'].lower() or policy['config_sha256'].lower() != identity['config_sha256'].lower():
        raise VariantError('The site policy was fitted for a different model than the one loaded.')
    config['grade_thresholds'] = [float(t) for t in thresholds]
    if 'head_threshold' in policy:
        head = float(policy['head_threshold'])
        if not 0 < head < 1:
            raise VariantError('head_threshold must lie in (0, 1).')
        config['referable_head_threshold'] = head
    site = policy.get('site', {})
    status.update(calibration_status=SITE_CALIBRATED, site_policy_path=str(path), site_policy_sha256=_sha256(path),
                  site_name=site.get('name', ''), site_referable_cases=site.get('referable'))
    return config, status


class E768Runtime:
    def __init__(self, variant_dir: Path = VARIANT_DIR, site_policy: str | Path | None = None):
        import onnxruntime as ort
        if not weights_present(variant_dir):
            raise VariantError('E_768 weights are not installed (matlab/variant_R3_E_768/netrai_fold0.onnx); see scripts/fetch_e768.sh or use --variant legacy.')
        self.identity = load_identity(variant_dir)
        config = json.loads((variant_dir / 'netrai_config.json').read_text())
        self.config, self.status = apply_site_policy(config, self.identity, site_policy)
        self.size = int(config['input_size'])
        self.temperature = float(config['confidence_temperature'])
        self.session = ort.InferenceSession(str(variant_dir / 'netrai_fold0.onnx'), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, images_rgb) -> dict:
        thresholds = np.asarray(self.config['grade_thresholds'], float)
        head_threshold = float(self.config['referable_head_threshold'])
        logits = []
        for image in images_rgb:
            x = preprocess_for_model(image, size=self.size).transpose(2, 0, 1)[None].astype(np.float32)
            logits.append(self.session.run(None, {self.input_name: x})[0][0])
        logits = np.asarray(logits, float)
        if not np.isfinite(logits).all():
            raise VariantError('Nonfinite model outputs')
        z = logits[:, :5] - logits[:, :5].max(1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        score = probs @ np.arange(5)
        grades = (score[:, None] >= thresholds[None]).sum(1).astype(int)
        referable_p = 1 / (1 + np.exp(-logits[:, 5]))
        grade_ref, head_pos = grades >= 2, referable_p >= head_threshold
        return {
            'grades': grades, 'grade_probabilities': probs, 'continuous_scores': score,
            'grade_head_referable_probabilities': probs[:, 2:].sum(1), 'referable_head_probabilities': referable_p,
            'grade_head_referable_predictions': grade_ref, 'referable_head_predictions': head_pos,
            'head_disagreements': grade_ref != head_pos,
            'triage_actions': np.array([assign_triage(quality_rejected=False, grade_referable=bool(g), referable_head_positive=bool(h))
                                        for g, h in zip(grade_ref, head_pos)], dtype=object),
            'logits': logits,
        }
