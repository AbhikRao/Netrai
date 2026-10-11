"""Opt-in IDRiD-trained binary referral probe; never changes default triage."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from utils.preprocessing import preprocess_for_model


@lru_cache(maxsize=2)
def _load_bound_probe(path, expected_sha256):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError('IDRiD referral probe differs from its evaluation record')
    return joblib.load(path)


def predict_idrid_referral_probe(image_rgb, model, evidence_dir=None):
    """Return research probability and decision without changing M3 grade/triage."""
    if evidence_dir is None:
        evidence_dir = (Path(__file__).resolve().parents[2] / 'results' /
                        'verification_2026-09-24' / 'idrid_feature_probe')
    folder = Path(evidence_dir)
    evidence = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if evidence['status'] != 'retrospective_target_domain_probe_not_deployed':
        raise ValueError('Expected a research-only IDRiD referral probe')
    if getattr(model, '_netrai_checkpoint_sha256', None) != evidence['checkpoint_sha256']:
        raise ValueError('Referral probe requires its evaluated backbone/checkpoint')
    probe = _load_bound_probe(str(folder/'referral_probe.joblib'),
                              evidence['probe_sha256'])
    array = preprocess_for_model(image_rgb, size=512)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)[None]).float().to(next(model.parameters()).device)
    with torch.inference_mode():
        backbone = model.backbone(tensor)
        if evidence['selection']['representation'] == 'backbone':
            features = backbone
        elif evidence['selection']['representation'] == 'head_256':
            features = model.head(backbone)
        else:
            raise ValueError('Unsupported IDRiD probe representation')
    # Historical probe fitting quantized cached features to float16 first.
    encoded = features.cpu().numpy().astype(np.float16).astype(np.float32)
    probability = float(probe.predict_proba(encoded)[:, 1][0])
    threshold = float(evidence['selection']['oof_threshold'])
    return {'probability': probability, 'positive': probability >= threshold,
            'threshold': threshold,
            'status': 'research_only_no_triage_effect'}
