"""Guard against inputs that are not fundus photographs (X15), checked after image quality and before grading.

Score: Mahalanobis distance of the image's 576-D embedding (the frozen ImageNet MobileNetV3-Small encoder inside the quality
candidate, full-frame 224-pixel input) to fundus images of the datasets that trained the published weights. Above the threshold the
image is not graded (``action`` = reject) or carries a warning (``action`` = flag); the parameters file says which, as decided by
the X15 evaluation. MATLAB computes the same score (src/m1_quality/checkFundusPhoto.m) from a byte-identical parameters file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / 'weights'
PARAMS = WEIGHTS / 'fundus_guard.json'
CHECKPOINT = WEIGHTS / 'quality_model_candidate.pth'
FEEDBACK = 'This image does not look like a fundus (retina) photograph; capture a fundus image of the eye.'


class GuardError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FundusGuard:
    def __init__(self, params: Path = PARAMS, checkpoint: Path = CHECKPOINT):
        import torch
        from models.quality_model import QualityModel
        meta = json.loads(Path(params).read_text())
        blob = (Path(params).parent / meta['parameters_file']).read_bytes()
        if _sha256(blob) != meta['parameters_sha256']:
            raise GuardError(f"{meta['parameters_file']} does not match the SHA-256 in {Path(params).name}")
        weights = Path(checkpoint).read_bytes()
        if _sha256(weights) != meta['encoder_checkpoint_sha256']:
            raise GuardError(f'{Path(checkpoint).name} is not the encoder the guard was fitted with')
        if meta.get('action') not in ('reject', 'flag'):
            raise GuardError('The guard parameters do not name an action (reject or flag)')
        dimension = int(meta['dimension'])
        values = np.frombuffer(blob, '<f4').astype(np.float64)
        if values.size != dimension + dimension * dimension:
            raise GuardError('Unexpected guard parameter size')
        self.mean, self.precision = values[:dimension], values[dimension:].reshape(dimension, dimension)
        self.threshold, self.action = float(meta['threshold']), meta['action']
        model = QualityModel(pretrained=False)
        model.load_state_dict(torch.load(Path(checkpoint), map_location='cpu', weights_only=True))
        self.model = model.eval()

    def embedding(self, image_rgb: np.ndarray) -> np.ndarray:
        import torch
        from models.quality_model import quality_input
        with torch.no_grad():
            return self.model.encode(torch.from_numpy(quality_input(image_rgb)[None]))[0].numpy().astype(np.float64)

    def score(self, image_rgb: np.ndarray) -> float:
        centred = self.embedding(image_rgb) - self.mean
        return float(np.sqrt(max(centred @ self.precision @ centred, 0.0)))

    def check(self, image_rgb: np.ndarray) -> dict:
        score = self.score(image_rgb)
        return {'is_fundus': bool(score <= self.threshold), 'score': score, 'threshold': self.threshold, 'action': self.action}


_DEFAULT = None


def check_fundus(image_rgb: np.ndarray) -> dict:
    """``check`` of the shipped guard (loaded once)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = FundusGuard()
    return _DEFAULT.check(image_rgb)
