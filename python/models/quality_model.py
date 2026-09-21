"""Small image-level EyeQ classifier and its explicit cross-runtime input contract."""

import numpy as np
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


INPUT_SIZE = 224
PREPROCESS_VERSION = 'rgb_half_pixel_bilinear_float32_224_v1'
CLASS_NAMES = ['good', 'usable', 'reject']


def quality_input(image):
    """Resize the complete RGB frame with explicit half-pixel bilinear sampling.

    No crop, enhancement, or automatic exposure correction: quality defects
    must remain visible. No antialiasing is applied. MATLAB uses these same
    coordinates and float32 arithmetic instead of its imresize defaults.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Quality model expects a uint8 RGB image')
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError('Quality model expects a nonempty HxWx3 RGB image')
    height, width = image.shape[:2]
    yy = np.clip((np.arange(INPUT_SIZE, dtype=np.float64) + 0.5)
                 * height / INPUT_SIZE - 0.5, 0, height - 1)
    xx = np.clip((np.arange(INPUT_SIZE, dtype=np.float64) + 0.5)
                 * width / INPUT_SIZE - 0.5, 0, width - 1)
    y0, x0 = np.floor(yy).astype(int), np.floor(xx).astype(int)
    y1, x1 = np.minimum(y0 + 1, height - 1), np.minimum(x0 + 1, width - 1)
    wy = (yy - y0).astype(np.float32)[:, None, None]
    wx = (xx - x0).astype(np.float32)[None, :, None]
    top = (image[y0[:, None], x0].astype(np.float32) * (1 - wx)
           + image[y0[:, None], x1].astype(np.float32) * wx)
    bottom = (image[y1[:, None], x0].astype(np.float32) * (1 - wx)
              + image[y1[:, None], x1].astype(np.float32) * wx)
    resized = (top * (1 - wy) + bottom * wy) / np.float32(255)
    normalized = (resized - np.array([.485, .456, .406], np.float32))
    normalized /= np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1))


class QualityModel(torch.nn.Module):
    """Frozen ImageNet MobileNetV3-Small encoder plus learned three-class head."""

    def __init__(self, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(576, 3)

    def encode(self, image):
        return self.pool(self.features(image)).flatten(1)

    def forward(self, image):
        return self.head(self.encode(image))


def quality_decisions(probabilities, reject_threshold):
    """Reject at a validation-selected threshold; otherwise choose Good/Usable."""
    probabilities = np.asarray(probabilities, dtype=float)
    if (probabilities.ndim != 2 or probabilities.shape[1] != 3
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            or not 0 < reject_threshold < 1):
        raise ValueError('Expected normalized Nx3 probabilities and threshold in (0,1)')
    predictions = probabilities[:, :2].argmax(axis=1)
    predictions[probabilities[:, 2] >= reject_threshold] = 2
    return predictions
