"""Higher-resolution, scored microaneurysm candidates for research evaluation.

This module does not use annotation masks at inference. Training and operating
point selection are handled separately on IDRiD's training split.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import joblib
import numpy as np


FEATURE_NAMES = (
    'blackhat_11', 'blackhat_19', 'blackhat_29',
    'spot_2_8', 'spot_3_15', 'spot_4_24',
    'hessian_blob_2', 'hessian_ratio_2',
    'hessian_blob_4', 'hessian_ratio_4',
    'local_std', 'green', 'red', 'blue', 'green_over_red',
)


def _hessian_features(inverted, sigma):
    blurred = cv2.GaussianBlur(inverted, (0, 0), sigma).astype(np.float32)
    xx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3)
    trace = xx + yy
    separation = np.sqrt(np.maximum((xx - yy)**2 + 4*xy**2, 0))
    large = (trace + separation) / 2
    small = (trace - separation) / 2
    # A dark round spot in the green channel is a concave maximum after inversion.
    blob = np.sqrt(np.maximum(large * small, 0)) * (large < 0) * (small < 0)
    ratio = np.abs(small) / (np.abs(large) + 1e-3)
    return blob, np.clip(ratio, 0, 20)


def extract_candidates(bgr, longest_edge=2048, floor=28, max_candidates=16000):
    """Return source-pixel coordinates, feature rows, and scaled-grid positions."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError('Expected a BGR image')
    source_h, source_w = bgr.shape[:2]
    scale = longest_edge / max(source_h, source_w)
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    image = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    blue, green, red = cv2.split(image)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(green)
    inverted = cv2.bitwise_not(enhanced)
    blackhats = []
    for size in (11, 19, 29):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        blackhats.append(cv2.morphologyEx(inverted, cv2.MORPH_TOPHAT, kernel))
    response = np.maximum.reduce(blackhats)
    fov = (cv2.GaussianBlur(red, (0, 0), 8) > 20).astype(np.uint8)
    fov = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    maxima = (response == cv2.dilate(response, np.ones((7, 7), np.uint8)))
    maxima &= (response >= floor) & (fov > 0)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        maxima.astype(np.uint8), connectivity=8)
    if count <= 1:
        return (np.empty((0, 2), np.float32),
                np.empty((0, len(FEATURE_NAMES)), np.float32),
                np.empty((0, 2), np.int32), (target_w, target_h))
    grid = np.rint(centroids[1:]).astype(np.int32)
    grid[:, 0] = np.clip(grid[:, 0], 0, target_w-1)
    grid[:, 1] = np.clip(grid[:, 1], 0, target_h-1)
    xs, ys = grid[:, 0], grid[:, 1]
    if len(grid) > max_candidates:
        keep = np.argsort(-response[ys, xs], kind='stable')[:max_candidates]
        grid, xs, ys = grid[keep], xs[keep], ys[keep]
    inv_float = inverted.astype(np.float32)
    spot = []
    for near, far in ((2, 8), (3, 15), (4, 24)):
        low = cv2.GaussianBlur(inv_float, (0, 0), near)
        high = cv2.GaussianBlur(inv_float, (0, 0), far)
        spot.append(low[ys, xs] - high[ys, xs])
    hessian_2 = _hessian_features(inverted, 2)
    hessian_4 = _hessian_features(inverted, 4)
    mean = cv2.blur(inv_float, (15, 15))
    variance = np.maximum(cv2.blur(inv_float**2, (15, 15)) - mean**2, 0)
    features = np.column_stack([
        *(layer[ys, xs] for layer in blackhats),
        *spot,
        hessian_2[0][ys, xs], hessian_2[1][ys, xs],
        hessian_4[0][ys, xs], hessian_4[1][ys, xs],
        np.sqrt(variance[ys, xs]), green[ys, xs], red[ys, xs], blue[ys, xs],
        (green[ys, xs].astype(float)+1) / (red[ys, xs].astype(float)+1),
    ]).astype(np.float32)
    native = np.column_stack(((xs + 0.5)*source_w/target_w - 0.5,
                              (ys + 0.5)*source_h/target_h - 0.5)).astype(np.float32)
    return native, features, grid, (target_w, target_h)


def score_and_suppress(native, features, model, minimum_spacing=14, max_output=3000):
    """Rank scored candidates and suppress near-duplicate peaks on the source grid."""
    if len(native) == 0:
        return native, np.empty(0, np.float32)
    probabilities = model.predict_proba(features)[:, 1]
    order = np.argsort(-probabilities, kind='stable')
    occupied = {}
    chosen = []
    # Sparse hash cells avoid allocating a full native-resolution occupancy map.
    cell = minimum_spacing
    for index in order:
        x, y = native[index]
        cx, cy = int(x // cell), int(y // cell)
        if any((x-native[prior, 0])**2 + (y-native[prior, 1])**2 < cell**2
               for gx in range(cx-1, cx+2) for gy in range(cy-1, cy+2)
               for prior in occupied.get((gx, gy), ())):
            continue
        chosen.append(index)
        occupied.setdefault((cx, cy), []).append(index)
        if len(chosen) >= max_output:
            break
    selected = np.asarray(chosen, dtype=int)
    return native[selected], probabilities[selected].astype(np.float32)


@lru_cache(maxsize=2)
def _load_ranker(path, expected_sha256):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError('MA candidate model hash differs from its evaluation record')
    return joblib.load(path)


def detect_ranked_microaneurysms(bgr, output_size=512, evidence_dir=None):
    """Run the IDRiD-trained research candidate and map points to M2 space.

    The returned mask marks ranked candidate locations, not lesion boundaries.
    This is not calibrated clinical evidence and is never used to alter M3 grade.
    """
    if evidence_dir is None:
        evidence_dir = (Path(__file__).resolve().parents[2] / 'results' /
                        'verification_2026-09-24' / 'ma_native_candidate')
    folder = Path(evidence_dir)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    if tuple(summary['feature_names']) != FEATURE_NAMES:
        raise ValueError('MA candidate feature contract differs from the model record')
    settings = summary['settings']
    model = _load_ranker(str(folder / 'ma_ranker.joblib'), summary['model_sha256'])
    native, features, _, _ = extract_candidates(
        bgr, longest_edge=settings['longest_edge'], floor=settings['floor'],
        max_candidates=settings['max_candidates'])
    coordinates, scores = score_and_suppress(native, features, model)
    threshold = float(summary['selected_on_development']['threshold'])
    coordinates = coordinates[scores >= threshold]
    height, width = bgr.shape[:2]
    scale = output_size / max(height, width)
    target_w, target_h = int(width*scale), int(height*scale)
    left, top = (output_size-target_w)//2, (output_size-target_h)//2
    mapped = [((x+0.5)*target_w/width-0.5+left,
               (y+0.5)*target_h/height-0.5+top) for x, y in coordinates]
    mask = np.zeros((output_size, output_size), np.uint8)
    for x, y in mapped:
        cv2.circle(mask, (int(round(x)), int(round(y))), 2, 255, -1)
    return mask, len(mapped), mapped
