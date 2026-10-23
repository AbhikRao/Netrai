"""Opt-in learned lesion evidence (EX, HE, MA, SE) for reports: probabilities -> masks -> counts, overlay and panel.

Research module. Weights come from the X8 lesion segmenter (not shipped by default); thresholds are the validation-chosen
Dice thresholds (``thresholds.json``). On the locked DDR test (202 images) the pixel Dice was EX 0.60, HE 0.43, MA 0.26:
overlays are candidate evidence for review, not a diagnosis and not lesion ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

LESIONS = ('EX', 'HE', 'MA', 'SE')
COLORS = {'EX': (255, 205, 0), 'HE': (230, 30, 60), 'MA': (0, 190, 255), 'SE': (255, 255, 255)}
LONG_SIDE, TILE, OVERLAP = 1536, 1024, 128


def load_thresholds(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    thresholds = dict(cfg['dice_thresholds'])
    thresholds['MA'] = float(cfg.get('ma_lesion_threshold', thresholds['MA']))  # the lesion-level operating point for MA counts
    return thresholds


def summarise_masks(probs: np.ndarray, thresholds: dict, valid: np.ndarray, min_pixels: int = 1) -> dict:
    """probs (H, W, 4) float in [0, 1]; valid (H, W) field mask. Returns masks, area fractions and component counts."""
    masks, area, count = {}, {}, {}
    field = max(int(valid.sum()), 1)
    for index, name in enumerate(LESIONS):
        mask = (probs[..., index] >= thresholds[name]) & valid
        masks[name] = mask
        area[name] = float(mask.sum() / field)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        count[name] = int(sum(stats[i, cv2.CC_STAT_AREA] >= min_pixels for i in range(1, n)))
    return {'masks': masks, 'area_fraction': area, 'count': count}


def overlay(image_rgb: np.ndarray, masks: dict, alpha: float = 0.55) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    for name in LESIONS:
        m = masks[name]
        if m.any():
            color = np.array(COLORS[name], np.float32)
            out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def segment(image_rgb: np.ndarray, weights: Path, thresholds_path: Path, device=None) -> dict:
    """Run the learned segmenter on a full RGB image (crop to the field, long side 1536, windows of 1024)."""
    import torch
    from lesion_data import FOREGROUND, foreground_box
    from locked_segmentation_eval import load_lesion_model, native_probabilities
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    top, bottom, left, right = foreground_box(image_rgb)
    crop = np.ascontiguousarray(image_rgb[top:bottom, left:right])
    model = load_lesion_model(Path(weights), device)
    probs = native_probabilities(model, crop, device, LONG_SIDE, TILE, OVERLAP, 'ben')
    valid = crop.max(axis=2) > FOREGROUND
    thresholds = load_thresholds(thresholds_path)
    summary = summarise_masks(probs, thresholds, valid)
    summary.update(image=crop, overlay=overlay(crop, summary['masks']), thresholds=thresholds)
    return summary


def save_panel(summary: dict, path: Path, title: str = 'Learned lesion candidates (research)') -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor='white', gridspec_kw={'width_ratios': [1.6, 1]})
    axes[0].imshow(summary['overlay']); axes[0].axis('off'); axes[0].set_title(title, fontsize=13)
    axes[1].axis('off')
    lines = ['Candidate lesions at the validation-chosen thresholds', '']
    for name in LESIONS:
        r, g, b = COLORS[name]
        lines.append(f"{name}:  {summary['count'][name]:4d} components   {100 * summary['area_fraction'][name]:5.2f}% of the field")
    lines += ['', 'Colour key:  yellow EX   red HE   cyan MA   white SE', '',
              'Locked DDR test (202 images), pixel Dice:', '  EX 0.60   HE 0.43   MA 0.26 (weak)', '',
              'Candidate evidence for clinician review;', 'not a diagnosis and not lesion ground truth.']
    axes[1].text(0.0, 0.95, '\n'.join(lines), va='top', ha='left', fontsize=11, family='monospace')
    fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
