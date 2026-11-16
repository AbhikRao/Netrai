#!/usr/bin/env python3
"""Label-free camera dust correction: find specks that sit at the same pixel position in many images from one camera, and inpaint them.

Dust on a fundus camera's sensor or optics leaves small dark specks at fixed pixel positions in every photograph that camera takes. After the
model's local-contrast normalisation (Ben Graham) such a speck looks like a microaneurysm, the defining lesion of mild DR. Diagnosis (plan v6,
tier TX): BRSET Canon images carry ~150 persistent dark specks (Nikon: ~6) and E_768 scores healthy Canon eyes like mild DR.

Estimation (unlabelled images of one camera, grouped by raw image size; parameters fixed on the BRSET Canon design site):
  per image: green channel at half resolution, high-pass residual = green - GaussianBlur(green, sigma 6);
  per pixel: median residual over the images and its standard error; a pixel is a speck when median < -3 and median / SE < -8;
  the mask is dilated by 1 pixel (half resolution) and stored with the image size it belongs to.
Application: the mask is upsampled to full resolution and inpainted (OpenCV Telea, radius 3) before the usual preprocessing. Images whose size has
no mask are left unchanged. Anatomy differs between patients, so it averages out of the median; only static camera artefacts persist.

``estimate`` writes ``<out>/dust_<H>x<W>.png`` plus ``dust_masks.json`` (speck counts per size: also a lens-cleaning signal for operators).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

MEDIAN_THRESHOLD, Z_THRESHOLD, SIGMA, DILATE, INPAINT_RADIUS, MIN_IMAGES = -3.0, -8.0, 6.0, 1, 3, 64


def residual(path: Path) -> np.ndarray:
    green = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_2)[..., 1].astype(np.float32)
    return green - cv2.GaussianBlur(green, (0, 0), SIGMA)


def full_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return int(image.shape[0]), int(image.shape[1])


def estimate(paths: list[Path], min_images: int = MIN_IMAGES) -> dict:
    """{(H, W): half-resolution boolean speck mask} for every image size with at least ``min_images`` images."""
    groups = collections.defaultdict(list)
    for path in paths:
        groups[full_size(path)].append(path)
    masks = {}
    for size, members in groups.items():
        if len(members) < min_images:
            continue
        stack = np.stack([residual(p) for p in members])
        median = np.median(stack, axis=0)
        z = median / (stack.std(axis=0) / np.sqrt(len(members)) + 1e-3)
        mask = (median < MEDIAN_THRESHOLD) & (z < Z_THRESHOLD)
        masks[size] = cv2.dilate(mask.astype(np.uint8), np.ones((2 * DILATE + 1, 2 * DILATE + 1), np.uint8)) > 0
    return masks


def save(masks: dict, out: Path, images_used: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    meta = {'parameters': {'median_threshold': MEDIAN_THRESHOLD, 'z_threshold': Z_THRESHOLD, 'sigma': SIGMA, 'dilate_half_res': DILATE,
                           'inpaint_radius': INPAINT_RADIUS, 'min_images_per_size': MIN_IMAGES}, 'images_used': images_used, 'sizes': {}}
    for (h, w), mask in masks.items():
        cv2.imwrite(str(out / f'dust_{h}x{w}.png'), mask.astype(np.uint8) * 255)
        count, _ = cv2.connectedComponents(mask.astype(np.uint8))
        meta['sizes'][f'{h}x{w}'] = {'speck_components': int(count - 1), 'mask_fraction': float(mask.mean())}
    (out / 'dust_masks.json').write_text(json.dumps(meta, indent=2))


def load(folder: Path) -> dict:
    masks = {}
    for path in Path(folder).glob('dust_*x*.png'):
        h, w = (int(v) for v in path.stem.split('_')[1].split('x'))
        masks[(h, w)] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
    return masks


def apply(image_bgr_or_rgb: np.ndarray, masks: dict) -> np.ndarray:
    """Inpaint the camera's static specks; unchanged when no mask matches the image size."""
    h, w = image_bgr_or_rgb.shape[:2]
    mask = masks.get((h, w))
    if mask is None or not mask.any():
        return image_bgr_or_rgb
    full = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.inpaint(image_bgr_or_rgb, full, INPAINT_RADIUS, cv2.INPAINT_TELEA)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--images', type=Path, required=True, help='folder of unlabelled images from ONE camera')
    parser.add_argument('--list', type=Path, help='optional text file naming the images to use (default: all)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    names = [l.strip() for l in args.list.read_text().splitlines() if l.strip()] if args.list else \
        sorted(p.name for p in args.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif'))
    masks = estimate([args.images / n for n in names])
    save(masks, args.out, len(names))
    print(json.dumps(json.loads((args.out / 'dust_masks.json').read_text())['sizes']))


if __name__ == '__main__':
    main()
