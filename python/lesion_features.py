#!/usr/bin/env python3
"""X12 feature extraction: run the frozen lesion segmenter on native images and store threshold-free summaries.

Per image and lesion (EX, HE, MA, SE) over the field of view: a 100-bin histogram of the probability map (bin width
0.01), the maximum probability, and connected-component counts of the thresholded map at probabilities 0.3/0.5/0.7/0.9.
Frozen Dice/lesion thresholds chosen later on validation data are applied to the histogram (rounded to the nearest 0.01),
so one inference pass serves any threshold. The segmenter never saw the images (it trained on DDR + IDRiD only) and
no label is read here.

``resolve`` maps a manifest image id to the native file name of the four leave-one-dataset-out sources.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import lesion_data as ld

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {
    'messidor': ROOT / 'data/messidor2/images',
    'fdr26': ROOT / 'data/fdr26/extracted/images',
    'odir': ROOT / 'data/odir/extracted/ODIR-5K/ODIR-5K/Training Images',
    'deepdrid': ROOT / 'data/DRiD/DeepDRiD-master/regular_fundus_images',
}
BINS = 100
COMPONENT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def file_name(image_id: str, source: str) -> str:
    """Native file name (no folder) for a manifest image id."""
    if source == 'messidor':   # messidor_20051020_43808_0100_PP_png -> 20051020_43808_0100_PP.png
        stem = image_id.removeprefix('messidor_')
        if stem.endswith('_png'):
            return stem[:-4] + '.png'
        return stem[:-4] + '.JPG' if stem.endswith('_JPG') else stem
    if source == 'fdr26':
        return image_id.removeprefix('fdr26_') + '.jpg'
    if source == 'odir':
        return image_id.removeprefix('odir_') + '.jpg'
    if source == 'deepdrid':
        return image_id.removeprefix('deepdrid_') + '.jpg'
    raise ValueError(source)


def local_path(image_id: str, source: str) -> Path:
    name = file_name(image_id, source)
    base = SOURCE_DIRS[source]
    if source == 'deepdrid':
        patient = name.split('_')[0]
        for split in ('regular-fundus-training', 'regular-fundus-validation', 'Online-Challenge1&2-Evaluation'):
            candidate = base / split / 'Images' / patient / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(image_id)
    return base / name


def summarise(probs: np.ndarray, valid: np.ndarray) -> dict:
    """probs (H, W, 4) in [0,1]; valid (H, W) bool field-of-view mask."""
    hist = np.zeros((4, BINS), np.int64)
    maxima = np.zeros(4, np.float32)
    components = np.zeros((4, len(COMPONENT_THRESHOLDS)), np.int32)
    for i in range(4):
        p = probs[..., i][valid]
        hist[i] = np.bincount(np.minimum((p * BINS).astype(np.int64), BINS - 1), minlength=BINS)
        maxima[i] = p.max() if p.size else 0.0
        for k, t in enumerate(COMPONENT_THRESHOLDS):
            binary = ((probs[..., i] >= t) & valid).astype(np.uint8)
            components[i, k] = max(0, cv2.connectedComponents(binary, connectivity=8)[0] - 1)
    return {'hist': hist, 'max': maxima, 'components': components, 'fov_pixels': int(valid.sum())}


def extract(weights: Path, ids: pd.DataFrame, images_dir: Path | None, output: Path, device, long_side=1536, tile=1024,
            overlap=128, enhance='ben', limit=0) -> None:
    import locked_segmentation_eval as lse
    model = lse.load_lesion_model(weights, device)
    output.mkdir(parents=True, exist_ok=True)
    keep = ids.iloc[:limit] if limit else ids
    done = {p.stem for p in output.glob('*.npz')}
    for i, row in enumerate(keep.itertuples()):
        if row.image_id in done:
            continue
        path = (images_dir / file_name(row.image_id, row.source)) if images_dir else local_path(row.image_id, row.source)
        rgb = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        top, bottom, left, right = ld.foreground_box(rgb)
        rgb = rgb[top:bottom, left:right]
        probs = lse.native_probabilities(model, rgb, device, long_side, tile, overlap, enhance)
        s = summarise(probs, rgb.max(axis=2) > ld.FOREGROUND)
        np.savez_compressed(output / f'{row.image_id}.npz', **s)
        if i % 100 == 0:
            print(f'{i}/{len(keep)}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--ids', type=Path, required=True, help='csv with image_id and source (a heldout_rows.csv works)')
    parser.add_argument('--images-dir', type=Path, help='flat folder of native files (remote chunks); default: local datasets')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') \
        if args.device == 'auto' else torch.device(args.device)
    extract(args.weights, pd.read_csv(args.ids), args.images_dir, args.output, device, limit=args.limit)


if __name__ == '__main__':
    sys.exit(main())
