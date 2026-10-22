#!/usr/bin/env python3
"""Lesion-segmentation data: records, guarded mask reading, cluster-based train/valid split, export, tile sampling.

* DDR lesion segmentation train/valid + IDRiD segmentation training images are the only labelled sources used
  for training and selection. The DDR **test** masks are reserved (``exposure``): they can only be addressed with a
  valid lock directory, and every mask read is checked against the exposure registry.
* The released train/valid split is NOT used for selection: DDR repeats the same retina across it (a 49-image
  cluster spans both), so train+valid are re-split by vessel-correlation cluster (``vessel_clusters.py``).
* A missing IDRiD mask means "not annotated", not "no lesion": ``annotated`` flags let the loss ignore that channel.
* Images are cropped to the field of view and exported with the long side at most ``long_side`` (INTER_AREA);
  masks are resized with INTER_AREA and re-binarised at >= 48/255 so tiny lesions survive.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import exposure
from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
LESIONS = ('EX', 'HE', 'MA', 'SE')
DDR_ROOT = ROOT / 'data/DDR/extracted/DDR-dataset/lesion_segmentation'
IDRID_ROOT = ROOT / 'data/idrid/A. Segmentation'
DDR_LABEL_DIR = {'train': 'label', 'valid': 'segmentation label', 'test': 'label'}
IDRID_MASK_DIR = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages', 'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
IDRID_SUFFIX = {'MA': 'MA', 'HE': 'HE', 'EX': 'EX', 'SE': 'SE'}
FOREGROUND = 7
MASK_BINARISE = 48
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CLUSTERS_CSV = ROOT / 'results/model_improvement/lesion_train_valid_clusters/clusters.csv'


@dataclass(frozen=True)
class Record:
    image_id: str
    dataset: str
    image_path: Path
    mask_paths: dict  # lesion -> Path or None (None = not annotated)


def ddr_records(split: str, lock_dir: Path | None = None) -> list[Record]:
    """DDR records for 'train' or 'valid'. 'test' needs a valid exposure lock (labels are reserved)."""
    if split not in DDR_LABEL_DIR:
        raise ValueError(f'unknown DDR split {split!r}')
    if split == 'test':
        if lock_dir is None:
            raise exposure.ExposureError('The DDR lesion test labels are reserved: pass the lock directory of the locked evaluation')
        exposure.assert_label_read_allowed('ddr_lesion_seg_test', lock_dir)
    images = DDR_ROOT / split / 'image'
    labels = DDR_ROOT / split / DDR_LABEL_DIR[split]
    out = []
    for path in sorted(images.glob('*.jpg')):
        masks = {lesion: labels / lesion / f'{path.stem}.tif' for lesion in LESIONS}
        out.append(Record(f'ddr_{path.stem}', 'ddr', path, {k: (v if v.is_file() else None) for k, v in masks.items()}))
    return out


def ddr_test_records_exposed() -> list[Record]:
    """DDR lesion-test records for POST HOC analyses after the single locked evaluation (registry status must be exposed)."""
    if exposure.status('ddr_lesion_seg_test') != 'exposed':
        raise exposure.ExposureError('The DDR lesion test is not exposed: its labels may only be read through the locked evaluation')
    images = DDR_ROOT / 'test' / 'image'
    labels = DDR_ROOT / 'test' / DDR_LABEL_DIR['test']
    out = []
    for path in sorted(images.glob('*.jpg')):
        masks = {lesion: labels / lesion / f'{path.stem}.tif' for lesion in LESIONS}
        out.append(Record(f'ddr_{path.stem}', 'ddr', path, {k: (v if v.is_file() else None) for k, v in masks.items()}))
    return out


def idrid_test_records() -> list[Record]:
    """IDRiD segmentation TEST images (27; exposed development data, post hoc use only). Missing masks = not annotated."""
    images = IDRID_ROOT / '1. Original Images' / 'b. Testing Set'
    truth = IDRID_ROOT / '2. All Segmentation Groundtruths' / 'b. Testing Set'
    out = []
    for path in sorted(images.glob('*.jpg')):
        masks = {}
        for lesion in LESIONS:
            candidate = truth / IDRID_MASK_DIR[lesion] / f'{path.stem}_{IDRID_SUFFIX[lesion]}.tif'
            masks[lesion] = candidate if candidate.is_file() else None
        out.append(Record(f'idrid_{path.stem}', 'idrid', path, masks))
    return out


def idrid_train_records() -> list[Record]:
    """IDRiD segmentation training images (54). Missing lesion masks are 'not annotated'."""
    images = IDRID_ROOT / '1. Original Images' / 'a. Training Set'
    truth = IDRID_ROOT / '2. All Segmentation Groundtruths' / 'a. Training Set'
    out = []
    for path in sorted(images.glob('*.jpg')):
        masks = {}
        for lesion in LESIONS:
            candidate = truth / IDRID_MASK_DIR[lesion] / f'{path.stem}_{IDRID_SUFFIX[lesion]}.tif'
            masks[lesion] = candidate if candidate.is_file() else None
        out.append(Record(f'idrid_{path.stem}', 'idrid', path, masks))
    return out


def read_masks(record: Record, shape: tuple[int, int], allow_reserved: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, L) uint8 masks in {0, 255} and an (L,) bool 'annotated' vector. Reserved label files are refused."""
    masks = np.zeros((*shape, len(LESIONS)), np.uint8)
    annotated = np.zeros(len(LESIONS), bool)
    for index, lesion in enumerate(LESIONS):
        path = record.mask_paths.get(lesion)
        if path is None:
            continue
        if not allow_reserved:
            exposure.assert_not_reserved_path(path)
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.shape[:2] != shape:
            raise ValueError(f'{path}: mask missing/unreadable or size {None if mask is None else mask.shape} != image {shape}')
        if mask.ndim == 3:  # some IDRiD test masks are multi-channel TIFFs: lesion = any colour channel set (alpha ignored)
            mask = mask[..., :3].max(axis=2)
        masks[:, :, index] = np.where(mask > 0, 255, 0)
        annotated[index] = True
    return masks, annotated


def foreground_box(image: np.ndarray, threshold: int = FOREGROUND) -> tuple[int, int, int, int]:
    mask = image.max(axis=2) > threshold
    if not mask.any():
        return 0, image.shape[0], 0, image.shape[1]
    rows, cols = mask.any(1), mask.any(0)
    return (int(rows.argmax()), int(len(rows) - rows[::-1].argmax()), int(cols.argmax()), int(len(cols) - cols[::-1].argmax()))


def prepare(image: np.ndarray, masks: np.ndarray, long_side: int = 2048) -> tuple[np.ndarray, np.ndarray, float]:
    """Crop to the field of view and downsize (never upsize) so the long side is at most ``long_side``."""
    top, bottom, left, right = foreground_box(image)
    image, masks = image[top:bottom, left:right], masks[top:bottom, left:right]
    scale = min(1.0, long_side / max(image.shape[:2]))
    if scale < 1.0:
        size = (max(32, round(image.shape[1] * scale)), max(32, round(image.shape[0] * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        resized = cv2.resize(masks, size, interpolation=cv2.INTER_AREA)
        masks = resized.reshape(size[1], size[0], -1) if resized.ndim == 2 else resized
        masks = np.where(masks >= MASK_BINARISE, 255, 0).astype(np.uint8)
    return image, masks, scale


def cluster_split(frame: pd.DataFrame, clusters: pd.DataFrame, valid_fraction: float = 0.2, seed: int = 20261020) -> pd.DataFrame:
    """Assign DDR images to train/valid by cluster; the largest cluster always goes to train; IDRiD is train only."""
    by_stem = {Path(p).stem: int(c) for p, c in zip(clusters.image, clusters.cluster)}
    frame = frame.copy()
    stems = frame.image_id.str.replace(r'^(ddr|idrid)_', '', regex=True)
    missing = [s for s in stems if s not in by_stem]
    if missing:
        raise ValueError(f'{len(missing)} images have no cluster, e.g. {missing[:3]}')
    frame['cluster'] = [by_stem[s] for s in stems]
    frame['split'] = 'train'
    ddr = frame[frame.dataset == 'ddr']
    sizes = ddr.groupby('cluster').size()
    largest = sizes.idxmax()
    order = [c for c in sizes.index if c != largest]
    np.random.default_rng(seed).shuffle(order)
    target, taken = valid_fraction * len(ddr), 0
    for cluster in order:
        if taken >= target:
            break
        frame.loc[(frame.dataset == 'ddr') & (frame.cluster == cluster), 'split'] = 'valid'
        taken += int(sizes[cluster])
    return frame


def _export_one(task):
    record, out, long_side = task
    image = cv2.cvtColor(cv2.imread(str(record.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    masks, annotated = read_masks(record, image.shape[:2])
    cropped, small, scale = prepare(image, masks, long_side)
    cv2.imwrite(str(out / 'images' / f'{record.image_id}.png'), cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    cv2.imwrite(str(out / 'masks' / f'{record.image_id}.png'), small, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    row = {'image_id': record.image_id, 'dataset': record.dataset, 'height': cropped.shape[0], 'width': cropped.shape[1],
           'scale': scale, 'source_sha256': hashlib.sha256(Path(record.image_path).read_bytes()).hexdigest()}
    for index, lesion in enumerate(LESIONS):
        row[f'annotated_{lesion}'] = bool(annotated[index])
        row[f'pixels_{lesion}'] = int((small[:, :, index] > 0).sum())
    return row


def export_dataset(out: Path, long_side: int = 2048, valid_fraction: float = 0.2, seed: int = 20261020,
                   clusters_csv: Path = CLUSTERS_CSV, workers: int = 4) -> pd.DataFrame:
    """Train/valid records only (there is deliberately no test option). The output directory is created once."""
    out = Path(out)
    if (out / 'manifest.csv').exists():
        raise FileExistsError(f'{out} exists; exported lesion datasets are immutable')
    records = ddr_records('train') + ddr_records('valid') + idrid_train_records()
    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'masks').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_export_one, [(r, out, long_side) for r in records], chunksize=4))
    frame = cluster_split(pd.DataFrame(rows), pd.read_csv(clusters_csv), valid_fraction, seed)
    frame.to_csv(out / 'manifest.csv', index=False)
    atomic_json(out / 'export_summary.json', {
        'schema_version': 1, 'long_side': long_side, 'valid_fraction': valid_fraction, 'seed': seed,
        'images': len(frame), 'by_split_dataset': {f'{s}/{d}': int(n) for (s, d), n in frame.groupby(['split', 'dataset']).size().items()},
        'positive_pixels': {l: {s: int(frame.loc[frame.split == s, f'pixels_{l}'].sum()) for s in ('train', 'valid')} for l in LESIONS},
        'annotated_images': {l: int(frame[f'annotated_{l}'].sum()) for l in LESIONS},
        'reserved': 'DDR lesion-segmentation/detection test masks were not read',
        'clusters_csv': str(Path(clusters_csv).relative_to(ROOT)) if str(clusters_csv).startswith(str(ROOT)) else str(clusters_csv)})
    return frame


# ----------------------------------------------------------------------------- training tiles


def ben_graham(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma), -4, 128)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((image.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1).copy())


class LesionTiles(Dataset):
    """Random training tiles centred on lesions with probability ``lesion_prob`` (else anywhere in the field)."""

    def __init__(self, frame: pd.DataFrame, root: Path, tile: int = 512, long_side: int | None = None, seed: int = 0,
                 lesion_prob: float = 0.7, enhance: str = 'ben', length: int | None = None, weights=None):
        if tile % 32:
            raise ValueError('tile must be a multiple of 32')
        self.frame, self.root, self.tile, self.long_side = frame.reset_index(drop=True), Path(root), tile, long_side
        self.seed, self.lesion_prob, self.enhance, self.epoch = seed, lesion_prob, enhance, 0
        self.length = length or len(self.frame)
        self.weights = None if weights is None else np.asarray(weights, float) / np.sum(weights)

    def __len__(self):
        return self.length

    def load(self, image_id: str):
        image = cv2.cvtColor(cv2.imread(str(self.root / 'images' / f'{image_id}.png'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        masks = cv2.imread(str(self.root / 'masks' / f'{image_id}.png'), cv2.IMREAD_UNCHANGED)
        if masks.ndim == 2:
            masks = masks[:, :, None]
        if self.long_side and max(image.shape[:2]) > self.long_side:
            scale = self.long_side / max(image.shape[:2])
            size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            masks = np.where(cv2.resize(masks, size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], -1) >= MASK_BINARISE, 255, 0).astype(np.uint8)
        return image, masks

    def sample_origin(self, masks, annotated, rng):
        h, w = masks.shape[:2]
        half = self.tile // 2
        positives = [i for i in range(masks.shape[2]) if annotated[i] and (masks[:, :, i] > 0).any()]
        if positives and rng.random() < self.lesion_prob:
            lesion = positives[rng.integers(len(positives))]  # uniform over lesion types present: rare ones are not drowned
            ys, xs = np.nonzero(masks[:, :, lesion])
            k = rng.integers(len(ys))
            cy, cx = ys[k] + rng.integers(-half // 2, half // 2 + 1), xs[k] + rng.integers(-half // 2, half // 2 + 1)
        else:
            cy, cx = rng.integers(0, h), rng.integers(0, w)
        top = int(np.clip(cy - half, 0, max(0, h - self.tile)))
        left = int(np.clip(cx - half, 0, max(0, w - self.tile)))
        return top, left

    def __getitem__(self, position):
        rng = np.random.default_rng((self.seed, self.epoch, int(position)))
        row = self.frame.iloc[int(rng.choice(len(self.frame), p=self.weights))]
        image, masks = self.load(row.image_id)
        annotated = np.array([bool(row[f'annotated_{lesion}']) for lesion in LESIONS])
        top, left = self.sample_origin(masks, annotated, rng)
        tile_img = np.zeros((self.tile, self.tile, 3), np.uint8)
        tile_mask = np.zeros((self.tile, self.tile, masks.shape[2]), np.uint8)
        crop, mcrop = image[top:top + self.tile, left:left + self.tile], masks[top:top + self.tile, left:left + self.tile]
        tile_img[:crop.shape[0], :crop.shape[1]] = crop
        tile_mask[:mcrop.shape[0], :mcrop.shape[1]] = mcrop
        if rng.random() < 0.5:
            tile_img, tile_mask = tile_img[:, ::-1], tile_mask[:, ::-1]
        if rng.random() < 0.5:
            tile_img, tile_mask = tile_img[::-1], tile_mask[::-1]
        k = int(rng.integers(4))
        tile_img, tile_mask = np.rot90(tile_img, k), np.rot90(tile_mask, k)
        tile_img = tile_img.astype(np.float32)
        tile_img = np.clip((tile_img - 128) * rng.uniform(0.8, 1.25) + 128 + rng.uniform(-20, 20), 0, 255)
        tile_img = np.clip(255 * (tile_img / 255) ** rng.uniform(0.8, 1.25), 0, 255).astype(np.uint8)
        if self.enhance == 'ben':
            tile_img = ben_graham(np.ascontiguousarray(tile_img), sigma=max(2.0, max(image.shape[:2]) / 100))
        mask_tensor = torch.from_numpy((np.ascontiguousarray(tile_mask) > 0).astype(np.float32).transpose(2, 0, 1).copy())
        return to_tensor(np.ascontiguousarray(tile_img)), mask_tensor, torch.from_numpy(annotated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--long-side', type=int, default=2048)
    parser.add_argument('--valid-fraction', type=float, default=0.2)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    frame = export_dataset(args.out, args.long_side, args.valid_fraction, workers=args.workers)
    print(frame.groupby(['split', 'dataset']).size())


if __name__ == '__main__':
    main()
