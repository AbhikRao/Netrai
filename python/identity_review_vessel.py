#!/usr/bin/env python3
"""Label-free identity screen of a fresh test against every training/validation image.

The vessel pattern of a retina survives re-encoding, resizing and cropping, which
defeated pHash/ORB for ODIR/DDR copies. Each image becomes a field-of-view-cropped,
green-channel high-pass thumbnail (unit vector). Stage 1 finds each test image's
nearest references by correlation of 64-px thumbnails (also against the mirrored
test image); stage 2 rescores those candidates at 128 px. Copies cluster at >= 0.6;
pairs from 0.45 upward are listed for visual review. Never reads labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.artifacts import atomic_json

IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def vessel_vector(path, size):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
    if image is None:
        raise ValueError(f'Unreadable image {path}')
    green = image[:, :, 1]
    mask = green > 8
    if mask.any():
        ys, xs = mask.nonzero()
        green = green[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    thumb = cv2.resize(green, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    high = thumb - cv2.GaussianBlur(thumb, (0, 0), size / 32)
    yy, xx = np.mgrid[:size, :size]
    centre, radius = (size - 1) / 2, size * 0.4375
    high[(yy - centre) ** 2 + (xx - centre) ** 2 > radius ** 2] = 0  # ignore the FOV rim
    high -= high.mean()
    return high / (np.linalg.norm(high) + 1e-6)


def vectors(paths, size, workers):
    with ProcessPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(vessel_vector, paths, [size] * len(paths), chunksize=64))).reshape(len(paths), -1)


def listing(folder):
    return sorted(p for p in Path(folder).rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--reference', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True, help='directory for pairs CSV and summary JSON')
    parser.add_argument('--candidates', type=int, default=20)
    parser.add_argument('--cache', type=Path, help='optional .npz cache of reference 64-px vectors')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tests = listing(args.test)
    references = [p for folder in args.reference for p in listing(folder)]
    print(f'{len(tests)} test images, {len(references)} reference images', flush=True)
    test64 = vectors(tests, 64, args.workers)
    mirrored = test64.reshape(-1, 64, 64)[:, :, ::-1].reshape(len(tests), -1)
    import hashlib
    key = hashlib.sha256('\n'.join(map(str, references)).encode()).hexdigest()
    if args.cache and args.cache.is_file() and str(np.load(args.cache)['key']) == key:
        reference64 = np.load(args.cache)['vectors']
    else:
        reference64 = vectors(references, 64, args.workers).astype(np.float32)
        if args.cache:
            np.savez(args.cache, key=key, vectors=reference64)
    best = np.full((len(tests), args.candidates), -1.0, np.float32)
    index = np.zeros((len(tests), args.candidates), np.int64)
    flipped = np.zeros((len(tests), args.candidates), bool)
    for start in range(0, len(references), 8192):
        block = reference64[start:start + 8192]
        for flip, query in ((False, test64), (True, mirrored)):
            corr = query.astype(np.float32) @ block.T
            top = np.argpartition(-corr, min(args.candidates, corr.shape[1] - 1), axis=1)[:, :args.candidates]
            values = np.take_along_axis(corr, top, 1)
            merged = np.concatenate([best, values], 1)
            merged_index = np.concatenate([index, top + start], 1)
            merged_flip = np.concatenate([flipped, np.full(values.shape, flip)], 1)
            keep = np.argsort(-merged, 1)[:, :args.candidates]
            best = np.take_along_axis(merged, keep, 1)
            index = np.take_along_axis(merged_index, keep, 1)
            flipped = np.take_along_axis(merged_flip, keep, 1)
        print(f'stage 1: {min(start + 8192, len(references))}/{len(references)}', flush=True)
    pairs = pd.DataFrame({'test': np.repeat([str(t) for t in tests], args.candidates),
                          'reference': [str(references[i]) for i in index.ravel()],
                          'mirrored': flipped.ravel(), 'corr64': best.ravel()})
    unique_paths = sorted(set(pairs.test) | set(pairs.reference))
    detail = dict(zip(unique_paths, vectors(unique_paths, 128, args.workers)))
    pairs['corr128'] = [float((detail[t].reshape(128, 128)[:, ::-1].ravel() if m else detail[t]) @ detail[r])
                        for t, r, m in zip(pairs.test, pairs.reference, pairs.mirrored)]
    pairs = pairs.sort_values('corr128', ascending=False)
    pairs.to_csv(args.output / 'vessel_pairs.csv', index=False)
    top = pairs.groupby('test').corr128.max()
    summary = {'schema_version': 1, 'test_images': len(tests), 'reference_images': len(references),
               'reference_folders': [str(r) for r in args.reference],
               'test_images_max_corr128_histogram': {f'>={c}': int((top >= c).sum()) for c in (0.4, 0.45, 0.5, 0.6, 0.7, 0.8)},
               'copy_threshold_note': 'ODIR/DDR copies cluster >= 0.6; 0.45-0.5 needs visual review'}
    atomic_json(args.output / 'vessel_summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
