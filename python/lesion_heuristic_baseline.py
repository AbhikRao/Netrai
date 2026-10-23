#!/usr/bin/env python3
"""Run the existing heuristic M2 lesion detectors on an exported lesion dataset split (the X8 comparator).

The heuristics output binary candidate masks (one exudate mask for hard+soft exudates, one haemorrhage mask, one
microaneurysm mask) at one operating point, so they are scored by pixel Dice/precision/recall and lesion-level
detection at that point (no AUPR). Exudates are scored against hard exudates (EX) and against EX+SE.
Tier TD (validation data of the cluster-split manifest). Never reads reserved test labels: it only accepts an
exported dataset directory (train/valid splits).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import lesion_data as ld
import lesion_metrics as lm
from evaluate_lesion_masks import restore_native_mask
from modules.segmentation import (detect_exudates, detect_hemorrhages, detect_microaneurysms, localize_fovea,
                                  localize_optic_disc, segment_vessels)
from netrai_eval import bootstrap as bs
from netrai_eval import report as er
from utils.preprocessing import preprocess_fundus

SCORED = {'MA': ('ma', ('MA',)), 'HE': ('hem', ('HE',)), 'EX': ('exudate', ('EX',)), 'EX+SE': ('exudate', ('EX', 'SE'))}


def heuristic_masks(rgb: np.ndarray) -> dict:
    """Binary masks at the image's own size from the deployed heuristic M2 detectors."""
    proc, green, fov = preprocess_fundus(rgb)
    od_center, od_radius = localize_optic_disc(proc)
    fovea = localize_fovea(green, od_center, od_radius)
    vessels = segment_vessels(green, fov)
    ma, _, _ = detect_microaneurysms(green, vessels, fov, od_center, od_radius)
    hem, _, _ = detect_hemorrhages(green, vessels, ma, fov, od_center, od_radius)
    exudate, _, _, _ = detect_exudates(proc, green, fovea, od_center, od_radius, fov)
    shape = rgb.shape[:2]
    return {key: restore_native_mask(value > 0, shape) for key, value in (('ma', ma), ('hem', hem), ('exudate', exudate))}


def pixel_counts(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> tuple[int, int, int]:
    p, t = prediction & valid, truth & valid
    return int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())


def ratio(a, b):
    return float(a / b) if b else float('nan')


def evaluate_split(dataset: Path, split: str = 'valid', max_images: int = 0, replicates: int = 1000, seed: int = 20261023) -> dict:
    if split not in ('train', 'valid'):
        raise ValueError('only exported train/valid splits can be scored')
    frame = pd.read_csv(Path(dataset) / 'manifest.csv')
    frame = frame[frame.split == split].reset_index(drop=True)
    if max_images:
        frame = frame.iloc[:max_images]
    loader = ld.LesionTiles(frame, dataset, tile=32)
    rows = {name: [] for name in SCORED}
    lesion_rows = {name: [] for name in SCORED}
    for image_id in frame.image_id:
        image, masks = loader.load(image_id)
        valid = image.max(axis=2) > ld.FOREGROUND
        predicted = heuristic_masks(image)
        annotated = {l: bool(frame.loc[frame.image_id == image_id, f'annotated_{l}'].iloc[0]) for l in ld.LESIONS}
        for name, (key, lesions) in SCORED.items():
            if not all(annotated[l] for l in lesions):
                continue
            truth = np.logical_or.reduce([masks[:, :, ld.LESIONS.index(l)] > 0 for l in lesions])
            rows[name].append(pixel_counts(predicted[key], truth, valid))
            lesion_rows[name].append(lm.lesion_level(predicted[key].astype(float), truth, 0.5))
    result = {}
    for name in SCORED:
        counts = np.array(rows[name], float)
        if not len(counts):
            continue

        def stat(index):
            tp, fp, fn = counts[index].sum(0)
            return {'dice': ratio(2 * tp, 2 * tp + fp + fn), 'precision': ratio(tp, tp + fp), 'recall': ratio(tp, tp + fn)}

        out = bs.bootstrap(stat, np.arange(len(counts)), replicates, seed)
        det, gt, fp = (sum(r[k] for r in lesion_rows[name]) for k in ('detected', 'gt_lesions', 'false_positives'))
        result[name] = {'images': len(counts), 'pixel': out,
                        'lesion_level': {'gt_lesions': int(gt), 'detected': int(det), 'sensitivity': ratio(det, gt),
                                         'false_positives_per_image': ratio(fp, len(counts))}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'valid'), default='valid')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    result = evaluate_split(args.dataset, args.split, args.max_images, args.replicates)
    n = max((v['images'] for v in result.values()), default=0)
    er.write_report(args.output, er.build_report(
        tier='TD', cohort=f'lesion export {args.split} (cluster split)', model='heuristic M2 (deployed detectors)',
        metrics=result, n_images=n, n_clusters=n,
        limitations=['Binary operating point: no AUPR; exudate mask does not separate hard and soft exudates.',
                     'Run at the exported resolution mapped back from the 512-px analysis grid.',
                     'Development data: valid rows of the cluster-split manifest.']))
    for name, v in result.items():
        print(name, v['images'], {k: round(x['estimate'], 4) for k, x in v['pixel'].items()}, v['lesion_level'])


if __name__ == '__main__':
    main()
