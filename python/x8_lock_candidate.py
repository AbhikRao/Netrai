#!/usr/bin/env python3
"""Build the X8 lock-candidate package from a trained lesion model, using VALIDATION images only (never the DDR test).

Scores the cluster-split validation images at NATIVE resolution exactly as ``locked_segmentation_eval.py`` will score
the test (bilinear upsampling of the probabilities, native masks binarised at > 0), chooses

* per-lesion Dice thresholds (best pooled Dice on the validation histograms), and
* the MA lesion-level threshold: highest sensitivity with <= ``--max-fp`` false-positive components per image,

and writes ``thresholds.json`` (the format ``locked_segmentation_eval.py freeze`` expects), a copy of the weights,
native-resolution validation metrics and a README. It does NOT freeze or lock anything.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import lesion_data as ld
import lesion_metrics as lm
import locked_segmentation_eval as lse
from utils.artifacts import atomic_json, sha256_file


def records_by_id() -> dict:
    out = {}
    for split in ('train', 'valid'):
        for r in ld.ddr_records(split):
            out[r.image_id] = r
    for r in ld.idrid_train_records():
        out[r.image_id] = r
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('../results/model_improvement/lesion_export_2048'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--long-side', type=int, default=1536)
    parser.add_argument('--tile', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--max-fp', type=float, default=8.0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists')
    manifest = pd.read_csv(args.dataset / 'manifest.csv')
    valid = manifest[manifest.split == 'valid'].reset_index(drop=True)
    if args.limit:
        valid = valid.iloc[:args.limit]
    lookup = records_by_id()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lse.load_lesion_model(args.weights, device)
    lesion_set, kept, grid = lm.LesionSet(), [], np.round(np.arange(0.10, 0.991, 0.025), 3)
    detection = {float(t): [0, 0, 0] for t in grid}  # detected, gt lesions, false positives (MA, annotated images)
    ma = ld.LESIONS.index('MA')
    for row in valid.itertuples():
        record = lookup[row.image_id]
        rgb = cv2.cvtColor(cv2.imread(str(record.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        masks, annotated = ld.read_masks(record, rgb.shape[:2])
        top, bottom, left, right = ld.foreground_box(rgb)
        rgb, masks = rgb[top:bottom, left:right], masks[top:bottom, left:right]
        probs = lse.native_probabilities(model, rgb, device, args.long_side, args.tile, args.overlap, 'ben')
        lesion_set.add(probs, masks, rgb.max(axis=2) > ld.FOREGROUND)
        kept.append(row.image_id)
        if annotated[ma]:
            for t in grid:
                r = lm.lesion_level(probs[..., ma], masks[..., ma], float(t))
                d = detection[float(t)]
                d[0] += r['detected']; d[1] += r['gt_lesions']; d[2] += r['false_positives']
        print(f'{len(kept)}/{len(valid)}', flush=True)
    summary = lesion_set.summary()
    thresholds = {name: summary[name]['threshold_chosen_on_this_set'] for name in ld.LESIONS}
    n_ma = sum(1 for i in kept if manifest.loc[manifest.image_id == i, 'annotated_MA'].iloc[0])
    curve = {str(t): {'sensitivity': d[0] / d[1] if d[1] else float('nan'), 'fp_per_image': d[2] / max(n_ma, 1)} for t, d in detection.items()}
    eligible = [(v['sensitivity'], -float(t), float(t)) for t, v in curve.items() if v['fp_per_image'] <= args.max_fp]
    ma_threshold = max(eligible)[2] if eligible else float(grid[-1])
    args.output.mkdir(parents=True)
    shutil.copy2(args.weights, args.output / 'best.pt')
    atomic_json(args.output / 'thresholds.json', {'dice_thresholds': thresholds, 'ma_lesion_threshold': ma_threshold,
                                                  'selected_on': 'validation images only, native resolution',
                                                  'max_fp_per_image': args.max_fp})
    atomic_json(args.output / 'validation_native.json', {
        'tier': 'TD (cluster-split validation images, native resolution; thresholds chosen on these images)',
        'images': len(kept), 'summary': summary, 'ma_lesion_curve': curve, 'ma_lesion_threshold': ma_threshold,
        'weights_sha256': sha256_file(args.weights), 'inference': {'long_side': args.long_side, 'tile': args.tile, 'overlap': args.overlap}})
    (args.output / 'README.md').write_text(
        'X8 lock candidate (NOT frozen). Weights: best.pt (sha256 in validation_native.json). thresholds.json is the input of\n'
        '`python locked_segmentation_eval.py freeze --weights best.pt --thresholds thresholds.json --exclusions '
        '../results/model_improvement/ddr_lesion_test_exclusions_v1.csv --output <lock dir>`; freezing needs the O4 review.\n')
    print(json.dumps({'dice_thresholds': thresholds, 'ma_lesion_threshold': ma_threshold}))


if __name__ == '__main__':
    main()
