#!/usr/bin/env python3
"""Fixtures for the MATLAB learned-lesion module (O3 §3.4): export the ONNX model and write parity references.

Produces, for a few cluster-split VALIDATION images (never the DDR test):
* ``<id>_work.png``   the work image (cropped to the field, long side 1536, INTER_AREA) that the MATLAB wrapper takes as input;
* ``<id>_ref.npz``    Python reference probabilities with the ONNX model run through the same sliding-window logic
                      (windows zero-padded to the fixed 1024 x 1024 ONNX input; ``probs`` (H, W, 4) float32), the
                      Ben-Graham-enhanced uint8 first window (``enhanced_window0``) and that window's network output
                      (``window0_probs``), so preprocessing and network parity can be measured separately;
* ``fixtures.json``   list of ids, thresholds and tolerances.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.io import savemat

import lesion_data as ld
import locked_segmentation_eval as lse


def export_onnx(weights: Path, output: Path, size: int = 1024) -> None:
    model = lse.load_lesion_model(weights, torch.device('cpu')).eval()
    x = torch.randn(1, 3, size, size)
    try:
        torch.onnx.export(model, x, str(output), opset_version=17, input_names=['input'], output_names=['logits'], dynamo=False)
    except TypeError:
        torch.onnx.export(model, x, str(output), opset_version=17, input_names=['input'], output_names=['logits'])


def window_input(window: np.ndarray, sigma: float, tile: int) -> tuple[np.ndarray, np.ndarray]:
    enhanced = ld.ben_graham(np.ascontiguousarray(window), sigma)
    padded = np.zeros((tile, tile, 3), np.uint8)
    padded[:enhanced.shape[0], :enhanced.shape[1]] = enhanced
    return enhanced, ld.to_tensor(padded)[None].numpy()


def predict_full_ort(session, image: np.ndarray, tile: int = 1024, overlap: int = 128) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    sigma = max(2.0, max(h, w) / 100)
    probs = np.zeros((h, w, 4), np.float32)
    counts = np.zeros((h, w, 1), np.float32)
    step = max(32, tile - overlap)
    first = {}
    for top in (list(range(0, max(1, h - overlap), step)) or [0]):
        for left in (list(range(0, max(1, w - overlap), step)) or [0]):
            window = image[top:top + tile, left:left + tile]
            enhanced, x = window_input(window, sigma, tile)
            logits = session.run(None, {'input': x})[0][0, :, :window.shape[0], :window.shape[1]]
            p = (1 / (1 + np.exp(-logits))).transpose(1, 2, 0).astype(np.float32)
            probs[top:top + window.shape[0], left:left + window.shape[1]] += p
            counts[top:top + window.shape[0], left:left + window.shape[1]] += 1
            if not first:
                first = {'enhanced_window0': enhanced, 'window0_probs': p}
    return probs / np.maximum(counts, 1), first


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=Path('../results/model_improvement/lesion_export_2048'))
    parser.add_argument('--thresholds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=int, default=3)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.onnx.exists():
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        export_onnx(args.weights, args.onnx)
    session = ort.InferenceSession(str(args.onnx), providers=['CPUExecutionProvider'])
    manifest = pd.read_csv(args.dataset / 'manifest.csv')
    ids = list(manifest[manifest.split == 'valid'].image_id.iloc[:args.images])
    for image_id in ids:
        rgb = cv2.cvtColor(cv2.imread(str(args.dataset / 'images' / f'{image_id}.png')), cv2.COLOR_BGR2RGB)
        scale = min(1.0, 1536 / max(rgb.shape[:2]))
        work = rgb if scale == 1.0 else cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.output / f'{image_id}_work.png'), cv2.cvtColor(work, cv2.COLOR_RGB2BGR))
        probs, first = predict_full_ort(session, work)
        np.savez_compressed(args.output / f'{image_id}_ref.npz', probs=probs, **first)
        savemat(args.output / f'{image_id}_ref.mat', {'probs': probs, **first})
    cfg = json.loads(args.thresholds.read_text())
    (args.output / 'fixtures.json').write_text(json.dumps({'ids': ids, 'dice_thresholds': cfg['dice_thresholds'],
        'ma_lesion_threshold': cfg['ma_lesion_threshold'], 'tile': 1024, 'overlap': 128,
        'network_tolerance_max_abs_prob': 1e-4, 'end_to_end_mask_disagreement_max_fraction': 1e-3}, indent=1))
    print('fixtures for', ids)


if __name__ == '__main__':
    main()
