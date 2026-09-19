#!/usr/bin/env python3
"""Evaluate NetrAI vessel candidates on the annotated DRIVE training split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.segmentation import segment_vessels
from utils.preprocessing import preprocess_fundus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVE = ROOT / 'data/drive'


def transform_mask(mask: np.ndarray, target_size: int = 512) -> np.ndarray:
    height, width = mask.shape[:2]
    scale = target_size / max(height, width)
    resized = cv2.resize(
        mask, (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_NEAREST)
    top = (target_size - resized.shape[0]) // 2
    bottom = target_size - resized.shape[0] - top
    left = (target_size - resized.shape[1]) // 2
    right = target_size - resized.shape[1] - left
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT) > 0


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1))


def metrics_from_counts(counts: pd.Series | dict) -> dict[str, float]:
    tp, tn, fp, fn = (float(counts[name]) for name in ('tp', 'tn', 'fp', 'fn'))
    return {
        'sensitivity': safe_ratio(tp, tp + fn),
        'specificity': safe_ratio(tn, tn + fp),
        'precision': safe_ratio(tp, tp + fp),
        'dice': safe_ratio(2 * tp, 2 * tp + fp + fn),
        'iou': safe_ratio(tp, tp + fp + fn),
        'accuracy': safe_ratio(tp + tn, tp + tn + fp + fn),
    }


def evaluate_case(image_path: Path, truth_path: Path, fov_path: Path) -> dict:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    truth_raw = cv2.imread(str(truth_path), cv2.IMREAD_GRAYSCALE)
    fov_raw = cv2.imread(str(fov_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or truth_raw is None or fov_raw is None:
        raise FileNotFoundError(
            f'Could not decode DRIVE case: {image_path}, {truth_path}, {fov_path}')
    processed, green, _ = preprocess_fundus(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    del processed
    official_fov = transform_mask(fov_raw)
    truth = transform_mask(truth_raw)
    prediction = segment_vessels(green, official_fov.astype(np.uint8) * 255) > 0
    tp = int(np.logical_and(prediction, truth & official_fov).sum())
    fp = int(np.logical_and(prediction, ~truth & official_fov).sum())
    fn = int(np.logical_and(~prediction, truth & official_fov).sum())
    tn = int(np.logical_and(~prediction, ~truth & official_fov).sum())
    counts = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    return {**counts, **metrics_from_counts(counts)}


def bootstrap(frame: pd.DataFrame, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    point = metrics_from_counts(frame[['tp', 'tn', 'fp', 'fn']].sum())
    samples = {name: [] for name in point}
    for _ in range(replicates):
        indices = rng.integers(0, len(frame), size=len(frame))
        sample = metrics_from_counts(
            frame.iloc[indices][['tp', 'tn', 'fp', 'fn']].sum())
        for name, value in sample.items():
            samples[name].append(value)
    result = {'images': len(frame)}
    for name, value in point.items():
        low, high = np.percentile(samples[name], [2.5, 97.5])
        result[name] = value
        result[f'{name}_ci95_low'] = float(low)
        result[f'{name}_ci95_high'] = float(high)
    return result


def main(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    image_dir = dataset / 'training/images'
    truth_dir = dataset / 'training/1st_manual'
    fov_dir = dataset / 'training/mask'
    images = sorted(image_dir.glob('*_training.tif'))
    if len(images) != 20:
        raise RuntimeError(f'Expected 20 DRIVE training images, found {len(images)}')
    rows = []
    for image_path in tqdm(images, desc='DRIVE vessel evaluation'):
        case_id = image_path.name.split('_', 1)[0]
        truth_path = truth_dir / f'{case_id}_manual1.gif'
        fov_path = fov_dir / f'{case_id}_training_mask.gif'
        rows.append({
            'case_id': case_id,
            'image_path': str(image_path),
            **evaluate_case(image_path, truth_path, fov_path),
        })
    frame = pd.DataFrame(rows)
    summary = bootstrap(frame, args.bootstrap_replicates, args.seed)
    summary['dataset'] = 'DRIVE annotated training split'
    summary['limitation'] = (
        'The downloaded official test package has no 1st_manual masks; '
        'evaluation is limited to the 20 public annotated training cases.')
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'drive_vessel_per_image.csv', index=False)
    (args.output / 'drive_vessel_metrics.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DRIVE)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/drive_vessel_validation')
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
