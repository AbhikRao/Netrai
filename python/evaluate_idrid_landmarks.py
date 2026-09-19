#!/usr/bin/env python3
"""Evaluate optic-disc and fovea localization on all 516 IDRiD images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.segmentation import localize_fovea, localize_optic_disc
from utils.preprocessing import preprocess_fundus


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDRID = ROOT / 'data/idrid'


def read_coordinates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[0, 1, 2])
    frame.columns = ['image_id', 'x', 'y']
    return frame.set_index('image_id')


def to_analysis_coordinates(x: float, y: float, width: int, height: int) -> tuple[float, float, float]:
    scale = 512 / max(height, width)
    resized_width = int(width * scale)
    resized_height = int(height * scale)
    left = (512 - resized_width) // 2
    top = (512 - resized_height) // 2
    return x * scale + left, y * scale + top, scale


def bootstrap_summary(values: np.ndarray, replicates: int, rng: np.random.Generator) -> dict:
    means, medians, p90s = [], [], []
    for _ in range(replicates):
        sample = values[rng.integers(0, len(values), size=len(values))]
        means.append(np.mean(sample))
        medians.append(np.median(sample))
        p90s.append(np.percentile(sample, 90))
    return {
        'images': int(len(values)),
        'mean': float(np.mean(values)),
        'mean_ci95': [float(v) for v in np.percentile(means, [2.5, 97.5])],
        'median': float(np.median(values)),
        'median_ci95': [float(v) for v in np.percentile(medians, [2.5, 97.5])],
        'p90': float(np.percentile(values, 90)),
        'p90_ci95': [float(v) for v in np.percentile(p90s, [2.5, 97.5])],
    }


def main(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    localization = dataset / 'C. Localization'
    specifications = [
        (
            'train', 'a. Training Set',
            localization / '2. Groundtruths/1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv',
            localization / '2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Training Set_Markups.csv',
        ),
        (
            'test', 'b. Testing Set',
            localization / '2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv',
            localization / '2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Testing Set_Markups.csv',
        ),
    ]
    rows = []
    for split, folder, od_csv, fovea_csv in specifications:
        od_truth = read_coordinates(od_csv)
        fovea_truth = read_coordinates(fovea_csv)
        image_dir = localization / '1. Original Images' / folder
        images = sorted(image_dir.glob('*.jpg'))
        for image_path in tqdm(images, desc=f'IDRiD {split} landmarks'):
            image_id = image_path.stem
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(image_path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            processed, green, _ = preprocess_fundus(rgb)
            od_prediction, od_radius = localize_optic_disc(processed)
            fovea_prediction = localize_fovea(green, od_prediction, od_radius)
            height, width = rgb.shape[:2]
            for landmark, prediction, truth in (
                ('optic_disc', od_prediction, od_truth.loc[image_id]),
                ('fovea', fovea_prediction, fovea_truth.loc[image_id]),
            ):
                truth_x, truth_y, scale = to_analysis_coordinates(
                    float(truth.x), float(truth.y), width, height)
                error_analysis = float(np.hypot(
                    prediction[0] - truth_x, prediction[1] - truth_y))
                rows.append({
                    'split': split,
                    'image_id': image_id,
                    'landmark': landmark,
                    'predicted_x_analysis': float(prediction[0]),
                    'predicted_y_analysis': float(prediction[1]),
                    'true_x_analysis': truth_x,
                    'true_y_analysis': truth_y,
                    'error_analysis_pixels': error_analysis,
                    'error_native_pixels': error_analysis / scale,
                    'error_fraction_image_diagonal': (
                        (error_analysis / scale) / np.hypot(width, height)),
                })
    frame = pd.DataFrame(rows)
    if len(frame) != 1032:
        raise RuntimeError(f'Expected 1,032 landmark rows, found {len(frame)}')
    rng = np.random.default_rng(args.seed)
    summary = {
        landmark: {
            metric: bootstrap_summary(
                group[metric].to_numpy(float), args.bootstrap_replicates, rng)
            for metric in (
                'error_analysis_pixels', 'error_native_pixels',
                'error_fraction_image_diagonal')
        }
        for landmark, group in frame.groupby('landmark')
    }
    payload = {
        'schema_version': 1,
        'dataset': 'IDRiD localization set (516 images)',
        'summary': summary,
        'limitation': (
            'Classical landmark localization has not been clinically accepted; '
            'errors are reported without selecting a favorable pass threshold.'),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'idrid_landmarks_per_image.csv', index=False)
    (args.output / 'idrid_landmark_metrics.json').write_text(
        json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, allow_nan=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_IDRID)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/idrid_landmark_validation')
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
