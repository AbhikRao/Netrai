#!/usr/bin/env python3
"""Audit M1 quality behavior under reproducible field-like corruptions.

The audit starts from officially EyeQ-labeled Good test images, applies
synthetic blur, exposure, illumination, alignment/FOV, and JPEG degradations,
then measures whether IQS and the calibrated Good/Usable/Reject policy respond
in the expected direction.  Synthetic corruptions are stress tests, not a
substitute for real portable-camera validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from calibrate_quality import resolve_eyeq_image
from modules.quality import assess_image_quality, generate_recapture_feedback
from utils.quality_calibration import apply_iqs_thresholds, load_iqs_thresholds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EYEQ = ROOT / 'data' / 'eyeq'
QUALITY_NAMES = {0: 'good', 1: 'usable', 2: 'reject'}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def apply_corruption(image: np.ndarray, name: str) -> np.ndarray:
    """Apply one deterministic, field-motivated image corruption."""
    if name == 'baseline':
        return image.copy()
    if name == 'defocus_blur':
        return cv2.GaussianBlur(image, (0, 0), 4.0)
    if name == 'low_light':
        return np.clip(image.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    if name == 'uneven_illumination':
        height, width = image.shape[:2]
        horizontal = np.linspace(0.25, 1.0, width, dtype=np.float32)
        vertical = np.linspace(0.70, 1.0, height, dtype=np.float32)[:, None]
        gain = (vertical * horizontal)[..., None]
        return np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if name == 'fov_misalignment':
        height, width = image.shape[:2]
        transform = np.float32([[1, 0, round(0.20 * width)], [0, 1, 0]])
        return cv2.warpAffine(
            image, transform, (width, height), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if name == 'jpeg_q25':
        success, encoded = cv2.imencode(
            '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 25])
        if not success:
            raise RuntimeError('OpenCV JPEG encoding failed')
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError('OpenCV JPEG decoding failed')
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    raise ValueError(f'Unknown corruption: {name}')


def summarize(records: pd.DataFrame) -> dict:
    baseline = records.loc[records['corruption'] == 'baseline'].set_index('image')
    summaries = {}
    for corruption, group in records.groupby('corruption', sort=False):
        paired = group.join(
            baseline[['iqs', 'focus', 'illumination', 'fov']],
            on='image', rsuffix='_baseline')
        quality_counts = group['predicted_quality'].value_counts()
        summaries[corruption] = {
            'images': int(len(group)),
            'median_iqs': float(group['iqs'].median()),
            'median_iqs_change_from_baseline': float(
                np.median(paired['iqs'] - paired['iqs_baseline'])),
            'fraction_iqs_decreased': float(
                np.mean(paired['iqs'] < paired['iqs_baseline'])),
            'predicted_quality_counts': {
                name: int(quality_counts.get(name, 0))
                for name in QUALITY_NAMES.values()
            },
            'predicted_quality_rates': {
                name: float(quality_counts.get(name, 0) / len(group))
                for name in QUALITY_NAMES.values()
            },
            'median_factor_change_from_baseline': {
                feature: float(np.median(
                    paired[feature] - paired[f'{feature}_baseline']))
                for feature in ('focus', 'illumination', 'fov')
            },
            'recapture_feedback_rate': float(
                np.mean(group['recapture_feedback'].astype(str).str.len() > 0)),
        }
    return summaries


def run(args: argparse.Namespace) -> dict:
    reject_threshold, good_threshold, threshold_artifact = load_iqs_thresholds(
        args.thresholds.resolve(), allow_evidence_only=True)
    labels = pd.read_csv(args.labels.resolve())
    if not {'image', 'quality'}.issubset(labels.columns):
        raise ValueError('EyeQ labels must contain image and quality columns')
    good = labels.loc[labels['quality'] == 0].copy()
    if args.sample_size > len(good):
        raise ValueError(
            f'--sample-size {args.sample_size} exceeds {len(good)} Good images')
    selected = good.sample(n=args.sample_size, random_state=args.seed)
    corruptions = (
        'baseline', 'defocus_blur', 'low_light', 'uneven_illumination',
        'fov_misalignment', 'jpeg_q25',
    )
    rows = []
    for item in tqdm(selected.itertuples(index=False), total=len(selected),
                     desc='M1 corruption audit'):
        path = resolve_eyeq_image(args.images.resolve(), str(item.image))
        if path is None:
            raise FileNotFoundError(f'Missing selected EyeQ image: {item.image}')
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'Could not decode selected EyeQ image: {path}')
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for corruption in corruptions:
            degraded = apply_corruption(image, corruption)
            iqs, metrics = assess_image_quality(degraded)
            quality = int(apply_iqs_thresholds(
                np.array([iqs]), reject_threshold, good_threshold)[0])
            feedback = ''
            if quality == 2:
                feedback = '; '.join(generate_recapture_feedback(metrics))
            rows.append({
                'image': str(item.image),
                'corruption': corruption,
                'iqs': float(iqs),
                'focus': float(metrics['focus']),
                'illumination': float(metrics['illumination']),
                'fov': float(metrics['fov']),
                'predicted_quality': QUALITY_NAMES[quality],
                'recapture_feedback': feedback,
            })

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = pd.DataFrame(rows)
    records.to_csv(output / 'per_image_quality_robustness.csv', index=False)
    summary = {
        'schema_version': 1,
        'status': 'synthetic_corruption_stress_test_not_camera_validation',
        'sample': {
            'source_split': 'EyeQ official test',
            'source_label': 'Good',
            'images': int(args.sample_size),
            'seed': int(args.seed),
        },
        'thresholds': {
            'artifact': str(args.thresholds.resolve()),
            'artifact_schema_version': threshold_artifact['schema_version'],
            'reject': reject_threshold,
            'good': good_threshold,
        },
        'corruptions': summarize(records),
        'sources': {
            'labels': str(args.labels.resolve()),
            'labels_sha256': file_sha256(args.labels.resolve()),
        },
        'limitations': [
            'These are deterministic synthetic corruptions, not images from portable rural cameras.',
            'EyeQ Good labels are global quality judgments and do not isolate capture failure causes.',
            'The sampled images are image-level, not a patient-level prospective cohort.',
        ],
    }
    with (output / 'quality_robustness_summary.json').open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--labels', type=Path,
        default=DEFAULT_EYEQ / 'data' / 'Label_EyeQ_test.csv')
    parser.add_argument(
        '--images', type=Path, default=DEFAULT_EYEQ / 'images' / 'test')
    parser.add_argument(
        '--thresholds', type=Path,
        default=ROOT / 'python' / 'weights' / 'quality_thresholds_eyeq.json')
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results' / 'quality_robustness')
    parser.add_argument('--sample-size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error('--sample-size must be at least one')
    return args


if __name__ == '__main__':
    run(parse_args())
