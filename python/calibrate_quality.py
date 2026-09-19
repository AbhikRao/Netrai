#!/usr/bin/env python3
"""Calibrate NetrAI M1 Good/Usable/Reject thresholds on EyeQ.

EyeQ quality labels use 0=Good, 1=Usable, and 2=Reject. The official label
files refer to EyePACS ``.jpeg`` names; EyeQ's preprocessing convention stores
the corresponding processed image as ``<stem>.png``. This runner resolves both
forms, caches the measured focus/illumination/FOV/IQS values, fits only on the
official train split, and evaluates once on the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

from modules.quality import assess_image_quality
from utils.quality_calibration import apply_iqs_thresholds, fit_iqs_thresholds


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EYEQ = ROOT / 'data' / 'eyeq'


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_eyeq_image(image_dir: Path, label_name: str) -> Path | None:
    source = Path(str(label_name))
    candidates = [
        image_dir / source.name,
        image_dir / f'{source.stem}.png',
        image_dir / f'{source.stem}.jpg',
        image_dir / f'{source.stem}.jpeg',
    ]
    return next((path for path in candidates if path.is_file()), None)


def extract_features(labels_path: Path, image_dir: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        required = {'image', 'quality', 'iqs', 'focus', 'illumination', 'fov', 'status'}
        if required.issubset(cached.columns):
            return cached
    labels = pd.read_csv(labels_path)
    if not {'image', 'quality'}.issubset(labels.columns):
        raise ValueError(f'{labels_path} must contain image and quality columns')
    rows = []
    for item in tqdm(labels.itertuples(index=False), total=len(labels), desc=labels_path.stem):
        path = resolve_eyeq_image(image_dir, str(item.image))
        if path is None:
            rows.append({
                'image': str(item.image), 'quality': int(item.quality),
                'status': 'missing', 'image_path': '',
            })
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            rows.append({
                'image': str(item.image), 'quality': int(item.quality),
                'status': 'decode_error', 'image_path': str(path),
            })
            continue
        score, metrics = assess_image_quality(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        rows.append({
            'image': str(item.image), 'quality': int(item.quality), 'status': 'success',
            'image_path': str(path), 'iqs': float(score),
            'focus': float(metrics['focus']),
            'illumination': float(metrics['illumination']),
            'fov': float(metrics['fov']),
        })
    result = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(cache_path, index=False)
    return result


def evaluate(frame: pd.DataFrame, reject: float, good: float) -> tuple[dict, np.ndarray]:
    valid = frame.loc[frame['status'] == 'success'].copy()
    labels = valid['quality'].to_numpy(dtype=int)
    predictions = apply_iqs_thresholds(valid['iqs'].to_numpy(dtype=float), reject, good)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1, 2])
    return {
        'samples': int(len(valid)),
        'missing_or_failed': int(len(frame) - len(valid)),
        'accuracy': float(np.mean(labels == predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(
            labels, predictions, labels=[0, 1, 2], average='macro', zero_division=0)),
        'per_class_recall': {
            name: float(matrix[index, index] / max(matrix[index].sum(), 1))
            for index, name in enumerate(('good', 'usable', 'reject'))
        },
    }, matrix


def main(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = extract_features(
        args.train_labels.resolve(), args.train_images.resolve(),
        output / 'eyeq_train_quality_features.csv')
    test = extract_features(
        args.test_labels.resolve(), args.test_images.resolve(),
        output / 'eyeq_test_quality_features.csv')
    train_valid = train.loc[train['status'] == 'success']
    if len(train_valid) != len(train):
        raise RuntimeError(
            f'{len(train)-len(train_valid)} EyeQ training images are missing/unreadable; '
            'calibration is blocked until the dataset is complete')
    test_valid = test.loc[test['status'] == 'success']
    if len(test_valid) != len(test):
        raise RuntimeError(
            f'{len(test)-len(test_valid)} EyeQ test images are missing/unreadable; '
            'evaluation is blocked until the dataset is complete')

    reject, good, optimizer = fit_iqs_thresholds(
        train_valid['iqs'].to_numpy(), train_valid['quality'].to_numpy(), seed=args.seed)
    train_metrics, train_matrix = evaluate(train, reject, good)
    test_metrics, test_matrix = evaluate(test, reject, good)
    pd.DataFrame(
        train_matrix, index=['true_good', 'true_usable', 'true_reject'],
        columns=['pred_good', 'pred_usable', 'pred_reject'],
    ).to_csv(output / 'train_confusion_matrix.csv', index_label='quality')
    pd.DataFrame(
        test_matrix, index=['true_good', 'true_usable', 'true_reject'],
        columns=['pred_good', 'pred_usable', 'pred_reject'],
    ).to_csv(output / 'test_confusion_matrix.csv', index_label='quality')

    artifact = {
        'schema_version': 1,
        'status': 'eyeq_calibrated_research_thresholds',
        'created_on': date.today().isoformat(),
        'label_mapping': {'0': 'Good', '1': 'Usable', '2': 'Reject'},
        'reject_threshold': reject,
        'good_threshold': good,
        'optimizer': optimizer,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'sources': {
            'train_labels': str(args.train_labels.resolve()),
            'train_labels_sha256': file_sha256(args.train_labels.resolve()),
            'test_labels': str(args.test_labels.resolve()),
            'test_labels_sha256': file_sha256(args.test_labels.resolve()),
        },
        'limitation': (
            'EyeQ is a re-annotated EyePACS subset. Thresholds require additional '
            'portable-camera/domain validation before rural clinical use.'),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, indent=2, allow_nan=False) + '\n'
    args.artifact.write_text(serialized, encoding='utf-8')
    (output / 'quality_calibration_metrics.json').write_text(serialized, encoding='utf-8')
    print(serialized, end='')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-labels', type=Path, default=DEFAULT_EYEQ / 'Label_EyeQ_train.csv')
    parser.add_argument('--test-labels', type=Path, default=DEFAULT_EYEQ / 'Label_EyeQ_test.csv')
    parser.add_argument(
        '--train-images', type=Path, default=DEFAULT_EYEQ / 'images' / 'train')
    parser.add_argument(
        '--test-images', type=Path, default=DEFAULT_EYEQ / 'images' / 'test')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'eyeq_quality_calibration')
    parser.add_argument('--artifact', type=Path, default=ROOT / 'python' / 'weights' / 'quality_thresholds_eyeq.json')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
