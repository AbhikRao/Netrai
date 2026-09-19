#!/usr/bin/env python3
"""Audit normalized NetrAI dataset layouts and write machine-readable evidence."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / 'data'


def count_files(path: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not path.is_dir():
        return 0
    files = (item for item in path.rglob('*') if item.is_file())
    if suffixes is not None:
        normalized = {suffix.lower() for suffix in suffixes}
        files = (item for item in files if item.suffix.lower() in normalized)
    return sum(1 for _ in files)


def csv_rows(path: Path, delimiter: str = ',') -> int:
    if not path.is_file():
        return 0
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return sum(1 for _ in csv.reader(handle, delimiter=delimiter)) - 1


def count_name_suffix(path: Path, suffix: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob(f'*{suffix}') if item.is_file())


def result(status: str, checks: dict, limitations: list[str] | None = None) -> dict:
    return {
        'status': status,
        'checks': checks,
        'limitations': limitations or [],
    }


def audit_eyeq() -> dict:
    root = DATA_ROOT / 'eyeq'
    train = csv_rows(root / 'Label_EyeQ_train.csv')
    test = csv_rows(root / 'Label_EyeQ_test.csv')
    images = count_files(root / 'images', ('.jpeg', '.jpg', '.png'))
    complete = train == 12543 and test == 16249 and images >= 28792
    return result(
        'ready' if complete else 'blocked_missing_source_images',
        {
            'train_label_rows': train,
            'expected_train_label_rows': 12543,
            'test_label_rows': test,
            'expected_test_label_rows': 16249,
            'source_images': images,
            'expected_source_images': 28792,
            'sha256_manifest_present': (root / 'SHA256SUMS.txt').is_file(),
        },
        [] if complete else [
            'EyeQ labels are complete, but the corresponding EyePACS images are missing.',
        ],
    )


def audit_idrid() -> dict:
    root = DATA_ROOT / 'idrid'
    grading = root / 'B. Disease Grading/1. Original Images'
    segmentation = root / 'A. Segmentation/1. Original Images'
    masks = root / 'A. Segmentation/2. All Segmentation Groundtruths'
    checks = {
        'grading_train_images': count_files(grading / 'a. Training Set', ('.jpg',)),
        'grading_test_images': count_files(grading / 'b. Testing Set', ('.jpg',)),
        'segmentation_train_images': count_files(
            segmentation / 'a. Training Set', ('.jpg',)),
        'segmentation_test_images': count_files(
            segmentation / 'b. Testing Set', ('.jpg',)),
        'microaneurysm_masks': count_name_suffix(masks, '_MA.tif'),
        'haemorrhage_masks': count_name_suffix(masks, '_HE.tif'),
        'hard_exudate_masks': count_name_suffix(masks, '_EX.tif'),
        'soft_exudate_masks': count_name_suffix(masks, '_SE.tif'),
        'optic_disc_masks': count_name_suffix(masks, '_OD.tif'),
        'all_tif_masks': count_files(masks, ('.tif',)),
        'sha256_manifest_present': (root / 'SHA256SUMS.txt').is_file(),
    }
    complete = (
        checks['grading_train_images'] == 413
        and checks['grading_test_images'] == 103
        and checks['segmentation_train_images'] == 54
        and checks['segmentation_test_images'] == 27
        and checks['all_tif_masks'] == 363
    )
    return result('ready' if complete else 'incomplete', checks)


def audit_drive() -> dict:
    root = DATA_ROOT / 'drive'
    checks = {
        'training_images': count_files(root / 'training/images', ('.tif',)),
        'training_fov_masks': count_files(root / 'training/mask', ('.gif',)),
        'training_manual_masks': count_files(root / 'training/1st_manual', ('.gif',)),
        'test_images': count_files(root / 'test/images', ('.tif',)),
        'test_fov_masks': count_files(root / 'test/mask', ('.gif',)),
        'test_manual_masks': count_files(root / 'test/1st_manual', ('.gif',)),
        'sha256_manifest_present': (root / 'SHA256SUMS.txt').is_file(),
    }
    training_ready = all(checks[key] == 20 for key in (
        'training_images', 'training_fov_masks', 'training_manual_masks'))
    return result(
        'training_benchmark_ready' if training_ready else 'incomplete',
        checks,
        [
            'The downloaded official test package has no public 1st_manual masks; '
            'vessel evaluation therefore uses the 20 annotated training cases.'
        ] if checks['test_manual_masks'] == 0 else [],
    )


def audit_messidor2() -> dict:
    root = DATA_ROOT / 'messidor2'
    images = count_files(root / 'images', ('.jpg', '.png'))
    pairs = csv_rows(root / 'messidor-2.csv', delimiter=';')
    return result(
        'images_ready_labels_unavailable' if images == 1748 else 'incomplete',
        {
            'images': images,
            'expected_images': 1748,
            'paired_exam_rows': pairs,
            'expected_paired_exam_rows': 874,
            'sha256_manifest_present': (root / 'SHA256SUMS.txt').is_file(),
        },
        [
            'The official Messidor-2 release has pairing metadata but no DR ground truth.',
        ],
    )


def main(args: argparse.Namespace) -> None:
    payload = {
        'schema_version': 1,
        'audited_on': date.today().isoformat(),
        'data_root': str(DATA_ROOT),
        'datasets': {
            'eyeq': audit_eyeq(),
            'idrid': audit_idrid(),
            'drive': audit_drive(),
            'messidor2': audit_messidor2(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, allow_nan=False))
    if args.strict and any(
        value['status'] in {'incomplete', 'blocked_missing_source_images'}
        for value in payload['datasets'].values()
    ):
        raise SystemExit(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/dataset_audit/dataset_audit.json')
    parser.add_argument('--strict', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
