#!/usr/bin/env python3
"""Build a normalized IDRiD segmentation manifest from the official layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDRID = ROOT / 'data/idrid'


LESIONS = {
    'ma_mask': ('1. Microaneurysms', 'MA'),
    'hem_mask': ('2. Haemorrhages', 'HE'),
    'hard_exudate_mask': ('3. Hard Exudates', 'EX'),
    'soft_exudate_mask': ('4. Soft Exudates', 'SE'),
    'optic_disc_mask': ('5. Optic Disc', 'OD'),
}


def relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def build_manifest(dataset: Path) -> pd.DataFrame:
    image_root = dataset / 'A. Segmentation/1. Original Images'
    mask_root = dataset / 'A. Segmentation/2. All Segmentation Groundtruths'
    rows = []
    for split_dir, split_name in (
        ('a. Training Set', 'train'), ('b. Testing Set', 'test')):
        for image_path in sorted((image_root / split_dir).glob('*.jpg')):
            stem = image_path.stem
            row = {
                'image_id': stem,
                'split': split_name,
                'image_path': relative(image_path, dataset),
            }
            for column, (folder, suffix) in LESIONS.items():
                mask_path = mask_root / split_dir / folder / f'{stem}_{suffix}.tif'
                row[column] = relative(mask_path, dataset) if mask_path.is_file() else ''
            rows.append(row)
    manifest = pd.DataFrame(rows)
    if len(manifest) != 81:
        raise RuntimeError(f'Expected 81 IDRiD segmentation images, found {len(manifest)}')
    expected = {
        'ma_mask': 81,
        'hem_mask': 80,
        'hard_exudate_mask': 81,
        'soft_exudate_mask': 40,
        'optic_disc_mask': 81,
    }
    for column, count in expected.items():
        actual = int((manifest[column] != '').sum())
        if actual != count:
            raise RuntimeError(f'Expected {count} {column} files, found {actual}')
    return manifest


def main(args: argparse.Namespace) -> None:
    dataset = args.dataset.resolve()
    manifest = build_manifest(dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f'IDRiD manifest written: {args.output.resolve()} ({len(manifest)} images)')
    print(manifest.notna().sum().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT_IDRID)
    parser.add_argument(
        '--output', type=Path,
        default=DEFAULT_IDRID / 'segmentation_manifest.csv')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
