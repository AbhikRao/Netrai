#!/usr/bin/env python3
"""Round-5 additions to the 768-px export (plan MODEL_IMPROVEMENT_PLAN_v5.md).

Adds FDR26 (exposed in round 4; 2,833 de-duplicated images, final grade_DR) split by
merged cluster 75/25 into train/validation, on top of the round-4 manifest. Only the
FDR26 images are exported (same ``legacy-area-v1`` lossless PNG contract). The
round-5 fresh tests (mBRSET, BRSET) are never exported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from export_round3 import export_one
from locked_evaluation import ROUND4_COHORTS, cohort_labels, inventory
from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def fdr26(seed):
    frame = inventory('fdr26')
    frame, gradable, grade = cohort_labels('fdr26', frame)
    frame, grade = frame.loc[gradable].reset_index(drop=True), grade[gradable].astype(int)
    clusters = frame.exam_id.drop_duplicates().sample(frac=1, random_state=seed)
    validation = set(clusters.iloc[:round(0.25 * len(clusters))])
    folder = ROUND4_COHORTS['fdr26']['images']
    return pd.DataFrame({'image_id': 'fdr26_' + frame.image_id.str.replace('.jpg', '', regex=False),
                         'path': [str(folder / i) for i in frame.image_id], 'grade': grade,
                         'group_id': frame.exam_id.values, 'source': 'fdr26',
                         'split': ['validation' if c in validation else 'train' for c in frame.exam_id]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--round4', type=Path, default=ROOT / 'results/model_improvement/train_export_768_r4')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_improvement/train_export_768_r5')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20261014)
    args = parser.parse_args()
    base = pd.read_csv(args.round4 / 'manifest.csv')
    added = fdr26(args.seed)
    frame = pd.concat([base, added.drop(columns='path')], ignore_index=True)
    if frame.image_id.duplicated().any():
        raise ValueError('Duplicate image ids')
    overlap = set(frame.loc[frame.split == 'train', 'group_id']) & set(frame.loc[frame.split == 'validation', 'group_id'])
    if overlap:
        raise ValueError(f'Groups cross train/validation: {sorted(overlap)[:5]}')
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    tasks = [(r.image_id, r.path, str(images), args.size, i % 25 == 0) for i, r in enumerate(added.itertuples())]
    with ProcessPoolExecutor(args.workers) as pool:
        digests = dict(pool.map(export_one, tasks, chunksize=8))
    frame.loc[frame.image_id.isin(digests), 'png_sha256'] = frame.image_id.map(digests)
    frame.to_csv(args.output / 'manifest.csv', index=False)
    atomic_json(args.output / 'export_summary.json', {
        'schema_version': 1, 'size': args.size, 'images_total': len(frame), 'images_exported_here': len(added),
        'by_split_source': {f'{a}/{b}': int(n) for (a, b), n in frame.groupby(['split', 'source']).size().items()},
        'fdr26_cluster_validation_share': 0.25, 'fdr26_split_seed': args.seed,
        'reserved_fresh_tests_not_exported': ['mBRSET', 'BRSET'],
        'deployed_input_reproduction': 'exact (array_equal) on a fixed 1-in-25 sample'})
    print(frame.groupby(['split', 'source']).size())


if __name__ == '__main__':
    main()
