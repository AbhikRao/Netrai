#!/usr/bin/env python3
"""Round-4 additions to the 768-px export (plan MODEL_IMPROVEMENT_PLAN_v4.md).

Adds ODIR-5K DR-endpoint eyes (keyword-mapped grades; patients split 75/25 into
train/validation) and the EyeQ test remainder (train) to the round-3 manifest.
Only the new images are exported (same ``legacy-area-v1`` lossless PNG contract);
the round-3 images are reused from ``train_export_768``. Both added sources are
already exposed; the round-4 fresh tests (mBRSET, BRSET) are never exported.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from export_round3 import build, export_one
from locked_evaluation import EYEQ, ODIR, ODIR_LABELS, inventory, odir_grade
from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def odir(seed):
    frame = inventory('odir')  # excludes the 108 DDR-overlap patients
    table = pd.read_excel(ODIR_LABELS)
    per_eye = {}
    for row in table.itertuples(index=False):
        per_eye[str(row[3]).strip()] = odir_grade(row[5])
        per_eye[str(row[4]).strip()] = odir_grade(row[6])
    grade = frame.image_id.map(per_eye)
    keep = grade.map(lambda g: isinstance(g, (int, np.integer)))
    frame, grade = frame.loc[keep], grade.loc[keep].astype(int)
    patients = frame.exam_id.drop_duplicates().sample(frac=1, random_state=seed)
    validation = set(patients.iloc[:round(0.25 * len(patients))])
    return pd.DataFrame({'image_id': 'odir_' + frame.image_id.str.replace('.jpg', '', regex=False),
                         'path': [str(ODIR / 'Training Images' / i) for i in frame.image_id],
                         'grade': grade.values, 'group_id': frame.exam_id.values, 'source': 'odir',
                         'split': ['validation' if p in validation else 'train' for p in frame.exam_id]})


def eyeq_remainder():
    frame = inventory('eyeq_remainder')
    labels = pd.read_csv(EYEQ / 'data/Label_EyeQ_test.csv', usecols=['image', 'DR_grade'])
    frame = frame.merge(labels.rename(columns={'image': 'label_image'}), on='label_image', validate='one_to_one')
    return pd.DataFrame({'image_id': 'eyeq_rem_' + frame.image_id.map(lambda i: Path(i).stem),
                         'path': [str(EYEQ / 'images/test' / i) for i in frame.image_id],
                         'grade': frame.DR_grade.astype(int), 'group_id': frame.exam_id.values,
                         'source': 'eyeq', 'split': 'train'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--round3', type=Path, default=ROOT / 'results/model_improvement/train_export_768')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_improvement/train_export_768_r4')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    base = pd.read_csv(args.round3 / 'manifest.csv')
    if len(base) != len(build(2026)):
        raise ValueError('Round-3 manifest does not match its builder')
    added = pd.concat([odir(args.seed), eyeq_remainder()], ignore_index=True)
    frame = pd.concat([base, added.drop(columns='path')], ignore_index=True)
    if frame.image_id.duplicated().any():
        raise ValueError('Duplicate image ids')
    overlap = set(frame.loc[frame.split == 'train', 'group_id']) & set(frame.loc[frame.split == 'validation', 'group_id'])
    if overlap:
        raise ValueError(f'Groups cross train/validation: {sorted(overlap)[:5]}')
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    tasks = [(r.image_id, r.path, str(images), args.size, i % 25 == 0) for i, r in enumerate(added.itertuples())]
    digests = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (image_id, png) in enumerate(pool.map(export_one, tasks, chunksize=8), 1):
            digests[image_id] = png
            if n % 1000 == 0:
                print(f'{n}/{len(tasks)}', flush=True)
    frame.loc[frame.image_id.isin(digests), 'png_sha256'] = frame.image_id.map(digests)
    frame.to_csv(args.output / 'manifest.csv', index=False)
    atomic_json(args.output / 'export_summary.json', {
        'schema_version': 1, 'size': args.size, 'images_total': len(frame), 'images_exported_here': len(added),
        'by_split_source': {f'{a}/{b}': int(n) for (a, b), n in frame.groupby(['split', 'source']).size().items()},
        'odir_patients_validation_share': 0.25, 'odir_split_seed': args.seed,
        'reserved_fresh_tests_not_exported': ['mBRSET', 'BRSET'],
        'deployed_input_reproduction': 'exact (array_equal) on a fixed 1-in-25 sample'})
    print(frame.groupby(['split', 'source']).size())


if __name__ == '__main__':
    main()
