#!/usr/bin/env python3
"""Round-3 lossless export at a higher input resolution (default 768 px).

Same ``legacy-area-v1`` contract as rounds 1–2 (crop > 7, INTER_AREA resize, PNG
before the Ben Graham blend with sigma 10); exactness against
``preprocess_for_model(size=...)`` is verified on a fixed 1-in-25 sample.
Split (exposed test sets are now development data):
train = APTOS, EyeQ train, IDRiD train, DeepDRiD train/validation/evaluation,
DDR train/valid/test; validation = EyeQ-test selection subset (3,000), IDRiD test,
Messidor-2. Reserved fresh tests (ODIR-5K, EyeQ test remainder) are never exported.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from export_training_images import build_manifest, crop_resize
from locked_evaluation import inventory, LABELS
from utils.artifacts import atomic_json
from utils.preprocessing import preprocess_for_model, IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parents[1]
DDR = ROOT / 'data/DDR/extracted/DDR-dataset/DR_grading'


def ddr(split):
    labels = pd.read_csv(DDR / f'{split}.txt', sep=' ', header=None, names=['file', 'grade'])
    labels = labels.loc[labels.grade <= 4]
    stems = labels.file.map(lambda f: Path(f).stem)
    return pd.DataFrame({'image_id': 'ddr_' + split + '_' + stems, 'path': [str(DDR / split / f) for f in labels.file],
                         'grade': labels.grade.astype(int), 'group_id': 'ddr_' + split + '_' + stems,
                         'source': 'ddr', 'split': 'train'})


def messidor():
    frame = inventory('messidor2')
    labels = pd.read_csv(LABELS)
    labels['key'] = labels.image_id.str.strip().str.lower()
    frame = frame.assign(key=frame.image_id.str.lower()).merge(labels.drop(columns='image_id'), on='key',
                                                               validate='one_to_one')
    frame = frame.loc[frame.adjudicated_gradable == 1]
    return pd.DataFrame({'image_id': 'messidor_' + frame.image_id.str.replace('.', '_', regex=False),
                         'path': [str(ROOT / 'data/messidor2/images' / i) for i in frame.image_id],
                         'grade': frame.adjudicated_dr_grade.astype(int), 'group_id': 'messidor_' + frame.exam_id,
                         'source': 'messidor', 'split': 'validation'})


def build(seed):
    base = build_manifest(3000, seed)
    base.loc[(base.source == 'deepdrid') & (base.split == 'validation'), 'split'] = 'train'  # exposed evaluation
    frame = pd.concat([base, ddr('train'), ddr('valid'), ddr('test'), messidor()], ignore_index=True)
    if frame.image_id.duplicated().any():
        raise ValueError('Duplicate image ids')
    overlap = set(frame.loc[frame.split == 'train', 'group_id']) & set(frame.loc[frame.split == 'validation', 'group_id'])
    if overlap:
        raise ValueError(f'Groups cross train/validation: {sorted(overlap)[:5]}')
    return frame


def export_one(task):
    image_id, path, output, size, verify = task
    data = Path(path).read_bytes()
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    small = crop_resize(rgb, size)
    if verify:
        blended = cv2.addWeighted(small, 4, cv2.GaussianBlur(small, (0, 0), 10), -4, 128)
        if not np.array_equal((blended.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD,
                              preprocess_for_model(rgb, size=size)):
            raise ValueError(f'Export does not reproduce the deployed input: {image_id}')
    target = Path(output) / f'{image_id}.png'
    existing = cv2.imread(str(target), cv2.IMREAD_COLOR) if target.exists() else None
    if existing is None or not np.array_equal(cv2.cvtColor(existing, cv2.COLOR_BGR2RGB), small):
        temporary = target.with_suffix('.tmp.png')
        cv2.imwrite(str(temporary), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        temporary.replace(target)
    return image_id, hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_improvement/train_export_768')
    parser.add_argument('--workers', type=int, default=7)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    frame = build(args.seed)
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    tasks = [(r.image_id, r.path, str(images), args.size, i % 25 == 0) for i, r in enumerate(frame.itertuples())]
    digests = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (image_id, png) in enumerate(pool.map(export_one, tasks, chunksize=8), 1):
            digests[image_id] = png
            if n % 1000 == 0:
                print(f'{n}/{len(tasks)}', flush=True)
    frame['png_sha256'] = frame.image_id.map(digests)
    frame.drop(columns='path').to_csv(args.output / 'manifest.csv', index=False)
    atomic_json(args.output / 'export_summary.json', {
        'schema_version': 1, 'size': args.size, 'images': len(frame),
        'by_split_source': {f'{a}/{b}': int(n) for (a, b), n in frame.groupby(['split', 'source']).size().items()},
        'reserved_fresh_tests_not_exported': ['ODIR-5K', 'EyeQ test remainder'],
        'deployed_input_reproduction': 'exact (array_equal) on a fixed 1-in-25 sample'})
    print(frame.groupby(['split', 'source']).size())


if __name__ == '__main__':
    main()
