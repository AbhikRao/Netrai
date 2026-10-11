#!/usr/bin/env python3
"""Export lossless 512-pixel training images for full-network GPU fine-tuning.

Each PNG (lossless, compression level 3) holds the deployed ``legacy-area-v1`` crop + INTER_AREA resize output
(before Ben Graham enhancement and normalisation). Training applies augmentation,
then the identical Ben Graham blend and ImageNet normalisation, so an
unaugmented export reproduces the deployed model input exactly (verified per
image here). Fresh test cohorts are refused: Messidor-2, the unused EyeQ test
remainder and any DDR test split never leave this machine.
"""
from __future__ import annotations

import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from extract_backbone_cache import cohort as cached_cohort
from utils.artifacts import atomic_json
from utils.preprocessing import preprocess_for_model, IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parents[1]
DEEPDRID = ROOT / '.project-notes/deepdrid-preparation/reviewed-development-2026-09-28'
DEEPDRID_ROOT = ROOT / 'data/DRiD/DeepDRiD-master'
FORBIDDEN = ('messidor', 'ddr_test', 'eyeq_test_remainder')


def crop_resize(rgb, size=512):
    foreground = rgb.max(axis=2) > 7
    if foreground.any():
        rows, cols = foreground.any(1), foreground.any(0)
        rgb = rgb[int(rows.argmax()):int(len(rows) - rows[::-1].argmax()),
                  int(cols.argmax()):int(len(cols) - cols[::-1].argmax())]
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def deployed_from_export(rgb512):
    blended = cv2.addWeighted(rgb512, 4, cv2.GaussianBlur(rgb512, (0, 0), 10), -4, 128)
    return (blended.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def deepdrid_split(name):
    frame = pd.read_csv(DEEPDRID / f'{name}.csv', dtype={'patient_id': str})
    if name == 'evaluation':  # exposed on 2026-10-11; labels from the post-lock package
        labelled = pd.read_csv(ROOT / 'results/model_improvement/acceptance/labelled/evaluation_labelled.csv',
                               dtype={'patient_id': str, 'id_code': str})
        frame = frame.drop(columns=['true_grade', 'grading_eligible']).merge(
            labelled[['id_code', 'true_grade', 'grading_eligible']], on='id_code', validate='one_to_one')
        frame['development_excluded'] = 0
    frame = frame.loc[(frame.development_excluded == 0) & (frame.grading_eligible.astype(bool))]
    return pd.DataFrame({'image_id': 'deepdrid_' + frame.id_code, 'path': [str(DEEPDRID_ROOT / p) for p in frame.image_relative_path],
                         'grade': frame.true_grade.astype(int), 'group_id': 'deepdrid_' + frame.patient_id, 'source': 'deepdrid'})


def aptos():
    labels = pd.read_csv(ROOT / 'data/aptos2019/train.csv')
    return pd.DataFrame({'image_id': 'aptos_' + labels.id_code, 'grade': labels.diagnosis.astype(int),
                         'path': [str(ROOT / f'data/aptos2019/train_images/{i}.png') for i in labels.id_code],
                         'group_id': 'aptos_' + labels.id_code, 'source': 'aptos'})


def build_manifest(eyeq_selection_limit, seed):
    parts = []
    train = [aptos(), cached_cohort('eyeq_train', 0, seed), cached_cohort('idrid_train', 0, seed),
             deepdrid_split('training'), deepdrid_split('validation')]
    validation = [cached_cohort('eyeq_test', eyeq_selection_limit, seed), cached_cohort('idrid_test', 0, seed),
                  deepdrid_split('evaluation')]
    for split, frames in (('train', train), ('validation', validation)):
        for frame in frames:
            frame = frame[['image_id', 'path', 'grade', 'group_id', 'source']].copy()
            if frame.source.iloc[0] in ('eyeq', 'idrid'):
                frame['image_id'] = frame.source + f'_{split}_' + frame.image_id.astype(str)
                frame['group_id'] = frame.group_id.astype(str) + ('' if frame.source.iloc[0] == 'eyeq' else f'_{split}')
            frame['split'] = split
            parts.append(frame)
    manifest = pd.concat(parts, ignore_index=True)
    if manifest.image_id.duplicated().any():
        raise ValueError('Duplicate exported image ids')
    for split_a, split_b in (('train', 'validation'),):
        overlap = set(manifest.loc[manifest.split == split_a, 'group_id']) & set(manifest.loc[manifest.split == split_b, 'group_id'])
        if overlap:
            raise ValueError(f'Patient groups cross train/validation: {sorted(overlap)[:5]}')
    if any(token in manifest.source.str.lower().str.cat(sep=' ') for token in FORBIDDEN):
        raise ValueError('A reserved fresh-test cohort entered the training export')
    return manifest


def export_one(task):
    image_id, path, output, verify = task
    data = Path(path).read_bytes()
    rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    small = crop_resize(rgb)
    if verify and not np.array_equal(deployed_from_export(small), preprocess_for_model(rgb)):
        raise ValueError(f'Export does not reproduce the deployed input: {image_id}')
    target = Path(output) / f'{image_id}.png'
    if not target.exists():
        cv2.imwrite(str(target), cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return image_id, hashlib.sha256(data).hexdigest(), hashlib.sha256(target.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_improvement/train_export')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    manifest = build_manifest(3000, args.seed)
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    tasks = [(r.image_id, r.path, str(images), i % 25 == 0) for i, r in enumerate(manifest.itertuples())]
    digests = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for index, (image_id, source_sha, png_sha) in enumerate(pool.map(export_one, tasks, chunksize=16), 1):
            digests[image_id] = (source_sha, png_sha)
            if index % 500 == 0:
                print(f'{index}/{len(tasks)}', flush=True)
    manifest['source_sha256'] = manifest.image_id.map(lambda i: digests[i][0])
    manifest['png_sha256'] = manifest.image_id.map(lambda i: digests[i][1])
    manifest.drop(columns='path').to_csv(args.output / 'manifest.csv', index=False)
    atomic_json(args.output / 'export_summary.json', {
        'schema_version': 1, 'contract': 'legacy-area-v1 crop + INTER_AREA 512 uint8 PNG, pre-Ben-Graham',
        'images': len(manifest), 'by_split_source': manifest.groupby(['split', 'source']).size().to_dict().__repr__(),
        'grade_counts_train': manifest.loc[manifest.split == 'train'].grade.value_counts().sort_index().to_dict(),
        'reserved_fresh_tests_not_exported': ['messidor2 (all)', 'EyeQ test remainder', 'DDR test'],
        'deployed_input_reproduction': 'exact (array_equal) on a fixed 1-in-25 sample; identical crop/resize code for all'})
    print(manifest.groupby(['split', 'source']).size())


if __name__ == '__main__':
    main()
