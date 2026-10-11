#!/usr/bin/env python3
"""Export DDR train/valid (gradable only) as extra round-2 training images.

Same lossless legacy-area-v1 crop/resize PNG contract as export_training_images.
The DDR **test** split is a reserved fresh test and is never read here.
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from export_training_images import export_one
from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DDR = ROOT / 'data/DDR/extracted/DDR-dataset/DR_grading'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_improvement/train_export_ddr')
    parser.add_argument('--workers', type=int, default=7)
    args = parser.parse_args()
    rows = []
    for split in ('train', 'valid'):
        labels = pd.read_csv(DDR / f'{split}.txt', sep=' ', header=None, names=['file', 'grade'])
        labels = labels.loc[labels.grade <= 4]  # grade 5 = ungradable
        for item in labels.itertuples():
            rows.append({'image_id': f'ddr_{split}_{Path(item.file).stem}', 'path': str(DDR / split / item.file),
                         'grade': int(item.grade), 'group_id': f'ddr_{split}_{Path(item.file).stem}',
                         'source': 'ddr', 'split': 'train'})
    frame = pd.DataFrame(rows)
    images = args.output / 'images'
    images.mkdir(parents=True, exist_ok=True)
    tasks = [(r.image_id, r.path, str(images), i % 25 == 0) for i, r in enumerate(frame.itertuples())]
    digests = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for n, (image_id, src, png) in enumerate(pool.map(export_one, tasks, chunksize=16), 1):
            digests[image_id] = (src, png)
            if n % 1000 == 0:
                print(f'{n}/{len(tasks)}', flush=True)
    frame['source_sha256'] = frame.image_id.map(lambda i: digests[i][0])
    frame['png_sha256'] = frame.image_id.map(lambda i: digests[i][1])
    frame.drop(columns='path').to_csv(args.output / 'manifest_ddr.csv', index=False)
    atomic_json(args.output / 'export_summary.json', {'images': len(frame),
                'grade_counts': frame.grade.value_counts().sort_index().to_dict(),
                'note': 'DDR train+valid gradable; DDR test reserved and untouched'})
    print(frame.grade.value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
