#!/usr/bin/env python3
"""Paired input-contract diagnostic with fixed legacy weights and thresholds.

This does not tune a threshold, recalibrate, retrain or promote a model. Results
on previously exposed cohorts remain development diagnostics.
"""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from inference import predict_batch_with_details
from utils.artifacts import atomic_json, sha256_file
from utils.metrics import quadratic_weighted_kappa, referable_dr_metrics
from utils.validation import grades, image_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--id-column', default='id_code')
    parser.add_argument('--label-column', default='diagnosis')
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new diagnostic directory')
    torch.set_num_threads(2)
    labels = pd.read_csv(args.csv, dtype={args.id_column: str})
    ids = image_ids(labels[args.id_column]).to_numpy()
    truth = grades(labels[args.label_column])
    versions = ('legacy-area-v1', 'training-linear-v2')
    outputs = {version: [] for version in versions}
    started = time.perf_counter()
    for start in range(0, len(ids), args.batch_size):
        images = []
        for image_id in ids[start:start+args.batch_size]:
            path = args.image_dir / (image_id if Path(image_id).suffix else image_id+args.extension)
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f'Could not decode {path}')
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        for version in versions:
            batch = predict_batch_with_details(images, args.batch_size, preprocessing_version=version)
            for offset in range(len(images)):
                index = start+offset
                outputs[version].append({'image_id': ids[index], 'true_grade': int(truth[index]),
                    'predicted_grade': int(batch['grades'][offset]),
                    'referable_probability': float(batch['grade_probabilities'][offset, 2:].sum()),
                    'triage_action': str(batch['triage_actions'][offset]),
                    **{f'prob_grade_{i}': float(batch['grade_probabilities'][offset, i]) for i in range(5)}})
        print(f'{min(start+args.batch_size,len(ids))}/{len(ids)} paired images', flush=True)
    args.output.mkdir(parents=True)
    summary = {'status': 'paired_development_diagnostic_no_promotion',
               'source_labels_sha256': sha256_file(args.csv), 'images': len(ids),
               'runtime_seconds': time.perf_counter()-started, 'variants': {}}
    for version, rows in outputs.items():
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output / f'{version}.csv', index=False)
        summary['variants'][version] = {
            'accuracy': float(np.mean(frame.predicted_grade.to_numpy() == truth)),
            'qwk': quadratic_weighted_kappa(truth, frame.predicted_grade),
            'referable': referable_dr_metrics(truth, frame.predicted_grade),
            'auc': float(roc_auc_score(truth >= 2, frame.referable_probability))
                if len(np.unique(truth >= 2)) == 2 else None,
        }
    atomic_json(args.output / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
