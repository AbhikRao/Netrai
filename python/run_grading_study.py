#!/usr/bin/env python3
"""Resume-safe GPU study of the corrected five-grade training objectives.

Three objectives share one content-grouped manifest and seed 42. The objective
with the lowest selection NLL is repeated at seeds 17 and 2026. All models start
from public ImageNet weights, never from the legacy APTOS checkpoint, because
that checkpoint has already seen images in the corrected selection/calibration
partitions. This script does not inspect calibration or external-test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import timm
import torch

from utils.artifacts import atomic_json, sha256_file


OBJECTIVES = ('ce', 'focal', 'ordinal')
SEEDS = (42, 17, 2026)
SOURCE_FILES = (
    'prepare_training_manifest.py', 'train_grading.py', 'run_grading_study.py',
    'models/netrai_model.py', 'models/losses.py', 'utils/artifacts.py',
    'utils/metrics.py', 'utils/preprocessing.py', 'utils/validation.py',
)
TRAINER_SOURCE_FILES = (
    'train_grading.py', 'models/netrai_model.py', 'models/losses.py',
    'utils/preprocessing.py',
)


def run_one(args, objective: str, seed: int, protocol: dict) -> tuple[dict, bool]:
    output = args.output / f'{objective}_seed{seed}'
    last = output / 'last.pt'
    history_path = output / 'history.json'
    best_path = output / 'best.pt'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if len(history) == args.epochs and best_path.is_file() and last.is_file():
            state = torch.load(last, map_location='cpu', weights_only=True)
            config = state['config']
            if (state['epoch'] != args.epochs - 1 or
                    config['manifest_sha256'] != protocol['manifest_sha256'] or
                    config['objective'] != objective or config['seed'] != seed or
                    config['epochs'] != args.epochs or config['batch_size'] != args.batch_size or
                    config['workers'] != args.workers or
                    config['learning_rate'] != args.learning_rate or
                    config['pretrained'] is not True or config['initial_checkpoint'] is not None or
                    config['runtime_versions'] != protocol['trainer_runtime_versions'] or
                    any(config['source_sha256'].get(name) != protocol['source_sha256'][name]
                        for name in TRAINER_SOURCE_FILES)):
                raise ValueError(f'Completed run identity changed: {output}')
            return ({'objective': objective, 'seed': seed, 'output': str(output),
                     'best_selection_nll': float(state['best_selection_nll']),
                     'best_checkpoint_sha256': sha256_file(best_path),
                     'epochs': args.epochs}, False)
    command = [sys.executable, str(Path(__file__).with_name('train_grading.py')),
               '--manifest', str(args.manifest), '--output', str(output),
               '--objective', objective, '--seed', str(seed), '--epochs', str(args.epochs),
               '--batch-size', str(args.batch_size), '--workers', str(args.workers),
               '--learning-rate', str(args.learning_rate), '--device', 'cuda',
               '--augmentation', 'dihedral', '--pretrained']
    if last.exists():
        command.extend(('--resume', str(last)))
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f'Partial run lacks a resumable checkpoint: {output}')
    print('Running', objective, seed, 'into', output, flush=True)
    subprocess.run(command, check=True)
    if not history_path.is_file() or len(json.loads(history_path.read_text())) != args.epochs:
        raise RuntimeError(f'Training did not finish all epochs: {output}')
    completed, _ = run_one(args, objective, seed, protocol)
    return completed, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-new-runs', type=int,
                        help='Stop after this many newly completed runs; rerun with unchanged settings to continue')
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    if (not args.manifest.is_file() or args.epochs < 1 or args.batch_size < 2 or
            args.workers < 0 or (args.max_new_runs is not None and args.max_new_runs < 1)):
        parser.error('Existing manifest, positive epochs, batch-size >=2, workers >=0 and positive run budget required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Enable a free GPU runtime before running the study.')
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'schema_version': 1, 'status': 'development_training_not_deployed',
                'manifest_sha256': sha256_file(args.manifest),
                'source_sha256': {name: sha256_file(Path(__file__).resolve().parent / name)
                                  for name in SOURCE_FILES},
                'trainer_runtime_versions': {'torch': str(torch.__version__),
                                             'numpy': np.__version__, 'opencv': cv2.__version__},
                'study_runtime_versions': {'timm': timm.__version__},
                'initialization': 'public ImageNet EfficientNet-B4; legacy APTOS checkpoint forbidden',
                'selection_rule': 'lowest NLL on grouped selection partition',
                'design': 'ce/focal/ordinal seed42, then winning objective seeds17/2026',
                'epochs': args.epochs, 'batch_size': args.batch_size,
                'workers': args.workers, 'learning_rate': args.learning_rate,
                'preprocessing': 'training-linear-v2',
                'calibration_and_external_test': 'never read during this model-selection study'}
    protocol_path = args.output / 'study_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Study configuration changed; use a new output directory')
    atomic_json(protocol_path, protocol)
    new_runs = 0
    screening = []
    for objective in OBJECTIVES:
        row, trained = run_one(args, objective, 42, protocol)
        screening.append(row)
        new_runs += int(trained)
        if trained and args.max_new_runs is not None and new_runs >= args.max_new_runs:
            print('Run budget reached. Save the study output and rerun this command to continue.', flush=True)
            return
    selected = min(screening, key=lambda row: (row['best_selection_nll'], OBJECTIVES.index(row['objective'])))
    finalists = [selected]
    for seed in SEEDS[1:]:
        row, trained = run_one(args, selected['objective'], seed, protocol)
        finalists.append(row)
        new_runs += int(trained)
        if (trained and seed != SEEDS[-1] and args.max_new_runs is not None and
                new_runs >= args.max_new_runs):
            print('Run budget reached. Save the study output and rerun this command to continue.', flush=True)
            return
    if not all(np.isfinite(row['best_selection_nll']) for row in screening + finalists):
        raise ValueError('Nonfinite selection NLL')
    chosen = min(finalists, key=lambda row: (row['best_selection_nll'], row['seed']))
    atomic_json(args.output / 'study_selection.json', {
        **protocol, 'screening_runs': screening, 'selected_objective': selected['objective'],
        'finalist_runs': finalists, 'selected_checkpoint': chosen,
        'interpretation': 'Selection-only development choice; calibration and external evaluation still required'})
    print('Selected development checkpoint:', chosen['output'], flush=True)


if __name__ == '__main__':
    main()
