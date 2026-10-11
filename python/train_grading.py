#!/usr/bin/env python3
"""Reproducible grading experiments; never overwrite the deployed checkpoint.

Selection chooses the epoch. Calibration data is reserved and never read by the
training loop. All runs are development evidence; clinical promotion is separate.
"""
import argparse
import json
import os
from pathlib import Path
import random
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.netrai_model import NetrAIModel
from models.losses import FocalLoss, OrdinalLoss
from utils.artifacts import atomic_json, sha256_file
from utils.preprocessing import preprocess_for_model
from utils.validation import grades, image_ids


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_manifest(path):
    frame = pd.read_csv(path, dtype={'image_id': str, 'group_id': str})
    required = {'image_id', 'image_path', 'grade', 'group_id', 'split', 'sha256'}
    if not required.issubset(frame):
        raise ValueError(f'Manifest requires {sorted(required)}')
    image_ids(frame.image_id)
    frame['grade'] = grades(frame.grade)
    if frame[list(required)].isna().any().any():
        raise ValueError('Manifest has missing fields')
    if not set(frame.split).issubset({'fit', 'selection', 'calibration'}):
        raise ValueError('Training manifest cannot include final/test rows')
    if not {'fit', 'selection', 'calibration'}.issubset(set(frame.split)):
        raise ValueError('Separate fit, selection and calibration partitions are required')
    for key in ('group_id', 'sha256'):
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f'Cross-split {key} overlap')
    if (frame.groupby('sha256').grade.nunique() > 1).any():
        raise ValueError('Conflicting grades for identical image content; adjudicate or quarantine')
    frame['image_path'] = [str((path.parent / name).resolve()) for name in frame.image_path]
    for row in frame.itertuples():
        if sha256_file(row.image_path) != row.sha256:
            raise ValueError(f'Image content changed: {row.image_id}')
    return frame


class GradingDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode training image {row.image_id}; no black-image substitution')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            image = np.rot90(image, random.randrange(4))
            if random.random() < 0.5:
                image = image[:, ::-1]
        array = preprocess_for_model(np.ascontiguousarray(image), version='training-linear-v2')
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), int(row.grade)


def save_checkpoint(path, state):
    descriptor, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, 'wb') as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def rng_state(generator):
    numpy_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    name, keys, pos, gauss, cached = state['numpy']
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, gauss, cached))
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['loader'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--objective', choices=('ce', 'focal', 'ordinal'), default='focal')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--augmentation', choices=('none', 'dihedral'), default='dihedral')
    parser.add_argument('--pretrained', action='store_true', help='May download public ImageNet weights')
    parser.add_argument('--initial-checkpoint', type=Path, help='Explicit adaptation; not a fresh model')
    parser.add_argument('--resume', type=Path, help='Resume last.pt at an epoch boundary')
    parser.add_argument('--smoke', action='store_true', help='Tiny mechanics check, not an accuracy experiment')
    args = parser.parse_args(argv)
    if args.batch_size < 2 or args.epochs < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error('batch-size >=2, epochs >=1, workers >=0 and learning-rate >0 required')
    if args.pretrained and args.initial_checkpoint:
        parser.error('Choose pretrained initialization or explicit adaptation, not both')
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError('Use an empty experiment directory; evidence is not overwritten')
    frame = load_manifest(args.manifest.resolve())
    fit = frame.loc[frame.split.eq('fit')]
    selection = frame.loc[frame.split.eq('selection')]
    if args.smoke:
        fit, selection = fit.head(args.batch_size), selection.head(args.batch_size)
    counts = np.bincount(fit.grade, minlength=5)
    if not args.smoke and (counts == 0).any():
        raise ValueError('Fit partition must contain every grade')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    generator = torch.Generator().manual_seed(args.seed)
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu or a configured GPU environment')
    fit_loader = DataLoader(GradingDataset(fit, args.augmentation == 'dihedral'),
                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, worker_init_fn=seed_worker,
                            generator=generator, pin_memory=args.device == 'cuda')
    selection_loader = DataLoader(GradingDataset(selection), batch_size=args.batch_size,
                                  num_workers=args.workers, worker_init_fn=seed_worker,
                                  generator=generator)
    if len(fit_loader) == 0:
        raise ValueError('Fit set smaller than a full training batch')
    model = NetrAIModel(pretrained=args.pretrained).to(device)
    if args.initial_checkpoint:
        from utils.bundle import load_legacy_checkpoint
        model.load_state_dict(load_legacy_checkpoint(args.initial_checkpoint, device)['model_state_dict'])
    weights = torch.tensor(len(fit) / (5 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = (OrdinalLoss(weights, args.smoothing) if args.objective == 'ordinal' else
                 FocalLoss(weights, gamma=2 if args.objective == 'focal' else 0,
                           label_smoothing=args.smoothing)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key not in ('resume', 'output')}
    config.update(manifest_sha256=sha256_file(args.manifest), preprocessing_version='training-linear-v2',
                  checkpoint_selection='lowest unweighted selection NLL',
                  initialization_sha256=sha256_file(args.initial_checkpoint) if args.initial_checkpoint else None)
    source = Path(__file__).resolve().parent
    config['source_sha256'] = {name: sha256_file(source / name) for name in
                              ('train_grading.py', 'models/netrai_model.py',
                               'models/losses.py', 'utils/preprocessing.py')}
    config['runtime_versions'] = {'torch': str(torch.__version__), 'numpy': np.__version__,
                                  'opencv': cv2.__version__}
    history, start, best = [], 0, float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if state['config'] != config:
            raise ValueError('Resume configuration differs; start a separate experiment')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'], generator)
        start, best, history = state['epoch'] + 1, state['best_selection_nll'], state['history']
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'protocol.json', {
        **config, 'status': 'smoke_only' if args.smoke else 'development_experiment_not_deployed',
        'class_weights': weights.cpu().tolist(), 'torch_version': str(torch.__version__),
        'calibration_rows_reserved_not_used': int(frame.split.eq('calibration').sum()),
    })
    for epoch in range(start, args.epochs):
        epoch_started = time.monotonic()
        model.train()
        losses = []
        for images, targets in fit_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.device == 'cuda'):
                logits, referral = model(images)
                loss = 0.8 * criterion(logits, targets) + 0.2 * nn.functional.binary_cross_entropy_with_logits(
                    referral, (targets >= 2).float())
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        model.eval()
        total_nll, seen, output_rows = 0.0, 0, []
        with torch.inference_mode():
            for images, targets in selection_loader:
                logits, referral = model(images.to(device))
                total_nll += float(nn.functional.cross_entropy(logits, targets.to(device), reduction='sum'))
                seen += len(targets)
                output_rows.extend(torch.cat([logits, referral[:, None]], 1).cpu().tolist())
        nll = total_nll / seen
        if not np.isfinite(nll):
            raise ValueError('Nonfinite selection loss')
        scheduler.step()
        improved = nll < best
        best = min(best, nll)
        history.append({'epoch': epoch, 'fit_loss': float(np.mean(losses)), 'selection_nll': nll})
        state = {'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                 'rng': rng_state(generator), 'epoch': epoch, 'config': config,
                 'best_selection_nll': best, 'history': history}
        if improved:
            save_checkpoint(args.output / 'best.pt', {'model_state_dict': model.state_dict(),
                            'epoch': epoch, 'config': config, 'selection_nll': nll})
            output = selection[['image_id', 'grade', 'split']].reset_index(drop=True).copy()
            output[[f'logit_{i}' for i in range(6)]] = output_rows
            output.to_csv(args.output / 'selection_logits.csv', index=False)
        atomic_json(args.output / 'history.json', history)
        # Publish the resumable state last, after best-model evidence is complete.
        save_checkpoint(args.output / 'last.pt', state)
        epoch_seconds = time.monotonic() - epoch_started
        print(json.dumps({**history[-1], 'epoch_wall_seconds': round(epoch_seconds, 1),
                          'rough_five_run_gpu_hours_at_this_pace':
                          round(epoch_seconds * args.epochs * 5 / 3600, 1)}), flush=True)
        if args.smoke:
            break


if __name__ == '__main__':
    main()
