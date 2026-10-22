#!/usr/bin/env python3
"""Manifest-v6 trainer (research branch): hardened, resumable, leave-one-dataset-out capable.

Differences from ``train_full_finetune.py`` (which stays untouched so rounds 2-5 remain reproducible):

* trains on ``train`` rows of manifest v6; selects on ``td`` + ``validation`` rows (``--select-on``);
* ``--exclude-source SRC`` withholds every row of a source from training *and* selection and scores it
  each epoch as a held-out leave-one-dataset-out fold (never used for selection);
* ``--init imagenet|random|<state_dict path>``; ``imagenet`` avoids carrying APTOS/legacy exposure;
* ``--input-size`` downsizes the exported 768-px PNGs on the fly with INTER_AREA (research contract);
* ``--loss`` ce | ce_nosmooth | emd | ordinal_ls | gce, ``--ema`` weight averaging, ``--source-weight``;
* every epoch writes history, ``last.pt``, selection and held-out outputs; ``best.pt`` improves only when
  the selection value does; a wall-clock guard stops cleanly before Kaggle's 12-hour limit;
* augmentation is seeded by (seed, epoch, sample position), so it is reproducible for any worker count;
* the loss scalar is ``.item()``-ed (no autograd-scalar warning) and ``run_info.json`` / ``compute.json``
  record configuration, data counts, versions and compute used.

Outputs are compatible with ``fit_validation_policy.py`` (``validation_rows.csv`` +
``best_validation_outputs.npy`` describe the selection rows).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import build_manifest_v6 as bm
from models.grade_losses import LOSSES, build_grade_loss
from models.netrai_model import NetrAIModel
from train_full_finetune import Rows, auc, augment, deployed_input, evaluate, sampling_weights, selection_value
from utils.artifacts import atomic_json, sha256_file

SELECT_SPLITS = {'td': ['td'], 'validation': ['validation'], 'both': ['td', 'validation']}


# ----------------------------------------------------------------------------- data


def resolve_selection_exclusions(init: str, exclude_sources, explicit=None) -> list:
    """Sources whose rows must not drive selection. A checkpoint initialisation (e.g. the legacy APTOS model)
    has already trained on part of APTOS, so APTOS selection rows would be optimistic: they are removed by default."""
    if init not in ('imagenet', 'random') and 'aptos' in exclude_sources:
        raise ValueError('--exclude-source aptos needs --init imagenet: the legacy initialisation was trained on APTOS')
    if explicit is not None:
        return sorted(set(explicit))
    return ['aptos'] if init not in ('imagenet', 'random') else []


def split_frames(manifest: pd.DataFrame, exclude_sources=(), select_on: str = 'both', selection_exclude=()):
    """(train, selection, heldout) frames. Held-out sources appear in none of the other two.
    ``selection_exclude`` sources stay in train but never enter the selection frame."""
    if select_on not in SELECT_SPLITS:
        raise ValueError(f'--select-on must be one of {sorted(SELECT_SPLITS)}')
    unknown = set(exclude_sources) - set(manifest.source)
    if unknown:
        raise ValueError(f'Unknown source(s) to exclude: {sorted(unknown)}')
    live = manifest[manifest.split != 'excluded']
    heldout = live[live.source.isin(exclude_sources)]
    rest = live[~live.source.isin(exclude_sources)]
    train = rest[rest.split == 'train'].reset_index(drop=True)
    selection = rest[rest.split.isin(SELECT_SPLITS[select_on]) & ~rest.source.isin(list(selection_exclude))].reset_index(drop=True)
    heldout = heldout.reset_index(drop=True)
    if train.empty or selection.empty:
        raise ValueError('Training or selection set is empty')
    for name, other in (('selection', selection), ('heldout', heldout)):
        if set(train.image_id) & set(other.image_id):
            raise AssertionError(f'train and {name} share images')
        if set(train.group_id) & set(other.group_id):
            raise AssertionError(f'train and {name} share groups')
    if len(heldout) and (set(selection.image_id) & set(heldout.image_id)):
        raise AssertionError('selection and held-out sets share images')
    return train, selection, heldout


def apply_source_multipliers(weights: np.ndarray, frame: pd.DataFrame, multipliers: dict) -> np.ndarray:
    """Scale the sampling weight of whole sources (e.g. down-weight a noisily labelled source)."""
    unknown = set(multipliers) - set(frame.source)
    if unknown:
        raise ValueError(f'Unknown source(s) in --source-weight: {sorted(unknown)}')
    scaled = weights * frame.source.map(lambda s: multipliers.get(s, 1.0)).to_numpy()
    return scaled / scaled.sum()


class RowsV6(Rows):
    """Rows with optional on-the-fly INTER_AREA downsizing and position-seeded augmentation."""

    def __init__(self, frame, roots, train, seed, glare_prob=0.0, input_size=None):
        super().__init__(frame, roots, train, seed, glare_prob)
        self.input_size = input_size
        self.order = None  # sampled row indices for the current epoch (training)

    def set_order(self, order, epoch):
        self.order, self.epoch = np.asarray(order), epoch

    def __len__(self):
        return len(self.frame) if self.order is None else len(self.order)

    def __getitem__(self, position):
        index = int(position if self.order is None else self.order[position])
        row = self.frame.iloc[index]
        bgr = cv2.imread(str(self.path(row.image_id)), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'Unreadable training image {row.image_id}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.input_size and rgb.shape[0] != self.input_size:
            rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, int(position)))
            rgb = augment(rgb, rng, self.glare_prob)
        return torch.from_numpy(deployed_input(rgb).transpose(2, 0, 1).copy()), int(row.grade)


def prefetch_files(paths, threads: int) -> dict:
    """Read every file once with many threads. Network-mounted datasets (Kaggle) have ~1 s latency per file and
    scale almost linearly with parallelism, so this turns the first epoch from I/O-bound to warm-cache."""
    from concurrent.futures import ThreadPoolExecutor

    def read(path):
        with open(path, 'rb') as stream:
            return len(stream.read())

    started = time.time()
    unique = sorted({str(p) for p in paths})
    with ThreadPoolExecutor(max(1, threads)) as pool:
        total = sum(pool.map(read, unique))
    seconds = max(time.time() - started, 1e-6)
    return {'files': len(unique), 'megabytes': round(total / 1e6, 1), 'seconds': round(seconds, 1),
            'files_per_second': round(len(unique) / seconds, 1)}


# ----------------------------------------------------------------------------- model


def build_model(init: str, freeze_blocks: int = 0):
    """EfficientNet-B4 dual head. ``imagenet`` loads timm weights (needs network); ``random`` is for tests."""
    model = NetrAIModel('efficientnet_b4', 5, pretrained=(init == 'imagenet'))
    if init not in ('imagenet', 'random'):
        model.load_state_dict(torch.load(init, map_location='cpu', weights_only=True))
    frozen = []
    if freeze_blocks:
        frozen = [model.backbone.conv_stem, model.backbone.bn1] + list(model.backbone.blocks[:freeze_blocks])
        for module in frozen:
            for parameter in module.parameters():
                parameter.requires_grad = False
    return model, frozen


class EMA:
    """Exponential moving average of weights and buffers, with warm-up of the decay."""

    def __init__(self, model: nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError('EMA decay must be in (0, 1)')
        self.decay, self.updates = decay, 0
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        for key, value in self.shadow.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach().to(value.device), alpha=1 - decay)
            else:
                value.copy_(source[key])


def past_deadline(started: float, max_hours: float, now: float | None = None) -> bool:
    return (time.time() if now is None else now) - started > max_hours * 3600


# ----------------------------------------------------------------------------- evaluation


def source_metrics(frame: pd.DataFrame, outputs: np.ndarray) -> dict:
    logits = outputs[:, :5] - outputs[:, :5].max(1, keepdims=True)
    probabilities = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    score = probabilities @ np.arange(5)
    out = {}
    for source, part in frame.groupby('source'):
        index = part.index.to_numpy()
        grades = part.grade.to_numpy()
        if (grades >= 2).any() and (grades < 2).any():
            out[source] = {'images': len(part), 'auc': auc(score[index], grades),
                           'selection_value': selection_value(score[index], grades)}
        else:
            out[source] = {'images': len(part), 'auc': float('nan'), 'selection_value': float('nan')}
    return out


def mean_selection(per_source: dict) -> float:
    values = [v['selection_value'] for v in per_source.values() if not math.isnan(v['selection_value'])]
    return float(np.mean(values)) if values else float('nan')


# ----------------------------------------------------------------------------- main


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True, help='manifest_v6.csv')
    parser.add_argument('--roots', type=Path, nargs='+', required=True, help='folders holding <image_id>.png')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init', default='imagenet', help='imagenet | random | path to a state_dict')
    parser.add_argument('--exclude-source', action='append', default=[], help='held-out LODO source (repeatable)')
    parser.add_argument('--selection-exclude-source', action='append', default=None,
                        help='sources kept out of selection (default: aptos when --init is a checkpoint)')
    parser.add_argument('--select-on', choices=sorted(SELECT_SPLITS), default='both')
    parser.add_argument('--input-size', type=int, default=0, help='downsize from the export size (0 = keep)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--freeze-blocks', type=int, default=0)
    parser.add_argument('--class-power', type=float, default=0.5)
    parser.add_argument('--source-power', type=float, default=0.5)
    parser.add_argument('--source-weight', action='append', default=[], metavar='SRC=X',
                        help='multiply the sampling weight of a source (repeatable)')
    parser.add_argument('--loss', choices=LOSSES, default='ce')
    parser.add_argument('--ordinal-sigma', type=float, default=0.7)
    parser.add_argument('--gce-q', type=float, default=0.7)
    parser.add_argument('--grade-weight', type=float, default=0.8)
    parser.add_argument('--referral-weight', type=float, default=0.2)
    parser.add_argument('--ema', type=float, default=0.0, help='EMA decay (0 disables); evaluation/checkpoints use EMA')
    parser.add_argument('--glare-prob', type=float, default=0.0)
    parser.add_argument('--prefetch-threads', type=int, default=0, help='warm a slow (network) image mount with this many reader threads')
    parser.add_argument('--max-train-per-epoch', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-hours', type=float, default=11.0, help='wall-clock guard (Kaggle limit is 12 h)')
    parser.add_argument('--limit-train', type=int, default=0, help='smoke/pilot only: random subset of train rows')
    parser.add_argument('--limit-select', type=int, default=0, help='smoke/pilot only: random subset of selection rows')
    parser.add_argument('--limit-heldout', type=int, default=0, help='smoke/pilot only: random subset of held-out rows')
    return parser.parse_args(argv)


def git_commit() -> str:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=10).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def subsample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if not limit or limit >= len(frame):
        return frame
    return frame.sample(n=limit, random_state=seed).sort_index().reset_index(drop=True)


def run(args) -> dict:
    started = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    bm.check_manifest(manifest)
    selection_exclude = resolve_selection_exclusions(args.init, args.exclude_source, args.selection_exclude_source)
    args.selection_exclude_source = selection_exclude
    train, selection, heldout = split_frames(manifest, args.exclude_source, args.select_on, selection_exclude)
    train = subsample(train, args.limit_train, args.seed)
    selection = subsample(selection, args.limit_select, args.seed)
    heldout = subsample(heldout, args.limit_heldout, args.seed)
    multipliers = {}
    for item in args.source_weight:
        name, _, value = item.partition('=')
        multipliers[name] = float(value)
    model, frozen = build_model(args.init, args.freeze_blocks)
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    per_epoch = args.max_train_per_epoch or len(train)
    steps = args.epochs * math.ceil(per_epoch / args.batch_size)
    warmup = max(1, steps // 30)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = EMA(model, args.ema) if args.ema else None
    grade_loss = build_grade_loss(args.loss, sigma=args.ordinal_sigma, q=args.gce_q)
    start_epoch, best, history = 0, -1.0, []
    last = args.output / 'last.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        if ema is not None and checkpoint.get('ema') is not None:
            ema.shadow.load_state_dict(checkpoint['ema']); ema.updates = checkpoint['ema_updates']
        start_epoch, best, history = checkpoint['epoch'] + 1, checkpoint['best'], checkpoint['history']
        print(f'Resumed after epoch {checkpoint["epoch"]}', flush=True)
    weights = apply_source_multipliers(sampling_weights(train, args.class_power, args.source_power), train, multipliers)
    size = args.input_size or None
    train_set = RowsV6(train, args.roots, True, args.seed, args.glare_prob, size)
    eval_workers = args.workers
    timeout = 600 if eval_workers else 0
    select_loader = DataLoader(RowsV6(selection, args.roots, False, args.seed, 0.0, size), batch_size=16,
                               num_workers=eval_workers, timeout=timeout)
    held_loader = (DataLoader(RowsV6(heldout, args.roots, False, args.seed, 0.0, size), batch_size=16,
                              num_workers=eval_workers, timeout=timeout) if len(heldout) else None)
    selection[['image_id', 'grade', 'source', 'group_id']].to_csv(args.output / 'validation_rows.csv', index=False)
    if len(heldout):
        heldout[['image_id', 'grade', 'source', 'group_id']].to_csv(args.output / 'heldout_rows.csv', index=False)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    atomic_json(args.output / 'run_info.json', {
        'schema_version': 1, 'config': config, 'git_commit': git_commit(), 'device': device.type,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'torch': torch.__version__, 'python': sys.version.split()[0],
        'manifest_sha256': sha256_file(args.manifest),
        'counts': {'train': len(train), 'selection': len(selection), 'heldout': len(heldout)},
        'train_by_source': train.source.value_counts().to_dict(),
        'selection_by_source': selection.source.value_counts().to_dict(),
        'heldout_by_source': heldout.source.value_counts().to_dict() if len(heldout) else {},
        'train_grade_counts': np.bincount(train.grade, minlength=5).tolist(),
        'source_multipliers': multipliers, 'evidence_tiers': {'selection': 'TD', 'heldout': 'TL'}})
    images_seen, truncated = 0, False
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        rng = np.random.default_rng((args.seed, epoch))
        train_set.set_order(rng.choice(len(train), size=per_epoch, replace=True, p=weights), epoch)
        if args.prefetch_threads:
            wanted = [train_set.path(train.image_id.iloc[i]) for i in np.unique(train_set.order)]
            print('prefetch', prefetch_files(wanted, args.prefetch_threads), flush=True)
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            drop_last=True, persistent_workers=False)
        model.train()
        for module in frozen:
            module.eval()
        losses = []
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                grade, referral = model(x)
                loss = (args.grade_weight * grade_loss(grade.float(), y)
                        + args.referral_weight * nn.functional.binary_cross_entropy_with_logits(
                            referral.float(), (y >= 2).float()))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            if ema is not None:
                ema.update(model)
            losses.append(loss.item())
            images_seen += len(y)
            if step % 200 == 0:
                print(f'epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-200:]):.4f} '
                      f'{(step + 1) * args.batch_size / (time.time() - epoch_started):.1f} img/s', flush=True)
            if step % 50 == 0 and past_deadline(started, args.max_hours):
                truncated = True
                print('Wall-clock guard reached: finishing this epoch early', flush=True)
                break
        evaluated = ema.shadow if ema is not None else model
        if args.prefetch_threads:  # training reads evicted the evaluation files from the page cache: re-warm them
            wanted = [select_loader.dataset.path(i) for i in selection.image_id]
            wanted += [held_loader.dataset.path(i) for i in heldout.image_id] if held_loader is not None else []
            print('prefetch (evaluation)', prefetch_files(wanted, args.prefetch_threads), flush=True)
        outputs = evaluate(evaluated, select_loader, device)
        per_source = source_metrics(selection, outputs)
        value = mean_selection(per_source)
        record = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else float('nan'),
                  'selection_value': value, 'per_source': per_source,
                  'seconds': round(time.time() - epoch_started, 1), 'truncated': truncated, 'ema': ema is not None}
        np.save(args.output / f'selection_outputs_epoch{epoch}.npy', outputs)
        if held_loader is not None:  # leave-one-dataset-out fold: scored every epoch, never used for selection
            held_outputs = evaluate(evaluated, held_loader, device)
            np.save(args.output / f'heldout_outputs_epoch{epoch}.npy', held_outputs)
            record['heldout_per_source'] = source_metrics(heldout, held_outputs)
        history.append(record)
        print(json.dumps(record), flush=True)
        if not math.isnan(value) and value > best:
            best = value
            torch.save({'model_state_dict': evaluated.state_dict(), 'epoch': epoch, 'selection_value': value,
                        'config': config}, args.output / 'best.pt')
            np.save(args.output / 'best_validation_outputs.npy', outputs)
            if held_loader is not None:
                np.save(args.output / 'best_heldout_outputs.npy', held_outputs)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best,
                    'history': history, 'ema': ema.shadow.state_dict() if ema is not None else None,
                    'ema_updates': ema.updates if ema is not None else 0}, last)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        if truncated:
            break
    wall = time.time() - started
    atomic_json(args.output / 'compute.json', {
        'wall_seconds': round(wall, 1), 'wall_hours': round(wall / 3600, 3), 'device': device.type,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
        'images_trained_this_session': images_seen,
        'images_per_second': round(images_seen / wall, 2) if wall else None,
        'epochs_completed': len(history), 'truncated_by_wall_clock': truncated})
    print('done; best selection value', best, flush=True)
    return {'best': best, 'history': history, 'truncated': truncated}


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
