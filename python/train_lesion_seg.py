#!/usr/bin/env python3
"""Lesion-segmentation trainer (X8). Train on tiles, validate on whole images, resumable, wall-clock guarded.

Input: an exported dataset from ``lesion_data.export_dataset`` (DDR train/valid + IDRiD training only). Selection
uses the ``valid`` rows of that manifest, which were split **by vessel cluster** (same retina never on both sides).
The reserved DDR lesion test masks are not reachable from this script.

Selection value = mean pixel-level AUPR over lesion types that occur in the validation images. Per-image
histograms of the best epoch are saved (``valid_histograms_best.npz``) so confidence intervals and thresholds can be
computed later without re-running inference. Dice thresholds are chosen on validation only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import lesion_data as ld
import lesion_metrics as lm
from models.lesion_unet import LesionNet, lesion_loss
from utils.artifacts import atomic_json


def pad_to_32(image: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)))
    return image, ph, pw


@torch.inference_mode()
def predict_full(model, image: np.ndarray, device, tile: int = 1024, overlap: int = 128, enhance: str = 'ben',
                 sigma: float | None = None) -> np.ndarray:
    """Sigmoid probability maps (H, W, L) for a whole RGB image, using overlapping windows of side ``tile``."""
    h, w = image.shape[:2]
    sigma = sigma or max(2.0, max(h, w) / 100)
    probs = None
    counts = np.zeros((h, w, 1), np.float32)
    step = max(32, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step)) or [0]
    xs = list(range(0, max(1, w - overlap), step)) or [0]
    for top in ys:
        for left in xs:
            window = image[top:top + tile, left:left + tile]
            if enhance == 'ben':
                window = ld.ben_graham(np.ascontiguousarray(window), sigma)
            padded, ph, pw = pad_to_32(window)
            x = ld.to_tensor(padded)[None].to(device)
            logits = model(x)[0, :, :window.shape[0], :window.shape[1]]
            p = torch.sigmoid(logits).permute(1, 2, 0).float().cpu().numpy()
            if probs is None:
                probs = np.zeros((h, w, p.shape[2]), np.float32)
            probs[top:top + window.shape[0], left:left + window.shape[1]] += p
            counts[top:top + window.shape[0], left:left + window.shape[1]] += 1
    return probs / np.maximum(counts, 1)


def evaluate_valid(model, root: Path, frame: pd.DataFrame, device, long_side, tile, overlap, enhance) -> lm.LesionSet:
    model.eval()
    lesion_set = lm.LesionSet()
    loader = ld.LesionTiles(frame, root, tile=32, long_side=long_side)  # only its .load() is used
    for image_id in frame.image_id:
        image, masks = loader.load(image_id)
        probs = predict_full(model, image, device, tile, overlap, enhance)
        lesion_set.add(probs, masks, valid=image.max(axis=2) > ld.FOREGROUND)
    return lesion_set


def selection_value(summary: dict) -> float:
    values = [v['aupr'] for v in summary.values() if not math.isnan(v['aupr'])]
    return float(np.mean(values)) if values else float('nan')


def positive_weights(frame: pd.DataFrame, mode: str, cap: float = 100.0) -> torch.Tensor:
    if mode == 'none':
        return torch.ones(len(ld.LESIONS))
    pixels = frame.height.astype(float) * frame.width.astype(float)
    weights = []
    for lesion in ld.LESIONS:
        ann = frame[f'annotated_{lesion}']
        positive = float(frame.loc[ann, f'pixels_{lesion}'].sum())
        total = float(pixels[ann].sum())
        weights.append(min(cap, max(1.0, math.sqrt((total - positive) / positive))) if positive else 1.0)
    return torch.tensor(weights, dtype=torch.float32)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, required=True, help='export_dataset output directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arch', choices=('unet', 'fpn'), default='unet')
    parser.add_argument('--encoder', default='efficientnet_b2')
    parser.add_argument('--no-pretrained', action='store_true', help='random init (tests); default loads ImageNet weights')
    parser.add_argument('--long-side', type=int, default=0, help='train/eval at this long side (0 = as exported)')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--eval-tile', type=int, default=1024)
    parser.add_argument('--overlap', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--samples-per-epoch', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--lesion-prob', type=float, default=0.7)
    parser.add_argument('--enhance', choices=('ben', 'none'), default='ben')
    parser.add_argument('--pos-weight', choices=('auto', 'none'), default='auto')
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-hours', type=float, default=11.0)
    parser.add_argument('--max-valid-images', type=int, default=0, help='smoke/pilot only')
    return parser.parse_args(argv)


def run(args) -> dict:
    started = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.dataset / 'manifest.csv')
    train = manifest[manifest.split == 'train'].reset_index(drop=True)
    valid = manifest[manifest.split == 'valid'].reset_index(drop=True)
    if args.max_valid_images:
        valid = valid.iloc[:args.max_valid_images].reset_index(drop=True)
    if train.empty or valid.empty:
        raise ValueError('Empty train or valid set')
    if set(train.cluster) & set(valid.cluster):
        raise AssertionError('A vessel cluster spans train and valid')
    long_side = args.long_side or None
    model = LesionNet(args.arch, args.encoder, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * math.ceil(args.samples_per_epoch / args.batch_size)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    pos_weight = positive_weights(train, args.pos_weight).to(device)
    dataset = ld.LesionTiles(train, args.dataset, tile=args.tile, long_side=long_side, seed=args.seed,
                             lesion_prob=args.lesion_prob, enhance=args.enhance, length=args.samples_per_epoch)
    start_epoch, best, history = 0, -1.0, []
    last = args.output / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler']); scaler.load_state_dict(state['scaler'])
        start_epoch, best, history = state['epoch'] + 1, state['best'], state['history']
        print(f'Resumed after epoch {state["epoch"]}', flush=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    atomic_json(args.output / 'run_info.json', {
        'schema_version': 1, 'config': config, 'device': device.type,
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None, 'torch': torch.__version__,
        'python': sys.version.split()[0], 'counts': {'train': len(train), 'valid': len(valid)},
        'by_dataset': train.dataset.value_counts().to_dict(), 'pos_weight': [float(v) for v in pos_weight.cpu()],
        'evidence_tiers': {'valid': 'TD (cluster-split development data)'}, 'test_labels_read': False})
    truncated, images_seen = False, 0
    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=True)
        model.train()
        losses = []
        for step, (x, y, annotated) in enumerate(loader):
            x, y, annotated = x.to(device), y.to(device), annotated.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = lesion_loss(logits.float(), y, annotated, pos_weight, args.dice_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(loss.item())
            images_seen += len(y)
            if step % 50 == 0:
                print(f'epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-50:]):.4f}', flush=True)
            if step % 20 == 0 and time.time() - started > args.max_hours * 3600:
                truncated = True
                break
        lesion_set = evaluate_valid(model, args.dataset, valid, device, long_side, args.eval_tile, args.overlap, args.enhance)
        summary = lesion_set.summary()
        value = selection_value(summary)
        record = {'epoch': epoch, 'loss': float(np.mean(losses)) if losses else float('nan'), 'selection_aupr': value,
                  'valid': summary, 'seconds': round(time.time() - epoch_started, 1), 'truncated': truncated}
        history.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != 'valid'}), flush=True)
        if not math.isnan(value) and value > best:
            best = value
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'selection_aupr': value, 'config': config},
                       args.output / 'best.pt')
            np.savez_compressed(args.output / 'valid_histograms_best.npz', histograms=np.stack(lesion_set.per_image).astype(np.float32),
                                image_ids=np.array(valid.image_id), has_lesion=np.stack(lesion_set.has_lesion))
            atomic_json(args.output / 'valid_summary_best.json', {
                'epoch': epoch, 'summary': summary, 'thresholds_chosen_on_valid':
                {name: summary[name]['threshold_chosen_on_this_set'] for name in lm.LESIONS}})
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'history': history}, last)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
        if truncated:
            break
    wall = time.time() - started
    atomic_json(args.output / 'compute.json', {'wall_seconds': round(wall, 1), 'wall_hours': round(wall / 3600, 3),
                                               'device': device.type, 'tiles_trained_this_session': images_seen,
                                               'epochs_completed': len(history), 'truncated_by_wall_clock': truncated})
    print('done; best selection AUPR', best, flush=True)
    return {'best': best, 'history': history, 'truncated': truncated}


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
