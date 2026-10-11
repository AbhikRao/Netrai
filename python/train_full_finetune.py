#!/usr/bin/env python3
"""Full-network multi-source fine-tuning of the deployed grading model (GPU).

Runs on CUDA (Kaggle) or Apple MPS. Inputs are lossless 512-pixel PNG exports
from ``export_training_images.py`` (the deployed ``legacy-area-v1`` crop/resize,
before Ben Graham). Each sample is augmented, then receives the exact deployed
Ben Graham blend and ImageNet normalisation, so unaugmented validation inputs
equal the deployed MATLAB/Python model input. Architecture and output schema
are unchanged (EfficientNet-B4, 5 grade logits + referral logit), so the result
exports to the existing ONNX/MATLAB contract.

Fresh test cohorts (Messidor-2, the EyeQ test remainder, DDR test) are never
present in the export; epochs are chosen on the validation split only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.netrai_model import NetrAIModel

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def deployed_input(rgb512):
    blended = cv2.addWeighted(rgb512, 4, cv2.GaussianBlur(rgb512, (0, 0), 10), -4, 128)
    return (blended.astype(np.float32) / 255.0 - MEAN) / STD


def augment(rgb, rng):
    """Acquisition-like augmentation applied before the deployed enhancement."""
    h, w = rgb.shape[:2]
    angle = rng.uniform(0, 360)
    scale = rng.uniform(0.88, 1.12)
    shift = rng.uniform(-0.06, 0.06, size=2) * w  # moves the field-of-view border
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[:, 2] += shift
    rgb = cv2.warpAffine(rgb, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    image = rgb.astype(np.float32)
    image *= rng.uniform(0.7, 1.3)                                   # exposure
    image = (image - image.mean()) * rng.uniform(0.75, 1.25) + image.mean()  # contrast
    image *= rng.uniform(0.9, 1.1, size=3)                           # colour cast
    if rng.random() < 0.3:                                           # uneven illumination
        ramp = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0), w, dtype=np.float32)
        image *= ramp[None, :, None] if rng.random() < 0.5 else ramp[:, None, None]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (0, 0), rng.uniform(0.5, 2.0))   # defocus
    if rng.random() < 0.3:
        ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 90))])
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    return np.ascontiguousarray(image)


class Rows(Dataset):
    def __init__(self, frame, root, train, seed):
        self.frame, self.root, self.train, self.seed = frame.reset_index(drop=True), root, train, seed
        self.epoch = 0

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        bgr = cv2.imread(str(self.root / f'{row.image_id}.png'), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, index, os.getpid()))
            rgb = augment(rgb, rng)
        array = deployed_input(rgb)
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), int(row.grade)


def sampling_weights(frame, class_power, source_power=0.5):
    """Source total weight proportional to size^source_power (sqrt: small sources are
    upweighted without being replayed tenfold); within a source, inverse grade
    frequency^class_power to lift rare severe grades."""
    weights = np.zeros(len(frame))
    for source, part in frame.groupby('source'):
        counts = part.grade.value_counts()
        per = part.grade.map(lambda g: (1.0 / counts[g]) ** class_power).to_numpy()
        weights[part.index] = per / per.sum() * len(part) ** source_power
    return weights / weights.sum()


def roc_points(score, positive):
    order = np.argsort(-score)
    positive = positive[order]
    tpr = np.cumsum(positive) / max(positive.sum(), 1)
    fpr = np.cumsum(~positive) / max((~positive).sum(), 1)
    return fpr, tpr


def selection_value(score, labels):
    fpr, tpr = roc_points(score, labels >= 2)
    return float(np.mean([tpr[(1 - fpr) >= m].max() if np.any((1 - fpr) >= m) else 0 for m in (0.80, 0.85, 0.90)]))


def auc(score, labels):
    positive = labels >= 2
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(model, loader, device):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for images, _ in loader:
            grade, referral = model(images.to(device))
            outputs.append(torch.cat([grade, referral[:, None]], 1).float().cpu())
    return torch.cat(outputs).numpy().astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--init', type=Path, required=True, help='legacy checkpoint (model_state_dict)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--class-power', type=float, default=0.5)
    parser.add_argument('--freeze-blocks', type=int, default=0, help='freeze stem + first N blocks (Mac speed)')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-train-per-epoch', type=int, default=0)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.export / 'manifest.csv')
    train = manifest.loc[manifest.split == 'train'].reset_index(drop=True)
    validation = manifest.loc[manifest.split == 'validation'].reset_index(drop=True)
    images = args.export / 'images'
    model = NetrAIModel('efficientnet_b4', 5, pretrained=False)
    model.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=True))
    frozen = []
    if args.freeze_blocks:
        frozen = [model.backbone.conv_stem, model.backbone.bn1] + list(model.backbone.blocks[:args.freeze_blocks])
        for module in frozen:
            for p in module.parameters():
                p.requires_grad = False
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    per_epoch = args.max_train_per_epoch or len(train)
    steps = args.epochs * math.ceil(per_epoch / args.batch_size)
    warmup = max(1, steps // 30)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    counts = np.bincount(train.grade, minlength=5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_epoch, best, history = 0, -1.0, []
    last = args.output / 'last.pt'
    if last.exists():
        checkpoint = torch.load(last, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler']); scaler.load_state_dict(checkpoint['scaler'])
        start_epoch, best, history = checkpoint['epoch'] + 1, checkpoint['best'], checkpoint['history']
        print(f'Resumed after epoch {checkpoint["epoch"]}', flush=True)
    weights = sampling_weights(train, args.class_power)
    train_set = Rows(train, images, True, args.seed)
    val_loader = DataLoader(Rows(validation, images, False, args.seed), batch_size=32, num_workers=args.workers)
    config = {k: str(v) for k, v in vars(args).items()} | {'device': device.type, 'train_images': len(train),
                                                          'validation_images': len(validation),
                                                          'train_grade_counts': counts.tolist()}
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        train_set.epoch = epoch
        rng = np.random.default_rng((args.seed, epoch))
        order = rng.choice(len(train), size=per_epoch, replace=True, p=weights)
        loader = DataLoader(torch.utils.data.Subset(train_set, order), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, drop_last=True, persistent_workers=False)
        model.train()
        for module in frozen:
            module.eval()
        losses = []
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                grade, referral = model(x)
                loss = 0.8 * criterion(grade.float(), y) + 0.2 * nn.functional.binary_cross_entropy_with_logits(
                    referral.float(), (y >= 2).float())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(float(loss))
            if step % 200 == 0:
                print(f'epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-200:]):.4f} '
                      f'{(step + 1) * args.batch_size / (time.time() - started):.1f} img/s', flush=True)
        outputs = evaluate(model, val_loader, device)
        score = (np.exp(outputs[:, :5] - outputs[:, :5].max(1, keepdims=True))
                 / np.exp(outputs[:, :5] - outputs[:, :5].max(1, keepdims=True)).sum(1, keepdims=True)) @ np.arange(5)
        per_source = {}
        for source, part in validation.groupby('source'):
            idx = part.index.to_numpy()
            per_source[source] = {'auc': auc(score[idx], part.grade.to_numpy()),
                                  'selection_value': selection_value(score[idx], part.grade.to_numpy())}
        value = float(np.mean([m['selection_value'] for m in per_source.values()]))
        record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'selection_value': value,
                  'per_source': per_source, 'seconds': round(time.time() - started, 1)}
        history.append(record)
        print(json.dumps(record), flush=True)
        if value > best:
            best = value
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'selection_value': value,
                        'config': config}, args.output / 'best.pt')
            np.save(args.output / 'best_validation_outputs.npy', outputs)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'history': history}, last)
        (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    validation[['image_id', 'grade', 'source', 'group_id']].to_csv(args.output / 'validation_rows.csv', index=False)
    print('done; best selection value', best, flush=True)


if __name__ == '__main__':
    main()
