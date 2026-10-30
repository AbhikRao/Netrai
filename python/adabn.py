#!/usr/bin/env python3
"""Label-free site adaptation by re-estimating batch-normalisation statistics (AdaBN) on unlabelled images of one site/camera.

A camera can shift the network's intermediate feature statistics (BRSET Canon: healthy eyes score like mild disease). AdaBN replaces the
running mean/variance of the selected BatchNorm layers with statistics measured on unlabelled images from the target site, leaving every
learned weight unchanged. Scopes:

* ``backbone``: all BatchNorm2d layers of the EfficientNet-B4 backbone (the two BatchNorm1d layers of the head keep training statistics);
* ``shallow``:  the stem and blocks 0-2 only (low-level appearance: colour, contrast, sharpness), deeper layers keep training statistics.

``score`` adapts one checkpoint on ``--adapt`` images (cumulative average over all adaptation batches) and writes the logits of ``--eval``
images (n x 6: five grade logits + referral logit), same preprocessing as every lock (legacy-area-v1 at ``--size``). Runs on CPU or Apple MPS.
No label is read. Research tool: the deployed models are unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from models.netrai_model import NetrAIModel
from utils.preprocessing import preprocess_for_model

SHALLOW_PREFIXES = ('backbone.bn1', 'backbone.blocks.0.', 'backbone.blocks.1.', 'backbone.blocks.2.')


def load_member(checkpoint: Path) -> NetrAIModel:
    model = NetrAIModel('efficientnet_b4', 5, pretrained=False)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model_state_dict', state))
    return model.eval()


def selected_bn(model: torch.nn.Module, scope: str) -> list[tuple[str, torch.nn.BatchNorm2d]]:
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d) and n.startswith('backbone.')]
    if scope == 'backbone':
        return layers
    if scope == 'shallow':
        return [(n, m) for n, m in layers if n.startswith(SHALLOW_PREFIXES)]
    raise ValueError(f'unknown scope {scope!r}')


DUST_MASKS: dict = {}  # optional camera dust masks (dust_correction.py), applied before preprocessing
LOWPASS: list = []     # optional [N]: crop the field and resize to N x N before the model's own resize (removes detail finer than N px)


def load_batch(folder: Path, names, size: int) -> torch.Tensor:
    arrays = []
    for name in names:
        bgr = cv2.imdecode(np.frombuffer((folder / name).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if DUST_MASKS:
            import dust_correction
            bgr = dust_correction.apply(bgr, DUST_MASKS)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if LOWPASS:
            fg = rgb.max(axis=2) > 7
            rows, cols = np.flatnonzero(fg.any(1)), np.flatnonzero(fg.any(0))
            rgb = cv2.resize(rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], (LOWPASS[0], LOWPASS[0]), interpolation=cv2.INTER_AREA)
        arrays.append(preprocess_for_model(rgb, size=size).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(arrays)).float()


@torch.no_grad()
def adapt(model: torch.nn.Module, folder: Path, names, scope: str, size: int, device, batch: int = 8) -> int:
    """Replace the running statistics of the selected BN layers by the cumulative average over ``names``. Returns #layers adapted."""
    layers = selected_bn(model, scope)
    model.eval()
    for _, bn in layers:
        bn.reset_running_stats()
        bn.momentum = None          # cumulative moving average over all adaptation batches
        bn.train()
    for start in range(0, len(names), batch):
        model(load_batch(folder, names[start:start + batch], size).to(device))
    model.eval()
    return len(layers)


@torch.no_grad()
def logits(model: torch.nn.Module, folder: Path, names, size: int, device, batch: int = 8) -> np.ndarray:
    out = []
    for start in range(0, len(names), batch):
        grade, referral = model(load_batch(folder, names[start:start + batch], size).to(device))
        out.append(torch.cat([grade, referral[:, None]], 1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='folder holding the listed images')
    parser.add_argument('--adapt', type=Path, required=True, help='text file: adaptation image names (unlabelled)')
    parser.add_argument('--eval', type=Path, required=True, help='text file: images to score after adaptation')
    parser.add_argument('--scope', choices=('backbone', 'shallow', 'none'), required=True)
    parser.add_argument('--size', type=int, default=768)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--lowpass', type=int, default=0, help='research: resize the cropped field to N px first (detail-sensitivity test)')
    parser.add_argument('--dust', type=Path, default=None, help='folder of dust masks (dust_correction.py estimate) applied before preprocessing')
    args = parser.parse_args(argv)
    if args.lowpass:
        LOWPASS.append(args.lowpass)
    if args.dust:
        import dust_correction
        DUST_MASKS.update(dust_correction.load(args.dust))
    torch.set_num_threads(args.threads)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    model = load_member(args.checkpoint).to(device)
    read = lambda p: [line.strip() for line in p.read_text().splitlines() if line.strip()]
    n = 0 if args.scope == 'none' else adapt(model, args.images, read(args.adapt), args.scope, args.size, device)
    np.save(args.out, logits(model, args.images, read(args.eval), args.size, device))
    print(f'{args.checkpoint.name}: scope={args.scope}, adapted {n} BN layers on {len(read(args.adapt))} images, scored {len(read(args.eval))} -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
