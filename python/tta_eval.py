#!/usr/bin/env python3
"""X14: test-time augmentation for leave-one-dataset-out grading runs (frozen weights, no training).

For one LODO run (``train_v6.py`` output with ``best.pt`` and ``heldout_rows.csv``), score the held-out images under
the views ``identity`` and ``hflip`` (horizontal flip of the network input) and save the logits of each view
(``logits_<view>.npy``, rows in ``heldout_rows.csv`` order). The decision rule of the tool ``compare`` is the paired
comparison of the averaged-logit TTA against the identity view on AUC, sensitivity at 85/90% specificity (cluster
bootstrap), computed with ``netrai_eval`` on the held-out source only; no label touches the model.

Runs on CPU, CUDA or Apple MPS. Image roots are folders holding ``<image_id>.png`` (the 768 exports).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import train_v6 as t6
from train_full_finetune import evaluate

VIEWS = ('identity', 'hflip')


class View(nn.Module):
    def __init__(self, model: nn.Module, view: str):
        super().__init__()
        if view not in VIEWS:
            raise ValueError(f'unknown view {view!r}')
        self.model, self.view = model, view

    def forward(self, x):
        return self.model(x.flip(-1) if self.view == 'hflip' else x)


def load_model(best: Path, device) -> nn.Module:
    model, _ = t6.build_model('random')
    sys.modules.setdefault('pathlib._local', pathlib)  # checkpoints saved by Python 3.13 (Kaggle) pickle pathlib._local.PosixPath
    checkpoint = torch.load(best, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def score(run: Path, roots, output: Path, device, input_size: int = 512, workers: int = 2, batch: int = 16,
          limit: int = 0) -> dict:
    rows = pd.read_csv(run / 'heldout_rows.csv')
    if limit:
        rows = rows.iloc[:limit].reset_index(drop=True)
    model = load_model(run / 'best.pt', device)
    dataset = t6.RowsV6(rows, [Path(r) for r in roots], False, 0, 0.0, input_size)
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers)
    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / 'rows.csv', index=False)
    for view in VIEWS:
        path = output / f'logits_{view}.npy'
        if path.exists():
            continue
        np.save(path, evaluate(View(model, view), loader, device))
        print(f'{run.parent.name}: {view} done ({len(rows)} images)', flush=True)
    return {'run': str(run), 'rows': len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    s = sub.add_parser('score')
    s.add_argument('--run', type=Path, required=True)
    s.add_argument('--roots', type=Path, nargs='+', required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--device', default='auto')
    s.add_argument('--workers', type=int, default=2)
    s.add_argument('--limit', type=int, default=0)
    c = sub.add_parser('compare')
    c.add_argument('--score-dir', type=Path, nargs='+', required=True, help='score output folders (one per LODO fold)')
    c.add_argument('--output', type=Path, required=True)
    c.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == 'score':
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu') \
            if args.device == 'auto' else torch.device(args.device)
        score(args.run, args.roots, args.output, device, workers=args.workers, limit=args.limit)
    else:
        compare(args)


def compare(args):
    from netrai_eval import bootstrap as bs
    from netrai_eval import metrics as m
    result = {'tier': 'TL (leave-dataset-out held-out sources, previously exposed data)', 'folds': {}}
    for folder in args.score_dir:
        rows = pd.read_csv(folder / 'rows.csv')
        a, b = np.load(folder / 'logits_identity.npy'), np.load(folder / 'logits_hflip.npy')
        tta = (a + b) / 2
        positive = rows.grade.to_numpy() >= 2
        clusters = rows.group_id.to_numpy()

        def stat(index):
            pos = positive[index]
            return {'auc': m.auc(pos, m.expected_grade_score(tta[index])) - m.auc(pos, m.expected_grade_score(a[index]))}

        base = m.auc(positive, m.expected_grade_score(a))
        out = bs.bootstrap(stat, clusters, args.replicates, 20261031)
        result['folds'][folder.name] = {'n': len(rows), 'auc_identity': base,
                                        'auc_tta': m.auc(positive, m.expected_grade_score(tta)),
                                        'auc_difference': out['auc']}
    args.output.write_text(json.dumps(result, indent=2, default=float))
    for k, v in result['folds'].items():
        print(k, round(v['auc_identity'], 4), round(v['auc_tta'], 4), v['auc_difference'])


if __name__ == '__main__':
    sys.exit(main())
