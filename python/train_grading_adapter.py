#!/usr/bin/env python3
"""Multi-source adaptation ladder for the deployed grading model (CPU feasible).

Rungs (all keep the deployed architecture, input contract and ONNX schema):

* ``A``  legacy model and thresholds, unchanged;
* ``B``  legacy logits with ordinal thresholds refit on the training pool;
* ``C``  legacy backbone frozen; head/grade/referral layers fine-tuned on pooled
         1792-D features;
* ``D``  ``blocks[6]``, ``conv_head``/``bn2`` affine and heads fine-tuned on cached
         ``blocks[5]`` maps with flip augmentation. BatchNorm statistics,
         dropout-path and earlier blocks stay frozen.

Patient groups never cross partitions. The training pool is split into
fit / holdout by patient; the holdout fixes thresholds and temperature. Model
epochs are chosen on separate selection cohorts (DeepDRiD validation and an EyeQ
test subset). IDRiD test is reported as exposed stress evidence only. The
reserved DeepDRiD evaluation set is never read here.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve

from evaluate_grading_candidate import fit_other_thresholds
from models.losses import FocalLoss
from utils.artifacts import atomic_json, sha256_file
from utils.calibration import fit_temperature
from utils.metrics import quadratic_weighted_kappa
from utils.validation import ordinal_thresholds

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'results/model_improvement/feature_cache'
LEGACY_THRESHOLDS = ROOT / 'python/weights/fold0_thresholds.npy'
POOL = ('deepdrid_training', 'idrid_train', 'eyeq_train')
SELECTION = ('deepdrid_validation', 'eyeq_test')
STRESS = ('idrid_test',)


def load_cohort(name, maps=False):
    folder = CACHE / name
    rows = pd.read_csv(folder / 'rows.csv', dtype={'image_id': str, 'group_id': str})
    shards = sorted(folder.glob('shard_*.npz'))
    parts = {}
    for shard in shards:
        with np.load(shard) as data:
            for key in data.files:
                if key.startswith('block5') and not maps:
                    continue
                parts.setdefault(key, []).append(data[key])
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    if len(arrays['image_id']) != len(rows) or not np.array_equal(arrays['image_id'], rows.image_id.to_numpy(str)):
        raise ValueError(f'{name}: cache incomplete or misaligned ({len(arrays["image_id"])}/{len(rows)})')
    return rows, arrays


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def scores(logits):
    return softmax(logits[:, :5]) @ np.arange(5)


def grade(score, thresholds):
    return (score[:, None] >= np.asarray(thresholds)[None]).sum(axis=1)


def midpoint_below(values, cut):
    """Place a boundary halfway below ``cut`` so no sample sits on it.

    Decisions on the fitting data are unchanged, and tiny cross-runtime
    numerical differences cannot flip the boundary sample.
    """
    lower = np.unique(values[values < cut])
    return float((cut + lower[-1]) / 2) if len(lower) else float(cut - 1e-6)


def referral_threshold(score, labels, target):
    """Lowest-FPR score boundary reaching ``target`` referable sensitivity."""
    fpr, tpr, cut = roc_curve(labels >= 2, score)
    eligible = np.flatnonzero(tpr >= target)
    best = eligible[np.argmin(fpr[eligible])]
    return float(np.clip(midpoint_below(score, cut[best]), 1e-3, 4 - 1e-3))


def fit_policy(score, labels, target, sources=None, rule='pooled'):
    """Fix the grade-2 boundary, then fit the other ordinal boundaries for QWK.

    ``pooled``: lowest-FPR boundary reaching ``target`` sensitivity on all rows.
    ``worst-source``: the most conservative of the per-source boundaries, so each
    source separately reaches ``target`` sensitivity.
    """
    if rule == 'pooled':
        t1 = referral_threshold(score, labels, target)
    elif rule == 'balanced':
        t1 = balanced_threshold(score, labels, sources, target)
    else:
        t1 = min(referral_threshold(score[sources == name], labels[sources == name], target)
                 for name in np.unique(sources) if np.any(labels[sources == name] >= 2))
    return ordinal_thresholds(fit_other_thresholds(score, labels, t1)).tolist()


def balanced_threshold(score, labels, sources, target):
    """Boundary maximizing source-averaged specificity at source-averaged
    sensitivity >= target, so no single large source dominates the operating point."""
    names = [n for n in np.unique(sources) if np.any(labels[sources == n] >= 2) and np.any(labels[sources == n] < 2)]
    candidates = np.unique(score)
    best = None
    for cut in candidates:
        sens = np.mean([np.mean(score[(sources == n) & (labels >= 2)] >= cut) for n in names])
        spec = np.mean([np.mean(score[(sources == n) & (labels < 2)] < cut) for n in names])
        if sens >= target and (best is None or spec > best[0]):
            best = (spec, cut)
    return float(np.clip(best[1], 1e-3, 4 - 1e-3))


def fit_head_threshold(logits, labels, target, sources=None, rule='pooled'):
    """Referral-head probability cut with the same sensitivity target and rule."""
    probability = 1 / (1 + np.exp(-logits[:, 5]))
    if rule == 'balanced':
        return float(np.clip(balanced_threshold(probability, labels, sources, target), 1e-4, 1 - 1e-4))
    fpr, tpr, cut = roc_curve(labels >= 2, probability)
    eligible = np.flatnonzero(tpr >= target)
    boundary = midpoint_below(probability, cut[eligible[np.argmin(fpr[eligible])]])
    return float(np.clip(boundary, 1e-6, 1 - 1e-6))


def selection_value(score, labels):
    """Smooth high-sensitivity criterion: mean sensitivity at 80/85/90% specificity."""
    return float(np.mean([sens_at_spec(score, labels, m) for m in (0.80, 0.85, 0.90)]))


def sens_at_spec(score, labels, minimum=0.85):
    fpr, tpr, _ = roc_curve(labels >= 2, score)
    allowed = (1 - fpr) >= minimum
    return float(tpr[allowed].max()) if allowed.any() else 0.0


def evaluate(logits, labels, thresholds, head_threshold=0.5):
    score = scores(logits)
    predicted = grade(score, thresholds)
    truth = labels >= 2
    positive = predicted >= 2
    head = 1 / (1 + np.exp(-logits[:, 5])) >= head_threshold
    return {
        'images': int(len(labels)), 'referable': int(truth.sum()),
        'sensitivity': float(positive[truth].mean()), 'specificity': float((~positive[~truth]).mean()),
        'auc': float(roc_auc_score(truth, score)), 'sens_at_spec85': sens_at_spec(score, labels),
        'sens_at_spec80': sens_at_spec(score, labels, 0.80), 'sens_at_spec90': sens_at_spec(score, labels, 0.90),
        'selection_value': selection_value(score, labels),
        'qwk': float(quadratic_weighted_kappa(labels, predicted)),
        'per_grade_recall': [float(np.mean(predicted[labels == g] == g)) if np.any(labels == g) else None
                             for g in range(5)],
        'head_sensitivity': float(head[truth].mean()), 'head_specificity': float((~head[~truth]).mean()),
        'head_disagreement_rate': float(np.mean(head != positive)),
    }


class Tail(nn.Module):
    """Trainable part of the deployed model starting from ``blocks[5]`` output."""

    def __init__(self, model, start):
        super().__init__()
        backbone = model.backbone
        self.start = start
        self.block6 = backbone.blocks[6]
        self.conv_head, self.bn2, self.pool = backbone.conv_head, backbone.bn2, backbone.global_pool
        self.head, self.grade_head, self.ref_head = model.head, model.grade_head, model.ref_head

    def forward(self, x):
        if self.start == 'block5':
            x = self.pool(self.bn2(self.conv_head(self.block6(x))))
        x = self.head(x)
        return torch.cat([self.grade_head(x), self.ref_head(x)], 1)

    def set_mode(self):
        self.eval()  # frozen BatchNorm statistics and no stochastic depth
        for module in self.head.modules():
            if isinstance(module, nn.Dropout):
                module.train()


def train_rung(rung, model, pool, selection, args, log):
    start = 'block5' if rung == 'D' else 'pooled'
    tail = Tail(copy.deepcopy(model), start)
    trainable = list(tail.head.parameters()) + list(tail.grade_head.parameters()) + list(tail.ref_head.parameters())
    backbone_params = []
    if rung == 'D':
        backbone_params = list(tail.block6.parameters()) + list(tail.conv_head.parameters()) + list(tail.bn2.parameters())
    for parameter in tail.parameters():
        parameter.requires_grad = False
    for parameter in trainable + backbone_params:
        parameter.requires_grad = True
    groups = [{'params': trainable, 'lr': args.head_lr}]
    if backbone_params:
        groups.append({'params': backbone_params, 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    key = 'block5' if rung == 'D' else 'pooled'
    x, x_flip, y, source = pool[key], pool.get(key + '_flip'), pool['grade'], pool['source']
    counts = np.bincount(y, minlength=5)
    criterion = FocalLoss(torch.tensor(len(y) / (5 * np.maximum(counts, 1)), dtype=torch.float32),
                          gamma=0, label_smoothing=0.1)
    # Equal total weight per source so EyePACS volume does not drown new cameras.
    source_weight = {name: len(y) / (len(set(source)) * np.sum(source == name)) for name in set(source)}
    sample_weight = np.array([source_weight[name] for name in source], dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    best, best_state, history = -np.inf, None, []
    for epoch in range(args.epochs):
        started = time.time()
        tail.set_mode()
        order = rng.choice(len(y), size=len(y), replace=True, p=sample_weight / sample_weight.sum())
        losses = []
        for begin in range(0, len(order), args.batch_size):
            index = order[begin:begin + args.batch_size]
            batch = x[index]
            if x_flip is not None:
                flip = rng.random(len(index)) < 0.5
                batch = np.where(flip.reshape((-1,) + (1,) * (batch.ndim - 1)), x_flip[index], batch)
            inputs = torch.from_numpy(batch.astype(np.float32))
            targets = torch.from_numpy(y[index].astype(np.int64))
            outputs = tail(inputs)
            loss = 0.8 * criterion(outputs[:, :5], targets) + 0.2 * nn.functional.binary_cross_entropy_with_logits(
                outputs[:, 5], (targets >= 2).float())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable + backbone_params, 1.0)
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()
        metrics = {name: evaluate(predict(tail, data[key], args), data['grade'], LEGACY) for name, data in selection.items()}
        criterion_value = float(np.mean([m['selection_value'] for m in metrics.values()]))
        tie = float(np.mean([m['auc'] for m in metrics.values()]))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'selection_value': criterion_value,
                        'selection_auc': tie, 'seconds': round(time.time() - started, 1)})
        log(f'{rung} epoch {epoch}: {history[-1]}')
        if criterion_value + 1e-3 * tie > best:
            best = criterion_value + 1e-3 * tie
            best_state = copy.deepcopy(tail.state_dict())
    tail.load_state_dict(best_state)
    return tail, history


def predict(tail, x, args):
    tail.eval()
    outputs = []
    with torch.inference_mode():
        for begin in range(0, len(x), 256):
            outputs.append(tail(torch.from_numpy(x[begin:begin + 256].astype(np.float32))).numpy())
    return np.concatenate(outputs).astype(np.float64)


def stack(cohorts, maps):
    frames, arrays = [], {}
    for name in cohorts:
        rows, data = load_cohort(name, maps)
        rows['cohort'] = name
        frames.append(rows)
        for key, value in data.items():
            arrays.setdefault(key, []).append(value)
    frame = pd.concat(frames, ignore_index=True)
    joined = {key: np.concatenate(value) for key, value in arrays.items() if len(value) == len(cohorts)}
    joined['grade'] = frame.grade.to_numpy(int)
    joined['source'] = frame.source.to_numpy(str)
    joined['group'] = frame.group_id.to_numpy(str)
    return frame, joined


def subset(data, mask):
    return {key: value[mask] for key, value in data.items()}


LEGACY = None


def main():
    global LEGACY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rungs', nargs='+', default=['A', 'B', 'C', 'D'])
    parser.add_argument('--pool', nargs='+', default=list(POOL))
    parser.add_argument('--selection', nargs='+', default=list(SELECTION))
    parser.add_argument('--stress', nargs='*', default=list(STRESS))
    parser.add_argument('--exclude-pool', nargs='*', default=[], help='e.g. deepdrid_training for D-noDeepDRiD')
    parser.add_argument('--target-sensitivity', type=float, default=0.92)
    parser.add_argument('--threshold-rule', choices=('pooled', 'balanced', 'worst-source'), default='pooled')
    parser.add_argument('--holdout-fraction', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--backbone-lr', type=float, default=5e-5)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    code_sha256 = sha256_file(Path(__file__))  # hash the code that actually runs
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    log_path = args.output / 'log.txt'

    def log(message):
        print(message, flush=True)
        with log_path.open('a') as handle:
            handle.write(message + '\n')

    LEGACY = np.load(LEGACY_THRESHOLDS).astype(float).tolist()
    maps = 'D' in args.rungs
    pool_names = [name for name in args.pool if name not in args.exclude_pool]
    pool_frame, pool = stack(pool_names, maps)
    groups = np.unique(pool['group'])
    holdout_groups = set(np.random.default_rng(args.seed).choice(
        groups, size=int(round(args.holdout_fraction * len(groups))), replace=False))
    holdout_mask = np.array([g in holdout_groups for g in pool['group']])
    fit, holdout = subset(pool, ~holdout_mask), subset(pool, holdout_mask)
    assert not set(fit['group']) & set(holdout['group'])
    selection = {name: stack([name], maps)[1] for name in args.selection}
    stress = {name: stack([name], maps)[1] for name in args.stress}
    log(f'pool {pool_names}: fit {len(fit["grade"])} / holdout {len(holdout["grade"])} images; '
        f'selection {[len(v["grade"]) for v in selection.values()]}; stress {[len(v["grade"]) for v in stress.values()]}')

    from inference import _get_model
    model, _, _ = _get_model()
    results, policies = {}, {}
    for rung in args.rungs:
        if rung in ('A', 'B'):
            outputs = {'holdout': holdout['logits'], 'pool': pool['logits'],
                       **{n: d['logits'] for n, d in {**selection, **stress}.items()}}
            if rung == 'A':
                thresholds, temperature, head_threshold = LEGACY, None, 0.5
            else:
                thresholds = fit_policy(scores(pool['logits']), pool['grade'], args.target_sensitivity,
                                        pool['source'], args.threshold_rule)
                head_threshold = fit_head_threshold(pool['logits'], pool['grade'], args.target_sensitivity)
                temperature = None
        else:
            tail, history = train_rung(rung, model, fit, selection, args, log)
            key = 'block5' if rung == 'D' else 'pooled'
            outputs = {'holdout': predict(tail, holdout[key], args),
                       **{n: predict(tail, d[key], args) for n, d in {**selection, **stress}.items()}}
            thresholds = fit_policy(scores(outputs['holdout']), holdout['grade'], args.target_sensitivity,
                                    holdout['source'], args.threshold_rule)
            head_threshold = fit_head_threshold(outputs['holdout'], holdout['grade'], args.target_sensitivity)
            temperature, _ = fit_temperature(outputs['holdout'][:, :5], holdout['grade'])
            torch.save({'state_dict': tail.state_dict(), 'start': tail.start}, args.output / f'rung_{rung}.pt')
            atomic_json(args.output / f'rung_{rung}_history.json', history)
        np.savez(args.output / f'rung_{rung}_outputs.npz', **outputs,
                 **{f'{name}_grade': data['grade'] for name, data in {**selection, **stress}.items()},
                 holdout_grade=holdout['grade'], holdout_source=holdout['source'])
        policies[rung] = {'thresholds': thresholds, 'temperature': temperature, 'head_threshold': head_threshold,
                          'threshold_source': 'legacy' if rung == 'A' else ('training pool' if rung == 'B' else 'pool holdout')}
        results[rung] = {name: evaluate(outputs[name], data['grade'], thresholds, head_threshold)
                         for name, data in {**selection, **stress}.items()}
        results[rung]['holdout'] = evaluate(outputs['holdout'], holdout['grade'], thresholds, head_threshold)
        log(f'{rung}: ' + json.dumps({n: {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}
                                       for n, m in results[rung].items()}))
    atomic_json(args.output / 'results.json', {
        'schema_version': 1, 'status': 'development_selection_not_acceptance',
        'pool': pool_names, 'selection': args.selection, 'stress_exposed': args.stress,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'code_sha256': code_sha256, 'policies': policies, 'results': results,
        'holdout_groups': len(holdout_groups),
        'limitations': ['DeepDRiD validation and EyeQ test subset choose epochs and rungs; they are development data.',
                        'IDRiD test was previously exposed and is not used for any choice.',
                        'Acceptance requires the locked DeepDRiD evaluation.']})


if __name__ == '__main__':
    main()
