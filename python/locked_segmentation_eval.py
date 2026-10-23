#!/usr/bin/env python3
"""Locked, single-use evaluation of a frozen lesion segmenter on the DDR lesion-segmentation test (X8).

``freeze``   binds the weights, the validation-chosen thresholds, the exclusion list (image-identity based), the test
             image inventory (file names only), the code and the metric code into ``lock.json`` with
             ``labels_read: false`` and records the cohort as *locked* in the exposure registry. No mask is read.
``evaluate`` verifies every binding, then reads the test masks once (``lesion_data.ddr_records('test', lock)``),
             scores the learned model and the deployed heuristic M2 on the *same* images at *native* resolution
             (probabilities are upsampled bilinearly to the native mask; masks are binarised at > 0, never
             re-thresholded), writes ``report.json`` and records the cohort as *exposed*. A second evaluation is refused.

Primary set = test images kept by the exclusion file; the full set is a labelled secondary (contains near-copies of
training images). Success rule (pre-registered in ``LESION_MODEL_PLAN.md``): for EX, HE and MA each, the learned
Dice at its frozen threshold exceeds the heuristic's with a paired image-bootstrap 95% interval above 0, and MA
lesion-level sensitivity at the frozen operating point exceeds the heuristic's. Nothing is tuned after the lock.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import exposure
import lesion_data as ld
import lesion_metrics as lm
from utils.artifacts import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
COHORT = 'ddr_lesion_seg_test'
SCORED = {'EX': 'exudate', 'HE': 'hem', 'MA': 'ma'}  # lesion -> heuristic key (the heuristic has no SE detector)
CODE = ['python/locked_segmentation_eval.py', 'python/lesion_data.py', 'python/lesion_metrics.py',
        'python/models/lesion_unet.py', 'python/train_lesion_seg.py', 'python/lesion_heuristic_baseline.py',
        'python/exposure.py']


def test_image_ids() -> list[str]:
    """Image inventory from file names only (no label file is opened)."""
    return sorted(p.stem for p in (ld.DDR_ROOT / 'test' / 'image').glob('*.jpg'))


def load_exclusions(path: Path) -> tuple[list[str], list[str]]:
    frame = pd.read_csv(path)
    ids = [str(i) for i in frame.image_id]
    keep = [str(r.image_id) for r in frame.itertuples() if r.decision == 'keep']
    return ids, keep


def freeze(args) -> None:
    lock_path = args.output / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    exposure.assert_fresh_allowed(COHORT)
    inventory = test_image_ids()
    listed, keep = load_exclusions(args.exclusions)
    if sorted(listed) != inventory:
        raise ValueError('The exclusion file must list exactly the test inventory')
    thresholds = json.loads(args.thresholds.read_text())
    if set(thresholds.get('dice_thresholds', {})) != set(ld.LESIONS) or 'ma_lesion_threshold' not in thresholds:
        raise ValueError('thresholds.json needs dice_thresholds for EX/HE/MA/SE and ma_lesion_threshold')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'inventory.json').write_text(json.dumps({'all': inventory, 'primary': keep}, indent=1))
    bindings = {str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p): sha256_file(p)
                for p in [args.weights, args.thresholds, args.exclusions, args.output / 'inventory.json',
                          *[ROOT / c for c in CODE]]}
    atomic_json(lock_path, {
        'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
        'tests': {COHORT: {'images_all': len(inventory), 'images_primary': len(keep)}}, 'fresh': True,
        'labels_read': False, 'weights': str(args.weights), 'thresholds': str(args.thresholds),
        'exclusions': str(args.exclusions), 'inference': {'long_side': args.long_side, 'tile': args.tile,
                                                          'overlap': args.overlap, 'enhance': args.enhance},
        'bootstrap_replicates': args.replicates, 'bootstrap_seed': args.seed, 'bindings': bindings})
    exposure.record_locked(COHORT, lock_path)
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def verify_bindings(lock: dict) -> None:
    for name, digest in lock['bindings'].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Bound artifact changed since the lock: {name}')


def load_lesion_model(weights: Path, device):
    from models.lesion_unet import LesionNet
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = LesionNet(config['arch'], config['encoder'], pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()


def native_probabilities(model, rgb: np.ndarray, device, long_side: int, tile: int, overlap: int, enhance: str) -> np.ndarray:
    """Probabilities at the native resolution of ``rgb``: predict at <= long_side, upsample bilinearly."""
    from train_lesion_seg import predict_full
    h, w = rgb.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    work = rgb if scale == 1.0 else cv2.resize(rgb, (max(32, round(w * scale)), max(32, round(h * scale))), interpolation=cv2.INTER_AREA)
    probs = predict_full(model, work, device, tile, overlap, enhance)
    if probs.shape[:2] != (h, w):
        probs = np.stack([cv2.resize(probs[..., i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(probs.shape[2])], axis=-1)
    return probs


def dice_from_counts(tp, fp, fn):
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float('nan')


def counts(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    p, t = pred & valid, truth & valid
    return np.array([(p & t).sum(), (p & ~t).sum(), (~p & t).sum()], float)


def paired_dice_bootstrap(a: np.ndarray, b: np.ndarray, replicates: int, seed: int) -> dict:
    """a, b: (images, 3) pixel counts of two methods on the same images. Pooled-Dice difference a - b."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(replicates):
        rows = rng.integers(n, size=n)
        diffs.append(dice_from_counts(*a[rows].sum(0)) - dice_from_counts(*b[rows].sum(0)))
    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]
    return {'difference': dice_from_counts(*a.sum(0)) - dice_from_counts(*b.sum(0)),
            'ci95': [float(v) for v in np.percentile(diffs, [2.5, 97.5])] if len(diffs) else [float('nan')] * 2}


def image_stats(probs, masks, annotated, valid, heuristic, thresholds: dict, ma_threshold: float) -> dict:
    """Everything the scoring needs from one image, so the (large) probability maps can be dropped immediately.

    ``heuristic`` is a dict of binary masks or ``None`` if the deployed heuristic failed on this image (then the image
    is left out of every paired/learned-vs-heuristic comparison, never counted as an empty prediction)."""
    one = lm.LesionSet()
    one.add(probs, masks, valid)
    stats = {'hist': one.per_image[0], 'annotated': np.asarray(annotated, bool), 'heuristic_ok': heuristic is not None, 'per_lesion': {}}
    for lesion, key in SCORED.items():
        index = ld.LESIONS.index(lesion)
        if not annotated[index]:
            continue
        truth = masks[..., index] > 0
        threshold = ma_threshold if lesion == 'MA' else thresholds[lesion]
        entry = {'learned_counts': counts(probs[..., index] >= thresholds[lesion], truth, valid)}
        r = lm.lesion_level(probs[..., index], truth, threshold)
        entry['learned_det'] = [r['detected'], r['gt_lesions'], r['false_positives']]
        if heuristic is not None:
            entry['heuristic_counts'] = counts(heuristic[key], truth, valid)
            r = lm.lesion_level(heuristic[key].astype(float), truth, 0.5)
            entry['heuristic_det'] = [r['detected'], r['gt_lesions'], r['false_positives']]
        stats['per_lesion'][lesion] = entry
    return stats


def score_stats(stats_list: list[dict], thresholds: dict, replicates: int, seed: int) -> dict:
    """Aggregate per-image stats (see ``image_stats``) into learned-vs-heuristic metrics."""
    out = {'images': len(stats_list), 'heuristic_failures': int(sum(not s['heuristic_ok'] for s in stats_list)), 'lesions': {}}
    lesion_set = lm.LesionSet()
    lesion_set.per_image = [s['hist'] for s in stats_list]
    summary = lesion_set.summary(thresholds)
    boot = lesion_set.bootstrap(replicates, seed, thresholds)
    for lesion in ld.LESIONS:
        out['lesions'][lesion] = {**summary[lesion], **boot.get(lesion, {})}
    for lesion in SCORED:
        rows = [s['per_lesion'][lesion] for s in stats_list if lesion in s['per_lesion'] and 'heuristic_counts' in s['per_lesion'][lesion]]
        if not rows:
            continue
        learned = np.array([r['learned_counts'] for r in rows])
        heur = np.array([r['heuristic_counts'] for r in rows])
        entry = {'images_annotated': len(rows), 'learned_dice': dice_from_counts(*learned.sum(0)),
                 'heuristic_dice': dice_from_counts(*heur.sum(0)),
                 'paired_dice_difference': paired_dice_bootstrap(learned, heur, replicates, seed)}
        det = {name: np.array([r[f'{name}_det'] for r in rows]).sum(0) for name in ('learned', 'heuristic')}
        entry['lesion_level'] = {name: {'detected': int(v[0]), 'gt_lesions': int(v[1]), 'sensitivity': (float(v[0] / v[1]) if v[1] else float('nan')),
                                        'false_positives_per_image': float(v[2] / len(rows))} for name, v in det.items()}
        out['lesions'][lesion]['vs_heuristic'] = entry
    return out


def success(primary: dict) -> dict:
    verdict = {}
    for lesion in SCORED:
        v = primary['lesions'].get(lesion, {}).get('vs_heuristic')
        verdict[lesion] = bool(v and v['paired_dice_difference']['ci95'][0] > 0)
    ma = primary['lesions'].get('MA', {}).get('vs_heuristic')
    verdict['MA_lesion_sensitivity'] = bool(ma and ma['lesion_level']['learned']['sensitivity'] > ma['lesion_level']['heuristic']['sensitivity'])
    verdict['all'] = all(verdict.values())
    return verdict


def evaluate(args) -> None:
    from lesion_heuristic_baseline import heuristic_masks
    lock_dir = args.lock
    lock = json.loads((lock_dir / 'lock.json').read_text())
    exposure.assert_label_read_allowed(COHORT, lock_dir)  # raises if no lock / already evaluated
    verify_bindings(lock)
    inventory = json.loads((lock_dir / 'inventory.json').read_text())
    primary_ids = set(inventory['primary'])
    thresholds_file = json.loads(Path(lock['thresholds']).read_text())
    thresholds = thresholds_file['dice_thresholds']
    ma_threshold = float(thresholds_file['ma_lesion_threshold'])
    cfg = lock['inference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_lesion_model(Path(lock['weights']), device)
    stats = {}
    for record in ld.ddr_records('test', lock_dir):  # the single guarded read of the test masks
        stem = record.image_id.removeprefix('ddr_')
        if stem not in set(inventory['all']):
            raise ValueError(f'{stem} is not in the locked inventory')
        rgb = cv2.cvtColor(cv2.imread(str(record.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        masks, annotated = ld.read_masks(record, rgb.shape[:2], allow_reserved=True)
        top, bottom, left, right = ld.foreground_box(rgb)
        rgb, masks = rgb[top:bottom, left:right], masks[top:bottom, left:right]
        probs = native_probabilities(model, rgb, device, cfg['long_side'], cfg['tile'], cfg['overlap'], cfg['enhance'])
        try:
            heuristic = heuristic_masks(rgb)
        except Exception as error:  # a heuristic failure must not abort the single-use evaluation
            print(f'heuristic failed on {stem}: {error}', flush=True)
            heuristic = None
        stats[stem] = image_stats(probs, masks, annotated, rgb.max(axis=2) > ld.FOREGROUND, heuristic, thresholds, ma_threshold)
        del probs, masks, rgb
        print(f'{len(stats)}/{len(inventory["all"])}', flush=True)
    primary = score_stats([v for k, v in stats.items() if k in primary_ids], thresholds, lock['bootstrap_replicates'], lock['bootstrap_seed'])
    secondary = score_stats(list(stats.values()), thresholds, lock['bootstrap_replicates'], lock['bootstrap_seed'])
    report = {'tier': 'TF', 'lock': str(lock_dir / 'lock.json'), 'primary': primary, 'success_rule_primary': success(primary),
              'secondary_all_images_contains_near_copies': secondary,
              'limitations': ['DDR has no patient ids: independence is image-identity evidence, not proof.',
                              'Test is in-distribution (same hospitals/cameras as most training data).',
                              'MA and SE have few annotated pixels: intervals are wide.',
                              'The heuristic has no soft-exudate detector; SE is learned-only.']}
    report_path = lock_dir / 'report.json'
    atomic_json(report_path, report)
    exposure.record_exposure(COHORT, lock_dir / 'lock.json', report_path)
    print(json.dumps(report['success_rule_primary']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze')
    f.add_argument('--weights', type=Path, required=True)
    f.add_argument('--thresholds', type=Path, required=True, help='json: dice_thresholds{EX,HE,MA,SE}, ma_lesion_threshold (validation-chosen)')
    f.add_argument('--exclusions', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--long-side', type=int, default=2048)
    f.add_argument('--tile', type=int, default=1024)
    f.add_argument('--overlap', type=int, default=128)
    f.add_argument('--enhance', default='ben')
    f.add_argument('--replicates', type=int, default=2000)
    f.add_argument('--seed', type=int, default=20261030)
    e = sub.add_parser('evaluate')
    e.add_argument('--lock', type=Path, required=True)
    args = parser.parse_args(argv)
    (freeze if args.command == 'freeze' else evaluate)(args)


if __name__ == '__main__':
    main()
