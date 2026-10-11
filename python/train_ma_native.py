#!/usr/bin/env python3
"""Fit a higher-resolution MA candidate ranker using IDRiD training masks only.

The official test split is scored once with a threshold chosen on a separate
training-subset validation partition. Results remain retrospective research
evidence because this project's earlier IDRiD test results were inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from modules.ma_native import FEATURE_NAMES, extract_candidates, score_and_suppress
from utils.artifacts import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def load_or_extract(row, root, cache, longest_edge, floor, max_candidates):
    image = root / row.image_path
    entry = cache / f'{row.image_id}.npz'
    feature_hash = sha256_file(Path(__file__).parent / 'modules/ma_native.py')
    signature = f'{sha256_file(image)}:{feature_hash}:{longest_edge}:{floor}:{max_candidates}:v2'
    if entry.is_file():
        with np.load(entry) as stored:
            if str(stored['signature']) == signature:
                return stored['native'], stored['features'], stored['grid'], tuple(stored['size'])
    bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image)
    native, features, grid, size = extract_candidates(
        bgr, longest_edge=longest_edge, floor=floor, max_candidates=max_candidates)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(entry, signature=signature, native=native, features=features,
                        grid=grid, size=np.asarray(size))
    return native, features, grid, size


def training_rows(row, root, grid, size, features, maximum_negatives, rng):
    mask_path = root / row.ma_mask
    truth = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if truth is None:
        raise FileNotFoundError(mask_path)
    scaled = cv2.resize((truth > 0).astype(np.uint8), size,
                        interpolation=cv2.INTER_NEAREST)
    if len(grid) == 0:
        return np.empty((0, len(FEATURE_NAMES)), np.float32), np.empty(0, np.uint8)
    if not scaled.any():
        indices = rng.choice(len(features), min(maximum_negatives, len(features)), replace=False)
        return features[indices], np.zeros(len(indices), dtype=np.uint8)
    distance, nearest = cv2.distanceTransformWithLabels(
        1-scaled, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)
    xs, ys = grid[:, 0], grid[:, 1]
    nearby = distance[ys, xs] <= 20 * size[0] / truth.shape[1]
    lesion_id = nearest[ys, xs]
    positives = []
    for label in np.unique(lesion_id[nearby]):
        available = np.flatnonzero(nearby & (lesion_id == label))
        if len(available):
            # Prefer a peak near the lesion and with a strong dark-spot response.
            rank = features[available, 3] - 1.5*distance[ys[available], xs[available]]
            positives.append(available[np.argmax(rank)])
    negatives = np.flatnonzero(distance[ys, xs] > 30 * size[0] / truth.shape[1])
    if len(negatives) > maximum_negatives:
        hard = negatives[np.argsort(-features[negatives, 3])[:maximum_negatives//2]]
        remainder = np.setdiff1d(negatives, hard, assume_unique=True)
        random = rng.choice(remainder, size=maximum_negatives-len(hard), replace=False)
        negatives = np.concatenate((hard, random))
    chosen = np.concatenate((np.asarray(positives, dtype=int), negatives))
    labels = np.concatenate((np.ones(len(positives), dtype=np.uint8),
                             np.zeros(len(negatives), dtype=np.uint8)))
    return features[chosen], labels


def match_trace(native, scores, truth, tolerance=20):
    """Incremental maximum matching; every score prefix is label-order invariant."""
    from collections import deque
    count, labels = cv2.connectedComponents(truth.astype(np.uint8), connectivity=8)
    height, width = truth.shape
    if len(native) != len(scores) or np.any(np.diff(scores) > 0):
        raise ValueError('Candidates must have aligned, descending finite scores')
    if not np.isfinite(native).all() or not np.isfinite(scores).all():
        raise ValueError('Nonfinite candidates')
    edges, matched = [], {}
    hits = np.zeros(len(scores), dtype=np.int32)
    for index, (x, y) in enumerate(native):
        x0, x1 = max(0, int(np.floor(x-tolerance))), min(width, int(np.ceil(x+tolerance))+1)
        y0, y1 = max(0, int(np.floor(y-tolerance))), min(height, int(np.ceil(y+tolerance))+1)
        candidates = []
        if x0 < x1 and y0 < y1:
            yy, xx = np.ogrid[y0:y1, x0:x1]
            values = np.unique(labels[y0:y1, x0:x1][(xx-x)**2+(yy-y)**2 <= tolerance**2])
            candidates = [int(value) for value in values if value > 0]
        edges.append(candidates)
        queue, parents, visited_labels = deque([index]), {index: None}, set()
        augmented = False
        while queue and not augmented:
            point = queue.popleft()
            for label in edges[point]:
                if label in visited_labels:
                    continue
                visited_labels.add(label)
                if label not in matched:
                    while True:
                        matched[label] = point
                        parent = parents[point]
                        if parent is None:
                            break
                        point, label = parent
                    augmented = True
                    break
                owner = matched[label]
                if owner not in parents:
                    parents[owner] = (point, label)
                    queue.append(owner)
        hits[index] = len(matched)
    return scores, hits, count-1


def score_split(rows, manifest_root, cache, model, settings):
    scored = []
    for index, row in enumerate(rows.itertuples(index=False), 1):
        native, features, _, _ = load_or_extract(
            row, manifest_root, cache, settings['longest_edge'],
            settings['floor'], settings['max_candidates'])
        coordinates, scores = score_and_suppress(native, features, model)
        truth = cv2.imread(str(manifest_root / row.ma_mask), cv2.IMREAD_GRAYSCALE)
        if truth is None:
            raise FileNotFoundError(row.ma_mask)
        scored.append((row.image_id, *match_trace(coordinates, scores, truth > 0)))
        print(f'{row.split} {index}/{len(rows)} {row.image_id}: '
              f'{len(coordinates)} ranked candidates', flush=True)
    return scored


def metrics_for_threshold(scored, threshold):
    hits = fp = lesions = 0
    for _, scores, cumulative_hits, total in scored:
        proposals = int(np.count_nonzero(scores >= threshold))
        found = int(cumulative_hits[proposals-1]) if proposals else 0
        hits += found
        fp += proposals - found
        lesions += total
    return {'threshold': float(threshold), 'images': len(scored),
            'lesions': lesions, 'hits': hits, 'false_positives': fp,
            'sensitivity': hits / lesions if lesions else None,
            'false_positives_per_image': fp / len(scored) if scored else None}


def per_image_at_threshold(scored, threshold):
    rows = []
    for image_id, scores, cumulative_hits, lesions in scored:
        proposals = int(np.count_nonzero(scores >= threshold))
        hits = int(cumulative_hits[proposals-1]) if proposals else 0
        rows.append({'image_id': image_id, 'threshold': threshold,
                     'lesions': lesions, 'hits': hits,
                     'false_positives': proposals-hits})
    return pd.DataFrame(rows)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def paired_baseline_comparison(candidate_rows, baseline_path, baseline_threshold):
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.loc[(baseline.split == 'test')
                            & (baseline.threshold == baseline_threshold)]
    paired = candidate_rows.merge(
        baseline[['image_id', 'lesion_hits', 'false_positives', 'lesion_total']],
        on='image_id', suffixes=('_candidate', '_baseline'), validate='one_to_one')
    if (len(paired) != 27 or not np.array_equal(
            paired.lesions.to_numpy(), paired.lesion_total.to_numpy())):
        raise ValueError('Candidate/baseline lesion cohorts differ')
    rng = np.random.default_rng(42)
    draws = rng.integers(len(paired), size=(5000, len(paired)))
    denominators = paired.lesions.to_numpy()[draws].sum(axis=1)
    sensitivity_difference = (
        paired.hits.to_numpy()[draws].sum(axis=1)
        - paired.lesion_hits.to_numpy()[draws].sum(axis=1)) / denominators
    fp_difference = (
        paired.false_positives_candidate.to_numpy()[draws]
        - paired.false_positives_baseline.to_numpy()[draws]).mean(axis=1)
    return {
        'baseline_source': str(baseline_path.resolve().relative_to(ROOT)),
        'baseline_source_sha256': sha256_file(baseline_path),
        'baseline_threshold': baseline_threshold,
        'baseline_sensitivity': float(paired.lesion_hits.sum()/paired.lesions.sum()),
        'baseline_false_positives_per_image': float(paired.false_positives_baseline.mean()),
        'candidate_sensitivity': float(paired.hits.sum()/paired.lesions.sum()),
        'candidate_false_positives_per_image': float(paired.false_positives_candidate.mean()),
        'sensitivity_difference': float((paired.hits.sum()-paired.lesion_hits.sum())
                                        / paired.lesions.sum()),
        'sensitivity_difference_paired_image_bootstrap_95':
            np.quantile(sensitivity_difference, [0.025, 0.975]).tolist(),
        'false_positives_per_image_difference': float(
            (paired.false_positives_candidate-paired.false_positives_baseline).mean()),
        'false_positives_difference_paired_image_bootstrap_95':
            np.quantile(fp_difference, [0.025, 0.975]).tolist(),
        'bootstrap_replicates': 5000,
    }


def run(args):
    cv2.setNumThreads(1)
    frame = pd.read_csv(args.manifest)
    training = frame.loc[frame.split == 'train'].reset_index(drop=True)
    testing = frame.loc[frame.split == 'test'].reset_index(drop=True)
    if len(training) != 54 or len(testing) != 27:
        raise ValueError('Expected the official 54/27 IDRiD segmentation partition')
    fit_ids, dev_ids = train_test_split(
        training.image_id.to_numpy(), test_size=12, random_state=42)
    fit_rows = training.loc[training.image_id.isin(fit_ids)]
    dev_rows = training.loc[training.image_id.isin(dev_ids)]
    root = args.manifest.resolve().parent
    cache = args.output / 'cache'
    rng = np.random.default_rng(42)
    matrices, labels = [], []
    for index, row in enumerate(fit_rows.itertuples(index=False), 1):
        _, features, grid, size = load_or_extract(
            row, root, cache, args.longest_edge, args.floor, args.max_candidates)
        x, y = training_rows(row, root, grid, size, features,
                             args.maximum_negatives_per_image, rng)
        matrices.append(x)
        labels.append(y)
        print(f'fit {index}/{len(fit_rows)} {row.image_id}: '
              f'{int(y.sum())} positive / {len(y)-int(y.sum())} negative', flush=True)
    x = np.vstack(matrices)
    y = np.concatenate(labels)
    model = HistGradientBoostingClassifier(
        max_iter=120, max_leaf_nodes=15, max_depth=5, min_samples_leaf=30,
        learning_rate=0.08, class_weight='balanced', l2_regularization=1.0,
        early_stopping=False, random_state=42)
    model.fit(x, y)
    print(f'fit complete: {len(x)} candidates, {int(y.sum())} positives', flush=True)
    settings = {'longest_edge': args.longest_edge, 'floor': args.floor,
                'max_candidates': args.max_candidates}
    dev = score_split(dev_rows, root, cache, model, settings)
    development_scores = np.concatenate([item[1] for item in dev])
    thresholds = np.unique(np.r_[0, 1,
        np.quantile(development_scores, np.r_[np.linspace(0, 0.9, 20),
                                               np.linspace(0.9, 0.9999, 201)])])
    dev_curve = pd.DataFrame([metrics_for_threshold(dev, threshold)
                              for threshold in thresholds])
    args.output.mkdir(parents=True, exist_ok=True)
    dev_curve.to_csv(args.output / 'development_froc.csv', index=False)
    eligible = dev_curve.loc[dev_curve.false_positives_per_image <= args.fp_budget]
    if eligible.empty:
        raise ValueError('No threshold satisfies the declared development FP budget')
    selected = eligible.sort_values(
        ['sensitivity', 'false_positives_per_image', 'threshold'],
        ascending=[False, True, False]).iloc[0]
    threshold = float(selected.threshold)
    print(f'development-selected threshold={threshold:.3f} '
          f'sensitivity={selected.sensitivity:.3f} '
          f'FP/image={selected.false_positives_per_image:.2f}', flush=True)
    test = score_split(testing, root, cache, model, settings)
    test_curve = pd.DataFrame([metrics_for_threshold(test, item)
                               for item in thresholds])
    model_path = args.output / 'ma_ranker.joblib'
    joblib.dump(model, model_path)
    test_curve.to_csv(args.output / 'test_froc.csv', index=False)
    test_per_image = per_image_at_threshold(test, threshold)
    test_per_image.to_csv(args.output / 'test_per_image_at_locked_threshold.csv', index=False)
    prior = ROOT / 'results/verification_2026-09-24'
    comparisons = {
        'classic_512_sigma_2_comparable_fp': paired_baseline_comparison(
            test_per_image, prior/'ma_froc/ma_froc_per_image.csv', 2.0),
        'research_1024_sigma_2_5': paired_baseline_comparison(
            test_per_image, prior/'ma_froc_1024/ma_froc_per_image.csv', 2.5),
    }
    summary = {'schema_version': 1,
               'status': 'retrospective_candidate_not_clinically_validated',
               'model': '2048-long-edge multiscale dark-spot features plus histogram gradient boosting',
               'feature_names': FEATURE_NAMES,
               'fit_image_ids': fit_ids.tolist(), 'development_image_ids': dev_ids.tolist(),
               'test_image_ids': testing.image_id.tolist(),
               'fit_candidates': int(len(x)), 'fit_positive_candidates': int(y.sum()),
               'model_sha256': sha256_file(model_path),
               'settings': settings,
               'selected_on_development': selected.to_dict(),
               'official_test_at_locked_threshold': metrics_for_threshold(test, threshold),
               'paired_baseline_comparisons': comparisons,
               'false_positive_budget_per_image': args.fp_budget,
               'limitations': ['Official IDRiD test results were exposed in earlier development.',
                               'No portable-camera or external lesion validation.',
                               'Detector operates at 2048-long-edge, not native sensor resolution.',
                               'The active MATLAB M2 path remains the classical detector.']}
    (args.output / 'summary.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary['official_test_at_locked_threshold'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT / 'data/idrid/segmentation_manifest.csv')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/verification_2026-09-24/ma_native_candidate')
    parser.add_argument('--longest-edge', type=int, default=2048)
    parser.add_argument('--floor', type=int, default=28)
    parser.add_argument('--max-candidates', type=int, default=16000)
    parser.add_argument('--maximum-negatives-per-image', type=int, default=2500)
    parser.add_argument('--fp-budget', type=float, default=25)
    run(parser.parse_args())
