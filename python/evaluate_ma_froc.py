#!/usr/bin/env python3
"""Evaluate fixed M2 microaneurysm proposals as lesion sensitivity vs FP/image.

No threshold is selected on IDRiD's test split. By default the actual detector
is rerun at each threshold. A fixed-proposal curve is explicitly opt-in. Native-space
candidate points are matched within 20 source-image pixels of an annotated
connected component; each annotation can be matched once. This is development evidence, not
sub-pixel or clinical detection validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from evaluate_lesion_masks import resolve
from modules.segmentation import (
    detect_microaneurysms, localize_optic_disc, segment_vessels,
)
from utils.preprocessing import preprocess_fundus


THRESHOLDS = (1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10)


def match_candidates(centres, scores, truth, threshold, tolerance=20, labeled=None):
    """Maximum-cardinality one-to-one matches; invariant to annotation numbering."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if len(centres) != len(scores) or tolerance < 0:
        raise ValueError('Invalid candidate lengths or tolerance')
    if not np.isfinite(np.asarray(centres)).all() or not np.isfinite(scores).all():
        raise ValueError('Candidate coordinates and scores must be finite')
    count, labels = labeled if labeled is not None else cv2.connectedComponents(
        truth.astype(np.uint8), 8)
    selected = [point for point, score in zip(centres, scores) if score >= threshold]
    rows, columns = [], []
    height, width = truth.shape
    for index, (x, y) in enumerate(selected):
        x0, x1 = max(0, int(np.floor(x-tolerance))), min(width, int(np.ceil(x+tolerance))+1)
        y0, y1 = max(0, int(np.floor(y-tolerance))), min(height, int(np.ceil(y+tolerance))+1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        nearby = (xx-x)**2 + (yy-y)**2 <= tolerance**2
        candidates = np.unique(labels[y0:y1, x0:x1][nearby])
        for label in candidates[candidates > 0]:
            rows.append(index)
            columns.append(int(label)-1)
    if not selected or count <= 1:
        return 0, len(selected), count-1
    graph = csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(selected), count-1))
    matched = maximum_bipartite_matching(graph, perm_type='column')
    hits = int(np.sum(matched >= 0))
    return hits, len(selected)-hits, count-1


def native_coordinates(centres, width, height, size):
    """Undo the exact integer resize and symmetric padding of preprocess_fundus."""
    scale = size / max(height, width)
    resized_width, resized_height = int(width*scale), int(height*scale)
    left, top = (size-resized_width)//2, (size-resized_height)//2
    return [((x-left+0.5)*width/resized_width-0.5,
             (y-top+0.5)*height/resized_height-0.5)
            for x, y in centres]


def run(args):
    if args.analysis_size < 128:
        raise ValueError('analysis-size must be >=128')
    manifest = pd.read_csv(args.manifest)
    required = {'image_id', 'split', 'image_path', 'ma_mask'}
    if not required.issubset(manifest):
        raise ValueError(f'manifest requires {sorted(required)}')
    if manifest.image_id.duplicated().any():
        raise ValueError('Duplicate image IDs in manifest')
    base = args.manifest.resolve().parent
    rows, proposals = [], []
    for item in tqdm(manifest.itertuples(index=False), total=len(manifest), desc='MA FROC'):
        bgr = cv2.imread(str(resolve(base, item.image_path)), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(resolve(base, item.ma_mask)), cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            raise FileNotFoundError(f'Missing image or MA annotation for {item.image_id}')
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        processed, green, fov = preprocess_fundus(rgb, args.analysis_size)
        disc, radius = localize_optic_disc(processed)
        vessels = segment_vessels(green, fov)
        truth = mask > 0
        labeled = cv2.connectedComponents(truth.astype(np.uint8), 8)
        mode = getattr(args, 'proposal_mode', 'detector-rerun')
        fixed = None
        for threshold in THRESHOLDS:
            if mode == 'detector-rerun' or fixed is None:
                generated = detect_microaneurysms(
                    green, vessels, fov, disc, radius,
                    threshold_sigma=threshold if mode == 'detector-rerun' else THRESHOLDS[0],
                    return_scores=True)
                _, _, centres, scores = generated
                natives = native_coordinates(centres, width, height, args.analysis_size)
                fixed = (natives, scores)
                for (nx, ny), score in zip(natives, scores):
                    proposals.append({'image_id': item.image_id, 'split': item.split,
                                      'generation_threshold': threshold,
                                      'native_x': nx, 'native_y': ny, 'score': score})
            hits, fp, total = match_candidates(
                natives, scores, truth, threshold if mode == 'fixed-proposals' else -1e30,
                tolerance=20, labeled=labeled)
            rows.append({'image_id': item.image_id, 'split': item.split,
                         'threshold': threshold, 'lesion_hits': hits,
                         'false_positives': fp, 'lesion_total': total})
    per_image = pd.DataFrame(rows)
    summary = []
    for (split, threshold), group in per_image.groupby(['split', 'threshold']):
        summary.append({'split': split, 'threshold': threshold, 'images': len(group),
                        'annotated_lesions': int(group.lesion_total.sum()),
                        'sensitivity': float(group.lesion_hits.sum()/max(group.lesion_total.sum(), 1)),
                        'false_positives_per_image': float(group.false_positives.mean())})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(output / 'ma_froc_per_image.csv', index=False)
    pd.DataFrame(proposals).to_csv(output / 'ma_proposals.csv', index=False)
    pd.DataFrame(summary).to_csv(output / 'ma_froc.csv', index=False)
    report = {'schema_version': 2, 'analysis_size': args.analysis_size,
              'proposal_mode': mode, 'matching': 'maximum_cardinality_component_distance',
              'proposal_floor_sigma': THRESHOLDS[0], 'thresholds': THRESHOLDS,
              'match_tolerance_native_pixels': 20,
              'status': 'development_benchmark_not_clinical_validation',
              'limitations': ['Fixed-scale analysis; native coordinates do not imply native-resolution detection.',
                              'IDRiD test results have already been inspected in project development.',
                              'No threshold selected on the official test split.']}
    (output / 'protocol.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data/idrid/segmentation_manifest.csv')
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results/verification_2026-09-24/ma_froc')
    parser.add_argument('--analysis-size', type=int, default=512)
    parser.add_argument('--proposal-mode', choices=('detector-rerun', 'fixed-proposals'),
                        default='detector-rerun')
    run(parser.parse_args())
