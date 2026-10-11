#!/usr/bin/env python3
"""Add ORB/RANSAC geometry evidence to the closest pHash overlap candidates.

A geometrically consistent pair is still a review candidate, not a confirmed
same-patient image. This consumes only the bounded candidate CSV, so absence
of matches cannot exclude duplication beyond the pHash search radius.
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from calibrate_quality import file_sha256


ROOT = Path(__file__).resolve().parents[1]


def source_roots():
    return {
        'eyeq_train': ROOT/'data/eyeq/images/train',
        'eyeq_test': ROOT/'data/eyeq/images/test',
        'aptos_train': ROOT/'data/aptos2019/train_images',
        'idrid_grading': ROOT/'data/idrid/B. Disease Grading/1. Original Images',
    }


def resolve_image(roots, source, relative_path):
    root = roots[source].resolve()
    path = (root/relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f'Invalid candidate path: {source}/{relative_path}')
    return path


@lru_cache(maxsize=4096)
def features(path):
    image = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if image is None:
        raise FileNotFoundError(path)
    mask = image > 8
    if mask.any():
        ys, xs = mask.nonzero()
        image = image[ys.min():ys.max()+1, xs.min():xs.max()+1]
    scale = min(1.0, 768/max(image.shape))
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(nfeatures=1000)
    return orb.detectAndCompute(image, None)


def geometric_evidence(first, second):
    keypoints_a, descriptor_a = features(str(first))
    keypoints_b, descriptor_b = features(str(second))
    if descriptor_a is None or descriptor_b is None or len(descriptor_a) < 2 or len(descriptor_b) < 2:
        return {'ratio_test_matches': 0, 'ransac_inliers': 0,
                'geometrically_consistent': False}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(descriptor_a, descriptor_b, k=2)
    good = [candidate for pair in pairs if len(pair) == 2
            for candidate, alternative in [pair]
            if candidate.distance < .75*alternative.distance]
    if len(good) < 4:
        return {'ratio_test_matches': len(good), 'ransac_inliers': 0,
                'geometrically_consistent': False}
    source = np.float32([keypoints_a[match.queryIdx].pt for match in good])
    target = np.float32([keypoints_b[match.trainIdx].pt for match in good])
    _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 5.0)
    count = int(inliers.sum()) if inliers is not None else 0
    return {'ratio_test_matches': len(good), 'ransac_inliers': count,
            'geometrically_consistent': bool(count >= 25 and count/max(len(good), 1) >= .5)}


def run(args):
    cv2.setNumThreads(1)
    candidates = pd.read_csv(args.candidates)
    required = {'source_a', 'source_b', 'image_a', 'image_b', 'hamming_distance'}
    if not required.issubset(candidates):
        raise ValueError(f'Candidate CSV requires {sorted(required)}')
    selected = candidates.loc[candidates.hamming_distance <= args.max_distance].copy()
    roots = source_roots()
    rows = []
    for item in selected.itertuples(index=False):
        first = resolve_image(roots, item.source_a, item.image_a)
        second = resolve_image(roots, item.source_b, item.image_b)
        rows.append({'source_a': item.source_a, 'source_b': item.source_b,
                     'image_a': item.image_a, 'image_b': item.image_b,
                     'hamming_distance': int(item.hamming_distance),
                     **geometric_evidence(first, second)})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'geometric_candidate_review.csv', index=False)
    report = {'schema_version': 1,
              'source_candidate_csv_sha256': file_sha256(args.candidates),
              'phash_max_hamming_distance_reviewed': args.max_distance,
              'pairs_reviewed': len(rows),
              'geometrically_consistent_candidates': sum(
                  row['geometrically_consistent'] for row in rows),
              'status': 'algorithmic_triage_not_identity_confirmation',
              'limitations': ['Only stored pHash candidate pairs within the selected radius are checked.',
                              'ORB/RANSAC can miss same-retina images after severe blur/crop/rotation.',
                              'Geometric consistency needs human adjudication before leakage claims.']}
    (output/'geometric_candidate_review.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    folder = ROOT/'results/verification_2026-09-24/perceptual_overlap'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path,
                        default=folder/'perceptual_overlap_candidates.csv')
    parser.add_argument('--output', type=Path, default=folder)
    parser.add_argument('--max-distance', type=int, default=2)
    run(parser.parse_args())
