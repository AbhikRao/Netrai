#!/usr/bin/env python3
"""Cluster fundus images that show the same retina, using vessel-pattern correlation (label-free).

Images whose 128-px FOV-cropped green high-pass thumbnails correlate at >= ``--threshold`` (default 0.6,
the level where ODIR/DDR copies and same-eye re-captures cluster) are joined (single linkage, also against
the mirrored image). Use the clusters to split data so one retina never lands on both sides of a split, and
to list test images that have a near-copy among training images (``--against``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from identity_review_vessel import listing, vectors


def cluster_labels(similarity: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of ``similarity >= threshold`` (symmetric matrix, diagonal ignored)."""
    n = len(similarity)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    rows, cols = np.nonzero(np.triu(similarity >= threshold, k=1))
    for a, b in zip(rows, cols):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def similarity_matrix(vecs: np.ndarray, size: int = 128) -> np.ndarray:
    """Max of plain and mirrored correlation for every pair."""
    mirrored = vecs.reshape(-1, size, size)[:, :, ::-1].reshape(len(vecs), -1)
    return np.maximum(vecs @ vecs.T, vecs @ mirrored.T)


def near_copies(test_vecs: np.ndarray, reference_vecs: np.ndarray, size: int = 128):
    """For each test vector: best correlation against the references and that reference's index."""
    mirrored = test_vecs.reshape(-1, size, size)[:, :, ::-1].reshape(len(test_vecs), -1)
    plain, flipped = test_vecs @ reference_vecs.T, mirrored @ reference_vecs.T
    best = np.maximum(plain, flipped)
    return best.max(axis=1), best.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, nargs='+', required=True, help='folders to cluster together')
    parser.add_argument('--against', type=Path, nargs='*', default=[],
                        help='optional reference folders; reports near-copies of --images among them')
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--output', type=Path, required=True, help='directory (created); never overwritten')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if (args.output / 'clusters.csv').exists():
        raise FileExistsError(args.output / 'clusters.csv')
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [p for folder in args.images for p in listing(folder)]
    vecs = vectors(paths, 128, args.workers).astype(np.float32)
    labels = cluster_labels(similarity_matrix(vecs), args.threshold)
    frame = pd.DataFrame({'image': [str(p) for p in paths], 'cluster': labels})
    sizes = frame.groupby('cluster').cluster.transform('size')
    frame['cluster_size'] = sizes
    frame.to_csv(args.output / 'clusters.csv', index=False)
    print(f'{len(paths)} images -> {frame.cluster.nunique()} clusters; '
          f'{int((sizes > 1).sum())} images share a cluster; largest cluster {int(sizes.max())}')
    if args.against:
        refs = [p for folder in args.against for p in listing(folder)]
        ref_vecs = vectors(refs, 128, args.workers).astype(np.float32)
        best, which = near_copies(vecs, ref_vecs)
        out = pd.DataFrame({'image': [str(p) for p in paths], 'max_correlation': best,
                            'nearest_reference': [str(refs[i]) for i in which]}).sort_values('max_correlation', ascending=False)
        out.to_csv(args.output / 'near_copies_against_reference.csv', index=False)
        print(f'{int((best >= args.threshold).sum())} of {len(paths)} images have a reference at >= {args.threshold}')


if __name__ == '__main__':
    main()
