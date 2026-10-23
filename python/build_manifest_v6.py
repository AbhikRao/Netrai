#!/usr/bin/env python3
"""Manifest v6: round-5 manifest plus patient/cluster-level development holdouts (TD).

Identical photographs (same exported PNG hash) are first merged with the patient/exam groups into
``cluster_id`` so an image and its exact duplicate can never fall on different sides of a split.
Rows of duplicate groups with conflicting grades, and training rows that duplicate a validation
image, are set to ``excluded``. For every training source a fixed fraction of clusters then moves
from ``train`` to ``td``. Existing ``validation`` rows are kept as they
are. New training runs fit on ``train`` only and select epochs/recipes/policies on ``td`` +
``validation``; leave-one-dataset-out folds additionally withhold every row of one source.

CAUTION: checkpoints R3-R5 were trained on rows that are ``td`` here, so TD scores of those
models are training-contaminated. TD is only valid for models trained with manifest v6.

The output is written once; an existing manifest is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MI = ROOT / 'results/model_improvement'
EXPORTS = {'768': MI / 'train_export_768/images', 'r4': MI / 'train_export_768_r4/images',
           'r5': MI / 'train_export_768_r5/images'}
LODO_SOURCES = ('messidor', 'fdr26', 'odir', 'deepdrid')  # folds of the leave-one-dataset-out study
OPTIONAL_LODO_SOURCES = ('aptos', 'idrid')
SPLITS = ('train', 'td', 'validation', 'excluded')
MIN_TD_PER_CLASS = 10
REQUIRED_COLUMNS = ('image_id', 'grade', 'group_id', 'source', 'split')


def add_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    """``cluster_id``: connected components of rows sharing a group_id or an exported-image hash."""
    frame = frame.copy()
    parent = list(range(len(frame)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    keys = ['group_id'] + (['png_sha256'] if 'png_sha256' in frame else [])
    for key in keys:
        first = {}
        for position, value in enumerate(frame[key].tolist()):
            if value in first:
                union(first[value], position)
            else:
                first[value] = position
    roots = [find(i) for i in range(len(frame))]
    frame['cluster_id'] = [f'k{r}' for r in roots]
    return frame


def apply_exclusions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Set ``excluded`` for (a) duplicate-photo groups with conflicting grades, (b) training rows that
    duplicate a validation image. Returns the frame and counts for the record."""
    frame = frame.copy()
    counts = {'conflicting_duplicate_rows': 0, 'train_duplicates_of_validation': 0}
    if 'png_sha256' not in frame:
        return frame, counts
    by_hash = frame.groupby('png_sha256')
    conflicting = by_hash.grade.transform('nunique') > 1
    frame.loc[conflicting, 'split'] = 'excluded'
    counts['conflicting_duplicate_rows'] = int(conflicting.sum())
    validation_hashes = set(frame.loc[frame.split == 'validation', 'png_sha256'])
    leaking = (frame.split == 'train') & frame.png_sha256.isin(validation_hashes)
    frame.loc[leaking, 'split'] = 'excluded'
    counts['train_duplicates_of_validation'] = int(leaking.sum())
    return frame, counts


def check_manifest(frame: pd.DataFrame, allowed_splits=SPLITS) -> None:
    """Invariants every manifest must satisfy; raises ValueError on the first violation."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame]
    if missing:
        raise ValueError(f'Manifest lacks columns {missing}')
    if frame.image_id.duplicated().any():
        raise ValueError('Duplicate image ids')
    if not set(frame.split) <= set(allowed_splits):
        raise ValueError(f'Unknown splits {sorted(set(frame.split) - set(allowed_splits))}')
    if not frame.grade.between(0, 4).all():
        raise ValueError('Grades must lie in 0..4')
    live = frame[frame.split != 'excluded']  # excluded rows belong to no split
    for key in ('group_id', 'cluster_id', 'png_sha256'):
        if key in live:
            crossing = live.groupby(key).split.nunique()
            if (crossing > 1).any():
                raise ValueError(f'{int((crossing > 1).sum())} {key} values cross splits, '
                                 f'e.g. {crossing[crossing > 1].index[:3].tolist()}')
    mixed = frame.groupby('group_id').source.nunique()
    if (mixed > 1).any():
        raise ValueError('A group spans several sources')


def carve_td(frame: pd.DataFrame, fraction: float = 0.10, seed: int = 20261020) -> pd.DataFrame:
    """Move ``fraction`` of each source's training clusters to ``td`` (deterministic per source and seed).

    Uses ``cluster_id`` when present (identical photographs stay together), else ``group_id``.
    """
    if not 0 < fraction < 0.5:
        raise ValueError('fraction must be in (0, 0.5)')
    frame = frame.copy()
    key = 'cluster_id' if 'cluster_id' in frame else 'group_id'
    for source, part in frame.loc[frame.split == 'train'].groupby('source'):
        groups = np.sort(part[key].unique())
        rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(source.encode())]))
        chosen = set(rng.choice(groups, size=max(1, round(fraction * len(groups))), replace=False))
        frame.loc[(frame.source == source) & frame[key].isin(chosen) & (frame.split == 'train'), 'split'] = 'td'
    return frame


def check_td_coverage(frame: pd.DataFrame, minimum: int = MIN_TD_PER_CLASS) -> dict:
    """Every source must have enough referable and non-referable TD images to be informative."""
    coverage = {}
    for source, part in frame.loc[frame.split == 'td'].groupby('source'):
        referable = int((part.grade >= 2).sum())
        coverage[source] = {'images': len(part), 'referable': referable, 'non_referable': len(part) - referable,
                            'grade_counts': np.bincount(part.grade, minlength=5).tolist()}
        if referable < minimum or len(part) - referable < minimum:
            raise ValueError(f'TD holdout of {source} has too few images per referral class: {coverage[source]}')
    return coverage


def locate_exports(frame: pd.DataFrame, exports: dict) -> pd.Series:
    names = {key: set(p.name for p in folder.iterdir()) for key, folder in exports.items()}
    where = []
    for image_id in frame.image_id:
        found = next((key for key in exports if f'{image_id}.png' in names[key]), None)
        if found is None:
            raise FileNotFoundError(f'No exported PNG for {image_id}')
        where.append(found)
    return pd.Series(where, index=frame.index)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(frame: pd.DataFrame) -> dict:
    out = {}
    for (split, source), part in frame.groupby(['split', 'source']):
        out[f'{split}/{source}'] = {'images': len(part), 'groups': int(part.group_id.nunique()),
                                    'grade_counts': np.bincount(part.grade, minlength=5).tolist()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=MI / 'train_export_768_r5/manifest.csv')
    parser.add_argument('--output', type=Path, default=MI / 'manifest_v6')
    parser.add_argument('--fraction', type=float, default=0.10)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--skip-file-check', action='store_true', help='do not verify the exported PNGs exist')
    args = parser.parse_args()
    target = args.output / 'manifest_v6.csv'
    if target.exists():
        raise FileExistsError(f'{target} exists; manifests are immutable')
    base = pd.read_csv(args.input)
    check_manifest(base.drop(columns=['png_sha256'], errors='ignore'), ('train', 'validation'))
    shared = base.groupby('png_sha256').split.nunique() if 'png_sha256' in base else pd.Series(dtype=int)
    input_cross_split_hashes = int((shared > 1).sum())  # pre-existing leaks in the round-5 manifest
    clustered = add_clusters(base)
    cleaned, exclusions = apply_exclusions(clustered)
    frame = carve_td(cleaned, args.fraction, args.seed)
    check_manifest(frame)
    untouched = frame.split == 'validation'
    if not (untouched.sum() + int(((cleaned.split == 'excluded') & (base.split == 'validation')).sum())
            == int((base.split == 'validation').sum())):
        raise AssertionError('validation rows changed other than by conflicting-duplicate exclusion')
    coverage = check_td_coverage(frame)
    frame['lodo_source'] = frame.source.isin(LODO_SOURCES + OPTIONAL_LODO_SOURCES)
    if not args.skip_file_check:
        frame['export'] = locate_exports(frame, EXPORTS)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    atomic_json(args.output / 'manifest_v6_summary.json', {
        'schema_version': 1, 'input': str(args.input.relative_to(ROOT)), 'input_sha256': sha256_file(args.input),
        'manifest_sha256': sha256_file(target), 'td_fraction': args.fraction, 'seed': args.seed,
        'rows': len(frame), 'splits': frame.split.value_counts().to_dict(),
        'lodo_sources': list(LODO_SOURCES), 'optional_lodo_sources': list(OPTIONAL_LODO_SOURCES),
        'exclusions': exclusions, 'input_cross_split_duplicate_hashes': input_cross_split_hashes, 'clusters': int(frame.cluster_id.nunique()),
        'td_coverage': coverage, 'by_split_source': summarize(frame),
        'caution': 'R3-R5 checkpoints were trained on rows that are td here; TD scores of those models are '
                   'training-contaminated. TD is valid only for models trained with manifest v6.'})
    print(json.dumps({'rows': len(frame), 'splits': frame.split.value_counts().to_dict(), 'td_coverage': coverage}, indent=1))


if __name__ == '__main__':
    main()
