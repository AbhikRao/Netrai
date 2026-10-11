#!/usr/bin/env python3
"""Create explicit fit/selection/calibration manifests from a development CSV.

No input supplied here is declared an untouched test set. Patient grouping must
come from a verified column; without it independence is limited to file identity.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from utils.artifacts import atomic_json, sha256_file
from utils.validation import grades, image_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--id-column', default='id_code')
    parser.add_argument('--label-column', default='diagnosis')
    parser.add_argument('--group-column')
    parser.add_argument('--extension', default='.png')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--conflicting-labels', choices=('error', 'quarantine'), default='error')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new manifest path; existing protocols are immutable')
    source = pd.read_csv(args.csv, dtype={args.id_column: str})
    frame = pd.DataFrame({'image_id': image_ids(source[args.id_column]),
                          'grade': grades(source[args.label_column])})
    paths = [args.image_dir.resolve() / (name if Path(name).suffix else name + args.extension)
             for name in frame.image_id]
    frame['image_path'] = [str(path) for path in paths]
    frame['sha256'] = [sha256_file(path) for path in paths]
    if args.group_column:
        if source[args.group_column].isna().any():
            raise ValueError('Missing patient/group IDs')
        frame['group_id'] = source[args.group_column].astype(str)
        group_kind = 'provided_identity_not_independently_verified'
    else:
        frame['group_id'] = frame.sha256
        group_kind = 'file_identity_only_patient_independence_unknown'
    # Merge file duplicates and supplied groups transitively before partitioning.
    parent = {group: group for group in frame.group_id}
    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    conflicts = []
    for digest, duplicate in frame.groupby('sha256'):
        if duplicate.grade.nunique() != 1:
            conflicts.append(digest)
        first = find(duplicate.group_id.iloc[0])
        for group in duplicate.group_id:
            parent[find(group)] = first
    frame['group_id'] = frame.group_id.map(find)
    excluded_groups = set(frame.loc[frame.sha256.isin(conflicts), 'group_id'])
    quarantined = frame.loc[frame.group_id.isin(excluded_groups)].copy()
    atomic_json(args.output.with_suffix('.conflicts.json'), {
        'conflicting_hashes': conflicts, 'quarantined': quarantined.to_dict('records'),
        'policy': args.conflicting_labels, 'source_images_modified': False})
    if conflicts and args.conflicting_labels == 'error':
        raise ValueError('Identical content has conflicting grades; see conflicts.json. '
                         'Review or explicitly use --conflicting-labels quarantine')
    frame = frame.loc[~frame.group_id.isin(excluded_groups)].reset_index(drop=True)
    frame['split'] = 'fit'
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold, (_, indices) in enumerate(splitter.split(frame, frame.grade, frame.group_id)):
        if fold == 0:
            frame.loc[indices, 'split'] = 'selection'
        elif fold == 1:
            frame.loc[indices, 'split'] = 'calibration'
    # Store paths relative to the manifest to permit relocating the whole tree.
    import os
    frame['image_path'] = [os.path.relpath(path, args.output.resolve().parent) for path in frame.image_path]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    atomic_json(args.output.with_suffix('.protocol.json'), {
        'schema_version': 1, 'manifest_sha256': sha256_file(args.output),
        'source_csv_sha256': sha256_file(args.csv), 'seed': args.seed,
        'group_kind': group_kind, 'split_method': 'grouped stratified five folds: 3 fit / 1 selection / 1 calibration',
        'counts': frame.groupby('split').size().to_dict(),
        'quarantined_rows': len(quarantined), 'conflicting_hash_groups': len(conflicts),
        'exposure': 'Development data only; no fresh final-test claim',
        'limitations': ['Content grouping cannot detect all transformed same-retina images',
                       'Verified identity/perceptual review and a fresh final cohort remain required'],
    })


if __name__ == '__main__':
    main()
