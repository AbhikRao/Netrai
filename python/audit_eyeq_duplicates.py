#!/usr/bin/env python3
"""Audit exact byte-identical files across EyeQ's official train/test splits."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return path.name, value.hexdigest()


def hashes(folder, workers):
    paths = sorted(path for path in folder.iterdir() if path.is_file())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(digest, paths))


def run(args):
    train = hashes(args.eyeq / 'images/train', args.workers)
    test = hashes(args.eyeq / 'images/test', args.workers)
    train_by_hash = {}
    for name, value in train.items():
        train_by_hash.setdefault(value, []).append(name)
    test_by_hash = {}
    for name, value in test.items():
        test_by_hash.setdefault(value, []).append(name)
    shared = sorted(set(train_by_hash) & set(test_by_hash))
    report = {
        'schema_version': 1,
        'method': 'SHA-256 of encoded source file bytes',
        'train_files': len(train), 'test_files': len(test),
        'cross_split_duplicate_hashes': len(shared),
        'cross_split_duplicate_files': [
            {'sha256': value, 'train': train_by_hash[value], 'test': test_by_hash[value]}
            for value in shared
        ],
        'within_train_duplicate_hashes': sum(len(names) > 1 for names in train_by_hash.values()),
        'within_test_duplicate_hashes': sum(len(names) > 1 for names in test_by_hash.values()),
        'limitation': 'Exact encoded-file duplicates only; perceptual near-duplicates are not detected.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eyeq', type=Path, default=ROOT / 'data/eyeq')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/verification_2026-09-21/eyeq_learned_quality/exact_duplicate_audit.json')
    run(parser.parse_args())
