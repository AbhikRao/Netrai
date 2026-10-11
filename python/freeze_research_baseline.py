#!/usr/bin/env python3
"""Freeze source/artifact identity without copying datasets or overwriting evidence."""
import argparse
import subprocess
from pathlib import Path

from utils.artifacts import atomic_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Baseline exists; choose a new immutable output path')
    root = Path(__file__).resolve().parents[1]
    paths = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root).decode().split('\0')
    hashes = {name: sha256_file(root / name) for name in paths if name and (root / name).is_file()}
    atomic_json(args.output, {
        'schema_version': 1,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=root).decode(),
        'files': hashes,
        'scope': 'Versioned/unignored source and evidence, not raw data; dirty state explicitly retained',
        'exposure': {'APTOS_fold0': 'checkpoint and ordinal-threshold selection; development only',
                     'IDRiD_test': 'repeated development exposure, including adaptation comparisons',
                     'EyeQ_test': 'development-exposed; five confirmed cross-split same-retina pairs'},
        'legacy_preprocessing': 'legacy-area-v1',
    })


if __name__ == '__main__':
    main()
