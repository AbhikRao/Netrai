#!/usr/bin/env python3
"""Package only corrected grading source for a user-run free GPU notebook.

No datasets, legacy APTOS checkpoint, patient images, cache or credentials.
"""
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'GPU_RUNBOOK.md',
    'python/prepare_training_manifest.py',
    'python/train_grading.py',
    'python/run_grading_study.py',
    'python/evaluate_grading_candidate.py',
    'python/models/__init__.py',
    'python/models/netrai_model.py',
    'python/models/losses.py',
    'python/utils/__init__.py',
    'python/utils/artifacts.py',
    'python/utils/calibration.py',
    'python/utils/metrics.py',
    'python/utils/preprocessing.py',
    'python/utils/validation.py',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh bundle path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, name)
    print(f'GPU source bundle: {args.output} ({args.output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
