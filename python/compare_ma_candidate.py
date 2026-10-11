#!/usr/bin/env python3
"""Recompare the frozen 2048 candidate with the real rerun of classic M2.

Both sources are retrospective, test-exposed IDRiD evidence. The trained
candidate's operating threshold remains the one fixed on training images.
"""
import argparse
from pathlib import Path

import pandas as pd

from train_ma_native import paired_baseline_comparison
from utils.artifacts import atomic_json, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT / 'results/verification_2026-09-24/ma_native_candidate/test_per_image_at_locked_threshold.csv')
    parser.add_argument('--baseline', type=Path, default=ROOT / 'results/implementation_2026-09-24/ma_detector_rerun/ma_froc_per_image.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh comparison output path')
    candidate = pd.read_csv(args.candidate)
    if len(candidate) != 27 or candidate.image_id.duplicated().any() or candidate.threshold.nunique() != 1:
        raise ValueError('Expected 27 unique images at one frozen candidate threshold')
    payload = {
        'schema_version': 1,
        'status': 'retrospective_paired_development_comparison_not_clinical_validation',
        'candidate_source': str(args.candidate.resolve()),
        'candidate_source_sha256': sha256_file(args.candidate),
        'candidate_threshold_selected_on_training_subset': float(candidate.threshold.iloc[0]),
        'baseline_protocol': 'actual_512_detector_rerun_per_threshold_native_coordinate_matching',
        'classic_sigma_2': paired_baseline_comparison(candidate, args.baseline, 2.0),
        'classic_sigma_3_default': paired_baseline_comparison(candidate, args.baseline, 3.0),
        'limitations': ['The IDRiD test masks were previously inspected during project development',
                        'Point FROC does not establish lesion-boundary segmentation or clinical utility'],
    }
    atomic_json(args.output, payload)
    print(f'Comparison written to {args.output}')


if __name__ == '__main__':
    main()
