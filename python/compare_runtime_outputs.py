#!/usr/bin/env python3
"""Compare Python and MATLAB validation exports on the same retinal images.

The comparison separates screening-decision parity from strict five-grade
parity.  That distinction matters because a small floating-point or image
preprocessing difference can cross an ordinal grade threshold without changing
the referable-DR decision or the four-way triage action.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIR.parent
DEFAULT_THRESHOLDS = PYTHON_DIR / 'weights' / 'fold0_thresholds.npy'

EXACT_FIELDS = {
    'true_grade': ('true_grade', 'true_grade', 'integer'),
    'predicted_grade': ('predicted_grade', 'predicted_grade', 'integer'),
    'referable_head_prediction': (
        'referable_head_prediction', 'referable_head_prediction', 'boolean'),
    'head_disagreement': ('head_disagreement', 'head_disagreement', 'boolean'),
    'triage_action': ('triage_action', 'triage_action', 'string'),
}

CONTINUOUS_FIELDS = {
    'continuous_score': ('continuous_score', 'continuous_score'),
    'confidence': ('confidence', 'confidence'),
    'referable_probability': ('referable_probability', 'referable_probability'),
    'calibrated_referable_probability': (
        'calibrated_referable_probability', 'calibrated_referable_probability'),
    'referable_head_probability': (
        'referable_head_probability', 'referable_head_probability'),
    'iqs': ('iqs', 'iqs'),
}

for _grade in range(5):
    CONTINUOUS_FIELDS[f'raw_prob_grade_{_grade}'] = (
        f'prob_grade_{_grade}', f'raw_prob_grade_{_grade}')
    CONTINUOUS_FIELDS[f'calibrated_prob_grade_{_grade}'] = (
        f'calibrated_prob_grade_{_grade}', f'prob_grade_{_grade}')


def _boolean_series(series, name):
    """Normalize common CSV boolean encodings and reject ambiguous values."""
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        '1': True, '1.0': True, 'true': True,
        '0': False, '0.0': False, 'false': False,
    }
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f'Unsupported boolean values in {name}: {invalid}')
    return normalized.map(mapping).astype(bool)


def _load_export(path, runtime):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'{runtime} CSV not found: {path}')
    frame = pd.read_csv(path, dtype={'id_code': str})
    if 'id_code' not in frame:
        raise ValueError(f'{runtime} CSV is missing id_code')
    frame['id_code'] = frame['id_code'].astype(str).str.strip()
    if frame['id_code'].eq('').any():
        raise ValueError(f'{runtime} CSV contains an empty id_code')
    duplicates = frame.loc[frame['id_code'].duplicated(), 'id_code'].tolist()
    if duplicates:
        raise ValueError(f'{runtime} CSV contains duplicate ids: {duplicates[:5]}')
    return frame.set_index('id_code', drop=False)


def _require_columns(frame, columns, runtime):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f'{runtime} CSV is missing columns: {missing}')


def _difference_stats(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {'count': 0, 'mean_abs_difference': None,
                'p95_abs_difference': None, 'max_abs_difference': None}
    return {
        'count': int(finite.size),
        'mean_abs_difference': float(np.mean(finite)),
        'p95_abs_difference': float(np.percentile(finite, 95)),
        'max_abs_difference': float(np.max(finite)),
    }


def compare_runtime_outputs(python_csv, matlab_csv, thresholds=None,
                            probability_tolerance=0.05,
                            threshold_edge_tolerance=0.05):
    """Return a machine-readable summary and per-image comparison table."""
    python_frame = _load_export(python_csv, 'Python')
    matlab_frame = _load_export(matlab_csv, 'MATLAB')

    python_columns = [value[0] for value in EXACT_FIELDS.values()]
    matlab_columns = [value[1] for value in EXACT_FIELDS.values()]
    python_columns += [value[0] for value in CONTINUOUS_FIELDS.values()]
    matlab_columns += [value[1] for value in CONTINUOUS_FIELDS.values()]
    _require_columns(python_frame, python_columns, 'Python')
    _require_columns(matlab_frame, matlab_columns, 'MATLAB')

    matlab_ids = set(matlab_frame.index)
    python_ids = set(python_frame.index)
    matched_ids = [image_id for image_id in matlab_frame.index if image_id in python_ids]
    matlab_only = sorted(matlab_ids - python_ids)
    python_only = sorted(python_ids - matlab_ids)

    if not matched_ids:
        raise ValueError('The runtime exports have no image ids in common')

    py = python_frame.loc[matched_ids].copy()
    ml = matlab_frame.loc[matched_ids].copy()
    if 'status' in py and 'status' in ml:
        successful = py['status'].eq('success') & ml['status'].eq('success')
    else:
        successful = pd.Series(True, index=py.index)
    compared_ids = list(py.index[successful])
    if not compared_ids:
        raise ValueError('No matched images succeeded in both runtimes')
    py = py.loc[compared_ids]
    ml = ml.loc[compared_ids]

    if thresholds is None:
        threshold_values = np.load(DEFAULT_THRESHOLDS).astype(float)
    else:
        threshold_values = np.asarray(thresholds, dtype=float)
    if threshold_values.shape != (4,) or not np.all(np.diff(threshold_values) > 0):
        raise ValueError('Expected four strictly increasing ordinal thresholds')

    details = pd.DataFrame({'id_code': compared_ids})
    exact_summary = {}
    exact_arrays = {}
    for label, (python_column, matlab_column, kind) in EXACT_FIELDS.items():
        if kind == 'boolean':
            python_values = _boolean_series(py[python_column], f'Python {python_column}')
            matlab_values = _boolean_series(ml[matlab_column], f'MATLAB {matlab_column}')
        elif kind == 'integer':
            python_values = pd.to_numeric(py[python_column], errors='raise').astype(int)
            matlab_values = pd.to_numeric(ml[matlab_column], errors='raise').astype(int)
        else:
            python_values = py[python_column].astype(str).str.strip()
            matlab_values = ml[matlab_column].astype(str).str.strip()
        matches = python_values.to_numpy() == matlab_values.to_numpy()
        exact_arrays[label] = matches
        details[f'python_{label}'] = python_values.to_numpy()
        details[f'matlab_{label}'] = matlab_values.to_numpy()
        details[f'{label}_match'] = matches
        exact_summary[label] = {
            'matches': int(np.sum(matches)),
            'mismatches': int(np.sum(~matches)),
            'match_rate': float(np.mean(matches)),
        }

    numerical_summary = {}
    numerical_differences = {}
    for label, (python_column, matlab_column) in CONTINUOUS_FIELDS.items():
        python_values = pd.to_numeric(py[python_column], errors='coerce').to_numpy(float)
        matlab_values = pd.to_numeric(ml[matlab_column], errors='coerce').to_numpy(float)
        differences = np.abs(python_values - matlab_values)
        numerical_differences[label] = differences
        details[f'python_{label}'] = python_values
        details[f'matlab_{label}'] = matlab_values
        details[f'{label}_abs_difference'] = differences
        numerical_summary[label] = _difference_stats(differences)

    python_scores = details['python_continuous_score'].to_numpy(float)
    matlab_scores = details['matlab_continuous_score'].to_numpy(float)
    grade_mismatch = ~exact_arrays['predicted_grade']
    crossed_threshold = np.full(len(details), np.nan)
    threshold_edge = np.zeros(len(details), dtype=bool)
    for index, (python_score, matlab_score) in enumerate(zip(python_scores, matlab_scores)):
        lower, upper = sorted((python_score, matlab_score))
        crossed = threshold_values[(threshold_values >= lower) & (threshold_values <= upper)]
        if crossed.size:
            selected = crossed[np.argmin(np.abs(crossed - np.mean([lower, upper])))]
            crossed_threshold[index] = selected
            threshold_edge[index] = bool(
                grade_mismatch[index]
                and max(abs(python_score - selected), abs(matlab_score - selected))
                <= threshold_edge_tolerance)
    details['crossed_ordinal_threshold'] = crossed_threshold
    details['threshold_edge_grade_mismatch'] = threshold_edge

    python_grade_referable = details['python_predicted_grade'].to_numpy(int) >= 2
    matlab_grade_referable = details['matlab_predicted_grade'].to_numpy(int) >= 2
    grade_referable_match = python_grade_referable == matlab_grade_referable
    details['python_grade_referable'] = python_grade_referable
    details['matlab_grade_referable'] = matlab_grade_referable
    details['grade_referable_match'] = grade_referable_match

    probability_fields = [
        key for key in numerical_differences
        if key.startswith('raw_prob_') or key.startswith('calibrated_prob_')
        or key in {
            'referable_probability', 'calibrated_referable_probability',
            'referable_head_probability',
        }
    ]
    probability_max = max(
        numerical_summary[field]['max_abs_difference'] or 0.0
        for field in probability_fields)
    all_matlab_rows_matched = not matlab_only
    all_compared = len(compared_ids) == len(matlab_frame)
    screening_fields = (
        grade_referable_match,
        exact_arrays['referable_head_prediction'],
        exact_arrays['head_disagreement'],
        exact_arrays['triage_action'],
    )
    screening_pass = bool(
        all_matlab_rows_matched and all_compared
        and all(np.all(values) for values in screening_fields))
    strict_grade_pass = bool(
        all_matlab_rows_matched and all_compared
        and np.all(exact_arrays['predicted_grade']))

    summary = {
        'schema_version': '1.0',
        'python_csv': str(Path(python_csv).resolve()),
        'matlab_csv': str(Path(matlab_csv).resolve()),
        'ordinal_thresholds': [float(value) for value in threshold_values],
        'tolerances': {
            'probability_absolute': float(probability_tolerance),
            'threshold_edge_score_distance': float(threshold_edge_tolerance),
        },
        'counts': {
            'python_rows': int(len(python_frame)),
            'matlab_rows': int(len(matlab_frame)),
            'matched_rows': int(len(matched_ids)),
            'compared_successful_rows': int(len(compared_ids)),
            'python_only_rows': int(len(python_only)),
            'matlab_only_rows': int(len(matlab_only)),
            'matched_rows_not_successful_in_both': int(len(matched_ids) - len(compared_ids)),
        },
        'unmatched': {
            'python_only_ids': python_only,
            'matlab_only_ids': matlab_only,
        },
        'exact_comparisons': exact_summary,
        'derived_screening_comparisons': {
            'grade_referable': {
                'matches': int(np.sum(grade_referable_match)),
                'mismatches': int(np.sum(~grade_referable_match)),
                'match_rate': float(np.mean(grade_referable_match)),
            },
        },
        'numerical_comparisons': numerical_summary,
        'threshold_edge_grade_mismatches': int(np.sum(threshold_edge)),
        'maximum_probability_abs_difference': float(probability_max),
        'gates': {
            'screening_decision_parity_pass': screening_pass,
            'strict_five_grade_parity_pass': strict_grade_pass,
            'probability_tolerance_pass': bool(probability_max <= probability_tolerance),
        },
        'interpretation': (
            'Screening parity covers grade-derived referability, the independent '
            'referable head, head-disagreement routing, and triage. Strict five-grade '
            'parity is reported separately and must not be inferred from screening parity.'
        ),
    }
    return summary, details


def write_comparison(summary, details, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'per_image_runtime_comparison.csv'
    json_path = output_dir / 'runtime_comparison.json'
    details.to_csv(csv_path, index=False)
    with json_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write('\n')
    return csv_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-csv', type=Path, required=True)
    parser.add_argument('--matlab-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--thresholds', type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument('--probability-tolerance', type=float, default=0.05)
    parser.add_argument('--threshold-edge-tolerance', type=float, default=0.05)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.probability_tolerance < 0 or args.threshold_edge_tolerance < 0:
        raise ValueError('Tolerances must be non-negative')
    thresholds = np.load(args.thresholds)
    summary, details = compare_runtime_outputs(
        args.python_csv,
        args.matlab_csv,
        thresholds=thresholds,
        probability_tolerance=args.probability_tolerance,
        threshold_edge_tolerance=args.threshold_edge_tolerance,
    )
    csv_path, json_path = write_comparison(summary, details, args.output_dir)
    print(json.dumps(summary['gates'], indent=2))
    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
