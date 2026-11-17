#!/usr/bin/env python3
"""End-to-end decision parity of the full Python and MATLAB pipelines (default settings) on a fixed sample of real images.

Five seeded random images from each fundus dataset of the X15 study (all exposed or training data) plus three non-fundus images
(a portrait, MATLAB's peppers, a black frame) run through ``inference.run_pipeline`` and MATLAB ``main_pipeline`` (quality gate,
input guard, default model; MATLAB without report and Grad-CAM). Compared: status and rejection reason, quality score, guard
score, grade, continuous score, routing and the input warning. Writes per-image tables locally and an aggregate JSON (no file names:
the sample includes PhysioNet DUA images).

  python compare_runtimes_e2e.py --work DIR --out results/benchmarks/runtime_comparison/e2e_parity.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEED, PER_SOURCE = 20261031, 5
MATLAB_SCRIPT = """
paths = string(readlines('{paths}')); paths = paths(strlength(paths) > 0); n = numel(paths);
addpath('{matlab}');
k = (0:n-1)'; status = strings(n,1); reason = strings(n,1); iqs = nan(n,1); guard = nan(n,1); grade = nan(n,1); score = nan(n,1);
triage = strings(n,1); warning = strings(n,1); variant = strings(n,1); message = strings(n,1);
for i = 1:n
    try
        r = main_pipeline(char(paths(i)), 'Verbose', false, 'GenerateReport', false, 'RunGradCAM', false, 'OutputDir', char(tempname));
        status(i) = r.status; iqs(i) = r.iqs; guard(i) = r.fundusGuard.score; triage(i) = r.triageAction; variant(i) = r.model_variant;
        if isfield(r, 'rejectionReason'), reason(i) = r.rejectionReason; end
        if r.status == "success", grade(i) = r.grade; score(i) = r.continuousScore; warning(i) = r.inputWarning; end
    catch e
        status(i) = "error"; message(i) = string(e.identifier) + ": " + string(e.message);
    end
end
writetable(table(k, status, reason, iqs, guard, grade, score, triage, warning, variant, message), '{out}');
"""


def draw_sample(work: Path) -> pd.DataFrame:
    import fundus_guard_study as study
    rng = np.random.default_rng(SEED)
    rows = []
    for source in study.FIT_SOURCES + study.EXTERNAL_SOURCES:
        table = study.inventory(source)
        rows += [{'source': source, 'path': table.path.iloc[k]} for k in rng.choice(len(table), PER_SOURCE, replace=False)]
    black = work / 'black.png'
    cv2.imwrite(str(black), np.zeros((256, 256, 3), np.uint8))
    rows += [{'source': 'non_fundus', 'path': str(Path(__import__('matplotlib').get_data_path()) / 'sample_data/grace_hopper.jpg')},
             {'source': 'non_fundus', 'path': str(study.matlab_installation() / 'toolbox/matlab/matlab_images/png/peppers.png')},
             {'source': 'non_fundus', 'path': str(black)}]
    return pd.DataFrame(rows).rename_axis('k').reset_index()


def run_python(sample: pd.DataFrame, work: Path) -> pd.DataFrame:
    import inference
    rows = []
    for row in sample.itertuples(index=False):
        with contextlib.redirect_stdout(io.StringIO()):
            result = inference.run_pipeline(row.path, output_dir=str(work / 'python' / str(row.k)), variant='auto')
        success = result['status'] == 'success'
        rows.append({'k': row.k, 'status': result['status'], 'reason': result.get('rejection_reason', ''), 'iqs': float(result['iqs']),
                     'guard': (result.get('fundus_guard') or {}).get('score', np.nan), 'grade': result['grade'] if success else np.nan,
                     'score': float(np.dot(np.arange(5), result['probs'])) if success else np.nan,
                     'triage': result['triage_action'], 'warning': result.get('input_warning', ''), 'variant': result.get('model_variant')})
    return pd.DataFrame(rows)


def run_matlab(sample: pd.DataFrame, work: Path) -> pd.DataFrame:
    paths, out = work / 'paths.txt', work / 'matlab.csv'
    paths.write_text('\n'.join(sample.path) + '\n')
    script = work / 'e2e_parity_matlab.m'
    script.write_text(MATLAB_SCRIPT.format(paths=paths, matlab=ROOT / 'matlab', out=out))
    subprocess.run([shutil.which('matlab'), '-batch', f"run('{script}')"], check=True, cwd=work)
    return pd.read_csv(out, keep_default_na=False, na_values=['NaN', ''])


def compare(sample: pd.DataFrame, py: pd.DataFrame, mt: pd.DataFrame) -> dict:
    df = sample.set_index('k').join(py.set_index('k').add_prefix('py_')).join(mt.set_index('k').add_prefix('m_'))
    graded = (df.py_status == 'success') & (df.m_status == 'success')
    flagged = lambda column: df[column].fillna('').astype(str) != ''
    guard = df.dropna(subset=['py_guard', 'm_guard'])
    return {
        'images': len(df), 'sources': int(df.source.nunique()), 'per_source': PER_SOURCE, 'seed': SEED,
        'errors': {'python': int((df.py_status == 'error').sum()), 'matlab': int((df.m_status == 'error').sum())},
        'variant': {'python': sorted(set(df.py_variant.astype(str))), 'matlab': sorted(set(df.m_variant.astype(str)))},
        'status_equal': int((df.py_status == df.m_status).sum()),
        'rejected': {'python': int((df.py_status == 'rejected').sum()), 'matlab': int((df.m_status == 'rejected').sum())},
        'graded_both': int(graded.sum()), 'grade_equal': int((df.py_grade[graded] == df.m_grade[graded]).sum()),
        'routing_equal': int((df.py_triage.fillna('') == df.m_triage.fillna('')).sum()),
        'input_warning_equal': int((flagged('py_warning') == flagged('m_warning')).sum()),
        'input_warnings': {'python': int(flagged('py_warning').sum()), 'matlab': int(flagged('m_warning').sum())},
        'max_abs_quality_score_difference': float((df.py_iqs - df.m_iqs).abs().max()),
        'max_relative_guard_score_difference': float(((guard.py_guard - guard.m_guard).abs() / guard.m_guard).max()),
        'max_abs_continuous_score_difference': float((df.py_score[graded] - df.m_score[graded]).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', type=Path, required=True, help='local working folder (per-image tables, pipeline outputs)')
    parser.add_argument('--out', type=Path, required=True, help='aggregate JSON (no file names)')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    sample = draw_sample(args.work)
    sample.to_csv(args.work / 'sample.csv', index=False)
    py = run_python(sample, args.work)
    py.to_csv(args.work / 'python.csv', index=False)
    mt = run_matlab(sample, args.work)
    summary = compare(sample, py, mt)
    summary['tier'] = 'runtime agreement on exposed/training images and local non-fundus images; not an accuracy result'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
