#!/usr/bin/env python3
"""Pre-registered secondary endpoint of the SUSTech-SYSU lock (plan v6, Addendum 6.4): confirmation of the selected site-calibration rule.

The rule (R2: always re-fit the referral cut-off to 95% sample sensitivity) was selected on exposed sites before any SUSTech-SYSU label was read;
this script's SHA-256 is recorded in the plan addendum, which the lock binds. Run it once, immediately after ``locked_fresh2.py evaluate``:

* verifies that the evaluation report exists and every lock binding still matches, and that this script's hash appears in the bound plan;
* rebuilds the primary population exactly as the evaluation did and draws the SAME calibration samples (seed + size) for R1 and R2;
* checks that R1 reproduces the evaluation's ``site_calibration_procedure`` numbers exactly;
* confirmation at 200 images: P(both targets) of R2 >= that of R1, and mean sensitivity of R2 >= 0.90.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import calibration_rules as cr
import locked_fresh2 as lf
from locked_evaluation import ROOT, sha, thresholds_for
from netrai_eval import metrics as m
from utils.artifacts import atomic_json

SELECTED, BASELINE, SIZES = 'R2', 'R1', (100, 200)


def verdict(selected: dict, baseline: dict) -> dict:
    passed = selected['p_both_targets'] >= baseline['p_both_targets'] and selected['sensitivity_mean'] >= cr.TARGET_SENSITIVITY
    return {'rule': f'P(both) {SELECTED} >= P(both) {BASELINE} and mean sensitivity {SELECTED} >= 0.90 (200 images)', 'confirmed': bool(passed)}


def run(args) -> dict:
    out = args.output
    lock = json.loads((out / 'lock.json').read_text())
    report = json.loads((out / 'report.json').read_text())
    if (out / 'secondary_report.json').exists():
        raise FileExistsError('The secondary endpoint was already computed')
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    plans = [ROOT / p for p in lock['bindings'] if p.endswith('MODEL_IMPROVEMENT_PLAN_v6.md')]
    if not plans or me not in plans[0].read_text():
        raise ValueError('This script is not the one pre-registered in the bound plan (SHA-256 not found)')
    config = json.loads(args.variants.read_text())
    frame = pd.read_csv(out / 'inventory.csv', dtype={'cluster': str})
    labelled, _ = lf.read_labels(args.cohort, frame)
    near = out / 'near_identical_pairs.csv'
    if near.is_file():
        labelled, _ = lf.exclude_conflicting_near_copies(labelled, pd.read_csv(near))
    primary = labelled.primary.to_numpy(bool)
    grade, clusters = labelled.grade.to_numpy(int)[primary], labelled.cluster.to_numpy()[primary]
    name = lock['primary_variant']
    spec = next(s for s in config['variants'] if s['id'] == name)
    score = m.expected_grade_score(np.load(out / f'logits_{name}.npy')[primary])
    frozen = float(thresholds_for(spec)[0][1])
    result = {}
    for size in SIZES:
        draws = cr.rule_draws(score, grade >= 2, clusters, frozen, size, lf.CALIBRATION_DRAWS, lock['bootstrap_seed'] + size, rules=(BASELINE, SELECTED))
        locked = report['site_calibration_procedure'][str(size)]
        for key in ('p_both_targets', 'sensitivity_mean', 'specificity_mean'):
            if abs(draws[BASELINE][key] - locked[key]) > 1e-12:
                raise ValueError(f'R1 does not reproduce the locked evaluation ({size} images, {key})')
        result[str(size)] = draws
    secondary = {'schema_version': 1, 'tier': 'TF (pre-registered secondary; rule selected on exposed sites before this lock)', 'cohort': args.cohort,
                 'lock_sha256': sha(out / 'lock.json'), 'script_sha256': me, 'evaluated_utc': datetime.now(timezone.utc).isoformat(),
                 'selected_rule': SELECTED, 'baseline_rule': BASELINE, 'results': result, 'verdict': verdict(result['200'][SELECTED], result['200'][BASELINE])}
    atomic_json(out / 'secondary_report.json', secondary)
    return secondary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cohort', choices=sorted(lf.COHORTS), required=True)
    parser.add_argument('--variants', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='the lock folder')
    args = parser.parse_args(argv)
    secondary = run(args)
    for size, block in secondary['results'].items():
        print(size, {r: {k: round(v[k], 3) for k in ('p_both_targets', 'sensitivity_mean', 'specificity_mean')} for r, v in block.items()})
    print(secondary['verdict'])


if __name__ == '__main__':
    main()
