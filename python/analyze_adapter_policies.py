#!/usr/bin/env python3
"""Compare predeclared threshold rules on saved adaptation-ladder outputs.

Thresholds are fitted only on the patient-grouped pool holdout (rung B on the
holdout too, for a like-for-like comparison) and then applied unchanged to each
selection/stress cohort. This chooses a development policy; it is not acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_grading_adapter import evaluate, fit_head_threshold, fit_policy, scores
from utils.artifacts import atomic_json

RULES = [('pooled', 0.90), ('pooled', 0.92), ('worst-source', 0.90), ('worst-source', 0.92)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    results = json.loads((args.run / 'results.json').read_text())
    cohorts = results['selection'] + results['stress_exposed']
    table = {}
    for rung in results['policies']:
        data = np.load(args.run / f'rung_{rung}_outputs.npz')
        holdout, grade, source = data['holdout'], data['holdout_grade'], data['holdout_source']
        for rule, target in RULES:
            thresholds = fit_policy(scores(holdout), grade, target, source, rule)
            head = fit_head_threshold(holdout, grade, target)
            key = f'{rung}|{rule}|{target}'
            table[key] = {'thresholds': thresholds, 'head_threshold': head, **{
                name: evaluate(data[name], data[f'{name}_grade'], thresholds, head) for name in cohorts}}
            row = ' '.join(f"{name}: {m['sensitivity']:.3f}/{m['specificity']:.3f} rev {m['head_disagreement_rate']:.3f}"
                           for name, m in table[key].items() if isinstance(m, dict))
            print(f'{key:26s} {row}')
    atomic_json(args.run / 'policy_comparison.json', table)


if __name__ == '__main__':
    main()
