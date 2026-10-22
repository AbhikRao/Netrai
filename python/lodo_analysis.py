#!/usr/bin/env python3
"""Analyse leave-one-dataset-out runs of ``train_v6.py`` (tier TL) and compare arms.

For one run directory:
* the decision policy (source-balanced 0.90 grade-2 boundary + referral-head threshold, QWK for the other
  boundaries; the same rule as ``fit_validation_policy.py``) is fitted on the **selection** rows only
  (td + validation of the remaining sources, tier TD);
* the frozen policy is then applied to the **held-out source** (tier TL, never used for selection), so the
  reported sensitivity/specificity show *threshold transfer* to an unseen dataset, next to the threshold-free
  AUC and sensitivity at fixed specificity;
* everything goes through ``netrai_eval`` with cluster-bootstrap intervals and a mandatory evidence tier.

``compare`` pairs two runs of the same held-out source (same images) and reports paired differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from netrai_eval import bootstrap as bs
from netrai_eval import grading as eg
from netrai_eval import metrics as m
from netrai_eval import report as er
from train_grading_adapter import fit_head_threshold, fit_policy, scores
from utils.calibration import fit_temperature

TARGET_SENSITIVITY = 0.90


def load_run(run: Path) -> dict:
    needed = ('validation_rows.csv', 'best_validation_outputs.npy', 'heldout_rows.csv', 'best_heldout_outputs.npy',
              'run_info.json')
    missing = [n for n in needed if not (run / n).is_file()]
    if missing:
        raise FileNotFoundError(f'{run} lacks {missing}; it is not a finished leave-one-dataset-out run')
    selection = pd.read_csv(run / 'validation_rows.csv')
    heldout = pd.read_csv(run / 'heldout_rows.csv')
    out_s, out_h = np.load(run / 'best_validation_outputs.npy'), np.load(run / 'best_heldout_outputs.npy')
    if len(selection) != len(out_s) or len(heldout) != len(out_h):
        raise ValueError(f'{run}: output rows do not match their row files')
    if set(selection.image_id) & set(heldout.image_id):
        raise ValueError(f'{run}: selection and held-out sets overlap')
    info = json.loads((run / 'run_info.json').read_text())
    return {'selection': selection, 'heldout': heldout, 'out_s': out_s, 'out_h': out_h, 'info': info}


def fit_selection_policy(rows: pd.DataFrame, outputs: np.ndarray, target: float = TARGET_SENSITIVITY) -> dict:
    """Policy fitted on selection rows only (never sees the held-out source)."""
    grade, source = rows.grade.to_numpy(), rows.source.to_numpy()
    thresholds = fit_policy(scores(outputs), grade, target, source, 'balanced')
    head = fit_head_threshold(outputs, grade, target, source, 'balanced')
    temperature, _ = fit_temperature(outputs[:, :5], grade)
    return {'thresholds': thresholds, 'head_threshold': head, 'temperature': float(temperature),
            'rule': 'balanced', 'target_sensitivity': target}


def analyze_run(run: Path, replicates: int = 1000, seed: int = 20261022) -> dict:
    loaded = load_run(run)
    policy = fit_selection_policy(loaded['selection'], loaded['out_s'])
    held = loaded['heldout']
    sources = sorted(held.source.unique())
    metrics = {'policy_fitted_on_selection_rows': policy, 'heldout_sources': sources, 'per_source': {}}
    for source in sources:
        mask = (held.source == source).to_numpy()
        metrics['per_source'][source] = eg.evaluate_grading(
            loaded['out_h'][mask], held.grade.to_numpy()[mask], held.group_id.to_numpy()[mask],
            thresholds=policy['thresholds'], head_threshold=policy['head_threshold'], replicates=replicates, seed=seed)
    selection_auc = {}
    for source, part in loaded['selection'].groupby('source'):
        index = part.index.to_numpy()
        selection_auc[source] = m.auc(part.grade.to_numpy() >= 2, m.expected_grade_score(loaded['out_s'][index]))
    metrics['selection_auc_by_source_TD'] = selection_auc
    limitations = ['Leave-one-dataset-out fold on previously exposed data: not fresh independent evidence.',
                   'The selection data (tier TD) chose the epoch and fitted the policy.',
                   'Label definitions differ between sources.']
    config = loaded['info']['config']
    return er.build_report(tier='TL', cohort='+'.join(sources), model=Path(run).name, metrics=metrics,
                           n_images=len(held), n_clusters=int(held.group_id.nunique()), limitations=limitations,
                           experiment=config.get('output', str(run)))


def compare(run_a: Path, run_b: Path, replicates: int = 1000, seed: int = 20261022) -> dict:
    """Paired threshold-free comparison on the shared held-out images (A minus B)."""
    a, b = load_run(run_a), load_run(run_b)
    if not a['heldout'].image_id.equals(b['heldout'].image_id):
        raise ValueError('The two runs did not hold out the same images in the same order')
    held = a['heldout']
    truth = held.grade.to_numpy()
    score_a, score_b = m.expected_grade_score(a['out_h']), m.expected_grade_score(b['out_h'])

    def stats(score):
        return lambda rows: {'auc': m.auc(truth[rows] >= 2, score[rows]),
                             'sensitivity_at_85_specificity': m.sensitivity_at_specificity(truth[rows] >= 2, score[rows], 0.85),
                             'sensitivity_at_90_specificity': m.sensitivity_at_specificity(truth[rows] >= 2, score[rows], 0.90)}

    return bs.paired_bootstrap(stats(score_a), stats(score_b), held.group_id.to_numpy(), replicates, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('analyze')
    one.add_argument('run', type=Path)
    one.add_argument('--output', type=Path, required=True)
    one.add_argument('--replicates', type=int, default=1000)
    two = sub.add_parser('compare')
    two.add_argument('run_a', type=Path)
    two.add_argument('run_b', type=Path)
    two.add_argument('--output', type=Path, required=True)
    two.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    if args.command == 'analyze':
        er.write_report(args.output, analyze_run(args.run, args.replicates))
    else:
        from utils.artifacts import atomic_json
        if args.output.exists():
            raise FileExistsError(args.output)
        atomic_json(args.output, {'schema_version': 1, 'tier': 'TL', 'a': str(args.run_a), 'b': str(args.run_b),
                                  'paired_a_minus_b': compare(args.run_a, args.run_b, args.replicates)})
    print('written', args.output)


if __name__ == '__main__':
    main()
