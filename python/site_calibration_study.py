#!/usr/bin/env python3
"""X7: how many labelled target-site clusters does a frozen model need to meet the screening targets?

Post-hoc study on already-exposed data. Two modes:

* ``--run`` (tier TD/mixed, v1, SUPERSEDED for deployment claims): validation outputs of a frozen model; only some
  sources were unseen by that model (the R4/R5 models trained on ODIR/EyeQ/FDR26 training rows), so most rows are
  in-domain and the result bounds the in-domain case only;
* ``--lodo-runs`` (tier TL): held-out outputs of leave-one-dataset-out runs of ``train_v6.py``, where each target
  source was never trained on or selected on, and the frozen boundary is that run's selection-fitted policy.

For each target source S and each calibration size N:

* ``frozen``  : referral boundary fitted on the *other* sources only (balanced 0.90 rule, leave-one-source-out);
* ``site(N)`` : N clusters of S are drawn at random, the referral boundary is re-fitted on them as the highest
  boundary (fewest false referrals) reaching ``--calib-sensitivity`` on the sample, falling back to the frozen
  boundary when the sample holds fewer than ``--min-positives`` referable cases;
* both are scored on the clusters of S that were **not** drawn: sensitivity, specificity, and whether both reach
  the screening targets (sensitivity >= 0.90, specificity >= 0.85).

``oracle`` is the best any single boundary can do on all of S (a ranking ceiling, not a deployable policy).
Resampling is by cluster (patient/exam), seeded, and the score is the expected grade (the deployed ranking score).
Nothing here is evidence of generalisation: sources were exposed, labels differ across sources.
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from netrai_eval import metrics as m
from train_grading_adapter import balanced_threshold, scores

TARGET_SENS, TARGET_SPEC = 0.90, 0.85


def operating_point(score: np.ndarray, positive: np.ndarray, threshold: float) -> tuple[float, float]:
    called = score >= threshold
    sens = float(called[positive].mean()) if positive.any() else float('nan')
    spec = float((~called[~positive]).mean()) if (~positive).any() else float('nan')
    return sens, spec


def site_threshold(score: np.ndarray, positive: np.ndarray, target: float, fallback: float, min_positives: int) -> float:
    """Highest threshold with sample sensitivity >= target (fewest false referrals); fallback if too few positives."""
    if positive.sum() < min_positives or (~positive).sum() < min_positives:
        return fallback
    pos = np.sort(score[positive])
    k = int(np.floor((1 - target) * len(pos)))  # allowed misses
    cut = pos[k]
    return float(cut) - 1e-9  # call everything >= the (k+1)-th lowest positive


def oracle(score: np.ndarray, positive: np.ndarray) -> dict:
    thresholds = np.unique(score)
    best = {'both_targets_possible': False}
    for t in thresholds:
        sens, spec = operating_point(score, positive, t)
        if sens >= TARGET_SENS and spec >= TARGET_SPEC:
            best = {'both_targets_possible': True, 'sensitivity': sens, 'specificity': spec, 'threshold': float(t)}
            break
    best['auc'] = m.auc(positive, score)
    best['spec_at_sens90'] = max((operating_point(score, positive, t)[1] for t in thresholds
                                  if operating_point(score, positive, t)[0] >= TARGET_SENS), default=float('nan'))
    return best


def study(rows: pd.DataFrame, outputs: np.ndarray, sources, sizes, replicates, calib_sens, min_positives, seed,
          frozen_override: dict | None = None) -> dict:
    score = scores(outputs)
    positive = rows.grade.to_numpy() >= 2
    result = {}
    for source in sources:
        in_site = (rows.source == source).to_numpy()
        if frozen_override and source in frozen_override:
            frozen = frozen_override[source]
        else:
            frozen = balanced_threshold(score[~in_site], rows.grade.to_numpy()[~in_site], rows.source[~in_site].to_numpy(), TARGET_SENS)
        s_score, s_pos = score[in_site], positive[in_site]
        clusters = rows.group_id[in_site].to_numpy()
        unique = np.unique(clusters)
        index_by_cluster = {c: np.flatnonzero(clusters == c) for c in unique}
        sens, spec = operating_point(s_score, s_pos, frozen)
        entry = {'n_images': int(in_site.sum()), 'n_clusters': int(len(unique)), 'referable_prevalence': float(s_pos.mean()),
                 'frozen_threshold': float(frozen), 'frozen': {'sensitivity': sens, 'specificity': spec,
                                                                'both_targets': bool(sens >= TARGET_SENS and spec >= TARGET_SPEC)},
                 'oracle': oracle(s_score, s_pos), 'site': {}}
        rng = np.random.default_rng((seed, zlib.crc32(source.encode())))
        for n in sizes:
            if n >= len(unique) * 0.7:
                continue  # need a remainder large enough to score
            outcome = []
            for _ in range(replicates):
                drawn = rng.choice(unique, size=n, replace=False)
                take = np.concatenate([index_by_cluster[c] for c in drawn])
                rest = np.setdiff1d(np.arange(len(s_score)), take)
                t = site_threshold(s_score[take], s_pos[take], calib_sens, frozen, min_positives)
                se, sp = operating_point(s_score[rest], s_pos[rest], t)
                outcome.append((se, sp, se >= TARGET_SENS and sp >= TARGET_SPEC, se >= TARGET_SENS))
            o = np.array(outcome, float)
            entry['site'][str(n)] = {
                'p_both_targets': float(o[:, 2].mean()), 'p_sensitivity_target': float(o[:, 3].mean()),
                'sensitivity_mean': float(np.nanmean(o[:, 0])), 'sensitivity_p10': float(np.nanpercentile(o[:, 0], 10)),
                'specificity_mean': float(np.nanmean(o[:, 1])), 'specificity_p10': float(np.nanpercentile(o[:, 1], 10))}
        result[source] = entry
    return result


def lodo_study(run_dirs, sizes, replicates, calib_sens, min_positives, seed) -> dict:
    """One entry per LODO run: the held-out source is the target, the run's selection-fitted policy is the frozen one."""
    from lodo_analysis import fit_selection_policy, load_run
    result = {}
    for run in run_dirs:
        loaded = load_run(Path(run))
        report = Path(run).parent.parent / f'{Path(run).parent.name}_report.json'
        if report.is_file():  # the selection-fitted policy was already computed (identical inputs) by lodo_analysis
            policy = json.loads(report.read_text())['metrics']['policy_fitted_on_selection_rows']
        else:
            policy = fit_selection_policy(loaded['selection'], loaded['out_s'])
        held = loaded['heldout']
        for source in sorted(held.source.unique()):
            part = (held.source == source).to_numpy()
            entry = study(held[part].reset_index(drop=True), loaded['out_h'][part], [source], sizes, replicates, calib_sens,
                          min_positives, seed, {source: float(policy['thresholds'][1])})
            key = f'{source}:{Path(run).parent.name}'
            result[key] = entry[source]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, help='v1: folder with validation_rows.csv and best_validation_outputs.npy')
    parser.add_argument('--lodo-runs', type=Path, nargs='+', help='v2: leave-one-dataset-out run folders (tier TL)')
    parser.add_argument('--sources', nargs='+', default=['odir', 'fdr26', 'messidor', 'eyeq'])
    parser.add_argument('--sizes', type=int, nargs='+', default=[25, 50, 100, 200, 400])
    parser.add_argument('--replicates', type=int, default=500)
    parser.add_argument('--calib-sensitivity', type=float, default=0.92)
    parser.add_argument('--min-positives', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; results are immutable')
    if bool(args.run) == bool(args.lodo_runs):
        raise SystemExit('give exactly one of --run or --lodo-runs')
    if args.lodo_runs:
        tier = 'TL (leave-dataset-out held-out sources: never trained on or selected on; previously exposed data)'
        result = lodo_study(args.lodo_runs, args.sizes, args.replicates, args.calib_sensitivity, args.min_positives, args.seed)
    else:
        tier = 'TD/mixed (SUPERSEDED: validation rows of a frozen model that trained on most of these sources; in-domain)'
        rows = pd.read_csv(args.run / 'validation_rows.csv')
        outputs = np.load(args.run / 'best_validation_outputs.npy')
        if len(rows) != len(outputs):
            raise ValueError('rows and outputs differ in length')
        result = study(rows, outputs, args.sources, args.sizes, args.replicates, args.calib_sensitivity, args.min_positives, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'tier': tier,
        'run': str(args.run or args.lodo_runs), 'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'targets': {'sensitivity': TARGET_SENS, 'specificity': TARGET_SPEC}, 'results': result}, indent=2, default=str))
    for source, e in result.items():
        print(f"{source}: frozen sens {e['frozen']['sensitivity']:.3f} spec {e['frozen']['specificity']:.3f}; oracle {e['oracle']}")
        for n, s in e['site'].items():
            print(f"   N={n}: P(both)={s['p_both_targets']:.2f} sens {s['sensitivity_mean']:.3f} (p10 {s['sensitivity_p10']:.3f}) spec {s['specificity_mean']:.3f}")


if __name__ == '__main__':
    main()
