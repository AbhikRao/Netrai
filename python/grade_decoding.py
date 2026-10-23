#!/usr/bin/env python3
"""X1: grade decoders that keep every referral decision and re-decide *within* each referral side.

See ``.project-notes/experiments/X1/spec.md`` (written before any result). Decoders, all fitted
out-of-fold by cluster so no row both fits and scores a decoder:

* A      the deployed policy thresholds (as shipped)
* A_cv   same procedure refitted out-of-fold: referral boundary fixed, other three boundaries by QWK
         (150-point quantile grid instead of every midpoint, for speed; same grid as C and E)
* B      restricted argmax (non-referable side: argmax over {0,1}; referable side: over {2,3,4})
* C      B plus a fitted log-prior bias on classes 1, 3, 4 (source-balanced macro-F1)
* E      expected-score boundaries refitted for source-balanced macro-F1 (referral boundary fixed)

Tier: TD (validation/selection data). Never used on a locked test cohort.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from netrai_eval import grading as eg
from netrai_eval import metrics as m
from netrai_eval import report as er

ROOT = Path(__file__).resolve().parents[1]
MI = ROOT / 'results/model_improvement'
BIAS_GRID = np.arange(-3.0, 3.0001, 0.25)
NON_REFERABLE, REFERABLE = (0, 1), (2, 3, 4)


# ----------------------------------------------------------------------------- decoders


def probabilities(outputs: np.ndarray) -> np.ndarray:
    return m.grade_probabilities(outputs)


def expected_score(p: np.ndarray) -> np.ndarray:
    return p @ np.arange(5)


def decode_thresholds(score: np.ndarray, thresholds) -> np.ndarray:
    return m.decode_thresholds(score, thresholds)


def restricted_argmax(p: np.ndarray, referable: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """Argmax within each referral side (optionally after adding a log-prior ``bias`` of shape (5,))."""
    z = np.log(np.clip(p, 1e-12, 1.0))
    if bias is not None:
        z = z + np.asarray(bias, float)[None, :]
    low = np.argmax(z[:, list(NON_REFERABLE)], axis=1)
    high = np.argmax(z[:, list(REFERABLE)], axis=1) + 2
    return np.where(referable, high, low)


def balanced_macro_f1(truth: np.ndarray, predicted: np.ndarray, sources: np.ndarray) -> float:
    """Mean over sources of the per-source macro-F1 (so a large source cannot dominate the objective)."""
    values = []
    for source in np.unique(sources):
        mask = sources == source
        values.append(m.macro_f1(m.confusion_matrix(truth[mask], predicted[mask])))
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float('nan')


def fit_bias(p: np.ndarray, truth: np.ndarray, sources: np.ndarray, referable: np.ndarray, passes: int = 2) -> np.ndarray:
    """Coordinate search of log-prior biases on classes 1, 3, 4 maximising source-balanced macro-F1."""
    bias = np.zeros(5)
    best = balanced_macro_f1(truth, restricted_argmax(p, referable, bias), sources)
    for _ in range(passes):
        for cls in (1, 3, 4):
            for candidate in BIAS_GRID:
                trial = bias.copy()
                trial[cls] = candidate
                value = balanced_macro_f1(truth, restricted_argmax(p, referable, trial), sources)
                if value > best + 1e-12:
                    best, bias = value, trial
    return bias


def fit_score_thresholds(score: np.ndarray, truth: np.ndarray, sources: np.ndarray, t_star: float,
                         passes: int = 2) -> np.ndarray:
    """Refit t0, t2, t3 (referral boundary t_star fixed) for source-balanced macro-F1."""
    if not 0 < t_star < 4:
        raise ValueError('referral boundary must lie inside (0, 4)')
    thresholds = np.array([t_star / 2, t_star, t_star + (4 - t_star) / 3, t_star + 2 * (4 - t_star) / 3])
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 150)))
    best = balanced_macro_f1(truth, decode_thresholds(score, thresholds), sources)
    for _ in range(passes):
        for index in (0, 2, 3):
            lower = thresholds[index - 1] if index else 0.0
            upper = thresholds[index + 1] if index < 3 else 4.0
            for candidate in candidates[(candidates > lower) & (candidates < upper)]:
                trial = thresholds.copy()
                trial[index] = candidate
                value = balanced_macro_f1(truth, decode_thresholds(score, trial), sources)
                if value > best + 1e-12:
                    best, thresholds = value, trial
    return thresholds


def fit_qwk_thresholds(score: np.ndarray, truth: np.ndarray, t_star: float, passes: int = 2) -> np.ndarray:
    """A_cv: same procedure as ``evaluate_grading_candidate.fit_other_thresholds`` (referral boundary fixed, the
    other three boundaries by pooled QWK, two coordinate passes) but over a 150-point quantile grid instead of every
    score midpoint, so it is fast enough to cross-fit. All fitted decoders use the same grid."""
    if not 0 < t_star < 4:
        raise ValueError('referral boundary must lie inside (0, 4)')
    thresholds = np.array([t_star / 2, t_star, t_star + (4 - t_star) / 3, t_star + 2 * (4 - t_star) / 3])
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 150)))
    best = m.quadratic_weighted_kappa(truth, decode_thresholds(score, thresholds))
    for _ in range(passes):
        for index in (0, 2, 3):
            lower = thresholds[index - 1] if index else 0.0
            upper = thresholds[index + 1] if index < 3 else 4.0
            for candidate in candidates[(candidates > lower) & (candidates < upper)]:
                trial = thresholds.copy()
                trial[index] = candidate
                value = m.quadratic_weighted_kappa(truth, decode_thresholds(score, trial))
                if np.isfinite(value) and value > best + 1e-12:
                    best, thresholds = value, trial
    return thresholds


# ----------------------------------------------------------------------------- cross-fitting


def make_folds(clusters: np.ndarray, sources: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold index per row; whole clusters stay together and every source is spread over all folds."""
    clusters, sources = np.asarray(clusters), np.asarray(sources)
    fold = np.empty(len(clusters), int)
    rng = np.random.default_rng(seed)
    for source in np.unique(sources):
        rows = np.flatnonzero(sources == source)
        unique = np.unique(clusters[rows])
        rng.shuffle(unique)
        assignment = {c: i % k for i, c in enumerate(unique)}
        fold[rows] = [assignment[c] for c in clusters[rows]]
    owners = pd.DataFrame({'c': clusters, 's': sources}).groupby('c').s.nunique()
    if (owners > 1).any():
        raise ValueError('A cluster spans several sources')
    return fold


def crossfit(p: np.ndarray, truth: np.ndarray, sources: np.ndarray, clusters: np.ndarray, policy_thresholds,
             k: int = 5, seed: int = 0, fitter=None) -> dict:
    """Out-of-fold predictions of every decoder. ``fitter`` is a test hook called with (rows_fit, rows_eval)."""
    policy_thresholds = np.asarray(policy_thresholds, float)
    t_star = float(policy_thresholds[1])
    score = expected_score(p)
    referable = score >= t_star
    folds = make_folds(clusters, sources, k, seed)
    out = {name: np.full(len(truth), -1, int) for name in ('A', 'A_cv', 'B', 'C', 'E')}
    out['A'] = decode_thresholds(score, policy_thresholds)  # as shipped: no fitting
    out['B'] = restricted_argmax(p, referable)
    biases, thresholds = [], []
    for fold in range(k):
        fit, ev = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if fitter:
            fitter(fit, ev)
        cv = fit_qwk_thresholds(score[fit], truth[fit], t_star)
        out['A_cv'][ev] = decode_thresholds(score[ev], cv)
        bias = fit_bias(p[fit], truth[fit], sources[fit], referable[fit])
        out['C'][ev] = restricted_argmax(p[ev], referable[ev], bias)
        score_thresholds = fit_score_thresholds(score[fit], truth[fit], sources[fit], t_star)
        out['E'][ev] = decode_thresholds(score[ev], score_thresholds)
        biases.append(bias.tolist())
        thresholds.append(score_thresholds.tolist())
    for name, predicted in out.items():
        if (predicted < 0).any():
            raise AssertionError(f'decoder {name} left rows undecoded')
        if not np.array_equal(predicted >= 2, referable):
            raise AssertionError(f'decoder {name} changed a referral decision')
    return {'predictions': out, 'referable': referable, 'folds': folds, 'fold_biases': biases,
            'fold_thresholds': thresholds}


# ----------------------------------------------------------------------------- reporting


def grade_table(truth: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = m.confusion_matrix(truth, predicted)
    return {'confusion_matrix': matrix.tolist(), 'support': matrix.sum(axis=1).tolist(),
            'recall': m.per_grade_recall(matrix).tolist(), 'precision': m.per_grade_precision(matrix).tolist(),
            'f1': m.per_grade_f1(matrix).tolist(), 'macro_f1': m.macro_f1(matrix),
            'qwk': m.quadratic_weighted_kappa(truth, predicted),
            'mae': m.mean_absolute_grade_error(truth, predicted),
            'exact_accuracy': float(np.mean(truth == predicted))}


def assess(candidate: dict, baseline: dict, paired: dict, per_source: dict, referral_changes: int,
           min_source_grade1: int = 30) -> dict:
    """The predeclared success criteria of the X1 spec for one decoder versus the A_cv baseline."""
    recall = np.array(candidate['recall'])
    base_recall = np.array(baseline['recall'])
    eligible = [s for s, t in per_source.items() if t['A_cv']['support'][1] >= min_source_grade1]
    grade1_sources = [s for s in eligible if per_source[s]['decoder']['recall'][1] >= 0.30]
    delta_f1 = paired['macro_f1']
    checks = {
        '1_referral_identical': referral_changes == 0,
        '2a_grade1_recall_pooled_ge_0.30': bool(recall[1] >= 0.30),
        '2b_grade1_recall_ge_0.30_in_>=3_eligible_sources': len(grade1_sources) >= 3,
        '3_macro_f1_gain_ge_0.03_ci_excludes_0': bool(delta_f1['difference'] >= 0.03 and delta_f1['difference_ci95'][0] > 0),
        '4_qwk_drop_le_0.01': bool(paired['qwk']['difference'] >= -0.01),
        '5_grade0_recall_drop_le_0.05': bool(recall[0] - base_recall[0] >= -0.05),
        '6_referable_grade_recall_drops_le_0.05': bool(np.all(recall[2:] - base_recall[2:] >= -0.05))}
    return {'checks': checks, 'all_met': all(checks.values()), 'eligible_sources': eligible,
            'sources_with_grade1_recall_ge_0.30': grade1_sources}


def load_model(spec: dict) -> dict:
    runs = [MI / r for r in spec['runs']]
    rows = [pd.read_csv(r / 'validation_rows.csv') for r in runs]
    if any(not r.image_id.equals(rows[0].image_id) for r in rows):
        raise ValueError(f"{spec['id']}: validation rows differ between runs")
    outputs = np.mean([np.load(r / 'best_validation_outputs.npy') for r in runs], axis=0)
    policy = json.loads((MI / spec['policy']).read_text())
    return {'rows': rows[0], 'outputs': outputs, 'policy': policy}


MODELS = [
    {'id': 'R4_768_s17', 'runs': ['kaggle_output_r4_s17/run'], 'policy': 'round4_eval/policy_R4_768_s17_balanced.json'},
    {'id': 'R5_768_s17', 'runs': ['kaggle_output_r5_s17/run'], 'policy': 'round5_eval/policy_R5_768_s17_balanced.json'},
    {'id': 'E4_768', 'runs': ['kaggle_output_r4_s2026/run', 'kaggle_output_r4_s17/run'],
     'policy': 'round4_eval/policy_E4_768_balanced.json'},
    {'id': 'E_768_round3', 'runs': ['kaggle_output_768_s2026/run', 'kaggle_output_768_s17/run'],
     'policy': 'round3_eval/policy_E_768_balanced.json'},
]


def evaluate_model(spec: dict, repeats: int = 3, folds: int = 5, replicates: int = 1000, seed: int = 20261021) -> dict:
    loaded = load_model(spec)
    rows, thresholds = loaded['rows'], loaded['policy']['thresholds']
    truth, sources, clusters = rows.grade.to_numpy(), rows.source.to_numpy(), rows.group_id.to_numpy()
    p = probabilities(loaded['outputs'])
    result = {'model': spec['id'], 'policy': spec['policy'], 'policy_thresholds': thresholds,
              'n_images': len(truth), 'n_clusters': int(len(np.unique(clusters))),
              'sources': {s: int((sources == s).sum()) for s in np.unique(sources)}, 'repeats': []}
    for repeat in range(repeats):
        fit = crossfit(p, truth, sources, clusters, thresholds, folds, seed + repeat)
        preds = fit['predictions']
        entry = {'seed': seed + repeat,
                 'pooled': {name: grade_table(truth, pred) for name, pred in preds.items()},
                 'source_balanced_macro_f1': {name: balanced_macro_f1(truth, pred, sources) for name, pred in preds.items()}}
        if repeat == 0:  # primary repeat: per-source tables, paired intervals, criteria
            per_source = {}
            for s in np.unique(sources):
                mask = sources == s
                per_source[s] = {name: grade_table(truth[mask], pred[mask]) for name, pred in preds.items()}
            entry['per_source'] = per_source
            entry['fold_biases'], entry['fold_thresholds'] = fit['fold_biases'], fit['fold_thresholds']
            referable = fit['referable']
            entry['paired_vs_A_cv'], entry['paired_vs_A'], entry['assessment'] = {}, {}, {}
            for name in ('A', 'B', 'C', 'E'):
                vs_cv = eg.compare_grade_decisions(truth, preds[name], preds['A_cv'], clusters, referral_a=referable,
                                                   referral_b=referable, replicates=replicates, seed=seed)
                entry['paired_vs_A_cv'][name] = vs_cv
                if name != 'A':
                    entry['paired_vs_A'][name] = eg.compare_grade_decisions(
                        truth, preds[name], preds['A'], clusters, referral_a=referable, referral_b=referable,
                        replicates=replicates, seed=seed)
                    per_src = {s: {'A_cv': per_source[s]['A_cv'], 'decoder': per_source[s][name]} for s in per_source}
                    entry['assessment'][name] = assess(entry['pooled'][name], entry['pooled']['A_cv'], vs_cv['paired'],
                                                       per_src, vs_cv['referral_decisions_changed'])
        result['repeats'].append(entry)
    result['referral_decisions_changed_any_decoder'] = 0  # crossfit() raises otherwise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MI / 'experiments/X1')
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    results = [evaluate_model(spec, args.repeats, 5, args.replicates) for spec in MODELS]
    limitations = ['Development/selection (TD) data: the checkpoints\' epoch was chosen on these rows.',
                   'Decoders are fitted and scored out-of-fold by cluster; the deployed policy A was fitted on all rows.',
                   'Label definitions differ between sources (EyePACS single grader, ODIR keywords, ALFEDIAM, Messidor-2 adjudicated).',
                   'No locked test cohort was used; any adoption needs its own pre-registered fresh evaluation.']
    rep = er.build_report(tier='TD', cohort='validation sources of R4/R5/E4/E3 runs', model='X1 decoders',
                          metrics={'models': results}, n_images=int(sum(r['n_images'] for r in results)),
                          n_clusters=int(sum(r['n_clusters'] for r in results)), limitations=limitations,
                          experiment='X1', notes='See .project-notes/experiments/X1/spec.md')
    er.write_report(args.output / 'x1_results.json', rep)
    for r in results:
        primary = r['repeats'][0]
        print(f"\n== {r['model']}  n={r['n_images']}")
        for name in ('A', 'A_cv', 'B', 'C', 'E'):
            t = primary['pooled'][name]
            print(f"  {name:5s} macroF1 {t['macro_f1']:.3f} qwk {t['qwk']:.3f} recall {[round(x, 2) for x in t['recall']]}")
        for name, a in primary['assessment'].items():
            print(f"  criteria {name}: all_met={a['all_met']}  failed={[k for k, v in a['checks'].items() if not v]}")


if __name__ == '__main__':
    main()
