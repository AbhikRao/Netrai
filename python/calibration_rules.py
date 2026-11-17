#!/usr/bin/env python3
"""Site-calibration rules for the referral cut-off and their resampling evaluation (plan v6, Addendum 6.4).

A site calibrates the frozen grade >= 2 cut-off of a model on a small labelled sample of its own images. Rules (``fit_cut``):

* ``R1`` always re-fit: the highest cut-off whose sample sensitivity reaches 0.92 (the procedure pre-registered for the fresh sets);
* ``R2`` the same with target 0.95;
* ``R3`` keep the frozen cut-off when, on the sample, it already reaches sensitivity >= 0.92 and specificity >= 0.85; otherwise ``R1``;
* ``R4`` keep the frozen cut-off when it reaches sensitivity >= 0.95 and specificity >= 0.85 on the sample; otherwise ``R2``.

With fewer than 5 referable or 5 non-referable images in the sample every rule keeps the frozen cut-off.

``rule_draws`` evaluates several rules on the SAME random calibration samples (whole clusters, >= ``size`` images; the cut is fitted on the
sample and scored on the remaining images), with the sampling of ``locked_fresh2.calibration_draws`` (identical draws for an identical seed).
``develop`` runs the development comparison on the exposed external sites (tier TX) and applies the selection rule of Addendum 6.4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from calibrate_site import fit_boundary

RULES = {'R1': (0.92, False), 'R2': (0.95, False), 'R3': (0.92, True), 'R4': (0.95, True)}   # name -> (target, keep frozen if it passes)
PREFERENCE = ('R1', 'R3', 'R2', 'R4')
MIN_CASES, TARGET_SENSITIVITY, TARGET_SPECIFICITY, TIE, MIN_SITE_SENSITIVITY = 5, 0.90, 0.85, 0.02, 0.89


def fit_cut(rule: str, score: np.ndarray, referable: np.ndarray, frozen: float) -> float:
    """Referral cut-off chosen by ``rule`` on a calibration sample (call referable when score >= cut)."""
    target, keep_if_passing = RULES[rule]
    referable = np.asarray(referable, bool)
    if referable.sum() < MIN_CASES or (~referable).sum() < MIN_CASES:
        return float(frozen)
    if keep_if_passing:
        called = score >= frozen
        if called[referable].mean() >= target and (~called[~referable]).mean() >= TARGET_SPECIFICITY:
            return float(frozen)
    return fit_boundary(score, referable, target)


def rule_draws(score: np.ndarray, referable: np.ndarray, clusters: np.ndarray, frozen: float, size: int, draws: int, seed: int,
               rules=tuple(RULES)) -> dict:
    """{rule: summary} over the same ``draws`` calibration samples (sampling identical to ``locked_fresh2.calibration_draws``)."""
    score, referable, clusters = np.asarray(score, float), np.asarray(referable, bool), np.asarray(clusters)
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    outcomes = {rule: [] for rule in rules}
    for _ in range(draws):
        order = rng.permutation(unique)
        take, count = [], 0
        for c in order:
            if count >= size:
                break
            take.extend(members[c]); count += len(members[c])
        take = np.array(take)
        rest = np.setdiff1d(np.arange(len(score)), take)
        for rule in rules:
            cut = fit_cut(rule, score[take], referable[take], frozen)
            called = score[rest] >= cut
            sens, spec = float(called[referable[rest]].mean()), float((~called[~referable[rest]]).mean())
            outcomes[rule].append((sens, spec, sens >= TARGET_SENSITIVITY and spec >= TARGET_SPECIFICITY, cut != frozen))
    called = score >= frozen
    out = {}
    for rule, rows in outcomes.items():
        r = np.array(rows, float)
        out[rule] = {'calibration_images': size, 'draws': draws, 'p_both_targets': float(r[:, 2].mean()),
                     'sensitivity_mean': float(r[:, 0].mean()), 'sensitivity_p10': float(np.percentile(r[:, 0], 10)),
                     'specificity_mean': float(r[:, 1].mean()), 'specificity_p10': float(np.percentile(r[:, 1], 10)),
                     'share_cut_changed': float(r[:, 3].mean()),
                     'frozen_sensitivity': float(called[referable].mean()), 'frozen_specificity': float((~called[~referable]).mean())}
    return out


def select_rule(per_site: dict) -> dict:
    """Addendum 6.4 selection: {site: {rule: summary}} at 200 images -> chosen rule and the reasoning numbers."""
    rules = list(next(iter(per_site.values())))
    score = {r: float(np.mean([per_site[s][r]['p_both_targets'] for s in per_site])) for r in rules}
    worst = {r: float(min(per_site[s][r]['sensitivity_mean'] for s in per_site)) for r in rules}
    eligible = [r for r in rules if worst[r] >= MIN_SITE_SENSITIVITY]
    if eligible:
        best = max(score[r] for r in eligible)
        chosen = next(r for r in PREFERENCE if r in eligible and score[r] >= best - TIE)
        reason = 'highest mean P(both) among rules with every site mean sensitivity >= 0.89 (preference order within 0.02)'
    else:
        chosen = max(rules, key=lambda r: worst[r])
        reason = 'no rule met the sensitivity constraint: highest worst-site mean sensitivity'
    return {'selected': chosen, 'reason': reason, 'mean_p_both': score, 'worst_site_mean_sensitivity': worst, 'eligible': eligible}


def development_sites() -> dict:
    """Exposed external sites for E_768 (labels exposed by earlier locks): {name: (score, referable, clusters)}. Tier TX."""
    import pandas as pd
    import locked_evaluation as le
    import locked_fresh2 as lf
    from netrai_eval import metrics as m
    R = le.ROOT / 'results/model_improvement'
    sites = {}
    inv = pd.read_csv(R / 'brset_lock/inventory_brset.csv', dtype=str)
    lab = pd.read_csv(le.ROOT / 'data/brset/label_brset.csv', dtype=str, keep_default_na=False)
    lab['f'] = lab.image_id + '.jpg'
    lab = lab.set_index('f').reindex(inv.image_id)
    score = m.expected_grade_score(np.load(R / 'brset_lock/logits_E_768.npy'))
    ok = (lab.quality == 'Adequate').to_numpy()
    for camera, name in (('Canon', 'BRSET Canon CR'), ('NIKON', 'BRSET Nikon NF5050')):
        sel = ok & lab.camera.str.startswith(camera).to_numpy()
        sites[name] = (score[sel], lab.DR_ICDR.astype(int).to_numpy()[sel] >= 2, inv.cluster.to_numpy()[sel])
    inv = pd.read_csv(R / 'mbrset_lock/inventory_mbrset.csv', dtype=str)
    lab = pd.read_csv(le.ROOT / 'data/mbrset/labels_mbrset.csv', dtype=str, keep_default_na=False).set_index('file').reindex(inv.image_id)
    sel = ((lab.final_quality == 'yes') & (lab.final_icdr != '')).to_numpy()
    sites['mBRSET'] = (m.expected_grade_score(np.load(R / 'mbrset_lock/logits_E_768.npy'))[sel],
                       lab.final_icdr[sel].astype(int).to_numpy() >= 2, inv.cluster.to_numpy()[sel])
    lock = R / 'fresh2_lock_paraguay_hc'
    frame = pd.read_csv(lock / 'inventory.csv', dtype={'cluster': str})
    labelled, _ = lf.read_labels('paraguay_hc', frame)
    labelled, _ = lf.exclude_conflicting_near_copies(labelled, pd.read_csv(lock / 'near_identical_pairs.csv'))
    sel = labelled.primary.to_numpy(bool)
    sites['Paraguay'] = (m.expected_grade_score(np.load(lock / 'logits_E_768.npy'))[sel], labelled.grade.to_numpy()[sel] >= 2,
                         labelled.cluster.to_numpy()[sel])
    for name, cohort, folder in (('ODIR', 'odir', 'round3_lock'), ('FDR26', 'fdr26', 'round4_lock_v3')):
        frame = pd.read_csv(R / folder / f'inventory_{cohort}.csv')
        frame, gradable, grade = le.cohort_labels(cohort, frame)
        g = np.asarray(gradable, bool)
        sites[name] = (m.expected_grade_score(np.load(R / folder / f'logits_{cohort}_E_768.npy'))[g], grade[g] >= 2,
                       frame.exam_id.to_numpy()[g])
    return sites


def develop(args) -> None:
    from locked_evaluation import thresholds_for
    frozen = float(thresholds_for({'policy': 'results/model_improvement/round3_eval/policy_E_768_balanced.json'})[0][1])
    sites = development_sites()
    result = {'tier': 'TX (labels of these sites were exposed by earlier locked evaluations)', 'model': 'E_768', 'frozen_cut': frozen,
              'protocol': 'random whole-cluster calibration samples, cut fitted on the sample and scored on the remaining images', 'sites': {}}
    for size in args.sizes:
        for name, (score, referable, clusters) in sites.items():
            entry = result['sites'].setdefault(name, {'images': int(len(score)), 'referable': int(referable.sum()),
                                                      'clusters': int(len(np.unique(clusters)))})
            entry[str(size)] = rule_draws(score, referable, clusters, frozen, size, args.draws, args.seed + size)
    result['selection_at_200'] = select_rule({name: entry['200'] for name, entry in result['sites'].items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1))
    for name, entry in result['sites'].items():
        line = ' | '.join(f"{r} {entry['200'][r]['p_both_targets']:.2f} ({entry['200'][r]['sensitivity_mean']:.3f}/{entry['200'][r]['specificity_mean']:.3f})"
                          for r in RULES)
        frozen_pt = f"frozen {entry['200']['R1']['frozen_sensitivity']:.3f}/{entry['200']['R1']['frozen_specificity']:.3f}"
        print(f"{name:20s} n={entry['images']:5d} ref={entry['referable']:4d} | {frozen_pt} | {line}")
    print(json.dumps(result['selection_at_200'], indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('develop')
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--sizes', type=int, nargs='+', default=[100, 200])
    d.add_argument('--draws', type=int, default=500)
    d.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args(argv)
    {'develop': develop}[args.command](args)


if __name__ == '__main__':
    main()
