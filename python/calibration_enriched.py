#!/usr/bin/env python3
"""How many referable cases does a site-calibration sample need? Enriched samples, rules R1 and R2 (post hoc, tier TX).

A screening site collects a calibration sample deliberately enriched with referable cases (random samples at screening prevalence contain too
few). For each exposed external site: draw ``k`` clusters containing a referable image plus 200 clusters without one, fit the cut-off with each
rule (``calibration_rules.fit_cut``) and score the remaining images; 500 seeded draws. Reported: P(sensitivity >= 0.90 and specificity >= 0.85),
mean and 10th-percentile sensitivity/specificity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import calibration_rules as cr

NON_REFERABLE_CLUSTERS = 200


def enriched_draws(score, referable, clusters, frozen, k, draws, seed, rules=('R1', 'R2')) -> dict | None:
    score, referable, clusters = np.asarray(score, float), np.asarray(referable, bool), np.asarray(clusters)
    unique = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    positive = np.array([c for c in unique if referable[members[c]].any()])
    negative = np.array([c for c in unique if not referable[members[c]].any()])
    if len(positive) < 2 * k or len(negative) < 1.5 * NON_REFERABLE_CLUSTERS:
        return None                                     # the remainder would be too small to score
    rng = np.random.default_rng(seed)
    rows = {rule: [] for rule in rules}
    for _ in range(draws):
        chosen = np.concatenate([rng.choice(positive, k, replace=False), rng.choice(negative, NON_REFERABLE_CLUSTERS, replace=False)])
        take = np.concatenate([members[c] for c in chosen])
        rest = np.setdiff1d(np.arange(len(score)), take)
        for rule in rules:
            called = score[rest] >= cr.fit_cut(rule, score[take], referable[take], frozen)
            sens, spec = called[referable[rest]].mean(), (~called[~referable[rest]]).mean()
            rows[rule].append((sens, spec, sens >= cr.TARGET_SENSITIVITY and spec >= cr.TARGET_SPECIFICITY))
    out = {}
    for rule, values in rows.items():
        r = np.array(values, float)
        out[rule] = {'p_both_targets': float(r[:, 2].mean()), 'sensitivity_mean': float(r[:, 0].mean()), 'sensitivity_p10': float(np.percentile(r[:, 0], 10)),
                     'specificity_mean': float(r[:, 1].mean()), 'specificity_p10': float(np.percentile(r[:, 1], 10))}
    return out


def sustech_site():
    import pandas as pd
    import locked_fresh2 as lf
    from locked_evaluation import ROOT
    from netrai_eval import metrics as m
    lock = ROOT / 'results/model_improvement/fresh2_lock_sustech_sysu'
    frame = pd.read_csv(lock / 'inventory.csv', dtype={'cluster': str})
    labelled, _ = lf.read_labels('sustech_sysu', frame)
    labelled, _ = lf.exclude_conflicting_near_copies(labelled, pd.read_csv(lock / 'near_identical_pairs.csv'))
    sel = labelled.primary.to_numpy(bool)
    return (m.expected_grade_score(np.load(lock / 'logits_E_768.npy'))[sel], labelled.grade.to_numpy()[sel] >= 2, labelled.cluster.to_numpy()[sel])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--k', type=int, nargs='+', default=[20, 40, 80])
    parser.add_argument('--draws', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261031)
    args = parser.parse_args(argv)
    from locked_evaluation import thresholds_for
    frozen = float(thresholds_for({'policy': 'results/model_improvement/round3_eval/policy_E_768_balanced.json'})[0][1])
    sites = cr.development_sites()
    sites['SUSTech-SYSU'] = sustech_site()
    result = {'tier': 'TX (labels exposed by earlier locked evaluations)', 'model': 'E_768', 'frozen_cut': frozen,
              'design': f'k referable clusters + {NON_REFERABLE_CLUSTERS} non-referable clusters per calibration sample; scored on the remaining images', 'sites': {}}
    for name, (score, referable, clusters) in sites.items():
        result['sites'][name] = {str(k): enriched_draws(score, referable, clusters, frozen, k, args.draws, args.seed + k) for k in args.k}
    args.output.write_text(json.dumps(result, indent=1))
    for name, block in result['sites'].items():
        cells = []
        for k, rules in block.items():
            if rules is None:
                cells.append(f'k={k}: n/a')
                continue
            cells.append(f"k={k}: " + ' '.join(f"{r} {v['p_both_targets']:.2f} ({v['sensitivity_mean']:.3f}/{v['specificity_mean']:.3f})" for r, v in rules.items()))
        print(f'{name:20s} ' + ' | '.join(cells))


if __name__ == '__main__':
    main()
