#!/usr/bin/env python3
"""Post hoc (tier TX) site-calibration curves of the frozen E_768 on the two exposed Brazilian sets (mBRSET, BRSET).

Uses ``site_calibration_study.study`` (X7 protocol): for each target site and N, N patients are drawn, the referral boundary is re-fitted on them
(highest boundary reaching ``--calib-sensitivity`` on the sample) and scored on the remaining patients. The ``frozen`` row is E_768's own
round-3 balanced boundary. Sites: mBRSET (Eyer smartphone), BRSET Canon CR, BRSET Nikon NF5050, BRSET all. Primary populations as in the locked
evaluations (gradable, graded, no exclusions). Labels are exposed (the locks were evaluated); aggregates only are written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import site_calibration_study as sc
from locked_evaluation import ROOT, thresholds_for

LOCKS = {'mbrset': ROOT / 'results/model_improvement/mbrset_lock', 'brset': ROOT / 'results/model_improvement/brset_lock'}
LABELS = {'mbrset': ROOT / 'data/mbrset/labels_mbrset.csv', 'brset': ROOT / 'data/brset/label_brset.csv'}
E768 = {'id': 'E_768', 'policy': 'results/model_improvement/round3_eval/policy_E_768_balanced.json'}


def load_site_rows(cohort: str) -> tuple[pd.DataFrame, np.ndarray]:
    inv = pd.read_csv(LOCKS[cohort] / f'inventory_{cohort}.csv', dtype=str)
    table = pd.read_csv(LABELS[cohort], dtype=str, keep_default_na=False)
    if cohort == 'mbrset':
        table = table.set_index('file')
        joined = table.reindex(inv.image_id)
        ok = (joined.final_quality == 'yes') & (joined.final_icdr != '')
        grade = joined.final_icdr.where(ok, '-1').astype(int)
        camera = pd.Series(['Eyer'] * len(inv))
        group = 'm' + inv.patient
    else:
        table['file'] = table.image_id + '.jpg'
        joined = table.set_index('file').reindex(inv.image_id)
        ok = joined.quality == 'Adequate'
        grade = joined.DR_ICDR.where(ok, '-1').astype(int)
        camera = joined.camera.reset_index(drop=True)
        group = 'b' + inv.patient
    keep = (grade >= 0).to_numpy()
    logits = np.load(LOCKS[cohort] / 'logits_E_768.npy')[keep]
    rows = pd.DataFrame({'grade': grade.to_numpy()[keep], 'group_id': group.to_numpy()[keep], 'camera': camera.to_numpy()[keep]})
    return rows, logits


def enriched_study(rows: pd.DataFrame, logits: np.ndarray, source: str, positives, negatives: int, replicates: int, calib_sens: float, frozen: float,
                   seed: int) -> dict:
    """Calibration samples enriched with referable patients: ``n`` referable + ``negatives`` non-referable patients, scored on the remaining patients."""
    import zlib
    score = sc.scores(logits)
    in_site = (rows.source == source).to_numpy()
    s_score, grade, group = score[in_site], rows.grade.to_numpy()[in_site], rows.group_id.to_numpy()[in_site]
    pos_groups = np.unique(group[grade >= 2])
    neg_groups = np.setdiff1d(np.unique(group), pos_groups)
    rng = np.random.default_rng((seed, zlib.crc32(source.encode())))
    out = {}
    for n in positives:
        runs = []
        for _ in range(replicates):
            drawn = np.concatenate([rng.choice(pos_groups, size=n, replace=False), rng.choice(neg_groups, size=negatives, replace=False)])
            take = np.isin(group, drawn)
            t = sc.site_threshold(s_score[take], grade[take] >= 2, calib_sens, frozen, 5)
            se, sp = sc.operating_point(s_score[~take], grade[~take] >= 2, t)
            runs.append((se, sp, se >= sc.TARGET_SENS and sp >= sc.TARGET_SPEC, se >= sc.TARGET_SENS))
        o = np.array(runs, float)
        out[str(n)] = {'referable_patients': n, 'non_referable_patients': negatives, 'p_both_targets': float(o[:, 2].mean()),
                       'p_sensitivity_target': float(o[:, 3].mean()), 'sensitivity_mean': float(o[:, 0].mean()),
                       'sensitivity_p10': float(np.percentile(o[:, 0], 10)), 'specificity_mean': float(o[:, 1].mean()),
                       'specificity_p10': float(np.percentile(o[:, 1], 10))}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[25, 50, 100, 200, 400])
    parser.add_argument('--replicates', type=int, default=500)
    parser.add_argument('--calib-sensitivity', type=float, default=0.92)
    parser.add_argument('--min-positives', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; results are immutable')
    boundary = float(thresholds_for(E768)[0][1])
    parts, outputs = [], []
    for cohort in ('mbrset', 'brset'):
        rows, logits = load_site_rows(cohort)
        rows['source'] = np.where(rows.camera == 'Eyer', 'mbrset_eyer_smartphone',
                                  np.where(rows.camera.str.startswith('Canon'), 'brset_canon', 'brset_nikon'))
        parts.append(rows)
        outputs.append(logits)
    rows = pd.concat(parts, ignore_index=True)
    logits = np.concatenate(outputs)
    result, sources = {}, ['mbrset_eyer_smartphone', 'brset_canon', 'brset_nikon']
    result.update(sc.study(rows, logits, sources, args.sizes, args.replicates, args.calib_sensitivity, args.min_positives, args.seed,
                           {s: boundary for s in sources}))
    brset = rows[rows.source.str.startswith('brset')].reset_index(drop=True)
    brset['source'] = 'brset_all_cameras'
    result.update(sc.study(brset, logits[rows.source.str.startswith('brset').to_numpy()], ['brset_all_cameras'], args.sizes, args.replicates,
                           args.calib_sensitivity, args.min_positives, args.seed, {'brset_all_cameras': boundary}))
    enriched = {src: enriched_study(rows, logits, src, [20, 40, 80, 160], 200, args.replicates, args.calib_sensitivity, boundary, args.seed)
                for src in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        'enriched_calibration_samples': enriched,
        'tier': 'TX (exposed mBRSET and BRSET labels; post hoc; E_768 frozen round-3 policy, no refit of the model)',
        'targets': {'sensitivity': sc.TARGET_SENS, 'specificity': sc.TARGET_SPEC}, 'unit': 'patients (clusters)',
        'config': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, 'frozen_boundary': boundary,
        'results': result}, indent=2, default=float))
    for source, e in result.items():
        print(f"{source}: {e['n_images']} images, {e['n_clusters']} patients, {e['referable_prevalence']:.3f} referable; frozen "
              f"{e['frozen']['sensitivity']:.3f}/{e['frozen']['specificity']:.3f}; oracle {e['oracle'].get('sensitivity', float('nan')):.3f}/"
              f"{e['oracle'].get('specificity', float('nan')):.3f}, AUC {e['oracle']['auc']:.3f}")
        for n, s in enriched.get(source, {}).items():
            print(f"   enriched {n} referable + 200 other: P(both)={s['p_both_targets']:.2f} P(sens)={s['p_sensitivity_target']:.2f} sens {s['sensitivity_mean']:.3f} "
                  f"(p10 {s['sensitivity_p10']:.3f}) spec {s['specificity_mean']:.3f} (p10 {s['specificity_p10']:.3f})")
        for n, s in e['site'].items():
            print(f"   N={n}: P(both)={s['p_both_targets']:.2f} P(sens)={s['p_sensitivity_target']:.2f} sens {s['sensitivity_mean']:.3f} "
                  f"(p10 {s['sensitivity_p10']:.3f}) spec {s['specificity_mean']:.3f} (p10 {s['specificity_p10']:.3f})")


if __name__ == '__main__':
    main()
