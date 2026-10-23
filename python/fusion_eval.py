#!/usr/bin/env python3
"""X12: does lesion evidence add ranking information to the grading score on UNSEEN sites?

Rows: the held-out images of the leave-one-dataset-out (LODO) runs (each site was never trained on or selected on by its
own run). Per image: the run's expected-grade score and lesion features from ``lesion_features.py`` (threshold-free
summaries of the frozen lesion segmenter, which never saw these images). Per target site S the fusion logistic
regression is fitted on the OTHER sites' rows only (leave-one-site-out), standardised with their statistics, and S's rows
are only scored; the paired cluster bootstrap compares fused vs grading-only AUC and specificity at 90% sensitivity on S.

Pre-registered rule (O2 review): success = mean AUC gain >= +0.010 over the sites, paired 95% CI excluding 0 on >= 2
sites and no site worse by more than 0.005. Features and model are fixed in this file before any lesion feature is read.
Lesion thresholds come from a frozen JSON (validation-chosen Dice thresholds); histogram resolution is 0.01.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lesion_features import BINS, COMPONENT_THRESHOLDS
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m

LESIONS = ('EX', 'HE', 'MA', 'SE')


def load_features(folder: Path, image_ids) -> dict:
    out = {}
    for image_id in image_ids:
        with np.load(folder / f'{image_id}.npz') as z:
            out[image_id] = {k: z[k] for k in ('hist', 'max', 'components', 'fov_pixels')}
    return out


def lesion_vector(item: dict, thresholds: dict, ma_threshold: float) -> np.ndarray:
    """12 numbers: per lesion log area fraction at the frozen threshold, maximum probability, log1p component count."""
    vector = []
    for i, lesion in enumerate(LESIONS):
        t = ma_threshold if lesion == 'MA' else thresholds[lesion]
        bin_index = min(BINS - 1, max(0, int(round(t * BINS))))
        area = item['hist'][i][bin_index:].sum() / max(item['fov_pixels'], 1)
        grid = int(np.argmin([abs(t - c) for c in COMPONENT_THRESHOLDS]))
        vector += [np.log(area + 1e-5), float(item['max'][i]), np.log1p(item['components'][i][grid])]
    return np.array(vector, float)


def standardise(train: np.ndarray, other: np.ndarray):
    mean, std = train.mean(0), train.std(0)
    std[std == 0] = 1.0
    return (train - mean) / std, (other - mean) / std


def fit_score(x_train, y_train, x_test, c: float = 1.0) -> np.ndarray:
    a, b = standardise(x_train, x_test)
    model = LogisticRegression(C=c, max_iter=2000, class_weight='balanced').fit(a, y_train)
    return model.decision_function(b)


def spec_at_sens(positive, score, sensitivity=0.90) -> float:
    cut = np.quantile(score[positive], 1 - sensitivity)
    return float((score[~positive] < cut).mean())


def within_site_standardise(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """z-score every column inside each site using that site's own (unlabelled) mean and SD."""
    out = np.array(values, float)
    one_d = out.ndim == 1
    if one_d:
        out = out[:, None]
    for site in frame.source.unique():
        mask = (frame.source == site).to_numpy()
        mean, std = out[mask].mean(0), out[mask].std(0)
        std[std == 0] = 1.0
        out[mask] = (out[mask] - mean) / std
    return out[:, 0] if one_d else out


def leave_one_site_out(frame: pd.DataFrame, grading: np.ndarray, lesion: np.ndarray, replicates=1000, seed=20261101,
                       within_site: bool = False) -> dict:
    """frame: image_id, source, grade, group_id; grading (N,) expected-grade score; lesion (N, 12)."""
    positive = frame.grade.to_numpy() >= 2
    sites = sorted(frame.source.unique())
    if within_site:  # X12b: label-free per-site standardisation of the score and every lesion feature
        grading, lesion = within_site_standardise(frame, grading), within_site_standardise(frame, lesion)
    scores = {name: np.zeros(len(frame)) for name in ('grading', 'lesion_only', 'fused')}
    scores['grading'][:] = grading
    for site in sites:
        test = (frame.source == site).to_numpy()
        train = ~test
        scores['lesion_only'][test] = fit_score(lesion[train], positive[train], lesion[test])
        fused = np.column_stack([grading, lesion])
        scores['fused'][test] = fit_score(fused[train], positive[train], fused[test])
    result = {'sites': {}}
    for site in sites:
        idx = np.flatnonzero((frame.source == site).to_numpy())
        pos, clusters = positive[idx], frame.group_id.to_numpy()[idx]

        def stat(a, b):
            return lambda rows: {'auc': m.auc(pos[rows], scores[a][idx][rows]) - m.auc(pos[rows], scores[b][idx][rows]),
                                 'spec90': spec_at_sens(pos[rows], scores[a][idx][rows]) - spec_at_sens(pos[rows], scores[b][idx][rows])}

        entry = {'n': int(len(idx)), 'prevalence': float(pos.mean())}
        for name in scores:
            entry[name] = {'auc': m.auc(pos, scores[name][idx]), 'spec_at_sens90': spec_at_sens(pos, scores[name][idx])}
        entry['fused_minus_grading'] = bs.bootstrap(stat('fused', 'grading'), clusters, replicates, seed)
        entry['lesion_only_minus_grading'] = bs.bootstrap(stat('lesion_only', 'grading'), clusters, replicates, seed)
        result['sites'][site] = entry
    gains = [result['sites'][s]['fused_minus_grading']['auc'] for s in sites]
    mean_gain = float(np.mean([g['estimate'] for g in gains]))
    result['rule'] = {'mean_auc_gain': mean_gain,
                      'sites_ci_excludes_zero': int(sum(g['ci95'][0] > 0 for g in gains)),
                      'worst_site_gain': float(min(g['estimate'] for g in gains)),
                      'success': bool(mean_gain >= 0.010 and sum(g['ci95'][0] > 0 for g in gains) >= 2
                                      and min(g['estimate'] for g in gains) >= -0.005)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='LODO run folders (heldout rows/outputs)')
    parser.add_argument('--features', type=Path, required=True, help='folder of <image_id>.npz from lesion_features.py')
    parser.add_argument('--thresholds', type=Path, required=True, help='json: dice_thresholds{EX,HE,MA,SE}, ma_lesion_threshold')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--within-site-standardise', action='store_true', help='X12b: z-score score and features within each site')
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; results are immutable')
    frames, grades, vectors = [], [], []
    cfg = json.loads(args.thresholds.read_text())
    for run in args.runs:
        rows = pd.read_csv(run / 'heldout_rows.csv')
        outputs = np.load(run / 'best_heldout_outputs.npy')
        feats = load_features(args.features, rows.image_id)
        frames.append(rows)
        grades.append(m.expected_grade_score(outputs))
        vectors.append(np.stack([lesion_vector(feats[i], cfg['dice_thresholds'], cfg['ma_lesion_threshold']) for i in rows.image_id]))
    frame = pd.concat(frames, ignore_index=True)
    result = leave_one_site_out(frame, np.concatenate(grades), np.vstack(vectors), args.replicates,
                                within_site=args.within_site_standardise)
    result.update({'tier': 'TL (leave-dataset-out held-out sources; lesion model never saw them)', 'within_site_standardise': args.within_site_standardise,
                   'thresholds': cfg, 'runs': [str(r) for r in args.runs]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float))
    for site, e in result['sites'].items():
        print(site, {k: round(e[k]['auc'], 4) for k in ('grading', 'lesion_only', 'fused')},
              'fused-grading', round(e['fused_minus_grading']['auc']['estimate'], 4), e['fused_minus_grading']['auc']['ci95'])
    print(result['rule'])


if __name__ == '__main__':
    main()
