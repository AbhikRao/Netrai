#!/usr/bin/env python3
"""X6: frozen RETFound (ViT-L/16 MAE, CC BY-NC 4.0, non-commercial research) features as a probe on UNSEEN sites.

``extract``  computes 224-px frozen features (average of patch tokens after the final norm, 1024-d) for the held-out
             images of the leave-one-dataset-out (LODO) runs. No label is used.
``analyze``  leave-one-site-out probes (pre-registered in the O2 review):
             (a) probe alone: logistic regression on the features of the other sites, scored on the target site;
             (b) stacking: [grading score, out-of-fold probe score]; the probe score of each training site comes from a
                 probe fitted on the remaining training sites (inner leave-one-site-out), the target site's score from a
                 probe fitted on all training sites; a second logistic regression combines the two scores.
             Paired cluster bootstrap against the grading score alone. Rules fixed in advance:
             (a) beats grading by >= +0.02 AUC on >= 3 of 4 sites before any fine-tuning is considered;
             (b) reported with the X12 rule (mean >= +0.010, CI above 0 on >= 2 sites, no site worse than -0.005).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_encoder(weights: Path | None, arch: str = 'vit_large_patch16_224'):
    import timm
    model = timm.create_model(arch, pretrained=False, num_classes=0, global_pool='avg')
    if weights is not None:
        state = torch.load(weights, map_location='cpu', weights_only=False)
        state = state.get('model', state)
        state = {k: v for k, v in state.items() if not k.startswith(('head.', 'decoder', 'mask_token'))}
        missing, unexpected = model.load_state_dict(state, strict=False)
        important = [k for k in missing if not k.startswith('fc_norm')]
        if important:
            raise ValueError(f'checkpoint does not match {arch}: missing {important[:5]}')
        print(f'loaded RETFound weights; missing {list(missing)}, unexpected {list(unexpected)[:5]}', flush=True)
    return model.eval()


def preprocess(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    image = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(((image - MEAN) / STD).transpose(2, 0, 1).copy())


def extract(weights: Path, runs, roots, output: Path, batch=32, limit=0, device='cpu', arch='vit_large_patch16_224') -> None:
    model = build_encoder(weights, arch).to(device)
    output.mkdir(parents=True, exist_ok=True)
    for run in runs:
        fold = Path(run).parent.name
        target = output / f'{fold}.npy'
        if target.exists():
            continue
        rows = pd.read_csv(Path(run) / 'heldout_rows.csv')
        if limit:
            rows = rows.iloc[:limit]
        feats = []
        for start in range(0, len(rows), batch):
            tensors = []
            for image_id in rows.image_id.iloc[start:start + batch]:
                path = next((Path(r) / f'{image_id}.png' for r in roots if (Path(r) / f'{image_id}.png').is_file()), None)
                if path is None:
                    raise FileNotFoundError(image_id)
                tensors.append(preprocess(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)))
            with torch.inference_mode():
                feats.append(model(torch.stack(tensors).to(device)).float().cpu().numpy())
            if (start // batch) % 20 == 0:
                print(f'{fold}: {start}/{len(rows)}', flush=True)
        np.save(target, np.concatenate(feats))


def probe_scores(x_train, y_train, x_test, c=0.05):
    mean, std = x_train.mean(0), x_train.std(0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=c, max_iter=3000, class_weight='balanced').fit((x_train - mean) / std, y_train)
    return clf.decision_function((x_test - mean) / std)


def analyze_sites(frame: pd.DataFrame, grading: np.ndarray, features: np.ndarray, c=0.05, replicates=1000, seed=20261102) -> dict:
    positive = frame.grade.to_numpy() >= 2
    sites = sorted(frame.source.unique())
    scores = {'grading': grading.astype(float), 'probe': np.zeros(len(frame)), 'stack': np.zeros(len(frame))}
    for site in sites:
        test = (frame.source == site).to_numpy()
        train_sites = [s for s in sites if s != site]
        scores['probe'][test] = probe_scores(features[~test], positive[~test], features[test], c)
        oof = np.zeros(len(frame))  # out-of-fold probe scores of the training sites
        for inner in train_sites:
            inner_mask = (frame.source == inner).to_numpy()
            fit = (~test) & (~inner_mask)
            oof[inner_mask] = probe_scores(features[fit], positive[fit], features[inner_mask], c)
        oof[test] = scores['probe'][test]
        stacked = np.column_stack([grading, oof])
        a, b = stacked[~test], stacked[test]
        mean, std = a.mean(0), a.std(0)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced').fit((a - mean) / std, positive[~test])
        scores['stack'][test] = clf.decision_function((b - mean) / std)
    result = {'sites': {}}
    for site in sites:
        idx = np.flatnonzero((frame.source == site).to_numpy())
        pos, clusters = positive[idx], frame.group_id.to_numpy()[idx]

        def diff(name):
            return lambda rows: {'auc': m.auc(pos[rows], scores[name][idx][rows]) - m.auc(pos[rows], scores['grading'][idx][rows])}

        entry = {'n': int(len(idx))}
        for name in scores:
            entry[name] = {'auc': m.auc(pos, scores[name][idx])}
        entry['probe_minus_grading'] = bs.bootstrap(diff('probe'), clusters, replicates, seed)['auc']
        entry['stack_minus_grading'] = bs.bootstrap(diff('stack'), clusters, replicates, seed)['auc']
        result['sites'][site] = entry
    probe = [result['sites'][s]['probe_minus_grading']['estimate'] for s in sites]
    stack = [result['sites'][s]['stack_minus_grading'] for s in sites]
    result['rule_probe_alone'] = {'sites_with_gain_ge_0.02': int(sum(g >= 0.02 for g in probe)), 'success': bool(sum(g >= 0.02 for g in probe) >= 3)}
    result['rule_stack'] = {'mean_auc_gain': float(np.mean([s['estimate'] for s in stack])),
                            'sites_ci_excludes_zero': int(sum(s['ci95'][0] > 0 for s in stack)),
                            'worst_site_gain': float(min(s['estimate'] for s in stack)),
                            'success': bool(np.mean([s['estimate'] for s in stack]) >= 0.010 and sum(s['ci95'][0] > 0 for s in stack) >= 2
                                            and min(s['estimate'] for s in stack) >= -0.005)}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    e = sub.add_parser('extract')
    e.add_argument('--weights', type=Path, required=True)
    e.add_argument('--runs', type=Path, nargs='+', required=True)
    e.add_argument('--roots', type=Path, nargs='+', required=True)
    e.add_argument('--output', type=Path, required=True)
    e.add_argument('--limit', type=int, default=0)
    e.add_argument('--device', default='cpu')
    a = sub.add_parser('analyze')
    a.add_argument('--runs', type=Path, nargs='+', required=True)
    a.add_argument('--features', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--c', type=float, default=0.05)
    a.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args(argv)
    if args.command == 'extract':
        extract(args.weights, args.runs, args.roots, args.output, limit=args.limit, device=args.device)
        return
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; results are immutable')
    frames, grades, feats = [], [], []
    for run in args.runs:
        rows = pd.read_csv(run / 'heldout_rows.csv')
        frames.append(rows)
        grades.append(m.expected_grade_score(np.load(run / 'best_heldout_outputs.npy')))
        feats.append(np.load(args.features / f'{run.parent.name}.npy'))
    result = analyze_sites(pd.concat(frames, ignore_index=True), np.concatenate(grades), np.vstack(feats), args.c, args.replicates)
    result.update({'tier': 'TL (leave-dataset-out held-out sources; RETFound never fine-tuned)', 'c': args.c})
    args.output.write_text(json.dumps(result, indent=2, default=float))
    for site, s in result['sites'].items():
        print(site, {k: round(s[k]['auc'], 4) for k in ('grading', 'probe', 'stack')}, 'probe-grading', round(s['probe_minus_grading']['estimate'], 4))
    print(result['rule_probe_alone'], result['rule_stack'])


if __name__ == '__main__':
    sys.exit(main())
