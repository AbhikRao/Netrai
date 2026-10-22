#!/usr/bin/env python3
"""X9: DeepDRiD-aware quality score on top of the frozen MobileNetV3-Small encoder (development stage; Challenge-2 labels never read).

Stage 0  embeddings (576-d, exact full-frame 224 bilinear contract) of DeepDRiD training (1,200) and validation (400) images,
         labels = the release's training/validation quality columns (development data; direction verified on them:
         Overall quality 1 = gradable, higher Clarity/Field = better, lower Artifact = better).
Stage 1  (a) the existing EyeQ candidate head: score = temperature-scaled P(reject);
         (b) DeepDRiD heads: ungradable (Overall quality = 0), heavy artifact (Artifact >= 6), poor clarity (Clarity <= 4),
             poor field (Field <= 4): logistic regressions on the embeddings, C selected by patient-grouped CV on training.
Stage 2  (c) final score = logistic regression on [EyeQ reject logit, the four head logits], fitted on DeepDRiD TRAINING with
         patient-grouped out-of-fold stage-1 predictions, evaluated on DeepDRiD VALIDATION (tier TD).
Operating point: the score threshold giving <= 5% false-reject of gradable training images (out-of-fold), fixed here.
Development gate (O5): validation AUC for ungradable of (c) >= that of (a) + 0.02; otherwise the Challenge-2 labels stay reserved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from models.quality_model import QualityModel, quality_input
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m

ROOT = Path(__file__).resolve().parents[1]
DRID = ROOT / 'data/DRiD/DeepDRiD-master/regular_fundus_images'
CANDIDATE = ROOT / 'python/weights/quality_model_candidate.pth'
CANDIDATE_CFG = ROOT / 'python/weights/quality_model_candidate.json'
HEADS = {'ungradable': lambda d: d['Overall quality'] == 0, 'artifact': lambda d: d['Artifact'] >= 6,
         'clarity': lambda d: d['Clarity'] <= 4, 'field': lambda d: d['Field definition'] <= 4}


def load_split(split: str) -> pd.DataFrame:
    frame = pd.read_csv(DRID / f'regular-fundus-{split}' / f'regular-fundus-{split}.csv')
    frame['path'] = [DRID / f'regular-fundus-{split}' / 'Images' / str(p) / f'{i}.jpg'
                     for p, i in zip(frame.patient_id, frame.image_id)]
    return frame


def embed(paths, model, batch=32) -> np.ndarray:
    out = []
    for start in range(0, len(paths), batch):
        arrays = [quality_input(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)) for p in paths[start:start + batch]]
        with torch.inference_mode():
            out.append(model.encode(torch.from_numpy(np.stack(arrays))).numpy())
    return np.concatenate(out)


def load_candidate() -> QualityModel:
    model = QualityModel()
    model.load_state_dict(torch.load(CANDIDATE, map_location='cpu', weights_only=True))
    return model.eval()


def eyeq_reject_logit(model: QualityModel, features: np.ndarray, temperature: float) -> np.ndarray:
    with torch.inference_mode():
        logits = model.head(torch.from_numpy(features).float()).numpy() / temperature
    shifted = logits - logits.max(1, keepdims=True)
    p = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
    p_reject = np.clip(p[:, 2], 1e-6, 1 - 1e-6)
    return np.log(p_reject / (1 - p_reject))


def fit_lr(x, y, c):
    mean, std = x.mean(0), x.std(0)
    std[std == 0] = 1.0
    clf = LogisticRegression(C=c, max_iter=3000, class_weight='balanced').fit((x - mean) / std, y)
    return lambda z: clf.decision_function((z - mean) / std)


def oof_scores(x, y, groups, c, folds=5):
    out = np.zeros(len(y))
    for tr, te in GroupKFold(folds).split(x, y, groups):
        out[te] = fit_lr(x[tr], y[tr], c)(x[te])
    return out


def select_c(x, y, groups, grid=(0.003, 0.01, 0.03, 0.1, 0.3)):
    scores = {c: m.auc(y, oof_scores(x, y, groups, c)) for c in grid}
    return max(scores, key=scores.get), scores


def threshold_for_false_reject(score, ungradable, rate=0.05):
    return float(np.quantile(score[~ungradable], 1 - rate))


def fit_pipeline(xt: np.ndarray, train: pd.DataFrame, qmodel: QualityModel, temperature: float):
    """Fit stage 1 + stage 2 on DeepDRiD training embeddings. Returns (predict, info); ``predict(features)`` gives every score."""
    groups = train.patient_id.to_numpy()
    y = {k: f(train).to_numpy() for k, f in HEADS.items()}
    eyeq_t = eyeq_reject_logit(qmodel, xt, temperature)
    chosen, head_oof, head_fit = {}, {}, {}
    for name in HEADS:
        c, _ = select_c(xt, y[name], groups)
        chosen[name] = c
        head_oof[name] = oof_scores(xt, y[name], groups, c)
        head_fit[name] = fit_lr(xt, y[name], c)
    stack_t = np.column_stack([eyeq_t] + [head_oof[n] for n in HEADS])
    final = fit_lr(stack_t, y['ungradable'], 1.0)
    oof_final = oof_scores(stack_t, y['ungradable'], groups, 1.0)
    threshold = threshold_for_false_reject(oof_final, y['ungradable'])

    def predict(features: np.ndarray) -> dict:
        eyeq = eyeq_reject_logit(qmodel, features, temperature)
        heads = {n: head_fit[n](features) for n in HEADS}
        stacked = final(np.column_stack([eyeq] + [heads[n] for n in HEADS]))
        return {'eyeq_reject_logit': eyeq, 'heads': heads, 'final': stacked}

    return predict, {'regularisation_C': chosen, 'operating_threshold_5pct_false_reject_oof': threshold, 'temperature': temperature}


def run(args) -> dict:
    train, valid = load_split('training'), load_split('validation')
    cfg = json.loads(CANDIDATE_CFG.read_text())
    temperature = float(cfg['confidence_temperature'])
    cache = args.cache
    if cache and cache.exists():
        z = np.load(cache)
        xt, xv = z['train'], z['valid']
    else:
        model = load_candidate()
        xt, xv = embed(train.path.tolist(), model), embed(valid.path.tolist(), model)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, train=xt, valid=xv)
    model = load_candidate()
    y_v = {k: f(valid).to_numpy() for k, f in HEADS.items()}
    predict, info = fit_pipeline(xt, train, model, temperature)
    scores = predict(xv)
    eyeq_v, head_val, score_v = scores['eyeq_reject_logit'], scores['heads'], scores['final']
    threshold = info['operating_threshold_5pct_false_reject_oof']
    result = {'tier': 'TD (DeepDRiD validation; development)',
              'prevalence_ungradable': {'train': float(HEADS['ungradable'](train).mean()), 'valid': float(y_v['ungradable'].mean())}, **info}
    clusters = valid.patient_id.to_numpy()
    pos = y_v['ungradable']

    def auc_of(score):
        return lambda rows: {'auc': m.auc(pos[rows], score[rows])}

    result['validation_auc'] = {
        'a_eyeq_candidate': m.auc(pos, eyeq_v), 'b_direct_ungradable_head': m.auc(pos, head_val['ungradable']), 'c_stacked': m.auc(pos, score_v)}
    result['stacked_minus_candidate'] = bs.paired_bootstrap(auc_of(score_v), auc_of(eyeq_v), clusters, 1000, 20261106)['auc']
    called = score_v >= threshold
    result['operating_point_on_validation'] = {'reject_recall': float(called[pos].mean()), 'false_reject_rate': float(called[~pos].mean())}
    result['factor_head_validation_auc'] = {n: m.auc(y_v[n], head_val[n]) for n in HEADS if n != 'ungradable' and y_v[n].any()}
    gain = result['validation_auc']['c_stacked'] - result['validation_auc']['a_eyeq_candidate']
    result['gate'] = {'rule': 'AUC(c) >= AUC(a) + 0.02', 'gain': float(gain), 'passed': bool(gain >= 0.02)}
    if args.save_model:
        np.savez(args.save_model, stack_weights=np.array([]))  # models are refit at lock time from the cached embeddings
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cache', type=Path, default=ROOT / 'results/model_improvement/experiments/X9/embeddings_dev.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--save-model', type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; results are immutable')
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps({k: result[k] for k in ('validation_auc', 'stacked_minus_candidate', 'operating_point_on_validation', 'gate')}, indent=1, default=float))


if __name__ == '__main__':
    main()
