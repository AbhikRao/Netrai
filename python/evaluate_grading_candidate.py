#!/usr/bin/env python3
"""Fit a candidate policy on APTOS calibration, then score frozen IDRiD.

Never replace the legacy deployment artifacts. IDRiD has been inspected before,
so its outputs are retrospective development evidence, not fresh validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from models.netrai_model import NetrAIModel
from train_grading import load_manifest
from utils.artifacts import atomic_json, sha256_file
from utils.calibration import fit_temperature, softmax, temperature_scale_logits, calibration_metrics
from utils.metrics import quadratic_weighted_kappa
from utils.preprocessing import preprocess_for_model
from utils.validation import grades, image_ids, ordinal_thresholds


def load_candidate(path: Path, device: str):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if (not config.get('pretrained') or config.get('initial_checkpoint') or
            config.get('preprocessing_version') != 'training-linear-v2'):
        raise ValueError('Candidate must use ImageNet initialization and the corrected input contract')
    model = NetrAIModel(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval().to(device)
    return model, config


def infer(files: list[Path], model, device: str, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(files), batch_size):
        arrays = []
        for path in files[start:start+batch_size]:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f'Cannot decode {path}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            arrays.append(preprocess_for_model(rgb, version='training-linear-v2'))
        tensor = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2).copy()).float().to(device)
        with torch.inference_mode():
            grade_logits, referral_logit = model(tensor)
        outputs.append(torch.cat([grade_logits, referral_logit[:, None]], dim=1).cpu().numpy())
        print(f'Inferred {min(start+batch_size, len(files))}/{len(files)}', flush=True)
    result = np.concatenate(outputs).astype(np.float64)
    if result.shape != (len(files), 6) or not np.isfinite(result).all():
        raise ValueError('Invalid candidate logits')
    return result


def choose_referral_threshold(scores: np.ndarray, labels: np.ndarray, min_specificity=0.85) -> dict:
    positive = labels >= 2
    if not positive.any() or positive.all():
        raise ValueError('Calibration needs referable and nonreferable labels')
    values = np.unique(scores)
    candidates = np.r_[np.nextafter(values[0], -np.inf),
                       (values[:-1]+values[1:])/2,
                       np.nextafter(values[-1], np.inf)]
    eligible = []
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = float(predicted[positive].mean())
        specificity = float((~predicted[~positive]).mean())
        if specificity >= min_specificity:
            eligible.append((sensitivity, specificity, float(threshold)))
    if not eligible:
        raise RuntimeError('No threshold meets specificity floor')
    sensitivity, specificity, threshold = max(eligible)
    return {'threshold': threshold, 'calibration_sensitivity': sensitivity,
            'calibration_specificity': specificity,
            'selection_rule': f'maximize sensitivity subject to specificity >= {min_specificity}',
            'calibration_target_passed': sensitivity > 0.90 and specificity > 0.85}


def fit_other_thresholds(scores: np.ndarray, labels: np.ndarray, referral_threshold: float) -> list[float]:
    # Optimize only the three non-referral boundaries; referral threshold stays
    # fixed after the prespecified sensitivity/specificity policy is selected.
    if not 0 < referral_threshold < 4:
        raise ValueError('Selected referral score boundary lies outside (0, 4)')
    thresholds = np.array([referral_threshold/2, referral_threshold,
                           referral_threshold+(4-referral_threshold)/3,
                           referral_threshold+2*(4-referral_threshold)/3])
    for _ in range(2):
        for index in (0, 2, 3):
            lower = thresholds[index-1] if index else 0.0
            upper = thresholds[index+1] if index < 3 else 4.0
            available = np.unique(scores[(scores > lower) & (scores < upper)])
            options = np.r_[thresholds[index], (available[:-1]+available[1:])/2]
            best = (-np.inf, thresholds[index])
            for candidate in options:
                trial = thresholds.copy()
                trial[index] = candidate
                predicted = (scores[:, None] >= trial).sum(axis=1)
                value = quadratic_weighted_kappa(labels, predicted)
                if np.isfinite(value) and value > best[0]:
                    best = (value, candidate)
            thresholds[index] = best[1]
    return ordinal_thresholds(thresholds).tolist()


def summarize(labels: np.ndarray, logits: np.ndarray, policy: dict,
              external_intervals=False) -> tuple[dict, pd.DataFrame]:
    raw = softmax(logits[:, :5])
    calibrated = temperature_scale_logits(logits[:, :5], policy['temperature'])
    score = raw @ np.arange(5)
    thresholds = ordinal_thresholds(policy['thresholds'])
    predicted = (score[:, None] >= thresholds).sum(axis=1)
    referable = labels >= 2
    positive = predicted >= 2
    summary = {
        'images': len(labels), 'referable': int(referable.sum()),
        'accuracy': float(np.mean(predicted == labels)),
        'qwk': float(quadratic_weighted_kappa(labels, predicted)),
        'per_grade_support': np.bincount(labels, minlength=5).astype(int).tolist(),
        'per_grade_recall': [float(np.mean(predicted[labels == grade] == grade))
                             if np.any(labels == grade) else None for grade in range(5)],
        'referable_sensitivity': float(positive[referable].mean()) if referable.any() else None,
        'referable_specificity': float((~positive[~referable]).mean()) if (~referable).any() else None,
        'referable_auc_score': float(roc_auc_score(referable, score))
            if referable.any() and (~referable).any() else None,
        'calibration': calibration_metrics(calibrated, labels, predicted),
    }
    if external_intervals and referable.any() and (~referable).any():
        rng = np.random.default_rng(42)
        positive_indices, negative_indices = np.flatnonzero(referable), np.flatnonzero(~referable)
        positive_draws = rng.choice(positive_indices, size=(5000, len(positive_indices)), replace=True)
        negative_draws = rng.choice(negative_indices, size=(5000, len(negative_indices)), replace=True)
        summary['image_bootstrap_95'] = {
            'referable_sensitivity': np.quantile(positive[positive_draws].mean(axis=1), [.025, .975]).tolist(),
            'referable_specificity': np.quantile((~positive[negative_draws]).mean(axis=1), [.025, .975]).tolist(),
            'replicates': 5000, 'unit': 'image; patient identity unavailable',
        }
    rows = pd.DataFrame({'true_grade': labels, 'predicted_grade': predicted,
                         'continuous_score': score, 'referable_score_threshold': thresholds[1],
                         'referable_prediction': positive,
                         'referral_head_probability': 1/(1+np.exp(-np.clip(logits[:, 5], -80, 80)))})
    for grade in range(5):
        rows[f'raw_prob_grade_{grade}'] = raw[:, grade]
        rows[f'calibrated_prob_grade_{grade}'] = calibrated[:, grade]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('fit-policy', 'test-idrid'))
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--manifest', type=Path, help='Corrected APTOS manifest for policy fitting')
    parser.add_argument('--policy', type=Path, help='Frozen policy for IDRiD scoring')
    parser.add_argument('--idrid-root', type=Path, help='Local complete IDRiD root for test scoring')
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error('Use a new output directory and a positive batch size')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    model, config = load_candidate(args.checkpoint, args.device)
    model_hash = sha256_file(args.checkpoint)
    if args.mode == 'fit-policy':
        if not args.manifest:
            parser.error('--manifest is required for fit-policy')
        frame = load_manifest(args.manifest.resolve())
        if sha256_file(args.manifest) != config['manifest_sha256']:
            raise ValueError('Checkpoint/manifest identity differs')
        selected = frame.loc[frame.split.eq('calibration')].reset_index(drop=True)
        ids = image_ids(selected.image_id).to_numpy()
        labels = grades(selected.grade)
        files = [Path(name) for name in selected.image_path]
        logits = infer(files, model, args.device, args.batch_size)
        raw = softmax(logits[:, :5])
        score = raw @ np.arange(5)
        referral = choose_referral_threshold(score, labels)
        temperature, optimizer = fit_temperature(logits[:, :5], labels)
        policy = {'schema_version': 1, 'status': 'APTOS_development_calibration_not_clinical_policy',
                  'checkpoint_sha256': model_hash, 'manifest_sha256': sha256_file(args.manifest),
                  'calibration_rows': len(selected), 'preprocessing_version': 'training-linear-v2',
                  'thresholds': fit_other_thresholds(score, labels, referral['threshold']),
                  'temperature': float(temperature), 'temperature_optimizer': optimizer,
                  'referable_selection': referral}
    else:
        if not args.policy or not args.idrid_root:
            parser.error('--policy and --idrid-root are required for test-idrid')
        policy = json.loads(args.policy.read_text())
        if (policy['checkpoint_sha256'] != model_hash or
                policy['preprocessing_version'] != 'training-linear-v2'):
            raise ValueError('Frozen policy and candidate model do not match')
        root = args.idrid_root.resolve() / 'B. Disease Grading'
        labels_path = root / '2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
        source = pd.read_csv(labels_path)
        ids = image_ids(source['Image name']).to_numpy()
        labels = grades(source['Retinopathy grade'])
        files = [root / '1. Original Images/b. Testing Set' / f'{name}.jpg' for name in ids]
        logits = infer(files, model, args.device, args.batch_size)
    summary, rows = summarize(labels, logits, policy, args.mode == 'test-idrid')
    rows.insert(0, 'image_id', ids)
    args.output.mkdir(parents=True)
    rows.to_csv(args.output / 'per_image.csv', index=False)
    if args.mode == 'fit-policy':
        atomic_json(args.output / 'policy.json', policy)
    atomic_json(args.output / 'summary.json', {
        'schema_version': 1, 'status': 'retrospective_development_not_clinical_validation',
        'mode': args.mode, 'checkpoint_sha256': model_hash,
        'policy_sha256': sha256_file(args.policy) if args.mode == 'test-idrid' else sha256_file(args.output / 'policy.json'),
        'metrics': summary,
        'limitations': ['APTOS has no verified patient IDs',
                        'IDRiD test labels were exposed during project development',
                        'Do not promote until an identity-reviewed fresh external cohort passes predeclared targets']})
    print('Wrote', args.output / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
