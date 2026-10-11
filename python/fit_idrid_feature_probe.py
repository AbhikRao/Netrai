#!/usr/bin/env python3
"""Train a target-domain referral probe from frozen APTOS model features.

Only IDRiD's 413 official training labels are used for representation,
regularization, and threshold selection. The 103 test labels are read after
that policy is fixed. This is retrospective domain-adaptation research, not
independent clinical validation or a replacement for the five-grade model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from inference import _get_model
from utils.preprocessing import preprocess_for_model
from utils.validation import grades as validate_grades, image_ids as validate_ids


ROOT = Path(__file__).resolve().parents[1]
GRADING = ROOT / 'data/idrid/B. Disease Grading'


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            sha.update(block)
    return sha.hexdigest()


def extract(split, labels_path, image_dir, cache, model, batch_size):
    labels = pd.read_csv(labels_path)
    image_ids = validate_ids(labels['Image name']).to_numpy()
    grades = validate_grades(labels['Retinopathy grade'])
    files = [image_dir/f'{image_id}.jpg' for image_id in image_ids]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError(f'Missing image in {split} source')
    signature = {'labels_sha256': digest(labels_path),
                 'checkpoint_sha256': digest(ROOT/'python/weights/netrai_fold0_best.pth'),
                 'images': [[path.name, digest(path)] for path in files],
                 'preprocessing': 'legacy-area-v1', 'feature_precision': 'float16_roundtrip',
                 'code_sha256': {name: digest(ROOT/'python'/name) for name in
                                 ('fit_idrid_feature_probe.py', 'utils/preprocessing.py',
                                  'models/netrai_model.py')},
                 'versions': {'torch': str(torch.__version__), 'opencv': cv2.__version__}}
    cache_path = cache/f'{split}_features.npz'
    if cache_path.is_file():
        with np.load(cache_path) as saved:
            if json.loads(str(saved['signature'])) == signature:
                return image_ids, grades, saved['backbone'].astype(np.float32), saved['head'].astype(np.float32)
            raise ValueError(f'Stale/unbound feature cache: {cache_path}; use a new output directory')
    backbones, heads = [], []
    for start in range(0, len(files), batch_size):
        batch = []
        for path in files[start:start+batch_size]:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f'Could not decode {path}')
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch.append(preprocess_for_model(rgb, size=512))
        tensor = torch.from_numpy(np.stack(batch).transpose(0, 3, 1, 2)).float().to(next(model.parameters()).device)
        with torch.inference_mode():
            backbone = model.backbone(tensor)
            head = model.head(backbone)
        backbones.append(backbone.cpu().numpy().astype(np.float16))
        heads.append(head.cpu().numpy().astype(np.float16))
        print(f'{split}: {min(start+batch_size, len(files))}/{len(files)} images', flush=True)
    backbone = np.vstack(backbones)
    head = np.vstack(heads)
    cache.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.features-', dir=cache)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            np.savez_compressed(stream, signature=json.dumps(signature), backbone=backbone, head=head)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return image_ids, grades, backbone.astype(np.float32), head.astype(np.float32)


def threshold_for_sensitivity(labels, probabilities, target=0.92):
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    eligible = np.flatnonzero(tpr >= target)
    if not len(eligible):
        raise ValueError('Sensitivity target cannot be reached')
    best = eligible[np.argmin(fpr[eligible])]
    return float(thresholds[best]), float(tpr[best]), float(1-fpr[best])


def performance(grades, scores, threshold):
    labels = np.asarray(grades) >= 2
    predicted = scores >= threshold
    return {'images': len(labels), 'referable': int(labels.sum()),
            'sensitivity': float(predicted[labels].mean()),
            'specificity': float((~predicted[~labels]).mean()),
            'auc': float(roc_auc_score(labels, scores)),
            'tp': int((predicted & labels).sum()),
            'fn': int((~predicted & labels).sum()),
            'tn': int((~predicted & ~labels).sum()),
            'fp': int((predicted & ~labels).sum())}


def compare_original(test_ids, test_grades, adapted_scores, threshold):
    source = ROOT/'results/verification_2026-09-18/idrid_external_grading/per_image_results.csv'
    original = pd.read_csv(source).set_index('id_code').loc[test_ids]
    if not np.array_equal(original.true_grade.to_numpy(int), test_grades):
        raise ValueError('Original and adapted IDRiD test labels differ')
    labels = test_grades >= 2
    old = original.referable_prediction.to_numpy(bool)
    new = adapted_scores >= threshold
    rng = np.random.default_rng(42)
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    positive_draws = rng.choice(positive, size=(5000, len(positive)), replace=True)
    negative_draws = rng.choice(negative, size=(5000, len(negative)), replace=True)
    sensitivity_diff = (new[positive_draws].astype(int)
                        - old[positive_draws].astype(int)).mean(axis=1)
    specificity_diff = (old[negative_draws].astype(int)
                        - new[negative_draws].astype(int)).mean(axis=1)
    return {'original_source': str(source.relative_to(ROOT)),
            'original_source_sha256': digest(source),
            'original_sensitivity': float(old[labels].mean()),
            'original_specificity': float((~old[~labels]).mean()),
            'original_auc': float(roc_auc_score(labels, original.referable_probability)),
            'adapted_sensitivity': float(new[labels].mean()),
            'adapted_specificity': float((~new[~labels]).mean()),
            'adapted_auc': float(roc_auc_score(labels, adapted_scores)),
            'sensitivity_difference_paired_bootstrap_95':
                np.quantile(sensitivity_diff, [0.025, 0.975]).tolist(),
            'specificity_difference_paired_bootstrap_95':
                np.quantile(specificity_diff, [0.025, 0.975]).tolist(),
            'bootstrap_replicates': 5000}


def simple_score_recalibration_attempt(test_ids, test_grades):
    train_path = ROOT/'results/verification_2026-09-24/idrid_train_grading/per_image_results.csv'
    test_path = ROOT/'results/verification_2026-09-18/idrid_external_grading/per_image_results.csv'
    training = pd.read_csv(train_path)
    testing = pd.read_csv(test_path).set_index('id_code').loc[test_ids]
    if not np.array_equal(testing.true_grade.to_numpy(int), test_grades):
        raise ValueError('Simple-threshold IDRiD labels differ from the probe cohort')
    threshold, train_sensitivity, train_specificity = threshold_for_sensitivity(
        (training.true_grade >= 2).to_numpy(int),
        training.continuous_score.to_numpy(float), 0.92)
    return {'train_predictions_source': str(train_path.relative_to(ROOT)),
            'train_predictions_sha256': digest(train_path),
            'test_predictions_source': str(test_path.relative_to(ROOT)),
            'test_predictions_sha256': digest(test_path),
            'selection_rule': 'minimum training false-positive rate at >=0.92 sensitivity',
            'selected_threshold': threshold,
            'training_sensitivity': train_sensitivity,
            'training_specificity': train_specificity,
            'official_test': performance(
                test_grades, testing.continuous_score.to_numpy(float), threshold),
            'decision': 'not_integrated_worse_test_specificity'}


def run(args):
    if (args.output/'referral_probe.joblib').exists():
        raise FileExistsError('Completed probe evidence exists; use a new output directory')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    model, _, _ = _get_model()
    train_label_path = GRADING/'2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'
    test_label_path = GRADING/'2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
    train_image_dir = GRADING/'1. Original Images/a. Training Set'
    test_image_dir = GRADING/'1. Original Images/b. Testing Set'
    cache = args.output/'cache'
    train_ids, train_grades, train_backbone, train_head = extract(
        'train', train_label_path, train_image_dir, cache, model, args.batch_size)
    test_ids, test_grades, test_backbone, test_head = extract(
        'test', test_label_path, test_image_dir, cache, model, args.batch_size)
    y_train = (train_grades >= 2).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    trials = []
    representations = {'head_256': (train_head, test_head),
                       'backbone': (train_backbone, test_backbone)}
    for name, (features, _) in representations.items():
        for regularization in (0.0001, 0.001, 0.01, 0.1, 1.0):
            estimator = make_pipeline(
                StandardScaler(), LogisticRegression(
                    C=regularization, max_iter=2000, solver='lbfgs'))
            oof = cross_val_predict(estimator, features, y_train, cv=cv,
                                    method='predict_proba')[:, 1]
            threshold, sensitivity, specificity = threshold_for_sensitivity(
                y_train, oof, 0.92)
            row = {'representation': name, 'C': regularization,
                   'oof_auc': float(roc_auc_score(y_train, oof)),
                   'oof_threshold': threshold,
                   'oof_sensitivity': sensitivity, 'oof_specificity': specificity}
            trials.append(row)
            print(row, flush=True)
    # Selection is entirely on OOF official-training predictions.
    selected = max(trials, key=lambda row: (row['oof_auc'], row['oof_specificity']))
    train_features, test_features = representations[selected['representation']]
    final = make_pipeline(StandardScaler(), LogisticRegression(
        C=selected['C'], max_iter=2000, solver='lbfgs'))
    final.fit(train_features, y_train)
    test_scores = final.predict_proba(test_features)[:, 1]
    test_metrics = performance(test_grades, test_scores, selected['oof_threshold'])
    original_comparison = compare_original(
        test_ids, test_grades, test_scores, selected['oof_threshold'])
    simple_attempt = simple_score_recalibration_attempt(test_ids, test_grades)
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output/'referral_probe.joblib'
    joblib.dump(final, model_path)
    pd.DataFrame(trials).to_csv(args.output/'training_oof_trials.csv', index=False)
    pd.DataFrame({'image_id': test_ids, 'true_grade': test_grades,
                  'adapted_referral_probability': test_scores,
                  'adapted_referral_prediction': test_scores >= selected['oof_threshold']}).to_csv(
                      args.output/'test_predictions.csv', index=False)
    result = {'schema_version': 1,
              'status': 'retrospective_target_domain_probe_not_deployed',
              'training_images': len(train_ids), 'testing_images': len(test_ids),
              'training_labels_sha256': digest(train_label_path),
              'testing_labels_sha256': digest(test_label_path),
              'checkpoint_sha256': digest(ROOT/'python/weights/netrai_fold0_best.pth'),
              'probe_sha256': digest(model_path),
              'selection': selected, 'official_test': test_metrics,
              'comparison_with_original': original_comparison,
              'simple_score_recalibration_attempt': simple_attempt,
              'limitations': ['IDRiD test labels were previously inspected in this project.',
                              'Training images may not have verifiable patient IDs.',
                              'This adapts a binary referral head, not five-grade classification.',
                              'No MATLAB export or prospective validation.']}
    (args.output/'summary.json').write_text(
        json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'selection': selected, 'official_test': test_metrics}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/verification_2026-09-24/idrid_feature_probe')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    run(parser.parse_args())
