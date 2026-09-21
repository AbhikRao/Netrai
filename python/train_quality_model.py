#!/usr/bin/env python3
"""Train a CPU-friendly image-level EyeQ candidate with patient-grouped selection.

Frozen pretrained image embeddings make head fitting inexpensive. Settings and
the Reject threshold are selected only within official training patients. The
official test split is evaluated after selection and never used for fitting.
All models remain research candidates until separate integration review.
"""
import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path
from statistics import NormalDist

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from calibrate_quality import file_sha256, resolve_eyeq_image
from compare_quality_baselines import metrics
from models.quality_model import (
    CLASS_NAMES, PREPROCESS_VERSION, QualityModel, quality_decisions, quality_input,
)

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def patient_id(filename):
    match = re.fullmatch(r'(\d+)_(left|right)\.(jpeg|jpg|png)', str(filename))
    if match is None:
        raise ValueError(f'Cannot establish patient identity from EyeQ name: {filename}')
    return match.group(1)


def read_labels(path):
    frame = pd.read_csv(path)
    if not {'image', 'quality'}.issubset(frame):
        raise ValueError('EyeQ labels require image and quality columns')
    if frame.empty or frame.image.isna().any() or frame.image.duplicated().any():
        raise ValueError('EyeQ labels are empty or contain missing/duplicate images')
    if not frame.quality.isin([0, 1, 2]).all():
        raise ValueError('EyeQ quality labels must be integers 0,1,2')
    frame['patient_id'] = frame.image.map(patient_id)
    return frame


def grouped_split(train, test, seed):
    overlap = set(train.patient_id) & set(test.patient_id)
    if overlap or set(train.image) & set(test.image):
        raise ValueError('Official EyeQ train/test patient or image overlap detected')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.full(len(train), -1, dtype=int)
    for fold, (_, held_out) in enumerate(
            splitter.split(train, train.quality, train.patient_id)):
        folds[held_out] = fold
    selection = np.flatnonzero(folds == 0)
    calibration = np.flatnonzero(folds == 1)
    fit = np.flatnonzero(folds >= 2)
    partitions = (fit, selection, calibration)
    patient_sets = [set(train.iloc[index].patient_id) for index in partitions]
    if any(patient_sets[left] & patient_sets[right]
           for left in range(3) for right in range(left+1, 3)):
        raise ValueError('Internal fit/selection/calibration patient leakage detected')
    for index in partitions:
        if set(train.iloc[index].quality) != {0, 1, 2}:
            raise ValueError('All three quality classes must occur in each split')
    return fit, selection, calibration


class ImageRows(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = cv2.imread(str(self.paths[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot read EyeQ image: {self.paths[index]}')
        return quality_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def initialize_worker(_):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def embeddings(frame, image_dir, model, cache_path, batch_size, workers):
    paths = [resolve_eyeq_image(image_dir, name) for name in frame.image]
    if any(path is None for path in paths):
        raise ValueError('Missing EyeQ images; refusing partial extraction')
    # Bind cache to order, labels, source files, preprocessing, and encoder weights.
    digest = hashlib.sha256(PREPROCESS_VERSION.encode())
    for name, value in model.features.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    for row, path in zip(frame.itertuples(), paths):
        stat = path.stat()
        digest.update(f'{row.image}|{row.quality}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}'.encode())
    signature = digest.hexdigest()
    features = np.empty((0, 576), np.float32)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if str(cache['signature']) != signature:
                raise ValueError(f'Stale embedding cache: {cache_path}')
            features = cache['features'].copy()
        if (features.ndim != 2 or features.shape[1] != 576
                or len(features) > len(frame) or not np.isfinite(features).all()):
            raise ValueError('Invalid embedding cache')
    completed = len(features)
    if completed == len(frame):
        print(f'{image_dir.name}: reused {completed} verified embeddings', flush=True)
        return features
    loader = DataLoader(ImageRows(paths[completed:]), batch_size=batch_size,
                        num_workers=workers, shuffle=False,
                        worker_init_fn=initialize_worker)
    chunks = [features]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            chunks.append(model.encode(batch).numpy())
            completed += len(batch)
            if batch_number % 16 == 0 or completed == len(frame):
                features = np.concatenate(chunks)
                temporary = cache_path.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.savez(stream, signature=signature, features=features)
                temporary.replace(cache_path)
                chunks = [features]
                print(f'{image_dir.name}: {completed}/{len(frame)} embeddings', flush=True)
    return features


def select_regularization(features, labels, fit, selection, seed):
    scaler = StandardScaler().fit(features[fit])
    x_fit = scaler.transform(features[fit])
    x_selection = scaler.transform(features[selection])
    trials = []
    best = None
    for regularization in (.001, .01, .1, 1.0):
        classifier = LogisticRegression(C=regularization, class_weight='balanced',
                                        max_iter=1500, random_state=seed)
        classifier.fit(x_fit, labels[fit])
        predictions = classifier.predict(x_selection)
        summary, _ = metrics(labels[selection], predictions)
        score = (summary['macro_f1'], summary['per_class_recall']['reject'])
        trial = {'C': regularization, 'metrics': summary}
        trials.append(trial)
        if best is None or score > best[0]:
            best = (score, trial)
    return best[1], trials


def fit_temperature(logits, labels):
    """Fit scalar temperature on the patient-separated calibration partition."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)
    result = minimize_scalar(
        lambda temperature: log_loss(labels, softmax(logits / temperature, axis=1),
                                     labels=[0, 1, 2]),
        bounds=(.25, 5.0), method='bounded', options={'xatol': 1e-6})
    if not result.success or not np.isfinite(result.x):
        raise ValueError('Quality temperature calibration failed')
    return float(result.x)


def wilson_lower_bound(successes, total, confidence=.95):
    """Return a one-sided Wilson lower confidence bound for a binomial rate."""
    if not 0 <= successes <= total or total < 1 or not .5 < confidence < 1:
        raise ValueError('Invalid binomial confidence-bound inputs')
    z = NormalDist().inv_cdf(confidence)
    rate = successes / total
    denominator = 1 + z*z/total
    centre = rate + z*z/(2*total)
    margin = z * np.sqrt(rate*(1-rate)/total + z*z/(4*total*total))
    return float((centre - margin) / denominator)


def select_reject_threshold(probabilities, labels):
    """Choose a conservative Reject threshold on calibration patients only.

    Requiring the one-sided 95% Wilson lower bound, rather than only the point
    estimate, to exceed 0.80 adds a pre-deployment safety margin for sampling
    uncertainty. Macro-F1 breaks ties among thresholds satisfying that rule.
    """
    best, trials = None, []
    labels = np.asarray(labels, dtype=int)
    reject_total = int(np.sum(labels == 2))
    for threshold in np.linspace(.15, .85, 29):
        predictions = quality_decisions(probabilities, float(threshold))
        summary, _ = metrics(labels, predictions)
        reject_hits = int(np.sum((labels == 2) & (predictions == 2)))
        lower_bound = wilson_lower_bound(reject_hits, reject_total)
        score = (lower_bound >= .8,
                 summary['macro_f1'], summary['per_class_recall']['reject'])
        trial = {
            'reject_threshold': float(threshold),
            'reject_recall_one_sided_95_wilson_lower': lower_bound,
            'metrics': summary,
        }
        trials.append(trial)
        if best is None or score > best[0]:
            best = (score, trial)
    return best[1], trials


def patient_bootstrap(labels, predictions, patients, seed, replicates=1000):
    """Patient-cluster bootstrap intervals for the two M1 engineering endpoints."""
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)
    patients = np.asarray(patients, dtype=str)
    unique = np.unique(patients)
    if len(labels) != len(predictions) or len(labels) != len(patients) or not len(unique):
        raise ValueError('Bootstrap inputs must be aligned and nonempty')
    indices = {patient: np.flatnonzero(patients == patient) for patient in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[patient] for patient in sampled])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            summary, _ = metrics(labels[selected], predictions[selected])
        values.append((summary['macro_f1'], summary['per_class_recall']['reject']))
    values = np.asarray(values)
    return {
        'unit': 'filename patient identifier', 'replicates': replicates,
        'macro_f1_95_ci': np.percentile(values[:, 0], [2.5, 97.5]).tolist(),
        'reject_recall_95_ci': np.percentile(values[:, 1], [2.5, 97.5]).tolist(),
    }


def retention_metrics(frame, predictions):
    """Expose false-recapture and DR-severity selection effects."""
    predictions = np.asarray(predictions, dtype=int)
    labels = frame.quality.to_numpy(int)
    accepted = predictions != 2
    nonreject = labels != 2
    result = {
        'false_reject_rate_among_good_or_usable': float(np.mean(~accepted[nonreject])),
        'acceptance_rate': float(np.mean(accepted)),
        'acceptance_rate_by_true_quality': {
            CLASS_NAMES[value]: float(np.mean(accepted[labels == value]))
            for value in range(3)
        },
    }
    if 'DR_grade' in frame:
        result['acceptance_rate_by_dr_grade'] = {
            str(value): float(np.mean(accepted[frame.DR_grade.to_numpy(int) == value]))
            for value in sorted(frame.DR_grade.dropna().astype(int).unique())
        }
    return result


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train = read_labels(args.eyeq / 'data/Label_EyeQ_train.csv')
    test = read_labels(args.eyeq / 'data/Label_EyeQ_test.csv')
    fit, selection, calibration = grouped_split(train, test, args.seed)
    manifest = train[['image', 'patient_id', 'quality']].copy()
    manifest['partition'] = 'fit'
    manifest.loc[selection, 'partition'] = 'selection'
    manifest.loc[calibration, 'partition'] = 'calibration'
    manifest.to_csv(output / 'training_split.csv', index=False)
    split_audit = {
        'patient_identifier': 'numeric EyePACS filename prefix; both eyes grouped',
        'official_train_images': len(train), 'official_test_images': len(test),
        'official_train_patients': int(train.patient_id.nunique()),
        'official_test_patients': int(test.patient_id.nunique()),
        'fit_images': len(fit), 'selection_images': len(selection),
        'calibration_images': len(calibration),
        'fit_patients': int(train.iloc[fit].patient_id.nunique()),
        'selection_patients': int(train.iloc[selection].patient_id.nunique()),
        'calibration_patients': int(train.iloc[calibration].patient_id.nunique()),
        'train_test_patient_overlap': 0,
        'internal_partition_patient_overlap': 0,
        'seed': args.seed,
        'train_labels_sha256': file_sha256(args.eyeq / 'data/Label_EyeQ_train.csv'),
        'test_labels_sha256': file_sha256(args.eyeq / 'data/Label_EyeQ_test.csv'),
        'exact_duplicate_audit': 'exact_duplicate_audit.json',
        'limitation': 'Filename identity and exact-file checks; perceptual near-duplicates are not audited.',
    }
    save_json(output / 'split_audit.json', split_audit)
    model = QualityModel(pretrained=True).eval()
    train_x = embeddings(train, args.eyeq / 'images/train', model,
                         output / 'cache/train_embeddings.npz', args.batch_size, args.workers)
    labels = train.quality.to_numpy(int)
    selected_model, model_trials = select_regularization(
        train_x, labels, fit, selection, args.seed)
    # Refit on fit+selection; calibration patients remain isolated.
    model_fit = np.concatenate((fit, selection))
    scaler = StandardScaler().fit(train_x[model_fit])
    classifier = LogisticRegression(C=selected_model['C'], class_weight='balanced',
                                    max_iter=1500, random_state=args.seed)
    classifier.fit(scaler.transform(train_x[model_fit]), labels[model_fit])
    calibration_logits = classifier.decision_function(
        scaler.transform(train_x[calibration]))
    temperature = fit_temperature(calibration_logits, labels[calibration])
    calibration_probabilities = softmax(calibration_logits / temperature, axis=1)
    selected_policy, policy_trials = select_reject_threshold(
        calibration_probabilities, labels[calibration])
    save_json(output / 'selection.json', {
        'model_selection_scope': 'patient-separated official-train selection partition',
        'model_selection_rule': 'maximize macro-F1, then Reject recall',
        'selected_model': selected_model, 'model_trials': model_trials,
        'policy_selection_scope': 'separate patient-grouped calibration partition',
        'policy_selection_rule': (
            'Require one-sided 95% Wilson lower bound for Reject recall >=0.80, '
            'then maximize macro-F1 and Reject recall'),
        'temperature': temperature,
        'selected_policy': selected_policy, 'policy_trials': policy_trials,
    })
    print('Selected model:', json.dumps(selected_model), flush=True)
    print('Calibrated policy:', json.dumps(selected_policy), flush=True)
    with torch.no_grad():
        coefficients = classifier.coef_ / scaler.scale_[None, :]
        intercept = classifier.intercept_ - coefficients @ scaler.mean_
        model.head.weight.copy_(torch.from_numpy(coefficients.astype(np.float32)))
        model.head.bias.copy_(torch.from_numpy(intercept.astype(np.float32)))
    # Verify folding the standardization into the exported neural head.
    with torch.inference_mode():
        logits = model.head(torch.from_numpy(train_x[:128])).numpy()
    error = float(np.max(np.abs(softmax(logits, axis=1)
                               - classifier.predict_proba(scaler.transform(train_x[:128])))))
    if error > 1e-5:
        raise ValueError(f'Standardized head conversion failed: {error}')
    checkpoint = output / 'quality_mobilenet_v3_small.pth'
    torch.save(model.state_dict(), checkpoint)
    config = {
        'schema_version': 1, 'architecture': 'MobileNetV3-Small frozen ImageNet encoder + EyeQ linear head',
        'pretrained_source': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
        'training_strategy': 'patient-grouped fit/selection/calibration within official train',
        'classes': CLASS_NAMES, 'input_size': 224,
        'imagenet_mean': [.485, .456, .406],
        'imagenet_std': [.229, .224, .225],
        'preprocessing': PREPROCESS_VERSION,
        'confidence_temperature': temperature,
        'reject_threshold': selected_policy['reject_threshold'],
        'deployment_status': 'research_candidate_not_deployed',
        'checkpoint_sha256': file_sha256(checkpoint), 'head_conversion_max_probability_error': error,
    }
    save_json(output / 'quality_model_config.json', config)
    test_x = embeddings(test, args.eyeq / 'images/test', model,
                        output / 'cache/test_embeddings.npz', args.batch_size, args.workers)
    with torch.inference_mode():
        probabilities = softmax(
            model.head(torch.from_numpy(test_x)).numpy() / temperature, axis=1)
    predictions = quality_decisions(probabilities, selected_policy['reject_threshold'])
    summary, matrix = metrics(test.quality.to_numpy(int), predictions)
    intervals = patient_bootstrap(
        test.quality.to_numpy(int), predictions, test.patient_id, args.seed,
        args.bootstrap_replicates)
    retention = retention_metrics(test, predictions)
    passed = summary['macro_f1'] >= .7 and summary['per_class_recall']['reject'] >= .8
    result = {
        'schema_version': 1, 'status': 'research_candidate_not_deployed',
        'split_audit': split_audit,
        'selected_on_training_selection': selected_model,
        'selected_on_training_calibration': selected_policy,
        'confidence_temperature': temperature,
        'official_test_metrics': summary,
        'patient_bootstrap': intervals,
        'retention_metrics': retention,
        'held_out_gate': {'minimum_macro_f1': .7, 'minimum_reject_recall': .8,
                          'passed': bool(passed), 'purpose': 'Engineering gate; not clinical validation'},
        'head_conversion_max_probability_error': error,
        'checkpoint_sha256': file_sha256(checkpoint),
        'limitations': ['Frozen ImageNet features; no retinal fine-tuning yet.',
                       'Test split has been examined for prior baseline development.',
                       'No portable-camera validation or factor-specific quality labels.',
                       'Cross-dataset content duplicates have not been audited.'],
    }
    save_json(output / 'quality_model_metrics.json', result)
    rows = test[['image', 'patient_id', 'quality']].copy()
    rows['prediction'] = predictions
    for index, name in enumerate(CLASS_NAMES):
        rows[f'probability_{name}'] = probabilities[:, index]
    rows.to_csv(output / 'test_predictions.csv', index=False)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output / 'test_confusion_matrix.csv', index_label='true_class')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eyeq', type=Path, default=ROOT / 'data/eyeq')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/verification_2026-09-21/eyeq_learned_quality')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap-replicates', type=int, default=1000)
    run(parser.parse_args())
