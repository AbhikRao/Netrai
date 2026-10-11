#!/usr/bin/env python3
"""Batch validation for the NetrAI screening pipeline.

Exports per-image predictions, aggregate diagnostic metrics, a fixed 5x5
confusion matrix, per-grade metrics, and optional segmentation summaries.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

os.environ.setdefault('MPLCONFIGDIR', '/tmp/netrai-matplotlib')

from inference import predict_batch_with_details
from modules.quality import assess_image_quality, enhance_fundus, generate_recapture_feedback
from modules.segmentation import (
    detect_exudates,
    detect_hemorrhages,
    detect_microaneurysms,
    detect_neovascularization,
    localize_fovea,
    localize_optic_disc,
    segment_vessels,
)
from utils.metrics import compute_metrics, quadratic_weighted_kappa, referable_dr_metrics
from utils.preprocessing import preprocess_fundus
from utils.calibration import load_calibration_temperature, temperature_scale_probabilities
from utils.safety import assign_triage
from utils.validation import grades as validate_grades, probabilities as validate_probabilities, image_ids
from utils.artifacts import atomic_json, json_value, sha256_file
from utils.bundle import verify_bundle


PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIR.parent
DEFAULT_CSV = PROJECT_ROOT / 'data' / 'aptos2019' / 'train.csv'
DEFAULT_IMAGE_DIR = PROJECT_ROOT / 'data' / 'aptos2019' / 'train_images'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'results' / 'batch_validation'
DEFAULT_CALIBRATION = PYTHON_DIR / 'weights' / 'calibration.json'
IMAGE_SUFFIXES = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def find_image(image_dir, image_id):
    """Resolve an image id with or without a filename extension."""
    image_dir = Path(image_dir)
    candidate = image_dir / str(image_id)
    if candidate.is_file():
        return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f'{image_id}{suffix}'
        if candidate.is_file():
            return candidate
    return None


def extract_segmentation(image_rgb):
    """Run M2 and return scalar features suitable for a CSV row."""
    proc_img, green_ch, fov_mask = preprocess_fundus(image_rgb, target_size=512)
    od_center, od_radius = localize_optic_disc(proc_img)
    fovea_center = localize_fovea(green_ch, od_center, od_radius)
    vessel_mask = segment_vessels(green_ch, fov_mask)
    ma_mask, ma_count, _ = detect_microaneurysms(
        green_ch, vessel_mask, fov_mask, od_center, od_radius)
    _, hem_count, hem_stats = detect_hemorrhages(
        green_ch, vessel_mask, ma_mask, fov_mask, od_center, od_radius)
    _, hard_count, soft_count, fovea_dist = detect_exudates(
        proc_img, green_ch, fovea_center, od_center, od_radius, fov_mask)
    _, nv_detected = detect_neovascularization(
        green_ch, vessel_mask, od_center, od_radius, fov_mask)
    fov_pixels = max(int(np.sum(fov_mask > 0)), 1)

    return {
        'od_x': int(od_center[0]),
        'od_y': int(od_center[1]),
        'od_radius': int(od_radius),
        'fovea_x': int(fovea_center[0]) if fovea_center is not None else None,
        'fovea_y': int(fovea_center[1]) if fovea_center is not None else None,
        'vessel_density': float(np.sum(vessel_mask > 0) / fov_pixels),
        'ma_count': int(ma_count),
        'hem_count': int(hem_count),
        'hem_dot_count': int(hem_stats['dot_count']),
        'hem_blot_count': int(hem_stats['blot_count']),
        'hard_exudate_count': int(hard_count),
        'soft_exudate_count': int(soft_count),
        'fovea_distance_dd': float(fovea_dist),
        'nv_detected': bool(nv_detected),
    }


def confusion_matrix_fixed(y_true, y_pred, num_classes=5):
    y_true, y_pred = validate_grades(y_true), validate_grades(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError('Truth and predictions must have the same length')
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        if 0 <= int(truth) < num_classes and 0 <= int(prediction) < num_classes:
            matrix[int(truth), int(prediction)] += 1
    return matrix


def calculate_aggregate(records):
    """Calculate coverage and diagnostic metrics from serializable row dicts."""
    total = len(records)
    successful = [row for row in records if row.get('status') == 'success']
    rejected = [row for row in records if row.get('status') == 'rejected']
    failed = [row for row in records if row.get('status') == 'error']
    evaluated = [
        row for row in successful
        if row.get('true_grade') is not None and row.get('predicted_grade') is not None
    ]
    for row in successful:
        for name in ('referable_probability', 'referable_head_probability',
                     'calibrated_referable_probability', 'confidence', 'raw_confidence'):
            if row.get(name) is not None:
                value = float(row[name])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f'{name} must be finite and in [0, 1]')

    aggregate = {
        'schema_version': 2,
        'metric_scope': 'Successful labeled images only; see full_cohort for unresolved cases',
        'counts': {
            'requested': total,
            'successful': len(successful),
            'quality_rejected': len(rejected),
            'errors': len(failed),
            'evaluated': len(evaluated),
        },
        'coverage': len(successful) / total if total else 0.0,
        'metrics': {},
    }
    labeled = [row for row in records if row.get('true_grade') is not None]
    validate_grades([row['true_grade'] for row in labeled], 'true_grade')
    resolved = [row for row in labeled if row.get('status') == 'success'
                and row.get('triage_action') in ('auto_clear', 'refer')]
    for row in resolved:
        validate_grades([row['predicted_grade']], 'predicted_grade')
    aggregate['full_cohort'] = {
        'labeled': len(labeled),
        'algorithmically_resolved': len(resolved),
        'unresolved': len(labeled) - len(resolved),
        'resolution_coverage': len(resolved) / len(labeled) if labeled else None,
        'correct_resolved_referral_decisions': sum(
            (row['true_grade'] >= 2) == (row['predicted_grade'] >= 2) for row in resolved),
        'missed_referable_auto_clear': sum(
            row['true_grade'] >= 2 and row['triage_action'] == 'auto_clear' for row in resolved),
        'interpretation': 'Research routing, not verified clinical screening completion; unresolved is never correct',
    }
    if not evaluated:
        return aggregate, np.zeros((5, 5), dtype=int), []

    y_true = validate_grades([row['true_grade'] for row in evaluated])
    y_pred = validate_grades([row['predicted_grade'] for row in evaluated])
    confusion = confusion_matrix_fixed(y_true, y_pred)
    class_metrics = compute_metrics(confusion)
    referable = referable_dr_metrics(y_true, y_pred)

    metrics = {
        'qwk': float(quadratic_weighted_kappa(y_true, y_pred)),
        'accuracy': float(np.mean(y_true == y_pred)),
        'mean_absolute_error': float(np.mean(np.abs(y_true - y_pred))),
        'within_one_grade_accuracy': float(np.mean(np.abs(y_true - y_pred) <= 1)),
        'macro_f1': float(np.nanmean(class_metrics['f1'])),
        'macro_f1_defined_classes': int(np.isfinite(class_metrics['f1']).sum()),
        'referable_sensitivity': float(referable['sensitivity']),
        'referable_specificity': float(referable['specificity']),
        'referable_precision': float(referable['precision']),
        'referable_accuracy': float(referable['accuracy']),
        'referable_f1': float(referable['f1']),
    }

    referable_true = (y_true >= 2).astype(int)
    if np.unique(referable_true).size == 2:
        referable_scores = np.array([float(row['referable_probability']) for row in evaluated])
        metrics['referable_auc'] = float(roc_auc_score(referable_true, referable_scores))
    else:
        metrics['referable_auc'] = None
    aggregate['metrics'] = metrics

    dual_head_rows = [
        row for row in evaluated
        if row.get('referable_head_probability') is not None
        and row.get('referable_head_prediction') is not None
    ]
    if dual_head_rows:
        dual_true = np.array([
            int(row['true_grade']) >= 2 for row in dual_head_rows], dtype=bool)
        dual_pred = np.array([
            bool(row['referable_head_prediction']) for row in dual_head_rows], dtype=bool)
        dual_scores = np.array([
            float(row['referable_head_probability']) for row in dual_head_rows])
        tp = int(np.sum(dual_true & dual_pred))
        tn = int(np.sum(~dual_true & ~dual_pred))
        fp = int(np.sum(~dual_true & dual_pred))
        fn = int(np.sum(dual_true & ~dual_pred))
        disagreements = np.array([
            bool(row.get('head_disagreement')) for row in dual_head_rows], dtype=bool)
        concordant = ~disagreements
        safety = {
            'referable_head_sensitivity': tp / (tp + fn) if tp + fn else None,
            'referable_head_specificity': tn / (tn + fp) if tn + fp else None,
            'referable_head_accuracy': (tp + tn) / len(dual_head_rows),
            'referable_head_auc': float(roc_auc_score(dual_true, dual_scores))
                if np.unique(dual_true).size == 2 else None,
            'head_disagreement_count': int(disagreements.sum()),
            'head_disagreement_rate': float(disagreements.mean()),
            'concordant_coverage': float(concordant.mean()),
        }
        if concordant.any():
            concordant_truth = dual_true[concordant]
            concordant_grade = np.array([
                int(row['predicted_grade']) >= 2 for row in dual_head_rows
            ], dtype=bool)[concordant]
            ctp = int(np.sum(concordant_truth & concordant_grade))
            ctn = int(np.sum(~concordant_truth & ~concordant_grade))
            cfp = int(np.sum(~concordant_truth & concordant_grade))
            cfn = int(np.sum(concordant_truth & ~concordant_grade))
            safety['concordant_sensitivity'] = (
                ctp / (ctp + cfn) if ctp + cfn else None)
            safety['concordant_specificity'] = (
                ctn / (ctn + cfp) if ctn + cfp else None)
        aggregate['dual_head_safety'] = safety

    per_grade = []
    for grade in range(5):
        per_grade.append({
            'grade': grade,
            'support': int(confusion[grade].sum()),
            'sensitivity_recall': float(class_metrics['sensitivity'][grade]),
            'specificity': float(class_metrics['specificity'][grade]),
            'precision': float(class_metrics['precision'][grade]),
            'f1': float(class_metrics['f1'][grade]),
        })
    return json_value(aggregate), confusion, json_value(per_grade)


def select_rows(frame, args):
    """Apply a reproducible fold or balanced-sample selection."""
    selected = frame.copy()
    if args.fold is not None:
        if args.label_column not in selected:
            raise ValueError('--fold requires a label column')
        if not 0 <= args.fold < args.n_folds:
            raise ValueError('--fold must be between 0 and n-folds - 1')
        splitter = StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_values = np.full(len(selected), -1, dtype=int)
        labels = selected[args.label_column].astype(int).to_numpy()
        for fold_number, (_, validation_indices) in enumerate(splitter.split(selected, labels)):
            fold_values[validation_indices] = fold_number
        selected = selected.loc[fold_values == args.fold]

    if args.sample_per_grade is not None:
        if args.label_column not in selected:
            raise ValueError('--sample-per-grade requires a label column')
        samples = []
        for _, group in selected.groupby(args.label_column, sort=True):
            samples.append(group.sample(
                n=min(args.sample_per_grade, len(group)), random_state=args.seed))
        selected = pd.concat(samples).sort_values(args.label_column)

    if args.limit is not None:
        selected = selected.head(args.limit)
    return selected.reset_index(drop=True)


def flatten_dict(value, prefix=''):
    rows = []
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            rows.extend(flatten_dict(item, name))
        else:
            rows.append({'metric': name, 'value': item})
    return rows


def batch_identity(frame, args, csv_path, image_dir):
    """Bind a resumable run to bytes, selected labels, configuration and code."""
    bundle = verify_bundle(PYTHON_DIR / 'weights' / 'bundle.json')
    cases = []
    for index, row in frame.iterrows():
        path = find_image(image_dir, row[args.id_column])
        label = row.get(args.label_column)
        cases.append({'source_index': int(index), 'id_code': str(row[args.id_column]),
                      'true_grade': int(label) if pd.notna(label) else None,
                      'image_path': str(path) if path else None,
                      'sha256': sha256_file(path) if path else None})
    source_files = [PYTHON_DIR / name for name in ('batch_validate.py', 'inference.py')]
    for directory in ('models', 'modules', 'utils'):
        source_files.extend(sorted((PYTHON_DIR / directory).glob('*.py')))
    identity = {'schema_version': 1, 'input_csv_sha256': sha256_file(csv_path),
                'bundle': bundle,
                'configuration': {key: value for key, value in vars(args).items()
                                  if key not in ('output', 'resume')},
                'cases': cases,
                'source_sha256': {str(path.relative_to(PYTHON_DIR)): sha256_file(path)
                                  for path in source_files},
                'versions': {name: importlib.metadata.version(name) for name in
                             ('torch', 'numpy', 'opencv-python-headless', 'timm')}}
    identity = json_value(identity)
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False).encode()
    return identity, hashlib.sha256(encoded).hexdigest()


def restore_records(path, identity_sha256, cases):
    checkpoint = json.loads(Path(path).read_text())
    if checkpoint.get('identity_sha256') != identity_sha256:
        raise ValueError('Resume identity changed (data, model, code or configuration); use a new output')
    expected = {row['id_code']: row for row in cases}
    restored, seen = [], set()
    for record in checkpoint['records']:
        image_id = record['id_code']
        if image_id not in expected or image_id in seen:
            raise ValueError('Invalid/duplicate checkpoint image ID')
        seen.add(image_id)
        if any(record.get(key) != expected[image_id][key]
               for key in ('source_index', 'true_grade')):
            raise ValueError('Checkpoint labels or row order do not match the manifest')
        if record['status'] in ('success', 'rejected', 'error'):
            restored.append(record)
        elif record['status'] != 'pending':
            raise ValueError('Unknown checkpoint status')
    return restored, float(checkpoint.get('elapsed_seconds', 0))


def run_batch(args):
    csv_path = Path(args.csv).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path, dtype={args.id_column: str})
    if args.id_column not in frame:
        raise ValueError(f'CSV is missing id column: {args.id_column}')
    frame[args.id_column] = image_ids(frame[args.id_column]).to_numpy()
    if args.label_column in frame:
        validate_grades(frame[args.label_column].dropna().to_numpy(), args.label_column)
    frame = select_rows(frame, args)
    has_labels = args.label_column in frame
    calibration_temperature = load_calibration_temperature(DEFAULT_CALIBRATION)

    identity, identity_sha256 = batch_identity(frame, args, csv_path, image_dir)
    progress_path = output_dir / 'progress.json'
    if getattr(args, 'resume', False):
        records, prior_elapsed = restore_records(progress_path, identity_sha256, identity['cases'])
    else:
        if any((output_dir / name).exists() for name in
               ('progress.json', 'run_identity.json', 'per_image_results.csv', 'aggregate_metrics.json')):
            raise FileExistsError('Existing batch evidence: use --resume or a new output directory')
        records, prior_elapsed = [], 0.0
    completed_ids = {row['id_code'] for row in records}
    atomic_json(output_dir / 'run_identity.json', {'sha256': identity_sha256, **identity})
    pending_images = []
    pending_record_indices = []
    started = time.perf_counter()

    def save_progress(complete=False):
        atomic_json(progress_path, {'identity_sha256': identity_sha256, 'records': records,
                    'complete': complete,
                    'elapsed_seconds': prior_elapsed + time.perf_counter() - started})

    save_progress()

    def predict_records(images, record_indices):
        inference_started = time.perf_counter()
        predictions = predict_batch_with_details(
            images, batch_size=args.batch_size)
        grades = validate_grades(predictions['grades'])
        probabilities = validate_probabilities(predictions['grade_probabilities'])
        if any(len(value) != len(images) for value in predictions.values()):
            raise ValueError('Model output batch length does not match inputs')
        continuous_scores = predictions['continuous_scores']
        if not np.isfinite(continuous_scores).all():
            raise ValueError('Nonfinite continuous scores')
        referral = np.asarray(predictions['referable_head_probabilities'], dtype=float)
        if not np.isfinite(referral).all() or np.any((referral < 0) | (referral > 1)):
            raise ValueError('Invalid referral-head probabilities')
        elapsed_each = (time.perf_counter() - inference_started) / len(images)
        if grades is None:
            for record_index in record_indices:
                records[record_index]['status'] = 'error'
                records[record_index]['error'] = 'model weights unavailable'
        else:
            for offset, record_index in enumerate(record_indices):
                grade = int(grades[offset])
                probs = probabilities[offset]
                calibrated_probs = temperature_scale_probabilities(
                    probs, calibration_temperature)
                record = records[record_index]
                record.update({
                    'status': 'success',
                    'predicted_grade': grade,
                    'continuous_score': float(continuous_scores[offset]),
                    'raw_confidence': float(probs[grade]),
                    'confidence': float(calibrated_probs[grade]),
                    'calibration_temperature': calibration_temperature,
                    'referable_prediction': bool(grade >= 2),
                    'referable_probability': float(probs[2:].sum()),
                    'calibrated_referable_probability': float(calibrated_probs[2:].sum()),
                    'referable_head_probability': float(
                        predictions['referable_head_probabilities'][offset]),
                    'referable_head_prediction': bool(
                        predictions['referable_head_predictions'][offset]),
                    'head_disagreement': bool(
                        predictions['head_disagreements'][offset]),
                    'triage_action': str(predictions['triage_actions'][offset]),
                    'model_seconds': elapsed_each,
                })
                for class_index, probability in enumerate(probs):
                    record[f'prob_grade_{class_index}'] = float(probability)
                    record[f'calibrated_prob_grade_{class_index}'] = float(
                        calibrated_probs[class_index])
                if record.get('true_grade') is not None:
                    record['correct'] = bool(record['true_grade'] == grade)
                    record['absolute_error'] = abs(record['true_grade'] - grade)
    def flush_predictions():
        if not pending_images:
            return
        images, indices = pending_images[:], pending_record_indices[:]
        pending_images.clear()
        pending_record_indices.clear()
        try:
            predict_records(images, indices)
        except Exception:
            # Retry one at a time to isolate a bad item or batch-memory failure.
            for image, index in zip(images, indices):
                try:
                    predict_records([image], [index])
                except Exception as exc:
                    records[index].update(status='error', error=f'{type(exc).__name__}: {exc}',
                                          triage_action='unresolved_error')
        save_progress()

    for source_index, row in tqdm(
            frame.iterrows(), total=len(frame), desc='NetrAI batch validation'):
        image_id = str(row[args.id_column])
        if image_id in completed_ids:
            continue
        if source_index % args.batch_size == 0:
            save_progress()
        true_grade = int(row[args.label_column]) if has_labels and pd.notna(row[args.label_column]) else None
        record = {
            'source_index': int(source_index),
            'id_code': image_id,
            'true_grade': true_grade,
            'status': 'pending',
            'error': '',
        }
        records.append(record)
        record_index = len(records) - 1

        image_path = find_image(image_dir, image_id)
        if image_path is None:
            record.update({'status': 'error', 'error': 'image file not found'})
            continue
        record['image_path'] = str(image_path)

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            record.update({'status': 'error', 'error': 'OpenCV could not decode image'})
            continue

        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            analysis_rgb = image_rgb
            iqs, quality = assess_image_quality(image_rgb)
            record.update({
                'iqs': float(iqs),
                'quality_focus': float(quality['focus']),
                'quality_illumination': float(quality['illumination']),
                'quality_fov': float(quality['fov']),
            })
            if iqs < args.quality_threshold:
                record.update({
                    'status': 'rejected',
                    'error': '; '.join(generate_recapture_feedback(quality)),
                    'triage_action': assign_triage(
                        quality_rejected=True, grade_referable=False,
                        referable_head_positive=False),
                })
                continue

            if not args.no_enhancement and iqs < args.enhancement_threshold:
                analysis_rgb = enhance_fundus(image_rgb)
                record['enhanced'] = True
            else:
                record['enhanced'] = False

            if not args.skip_segmentation:
                record.update(extract_segmentation(analysis_rgb))

            # Keep model grading on the original image so it matches training.
            pending_images.append(image_rgb)
            pending_record_indices.append(record_index)
            if len(pending_images) >= args.batch_size:
                flush_predictions()
        except Exception as exc:
            record.update({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})

    flush_predictions()
    records.sort(key=lambda record: record['source_index'])
    elapsed = prior_elapsed + time.perf_counter() - started
    aggregate, confusion, per_grade = calculate_aggregate(records)
    aggregate['run_identity_sha256'] = identity_sha256
    aggregate['runtime'] = {
        'total_seconds': elapsed,
        'images_per_second': len(records) / elapsed if elapsed else 0.0,
    }
    aggregate['configuration'] = {
        'csv': str(csv_path),
        'image_dir': str(image_dir),
        'quality_threshold': args.quality_threshold,
        'enhancement_threshold': args.enhancement_threshold,
        'enhancement_enabled': not args.no_enhancement,
        'segmentation_enabled': not args.skip_segmentation,
        'batch_size': args.batch_size,
        'fold': args.fold,
        'n_folds': args.n_folds,
        'sample_per_grade': args.sample_per_grade,
        'seed': args.seed,
        'confidence_temperature': calibration_temperature,
    }

    per_image_path = output_dir / 'per_image_results.csv'
    summary_path = output_dir / 'metrics_summary.csv'
    json_path = output_dir / 'aggregate_metrics.json'
    confusion_path = output_dir / 'confusion_matrix.csv'
    per_grade_path = output_dir / 'per_grade_metrics.csv'

    result_frame = pd.DataFrame(records)
    result_frame.to_csv(per_image_path, index=False)
    pd.DataFrame(flatten_dict(aggregate)).to_csv(summary_path, index=False)
    pd.DataFrame(
        confusion,
        index=[f'true_{grade}' for grade in range(5)],
        columns=[f'pred_{grade}' for grade in range(5)],
    ).to_csv(confusion_path, index_label='true_grade')
    pd.DataFrame(per_grade).to_csv(per_grade_path, index=False)
    atomic_json(json_path, aggregate)
    save_progress(complete=True)

    if not args.skip_segmentation and has_labels and not result_frame.empty:
        segmentation_columns = [
            'vessel_density', 'ma_count', 'hem_count', 'hard_exudate_count',
            'soft_exudate_count', 'fovea_distance_dd', 'nv_detected',
        ]
        available = [column for column in segmentation_columns if column in result_frame]
        if available:
            lesion_summary = (
                result_frame[result_frame['status'] == 'success']
                .groupby('true_grade')[available]
                .agg(['mean', 'median'])
            )
            lesion_summary.columns = [f'{name}_{stat}' for name, stat in lesion_summary.columns]
            lesion_summary.to_csv(output_dir / 'segmentation_summary_by_grade.csv')

    print(json.dumps(aggregate, indent=2, allow_nan=False))
    print(f'Outputs written to: {output_dir}')
    return aggregate


def build_parser():
    parser = argparse.ArgumentParser(description='Batch validation for NetrAI')
    parser.add_argument('--csv', default=str(DEFAULT_CSV), help='Input CSV containing image ids')
    parser.add_argument('--image-dir', default=str(DEFAULT_IMAGE_DIR), help='Directory containing images')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_DIR), help='Output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume only an identical content/configuration-bound run; retain recorded errors')
    parser.add_argument('--id-column', default='id_code', help='CSV image-id column')
    parser.add_argument('--label-column', default='diagnosis', help='CSV ground-truth column')
    parser.add_argument('--batch-size', type=int, default=8, help='Model inference batch size')
    parser.add_argument('--limit', type=int, help='Process only the first N selected rows')
    parser.add_argument('--sample-per-grade', type=int, help='Balanced random sample size per grade')
    parser.add_argument('--fold', type=int, help='Evaluate one reproducible stratified fold')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds used with --fold')
    parser.add_argument('--seed', type=int, default=42, help='Selection/fold random seed')
    parser.add_argument('--quality-threshold', type=float, default=0.30, help='Reject below this IQS')
    parser.add_argument('--enhancement-threshold', type=float, default=0.70, help='Enhance below this IQS')
    parser.add_argument('--no-enhancement', action='store_true', help='Disable adaptive image enhancement')
    parser.add_argument('--skip-segmentation', action='store_true', help='Run only M1 and model grading')
    return parser


def main():
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be at least 1')
    if args.limit is not None and args.limit < 1:
        raise SystemExit('--limit must be at least 1')
    if args.sample_per_grade is not None and args.sample_per_grade < 1:
        raise SystemExit('--sample-per-grade must be at least 1')
    run_batch(args)


if __name__ == '__main__':
    main()
