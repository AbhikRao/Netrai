#!/usr/bin/env python3
"""Build an auditable, machine-readable passport for a NetrAI model release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading large models in RAM."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Return a project-relative POSIX path for portable evidence records."""
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def artifact_record(path: Path) -> dict:
    """Describe and fingerprint one required release artifact."""
    if not path.is_file():
        raise FileNotFoundError(f'Required artifact is missing: {path}')
    return {
        'path': relative(path),
        'sha256': sha256_file(path),
        'bytes': path.stat().st_size,
    }


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Required evidence is missing: {path}')
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def csv_data_rows(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f'Required dataset index is missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return sum(1 for _ in csv.DictReader(handle))


def load_csv_records(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f'Required evidence is missing: {path}')
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def build_passport(validation_path: Path, gradcam_path: Path,
                   runtime_comparison_path: Path, matlab_validation_path: Path,
                   quality_validation_path: Path,
                   quality_robustness_path: Path,
                   quality_ablation_path: Path,
                   learned_quality_path: Path,
                   learned_quality_robustness_path: Path,
                   learned_quality_duplicate_path: Path,
                   learned_quality_matlab_parity_path: Path) -> dict:
    """Assemble release artifacts, evidence, scope, and limitations."""
    checkpoint_path = PROJECT_ROOT / 'python/weights/netrai_fold0_best.pth'
    thresholds_path = PROJECT_ROOT / 'python/weights/fold0_thresholds.npy'
    calibration_path = PROJECT_ROOT / 'python/weights/calibration.json'
    safety_path = PROJECT_ROOT / 'python/weights/safety_policy.json'
    quality_thresholds_path = PROJECT_ROOT / 'python/weights/quality_thresholds_eyeq.json'
    learned_quality_checkpoint_path = learned_quality_path.parent / 'quality_mobilenet_v3_small.pth'
    learned_quality_config_path = learned_quality_path.parent / 'quality_model_config.json'
    learned_quality_onnx_path = PROJECT_ROOT / 'matlab/models/quality_model_candidate.onnx'
    learned_quality_onnx_config_path = PROJECT_ROOT / 'matlab/models/quality_model_candidate.json'
    learned_quality_parity_reference_path = PROJECT_ROOT / 'matlab/models/quality_parity_reference.json'
    onnx_path = PROJECT_ROOT / 'matlab/models/netrai_fold0.onnx'
    matlab_config_path = PROJECT_ROOT / 'matlab/models/netrai_config.json'
    parity_path = PROJECT_ROOT / 'matlab/models/parity_reference.json'
    simulink_model_path = PROJECT_ROOT / 'matlab/models/NetrAI_Telemedicine_Model.slx'
    matlab_environment_path = PROJECT_ROOT / 'matlab/results/environment_2026-09-18.json'
    simulink_sweep_path = PROJECT_ROOT / 'matlab/results/m5_scenario_sweep/simulink_scenario_sweep.json'
    python_m5_path = PROJECT_ROOT / 'results/verification_2026-09-18/m5_python_reference/python_scenario_sweep.json'
    dataset_audit_path = PROJECT_ROOT / 'results/verification_2026-09-18/dataset_audit.json'
    idrid_validation_path = PROJECT_ROOT / 'results/verification_2026-09-18/idrid_external_grading/aggregate_metrics.json'
    idrid_bootstrap_path = PROJECT_ROOT / 'results/verification_2026-09-18/idrid_external_grading/bootstrap_metrics.json'
    idrid_lesion_path = PROJECT_ROOT / 'results/verification_2026-09-18/idrid_lesion_validation/lesion_mask_aggregate.csv'
    idrid_landmark_path = PROJECT_ROOT / 'results/verification_2026-09-18/idrid_landmark_validation/idrid_landmark_metrics.json'
    drive_vessel_path = PROJECT_ROOT / 'results/verification_2026-09-18/drive_vessel_validation/drive_vessel_metrics.json'
    selective_path = PROJECT_ROOT / 'results/verification_2026-09-18/selective_prediction/coverage_risk_curve.json'
    labels_path = PROJECT_ROOT / 'data/aptos2019/train.csv'
    fold_path = PROJECT_ROOT / 'matlab/models/fold0_validation.csv'

    validation = load_json(validation_path)
    gradcam = load_json(gradcam_path)
    calibration = load_json(calibration_path)
    safety = load_json(safety_path)
    matlab_config = load_json(matlab_config_path)
    parity = load_json(parity_path)
    matlab_environment = load_json(matlab_environment_path)
    dataset_audit = load_json(dataset_audit_path)
    idrid_validation = load_json(idrid_validation_path)
    idrid_bootstrap = load_json(idrid_bootstrap_path)
    idrid_landmarks = load_json(idrid_landmark_path)
    drive_vessels = load_json(drive_vessel_path)
    selective = load_json(selective_path)
    simulink_sweep = load_json(simulink_sweep_path)
    python_m5 = load_json(python_m5_path)
    runtime_comparison = load_json(runtime_comparison_path)
    matlab_validation = load_json(matlab_validation_path)
    quality_validation = load_json(quality_validation_path)
    quality_robustness = load_json(quality_robustness_path)
    quality_ablation = load_json(quality_ablation_path)
    learned_quality = load_json(learned_quality_path)
    learned_quality_robustness = load_json(learned_quality_robustness_path)
    learned_quality_duplicates = load_json(learned_quality_duplicate_path)
    learned_quality_matlab_parity = load_json(learned_quality_matlab_parity_path)

    learned_checkpoint = artifact_record(learned_quality_checkpoint_path)
    if learned_quality['checkpoint_sha256'] != learned_checkpoint['sha256']:
        raise ValueError('Learned quality checkpoint does not match its evaluation evidence')
    if not learned_quality['held_out_gate']['passed']:
        raise ValueError('Passport expects the evaluated learned M1 candidate to pass its engineering gate')
    if not learned_quality_matlab_parity.get('passed'):
        raise ValueError('Passport requires passing learned M1 MATLAB parity evidence')

    thresholds = np.load(thresholds_path).astype(float).tolist()
    if len(thresholds) != 4:
        raise ValueError('Expected exactly four ordinal grade thresholds')
    if validation.get('counts', {}).get('evaluated') != 733:
        raise ValueError('Passport requires the complete 733-image Fold-0 evidence')

    return {
        'schema_version': 4,
        'passport_id': 'netrai-efficientnet-b4-fold0-2026-09-21',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'clinical_claim_status': 'research_screening_prototype_not_clinically_validated',
        'intended_use': (
            'Research prototype for diabetic-retinopathy screening triage from '
            'color fundus photographs with mandatory clinician oversight.'
        ),
        'prohibited_use': [
            'standalone diagnosis',
            'treatment selection',
            'autonomous clinical deployment',
            'claims of external or prospective validation',
        ],
        'model': {
            'architecture': matlab_config['architecture'],
            'training_dataset': 'APTOS 2019 Blindness Detection',
            'training_workflow': 'netrai_kaggle_train.ipynb',
            'fold': matlab_config['fold'],
            'input_size': matlab_config['input_size'],
            'output_schema': matlab_config['output_schema'],
            'ordinal_grade_thresholds': thresholds,
            'referable_head_threshold': safety['referable_head_threshold'],
            'artifacts': {
                'pytorch_checkpoint': artifact_record(checkpoint_path),
                'ordinal_thresholds': artifact_record(thresholds_path),
                'confidence_calibration': artifact_record(calibration_path),
                'safety_policy': artifact_record(safety_path),
                'matlab_onnx': artifact_record(onnx_path),
                'matlab_config': artifact_record(matlab_config_path),
                'onnx_parity_reference': artifact_record(parity_path),
                'simulink_digital_twin': artifact_record(simulink_model_path),
            },
        },
        'data_provenance': {
            'label_index': {
                **artifact_record(labels_path),
                'rows': csv_data_rows(labels_path),
            },
            'fold0_selection': {
                **artifact_record(fold_path),
                'rows': csv_data_rows(fold_path),
            },
            'validation_scope': (
                'Internal held-out Fold 0. Checkpoint selection and ordinal '
                'threshold optimization used this fold; it is not external validation.'
            ),
            'dataset_audit': {
                'source': artifact_record(dataset_audit_path),
                'datasets': dataset_audit['datasets'],
            },
        },
        'validation_evidence': {
            'source': artifact_record(validation_path),
            'counts': validation['counts'],
            'grade_and_referable_metrics': validation['metrics'],
            'dual_head_safety': validation['dual_head_safety'],
            'configuration': validation['configuration'],
        },
        'confidence_calibration': {
            'temperature': calibration['temperature'],
            'evaluation_split_deployed_ordinal_policy': calibration['metrics'][
                'evaluation_split_deployed_ordinal_policy'
            ],
            'limitation': calibration['limitation'],
        },
        'm1_quality_validation': {
            'source': artifact_record(quality_validation_path),
            'threshold_artifact': artifact_record(quality_thresholds_path),
            'deployment_status': 'not_deployed_failed_held_out_quality_gate',
            'status': quality_validation['status'],
            'held_out_gate': quality_validation['held_out_gate'],
            'label_mapping': quality_validation['label_mapping'],
            'reject_threshold': quality_validation['reject_threshold'],
            'good_threshold': quality_validation['good_threshold'],
            'train_metrics': quality_validation['train_metrics'],
            'held_out_test_metrics': quality_validation['test_metrics'],
            'factor_diagnostics': quality_validation['factor_diagnostics'],
            'synthetic_robustness_source': artifact_record(quality_robustness_path),
            'synthetic_robustness': {
                'status': quality_robustness['status'],
                'sample': quality_robustness['sample'],
                'corruptions': quality_robustness['corruptions'],
                'limitations': quality_robustness['limitations'],
            },
            'feature_baseline_ablation_source': artifact_record(quality_ablation_path),
            'feature_baseline_ablation': {
                'status': quality_ablation['status'],
                'ranking_by_macro_f1': quality_ablation['ranking_by_macro_f1'],
                'results': quality_ablation['results'],
                'deployment_decision': quality_ablation['deployment_decision'],
            },
            'limitation': quality_validation['limitation'],
        },
        'm1_learned_quality_candidate': {
            'deployment_status': (
                'research_candidate_not_default; global EyeQ gate passed, but '
                'factor-specific and portable-camera gates remain open'),
            'evaluation_source': artifact_record(learned_quality_path),
            'evaluation': learned_quality,
            'exact_duplicate_audit_source': artifact_record(learned_quality_duplicate_path),
            'exact_duplicate_audit': learned_quality_duplicates,
            'synthetic_robustness_source': artifact_record(
                learned_quality_robustness_path),
            'synthetic_robustness': learned_quality_robustness,
            'artifacts': {
                'pytorch_checkpoint': learned_checkpoint,
                'pytorch_config': artifact_record(learned_quality_config_path),
                'matlab_onnx_candidate': artifact_record(learned_quality_onnx_path),
                'matlab_onnx_config': artifact_record(learned_quality_onnx_config_path),
                'parity_reference': artifact_record(learned_quality_parity_reference_path),
                'matlab_parity_evidence': artifact_record(
                    learned_quality_matlab_parity_path),
            },
            'matlab_parity': learned_quality_matlab_parity,
            'decision': (
                'Do not replace the default research routing policy yet. The model '
                'passes the global EyeQ engineering gate and cross-runtime parity, '
                'but rejects only 0.4% of synthetic defocus, 1.6% of FOV shifts, and '
                '0% of JPEG-Q25 cases in the fixed stress audit.'),
        },
        'explainability_qc': {
            'source': artifact_record(gradcam_path),
            **gradcam,
        },
        'external_validation': {
            'dataset': 'IDRiD official testing split (103 images)',
            'source': artifact_record(idrid_validation_path),
            'bootstrap_source': artifact_record(idrid_bootstrap_path),
            'metrics': idrid_validation['metrics'],
            'bootstrap': idrid_bootstrap,
            'target_status': 'failed_referable_sensitivity_and_specificity_targets',
        },
        'm2_validation': {
            'idrid_lesions': {
                'source': artifact_record(idrid_lesion_path),
                'aggregate_rows': load_csv_records(idrid_lesion_path),
            },
            'idrid_landmarks': {
                'source': artifact_record(idrid_landmark_path),
                'summary': idrid_landmarks['summary'],
            },
            'drive_vessels': {
                'source': artifact_record(drive_vessel_path),
                'summary': drive_vessels,
            },
        },
        'selective_prediction': {
            'source': artifact_record(selective_path),
            'policy': selective['policy'],
            'curve': selective['curve'],
            'limitations': selective['limitations'],
        },
        'interoperability': {
            'onnx_output_schema': matlab_config['output_schema'],
            'onnx_runtime_parity': parity.get('summary', parity),
            'matlab_environment_source': artifact_record(matlab_environment_path),
            'matlab_runtime_status': matlab_environment['status'],
            'matlab_version': matlab_environment['matlab_version'],
            'simulink_license': matlab_environment['simulink_license'],
            'onnx_converter_ready': matlab_environment['onnx_converter_ready'],
            'matlab_fold0_source': artifact_record(matlab_validation_path),
            'matlab_fold0_summary': matlab_validation,
            'cross_runtime_source': artifact_record(runtime_comparison_path),
            'cross_runtime_gates': runtime_comparison['gates'],
            'cross_runtime_exact_comparisons': runtime_comparison['exact_comparisons'],
            'cross_runtime_numerical_comparisons': runtime_comparison['numerical_comparisons'],
        },
        'district_simulation': {
            'simulink_source': artifact_record(simulink_sweep_path),
            'python_reference_source': artifact_record(python_m5_path),
            'scenario_count': simulink_sweep['scenario_count'],
            'cross_check': python_m5.get('simulink_cross_check'),
            'limitations': simulink_sweep['limitations'] + python_m5['limitations'],
        },
        'decision_policy': safety,
        'known_limitations': [
            'No independent external or prospective clinical validation.',
            'The handcrafted M1 quality policy fails the held-out EyeQ gate; its fitted thresholds are evidence-only and not deployed.',
            'The learned M1 candidate passes the global EyeQ engineering gate but remains non-default because synthetic defect coverage, portable-camera validation, and factor-specific feedback validation are incomplete.',
            'Minority-grade recall is materially weaker than the referable-DR endpoint.',
            'External IDRiD sensitivity and specificity fail the problem target.',
            'M2 is now measured on IDRiD/DRIVE; lesion and vessel accuracy remains below clinical quality.',
            'The 25-image Grad-CAM audit flagged possible shortcut attention in 52% of maps.',
            'No ophthalmologist usefulness or under-30-second review-time study yet.',
            'MATLAB and Python retain separate image-library preprocessing paths; exact screening, strict five-grade, and numerical parity gates currently fail and are reported separately.',
            'M5 routing rates and continuous annual calendar are planning assumptions, not field measurements.',
            'M5 does not yet contain a timed SimEvents abandonment/escalation branch.',
        ],
        'required_next_gates': [
            'improve the learned M1 candidate on factor-specific blur/FOV/compression failures and validate portable-camera domains before default integration',
            'replace weak M2 lesion/vessel methods and run native-resolution MA FROC',
            'retrain and repeat external IDRiD grading validation',
            'shortcut-robust retraining and repeat Grad-CAM QC',
            'resolve any remaining MATLAB/Python strict grade or probability mismatch',
            'ophthalmologist usefulness and review-time study',
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--validation', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-18/fold0_dual_head/aggregate_metrics.json',
    )
    parser.add_argument(
        '--gradcam', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-18/gradcam_qc_25/gradcam_qc_summary.json',
    )
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-20/validation_passport.json',
    )
    parser.add_argument(
        '--runtime-comparison', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-20/matlab_python_fold0_733_aligned/runtime_comparison.json',
    )
    parser.add_argument(
        '--matlab-validation', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-20/matlab_fold0_733_aligned/aggregate_metrics.json',
    )
    parser.add_argument(
        '--quality-validation', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-20/eyeq_quality_calibration/quality_calibration_metrics.json',
    )
    parser.add_argument(
        '--quality-robustness', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-20/eyeq_quality_robustness/quality_robustness_summary.json',
    )
    parser.add_argument(
        '--quality-ablation', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-20/eyeq_quality_ablation/quality_baseline_ablation.json',
    )
    parser.add_argument(
        '--learned-quality', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-21/eyeq_learned_quality/quality_model_metrics.json',
    )
    parser.add_argument(
        '--learned-quality-robustness', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-21/eyeq_learned_quality/robustness/learned_quality_robustness.json',
    )
    parser.add_argument(
        '--learned-quality-duplicates', type=Path,
        default=PROJECT_ROOT / 'results/verification_2026-09-21/eyeq_learned_quality/exact_duplicate_audit.json',
    )
    parser.add_argument(
        '--learned-quality-matlab-parity', type=Path,
        default=PROJECT_ROOT / 'matlab/results/quality_model_candidate_parity.json',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passport = build_passport(
        args.validation.resolve(), args.gradcam.resolve(),
        args.runtime_comparison.resolve(), args.matlab_validation.resolve(),
        args.quality_validation.resolve(), args.quality_robustness.resolve(),
        args.quality_ablation.resolve(), args.learned_quality.resolve(),
        args.learned_quality_robustness.resolve(),
        args.learned_quality_duplicates.resolve(),
        args.learned_quality_matlab_parity.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(passport, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Validation Passport written to: {args.output.resolve()}')


if __name__ == '__main__':
    main()
