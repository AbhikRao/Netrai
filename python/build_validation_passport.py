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


def build_passport(validation_path: Path, gradcam_path: Path) -> dict:
    """Assemble release artifacts, evidence, scope, and limitations."""
    checkpoint_path = PROJECT_ROOT / 'python/weights/netrai_fold0_best.pth'
    thresholds_path = PROJECT_ROOT / 'python/weights/fold0_thresholds.npy'
    calibration_path = PROJECT_ROOT / 'python/weights/calibration.json'
    safety_path = PROJECT_ROOT / 'python/weights/safety_policy.json'
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

    thresholds = np.load(thresholds_path).astype(float).tolist()
    if len(thresholds) != 4:
        raise ValueError('Expected exactly four ordinal grade thresholds')
    if validation.get('counts', {}).get('evaluated') != 733:
        raise ValueError('Passport requires the complete 733-image Fold-0 evidence')

    return {
        'schema_version': 2,
        'passport_id': 'netrai-efficientnet-b4-fold0-2026-09-18',
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
            'EyeQ labels are present but the 28,792 EyePACS source images are missing; M1 thresholds remain uncalibrated.',
            'Minority-grade recall is materially weaker than the referable-DR endpoint.',
            'External IDRiD sensitivity and specificity fail the problem target.',
            'M2 is now measured on IDRiD/DRIVE; lesion and vessel accuracy remains below clinical quality.',
            'The 25-image Grad-CAM audit flagged possible shortcut attention in 52% of maps.',
            'No ophthalmologist usefulness or under-30-second review-time study yet.',
            'The MATLAB ONNX converter support package is missing, blocking native M3/M4 parity.',
            'M5 routing rates and continuous annual calendar are planning assumptions, not field measurements.',
            'M5 does not yet contain a timed SimEvents abandonment/escalation branch.',
        ],
        'required_next_gates': [
            'EyeQ quality calibration',
            'replace weak M2 lesion/vessel methods and run native-resolution MA FROC',
            'retrain and repeat external IDRiD grading validation',
            'shortcut-robust retraining and repeat Grad-CAM QC',
            'install ONNX converter and run desktop MATLAB parity/holdout',
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
        default=PROJECT_ROOT / 'results/verification_2026-09-18/validation_passport.json',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passport = build_passport(args.validation.resolve(), args.gradcam.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(passport, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(f'Validation Passport written to: {args.output.resolve()}')


if __name__ == '__main__':
    main()
