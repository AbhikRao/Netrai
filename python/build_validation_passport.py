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

from utils.artifacts import atomic_json


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
                   learned_quality_matlab_parity_path: Path,
                   integrated_ablation_path: Path,
                   ma_froc_512_path: Path,
                   ma_froc_1024_path: Path,
                   ma_native_path: Path,
                   idrid_probe_path: Path,
                   perceptual_overlap_path: Path,
                   geometric_review_path: Path,
                   confirmed_overlap_path: Path,
                   overlap_impact_path: Path) -> dict:
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
    implementation = PROJECT_ROOT / 'results/implementation_2026-09-24'
    simulink_sweep_path = implementation / 'm5_simevents/simulink_scenario_sweep.json'
    python_m5_path = implementation / 'm5_crosscheck/python_scenario_sweep.json'
    legacy_overlap_path = implementation / 'legacy_aptos_overlap.json'
    manifest_protocol_path = implementation / 'manifests/aptos_development.protocol.json'
    development_manifest_path = implementation / 'manifests/aptos_development.csv'
    rerun_froc_path = implementation / 'ma_detector_rerun/ma_froc.csv'
    rerun_protocol_path = implementation / 'ma_detector_rerun/protocol.json'
    rerun_comparison_path = implementation / 'ma_candidate_vs_detector_rerun.json'
    preprocessing_pair_path = implementation / 'idrid_preprocessing_pair/summary.json'
    dataset_audit_path = PROJECT_ROOT / 'results/dataset_audit/dataset_audit.json'
    idrid_validation_path = PROJECT_ROOT / 'results/verification_2026-09-18/idrid_external_grading/aggregate_metrics.json'
    idrid_bootstrap_path = PROJECT_ROOT / 'results/verification_2026-09-18/idrid_external_grading/bootstrap_metrics.json'
    idrid_lesion_path = implementation / 'lesions_native/lesion_mask_aggregate.csv'
    idrid_landmark_path = implementation / 'landmarks_test/idrid_landmark_metrics.json'
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
    if (dataset_audit['datasets']['eyeq']['status'] != 'ready' or
            dataset_audit['datasets']['eyeq']['checks']['source_images'] != 28792):
        raise ValueError('Passport requires the current complete EyeQ dataset audit')
    idrid_validation = load_json(idrid_validation_path)
    idrid_bootstrap = load_json(idrid_bootstrap_path)
    idrid_landmarks = load_json(idrid_landmark_path)
    drive_vessels = load_json(drive_vessel_path)
    selective = load_json(selective_path)
    simulink_sweep = load_json(simulink_sweep_path)
    python_m5 = load_json(python_m5_path)
    legacy_overlap = load_json(legacy_overlap_path)
    manifest_protocol = load_json(manifest_protocol_path)
    rerun_protocol = load_json(rerun_protocol_path)
    rerun_comparison = load_json(rerun_comparison_path)
    preprocessing_pair = load_json(preprocessing_pair_path)
    rerun_curve = load_csv_records(rerun_froc_path)
    native_lesions = load_csv_records(idrid_lesion_path)
    if (legacy_overlap['validation_images_with_identical_training_content'] != 35
            or legacy_overlap['conflicting_grade_images'] != 62
            or manifest_protocol['quarantined_rows'] != 62
            or csv_data_rows(development_manifest_path) != 3600):
        raise ValueError('APTOS duplicate/quarantine evidence changed')
    if (rerun_protocol['proposal_mode'] != 'detector-rerun'
            or rerun_protocol['analysis_size'] != 512
            or not any(row['split'] == 'test' and float(row['threshold']) == 3.0
                       and int(row['images']) == 27 for row in rerun_curve)):
        raise ValueError('Corrected classic MA detector rerun is incomplete')
    if (rerun_comparison['candidate_source_sha256'] != sha256_file(
            PROJECT_ROOT / 'results/verification_2026-09-24/ma_native_candidate/test_per_image_at_locked_threshold.csv')
            or rerun_comparison['classic_sigma_3_default']['baseline_source_sha256']
            != sha256_file(implementation / 'ma_detector_rerun/ma_froc_per_image.csv')):
        raise ValueError('MA paired rerun comparison is no longer bound to its inputs')
    if (len(native_lesions) != 4 or {row['split'] for row in native_lesions} != {'test'}
            or any(int(row['images']) != 27 for row in native_lesions)):
        raise ValueError('Native-mask IDRiD lesion evidence is incomplete')
    if (simulink_sweep['schema_version'] != 2 or
            python_m5['simulink_cross_check']['scenarios_passed'] != 18 or
            not python_m5['simulink_cross_check']['all_passed'] or
            any(row['endpoint_version'] != 2 for row in simulink_sweep['results'])):
        raise ValueError('M5 endpoint-v2 cross-check is incomplete')
    runtime_comparison = load_json(runtime_comparison_path)
    matlab_validation = load_json(matlab_validation_path)
    quality_validation = load_json(quality_validation_path)
    quality_robustness = load_json(quality_robustness_path)
    quality_ablation = load_json(quality_ablation_path)
    learned_quality = load_json(learned_quality_path)
    learned_quality_robustness = load_json(learned_quality_robustness_path)
    learned_quality_duplicates = load_json(learned_quality_duplicate_path)
    learned_quality_matlab_parity = load_json(learned_quality_matlab_parity_path)
    integrated_ablation = load_csv_records(integrated_ablation_path)
    integrated_protocol_path = integrated_ablation_path.parent / 'protocol.json'
    integrated_protocol = load_json(integrated_protocol_path)
    ma_froc_512 = load_csv_records(ma_froc_512_path)
    ma_froc_1024 = load_csv_records(ma_froc_1024_path)
    ma_froc_512_protocol_path = ma_froc_512_path.parent / 'protocol.json'
    ma_froc_1024_protocol_path = ma_froc_1024_path.parent / 'protocol.json'
    ma_froc_512_protocol = load_json(ma_froc_512_protocol_path)
    ma_froc_1024_protocol = load_json(ma_froc_1024_protocol_path)
    ma_native = load_json(ma_native_path)
    ma_native_model_path = ma_native_path.parent / 'ma_ranker.joblib'
    ma_native_development_path = ma_native_path.parent / 'development_froc.csv'
    ma_native_test_path = ma_native_path.parent / 'test_froc.csv'
    ma_native_per_image_path = ma_native_path.parent / 'test_per_image_at_locked_threshold.csv'
    if ma_native['model_sha256'] != sha256_file(ma_native_model_path):
        raise ValueError('MA candidate model differs from its evaluation record')
    if ma_native['official_test_at_locked_threshold']['images'] != 27:
        raise ValueError('MA candidate evidence must cover the complete IDRiD test cohort')
    for comparison in ma_native['paired_baseline_comparisons'].values():
        baseline = PROJECT_ROOT / comparison['baseline_source']
        if comparison['baseline_source_sha256'] != sha256_file(baseline):
            raise ValueError('MA candidate comparison baseline has changed')
    idrid_probe = load_json(idrid_probe_path)
    idrid_probe_model_path = idrid_probe_path.parent / 'referral_probe.joblib'
    idrid_probe_trials_path = idrid_probe_path.parent / 'training_oof_trials.csv'
    idrid_probe_predictions_path = idrid_probe_path.parent / 'test_predictions.csv'
    if (idrid_probe['probe_sha256'] != sha256_file(idrid_probe_model_path)
            or idrid_probe['checkpoint_sha256'] != sha256_file(checkpoint_path)
            or idrid_probe['training_images'] != 413
            or idrid_probe['testing_images'] != 103):
        raise ValueError('IDRiD referral probe artifacts or cohort changed')
    comparison = idrid_probe['comparison_with_original']
    if comparison['original_source_sha256'] != sha256_file(
            PROJECT_ROOT / comparison['original_source']):
        raise ValueError('IDRiD referral probe baseline changed')
    simple_attempt = idrid_probe['simple_score_recalibration_attempt']
    if (simple_attempt['train_predictions_sha256'] != sha256_file(
            PROJECT_ROOT / simple_attempt['train_predictions_source'])
            or simple_attempt['test_predictions_sha256'] != sha256_file(
                PROJECT_ROOT / simple_attempt['test_predictions_source'])):
        raise ValueError('IDRiD simple score comparison evidence changed')
    perceptual_overlap = load_json(perceptual_overlap_path)
    geometric_review = load_json(geometric_review_path)
    geometric_review_csv_path = geometric_review_path.parent / 'geometric_candidate_review.csv'
    overlap_impact = load_json(overlap_impact_path)
    confirmed_pairs = load_csv_records(confirmed_overlap_path)
    if (len(confirmed_pairs) != overlap_impact['confirmed_cross_split_same_retina_pairs']
            or overlap_impact['sources_sha256']['confirmed_pairs'] != sha256_file(confirmed_overlap_path)):
        raise ValueError('Confirmed overlap impact is not bound to reviewed pairs')
    perceptual_candidates_path = perceptual_overlap_path.parent / 'perceptual_overlap_candidates.csv'
    if geometric_review['source_candidate_csv_sha256'] != sha256_file(perceptual_candidates_path):
        raise ValueError('Geometric review is not bound to the current pHash candidates')
    if len(integrated_ablation) != 5 or not ma_froc_512 or not ma_froc_1024:
        raise ValueError('Passport requires complete integrated and MA FROC evidence')
    if ma_froc_512_protocol['analysis_size'] != 512 or ma_froc_1024_protocol['analysis_size'] != 1024:
        raise ValueError('MA FROC analysis-size protocol does not match passport slots')
    if integrated_protocol['status'] != 'post_hoc_development_ablation_not_independent_validation':
        raise ValueError('Unexpected integrated ablation status')

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
        'schema_version': 7,
        'passport_id': 'netrai-efficientnet-b4-fold0-2026-09-25',
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
            'training_workflow': 'research/legacy_training.ipynb',
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
                'Internal development Fold 0. Checkpoint and thresholds used this fold; '
                '35 validation images have exact training duplicates. Not independent validation.'
            ),
            'legacy_exact_content_overlap': {
                'source': artifact_record(legacy_overlap_path), 'summary': legacy_overlap},
            'corrected_development_manifest': {
                'source': artifact_record(development_manifest_path),
                'protocol_source': artifact_record(manifest_protocol_path),
                'protocol': manifest_protocol,
                'patient_independence': 'unknown'},
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
        'integrated_ablation': {
            'source': artifact_record(integrated_ablation_path),
            'protocol_source': artifact_record(integrated_protocol_path),
            'protocol': integrated_protocol,
            'cohort': 'APTOS Fold-0 development cohort, 733 images; 35 exact-content training overlaps',
            'variants': integrated_ablation,
            'interpretation': (
                'Recapture and human review are unresolved outcomes, not correct decisions. '
                'APTOS lacks quality labels and the learned M1 candidate remains non-default.'),
        },
        'perceptual_overlap_audit': {
            'source': artifact_record(perceptual_overlap_path),
            'candidate_source': artifact_record(perceptual_candidates_path),
            'summary': perceptual_overlap,
            'geometric_review_source': artifact_record(geometric_review_path),
            'geometric_review_rows_source': artifact_record(geometric_review_csv_path),
            'geometric_review': geometric_review,
            'visually_confirmed_pairs_source': artifact_record(confirmed_overlap_path),
            'overlap_impact_source': artifact_record(overlap_impact_path),
            'overlap_impact': overlap_impact,
            'interpretation': (
                'Five same-retina EyeQ official train/test pairs were visually confirmed. '
                'Post-hoc exclusion barely changes metrics but does not restore an independent test.'),
        },
        'explainability_qc': {
            'source': artifact_record(gradcam_path),
            **gradcam,
        },
        'external_validation': {
            'dataset': 'IDRiD official testing split (103 images; development-exposed)',
            'source': artifact_record(idrid_validation_path),
            'bootstrap_source': artifact_record(idrid_bootstrap_path),
            'metrics': idrid_validation['metrics'],
            'bootstrap': idrid_bootstrap,
            'target_status': 'failed_referable_sensitivity_and_specificity_targets',
        },
        'target_domain_referral_probe': {
            'source': artifact_record(idrid_probe_path),
            'model': artifact_record(idrid_probe_model_path),
            'training_source': artifact_record(
                PROJECT_ROOT / 'python/fit_idrid_feature_probe.py'),
            'inference_source': artifact_record(
                PROJECT_ROOT / 'python/modules/idrid_probe.py'),
            'training_oof_trials_source': artifact_record(idrid_probe_trials_path),
            'test_predictions_source': artifact_record(idrid_probe_predictions_path),
            'summary': idrid_probe,
            'deployment_status': 'Python_explicit_opt_in_research_score_only; no triage_effect; MATLAB_not_ported',
        },
        'm2_validation': {
            'trained_ma_candidate': {
                'source': artifact_record(ma_native_path),
                'model': artifact_record(ma_native_model_path),
                'extractor_source': artifact_record(
                    PROJECT_ROOT / 'python/modules/ma_native.py'),
                'training_source': artifact_record(
                    PROJECT_ROOT / 'python/train_ma_native.py'),
                'development_froc_source': artifact_record(ma_native_development_path),
                'test_froc_source': artifact_record(ma_native_test_path),
                'test_per_image_source': artifact_record(ma_native_per_image_path),
                'summary': ma_native,
                'deployment_status': 'Python_explicit_opt_in_only; MATLAB_classic_path_unchanged',
            },
            'ma_froc_512': {
                'source': artifact_record(ma_froc_512_path),
                'protocol_source': artifact_record(ma_froc_512_protocol_path),
                'protocol': ma_froc_512_protocol,
                'curve': ma_froc_512,
                'status': 'historical_fixed_proposals_superseded_for_active_detector_claims',
            },
            'actual_classic_detector_rerun': {
                'source': artifact_record(rerun_froc_path),
                'protocol_source': artifact_record(rerun_protocol_path),
                'protocol': rerun_protocol, 'curve': rerun_curve,
                'paired_candidate_comparison_source': artifact_record(rerun_comparison_path),
                'paired_candidate_comparison': rerun_comparison},
            'ma_froc_1024': {
                'source': artifact_record(ma_froc_1024_path),
                'protocol_source': artifact_record(ma_froc_1024_protocol_path),
                'protocol': ma_froc_1024_protocol,
                'curve': ma_froc_1024,
            },
            'idrid_lesions': {
                'source': artifact_record(idrid_lesion_path),
                'evaluation_space': 'native_annotation_v2',
                'aggregate_rows': native_lesions,
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
            'endpoint_version': 2,
            'cross_check': python_m5.get('simulink_cross_check'),
            'limitations': simulink_sweep['limitations'] + python_m5['limitations'],
        },
        'decision_policy': safety,
        'paired_preprocessing_diagnostic': {
            'source': artifact_record(preprocessing_pair_path),
            'result': preprocessing_pair,
            'decision': 'do_not_promote_linear_resize_on_old_checkpoint'},
        'known_limitations': [
            'No independent external or prospective clinical validation. Original APTOS Fold-0 contains 35 validation images with exact training duplicates.',
            'The handcrafted M1 quality policy fails the held-out EyeQ gate; its fitted thresholds are evidence-only and not deployed.',
            'The learned M1 candidate passes the global EyeQ engineering gate but remains non-default because synthetic defect coverage, portable-camera validation, and factor-specific feedback validation are incomplete.',
            'Minority-grade recall is materially weaker than the referable-DR endpoint.',
            'External IDRiD sensitivity and specificity fail the problem target.',
            'The IDRiD-trained frozen-feature referral probe improves retrospective IDRiD point estimates to 85.94% sensitivity and 87.18% specificity but still fails >90% sensitivity, has no independent new cohort, and is opt-in research output only.',
            'M2 is measured on IDRiD/DRIVE. The new 2048-long-edge trained MA candidate improves the retrospective FROC over the classic 512 method but remains far below clinical lesion sensitivity, is Python opt-in only, and has not been ported to MATLAB.',
            'Five same-retina images cross the official EyeQ train/test split despite distinct filename patient identifiers; M1 test metrics are development benchmarks, not cleanly image-independent validation.',
            'The pHash/ORB search can miss further transformed duplicates; remaining candidates are not exhaustively adjudicated.',
            'The 25-image Grad-CAM audit flagged possible shortcut attention in 52% of maps.',
            'No ophthalmologist usefulness or under-30-second review-time study yet.',
            'MATLAB and Python retain separate image-library preprocessing paths; exact screening, strict five-grade, and numerical parity gates currently fail and are reported separately.',
            'M5 routing rates and continuous annual calendar are planning assumptions, not field measurements.',
            'M5 does not yet contain a timed SimEvents abandonment/escalation branch.',
            'M5 recapture requests remain pending; retries and returns are not simulated.',
        ],
        'required_next_gates': [
            'improve the learned M1 candidate on factor-specific blur/FOV/compression failures and validate portable-camera domains before default integration',
            'obtain a fresh image- and patient-independent M1 validation cohort after the confirmed EyeQ split overlap',
            'improve the trained MA candidate, validate on a fresh lesion cohort, and port any supported operating point to MATLAB before default use',
            'retrain and repeat external IDRiD grading validation',
            'validate the target-domain referral probe on a genuinely new camera/site cohort before any policy integration or MATLAB port',
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
        default=PROJECT_ROOT / 'results/implementation_2026-09-24/validation_passport_v7.json',
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
    parser.add_argument('--integrated-ablation', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/integrated_ablation/integrated_ablation.csv')
    parser.add_argument('--ma-froc-512', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/ma_froc/ma_froc.csv')
    parser.add_argument('--ma-froc-1024', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/ma_froc_1024/ma_froc.csv')
    parser.add_argument('--ma-native', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/ma_native_candidate/summary.json')
    parser.add_argument('--idrid-probe', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/idrid_feature_probe/summary.json')
    parser.add_argument('--perceptual-overlap', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/perceptual_overlap/perceptual_overlap_summary.json')
    parser.add_argument('--geometric-review', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/perceptual_overlap/geometric_candidate_review.json')
    parser.add_argument('--confirmed-overlap', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/perceptual_overlap/visually_confirmed_pairs.csv')
    parser.add_argument('--overlap-impact', type=Path,
                        default=PROJECT_ROOT / 'results/verification_2026-09-24/perceptual_overlap/overlap_impact.json')
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
        args.learned_quality_matlab_parity.resolve(),
        args.integrated_ablation.resolve(), args.ma_froc_512.resolve(),
        args.ma_froc_1024.resolve(), args.ma_native.resolve(),
        args.idrid_probe.resolve(),
        args.perceptual_overlap.resolve(),
        args.geometric_review.resolve(), args.confirmed_overlap.resolve(),
        args.overlap_impact.resolve())
    if args.output.exists():
        raise FileExistsError('Use a fresh passport path; prior evidence is immutable')
    atomic_json(args.output, passport)
    print(f'Validation Passport written to: {args.output.resolve()}')


if __name__ == '__main__':
    main()
