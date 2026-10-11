#!/usr/bin/env python3
"""
NetrAI Inference & Demo — Generate research reports with trained model predictions.
Uses the versioned EfficientNet-B4 deployment artifacts.

Usage:
    python inference.py --image ../data/patient_sample.jpg
    python inference.py --demo   # runs on sample + generates all visualizations
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.gridspec as gridspec
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.quality import assess_image_quality, enhance_fundus, generate_recapture_feedback
from modules.segmentation import (localize_optic_disc, localize_fovea, segment_vessels,
                                   detect_microaneurysms, detect_hemorrhages,
                                   detect_exudates, detect_neovascularization,
                                   extract_clinical_features)
from modules.ma_native import detect_ranked_microaneurysms
from modules.idrid_probe import predict_idrid_referral_probe
from modules.explainability import (assess_gradcam_quality, generate_lesion_overlay,
                                     calibrate_confidence, generate_evidence_checklist)
from utils.preprocessing import (
    IMAGENET_MEAN, IMAGENET_STD, model_space_fov_mask, preprocess_fundus,
    preprocess_for_model)
from utils.calibration import load_calibration_temperature
from utils.safety import assign_triage, load_safety_policy
from utils.artifacts import atomic_json, json_value
from utils.validation import ordinal_thresholds, probabilities

GRADE_NAMES = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR']
GRADE_COLORS = ['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#B71C1C']
GRADE_ACTIONS = [
    'Research result; clinician determines follow-up',
    'Research result; clinician determines follow-up',
    'Referable prediction; clinician review required',
    'High-severity prediction; clinician review required',
    'High-severity prediction; clinician review required'
]
CALIBRATION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'weights', 'calibration.json')
SAFETY_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'weights', 'safety_policy.json')
_CALIBRATION_TEMPERATURE = None
_SAFETY_POLICY = None


def get_calibration_temperature():
    """Return the strictly validated, versioned confidence temperature."""
    global _CALIBRATION_TEMPERATURE
    if _CALIBRATION_TEMPERATURE is None:
        _CALIBRATION_TEMPERATURE = load_calibration_temperature(CALIBRATION_PATH)
    return _CALIBRATION_TEMPERATURE


def get_safety_policy():
    """Return the strictly validated dual-head screening policy."""
    global _SAFETY_POLICY
    if _SAFETY_POLICY is None:
        _SAFETY_POLICY = load_safety_policy(SAFETY_POLICY_PATH)
    return _SAFETY_POLICY

# ── Shared Preprocessing ─────────────────────────────────────────────────────

def _preprocess_for_model(image_rgb):
    """Historical legacy-area-v1 input; retained for benchmark reproducibility."""
    return preprocess_for_model(image_rgb, size=512)


def _model_display_image(image_rgb):
    """Return the exact 512px model input in displayable RGB form."""
    normalized = _preprocess_for_model(image_rgb)
    restored = (normalized * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.clip(restored, 0, 255).astype(np.uint8)


# ── Real Model Loading ───────────────────────────────────────────────────────

def _load_model():
    """Load the trained NetrAI model from weights directory."""
    import torch
    from models.netrai_model import NetrAIModel

    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
    pth_path = os.path.join(weights_dir, 'netrai_fold0_best.pth')
    thresh_path = os.path.join(weights_dir, 'fold0_thresholds.npy')

    if not os.path.exists(pth_path):
        raise FileNotFoundError(
            f'Model weights not found at {pth_path}; refusing to fabricate a grade')

    device = torch.device('cpu')
    model = NetrAIModel('efficientnet_b4', 5, pretrained=False)
    from utils.bundle import load_legacy_checkpoint, verify_bundle
    bundle = verify_bundle(Path(weights_dir) / 'bundle.json')
    ckpt = load_legacy_checkpoint(pth_path, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model._netrai_checkpoint_sha256 = bundle['files']['netrai_fold0_best.pth']
    model.eval()
    model.to(device)

    thresholds = ordinal_thresholds(np.load(thresh_path))

    qwk = ckpt.get('qwk', None)
    qwk_str = f'{qwk:.4f}' if isinstance(qwk, (int, float)) else 'N/A'
    print(f"[+] Loaded legacy model (historical internal QWK={qwk_str}, epoch {ckpt.get('epoch', '?')}; Fold-0 overlaps training)")
    return model, thresholds, device


# Global model (loaded once)
_MODEL = None
_THRESHOLDS = None
_DEVICE = None
_MODEL_LOADED = False


def _get_model():
    """Lazy-load model on first use."""
    global _MODEL, _THRESHOLDS, _DEVICE, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL, _THRESHOLDS, _DEVICE = _load_model()
        _MODEL_LOADED = True
    return _MODEL, _THRESHOLDS, _DEVICE


def predict_with_model(image_rgb):
    """
    Run real model inference on a fundus image.
    Returns (grade, probs) where probs is a 5-element array.
    """
    grades, probs, _ = predict_batch_with_model([image_rgb], batch_size=1)
    if grades is None:
        return None, None
    return int(grades[0]), probs[0]


def predict_with_model_details(image_rgb):
    """Run one image and retain grade/ref-head safety evidence."""
    details = predict_batch_with_details([image_rgb], batch_size=1)
    single = {}
    for key, value in details.items():
        if not isinstance(value, np.ndarray):
            single[key] = value
            continue
        first = value[0]
        # Numeric/bool NumPy scalars expose ``item``; object arrays may hold
        # ordinary Python strings (for example the triage action), which do not.
        single[key] = (
            first.item()
            if np.ndim(first) == 0 and hasattr(first, 'item')
            else first
        )
    return single


def predict_batch_with_model(images_rgb, batch_size=8):
    """Backward-compatible grade prediction tuple."""
    details = predict_batch_with_details(images_rgb, batch_size=batch_size)
    return (
        details['grades'], details['grade_probabilities'],
        details['continuous_scores'])


def predict_batch_with_details(images_rgb, batch_size=8, *, preprocessing_version='legacy-area-v1'):
    """Run both trained heads and derive explicit disagreement/triage signals."""
    import torch
    import torch.nn.functional as F

    if batch_size < 1:
        raise ValueError('batch_size must be at least 1')
    if not images_rgb:
        return {
            'grades': np.array([], dtype=int),
            'grade_probabilities': np.empty((0, 5)),
            'continuous_scores': np.array([], dtype=float),
            'grade_head_referable_probabilities': np.array([], dtype=float),
            'referable_head_probabilities': np.array([], dtype=float),
            'grade_head_referable_predictions': np.array([], dtype=bool),
            'referable_head_predictions': np.array([], dtype=bool),
            'head_disagreements': np.array([], dtype=bool),
            'triage_actions': np.array([], dtype=object),
        }

    model, thresholds, device = _get_model()
    all_probs = []
    all_referable_probs = []
    for start in range(0, len(images_rgb), batch_size):
        batch = images_rgb[start:start + batch_size]
        arrays = [preprocess_for_model(image, version=preprocessing_version) for image in batch]
        tensor = torch.from_numpy(
            np.stack(arrays).transpose(0, 3, 1, 2)
        ).float().to(device)

        with torch.inference_mode():
            grade_logits, referable_logits = model(tensor)
            all_probs.append(F.softmax(grade_logits, dim=1).cpu().numpy())
            all_referable_probs.append(torch.sigmoid(referable_logits).cpu().numpy())

    probs = probabilities(np.concatenate(all_probs, axis=0))
    referable_head_probs = np.concatenate(all_referable_probs, axis=0)
    continuous_scores = (probs * np.arange(5)).sum(axis=1)
    grades = np.digitize(continuous_scores, ordinal_thresholds(thresholds)).clip(0, 4)
    if not np.isfinite(referable_head_probs).all():
        raise ValueError('Nonfinite referral probabilities')

    policy = get_safety_policy()
    grades = grades.astype(int)
    grade_head_referable = grades >= 2
    referable_head_positive = referable_head_probs >= policy['referable_head_threshold']
    disagreements = grade_head_referable != referable_head_positive
    triage_actions = np.array([
        assign_triage(
            quality_rejected=False,
            grade_referable=bool(grade_head_referable[index]),
            referable_head_positive=bool(referable_head_positive[index]))
        for index in range(len(grades))
    ], dtype=object)
    return {
        'grades': grades,
        'grade_probabilities': probs,
        'continuous_scores': continuous_scores,
        'grade_head_referable_probabilities': probs[:, 2:].sum(axis=1),
        'referable_head_probabilities': referable_head_probs,
        'grade_head_referable_predictions': grade_head_referable,
        'referable_head_predictions': referable_head_positive,
        'head_disagreements': disagreements,
        'triage_actions': triage_actions,
    }


def generate_real_gradcam(image_rgb, target='referral'):
    """Attribution of the referral logit (default) or continuous ordinal score."""
    from modules.explainability import generate_gradcam
    model, _, _ = _get_model()
    return generate_gradcam(model, image_rgb, target_class=target)


def generate_clinical_report(image_rgb, seg_results, grade, probs, heatmap,
                              overlay, output_dir, filename='ClinicalReport',
                              gradcam_image=None, gradcam_qc=None,
                              calibration_temperature=None, safety=None):
    """Generate a six-panel research screening report."""
    os.makedirs(output_dir, exist_ok=True)

    if calibration_temperature is None:
        calibration_temperature = get_calibration_temperature()
    cal_probs = calibrate_confidence(probs, temperature=calibration_temperature)
    checklist = generate_evidence_checklist(seg_results, grade, cal_probs)
    if safety is not None:
        checklist.append({
            'criterion': 'Shared-backbone Referral Head',
            'status': 'HIGH' if safety['head_disagreement'] else 'AGREE',
            'detail': (
                f"{safety['referable_head_probability']*100:.1f}%; "
                f"{'DISAGREES - human review' if safety['head_disagreement'] else 'agrees with grade policy'}"
            ),
        })
        research = safety.get('research_referral')
        if research is not None:
            checklist.append({
                'criterion': 'IDRiD Referral Probe',
                'status': 'CANDIDATE',
                'detail': (f"{research['probability']*100:.1f}% research score; "
                           'no triage effect'),
            })

    fig = plt.figure(figsize=(20, 12), facecolor='white')
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.25)

    # Title banner
    fig.suptitle('NetrAI Research Screening Report', fontsize=22, fontweight='bold', y=0.965)
    fig.text(0.5, 0.935, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | '
             f'Not a diagnosis | Unverified lesion candidates | Requires expert review',
             ha='center', fontsize=10, color='gray')

    # Panel 1: Enhanced Fundus
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image_rgb)
    ax1.set_title('Enhanced Fundus Image', fontsize=13, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Grad-CAM
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gradcam_image if gradcam_image is not None else image_rgb)
    if heatmap is None:
        ax2.text(0.5, 0.5, 'Grad-CAM unavailable', ha='center', va='center',
                 transform=ax2.transAxes, color='white', backgroundcolor='black')
        ax2.set_title('Explanation unavailable', fontsize=13, fontweight='bold')
    else:
        hm_colored = plt.cm.jet(heatmap)[:, :, :3]
        ax2.imshow(hm_colored, alpha=0.45)
        qc_suffix = ' - QC FLAG' if gradcam_qc and gradcam_qc['shortcut_flag'] else ''
        ax2.set_title(f'Referral-logit Grad-CAM{qc_suffix}', fontsize=13, fontweight='bold')
    ax2.axis('off')

    # Panel 3: Lesion Overlay
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ma_candidate = seg_results.get('ma_detector') == 'trained-2048'
    ax3.set_title('Unverified Lesion Candidates',
                  fontsize=13, fontweight='bold')
    ax3.axis('off')
    legend_items = [
        ('red', 'Microaneurysms'), ('darkred', 'Hemorrhages'),
        ('yellow', 'Hard Exudates'), ('lime', 'Optic Disc'), ('cyan', 'Fovea')
    ]
    for i, (color, label) in enumerate(legend_items):
        ax3.plot([], [], 's', color=color, markersize=8, label=label)
    ax3.legend(loc='lower right', fontsize=7, framealpha=0.8)

    # Panel 4: Diagnosis Summary
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    grade_color = GRADE_COLORS[grade]
    is_referable = grade >= 2
    triage_action = safety['triage_action'] if safety else ('refer' if is_referable else 'auto_clear')
    if triage_action == 'uncertain_human_review':
        ref_text = 'UNCERTAIN - HEAD DISAGREEMENT; HUMAN REVIEW'
        ref_color = '#B71C1C'
        action_text = 'Human review before screening disposition'
    else:
        ref_text = 'Predicted referable - expert review' if is_referable else 'Predicted non-referable - not clearance'
        ref_color = '#D32F2F' if is_referable else '#388E3C'
        action_text = GRADE_ACTIONS[grade]

    text_lines = [
        (0.5, 0.85, f'DR Grade: Level {grade}', 20, grade_color, 'bold'),
        (0.5, 0.72, GRADE_NAMES[grade], 16, grade_color, 'bold'),
        (0.5, 0.58, ref_text, 12, ref_color, 'bold'),
        (0.5, 0.42, f'Post-hoc confidence: {cal_probs[grade]*100:.1f}%', 14, '#333', 'normal'),
        (0.5, 0.28, f'Action: {action_text}', 11, '#555', 'normal'),
        (0.5, 0.14, 'EfficientNet-B4 | Internal post-hoc calibration only', 10, '#888', 'normal'),
    ]
    for x, y, text, size, color, weight in text_lines:
        ax4.text(x, y, text, ha='center', va='center', fontsize=size,
                 color=color, fontweight=weight, transform=ax4.transAxes)
    ax4.set_title('Research Prediction', fontsize=13, fontweight='bold')

    # Panel 5: Evidence Checklist
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    ax5.set_title('Unverified Candidate Checklist', fontsize=13, fontweight='bold')
    y_pos = 0.92
    checklist_step = 0.11 if len(checklist) > 8 else 0.13
    for item in checklist:
        icon = ('[?]' if item['status'] == 'CANDIDATE' else
                ('[+]' if item['status'] == 'FOUND' else
                 ('[!]' if item['status'] == 'HIGH' else '[-]')))
        status_color = '#D32F2F' if item['status'] == 'HIGH' else '#806000'
        ax5.text(0.05, y_pos, f"{icon} {item['criterion']}: {item['detail']}",
                 fontsize=9, color=status_color, transform=ax5.transAxes, verticalalignment='top')
        y_pos -= checklist_step

    # Panel 6: Probability Bar Chart
    ax6 = fig.add_subplot(gs[1, 2])
    bars = ax6.bar(range(5), cal_probs, color=['#90CAF9'] * 5, edgecolor='#1565C0', linewidth=1.2)
    bars[grade].set_color(grade_color)
    bars[grade].set_edgecolor('#333')
    bars[grade].set_linewidth(2)
    ax6.set_xticks(range(5))
    ax6.set_xticklabels(['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'], fontsize=9)
    ax6.set_ylabel('Calibrated Probability', fontsize=11)
    ax6.set_ylim(0, 1.05)
    ax6.set_title(
        f'DR Grade Probabilities (T={calibration_temperature:.3f})',
        fontsize=13, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    # Save
    png_path = os.path.join(output_dir, f'{filename}.png')
    pdf_path = os.path.join(output_dir, f'{filename}.pdf')
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Research screening report saved: {png_path}")
    return png_path


def run_pipeline(image_path, output_dir='../results', ma_detector='classic',
                 referral_probe='none'):
    """Run the full NetrAI pipeline on a single image."""
    print(f"\n{'='*60}")
    print(f"  NetrAI — Analyzing: {os.path.basename(image_path)}")
    print(f"{'='*60}\n")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'Could not decode image: {image_path}')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    analysis_rgb = image_rgb

    # Module 1: Quality Assessment
    print("[M1] Image Quality Assessment...")
    iqs, metrics = assess_image_quality(image_rgb)
    print(f"     IQS Score: {iqs:.3f} (Focus: {metrics['focus']:.3f}, "
          f"Illumination: {metrics['illumination']:.3f}, FOV: {metrics['fov']:.3f})")

    if iqs < 0.3:
        feedback = generate_recapture_feedback(metrics)
        print(f"     [✗] Image REJECTED — {feedback}")
        result = {
            'status': 'rejected', 'iqs': iqs, 'feedback': feedback,
            'triage_action': assign_triage(
                quality_rejected=True, grade_referable=False,
                referable_head_positive=False),
        }
        result.update(schema_version=2, research_only=True, quality_metrics=metrics)
        atomic_json(Path(output_dir) / 'result.json', result)
        return result

    if iqs < 0.7:
        print("     [~] Borderline quality — applying adaptive enhancement...")
        analysis_rgb = enhance_fundus(image_rgb)

    # Preprocess to 512x512
    proc_img, green_ch, fov_mask = preprocess_fundus(analysis_rgb, target_size=512)

    # Module 2: Segmentation
    print("[M2] Retinal Structure Segmentation...")
    od_center, od_radius = localize_optic_disc(proc_img)
    print(f"     Optic Disc: center=({od_center[0]:.0f}, {od_center[1]:.0f}), radius={od_radius:.0f}")

    fovea_center = localize_fovea(green_ch, od_center, od_radius)
    print(f'     Fovea candidate: {fovea_center}')

    vessel_mask = segment_vessels(green_ch, fov_mask)
    vessel_pct = (vessel_mask > 0).sum() / max((fov_mask > 0).sum(), 1) * 100
    print(f"     Vessels: {vessel_pct:.1f}% coverage")

    if ma_detector == 'trained-2048':
        ma_mask, ma_count, ma_centroids = detect_ranked_microaneurysms(image)
        print(f"     MA research candidates: {ma_count} (IDRiD-trained; not clinical evidence)")
    elif ma_detector == 'classic':
        ma_mask, ma_count, ma_centroids = detect_microaneurysms(
            green_ch, vessel_mask, fov_mask, od_center, od_radius)
        print(f"     Microaneurysms: {ma_count} detected")
    else:
        raise ValueError(f'Unsupported MA detector: {ma_detector}')

    hem_mask, hem_count, hem_stats = detect_hemorrhages(
        green_ch, vessel_mask, ma_mask, fov_mask, od_center, od_radius)
    print(f"     Hemorrhages: {hem_count} (dot: {hem_stats['dot_count']}, blot: {hem_stats['blot_count']})")

    ex_mask, hard_count, soft_count, fovea_dist = detect_exudates(
        proc_img, green_ch, fovea_center, od_center, od_radius, fov_mask)
    print(f"     Exudates: {hard_count} hard, {soft_count} soft, fovea dist: {fovea_dist:.2f} DD")

    nv_mask, nv_detected = detect_neovascularization(green_ch, vessel_mask, od_center, od_radius, fov_mask=fov_mask)
    print(f"     Neovascularization heuristic: {'CANDIDATE' if nv_detected else 'No candidate; absence unverified'}")

    seg_results = {
        'od_center': od_center, 'od_radius': od_radius, 'fovea_center': fovea_center,
        'vessel_mask': vessel_mask, 'ma_mask': ma_mask, 'ma_count': ma_count,
        'ma_centroids': ma_centroids, 'hem_mask': hem_mask, 'hem_count': hem_count,
        'hem_stats': hem_stats, 'ex_mask': ex_mask, 'hard_count': hard_count,
        'soft_count': soft_count, 'fovea_dist': fovea_dist, 'nv_mask': nv_mask,
        'nv_detected': nv_detected, 'fov_mask': fov_mask,
        'ma_detector': ma_detector,
    }

    clinical_features = extract_clinical_features(seg_results)
    print(f"     12-D Retinal Candidate Vector: [{', '.join(f'{v:.3f}' for v in clinical_features)}]")

    # Module 3: DR Severity Grading (REAL MODEL)
    print("[M3] DR Severity Grading (EfficientNet-B4)...")
    # Model grading always uses the original RGB image.  The model's own
    # Legacy-area-v1 preprocessing reproduces the published inference path;
    # the old notebook's training resize differed. M1 enhancement is for M2/report.
    prediction = predict_with_model_details(image_rgb)
    grade = int(prediction['grades'])
    probs = prediction['grade_probabilities']
    calibration_temperature = get_calibration_temperature()
    calibrated_probs = calibrate_confidence(probs, calibration_temperature)
    print(f"     Grade: Level {grade} ({GRADE_NAMES[grade]})")
    print(f"     Probabilities: [{', '.join(f'{p:.3f}' for p in probs)}]")
    print(f"     Post-hoc confidence: {calibrated_probs[grade]*100:.1f}% "
          f"(temperature={calibration_temperature:.3f})")
    print(f"     Referable: {'YES' if grade >= 2 else 'No'}")
    print(f"     Shared-backbone referable head: "
          f"{prediction['referable_head_probabilities']*100:.1f}%")
    print(f"     Head disagreement: {bool(prediction['head_disagreements'])}")
    print(f"     Triage: {prediction['triage_actions']}")
    print("     Method: Legacy EfficientNet-B4 research checkpoint")
    research_referral = None
    if referral_probe == 'idrid':
        model, _, _ = _get_model()
        research_referral = predict_idrid_referral_probe(image_rgb, model)
        print(f"     IDRiD-trained research referral probability: "
              f"{research_referral['probability']:.3f}; "
              f"positive={research_referral['positive']} (no triage effect)")
    elif referral_probe != 'none':
        raise ValueError(f'Unsupported referral probe: {referral_probe}')

    # Module 4: Model attribution and candidate evidence
    print("[M4] Model attribution and candidate evidence...")
    heatmap = generate_real_gradcam(image_rgb)
    gradcam_qc = None
    if heatmap is not None:
        gradcam_qc = assess_gradcam_quality(
            heatmap, model_space_fov_mask(image_rgb, size=heatmap.shape[0]))
        heatmap = cv2.resize(heatmap, (proc_img.shape[1], proc_img.shape[0]))
        print("     Grad-CAM: Real (from trained model)")
        print(f"     Explanation QC: border enrichment {gradcam_qc['border_enrichment']:.3f}, "
              f"outside-FOV attention {gradcam_qc['outside_fov_attention_fraction']:.3f}, "
              f"flag={gradcam_qc['shortcut_flag']}")
    else:
        print("     Grad-CAM: unavailable (no synthetic substitute generated)")

    overlay = generate_lesion_overlay(proc_img, seg_results)

    model_display = _model_display_image(image_rgb)
    report_path = generate_clinical_report(
        proc_img, seg_results, grade, probs, heatmap, overlay, output_dir,
        gradcam_image=model_display, gradcam_qc=gradcam_qc,
        calibration_temperature=calibration_temperature,
        safety={
            'referable_head_probability': float(
                prediction['referable_head_probabilities']),
            'head_disagreement': bool(prediction['head_disagreements']),
            'triage_action': str(prediction['triage_actions']),
            'research_referral': research_referral,
        })

    # Also save individual panels for PPT
    save_individual_panels(
        proc_img, vessel_mask, heatmap, overlay, seg_results, output_dir,
        gradcam_image=model_display)

    print(f"\n[✓] Analysis complete. Results in {output_dir}/")
    result = {
        'status': 'success', 'iqs': iqs, 'grade': grade,
        'grade_name': GRADE_NAMES[grade], 'probs': probs,
        'calibrated_probs': calibrated_probs,
        'confidence': float(calibrated_probs[grade]),
        'calibration_temperature': calibration_temperature,
        'is_referable': grade >= 2, 'report_path': report_path,
        'grade_head_referable_probability': float(
            prediction['grade_head_referable_probabilities']),
        'referable_head_probability': float(
            prediction['referable_head_probabilities']),
        'referable_head_prediction': bool(
            prediction['referable_head_predictions']),
        'head_disagreement': bool(prediction['head_disagreements']),
        'triage_action': str(prediction['triage_actions']),
        'research_referral': research_referral,
        'real_model': True, 'quality_metrics': metrics,
        'clinical_features': clinical_features, 'gradcam_qc': gradcam_qc,
        'segmentation': {
            'ma_count': ma_count, 'hem_count': hem_count,
            'hard_count': hard_count, 'soft_count': soft_count,
            'fovea_dist': fovea_dist, 'nv_detected': nv_detected,
            'vessel_pct': vessel_pct,
        },
    }
    result.update(schema_version=2, research_only=True,
                  preprocessing_version='legacy-area-v1',
                  explanation_target='referral_logit',
                  explanation_coordinate_frame='classifier_crop_resize',
                  lesion_evidence_status='unverified_candidates')
    from utils.artifacts import sha256_file
    result['bundle_sha256'] = sha256_file(Path(__file__).parent / 'weights/bundle.json')
    result = json_value(result)
    atomic_json(Path(output_dir) / 'result.json', result)
    return result


def save_individual_panels(image, vessel_mask, heatmap, overlay, seg_results,
                           output_dir, gradcam_image=None):
    """Save individual high-res panels for PPT slides."""
    os.makedirs(output_dir, exist_ok=True)

    # Vessel segmentation
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    vessel_overlay = np.zeros((*vessel_mask.shape, 4))
    vessel_overlay[vessel_mask > 0] = [0, 1, 0, 0.5]
    ax.imshow(vessel_overlay)
    ax.set_title('Retinal Vessel Segmentation', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'vessel_segmentation.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Grad-CAM standalone
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(gradcam_image if gradcam_image is not None else image)
    if heatmap is None:
        ax.text(0.5, 0.5, 'Grad-CAM unavailable', ha='center', va='center',
                transform=ax.transAxes, color='white', backgroundcolor='black')
        ax.set_title('Explanation unavailable', fontsize=16, fontweight='bold')
    else:
        hm_colored = plt.cm.jet(heatmap)[:, :, :3]
        ax.imshow(hm_colored, alpha=0.5)
        ax.set_title('Grad-CAM Attention Heatmap', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'gradcam_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # Lesion overlay standalone
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(overlay)
    ax.set_title('Unverified Retinal Lesion Candidates', fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(output_dir, 'lesion_overlay.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"[✓] Individual panels saved for PPT")


def main():
    parser = argparse.ArgumentParser(description='NetrAI Inference')
    parser.add_argument('--image', type=str, help='Path to fundus image')
    parser.add_argument('--demo', action='store_true', help='Run demo on sample image')
    parser.add_argument('--output', type=str, default='../results', help='Output directory')
    parser.add_argument('--ma-detector', choices=('classic', 'trained-2048'),
                        default='classic',
                        help='Experimental IDRiD-trained MA candidate, off by default')
    parser.add_argument('--referral-probe', choices=('none', 'idrid'), default='none',
                        help='Experimental IDRiD-trained binary referral score; no triage effect')
    args = parser.parse_args()

    if args.demo:
        # Try APTOS images first, then patient sample
        aptos_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'aptos2019', 'train_images')
        sample = os.path.join(os.path.dirname(__file__), '..', 'data', 'patient_sample.jpg')

        if os.path.isdir(aptos_dir):
            import pandas as pd
            csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'aptos2019', 'train.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # Pick one image per grade for demo
                for grade_val in [0, 2, 4]:
                    subset = df[df['diagnosis'] == grade_val]
                    if len(subset) > 0:
                        img_id = subset.iloc[0]['id_code']
                        img_path = os.path.join(aptos_dir, f'{img_id}.png')
                        if os.path.exists(img_path):
                            grade_dir = os.path.join(args.output, f'grade{grade_val}')
                            run_pipeline(img_path, grade_dir, ma_detector=args.ma_detector,
                                         referral_probe=args.referral_probe)
                return
            # If no CSV, just pick first 3 images
            imgs = sorted([f for f in os.listdir(aptos_dir) if f.endswith('.png')])[:3]
            for img_file in imgs:
                run_pipeline(os.path.join(aptos_dir, img_file), args.output,
                             ma_detector=args.ma_detector,
                             referral_probe=args.referral_probe)
            return

        if os.path.exists(sample):
            run_pipeline(sample, args.output, ma_detector=args.ma_detector,
                         referral_probe=args.referral_probe)
        else:
            parser.error('No local sample images found. Provide --image PATH; datasets are not bundled.')
    elif args.image:
        run_pipeline(args.image, args.output, ma_detector=args.ma_detector,
                     referral_probe=args.referral_probe)
    else:
        parser.error('Provide --image PATH or --demo')


if __name__ == '__main__':
    main()
