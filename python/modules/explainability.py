"""
NetrAI Module 4: CAFE Explainability
Grad-CAM, lesion overlay, calibrated confidence, evidence checklist.
"""

import cv2
import numpy as np

from utils.calibration import temperature_scale_probabilities


def generate_gradcam(model, image, target_class):
    """Real Grad-CAM using pytorch-grad-cam library."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import torch

        target_layers = [model.backbone.conv_head]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [ClassifierOutputTarget(target_class)]

        if not isinstance(image, torch.Tensor):
            input_tensor = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        else:
            input_tensor = image

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        return grayscale_cam
    except Exception as e:
        import warnings
        warnings.warn(f'Grad-CAM failed: {e}')
        return None


def generate_lesion_overlay(image, seg_results):
    """Color-coded overlay: red MAs, dark red hemorrhages, yellow exudates, green OD, cyan fovea."""
    overlay = image.copy()

    # MAs in red
    if seg_results.get('ma_mask') is not None and np.any(seg_results['ma_mask']):
        mask = seg_results['ma_mask'] > 0
        overlay[mask] = [255, 0, 0]

    # Hemorrhages in dark red
    if seg_results.get('hem_mask') is not None and np.any(seg_results['hem_mask']):
        mask = seg_results['hem_mask'] > 0
        overlay[mask] = [180, 0, 0]

    # Exudates in yellow
    if seg_results.get('ex_mask') is not None and np.any(seg_results['ex_mask']):
        mask = seg_results['ex_mask'] > 0
        overlay[mask] = [255, 255, 0]

    # Neovascularization in magenta
    if seg_results.get('nv_mask') is not None and np.any(seg_results['nv_mask']):
        mask = seg_results['nv_mask'] > 0
        overlay[mask] = [255, 0, 255]

    # OD circle in green
    od_center = seg_results.get('od_center')
    od_radius = seg_results.get('od_radius')
    if od_center is not None and od_radius is not None:
        cv2.circle(overlay, od_center, int(od_radius), (0, 255, 0), 2)

    # Fovea circle in cyan
    fovea_center = seg_results.get('fovea_center')
    if fovea_center is not None:
        cv2.circle(overlay, fovea_center, 10, (0, 255, 255), -1)
        cv2.circle(overlay, fovea_center, 12, (0, 200, 200), 2)

    # Vessel overlay in blue (subtle)
    if seg_results.get('vessel_mask') is not None:
        vessel_overlay = np.zeros_like(overlay)
        vessel_overlay[seg_results['vessel_mask'] > 0] = [0, 100, 255]
        overlay = cv2.addWeighted(overlay, 0.85, vessel_overlay, 0.15, 0)

    # Blend with original
    blended = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
    return blended


def calibrate_confidence(probs, temperature=1.0):
    """Temperature-scale probabilities; the identity is the safe default."""
    return temperature_scale_probabilities(probs, temperature)


def assess_gradcam_quality(heatmap, fov_mask=None, border_fraction=0.10):
    """Compute heuristic Grad-CAM shortcut indicators.

    These metrics are quality-control signals, not clinical validation.  A
    border-enrichment value above one means average attention is higher in the
    outer band than in the interior.
    """
    cam = np.asarray(heatmap, dtype=np.float64)
    if cam.ndim != 2 or cam.size == 0 or not np.isfinite(cam).all():
        raise ValueError('heatmap must be a finite non-empty 2-D array')
    cam = np.clip(cam, 0, None)
    total = float(cam.sum())
    if total <= 0:
        return {
            'border_attention_fraction': 0.0,
            'border_enrichment': 0.0,
            'max_corner_enrichment': 0.0,
            'top5_border_fraction': 0.0,
            'outside_fov_attention_fraction': 0.0,
            'top20_outside_fov_fraction': 0.0,
            'shortcut_flag': False,
        }

    h, w = cam.shape
    band = max(1, int(round(min(h, w) * border_fraction)))
    border = np.zeros((h, w), dtype=bool)
    border[:band] = border[-band:] = True
    border[:, :band] = border[:, -band:] = True
    border_mean = float(cam[border].mean())
    interior_mean = float(cam[~border].mean()) if (~border).any() else 0.0
    border_enrichment = border_mean / max(interior_mean, 1e-12)
    corner_size = max(1, int(round(min(h, w) * 0.15)))
    corners = [
        cam[:corner_size, :corner_size], cam[:corner_size, -corner_size:],
        cam[-corner_size:, :corner_size], cam[-corner_size:, -corner_size:],
    ]
    global_mean = float(cam.mean())
    max_corner_enrichment = max(float(corner.mean()) for corner in corners) / max(global_mean, 1e-12)
    top5 = cam >= float(np.quantile(cam, 0.95))
    top5_border_fraction = float(np.mean(border[top5])) if top5.any() else 0.0

    outside_fraction = 0.0
    top20_outside = 0.0
    if fov_mask is not None:
        fov = np.asarray(fov_mask).astype(bool)
        if fov.shape != cam.shape:
            fov = cv2.resize(fov.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        outside_fraction = float(cam[~fov].sum() / total)
        threshold = float(np.quantile(cam, 0.80))
        top = cam >= threshold
        top20_outside = float(np.mean(~fov[top])) if top.any() else 0.0

    shortcut_flag = bool(
        border_enrichment > 1.5 or max_corner_enrichment > 2.0
        or top5_border_fraction > 0.60 or outside_fraction > 0.20
        or top20_outside > 0.25
    )
    return {
        'border_attention_fraction': float(cam[border].sum() / total),
        'border_enrichment': border_enrichment,
        'max_corner_enrichment': max_corner_enrichment,
        'top5_border_fraction': top5_border_fraction,
        'outside_fov_attention_fraction': outside_fraction,
        'top20_outside_fov_fraction': top20_outside,
        'shortcut_flag': shortcut_flag,
    }


def generate_evidence_checklist(seg_results, grade, probs):
    """
    Structured clinical evidence checklist.
    Returns list of dicts with 'criterion', 'status', 'detail'.
    """
    checklist = []

    # Microaneurysms
    ma_count = seg_results.get('ma_count', 0)
    checklist.append({
        'criterion': 'Microaneurysms',
        'status': 'FOUND' if ma_count > 0 else 'ABSENT',
        'detail': f'{ma_count} detected' if ma_count > 0 else 'None detected'
    })

    # Hemorrhages
    hem_count = seg_results.get('hem_count', 0)
    hem_stats = seg_results.get('hem_stats', {})
    dot = hem_stats.get('dot_count', 0)
    blot = hem_stats.get('blot_count', 0)
    checklist.append({
        'criterion': 'Hemorrhages',
        'status': 'FOUND' if hem_count > 0 else 'ABSENT',
        'detail': f'{hem_count} ({dot} dot, {blot} blot)' if hem_count > 0 else 'None detected'
    })

    # Hard Exudates
    hard_count = seg_results.get('hard_count', 0)
    checklist.append({
        'criterion': 'Hard Exudates',
        'status': 'FOUND' if hard_count > 0 else 'ABSENT',
        'detail': f'{hard_count} clusters' if hard_count > 0 else 'None detected'
    })

    # Soft Exudates (Cotton Wool Spots)
    soft_count = seg_results.get('soft_count', 0)
    checklist.append({
        'criterion': 'Soft Exudates (CWS)',
        'status': 'FOUND' if soft_count > 0 else 'ABSENT',
        'detail': f'{soft_count} detected' if soft_count > 0 else 'None detected'
    })

    # DME Risk
    fovea_dist = seg_results.get('fovea_dist', -1)
    if fovea_dist >= 0 and fovea_dist < 1.0:
        dme_status = 'HIGH'
        dme_detail = f'Exudates within {fovea_dist:.2f} DD of fovea - DME risk'
    elif fovea_dist >= 0:
        dme_status = 'LOW'
        dme_detail = f'Exudates {fovea_dist:.2f} DD from fovea'
    else:
        dme_status = 'ABSENT'
        dme_detail = 'No exudates near fovea'
    checklist.append({
        'criterion': 'Macular Edema Risk',
        'status': dme_status,
        'detail': dme_detail
    })

    # Neovascularization
    nv = seg_results.get('nv_detected', False)
    checklist.append({
        'criterion': 'Neovascularization',
        'status': 'FOUND' if nv else 'ABSENT',
        'detail': 'Abnormal vessel growth detected' if nv else 'Not detected'
    })

    # ICDR Grade Concordance
    cal_probs = np.array(probs)
    confidence = cal_probs[grade] * 100
    checklist.append({
        'criterion': 'AI Confidence',
        'status': 'HIGH' if confidence > 60 else 'LOW',
        'detail': f'{confidence:.1f}% for Grade {grade}'
    })

    return checklist
