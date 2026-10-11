"""
NetrAI Module 4: model attribution and candidate overlays
Grad-CAM, lesion overlay, calibrated confidence, evidence checklist.
"""

import cv2
import numpy as np

from utils.calibration import temperature_scale_probabilities


def generate_gradcam(model, image, target_class='referral', preprocessing_version='legacy-area-v1'):
    """Model-space attribution for an explicitly named target, not lesion proof."""
    try:
        from pytorch_grad_cam import GradCAM
        import torch
        from utils.preprocessing import preprocess_for_model

        target_layers = [model.backbone.conv_head]
        class BothHeads(torch.nn.Module):
            def __init__(self, network):
                super().__init__()
                self.network = network

            def forward(self, inputs):
                logits, referral = self.network(inputs)
                return torch.cat([logits, referral[:, None]], dim=1)

        def target(output):
            if target_class == 'referral':
                return output[5]
            if target_class == 'ordinal':
                return (output[:5].softmax(0) * torch.arange(5, device=output.device)).sum()
            if isinstance(target_class, int) and 0 <= target_class <= 4:
                return output[target_class]
            raise ValueError('Target must be referral, ordinal, or a class index 0–4')

        array = preprocess_for_model(image, version=preprocessing_version)
        input_tensor = torch.from_numpy(array.transpose(2, 0, 1)[None]).float().to(
            next(model.parameters()).device)
        with GradCAM(model=BothHeads(model).eval(), target_layers=target_layers) as cam:
            return cam(input_tensor=input_tensor, targets=[target])[0]
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
            'shortcut_flag': True,
            'informative': False,
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
        'informative': True,
    }


def generate_evidence_checklist(seg_results, grade, probs):
    """
    Structured retinal-candidate evidence checklist.
    Returns list of dicts with 'criterion', 'status', 'detail'.
    """
    checklist = []
    for label, key in (('MA point candidates', 'ma_count'),
                       ('Hemorrhage candidates', 'hem_count'),
                       ('Hard exudate candidates', 'hard_count'),
                       ('Soft exudate candidates', 'soft_count')):
        count = seg_results.get(key)
        checklist.append({'criterion': label,
                          'status': 'UNKNOWN' if count is None else 'CANDIDATE',
                          'detail': 'Not assessed' if count is None else
                          f'{count} unverified; zero does not exclude lesions'})
    distance = seg_results.get('fovea_dist')
    valid_distance = distance is not None and np.isfinite(distance) and distance >= 0
    checklist.append({'criterion': 'Exudate–fovea geometry',
                      'status': 'CANDIDATE' if valid_distance else 'UNKNOWN',
                      'detail': f'{distance:.2f} disc diameters; not a DME diagnosis'
                      if valid_distance else 'Not assessable; no DME conclusion'})
    nv = seg_results.get('nv_detected')
    checklist.append({'criterion': 'Vessel-proliferation heuristic',
                      'status': 'UNKNOWN' if nv is None else 'CANDIDATE',
                      'detail': 'Not assessed' if nv is None else
                      ('Flagged; needs expert confirmation' if nv else 'Not flagged; NV not excluded')})
    checklist.append({'criterion': 'Grade attribution', 'status': 'UNVERIFIED',
                      'detail': f'Grade {grade}; lesion concordance not established'})
    return checklist
