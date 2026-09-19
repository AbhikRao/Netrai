# NetrAI — Pipeline Status & Assets Walkthrough

## What's Completed

### 1. Model Training & Validation (Kaggle T4) ✅
- Architecture: **EfficientNet-B4 Dual-Head** (5-class ordinal grading + binary referable DR).
- Trained on **APTOS 2019 Blindness Detection** (3,662 images) using Fold 0 of a reproducible stratified 5-fold split, Focal Loss, and Cosine Annealing.
- **Weights Downloaded & Verified**: [`netrai_fold0_best.pth`](file:///home/ovoabhik/Downloads/Larp/Netrai/python/weights/netrai_fold0_best.pth) (72 MB) + [`fold0_thresholds.npy`](file:///home/ovoabhik/Downloads/Larp/Netrai/python/weights/fold0_thresholds.npy).
- **Validation Results**:
  - **QWK**: **0.9330** (SIH target: >0.85)
  - **Sensitivity**: **94.63%** (SIH target: >90%)
  - **Specificity**: **94.02%** (SIH target: >85%)
  - **AUC-ROC**: **0.9842**
  - Reproduced locally on all 733 held-out Fold-0 images with zero processing errors. Evidence: `results/verification_2026-09-17/fold0_model/`.
  - Validation curves and confusion matrix: [`validation_metrics.png`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/validation_metrics.png)

### 2. End-to-End Real Inference Pipeline ✅
All 5 modules are complete and integrated in [`python/inference.py`](file:///home/ovoabhik/Downloads/Larp/Netrai/python/inference.py):
- **M1 Quality Assessment**: Resolution-adaptive focus score, quadrant illumination, FOV coverage.
- **M2 Retinal Segmentation**: Adaptive thresholds for MAs, hemorrhages, exudates, optic disc, fovea, and vessel density within FOV.
- **M3 DR Severity Grading**: Real PyTorch model inference (~3–5s on CPU) with optimal threshold mapping.
- **M4 Explainability (CAFE)**: Real Grad-CAM from EfficientNet `conv_head` layer + ICDR compliance evidence checklist.
- **M5 Telemedicine Simulation**: Implemented in [`simulation.py`](file:///home/ovoabhik/Downloads/Larp/Netrai/python/modules/simulation.py) for PHC capacity, referral rate, and cost estimation.

### 3. Pipeline Bug Fixes Applied ✅
- Eliminated double calibration in `explainability.py`.
- Standardized Ben Graham preprocessing sigma to `10` across training and inference.
- Corrected RGB color space handling in FOV mask generation.
- Replaced hardcoded pixel-count thresholds with resolution-adaptive formulas.
- Cleaned glyph warnings in PDF/PNG clinical reports.
- Corrected optic-disc localization for partially clipped discs and removed FOV-edge exudate artifacts.
- Kept M1 enhancement out of M3 grading/Grad-CAM so model inputs exactly match training preprocessing.
- Added `python/batch_validate.py` plus automated pipeline tests.

### 4. MATLAB Online Deployment ✅ (Local package complete)
- Exported the real Fold-0 EfficientNet-B4 grade head to ONNX; no retraining or rule-based substitution.
- Added shared preprocessing/threshold config and an exact MATLAB parity fixture.
- Replaced the placeholder MATLAB grade, probability, segmentation, report, and synthetic Grad-CAM fallbacks with explicit real execution or honest errors/unavailable states.
- Rebuilt `validatePipeline.m` to export per-image predictions, aggregate CSV/JSON, confusion matrix, per-grade metrics, QWK, referable sensitivity/specificity, and AUC.
- PyTorch vs ONNX Runtime parity passed on five real images with maximum absolute logit error `5.25e-06`.
- Final execution of `verifyMatlabPackage` remains pending in MATLAB Online because this workstation has no MATLAB/Octave executable.

---

## Output Assets Ready for PPT

All refreshed real-model visual outputs are saved in `results/verification_2026-09-17/end_to_end_grade2/`:
- **6-Panel Clinical Report**: [`ClinicalReport.png`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/verification_2026-09-17/end_to_end_grade2/ClinicalReport.png) / [`ClinicalReport.pdf`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/verification_2026-09-17/end_to_end_grade2/ClinicalReport.pdf)
- **Grad-CAM Heatmap**: [`gradcam_heatmap.png`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/verification_2026-09-17/end_to_end_grade2/gradcam_heatmap.png)
- **Lesion Overlay**: [`lesion_overlay.png`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/verification_2026-09-17/end_to_end_grade2/lesion_overlay.png)
- **Vessel Segmentation**: [`vessel_segmentation.png`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/verification_2026-09-17/end_to_end_grade2/vessel_segmentation.png)
- **Validation Metrics**: [`results/validation_metrics.png`](file:///home/ovoabhik/Downloads/Larp/Netrai/results/validation_metrics.png)

---

## Roadmap / Next Steps

1. **SIH PPT Presentation (Deadline: September 30, 2026)**:
   - Use the reproduced metrics (0.9330 QWK, 94.63% sensitivity, 94.02% specificity) and refreshed assets from `results/verification_2026-09-17/`.
2. **Interactive Streamlit Web App (Optional)**:
   - Create a clean drag-and-drop web UI for live demonstration.
3. **Clinical Feature Validation**:
   - Supply a pixel-annotation manifest and run `python/evaluate_lesion_masks.py`; the precision/recall/Dice/IoU evaluator is ready, but the required masks are not in the repository.
   - Expand the new Grad-CAM border/FOV audit beyond the five-image smoke sample. The audit completed 5/5 but flagged 3/5 for localized corner attention; reports now expose the warning. True mitigation requires retraining with border masking/augmentation.
