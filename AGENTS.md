# NetrAI — Agent Memory & Project Handover Context

> **SIH 2026 | Problem Statement 26038 | Team Biscuit | MathWorks Sponsored**  
> **Topic**: Explainable AI for Diabetic Retinopathy Screening in Rural India  
> **Submission Deadline**: September 30, 2026  
> **Workspace**: `/home/ovoabhik/Downloads/Larp/Netrai/`
> **Public repository**: <https://github.com/AbhikRao/Netrai>

---

## 1. System & Environment Context

- **Host OS**: Ubuntu 24.04 LTS (x86_64), Python 3.12
- **Hardware**: Dell Inspiron 5590 (Intel Core i5 10th Gen, 20GB RAM, CPU only, no local GPU)
- **Local Virtual Environment**: `/home/ovoabhik/Downloads/Larp/Netrai/python/.venv`
  - Activate: `source python/.venv/bin/activate`
  - PyTorch CPU (`torch 2.14.0+cpu`, `torchvision`), `timm`, `opencv-python-headless`, `albumentations`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `grad-cam`, `pandas`, `pillow`, `tqdm`.
- **Datasets**: APTOS 2019 (3,662 images), complete IDRiD, DRIVE public training/test packages, complete Messidor-2 images/pair metadata, and all 28,792 EyeQ/EyePACS images are normalized under `data/`. See `results/dataset_audit/dataset_audit.json`.
- **Authoritative backlog**: `task.md`. UI and presentation work are deferred; pipeline, validation, MATLAB/Simulink and safety evidence are active.
- **MathWorks runtime status**: MATLAB R2026a Update 5, Simulink, SimEvents, every requested toolbox, and the ONNX converter support package are installed and licensed. Evidence: `matlab/results/environment_2026-09-18.json`. The native six-output ONNX fixture passes (`7.63e-06` maximum error) and the end-to-end grade matches; the `1.0163` preprocessing-path logit difference remains an explicit warning. The full smoke and 733-image MATLAB holdout remain pending.

---

## 2. Project Architecture & File Inventory

```
Netrai/
├── README.md                          # Main project overview and documentation
├── AGENTS.md                          # Handover memory and agent context (this file)
├── netrai_kaggle_train.ipynb          # Kaggle training notebook (tested & verified)
├── data/
│   ├── README.md                      # Dataset acquisition, layout, licensing, hashes
│   ├── patient_sample.jpg             # High-res sample image
│   ├── aptos2019/                     # Local dataset: train.csv + train_images/
│   ├── idrid/                         # Complete grading/segmentation/localization release
│   ├── drive/                         # Annotated training + unannotated public test split
│   ├── messidor2/                     # Complete images; no official DR grades
│   └── eyeq/                          # Complete labels + local train/test images
├── matlab/                            # First-class MATLAB/Simulink deployment
│   ├── README_MATLAB.md               # Setup, chunked validation, limitations
│   ├── main_pipeline.m                # Real ONNX-backed pipeline
│   ├── validatePipeline.m             # CSV/JSON aggregate validator
│   ├── verifyMatlabPackage.m          # ONNX parity self-test
│   ├── captureMatlabEnvironment.m     # Installed-product evidence
│   ├── aggregateValidationRuns.m      # Merge free-session validation chunks
│   ├── models/                        # ONNX/config/fixture + executable M5 .slx
│   └── src/ (m1-m5 modules)
├── presentation/
│   └── Team Biscuit.pdf               # Presentation deck / template
├── python/                            # Core Python production pipeline
│   ├── .venv/                         # Fully configured virtual environment
│   ├── inference.py                   # Main pipeline CLI & clinical report generator
│   ├── batch_validate.py              # Batch validation + CSV/JSON metrics export
│   ├── calibrate_model.py             # Fitted scalar temperature calibration
│   ├── calibrate_quality.py           # EyeQ Good/Usable/Reject calibration runner
│   ├── audit_explainability.py        # Grade-stratified shortcut-attention QC
│   ├── build_validation_passport.py   # Release hashes + evidence/limits passport
│   ├── evaluate_lesion_masks.py       # Pixel/lesion metrics when masks are present
│   ├── evaluate_drive_vessels.py      # DRIVE vessel benchmark + bootstrap CIs
│   ├── evaluate_idrid_landmarks.py    # IDRiD OD/fovea localization evidence
│   ├── bootstrap_grading_metrics.py   # External grade metrics + bootstrap CIs
│   ├── analyze_selective_prediction.py # Coverage-risk/workload curve
│   ├── simulate_district_workflow.py  # Independent patient-level M5 reference
│   ├── audit_datasets.py              # Normalized dataset completeness evidence
│   ├── tests/test_pipeline.py         # 27 automated edge-case/invariant tests
│   ├── requirements.txt               # Pinned dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   └── netrai_model.py            # EfficientNet-B4 dual-head architecture
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── quality.py                 # M1: Image Quality Assessment (IQS)
│   │   ├── segmentation.py            # M2: OD, fovea, vessels, lesions, 12-D features
│   │   ├── explainability.py          # M4: Real Grad-CAM, overlays, evidence checklist
│   │   └── simulation.py              # M5: Telemedicine PHC workflow simulation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py           # Ben Graham (sigma=10), aspect-ratio crop
│   │   ├── calibration.py             # Temperature fitting/application helpers
│   │   ├── quality_calibration.py     # Ordered EyeQ threshold fitting
│   │   ├── safety.py                  # Four-way dual-head triage
│   │   └── metrics.py                 # QWK, sensitivity, specificity, AUC
│   └── weights/
│       ├── netrai_fold0_best.pth      # Trained weights (72MB, raw checkpoint QWK=0.9285)
│       ├── fold0_thresholds.npy       # Boundaries [0.5422, 1.4859, 2.5547, 3.3031]
│       ├── calibration.json           # Fitted T=0.839077 + evidence/limitation
│       └── safety_policy.json         # Versioned four-way routing policy
└── results/
    ├── validation_metrics.png         # Confusion matrix + ROC curve from training
    ├── verification_2026-09-18/       # Dual-head, 25-map QC, report, passport
    └── verification_2026-09-17/       # Original benchmark + calibration evidence
```

The user deleted the redundant dataset download archives after extraction.
Superseded `real_demo`
and `test_fixed` outputs plus the superseded M5 smoke sweep were moved recoverably to
`/home/ovoabhik/Downloads/Netrai_project_archive/legacy_results_2026-09-18/`.

---

## 3. Training & Validation Performance

Trained on Kaggle GPU (T4) using `netrai_kaggle_train.ipynb` over 30 epochs. The notebook defines a 5-fold stratified split but `train_folds: [0]`; the checked-in model is the best Fold-0 checkpoint:

| Metric | SIH Target | Achieved by NetrAI | Status |
|--------|------------|-------------------|--------|
| **Quadratic Weighted Kappa (QWK)** | > 0.85 | **0.9330** | 🏆 Exceeded |
| **Referable DR Sensitivity (Grade $\ge$ 2)** | > 90% | **94.63%** | 🏆 Exceeded |
| **Referable DR Specificity** | > 85% | **94.02%** | 🏆 Exceeded |
| **AUC-ROC (Referable DR)** | — | **0.9842** | 🏆 Outstanding |
| **Inference Latency** | < 30s | **~3–5s / image** (CPU) | 🏆 Real-time |

Weights are located at `python/weights/netrai_fold0_best.pth` and loaded automatically by `python/inference.py`.

The table above is the reproducible result from the current checked-in artifacts on all 733 Fold-0 validation images. The latest dual-head rerun is in `results/verification_2026-09-18/fold0_dual_head/`. Older handover figures (0.9362 QWK / 96.6% sensitivity) did not exactly match the checked-in threshold file and should not be quoted as current results.

The independent official IDRiD 103-image test split is the current release
blocker: QWK `0.6652`, referable sensitivity `79.69%` (95% bootstrap CI
`68.75–89.06%`), specificity `82.05%` (`69.23–92.31%`), and AUC `0.8746`
(`0.7985–0.9387`). It does not meet the problem target. Evidence is in
`results/verification_2026-09-18/idrid_external_grading/`; never substitute the
internal APTOS metric when discussing external performance.

Measured M2 baselines: DRIVE vessel sensitivity `0.4605`, specificity
`0.9096`, Dice `0.4405` after replacing the original `0.1736`-Dice Gabor
method; IDRiD optic-disc Dice `0.8042`; exudate/haemorrhage/MA Dice
`0.1869/0.0203/0.0135`. These are honest engineering baselines, not clinical
segmentation claims.

---

## 4. Work Completed in Latest Sessions

1. **Model Training & Integration**:
   - Resolved Kaggle dataset pathing and offline weights issues.
   - Successfully trained EfficientNet-B4 backbone with dual head (5-class grade + binary referable).
   - Exported model weights (`netrai_fold0_best.pth`), optimal thresholds (`fold0_thresholds.npy`), and validation plots (`validation_metrics.png`).
   - Wired real model weights into `inference.py` with real Grad-CAM (`model.backbone.conv_head`).
2. **Directory & Workspace Cleanup**:
   - Removed duplicate/temporary files, stale fake demo outputs, and unneeded scripts.
   - Recreated and verified clean Python 3.12 virtual environment (`python/.venv`).
3. **Pipeline Bug Fixes**:
   - **Double Calibration**: Removed duplicate temperature scaling in `explainability.py`.
   - **Ben Graham Sigma**: Aligned `preprocessing.py` sigma to 10 matching training.
   - **Color Space**: Fixed `COLOR_BGR2GRAY` to `COLOR_RGB2GRAY` in preprocessing.
   - **Adaptive Segmentation**: Added FOV/optic-disc exclusion, stricter MA candidates, conservative clustered NV detection, brightness-guided optic-disc localization, and local-contrast exudate detection.
   - **FOV NV Detection**: Scoped neovascularization density calculation strictly to the active fundus FOV.
   - **Model Class Modularity**: Moved `NetrAIModel` to `python/models/netrai_model.py`.
   - **M5 Simulation Module**: Implemented `python/modules/simulation.py` for PHC capacity, referral rate, and cost analysis.
   - **Report Formatting**: Cleaned up font rendering glyph warnings in Matplotlib reports.
4. **End-to-End Verification**:
   - Ran all 733 held-out Fold-0 images with zero errors (QWK 0.9330, sensitivity 94.63%, specificity 94.02%, AUC 0.9842).
   - Ran a balanced 25-image full-pipeline sample with all stages enabled and zero failures.
   - Rendered and visually inspected the refreshed Grade-2 PDF report with Poppler.
   - Added 27 automated tests for preprocessing, quality edge cases, segmentation invariants, metrics, calibration, dual-head triage, report safety fields, EyeQ fitting, lesion matching, M5 queue conservation, external bootstrap metrics, selective prediction, and Grad-CAM QC.
5. **MATLAB Deployment and Limitation Controls**:
   - Exported both trained heads as six outputs (five grade logits + one referable logit) to `matlab/models/netrai_fold0.onnx` (71 MB).
   - Verified PyTorch/ONNX Runtime parity on five real APTOS images (maximum absolute output error `5.25e-06`).
   - Removed fabricated MATLAB/Python grade and synthetic Grad-CAM fallbacks; failures are now explicit.
   - Rebuilt MATLAB batch validation and added chunk aggregation for MATLAB Online Basic's compute limit.
   - Added `python/evaluate_lesion_masks.py` for pixel- and lesion-level validation when annotated masks are supplied.
   - Added Grad-CAM border/FOV/corner QC; the 25-image grade-stratified audit completed 25/25 and flagged 13/25. All flagged overlays are saved. Reports surface the flag. True mitigation requires retraining, not an inference-only preprocessing change.
   - MATLAB R2026a M2 and SimEvents runtime execution pass, MISS_HIT structural lint passes 41/41 files, and the ONNX converter is installed. The six-output fixture passes at `7.63e-06`; the end-to-end grade passes with a `1.0163` preprocessing-logit warning. Full style mode reports legacy formatting debt; the full smoke and fixed MATLAB holdout are still pending.
   - M5 now has a generated executable `matlab/models/NetrAI_Telemedicine_Model.slx`, 18 annual 100,000-patient scenarios, and an independent patient-level Python FCFS reference. All 18 rate comparisons pass at 1.5 percentage-point tolerance. The cross-check exposed and removed a misleading 10,000-entity buffer cap; the corrected 0.25 Mbps backlog is 47,857–62,974 uploads.
6. **Measured Open Risks (do not obscure these with the strong referable endpoint)**:
   - All 733 Fold-0 images currently score below the nominal M1 “good” IQS threshold (`0.7`), demonstrating that quality thresholds require EyeQ calibration.
   - The stratified Grad-CAM audit flags 13/25 maps: G0 20%, G1 40%, G2 40%, G3 60%, G4 100%. This heuristic signal requires visual/clinical review and mitigation/re-validation.
   - Per-grade recall is G0 `99.72%`, G1 `63.51%`, G2 `72.00%`, G3 `58.97%`, G4 `59.32%`; the system is a stronger referable screener than a five-grade classifier.
7. **Confidence Calibration (completed 2026-09-17)**:
   - Replaced hard-coded `T=1.5` with a versioned fitted scalar temperature `T=0.839077` in `python/weights/calibration.json` and MATLAB config.
   - Used a reproducible, grade-stratified 366/367 split of the stored Fold-0 outputs. On the evaluation half, deployed-policy ECE improved `0.0459 -> 0.0258`, NLL `0.3932 -> 0.3837`, and Brier `0.2101 -> 0.2079`; grade decisions do not change.
   - Evidence: `results/verification_2026-09-17/calibration/`. A regenerated Grade-2 report was visually verified in `results/verification_2026-09-17/calibrated_report_smoke/`.
   - This remains post-hoc evidence because Fold 0 was also used for checkpoint selection and threshold optimization. Do not call it external calibration.
8. **Dual-Head Safety and Four-Way Triage (completed 2026-09-18)**:
   - Python, ONNX and MATLAB schemas retain the independent binary referable head instead of discarding it.
   - Full Fold-0 evidence: binary-head sensitivity `92.95%`, specificity `94.94%`, AUC `0.9818`; 9/733 (`1.23%`) disagree with the ordinal grade policy.
   - Every disagreement maps to `uncertain_human_review`; concordant outputs cover `98.77%` with sensitivity `94.54%` and specificity `94.90%`. The policy is explicitly research-only pending external validation.
   - A single-image report integration bug involving string triage values was found by the report smoke test, fixed, and covered by a regression test. The regenerated report was visually inspected.
9. **Dataset Normalization and External Validation (2026-09-18)**:
   - Complete IDRiD, DRIVE, Messidor-2, and all 28,792 EyeQ/EyePACS images are normalized and checksummed. Redundant download archives were deleted after extraction. `audit_datasets.py` records exact completeness.
   - `calibrate_quality.py` now targets the real `images/train` and `images/test` layout. Calibration remains the next M1 evidence run; no EyeQ-derived thresholds are claimed yet.
   - The official IDRiD test split was run as an independent Indian-domain grading check with 2,000 bootstrap replicates; its 79.69%/82.05% sensitivity/specificity fails the required endpoint.
   - IDRiD lesion/landmark and DRIVE vessel benchmarks now quantify M2. The vessel method improved materially, but MA/haemorrhage/exudate candidates remain inadequate.
   - Quadratic sub-pixel MA peak refinement is implemented in Python and MATLAB and synthetic-tested; it remains analysis-grid localization, not native-resolution or FROC validation.
   - `data/README.md` records the EyeQ, IDRiD, DRIVE and external-grading acquisition order, layouts, licensing cautions and hash procedure.
10. **Validation Passport v2 (completed 2026-09-18)**:
   - `build_validation_passport.py` hashes eight release artifacts plus dataset/evidence indexes, then binds them to internal/external grading, M2, calibration, attention QC, selective prediction, MATLAB environment, M5 cross-check, intended/prohibited use and known limitations.
   - Current artifact: `results/verification_2026-09-18/validation_passport.json`. It is traceability evidence, not clinical validation or regulatory certification.
11. **Selective prediction (2026-09-18)**:
   - `analyze_selective_prediction.py` exports coverage, human-review workload, retained endpoint metrics and false decisions for 20 confidence thresholds after mandatory dual-head disagreement review.
   - The 0.75 confidence row is illustrative only: 70.94% coverage, 29.06% workload, 94.21% retained sensitivity and 98.50% specificity. It is post-hoc internal evidence, not a deployment threshold.
12. **Workspace cleanup (2026-09-18)**:
   - Dataset ZIPs/split archives were removed by the user after verified extraction; legacy result/smoke duplicates remain under `/home/ovoabhik/Downloads/Netrai_project_archive/legacy_results_2026-09-18/`; EyeQ IDE, Python bytecode and Simulink build caches were trashed.

---

## 5. How to Run the Pipeline

```bash
cd /home/ovoabhik/Downloads/Larp/Netrai/python
source .venv/bin/activate

# 1. Run inference on a specific fundus image:
python3 inference.py --image ../data/aptos2019/train_images/000c1434d8d7.png --output ../results/demo_run

# 2. Run demo on default dataset samples (Grade 0, 2, 4):
python3 inference.py --demo --output ../results/demo_run

# 3. Run/cross-check the independent 100,000-patient M5 reference:
python3 simulate_district_workflow.py \
  --simulink-csv ../matlab/results/m5_scenario_sweep/simulink_scenario_sweep.csv

# 4. MATLAB: open matlab/ and run the installed ONNX deployment verifier:
# report = verifyMatlabPackage();
# demo_netrai;

# 5. Rebuild the current evidence passport:
python3 build_validation_passport.py

# 6. After the complete EyeQ layout in data/README.md is present:
python3 calibrate_quality.py

# 7. Rebuild internal abstention/workload evidence:
python3 analyze_selective_prediction.py
```

---

## 6. Planned Tasks & Roadmap

`task.md` is the sole authoritative backlog and must remain priority ordered. `README.md`, `AGENTS.md`, and `task.md` must be synchronized in every batch that changes project behavior or evidence.

Current execution order:

1. **P0 correctness**: execute MATLAB parity, full M1–M4 smoke and the 733-image holdout now that the ONNX converter is installed. M5 `.slx` runtime and corrected annual cross-check evidence are complete.
2. **Four measured weaknesses**: calibrate M1 on the complete EyeQ split; retrain for failed IDRiD generalization/minority grades and Grad-CAM shortcut risk; replace weak lesion candidates; add native-resolution MA FROC.
3. **Standout system**: four-way triage, Validation Passport and coverage-risk evidence are implemented; next add clinical concordance, store-and-forward compression, timed M5 escalation, failure galleries, and reviewer evidence.
4. **Clinical rigor**: external patient-level metrics with bootstrap CIs, leakage audit, integrated ablation, source-verified benchmark table, and a clinician usefulness/under-30-second review study.
5. **Deferred at user request**: interactive UI and presentation construction.

No required dataset acquisition remains for current benchmarks. Messidor-2
remains image-only and cannot support a grading claim without a lawful,
separately verified label source; never use an unverified internet CSV.
