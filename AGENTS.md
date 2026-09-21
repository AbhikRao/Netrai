# NetrAI — Agent Memory & Project Handover Context

> **SIH 2026 | Problem Statement 26038 | Team Biscuit | MathWorks Sponsored**  
> **Topic**: Explainable AI for Diabetic Retinopathy Screening in Rural India  
> **Submission Deadline**: September 30, 2026  
> **Workspace**: `/home/ovoabhik/Downloads/Larp/Netrai/`
> **Public repository**: <https://github.com/AbhikRao/Netrai>

## Active instruction — pipeline implementation resumed (2026-09-21)

The user approved implementation of the priority plan. Presentation remains
user-owned and the interactive UI remains deferred. Continue evidence-first
pipeline work, synchronize `task.md`, `README.md`, and this file after material
changes, and never promote a candidate merely because one aggregate gate passes.

Latest completed M1 work:

- `python/models/quality_model.py` and `python/train_quality_model.py` now implement a frozen ImageNet MobileNetV3-Small encoder, exact full-frame 224-pixel float bilinear input, resumable integrity-bound embeddings, and a balanced three-class EyeQ head.
- All 12,543 training and 16,249 test embeddings are complete. Official train/test have 8,614/12,048 filename patients with zero overlap. Internal fit/selection/calibration are 7,527/2,508/2,508 images and 5,159/1,736/1,719 patients, with both eyes grouped and no overlap.
- Exact encoded-file SHA-256 audit finds no duplicates within or across EyeQ official splits. Perceptual near-duplicates and cross-dataset content duplication remain open.
- Regularization is selected on selection patients; scalar temperature and a conservative Reject threshold are selected on separate calibration patients. The threshold requires the one-sided 95% Wilson lower confidence bound for calibration Reject recall to be at least 0.80.
- Complete official-test results: macro-F1 `0.8289`, balanced accuracy `0.8299`, Reject recall `86.37%` (patient-bootstrap 95% CI `85.19–87.56%`), and Good/Usable false-reject rate `4.63%`. This passes the engineering gate but is development evidence because EyeQ test outcomes have already informed the project.
- `matlab/models/quality_model_candidate.onnx` is 3.6 MB. The fixed real-image MATLAB fixture has exact input parity, the same decision, and maximum logit error `2.65e-05`; evidence is in `matlab/results/quality_model_candidate_parity.json`.
- The 250-image paired stress audit remains a promotion blocker: hard-reject rates are defocus `0.4%`, low light `7.6%`, uneven illumination `6.4%`, FOV shift `1.6%`, and JPEG Q25 `0%`. The candidate stays non-default pending factor-specific augmentation/fine-tuning and portable-camera validation.
- Validation Passport v4 hashes and records the candidate checkpoint/config/ONNX/parity evidence, split and exact-duplicate audits, global performance, corruption failures, and non-default decision.

EyeQ/IDRiD official test results have already been examined during development;
future reports must disclose this exposure and reserve fresh evidence for final
generalization claims.

---

## 1. System & Environment Context

- **Host OS**: Ubuntu 24.04 LTS (x86_64), Python 3.12
- **Hardware**: Dell Inspiron 5590 (Intel Core i5 10th Gen, 20GB RAM, CPU only, no local GPU)
- **Local Virtual Environment**: `/home/ovoabhik/Downloads/Larp/Netrai/python/.venv`
  - Activate: `source python/.venv/bin/activate`
  - PyTorch CPU (`torch 2.14.0+cpu`, `torchvision`), `timm`, `opencv-python-headless`, `albumentations`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `grad-cam`, `pandas`, `pillow`, `tqdm`.
- **Datasets**: APTOS 2019 (3,662 images), complete IDRiD, DRIVE public training/test packages, complete Messidor-2 images/pair metadata, and all 28,792 EyeQ/EyePACS images are normalized under `data/`. See `results/dataset_audit/dataset_audit.json`.
- **EyeQ caches**: completed handcrafted features are under `results/verification_2026-09-20/eyeq_quality_calibration/`; learned 576-D embeddings are under `results/verification_2026-09-21/eyeq_learned_quality/cache/`. Both caches are local/Git-ignored; aggregate evidence and hashes are shareable.
- **Authoritative backlog**: `task.md`. Its active six-stage plan records completed learned-M1 work and the remaining promotion gates. UI remains deferred and presentation editing is user-owned.
- **Presentation status**: `presentation/Netrai_SIH_2026_Final.pptx` and `.pdf` are visually reviewed drafts. Do not modify them unless the user requests presentation work again.
- **MathWorks runtime status**: MATLAB R2026a Update 5, Simulink, SimEvents, every requested toolbox, and the ONNX converter support package are installed and licensed. Evidence: `matlab/results/environment_2026-09-18.json`. The native six-output ONNX fixture passes (`7.63e-06` maximum raw-output error). The aligned MATLAB holdout completes 733/733 with zero failures (QWK `0.9297`, sensitivity `94.97%`, specificity `93.33%`, AUC `0.9841`, `1.55 s/image`). Cross-runtime grades match 717/733 and triage matches 724/733; exact screening, grade, and probability-tolerance gates remain failed and must not be described as parity.

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
│   ├── Netrai_SIH_2026_Final.pptx      # Editable official-format SIH deck
│   ├── Netrai_SIH_2026_Final.pdf       # Six-page portal submission render
│   ├── SIH2026-IDEA-Presentation-Format.pptx # Official SIH template
│   └── Team Biscuit.pdf                # Previous-round reference deck
├── python/                            # Core Python production pipeline
│   ├── .venv/                         # Fully configured virtual environment
│   ├── inference.py                   # Main pipeline CLI & clinical report generator
│   ├── batch_validate.py              # Batch validation + CSV/JSON metrics export
│   ├── calibrate_model.py             # Fitted scalar temperature calibration
│   ├── calibrate_quality.py           # EyeQ Good/Usable/Reject calibration runner
│   ├── audit_quality_robustness.py    # Synthetic M1 corruption stress audit
│   ├── audit_learned_quality_robustness.py # Learned-candidate paired stress audit
│   ├── audit_eyeq_duplicates.py       # Official-split exact-file duplicate audit
│   ├── train_quality_model.py         # Patient-disjoint frozen-encoder EyeQ experiment
│   ├── export_quality_model.py        # Gated ONNX + MATLAB fixture export
│   ├── compare_quality_baselines.py   # Non-deployed 4-feature model ablation
│   ├── audit_explainability.py        # Grade-stratified shortcut-attention QC
│   ├── build_validation_passport.py   # Release hashes + evidence/limits passport
│   ├── evaluate_lesion_masks.py       # Pixel/lesion metrics when masks are present
│   ├── evaluate_drive_vessels.py      # DRIVE vessel benchmark + bootstrap CIs
│   ├── evaluate_idrid_landmarks.py    # IDRiD OD/fovea localization evidence
│   ├── bootstrap_grading_metrics.py   # External grade metrics + bootstrap CIs
│   ├── analyze_selective_prediction.py # Coverage-risk/workload curve
│   ├── simulate_district_workflow.py  # Independent patient-level M5 reference
│   ├── audit_datasets.py              # Normalized dataset completeness evidence
│   ├── compare_runtime_outputs.py     # Layered MATLAB/Python parity audit
│   ├── tests/                         # 45 automated edge-case/invariant tests
│   ├── requirements.txt               # Pinned dependencies
│   ├── models/
│   │   ├── __init__.py
│   │   ├── netrai_model.py            # EfficientNet-B4 dual-head architecture
│   │   └── quality_model.py           # MobileNetV3-Small M1 candidate + input contract
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
│       ├── quality_thresholds_eyeq.json # Evidence-only failed EyeQ baseline; not deployed
│       └── safety_policy.json         # Versioned four-way routing policy
└── results/
    ├── validation_metrics.png         # Confusion matrix + ROC curve from training
    ├── verification_2026-09-17/       # Original benchmark + calibration evidence
    ├── verification_2026-09-18/       # Dual-head, external, M2, M5, 25-map QC
    └── verification_2026-09-20/       # EyeQ, MATLAB full holdout/parity, passport v3
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
   - All 45 automated tests pass across preprocessing, quality edge cases, segmentation invariants, metrics, calibration, dual-head triage, report safety fields, EyeQ fitting/cache resumption/atomic writes/corruptions, patient-disjoint learned-M1 selection/calibration, feature baselines, lesion matching, runtime comparison, M5 queue conservation, external bootstrap metrics, selective prediction, and Grad-CAM QC.
5. **MATLAB Deployment and Limitation Controls**:
   - Exported both trained heads as six outputs (five grade logits + one referable logit) to `matlab/models/netrai_fold0.onnx` (71 MB).
   - Verified PyTorch/ONNX Runtime parity on five real APTOS images (maximum absolute output error `5.25e-06`).
   - Removed fabricated MATLAB/Python grade and synthetic Grad-CAM fallbacks; failures are now explicit.
   - Rebuilt MATLAB batch validation and added chunk aggregation for MATLAB Online Basic's compute limit.
   - Added `python/evaluate_lesion_masks.py` for pixel- and lesion-level validation when annotated masks are supplied.
   - Added Grad-CAM border/FOV/corner QC; the 25-image grade-stratified audit completed 25/25 and flagged 13/25. All flagged overlays are saved. Reports surface the flag. True mitigation requires retraining, not an inference-only preprocessing change.
   - MATLAB R2026a M2 and SimEvents runtime execution pass, the existing MATLAB structural checks pass, and the ONNX converter is installed. The six-output fixture passes at `7.63e-06`; preprocessing alignment lowers its maximum end-to-end logit difference from `1.0163` to `0.2210`.
   - The aligned fixed holdout completes 733/733 with zero errors: QWK `0.9297`, sensitivity `94.97%`, specificity `93.33%`, AUC `0.9841`, and `1.55 s/image`. The layered comparator reports 717/733 grade matches, 729/733 grade-referable matches, 727/733 independent-head matches, and 724/733 triage matches. All three exact/numerical parity gates remain failed; the MATLAB deployment works but is not bit/decision interchangeable with Python.
   - M5 now has a generated executable `matlab/models/NetrAI_Telemedicine_Model.slx`, 18 annual 100,000-patient scenarios, and an independent patient-level Python FCFS reference. All 18 rate comparisons pass at 1.5 percentage-point tolerance. The cross-check exposed and removed a misleading 10,000-entity buffer cap; the corrected 0.25 Mbps backlog is 47,857–62,974 uploads.
6. **Measured Open Risks (do not obscure these with the strong referable endpoint)**:
   - The completed EyeQ scalar-IQS baseline fails its held-out gate: test macro-F1 `0.3976`, balanced accuracy `0.4101`, and Reject recall `25.09%`. Its fitted artifact is evidence-only and not deployed.
   - Synthetic paired stress testing on 250 Good EyeQ images catches severe FOV misalignment (99.2% Reject) but misses defocus blur, low light, and JPEG Q25 (2.4% Reject each). This is not real portable-camera validation.
   - The learned MobileNetV3-Small candidate passes the global EyeQ gate (macro-F1 `0.8289`, Reject recall `86.37%`) and fixed MATLAB parity, but remains non-default because the paired stress audit hard-rejects only `0.4%` defocus, `7.6%` low light, `6.4%` uneven illumination, `1.6%` FOV shift, and `0%` JPEG Q25.
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
   - `calibrate_quality.py` processed all 12,543 train and 16,249 official test images with safely parallel resumable caches and zero failed rows. The scalar threshold baseline failed the engineering gate and remains undeployed. `load_iqs_thresholds` rejects any non-`deployed` artifact by default; diagnostic scripts must explicitly opt into evidence-only access.
   - Factor association analysis finds illumination most useful for Reject labels (test AUC `0.7043`), while focus and the scalar FOV measure are not valid factor classifiers. EyeQ has no factor-specific labels.
   - Four balanced handcrafted-feature models were evaluated. HistGradientBoosting is best at macro-F1 `0.5853` and Reject recall `55.81%`, still below the gate. The completed frozen-encoder image model improves global performance materially; its next step is factor-specific augmentation/fine-tuning and portable-camera validation, not default integration.
   - The official IDRiD test split was run as an independent Indian-domain grading check with 2,000 bootstrap replicates; its 79.69%/82.05% sensitivity/specificity fails the required endpoint.
   - IDRiD lesion/landmark and DRIVE vessel benchmarks now quantify M2. The vessel method improved materially, but MA/haemorrhage/exudate candidates remain inadequate.
   - Quadratic sub-pixel MA peak refinement is implemented in Python and MATLAB and synthetic-tested; it remains analysis-grid localization, not native-resolution or FROC validation.
   - `data/README.md` records the EyeQ, IDRiD, DRIVE and external-grading acquisition order, layouts, licensing cautions and hash procedure.
10. **Validation Passport v4 (completed 2026-09-21)**:
   - `build_validation_passport.py` retains the complete release/data evidence and additionally hashes the learned-M1 checkpoint, config, ONNX/config, parity reference and MATLAB result; it binds these to the patient split, exact-duplicate audit, complete EyeQ evaluation, corruption failures and non-default promotion decision.
   - Current artifact: `results/verification_2026-09-20/validation_passport.json`. It is traceability evidence, not clinical validation or regulatory certification.
11. **Selective prediction (2026-09-18)**:
   - `analyze_selective_prediction.py` exports coverage, human-review workload, retained endpoint metrics and false decisions for 20 confidence thresholds after mandatory dual-head disagreement review.
   - The 0.75 confidence row is illustrative only: 70.94% coverage, 29.06% workload, 94.21% retained sensitivity and 98.50% specificity. It is post-hoc internal evidence, not a deployment threshold.
12. **Workspace cleanup (2026-09-18)**:
   - Dataset ZIPs/split archives were removed by the user after verified extraction; legacy result/smoke duplicates remain under `/home/ovoabhik/Downloads/Netrai_project_archive/legacy_results_2026-09-18/`; EyeQ IDE, Python bytecode and Simulink build caches were trashed.
13. **Public source release (2026-09-19)**:
   - Published <https://github.com/AbhikRao/Netrai> as a public, source-only repository. Datasets, credentials, the local virtual environment, presentation drafts, generated patient-image derivatives, and MATLAB import caches are excluded; the checkpoint and ONNX model are included for reproducibility.

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

# 6. Reproduce the complete EyeQ baseline (the current artifact fails its gate):
python3 calibrate_quality.py \
  --output ../results/verification_2026-09-20/eyeq_quality_calibration

# 7. Rebuild internal abstention/workload evidence:
python3 analyze_selective_prediction.py
```

---

## 6. Planned Tasks & Roadmap

`task.md` is the sole authoritative backlog and must remain priority ordered. `README.md`, `AGENTS.md`, and `task.md` must be synchronized in every batch that changes project behavior or evidence.

Active execution order:

1. **M1 protocol/evaluation complete for the frozen candidate**: patient-grouped fit/selection/calibration, exact-file duplicate audit, fixed gate, patient-bootstrap intervals, retention and corruption response are recorded. Perceptual/cross-dataset duplication and fresh final evidence remain open.
2. **M1 promotion blockers**: prepare factor-specific augmentation/fine-tuning, broader synthetic edge-case parity, and real portable-camera validation. Do not enable the learned candidate by default until these checks justify it.
3. **MATLAB consistency**: M1 fixed-fixture parity passes. Resolve the grade model's existing 16 grade / 9 triage mismatches through stagewise preprocessing checks and rerun the fixed 733 rows. Never weaken parity gates to claim success.
4. **M3/M4 together**: evaluate class balancing, ordinal loss and retinal-field/background augmentation; refit calibration and repeat external grading, per-grade recall and fixed Grad-CAM audits. Keep IDRiD test labels out of fitting and threshold selection.
5. **M2 validity**: prioritize native-resolution MA/haemorrhage improvements and MA FROC, then exudates/vessels; keep annotation-based splits and separate evidence gaps for NV/haemorrhage subtypes.
6. **Integration and evidence**: run same-cohort ablations including rejected/abstained images, review workload and missed referrals; add store-and-forward/compression and timed M5 escalation; refresh reports/passport/tests and run a clinician explanation/review-time study when reviewers are available.

The detailed checklist and completion evidence for each stage are in `task.md`.
Further presentation work stays with the user; UI remains deferred.

No required dataset acquisition remains for current benchmarks. Messidor-2
remains image-only and cannot support a grading claim without a lawful,
separately verified label source; never use an unverified internet CSV.
