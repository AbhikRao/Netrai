# 🔬 NetrAI — Explainable AI for Diabetic Retinopathy Screening

> **SIH 2026 | Problem Statement 26038 | Team Biscuit | MathWorks Sponsored**  
> *Automated, explainable, validated DR screening for rural India*

Public repository: <https://github.com/AbhikRao/Netrai>

---

## 📌 Problem

India has **77 million diabetic adults** — the 2nd highest globally. Diabetic Retinopathy (DR) affects ~18% of this population and is a **leading cause of preventable blindness**. Early screening can prevent 90% of vision loss, but India has only ~1 ophthalmologist per 100,000 rural population, making mass manual screening infeasible.

## 💡 Solution

NetrAI is a **5-module research pipeline** that analyzes a retinal fundus photograph and produces an explainable screening report. The internal grading benchmark is reproducible and the SimEvents district digital twin now runs, but independent IDRiD testing shows the system is not yet clinically validated:

| Module | Function | Output | Status |
|---|---|---|---|
| **M1** Image Quality Assessment | Evaluates focus, illumination, FOV coverage | IQS Score + Recapture Guidance | ⚠️ All 28,792 EyeQ/EyePACS images are ready; threshold calibration is the next run |
| **M2** Retinal Segmentation | Detects optic disc, fovea, vessels, lesions | 12-D Clinical Feature Vector | ⚠️ Benchmarked on IDRiD/DRIVE; OD is promising, vessel/lesion candidates remain below clinical quality |
| **M3** DR Severity Grading | EfficientNet-B4 dual head + ordinal thresholds | ICDR Grade 0–4 + Referable DR + four-way triage | ✅ Trained (**QWK = 0.9330**) and dual-head safety verified locally |
| **M4** Explainability (CAFE) | Real Grad-CAM attention + evidence checklist | 6-Panel Clinical Report (PNG + PDF)| ⚠️ Working; 13/25 stratified maps received shortcut-attention QC flags |
| **M5** Telemedicine Simulation | PHC workflow modeling & capacity planning | Throughput, queues, waits, recapture load, staffing | ✅ Executable `.slx`, 18-scenario 100,000-patient sweep, independent Python reference |

---

## 🎯 Benchmark Validation Results

The model was trained on the **APTOS 2019 Blindness Detection** dataset (3,662 images). The notebook creates a 5-fold stratified split and the checked-in checkpoint is the trained Fold-0 model:

| Metric | SIH Target | Achieved by NetrAI | Status |
|---|---|---|---|
| **Quadratic Weighted Kappa (QWK)** | > 0.85 | **0.9330** | 🏆 **Exceeded** |
| **Referable DR Sensitivity (Grade $\ge$ 2)** | > 90.0% | **94.63%** | 🏆 **Exceeded** |
| **Referable DR Specificity** | > 85.0% | **94.02%** | 🏆 **Exceeded** |
| **AUC-ROC (Referable DR)** | — | **0.9842** | 🏆 **Outstanding** |
| **Inference Time (CPU)** | < 30s | **~3–5 seconds** | 🏆 **Real-time** |

These figures were reproduced on all 733 images in held-out Fold 0 using the checked-in checkpoint and thresholds. The latest machine-readable rerun is in `results/verification_2026-09-18/fold0_dual_head/`; the original benchmark evidence remains in `results/verification_2026-09-17/fold0_model/`.

The primary endpoint is strong, but it must not be confused with complete clinical validation. Current grade recalls are G0 99.72%, G1 63.51%, G2 72.00%, G3 58.97%, and G4 59.32%; threshold optimization and checkpoint selection also used Fold 0. Independent external validation is therefore a blocking task.

### Independent Indian-domain check (IDRiD)

The official 103-image IDRiD test split was evaluated without tuning the model
or ordinal thresholds. It achieved QWK **0.6652**, referable sensitivity
**79.69%** (95% bootstrap CI **68.75–89.06%**), specificity **82.05%**
(**69.23–92.31%**), and AUC **0.8746** (**0.7985–0.9387**). This fails the
problem's >90% sensitivity and >85% specificity endpoint. The result is kept as
a release blocker, not hidden behind the stronger APTOS Fold-0 number. Evidence
is in `results/verification_2026-09-18/idrid_external_grading/`.

IDRiD/DRIVE also converted M2's former qualitative outputs into measured
baselines. Optic-disc segmentation reaches Dice **0.8042**; the revised
multi-scale line vessel detector improves DRIVE Dice from **0.1736** to
**0.4405** (sensitivity **0.4605**, specificity **0.9096**). Exudate Dice is
**0.1869**, haemorrhage Dice **0.0203**, and microaneurysm Dice **0.0135**.
These candidates are useful for explainability experiments, but are not yet
clinically meaningful lesion segmentation.

### Confidence calibration evidence

The former arbitrary temperature `1.5` has been removed. A reproducible, grade-stratified 366/367 calibration/evaluation split fits `T=0.839077`; on the disjoint evaluation half, deployed-policy ECE improves from **4.59% to 2.58%**, NLL from **0.3932 to 0.3837**, and multiclass Brier from **0.2101 to 0.2079**, without changing grade decisions. The versioned artifact is `python/weights/calibration.json`, and the split manifest, metrics, reliability bins, and diagram are in `results/verification_2026-09-17/calibration/`.

This is post-hoc Fold-0 evidence, not independent external or prospective clinical validation, because Fold 0 also informed checkpoint selection and ordinal thresholds.

### Dual-head safety and abstention evidence

The MATLAB/ONNX and Python deployments now retain both trained outputs: five grade logits and the independent referable-DR logit. On all 733 Fold-0 images, the grade-derived policy retains **94.63% sensitivity / 94.02% specificity**; the independent head obtains **92.95% / 94.94%**. They disagree on **9/733 images (1.23%)**. Those cases are routed to `uncertain_human_review`; neither head is silently preferred. Concordant coverage is **98.77%** with **94.54% sensitivity / 94.90% specificity** on the retained set. Evidence is in `results/verification_2026-09-18/fold0_dual_head/`.

This four-way research policy produces `auto_clear`, `refer`, `recapture`, or `uncertain_human_review`. Its `0.5` binary-head threshold must still be validated externally before clinical use.

### Selective-prediction workload evidence

`python/analyze_selective_prediction.py` turns abstention into an auditable
coverage/workload curve instead of a vague “low confidence” promise. On the
internal Fold-0 evidence, mandatory head-disagreement review alone retains
98.77% coverage. As an illustrative, post-hoc operating point, requiring 0.75
calibrated grade confidence retains 70.94% for automatic decisions and routes
29.06% to review, with retained sensitivity 94.21%, specificity 98.50%, and a
5.19% five-grade error rate. This threshold has not been externally selected
and is not a deployment recommendation.

### Auditable release evidence

`python/build_validation_passport.py` generates Validation Passport v2 at
`results/verification_2026-09-18/validation_passport.json`. It fingerprints
eight release artifacts and binds them to internal/external grading, M2,
calibration, Grad-CAM, selective-prediction, MATLAB-environment and M5
cross-check evidence, intended/prohibited use, and known limitations. It is a
traceability artifact—not a regulatory certificate.

### MATLAB/SimEvents status

MATLAB R2026a Update 5, Simulink, SimEvents, and every requested toolbox are
installed and runtime-tested. `matlab/models/NetrAI_Telemedicine_Model.slx`
models acquisition, recapture, bandwidth upload, AI inference, four-way
routing, and clinician review for 100,000 patients/year. All 18 annual
scenarios pass the independent patient-level Python FCFS rate cross-check at a
1.5 percentage-point tolerance. That cross-check found and fixed an artificial
10,000-entity queue cap that had hidden the true low-bandwidth backlog:
0.25 Mbps leaves 47,857–62,974 uploads queued, while 1 Mbps or more completes
at least 99.997% of arrivals under the stated assumptions.

The free **Deep Learning Toolbox Converter for ONNX Model Format** support
package is now installed. The native six-output ONNX fixture passes with
maximum error `7.63e-06`, and the end-to-end grade matches; a `1.0163`
preprocessing-path logit difference remains flagged for alignment. Environment
evidence is kept in `matlab/results/environment_2026-09-18.json`; the small
full-pipeline smoke and fixed 733-image MATLAB holdout remain pending.

## Current Priority Roadmap

[`task.md`](task.md) is the authoritative, priority-ordered backlog. Work currently proceeds in this order:

1. Execute MATLAB ONNX parity/full-pipeline smoke, then run the fixed 733-image MATLAB holdout.
2. Calibrate M1 on the complete EyeQ/EyePACS train/test split. Extend MA detection to native resolution and FROC, and replace weak lesion candidates with trained/validated models.
3. Retrain for IDRiD generalization, minority-grade recall, and measured Grad-CAM shortcut risk; the current external IDRiD result fails the target.
4. Add a clinical-concordance gate, store-and-forward compression tests, timed SimEvents escalation, passport evidence, and a clinician under-30-second review study.

UI and presentation work are intentionally deferred while this evidence is being built.

The user deleted the redundant dataset download archives after extraction.
Normalized datasets and checksums remain local and are excluded from Git;
superseded demo/smoke outputs remain recoverably archived outside the project.

---

## 🏗️ Project Structure

```
Netrai/
├── README.md                          # Main project overview
├── AGENTS.md                          # Comprehensive agent memory & handover context
├── netrai_kaggle_train.ipynb          # Full training notebook for Kaggle GPU
├── data/
│   ├── README.md                      # Dataset sources, layouts, licenses, integrity steps
│   ├── patient_sample.jpg             # Reference sample image
│   ├── aptos2019/                     # 3,662 images + train.csv
│   ├── idrid/                         # Complete grading/segmentation/localization release
│   ├── drive/                         # 20 annotated training + 20 unannotated test images
│   ├── messidor2/                     # 1,748 images; no official DR labels
│   └── eyeq/                          # Complete labels + 28,792 local EyePACS images
├── presentation/
│   └── Team Biscuit.pdf               # Presentation deck / template
├── matlab/                            # Native MATLAB + exported ONNX deployment
│   ├── README_MATLAB.md               # MATLAB/Simulink install and run guide
│   ├── main_pipeline.m                # Real ONNX-backed end-to-end pipeline
│   ├── validatePipeline.m             # Metrics + CSV/JSON batch validator
│   ├── verifyMatlabPackage.m          # MATLAB/ONNX parity self-test
│   ├── captureMatlabEnvironment.m     # Installed-product/runtime evidence
│   ├── models/                        # ONNX, thresholds/config, parity fixture
│   │   └── NetrAI_Telemedicine_Model.slx # Executable SimEvents digital twin
│   └── src/ (m1-m5 modules)
├── python/                            # Production AI pipeline
│   ├── .venv/                         # Configured Python 3.12 venv
│   ├── inference.py                   # Main CLI & report generation entry point
│   ├── batch_validate.py              # Batch CSV/metrics validation runner
│   ├── calibrate_model.py             # Reproducible temperature fitting/evaluation
│   ├── calibrate_quality.py           # EyeQ train/test IQS threshold calibration
│   ├── audit_explainability.py        # Border/FOV Grad-CAM quality audit
│   ├── build_validation_passport.py   # Artifact hashes + evidence/limitation passport
│   ├── audit_datasets.py              # Dataset completeness/integrity audit
│   ├── evaluate_lesion_masks.py       # IDRiD pixel/lesion bootstrap metrics
│   ├── evaluate_drive_vessels.py      # DRIVE vessel bootstrap metrics
│   ├── evaluate_idrid_landmarks.py    # OD/fovea localization error evidence
│   ├── bootstrap_grading_metrics.py   # External grading 95% confidence intervals
│   ├── analyze_selective_prediction.py # Coverage-risk/workload evidence
│   ├── simulate_district_workflow.py  # Independent patient-level M5 reference
│   ├── export_matlab_model.py         # PyTorch checkpoint -> MATLAB ONNX
│   ├── tests/test_pipeline.py         # Automated pipeline/edge-case checks
│   ├── requirements.txt               # Dependencies (torch, timm, grad-cam, albumentations...)
│   ├── models/
│   │   └── netrai_model.py            # EfficientNet-B4 dual-head architecture
│   ├── modules/
│   │   ├── quality.py                 # M1: IQS assessment
│   │   ├── segmentation.py            # M2: Retinal structure & lesion detection
│   │   ├── explainability.py          # M4: Real Grad-CAM & clinical checklist
│   │   └── simulation.py              # M5: Telemedicine deployment simulation
│   ├── utils/
│   │   ├── preprocessing.py           # Ben Graham preprocessing (sigma=10)
│   │   ├── quality_calibration.py     # Ordered EyeQ threshold fitting
│   │   ├── safety.py                  # Four-way dual-head triage policy
│   │   └── metrics.py                 # Evaluation metrics
│   └── weights/
│       ├── netrai_fold0_best.pth      # Best model checkpoint (72 MB)
│       ├── fold0_thresholds.npy       # Optimal ordinal decision boundaries
│       ├── calibration.json           # Versioned fitted confidence temperature
│       └── safety_policy.json         # Versioned four-way triage policy
└── results/
    ├── validation_metrics.png         # Confusion matrix & ROC curve
    ├── verification_2026-09-17/       # Original benchmark + calibration evidence
    └── verification_2026-09-18/       # Dual-head, external, M2, M5, QC + passport evidence
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd Netrai/python
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Inference on a Retinal Image
```bash
# Analyze any fundus image and generate the 6-panel report:
python3 inference.py --image ../data/aptos2019/train_images/000c1434d8d7.png --output ../results/demo_output
```

### 3. Run Demo on Dataset Samples
```bash
python3 inference.py --demo --output ../results/demo_output
```

The script will produce:
- `ClinicalReport.png` & `ClinicalReport.pdf`: 6-panel clinical diagnostic sheet.
- `gradcam_heatmap.png`: High-resolution Grad-CAM attention map.
- `lesion_overlay.png`: Segmented microaneurysms, hemorrhages, and exudates.
- `vessel_segmentation.png`: Segmented retinal vasculature.

### 4. Run Reproducible Batch Validation
```bash
# Reproduce the complete held-out Fold-0 benchmark:
python3 batch_validate.py --fold 0 --skip-segmentation --no-enhancement \
  --quality-threshold 0 --batch-size 16 --output ../results/fold0_validation

# Exercise all grading and segmentation stages on a balanced sample:
python3 batch_validate.py --sample-per-grade 5 --batch-size 5 \
  --output ../results/full_pipeline_sample
```

The runner exports `per_image_results.csv`, `metrics_summary.csv`, `aggregate_metrics.json`, `confusion_matrix.csv`, and `per_grade_metrics.csv`. Its per-image and aggregate outputs include both model heads, disagreement, triage, concordant coverage and retained sensitivity/specificity. With segmentation enabled it also writes `segmentation_summary_by_grade.csv`.

### 5. Install and run MATLAB/Simulink

The desktop host has MATLAB R2026a Update 5, Simulink, SimEvents, the required
toolboxes, and the **Deep Learning Toolbox Converter for ONNX Model Format**
support package.

The implementation uses the exported trained dual-head model—there are no placeholder grades or synthetic Grad-CAM fallbacks. Open the `matlab` folder, then run:

```matlab
report = verifyMatlabPackage();
demo_netrai;
```

See [`matlab/README_MATLAB.md`](matlab/README_MATLAB.md) for exact environment
evidence, validation, and CSV export. M5 is the executable checked-in SimEvents
model and ONNX-backed M3/M4 dependencies are installed.

### 6. Audit the Remaining Scientific Limitations

```bash
# Quantify border/FOV Grad-CAM attention on a grade-stratified sample:
python3 audit_explainability.py --per-grade 5 \
  --output ../results/gradcam_qc

# Re-run the normalized IDRiD/DRIVE evidence:
python3 evaluate_lesion_masks.py \
  --manifest ../data/idrid/segmentation_manifest.csv \
  --output ../results/idrid_lesion_validation
python3 evaluate_drive_vessels.py --output ../results/drive_vessel_validation
python3 evaluate_idrid_landmarks.py --output ../results/idrid_landmarks

# Rebuild the internal coverage-risk/workload curve:
python3 analyze_selective_prediction.py

# Cross-check the annual SimEvents sweep independently:
python3 simulate_district_workflow.py \
  --simulink-csv ../matlab/results/m5_scenario_sweep/simulink_scenario_sweep.csv

# Reproduce confidence calibration evidence from the stored Fold-0 outputs:
python3 calibrate_model.py

# After placing the complete EyeQ data as documented in data/README.md:
python3 calibrate_quality.py

# Regenerate the auditable release passport from current evidence:
python3 build_validation_passport.py
```

The current 25-image, grade-stratified Grad-CAM audit completed with zero execution errors and flagged 13/25 maps. Flag rate rises from 20% for true Grade 0 to 100% for true Grade 4. Thirteen visual overlays are saved in `results/verification_2026-09-18/gradcam_qc_25/flagged_maps/`. A flag is a heuristic review signal, not proof that a prediction is wrong. Removing the shortcut risk itself requires retraining with suitable FOV/background masking and border/camera augmentation, followed by repeat accuracy, calibration and attention audits.

Current verification gates: **27/27 Python tests pass**; MATLAB M2 and the
SimEvents `.slx` execute under R2026a, and **41/41 MATLAB files pass MISS_HIT
structural lint**. Full ONNX parity and the 733-image
MATLAB holdout remain blocked by the missing ONNX converter support package.

> M2 lesion outputs are explainability candidates from classical computer vision, not pixel-level clinically validated segmentations. Grad-CAM shows model attention and should not be interpreted as a lesion boundary.

---

## 👥 Team Biscuit
**Smart India Hackathon (SIH) 2026** — Problem Statement 26038 (MathWorks Sponsored)
