# 🔬 NetrAI — Explainable AI for Diabetic Retinopathy Screening

> **SIH 2026 | Problem Statement 26038 | Team Biscuit | MathWorks Sponsored**  
> *Automated, explainable, validated DR screening for rural India*

Public repository: <https://github.com/AbhikRao/Netrai>

## Current work status — pipeline implementation resumed (2026-09-21)

The first planned improvement stage is complete: a frozen ImageNet
MobileNetV3-Small quality candidate was trained and evaluated on all 28,792
EyeQ images with patient-disjoint fitting, selection, and calibration. On the
16,249-image official test split it reaches macro-F1 **0.8289** and Reject
recall **86.37%** (patient-bootstrap 95% CI **85.19–87.56%**), compared with
0.3976 for scalar IQS and 0.5853 for the best handcrafted-feature model. Its
Good/Usable false-reject rate is **4.63%**.

The 3.6 MB ONNX candidate has an explicit shared preprocessing contract.
MATLAB reproduces the fixed real-image input exactly and matches the candidate
decision with maximum logit error **2.65e-05**. Exact-file hashing found no
duplicates within or across official EyeQ train/test splits. Both eyes from a
patient remain in one internal partition.

The model is intentionally **not the default M1 route yet**. On the paired
synthetic audit it hard-rejects only 0.4% of defocus, 7.6% of low-light, 6.4%
of uneven-light, 1.6% of shifted-FOV, and 0% of JPEG-Q25 cases. EyeQ provides
global quality labels rather than factor-specific failure labels, the official
test set has already informed development, and no portable-camera cohort has
been tested. Validation Passport v4 records the passing global-label gate and
these promotion blockers together.

## SIH submission deck

The evidence-backed six-slide SIH deck is complete in both editable PowerPoint
and portal-ready PDF form:

- `presentation/Netrai_SIH_2026_Final.pptx`
- `presentation/Netrai_SIH_2026_Final.pdf`

It preserves the official SIH template, uses actual prototype outputs, includes
editable native charts for M5 capacity and internal-vs-external grading, and
states the IDRiD generalization failure and clinical claim boundary explicitly.
Before submission, replace `[ADD REGISTERED ID]` on slide 1 with the portal Team
ID. The user owns further presentation editing; current engineering work does
not modify the deck. Presentation artifacts remain local-only.

---

## 📌 Problem

India has **77 million diabetic adults** — the 2nd highest globally. Diabetic Retinopathy (DR) affects ~18% of this population and is a **leading cause of preventable blindness**. Early screening can prevent 90% of vision loss, but India has only ~1 ophthalmologist per 100,000 rural population, making mass manual screening infeasible.

## 💡 Solution

NetrAI is a **5-module research pipeline** that analyzes a retinal fundus photograph and produces an explainable screening report. The internal grading benchmark is reproducible and the SimEvents district digital twin now runs, but independent IDRiD testing shows the system is not yet clinically validated:

| Module | Function | Output | Status |
|---|---|---|---|
| **M1** Image Quality Assessment | Evaluates global gradability; candidate factor checks remain under study | Good/Usable/Reject + recapture research evidence | ⚠️ Learned candidate passes the EyeQ global-label gate and MATLAB parity, but remains non-default after synthetic defect gaps and no portable-camera validation |
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

### Image-quality validation: stronger global model, unresolved capture defects

The official EyeQ train/test run completed on all **12,543 training** and
**16,249 held-out test** images with zero missing rows. Fitted scalar IQS
thresholds achieve only **0.3976 macro-F1**, **0.4101 balanced accuracy**, and
**25.09% Reject recall** on the test split, failing the project engineering
gate of 0.70 macro-F1 and 0.80 Reject recall. The threshold artifact is retained
for reproducibility but marked `not_deployed`; the shared loader rejects it by
default unless a diagnostic explicitly requests evidence-only access.

Using the same four handcrafted scores, balanced HistGradientBoosting improves
test macro-F1 to **0.5853** and Reject recall to **55.81%**, still below the gate
and therefore also not deployed. A paired 250-image synthetic stress test shows
why: severe FOV misalignment is rejected **99.2%** of the time, but defocus
blur, low light, and JPEG Q25 are each rejected only **2.4%** of the time.
These synthetic corruptions are diagnostic evidence, not portable-camera
validation.

The completed frozen MobileNetV3-Small candidate uses 7,527 patient-grouped
fit images, 2,508 selection images, and a separate 2,508-image calibration
partition. A conservative Reject policy is chosen only on calibration patients
by requiring the one-sided 95% Wilson lower bound for Reject recall to exceed
0.80. On all 16,249 official test images it reaches **0.8289 macro-F1**,
**0.8299 balanced accuracy**, and **86.37% Reject recall**; the patient-cluster
95% CIs are **0.8227–0.8351** and **85.19–87.56%**, respectively. It falsely
rejects **4.63%** of truly Good/Usable images. Exact encoded-file duplicate
hashes are zero within and across the official splits.

This global EyeQ improvement does not solve factor-specific adequacy. On the
same paired stress set the learned model rejects only **0.4%** of defocus,
**7.6%** low light, **6.4%** uneven illumination, **1.6%** shifted FOV, and
**0%** JPEG Q25. Its probability generally moves in the correct direction,
but the routing decision is not safe enough to become the default. The ONNX
and MATLAB candidate is therefore retained for reproducible research while
GPU augmentation/fine-tuning, real portable-camera testing, and labeled
focus/illumination/FOV validation remain open. EyeQ test-set exposure is
explicitly disclosed, so final generalization requires fresh evidence.

### Auditable release evidence

`python/build_validation_passport.py` generates Validation Passport v4 at
`results/verification_2026-09-20/validation_passport.json`. It fingerprints
the release and learned-M1 candidate artifacts and binds them to the failed
handcrafted M1 baselines, passing global learned-M1 gate, corruption failures,
exact-duplicate and MATLAB parity audits, internal/external grading, M2,
calibration, Grad-CAM, selective prediction, full MATLAB/runtime comparison,
MATLAB environment, M5 cross-check, intended/prohibited use, and known
limitations. It is a traceability artifact—not a regulatory certificate.

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
package is installed. The native six-output ONNX fixture passes with maximum
raw-output error `7.63e-06`, and preprocessing alignment reduced its maximum
end-to-end logit difference from `1.0163` to `0.2210`.

The fixed full MATLAB holdout now completes **733/733 images with zero errors**:
QWK **0.9297**, referable sensitivity **94.97%**, specificity **93.33%**, AUC
**0.9841**, and mean model time **1.55 s/image**. The row-for-row Python audit
matches 717/733 five-grade decisions (97.82%), 729/733 grade-derived referable
decisions (99.45%), and 724/733 final triage actions (98.77%). Six independent
referable-head decisions differ, and the maximum comparable probability drift
is `0.2990`, so screening, strict-grade, and `0.05` numerical-tolerance gates
remain failed. This is a functioning MATLAB deployment with a bounded
cross-library preprocessing incompatibility—not exact runtime equivalence.
Evidence is in `results/verification_2026-09-20/matlab_fold0_733_aligned/` and
`results/verification_2026-09-20/matlab_python_fold0_733_aligned/`.

## Current Priority Roadmap

[`task.md`](task.md) is the authoritative backlog and contains the detailed
active sequence with completion criteria:

1. **Protocol frozen for the current M1 experiment.** Patient-disjoint fit/selection/calibration roles, cache signatures, exact-file duplicate checks, acceptance metrics, and prior test exposure are recorded. Perceptual/cross-dataset duplication remains open.
2. **Learned M1 candidate evaluated.** The global EyeQ gate and fixed MATLAB parity pass, while factor-specific stress and real-camera gates fail or remain unavailable; the candidate stays non-default and needs augmentation/fine-tuning.
3. **Continue MATLAB consistency.** M1 fixed-fixture parity passes. Trace the grade model's remaining 16 grade and 9 triage mismatches through stagewise preprocessing and add broader M1 edge-case parity before promotion.
4. **Improve grading and explanation quality together.** Test class balancing, ordinal loss, and retinal-field/background augmentation; recalibrate and repeat external grading, minority-grade recall, and Grad-CAM audits. External sensitivity >90% and specificity >85% remain unmet targets.
5. **Improve lesion validation.** Prioritize MA/haemorrhage candidates, native-resolution sub-pixel MA coordinates and FROC, then exudate/vessel models. Validate against annotations and identify missing evidence for neovascularization and haemorrhage subtypes.
6. **Validate the integrated workflow.** Compare all variants on the same cohort, counting recapture/review coverage and missed referable cases; add store-and-forward/compression and timed SimEvents escalation, refresh the passport/reports, and conduct a separate clinician usefulness/review-time study.

Local data is sufficient for the first experiments. Real portable-camera data,
fresh independent grading evidence, and clinician assessments are later
dependencies. Learned candidates will remain experimental until their measured
gates and integration checks pass.

The user owns further presentation work. Interactive UI work remains deferred
while the evidence roadmap continues.

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
│   ├── Netrai_SIH_2026_Final.pptx      # Editable six-slide SIH submission deck
│   ├── Netrai_SIH_2026_Final.pdf       # Portal-ready rendered submission
│   ├── SIH2026-IDEA-Presentation-Format.pptx # Official source template
│   └── Team Biscuit.pdf                # Previous-round reference deck
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
│   ├── audit_quality_robustness.py    # Paired synthetic M1 corruption stress audit
│   ├── audit_learned_quality_robustness.py # Learned-candidate paired stress audit
│   ├── audit_eyeq_duplicates.py       # Exact EyeQ split duplicate audit
│   ├── train_quality_model.py         # Patient-disjoint frozen-encoder M1 experiment
│   ├── export_quality_model.py        # Gated ONNX/MATLAB candidate export
│   ├── compare_quality_baselines.py   # Non-deployed handcrafted-feature ablation
│   ├── compare_runtime_outputs.py     # MATLAB/Python decision + numerical parity audit
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
│   ├── tests/                         # 45 automated pipeline/edge-case checks
│   ├── requirements.txt               # Dependencies (torch, timm, grad-cam, albumentations...)
│   ├── models/
│   │   ├── netrai_model.py            # EfficientNet-B4 dual-head architecture
│   │   └── quality_model.py           # MobileNetV3-Small M1 input/model contract
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
    ├── verification_2026-09-18/       # Dual-head, external, M2, M5 + QC evidence
    ├── verification_2026-09-20/       # Baselines, MATLAB holdout/parity + passport v4
    └── verification_2026-09-21/       # Learned M1 split/metrics/stress evidence
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

# Reproduce the complete EyeQ threshold baseline (evidence-only when gate fails):
python3 calibrate_quality.py \
  --output ../results/verification_2026-09-20/eyeq_quality_calibration

# Reproduce paired corruption stress evidence and feature-only baselines:
python3 audit_quality_robustness.py \
  --output ../results/verification_2026-09-20/eyeq_quality_robustness
python3 compare_quality_baselines.py

# Reproduce the learned M1 experiment from complete/resumable embeddings,
# then its gated ONNX export and paired stress audit:
python3 train_quality_model.py
python3 export_quality_model.py
python3 audit_learned_quality_robustness.py

# Regenerate the auditable release passport from current evidence:
python3 build_validation_passport.py
```

The current 25-image, grade-stratified Grad-CAM audit completed with zero execution errors and flagged 13/25 maps. Flag rate rises from 20% for true Grade 0 to 100% for true Grade 4. Thirteen visual overlays are saved in `results/verification_2026-09-18/gradcam_qc_25/flagged_maps/`. A flag is a heuristic review signal, not proof that a prediction is wrong. Removing the shortcut risk itself requires retraining with suitable FOV/background masking and border/camera augmentation, followed by repeat accuracy, calibration and attention audits.

All **45/45 Python regression tests pass** across the established pipeline and learned-M1
protocol/input/decision invariants. MATLAB M2 and the SimEvents `.slx` execute
under R2026a, the MATLAB source passes the existing
structural checks, and native six-output ONNX parity passes with maximum raw
error `7.63e-06`. The aligned 733-image MATLAB holdout completes without
execution errors and exceeds the internal screening target, but the exact
cross-runtime screening, five-grade, and probability-tolerance gates remain
open as quantified above.

> M2 lesion outputs are explainability candidates from classical computer vision, not pixel-level clinically validated segmentations. Grad-CAM shows model attention and should not be interpreted as a lesion boundary.

---

## 👥 Team Biscuit
**Smart India Hackathon (SIH) 2026** — Problem Statement 26038 (MathWorks Sponsored)
