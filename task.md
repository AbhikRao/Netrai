# NetrAI Authoritative Project Backlog

> Last synchronized: 2026-09-19  
> Scope now: pipeline, MATLAB/Simulink compliance, validation, and defensible differentiators.  
> Deferred by user: UI and presentation construction.  
> Rule: when project behavior or evidence changes, update this file, `README.md`, and `AGENTS.md` in the same work batch.

## Current evidence baseline

- [x] Reproduce Fold-0 evaluation on 733/733 APTOS images: QWK `0.9330`, referable sensitivity `94.63%`, specificity `94.02%`, AUC `0.9842`.
- [x] Complete a balanced 25-image full-pipeline smoke run with zero execution failures.
- [x] Export both trained heads in one six-output ONNX tensor and verify PyTorch/ONNX Runtime parity on five images (maximum absolute output error `5.25e-06`).
- [x] Add batch CSV/JSON validation, lesion/vessel/landmark evaluation, Grad-CAM QC, external bootstrap CIs, selective-prediction evidence, and 27 automated tests.
- [x] Remove fabricated grades, synthetic Grad-CAM fallbacks, and stale benchmark claims.
- [x] Parse/lint all 41 MATLAB files with zero MISS_HIT structural findings; full style mode still reports legacy formatting debt.
- [x] Fix `verifyMatlabPackage` product discovery for R2026a and make the missing ONNX converter fail with the explicit `NetrAI:MissingONNXConverter` identifier.
- [x] Validate the dual-head policy on all 733 Fold-0 images: 9 disagreements (`1.23%`) are routed to human review; concordant coverage is `98.77%`.
- [x] Expand Grad-CAM QC to 25 grade-stratified images and save all 13 flagged overlays; flag rate is `52%` and remains an open shortcut-risk finding.
- [x] Generate Validation Passport v2: hash eight release artifacts plus evidence/data indexes and bind internal/external grading, M2, calibration, selective prediction, MATLAB environment, M5 cross-check, intended use, and limitations.
- [x] Normalize and checksum IDRiD, DRIVE, Messidor-2, and EyeQ metadata; archive redundant downloads outside the project and save a machine-readable dataset audit.
- [x] Publish a source-only public repository at <https://github.com/AbhikRao/Netrai>; datasets, credentials, local environments, presentation drafts, and generated patient-image derivatives are excluded.
- [x] Run the official 103-image IDRiD test split as an independent grading check: QWK `0.6652`, referable sensitivity `79.69%` (95% CI `68.75–89.06%`), specificity `82.05%` (`69.23–92.31%`), and AUC `0.8746` (`0.7985–0.9387`). This does **not** meet the target and is the current generalization blocker.
- [x] Measure M2 on IDRiD/DRIVE. Optic-disc Dice is `0.8042`; the improved vessel baseline reaches Dice `0.4405` (from `0.1736`), while MA, haemorrhage, and exudate results remain inadequate.
- [ ] Treat the system as clinically validated. This remains prohibited until the external, lesion-level, calibration, and reviewer studies below are complete.

## P0 — Blocking correctness and MathWorks compliance

### P0.1 Install and verify the required MathWorks runtime (user + project)

- [x] Activate a working MATLAB R2026a Update 5 license on the desktop host.
- [x] Install MATLAB, Simulink, Image Processing Toolbox, Deep Learning Toolbox, Statistics and Machine Learning Toolbox, Computer Vision Toolbox, Medical Imaging Toolbox, Optimization Toolbox, and SimEvents.
- [x] Install the free **Deep Learning Toolbox Converter for ONNX Model Format** support package.
- [x] Save `ver`, Simulink license, ONNX-import, and SimEvents path evidence in `matlab/results/environment_2026-09-18.json`; it correctly records the missing ONNX converter.
- [x] Run `verifyMatlabPackage` in desktop MATLAB; the six-output fixture and end-to-end grade pass, with the preprocessing-logit warning recorded below.
- [ ] Run a small `validatePipeline(..., Mode='full')` smoke test in desktop MATLAB.
- [ ] Run or chunk the fixed 733-image Fold-0 validation in MATLAB and compare it with the Python evidence.

### P0.2 Replace misleading confidence with fitted post-hoc calibration

- [x] Add a reproducible stratified calibration/evaluation split and fit scalar temperature `T=0.839077` instead of the arbitrary `T=1.5` constant.
- [x] Export `python/weights/calibration.json` plus raw/calibrated NLL, ECE, Brier, split manifest, reliability bins, and diagram.
- [x] Load the same fitted temperature in Python and MATLAB reports; malformed/missing artifacts fail clearly.
- [x] Add regression tests and visually verify a regenerated Grade-2 sample report.
- [x] Keep the limitation explicit: the 366/367 split of Fold 0 is post-hoc evidence, not independent external validation.

Calibration evaluation evidence: deployed-policy ECE `4.59% -> 2.58%`, NLL `0.3932 -> 0.3837`, and multiclass Brier `0.2101 -> 0.2079`; grade decisions are unchanged. Evidence is in `results/verification_2026-09-17/calibration/`.

### P0.3 Use both trained model heads as a safety signal

- [x] Export both the five grade logits and the independently trained referable-DR logit to ONNX.
- [x] Return grade-head probability, referable-head probability, and head-disagreement status in Python and MATLAB.
- [x] Route every head disagreement to human review instead of silently presenting a confident result.
- [x] Re-run PyTorch/ONNX Runtime parity fixtures and update the six-output schema documentation.
- [x] Execute the updated parity fixture in desktop MATLAB: six ONNX outputs pass with maximum error `7.63e-06`, and the end-to-end grade matches. Keep the `1.0163` preprocessing-path logit difference as an open parity warning.

Fold-0 safety evidence: the grade policy retains `94.63%` sensitivity / `94.02%` specificity; the independent binary head obtains `92.95%` / `94.94%`; 9/733 decisions disagree. Neither head is silently preferred on those nine cases. Evidence is in `results/verification_2026-09-18/fold0_dual_head/`.

### P0.4 Build the required real Simulink/SimEvents M5 digital twin

- [x] Replace the script-only placeholder with generated `matlab/models/NetrAI_Telemedicine_Model.slx` and verify SimEvents execution.
- [x] Model acquisition arrivals, ungradeable-image recapture, upload bandwidth, AI service time, four-way routing, referral/review queues, and ophthalmologist capacity.
- [ ] Add an explicit timed queue-abandonment/escalation branch to SimEvents; the Python reference currently reports a declared 24-hour planning threshold.
- [x] Parameterize a district program serving 100,000 patients/year across 18 bandwidth, staffing, and quality scenarios.
- [x] Export throughput, utilization, running-average wait, queue size, recapture load, and annual staffing requirements; retain patient-level P50/P90/P95 waits in the independent Python reference.
- [x] Finish the corrected 100,000-patient annual sweep and pass all 18 SimEvents/Python rate cross-checks (1.5 percentage-point tolerance) after removing the discovered 10,000-entity back-pressure cap. At 0.25 Mbps the true upload backlog is 47,857–62,974 patients; at 1+ Mbps completion is `>=99.997%`.

## P1 — The four measured pipeline weaknesses

### P1.1 M1 quality assessment calibration (currently all Fold-0 images are “borderline”)

- [x] Implement `python/calibrate_quality.py` with official EyeQ train-only fitting, test-only evaluation, feature caching, strict missing-image checks, and versioned evidence output.
- [x] Normalize and checksum the complete EyeQ labels: 12,543 train + 16,249 test rows.
- [x] Download and audit all 28,792 EyePACS source images: 12,543 train + 16,249 test images under `data/eyeq/images/`.
- [ ] Calibrate Good / Usable / Reject thresholds on labeled data; report macro-F1, balanced accuracy, confusion matrix, and rejection sensitivity.
- [ ] Validate focus, illumination, and FOV sub-scores separately and make recapture feedback correspond to the failed factor.
- [ ] Test camera/domain robustness and compression/low-light corruptions representative of portable rural fundus cameras.

### P1.2 Fine-grained ICDR grading (referable screening is strong; minority grade recall is not)

- [x] Preserve and surface the current per-grade recall baseline: G0 `99.72%`, G1 `63.51%`, G2 `72.00%`, G3 `58.97%`, G4 `59.32%`.
- [x] Establish and run the official IDRiD 103-image test set as an external Indian-domain grading check with bootstrap CIs; it currently fails the required endpoint.
- [ ] Retrain/evaluate with ordinal-aware loss, class-balanced sampling or loss, higher-resolution crops, and cross-validation/ensembling ablations.
- [ ] Compare ordinal thresholds with plain argmax and the final safety-aware policy using bootstrap confidence intervals.

### P1.3 Grad-CAM shortcut attention (13/25 stratified maps flagged)

- [x] Expand the audit to 25 grade-stratified images; export FOV/border/corner metrics and save 13 flagged overlays. Flag rates by true grade are G0 `20%`, G1 `40%`, G2 `40%`, G3 `60%`, G4 `100%`.
- [ ] Have a clinician inspect the saved flagged overlays and define a larger pre-registered audit sample; the heuristic flag is not proof that a prediction is wrong.
- [ ] Retrain with circular-FOV/background masking and border/camera augmentation; do not “fix” the evidence by hiding borders only at inference.
- [ ] Re-run accuracy, calibration, and Grad-CAM QC to demonstrate that mitigation did not create a harmful distribution shift.

### P1.4 M2 lesion and landmark validity, including sub-pixel microaneurysms

- [x] Download IDRiD images, lesion masks, grading labels, and optic-disc/fovea landmarks; generate an 81-case manifest and verify exact completeness.
- [x] Download DRIVE images/vessel masks; record that public manual masks are available for the 20-image training split, not the downloaded test split.
- [x] Run pixel Dice/IoU and lesion-level sensitivity/precision with bootstrap confidence intervals for exudates, haemorrhages, and microaneurysms; the measured baselines remain too weak for clinical claims.
- [x] Add separable quadratic sub-pixel MA peak refinement in Python and MATLAB and verify it on a synthetic Gaussian peak.
- [ ] Extend MA detection to native-resolution, multi-scale candidates and transform refined coordinates back to native pixels; the current refinement operates on the analysis grid only.
- [ ] Report an MA FROC curve (sensitivity versus false positives/image), not only pixel overlap.
- [x] Report DRIVE vessel Dice/sensitivity/specificity and IDRiD optic-disc/fovea localization error with bootstrap confidence intervals.

## P2 — Standout safety system ("knows when not to decide")

- [x] Implement a versioned four-way triage policy: **auto-clear**, **refer**, **recapture**, and **uncertain—human review**. Its thresholds remain research-only pending external validation.
- [ ] Build a clinical-concordance gate across grade prediction, referable head, lesion evidence, image quality, and Grad-CAM QC.
- [x] Add selective-prediction evidence: coverage-risk curves, retained sensitivity/specificity, false negatives/positives, and workload for 20 confidence thresholds after mandatory dual-head disagreement review.
- [ ] Add offline/store-and-forward operation plus bandwidth-aware image compression testing for rural PHCs.
- [x] Create Validation Passport v2 with release/data-index hashes, internal and external bootstrap evidence, M2 metrics, calibration, Grad-CAM QC, selective prediction, MATLAB environment, SimEvents/Python cross-check, intended use, and known failures.
- [ ] Add subgroup metadata/domain results and clinician reviewer-study evidence when those studies exist.
- [ ] Ensure every report clearly states screening/not diagnosis, reasons for recapture/review, model/data version, and uncertainty evidence.

## P3 — Validation rigor required by the problem statement

- [x] Use the official labeled IDRiD test split for the first external Indian-domain grading evaluation. The complete Messidor-2 images remain domain-test-only because the official release has no DR labels.
- [x] Run external IDRiD referable-DR sensitivity/specificity/AUC/QWK with image-level stratified 95% bootstrap confidence intervals; record that sensitivity/specificity fail the target.
- [ ] Validate all thresholds on data separate from final evaluation and record dataset/patient leakage checks.
- [ ] Add subgroup/domain analysis where metadata permits (camera, site, image quality, sex/age) and document unavailable attributes.
- [ ] Create an integrated-vs-single-technique ablation table: classifier only; +quality gate; +ordinal grading; +dual-head safety; +clinical concordance.
- [ ] Build a source-verified published-benchmark comparison with identical endpoints/datasets or clearly mark non-comparable rows.
- [ ] Conduct a small ophthalmologist/reviewer study: explanation usefulness rubric, disagreement reasons, and median/P90 validation time below 30 seconds.
- [ ] Perform failure-mode review and document false-negative examples, especially grades 3–4.

## P4 — Deferred until the pipeline evidence is credible

- [ ] Build the interactive demo UI. **Deferred by user.**
- [ ] Build or revise the presentation deck. **Deferred by user.**
- [ ] Use only evidence generated by completed tasks above; do not market planned features as implemented.

## Remaining dataset acquisition order

1. **No required dataset download remains for the current public benchmarks.**
   EyeQ/EyePACS, IDRiD, DRIVE, APTOS, and Messidor-2 image payloads are local.
2. **Optional provenance-verified Messidor-2 grades** — only if a lawful,
   source-verified label release is found; the official image package has no DR
   ground truth. Do not substitute an unverified internet CSV.

IDRiD and DRIVE acquisition is complete for their public benchmarks. IDRiD is
already the first external Indian-domain grading evaluation.

Store each dataset outside Git under `data/<dataset>/`, retain the original archive/manifest checksums, record its license/source, and never commit patient images or credentials.

## Completed engineering checklist

- [x] Normalize dataset folders, move 5.6 GB of redundant archives outside the project, archive superseded demo/smoke outputs recoverably, and remove IDE/Python/Simulink caches.
- [x] Fix preprocessing sigma and RGB/gray conversion mismatches.
- [x] Make classical segmentation thresholds resolution-adaptive and scope NV density to the FOV.
- [x] Implement real quadrant feature extraction and remove dead skeleton code.
- [x] Extract the shared model class and preprocessing paths.
- [x] Build reproducible batch metrics/CSV export and aggregate validation.
- [x] Add M1–M4 MATLAB functions and real ONNX-backed grading.
- [x] Add the executable SimEvents `.slx`, an 18-scenario annual sweep, and an independent Python patient-level FCFS reference.
- [x] Fit and deploy post-hoc scalar temperature calibration with reliability evidence.
- [x] Export and use the independent referable head with explicit disagreement abstention.
- [x] Add the EyeQ quality-calibration runner without inventing thresholds before data exists.
- [x] Add analysis-grid quadratic sub-pixel MA coordinates in Python and MATLAB.
- [x] Add a machine-readable, artifact-hashed Validation Passport.
