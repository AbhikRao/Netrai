# NetrAI Authoritative Project Backlog

> Last synchronized: 2026-09-21
> Scope now: pipeline, MATLAB/Simulink compliance, validation, and defensible differentiators.
> Deferred by user: interactive UI. Further presentation work is user-owned.
> Rule: when project behavior or evidence changes, update this file, `README.md`, and `AGENTS.md` in the same work batch.
> Current instruction (2026-09-21): implementation resumed. Preserve measured failures and do not promote candidates past incomplete safety gates.

## Implementation checkpoint — learned M1 candidate

The frozen-encoder M1 experiment is now complete as a reproducible research
candidate. It is deliberately not the default routing policy because its
factor-specific stress behavior and portable-camera validity remain inadequate.

- [x] Implement explicit full-frame 224-pixel RGB half-pixel bilinear preprocessing and a frozen ImageNet MobileNetV3-Small encoder with an EyeQ Good/Usable/Reject head.
- [x] Extract and integrity-bind all 12,543 training and 16,249 test embeddings with resumable caches.
- [x] Use three patient-disjoint official-training partitions: fit 7,527 images/5,159 patients; selection 2,508/1,736; calibration 2,508/1,719. Both eyes stay together and official train/test have zero filename-patient overlap.
- [x] Audit exact source-file hashes: zero within-train, within-test, or cross-split duplicates. Perceptual near-duplicates and cross-dataset duplication remain untested.
- [x] Fit regularization on selection patients and temperature/Reject policy on calibration patients. The policy requires a one-sided 95% Wilson lower bound for calibration Reject recall of at least 0.80.
- [x] Evaluate all 16,249 official test images: macro-F1 `0.8289`, balanced accuracy `0.8299`, Reject recall `86.37%` (patient-bootstrap 95% CI `85.19–87.56%`), and Good/Usable false-reject rate `4.63%`. The engineering gate passes.
- [x] Export the 3.6 MB ONNX candidate and reproduce the fixed real-image input exactly in MATLAB; decision parity passes and the maximum MATLAB/Python logit error is `2.65e-05`.
- [x] Run the paired 250-image corruption audit. The candidate increases Reject probability for most blur/lighting/FOV corruptions, but hard rejection remains only `0.4%` for defocus, `7.6%` low light, `6.4%` uneven light, `1.6%` FOV shift, and `0%` JPEG Q25.
- [x] Keep the learned candidate non-default and record it in Validation Passport v4. EyeQ test outcomes were already exposed during baseline development and threshold safety review, so these are development-benchmark results, not a pristine final evaluation.

Evidence is under `results/verification_2026-09-21/eyeq_learned_quality/`,
`matlab/models/quality_model_candidate.*`, and
`matlab/results/quality_model_candidate_parity.json`. Embedding caches are
Git-ignored. The active Python/MATLAB inference routes still use the existing
research quality policy until factor-specific and portable-camera checks pass.

## Active execution sequence

This sequence orders the work; the P0–P4 sections below retain the detailed
backlog and release blockers. Each stage must leave runnable code, measured
evidence, and synchronized README/AGENTS/task records.

### 1. Freeze the evaluation protocol and review the unfinished scaffolding

- [x] Review preprocessing, patient grouping, cache validity, and split selection; bind caches to preprocessing, encoder weights, row order, labels, source paths, file sizes, and modification times.
- [x] Keep both eyes from each identified EyeQ patient in the same partition and audit exact official-split file duplicates.
- [ ] Audit perceptual near-duplicates and cross-dataset content duplicates; exact byte hashes cannot detect transformed copies.
- [x] Choose regularization on selection patients and calibration/decision policy on a separate calibration partition. Record prior EyeQ/IDRiD test exposure and require a fresh source for final generalization claims.
- [x] Apply the fixed M1 gate (macro-F1 >=0.70 and Reject recall >=0.80) and report false rejection, patient-bootstrap intervals, and DR-grade-stratified retention.

Completion evidence: reviewed experiment configuration, split manifests,
leakage audit, and a fixed evaluation/selection protocol. No new accuracy claim
is justified by a split audit alone.

### 2. Build and evaluate the learned M1 quality candidate

- [x] Train the CPU-feasible frozen MobileNetV3-Small encoder plus balanced linear Good/Usable/Reject head using explicit full-frame sampling.
- [x] Fit on EyeQ training patients and compare with scalar IQS (`0.3976` macro-F1) and four-feature HGB (`0.5853`); the learned candidate reaches `0.8289`.
- [x] Evaluate the locked candidate on the complete official test split with per-class metrics, confusion matrix, patient-bootstrap intervals, calibration, timing, false rejection, and DR-grade retention.
- [x] Repeat paired blur, low-light, uneven-light, FOV, and compression stress tests; retain the measured synthetic failure modes and reserve real portable-camera testing.
- [ ] Prepare a GPU fine-tuning/augmentation experiment for blur, FOV shift, compression, and illumination robustness using the same patient-disjoint protocol. Do not launch paid/external compute automatically.

Completion evidence: trained candidate, complete metrics, and documented
accept/reject decision against predeclared gates. Integration follows only
after the MATLAB and screening-safety checks in stages 3 and 6.

### 3. Establish MATLAB/Python consistency for both models

- [ ] Trace the existing grade model's 16 grade mismatches and 9 triage mismatches through decoding, cropping, resize, Gaussian filtering, rounding, and normalization using identical intermediate tensors.
- [ ] Implement shared preprocessing semantics and rerun the fixed 733-image comparison. Preserve exact decision gates and the existing 0.05 probability tolerance; explicitly retain any unresolved mismatch.
- [x] Export the learned M1 candidate to ONNX, implement its exact input/decision contract in MATLAB, and pass the fixed real-image input/logit/probability/decision parity fixture. A broader edge-case parity set remains desirable before promotion.

Completion evidence: reproducible native MATLAB verification and comparison
artifacts. If exact parity still fails, keep the versions separately identified
and the interoperability blocker open.

### 4. Improve M3 grading and M4 attention together

- [ ] Use patient-aware training/validation where identifiers permit, class balancing and ordinal-aware loss, and retinal-field/background augmentation. Keep IDRiD's official test labels out of fitting and threshold selection.
- [ ] Start with one revised model and isolate changes through ablations before adding ensembles or higher-resolution models. Plan substantial fine-tuning for GPU execution when available.
- [ ] Refit confidence calibration on a separate calibration partition; repeat per-grade recall, QWK, referable sensitivity/specificity/AUC, subgroup/domain metrics, and the fixed Grad-CAM audit.
- [ ] Report external sensitivity >90% and specificity >85% as acceptance targets, with confidence intervals and failures. The current IDRiD result is 79.69% / 82.05%; prospective clinical validity and clinician-rated explanation usefulness remain unproven.

Completion evidence: a reproducible comparison showing whether generalization,
minority-grade recall, calibration, and attention improve together. An
attention flag is a review signal, not a lesion label.

### 5. Improve M2 with annotation-based validation

- [ ] Prioritize the weakest measured components: microaneurysms and haemorrhages, then exudates and vessels. Use IDRiD/DRIVE annotations with fixed train/evaluation splits.
- [ ] Extend MA candidates to native-resolution, multi-scale processing and transform sub-pixel locations back to native coordinates; measure FROC sensitivity versus false positives/image.
- [ ] Compare trained lesion/vessel candidates with the current methods using Dice/IoU, lesion sensitivity/precision, localization error, and runtime. Retain the measured optic-disc baseline.
- [ ] Track haemorrhage subtype and neovascularization detection as separate requirements needing appropriate labels; current heuristics cannot establish their clinical accuracy.

Completion evidence: per-structure benchmark results and visible failures;
lesion candidates remain advisory until they justify stronger clinical use.

### 6. Integrate safety, district workflow, and final evidence

- [ ] Evaluate classifier-only, +quality, +ordinal, +dual-head safety, and +concordance variants on the same cohort. Include rejected/abstained images in total coverage, missed referable cases, and review workload rather than reporting retained-set accuracy alone.
- [ ] Introduce quality-aware recapture/review with reasons and model versions; use uncertain lesion/attention signals for clinician review only when supported by the ablations.
- [ ] Add store-and-forward/compression handling and timed review escalation in SimEvents; rerun the 18-scenario Python/Simulink cross-check with clearly identified measured inputs and planning assumptions.
- [x] Update Validation Passport v4, learned-M1 regression tests, CPU stress timing, and fixed MATLAB parity evidence.
- [ ] Add active report fields/end-to-end learned-M1 integration only after factor-specific/portable-camera evidence supports promotion; clinician usefulness and <30-second review testing still require reviewers.

Completion evidence: an integrated benchmark, updated reproducible release
artifacts, and an explicit record of any remaining failures. Existing public
datasets are sufficient for the first experiments; real-camera images,
independent grading labels, and clinician assessments address later evidence
gaps.

## Current evidence baseline

- [x] Reproduce Fold-0 evaluation on 733/733 APTOS images: QWK `0.9330`, referable sensitivity `94.63%`, specificity `94.02%`, AUC `0.9842`.
- [x] Complete a balanced 25-image full-pipeline smoke run with zero execution failures.
- [x] Export both trained heads in one six-output ONNX tensor and verify PyTorch/ONNX Runtime parity on five images (maximum absolute output error `5.25e-06`).
- [x] Add batch CSV/JSON validation, lesion/vessel/landmark evaluation, Grad-CAM QC, external bootstrap CIs, selective-prediction evidence, runtime-comparison evidence, and learned-M1 protocol/parity regression tests; all 45 Python tests pass.
- [x] Remove fabricated grades, synthetic Grad-CAM fallbacks, and stale benchmark claims.
- [x] Parse/lint all 41 MATLAB files with zero MISS_HIT structural findings; full style mode still reports legacy formatting debt.
- [x] Fix `verifyMatlabPackage` product discovery for R2026a and make the missing ONNX converter fail with the explicit `NetrAI:MissingONNXConverter` identifier.
- [x] Validate the dual-head policy on all 733 Fold-0 images: 9 disagreements (`1.23%`) are routed to human review; concordant coverage is `98.77%`.
- [x] Expand Grad-CAM QC to 25 grade-stratified images and save all 13 flagged overlays; flag rate is `52%` and remains an open shortcut-risk finding.
- [x] Generate Validation Passport v4: retain the failed handcrafted M1 baseline and add the learned candidate checkpoint/config/ONNX/parity hashes, three-way patient split, exact-duplicate audit, complete EyeQ metrics, corruption failures, and non-default decision.
- [x] Normalize and checksum IDRiD, DRIVE, Messidor-2, and EyeQ metadata; archive redundant downloads outside the project and save a machine-readable dataset audit.
- [x] Publish a source-only public repository at <https://github.com/AbhikRao/Netrai>; datasets, credentials, local environments, presentation drafts, and generated patient-image derivatives are excluded.
- [x] Run the official 103-image IDRiD test split as an independent grading check: QWK `0.6652`, referable sensitivity `79.69%` (95% CI `68.75–89.06%`), specificity `82.05%` (`69.23–92.31%`), and AUC `0.8746` (`0.7985–0.9387`). This does **not** meet the target and is the current generalization blocker.
- [x] Measure M2 on IDRiD/DRIVE. Optic-disc Dice is `0.8042`; the improved vessel baseline reaches Dice `0.4405` (from `0.1736`), while MA, haemorrhage, and exudate results remain inadequate.
- [x] Build and visually verify the official six-slide SIH deck in editable PPTX and portal-ready PDF form, using actual prototype outputs and native charts; package/layout/font/chart-data validation passes.
- [ ] Treat the system as clinically validated. This remains prohibited until the external, lesion-level, calibration, and reviewer studies below are complete.

## P0 — Blocking correctness and MathWorks compliance

### P0.1 Install and verify the required MathWorks runtime (user + project)

- [x] Activate a working MATLAB R2026a Update 5 license on the desktop host.
- [x] Install MATLAB, Simulink, Image Processing Toolbox, Deep Learning Toolbox, Statistics and Machine Learning Toolbox, Computer Vision Toolbox, Medical Imaging Toolbox, Optimization Toolbox, and SimEvents.
- [x] Install the free **Deep Learning Toolbox Converter for ONNX Model Format** support package.
- [x] Refresh `ver`, Simulink license, ONNX-import, and SimEvents path evidence in `matlab/results/environment_2026-09-18.json`; it records `onnx_converter_ready: true`.
- [x] Run `verifyMatlabPackage` in desktop MATLAB; the six-output fixture and end-to-end grade pass, with the preprocessing-logit warning recorded below.
- [x] Run a 25-image `validatePipeline(..., Mode='full')` smoke test in desktop MATLAB: 25/25 complete, no errors, QWK `0.9757`, referable sensitivity/specificity `100%`, and average end-to-end time `8.18 s/image`.
- [x] Run the fixed 733-image Fold-0 validation in MATLAB and compare all 733 rows with the Python evidence.
- [ ] Resolve or explicitly accept the remaining cross-library parity gate failure before calling the runtimes interchangeable.

The aligned full MATLAB run completes 733/733 with zero errors: QWK `0.9297`,
referable sensitivity `94.97%`, specificity `93.33%`, AUC `0.9841`, and mean
model time `1.55 s/image`. Against the matching Python export, strict grades
match 717/733 (`97.82%`) and grade-derived referability matches 729/733
(`99.45%`), but 6 independent-head decisions and 9 final triage actions differ.
The exact screening, strict-grade, and `0.05` numerical-tolerance gates
therefore remain **failed**. Preprocessing alignment cut grade mismatches from
30 to 16 and reduced the worst ordinal-score drift from `0.4458` to `0.2622`,
but this is a bounded incompatibility—not exact parity.

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
- [x] Execute the updated parity fixture in desktop MATLAB: six ONNX outputs pass with maximum error `7.63e-06`, and the end-to-end grade matches. The measured preprocessing change reduced the fixture's maximum end-to-end logit difference from `1.0163` to `0.2210`; the full-holdout comparator above remains the authority for decision parity.

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

- [x] Implement `python/calibrate_quality.py` with official EyeQ train-only fitting, test-only evaluation, exact dataset paths, safely parallel/resumable feature caching, strict completeness checks, and versioned evidence output.
- [x] Normalize and checksum the complete EyeQ labels: 12,543 train + 16,249 test rows.
- [x] Download and audit all 28,792 EyePACS source images: 12,543 train + 16,249 test images under `data/eyeq/images/`.
- [x] Fit Good / Usable / Reject thresholds on official EyeQ train labels and evaluate once on the untouched official test split. The scalar IQS baseline fails the engineering gate and is **not deployed**: test macro-F1 `0.3976`, balanced accuracy `0.4101`, Reject recall `25.09%` versus required `0.70`/`0.80`.
- [x] Audit focus, illumination, and FOV score association with global EyeQ labels. Illumination is the strongest Reject signal (test AUC `0.7043`); focus and the current scalar FOV score are not valid factor classifiers. EyeQ has no factor-specific labels, so this does not complete factor validation.
- [x] Run a paired synthetic stress audit on 250 officially Good EyeQ test images. The baseline catches severe FOV misalignment (`99.2%` Reject) and some uneven illumination (`40.8%` Reject), but misses defocus blur, low light, and JPEG Q25 (each only `2.4%` Reject).
- [x] Compare four balanced classifiers using the existing handcrafted IQS/focus/illumination/FOV features. HistGradientBoosting is best at test macro-F1 `0.5853` and Reject recall `55.81%`; it remains below the gate and is not deployed.
- [x] Make the quality-threshold loader reject every non-`deployed` artifact by default; diagnostic stress tests require an explicit evidence-only override.
- [x] Train/evaluate the frozen MobileNetV3-Small EyeQ candidate, export it to ONNX, and reproduce its fixed-fixture decision in MATLAB. It passes the global-label gate but remains non-default because factor-specific stress and portable-camera gates are open.
- [ ] Improve the learned candidate's synthetic defect coverage before promotion: current hard-reject rates are defocus `0.4%`, low light `7.6%`, uneven illumination `6.4%`, shifted FOV `1.6%`, and JPEG Q25 `0%` on 250 Good source images.
- [ ] Validate factor-specific recapture feedback on labeled focus/illumination/FOV failures; global EyeQ labels cannot establish this.
- [ ] Test real portable-camera/domain robustness. Synthetic compression/low-light stress evidence is complete but is not camera validation.

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
- [x] Create Validation Passport v4 with release/data-index hashes, failed handcrafted M1 evidence, the learned candidate's artifacts/global gate/duplicate/parity/stress evidence and non-default decision, internal/external grading, M2, calibration, Grad-CAM QC, selective prediction, full MATLAB runtime gates, SimEvents/Python cross-check, intended use, and known failures.
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

## P4 — Product surface and submission material

- [ ] Build the interactive demo UI. **Deferred by user.**
- [x] Build an evidence-backed six-slide draft and PDF. Further presentation editing and submission are user-owned; pipeline work must not modify them unless requested again.
- [ ] Replace `[ADD REGISTERED ID]` on slide 1 with the registered SIH portal Team ID before submission.
- [ ] Run a final team rehearsal against the slide notes and verify every member can explain the external IDRiD gap, MATLAB parity warning, M2 limitations, and M5 assumptions.
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
- [x] Add and execute the EyeQ quality-calibration runner; retain its fitted thresholds as evidence-only because the held-out gate failed.
- [x] Add analysis-grid quadratic sub-pixel MA coordinates in Python and MATLAB.
- [x] Add a machine-readable, artifact-hashed Validation Passport.
- [x] Add a tested MATLAB/Python runtime comparator that separates screening-decision parity, strict five-grade parity, threshold-edge mismatches, and probability tolerances.
- [x] Make the full EyeQ feature run resumable and reject incomplete/stale caches instead of silently calibrating on a partial dataset.
- [x] Add deterministic M1 corruption stress testing and a four-model handcrafted-feature ablation without deploying a weak candidate.
- [x] Complete the aligned 733-image MATLAB holdout and publish layered runtime-parity evidence instead of treating small-smoke agreement as proof.
- [x] Produce and QA `presentation/Netrai_SIH_2026_Final.pptx` plus the six-page PDF submission render.
