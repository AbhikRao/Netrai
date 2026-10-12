# Validation and prototype evidence

NetrAI has implementations for acquisition quality, retinal analysis, grading,
explanation reporting and capacity simulation. This describes component coverage,
not a percentage of clinical readiness or completion of all challenge requirements.

## MATLAB grading

The September 28, 2026 model-mode evaluation processed all 733 legacy APTOS
Fold-0 images with zero errors and verified complete expected-cohort coverage:

| Metric | Value |
| --- | ---: |
| Referable-DR sensitivity | 94.63% |
| Referable-DR specificity | 94.02% |
| Quadratic weighted kappa | 0.9330 |
| Referable AUC | 0.9842 |
| Mean model-mode time on CPU | 2.33 s/image |

[Aggregate metrics](../results/benchmarks/matlab_runtime_2026-09-28/combined_metrics.json)

The per-image timer includes image-byte verification/decoding, native quality
assessment, preprocessing, inference, calibration and routing. It excludes
retinal analysis, Grad-CAM, report generation, clinician review and the initial
model import. Timing is from this development run on the tested CPU host.
[Run metrics](../results/benchmarks/matlab_runtime_2026-09-28/matlab_run_metrics.json)

The cohort informed checkpoint/policy selection. A later exact-content audit found
35 evaluation images with training counterparts; these figures are development
evidence and cannot establish independent accuracy. No amount of metric inflation
has been estimated. [Overlap audit](../results/benchmarks/split_integrity/legacy_aptos_overlap.json)

The matched comparison passes all three declared gates: exact five-grade outputs,
screening decisions and probability tolerance. All 733 grades, grade-derived
referral decisions, referral-head decisions, disagreement flags and triage actions
match. The maximum absolute probability difference is 2.82 × 10⁻⁶, below the
unchanged 0.05 tolerance; probabilities are not bit-identical.
[Current runtime comparison](../results/benchmarks/matlab_runtime_2026-09-28/runtime_comparison.json)

The repair reproduces `legacy-area-v1` preprocessing through area resampling,
fixed-point uint8 Gaussian filtering and integer illumination blending. All 41
synthetic operator cases and two identical decoded real arrays match the reference
operations exactly. Model weights, calibration and routing policy are unchanged.
[Operator verification](../results/benchmarks/matlab_runtime_2026-09-28/operator_verification.json)

This closes model-mode runtime consistency on the recorded cohort. It does not
establish quality-gate, retinal-analysis or report equivalence, independent
accuracy, portable-camera robustness or clinical safety. The earlier comparison
with 717/733 grades and 724/733 triage actions remains
[historical failed evidence](../results/benchmarks/runtime_comparison/runtime_comparison.json).

## External-domain development evidence

The legacy reference pipeline evaluated 103 IDRiD grading-test images: sensitivity
79.69%, specificity 82.05%, QWK 0.6652 and AUC 0.8746.
[Recorded reference metrics](../results/benchmarks/reference_grading/aggregate_metrics.json)
These are reference-runtime results, not a MATLAB external-cohort measurement.
They miss the challenge's >90% sensitivity and >85% specificity targets.

A separately trained corrected candidate scored 84.38% sensitivity, 69.23%
specificity and QWK 0.5843 on the same 103 images; Grade-4 recall was 2/13.
[Candidate summary](../results/benchmarks/grading_candidate/summary.json)
It also fails screening acceptance and has not replaced the deployed MATLAB model.
The workflows differ, so the change cannot isolate a loss-function effect.
IDRiD test outcomes were already inspected during development. Neither result
constitutes fresh external or prospective clinical validation.

### DeepDRiD development validation (MATLAB, September 28)

The deployed MATLAB pipeline processed the DeepDRiD v1.1 regular-fundus
**validation** partition: 400 requested images from 100 patients, four images of
one patient excluded before prediction for an identity overlap with the reserved
evaluation set, 396 processed and zero model errors. This is the first result on
a source new to the project, but it is a development partition, not the reserved
evaluation set, and it is now exposed.

| Model-only endpoint (396 images) | Value |
| --- | ---: |
| Referable sensitivity | 63.33% (114/180) |
| Referable specificity | 96.76% (209/216) |
| Referable AUC | 0.910 |
| Quadratic weighted kappa | 0.749 |
| Grade-2 / Grade-4 recall | 32/92 / 6/20 |

With the quality gate, 61 of 180 known-referable images were auto-cleared and 20
remained unresolved (recapture or head disagreement). The model ranks referable
disease reasonably (AUC 0.91) but systematically under-grades this camera domain,
so the screening sensitivity target fails by a wide margin. Metrics are
image-level without confidence intervals; views of one patient are correlated.
[Summary](../results/benchmarks/deepdrid_development/summary.json),
[confusion matrix](../results/benchmarks/deepdrid_development/confusion_matrix.csv),
[per-grade metrics](../results/benchmarks/deepdrid_development/per_grade_metrics.csv)

## Round 2: full-network models on two fresh tests (October 12)

Four EfficientNet-B4 models were fine-tuned end to end on GPUs (Kaggle and an
Apple M4) from the deployed weights, with the architecture, input contract and
six-output ONNX schema unchanged. Training used up to 24,641 images (APTOS, EyePACS,
IDRiD, DeepDRiD, and for two models DDR train/valid), acquisition-style
augmentation (rotation, field-of-view shift, exposure, colour, blur, JPEG) and
source- and grade-balanced sampling. The primary candidate, `K3` (first four
backbone stages frozen, DDR added, seed 17), was chosen on development validation by
a rule fixed in advance; thresholds were fitted on validation data only.

Everything was then locked (53 artifacts: predictions, checkpoints, policies, code,
protocol, label-file hashes) and two never-used tests were scored once:
**Messidor-2** with adjudicated grades (Krause et al., 2018; a camera no model saw,
874 exams) and the **DDR** test split (4,105 images; in-dataset for `K3`, which trained
on DDR train/valid). Identity reviews found no overlap with any training image.

| Messidor-2 (1,744 gradable images) | Sensitivity (95% CI) | Specificity (95% CI) | AUC | Sens. at 85% spec. |
| --- | --- | --- | ---: | ---: |
| Deployed model | 54.1% (48.7–59.4) | 96.3% (95.1–97.4) | 0.909 | 82.3% |
| **`K3` primary operating point** | **93.9% (91.7–95.9)** | **69.3% (66.5–72.2)** | **0.937** | **88.2%** |
| `K3` balanced (Youden) point | 91.9% (89.4–94.4) | 76.5% (73.8–79.1) | 0.937 | 88.2% |

| DDR test (3,759 gradable images) | Sensitivity | Specificity | AUC |
| --- | --- | --- | ---: |
| Deployed model | 34.4% | 99.2% | 0.891 |
| **`K3`** | **86.1%** (84.5–87.7) | **94.3%** (93.3–95.3) | **0.962** |

Messidor-2 intervals are exam-cluster bootstrap; DDR has no patient identifiers, so
its intervals are image-level and likely optimistic. AUCs use the expected-grade score.

What this shows:

- On a camera never seen in training, `K3` ranks referable disease better than the
  deployed model (AUC 0.937 vs 0.909) and is far more sensitive (+39.8 points, paired
  interval 34.8–45.0), at a cost of 27.0 specificity points.
- The thresholds, fitted on other cameras, are too low for Messidor-2: specificity
  is 69%, below the 85% target. Even at the best possible threshold, sensitivity at 85%
  specificity is 88.2%, so both targets are narrowly missed. Many of the false
  positives are mild (grade 1) eyes graded as referable; `K3` never predicts grade 1.
- Per the predeclared rule, `K3` is therefore a **research variant, not the default**.
  A single global threshold still does not transfer across cameras; per-camera threshold
  calibration with a small labelled set remains the practical deployment requirement.
- MATLAB reproduces `K3`'s logits within 1.1e-5 on label-free test images. Messidor-2
  and DDR test are now exposed and cannot serve as fresh tests again.

[Locked report](../results/benchmarks/round2_fresh_tests/report.json),
[variants and policies](../results/benchmarks/round2_fresh_tests/),
[identity reviews](../results/benchmarks/round2_fresh_tests/messidor_identity_review.json)

## Locked DeepDRiD evaluation of site-adapted candidates (October 11)

A new grading candidate was developed without the reserved data and then tested
once on the DeepDRiD v1.1 regular-fundus **evaluation** set, under an acceptance
lock that bound the protocol, the five deployment bundles, the decision policies
and the 50-file MATLAB evaluation source list before any label was read. The
evaluation ran in MATLAB, and an independent Python recomputation from the
per-image outputs matches every confusion count.

How the candidates were built: the deployed EfficientNet-B4 architecture, input
contract and six-output ONNX schema are unchanged. The backbone stays frozen;
its head layers (`C`) or last stage plus heads (`D`) were retrained on EyePACS
(EyeQ train, gradable), IDRiD train and DeepDRiD train images. Decision
thresholds were then calibrated on DeepDRiD **validation** (site calibration).
The candidate `C_site` was chosen by a rule fixed before `D` finished training.
The first lock halted at label attachment because the workbook column is named
`DR_Levels` rather than the documented `DR_Level`; only the header and image IDs
were inspected, and the parser was corrected and re-locked before any grade value
was read.

Primary population: 392 images from 98 patients, all gradable (four images each
from two patients are kept in coverage but excluded for identity review). Model
only; intervals are 95% patient-cluster bootstrap.

| Variant | Sensitivity | Specificity | AUC | QWK |
| --- | --- | --- | ---: | ---: |
| Deployed legacy model | 62.8% (52.2–73.1) | 96.9% (94.6–99.0) | 0.951 | 0.812 |
| Legacy + site thresholds | 92.7% (87.5–97.1) | 82.9% (75.6–89.2) | 0.951 | 0.809 |
| **`C_site` primary candidate** | **93.3% (88.4–97.5)** | **89.0% (83.6–93.9)** | **0.971** | **0.865** |
| `C_site` recipe without DeepDRiD training | 93.9% (89.2–97.7) | 79.4% (72.5–85.7) | 0.946 | 0.787 |
| `D_site` secondary (information only) | 95.1% (90.3–98.9) | 90.8% (85.4–95.3) | 0.975 | 0.891 |

`C_site` meets the >90% sensitivity and >85% specificity targets on its point
estimates; the lower interval bounds (88.4% / 83.6%) do not exclude missing them.
Against the deployed model it gains 30.5 sensitivity points (paired interval
21.5–39.7) and loses 7.9 specificity points (3.3–12.9). With the quality gate,
only 7 of 164 referable images are auto-cleared (deployed model: 58). Grade-4
recall remains weak (4/20). AUCs use the expected-grade score.

What this shows and what it does not:

- The gain needs labelled images from the target camera: the same recipe without
  DeepDRiD training images fails specificity, and thresholds fitted on other
  sources did not transfer during development. A deployment would need a
  calibration set from each camera.
- It is a retrospective public benchmark, not prospective or rural-India
  validation, and the DeepDRiD evaluation set is now exposed.
- The deployed MATLAB model is unchanged; promoting a candidate is a separate
  decision. On the same 25 audit images and the current QC code, Grad-CAM
  shortcut flags fall from 80% (deployed model) to 64% (`C_site`); flags remain
  common. The earlier 13/25 (52%) figure below used the previous QC rules, so the
  two audits are not directly comparable.

[Acceptance report](../results/benchmarks/deepdrid_acceptance/acceptance_report.json),
[independent recomputation](../results/benchmarks/deepdrid_acceptance/python_crosscheck.json),
[development ladder and policies](../results/benchmarks/model_adaptation/)

## SimEvents capacity scenarios

The September 28 endpoint-v3 sweep completed all 18 combinations of 0.25/1/5 Mbps,
one/two reviewers and 5%/10%/20% assumed quality rejection. A rejected patient
returns to the same acquisition queue with the same identity, with at most two
recaptures. Completion counts disjoint original patients automatically routed or
reviewed; abandonment, exhausted retries and patients still in the system remain
unfinished. This completion endpoint does not measure correct diagnosis.

All 18 native MATLAB scenarios satisfy
`generated = auto_cleared + reviewed + abandoned + quality_exhausted + in_system`,
with in-system counts checked against independent queue/server occupancy sums.
Attempt and return counters have separate denominators.
[Conservation checks](../results/benchmarks/matlab_workflow_2026-09-28/annual_conservation_checks.json)

The annual horizon is 250 × 8 working-hour-equivalent hours, with a 72-second
arrival interval: 100,000 target patients and 100,001 observed arrivals including
the start and cutoff. Each patient uploads two 3.5 MB images (56 Mb total).
Acquisition takes 45 seconds per attempt, the return delay is 120 seconds,
inference takes 3.5 seconds and review takes 30 seconds. Initial and retry
rejection probabilities are equal; dropout is explicitly zero. Referral and
uncertain routing probabilities are 8% and 1.23%, respectively. These are
configured assumptions, not fitted field measurements.

| One reviewer, 10% assumed rejection | Completed / generated | Quality-exhausted | In system at cutoff | Completion |
| --- | ---: | ---: | ---: | ---: |
| 0.25 Mbps | 32,142 / 100,001 | 87 | 67,772 | 32.14% |
| 1 Mbps | 99,893 / 100,001 | 105 | 3 | 99.89% |
| 5 Mbps | 99,904 / 100,001 | 95 | 2 | 99.90% |

[Scenario JSON](../results/benchmarks/matlab_workflow_2026-09-28/simulink_scenario_sweep.json) · [Scenario CSV](../results/benchmarks/matlab_workflow_2026-09-28/simulink_scenario_sweep.csv)

At 0.25 Mbps, the 224-second upload service exceeds the 72-second arrival interval;
the upload backlog dominates, and adding a second reviewer does not increase
completed throughput. Under the assumed 30-second review and 9.23% review demand,
reviewer utilization is low. It cannot establish real staffing requirements.
Each annual scenario uses different seeds, so small differences at adequate
bandwidth are not paired causal effects or estimates of uncertainty across seeds.

Verification has three distinct scopes. Seven deterministic native fixtures cover
no rejection, zero retries, successful return, retry exhaustion, abandonment,
a return pending at cutoff and review congestion. Their counts, return occupancy,
mean queue wait and cutoff utilization match a separate event-calendar reference.
Three native stochastic smoke runs verify identical-seed repeatability and that
changing only the routing seed preserves upstream quality events. The 18 annual
scenarios were checked for native conservation; they were not compared against
an independent annual runtime.
[Fixture summary](../results/benchmarks/matlab_workflow_2026-09-28/fixture_summary.json) · [Independent fixture comparison](../results/benchmarks/matlab_workflow_2026-09-28/independent_fixture_comparison.json) · [Repeatability summary](../results/benchmarks/matlab_workflow_2026-09-28/repeatability_summary.json)

Mean review queue wait covers patients who started review and is separate from
service time; it is not a patient wait percentile. Censored waiting patients stay
in the in-system count. The model does not implement closed hours, shifts, outages
or timed escalation, and does not demonstrate field benefit or cost savings.
The [endpoint-v2 sweep](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.json)
and its [earlier reference comparison](../results/benchmarks/capacity_simulation/reference_crosscheck.json)
remain historical; their unresolved-recapture definition is different.

## Retinal analysis and image quality

The supporting reference runtime was evaluated against native IDRiD annotation
masks on 27 test images. Recorded Dice was 0.0134 for microaneurysms, 0.0170 for
haemorrhage, 0.2037 for exudates and 0.8603 for optic disc.
[Native-mask metrics](../results/benchmarks/retinal_analysis/idrid_native_mask_metrics.csv)
These weak lesion baselines support retaining the candidate label on overlays.
They are not measured MATLAB segmentation accuracy.

The non-default learned quality model recorded macro-F1 0.8289 and Reject recall
86.37% on 16,249 EyeQ test images.
[Quality metrics](../results/benchmarks/image_quality/quality_model_metrics.json)
Five confirmed same-retina pairs cross that official split, and its test outcomes
were inspected during development.
[Overlap audit](../results/benchmarks/image_quality/overlap_impact.json)

On a 250-image paired synthetic stress audit, that candidate rejected only 0.4%
of defocused images, 7.6% of low-light images and 0% of JPEG-Q25 images. It remains
disabled in the default MATLAB pipeline; global quality-label performance does
not establish reliable acquisition-defect detection or portable-camera robustness.
[Stress results](../results/benchmarks/image_quality/learned_quality_robustness.json)

The reference attribution audit flagged 13 of 25 grade-stratified Grad-CAM maps
for possible border/FOV/corner attention. This is a heuristic review signal,
not an established error rate. MATLAB reports also display their per-image QC
warnings; clinician explanation usefulness remains unmeasured.
[Attribution QC](../results/benchmarks/explanation_qc/gradcam_qc_summary.json)

## Acceptance work remaining

- Portable-camera quality validation and factor-specific robustness.
- Reliable annotated lesion evaluation, including appropriate subtype/NV labels.
- Prospective, camera-specific validation of the site-adapted candidate and severe-grade (Grade-4) recall.
- Broader full-image runtime checks, including quality and retinal analysis.
- Clinician explanation usefulness and measured review time.
- Field-derived recapture-return assumptions, shift/outage modeling and timed referral escalation.

Candidate masks are not confirmed diagnoses. Grad-CAM is model attribution,
not validated lesion localization. Disagreement-to-review is an implemented rule;
its clinical safety benefit has not been established.
