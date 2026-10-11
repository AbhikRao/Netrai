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
- Fresh, identity-reviewed external grading and severe-grade recall.
- Broader full-image runtime checks, including quality and retinal analysis.
- Clinician explanation usefulness and measured review time.
- Field-derived recapture-return assumptions, shift/outage modeling and timed referral escalation.

Candidate masks are not confirmed diagnoses. Grad-CAM is model attribution,
not validated lesion localization. Disagreement-to-review is an implemented rule;
its clinical safety benefit has not been established.
