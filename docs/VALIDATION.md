# Validation and prototype evidence

NetrAI has implementations for acquisition quality, retinal analysis, grading,
explanation reporting and capacity simulation. This describes component coverage,
not a percentage of clinical readiness or completion of all challenge requirements.

## MATLAB grading

The recorded model-only evaluation processed all 733 legacy APTOS Fold-0 images
with zero errors:

| Metric | Value |
| --- | ---: |
| Referable-DR sensitivity | 94.97% |
| Referable-DR specificity | 93.33% |
| Quadratic weighted kappa | 0.9297 |
| Referable AUC | 0.9841 |
| Mean model time | 1.55 s/image |

[Aggregate metrics](../results/benchmarks/matlab_grading/aggregate_metrics.json)

The cohort informed checkpoint/policy selection. A later exact-content audit found
35 evaluation images with training counterparts; these figures are development
evidence and cannot establish independent accuracy. No amount of metric inflation
has been estimated. [Overlap audit](../results/benchmarks/split_integrity/legacy_aptos_overlap.json)

Timing excludes report generation and clinician review. The fixed ONNX fixture
passes, but it does not establish whole-pipeline equivalence. The recorded
733-case comparison matched 717 grades and 724 triage actions; numerical and
exact-decision acceptance gates remain failed.
[Runtime comparison](../results/benchmarks/runtime_comparison/runtime_comparison.json)

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

The endpoint-v2 sweep covers 18 combinations of bandwidth, clinician count and
quality-rejection assumptions, with 100,000 intended annual patients and 100,001
scheduled arrivals. Completion means automatically routed or clinician-reviewed;
recapture requests remain unresolved.

| One clinician, 10% assumed quality rejection | Simulated completion |
| --- | ---: |
| 0.25 Mbps | 32.14% |
| 1 Mbps | 89.99% |

[Scenario JSON](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.json) · [Scenario CSV](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.csv)

The simulation assumes acquisition, transfer, inference and review service times.
It does not demonstrate field benefit or cost savings. A recapture-return loop,
timed escalation and field-derived assumptions remain future work.

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
- Complete runtime comparison with fixed case coverage and numerical gates.
- Clinician explanation usefulness and measured review time.
- Recapture-return and referral-escalation workflow validation.

Candidate masks are not confirmed diagnoses. Grad-CAM is model attribution,
not validated lesion localization. Disagreement-to-review is an implemented rule;
its clinical safety benefit has not been established.
