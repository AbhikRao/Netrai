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

[Aggregate metrics](../results/verification_2026-09-20/matlab_fold0_733_aligned/aggregate_metrics.json)

The cohort informed checkpoint/policy selection. A later exact-content audit found
35 evaluation images with training counterparts; these figures are development
evidence and cannot establish independent accuracy. No amount of metric inflation
has been estimated. [Overlap audit](../results/implementation_2026-09-24/legacy_aptos_overlap.json)

Timing excludes report generation and clinician review. The fixed ONNX fixture
passes, but it does not establish whole-pipeline equivalence. The recorded
733-case comparison matched 717 grades and 724 triage actions; numerical and
exact-decision acceptance gates remain failed.
[Runtime comparison](../results/verification_2026-09-20/matlab_python_fold0_733_aligned/runtime_comparison.json)

## External-domain development evidence

The legacy reference pipeline evaluated 103 IDRiD grading-test images: sensitivity
79.69%, specificity 82.05%, QWK 0.6652 and AUC 0.8746.
[Recorded reference metrics](../results/verification_2026-09-18/idrid_external_grading/aggregate_metrics.json)
These are reference-runtime results, not a MATLAB external-cohort measurement.
They miss the challenge's >90% sensitivity and >85% specificity targets.

A separately trained corrected candidate scored 84.38% sensitivity, 69.23%
specificity and QWK 0.5843 on the same 103 images; Grade-4 recall was 2/13.
[Candidate summary](../results/verification_2026-09-26/grading_idrid_test/summary.json)
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

[Scenario JSON](../results/implementation_2026-09-24/m5_simevents/simulink_scenario_sweep.json) · [Scenario CSV](../results/implementation_2026-09-24/m5_simevents/simulink_scenario_sweep.csv)

The simulation assumes acquisition, transfer, inference and review service times.
It does not demonstrate field benefit or cost savings. A recapture-return loop,
timed escalation and field-derived assumptions remain future work.

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
