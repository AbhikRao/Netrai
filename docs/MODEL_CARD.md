# NetrAI model card

## Deployed MATLAB model

The MATLAB prototype imports EfficientNet-B4 with a shared backbone, five-grade
output and binary referral output. The legacy APTOS Fold-0 checkpoint is exported
as [`netrai_fold0.onnx`](../matlab/models/netrai_fold0.onnx). Deployment thresholds,
normalization and fitted scalar confidence temperature are recorded in
[`netrai_config.json`](../matlab/models/netrai_config.json). The original training
recipe is preserved in [training provenance](../research/README.md).

The two heads share their representation. Disagreement routes to human review;
it is not independent model consensus. The learned image-quality candidate is
separate and remains disabled in the default MATLAB pipeline.

## Input and artifact identity

The deployed `legacy-area-v1` contract uses a foreground crop, pixel-area
resampling, uint8 Gaussian illumination normalization with reflected borders,
and single-precision ImageNet normalization. MATLAB regression fixtures cover
enlargement, reduction, unequal dimensions, crop boundaries and empty frames.
The preprocessing repair changes the MATLAB numerical implementation while
preserving the released reference contract, model weights and grading policy.

[`netrai_bundle.json`](../matlab/models/netrai_bundle.json) binds the model,
configuration, image decoding, preprocessing, grading, calibration and routing
source by SHA-256 (schema 2, seven required roles).
The loader checks content before reuse. A matching hash establishes artifact
identity; clinical acceptance still requires the evaluations below.
It also verifies the source bytes selected by MATLAB's function resolution for
decoding, preprocessing, grading, calibration and routing, including cached-model
reuse. A different implementation earlier on the MATLAB path fails the check.

## Recorded performance and limits

The September 28, 2026 MATLAB model-mode development evaluation processed all
733 images and recorded sensitivity 94.63%, specificity 94.02%, QWK 0.9330 and
AUC 0.9842.
The cohort informed model/policy selection and contains 35 images with exact
training counterparts.
The [validation report](VALIDATION.md) links the complete aggregate evidence.

Mean per-image CPU time was 2.33 seconds in model mode, including verified
decoding, quality assessment, preprocessing and grading. It excludes retinal
analysis, Grad-CAM, report generation, clinician review and the initial model
import; this is development-run timing rather than a measured clinical workflow.
[Run metrics](../results/benchmarks/matlab_runtime_2026-09-28/matlab_run_metrics.json)

All 733 grades and screening routes match the reference runtime after the
`legacy-area-v1` preprocessing repair. Maximum absolute probability difference
is 2.82 × 10⁻⁶, passing the unchanged 0.05 tolerance. This verifies numerical
consistency for the recorded model-mode cohort; it does not validate the entire
image pipeline or establish new independent accuracy. Model weights, calibration
and routing policy were unchanged.
[Runtime comparison](../results/benchmarks/matlab_runtime_2026-09-28/runtime_comparison.json)

The legacy supporting reference runtime scored sensitivity 79.69%, specificity
82.05%, QWK 0.6652 and AUC 0.8746 on 103 IDRiD grading-test images. This is not
an external-cohort MATLAB result. The test had prior development exposure and
the screening acceptance targets remain unmet.

A separately trained corrected candidate scored 84.38% sensitivity, 69.23%
specificity, QWK 0.5843 and AUC 0.8654 on those same 103 images, with Grade-4
recall 2/13. It remains non-default and has no MATLAB export.
[Candidate evidence](../results/benchmarks/grading_candidate/summary.json)

The candidate uses a corrected loss and a separate preprocessing contract. Its
calibration partition fitted its policy, so calibration scores are not independent
validation. Comparisons with the legacy workflow cannot isolate one training change.

## Intended use

Reproducible research and prototype demonstrations. The system has not been
validated for diagnosis, treatment selection or autonomous patient clearance.
Lesion overlays are unverified candidates. Grad-CAM targets model attribution;
it does not establish lesion boundaries or clinical explanation usefulness.
Confidence calibration is post-hoc internal evidence, not demonstrated cross-site
reliability. Attribution QC warnings remain visible in generated reports.

Fresh identity-reviewed external grading, portable-camera robustness, severe-grade
recall, broader full-image runtime checks and clinician review-time studies remain open.
See [research protocol](RESEARCH_PROTOCOL.md) for evaluation boundaries.

## Capacity-model scope

The separate SimEvents endpoint-v3 planning model preserves patient identity
through at most two recapture returns. Abandonment, exhausted retries and patients
still queued or in service are unresolved outcomes. All 18 native annual scenarios
pass patient conservation; seven deterministic fixtures match a separate reference,
and three native seeded runs verify repeatability and routing-seed independence.
These scopes do not establish annual cross-runtime agreement or field calibration.

The published scenarios assume 56 Mb uploads, 45-second acquisition attempts,
120-second recapture delays, 3.5-second inference, 30-second review and zero
dropout. They use working-hour-equivalent time and omit shifts, outages and timed
escalation. Completion is a workflow outcome, not diagnostic accuracy or evidence
for autonomous clinical clearance.
[Workflow evidence and assumptions](../results/benchmarks/matlab_workflow_2026-09-28/workflow_evidence_summary.json)

## Artifact terms

Original source and documentation use [MIT](../LICENSE). Training data,
pretrained backbones, checkpoints and medical-image fixtures retain separate terms;
see [licensing scope](../LICENSE_STATUS.md). Load model artifacts only from trusted
sources and verify their recorded hashes.
