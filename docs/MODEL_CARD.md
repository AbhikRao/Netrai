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

## Recorded performance and limits

The 733-image MATLAB model-only development evaluation recorded sensitivity
94.97%, specificity 93.33%, QWK 0.9297 and mean execution time 1.55 s/image.
The cohort informed model/policy selection and contains 35 images with exact
training counterparts. Timing excludes report generation and clinician review.
The [validation report](VALIDATION.md) links the complete aggregate evidence.

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
recall, full runtime equivalence and clinician review-time studies remain open.
See [research protocol](RESEARCH_PROTOCOL.md) for evaluation boundaries.

## Artifact terms

Original source and documentation use [MIT](../LICENSE). Training data,
pretrained backbones, checkpoints and medical-image fixtures retain separate terms;
see [licensing scope](../LICENSE_STATUS.md). Load model artifacts only from trusted
sources and verify their recorded hashes.
