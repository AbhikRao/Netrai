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

## Site-adapted candidate (not deployed)

`matlab/variant_C_site/` holds a research candidate with the same architecture,
input contract and ONNX schema. Its head layers were retrained on EyePACS, IDRiD
and DeepDRiD training images over a frozen backbone, and its thresholds were
calibrated on DeepDRiD validation. In one locked DeepDRiD evaluation it reached
93.3% sensitivity and 89.0% specificity (see [validation](VALIDATION.md)). It
depends on target-camera calibration data, has weak Grade-4 recall, and is not
the default model. Its ONNX weights are kept outside Git.

## Round-2 candidate K3 (not deployed)

`matlab/variant_R2_K3_freeze4_ddr_s17/` holds a full-network fine-tuned candidate
(same architecture and ONNX schema; first four backbone stages frozen; trained on
APTOS, EyePACS, IDRiD, DeepDRiD and DDR train/valid). In one locked test it reached
93.9% sensitivity / 69.3% specificity on Messidor-2 and 86.1% / 94.3% on DDR test
(see [validation](VALIDATION.md)). Specificity on the unseen camera is below target and
grade 1 is never predicted, so it is a research variant; its weights are kept outside Git.

## E_768: presented default, with required site calibration (owner decision, October 25, 2026)

`matlab/variant_R3_E_768/` holds the mean of two full-network models trained at 768 × 768 pixels (same architecture, `legacy-area-v1`
preprocessing at 768 px, one two-member ONNX graph with the standard six outputs; weights, 143 MB, are kept outside Git).
Locked results (all against the deployed legacy model): ODIR-5K 90.5% / 84.5% (AUC 0.948 vs 0.885); mBRSET smartphone images 89.2% / 89.3%
(AUC 0.948 vs 0.918; legacy 52.6% / 96.6%); BRSET desktop cameras 98.5% / 56.0% (AUC 0.957 vs 0.887; legacy 76.8% / 84.6%). No pre-registered
rule (> 90% / > 85% at the frozen cut-off) was met, so this is a product decision informed by the results, not a promotion by rule. What the
evidence supports: E_768 **ranks** referable DR better on three fresh external sets; its frozen **cut-off does not transfer** between cameras, so
a per-site calibration step is required and every report is marked `UNCALIBRATED` until one is loaded. On two further fresh sets (Paraguay, Zeiss;
SUSTech-SYSU, Topcon; both easier than screening populations) its frozen cut-off met both targets (96.7% / 96.8% and 99.4% / 95.6%), and the
calibration rule with a 95% sample-sensitivity target was confirmed there (both targets met in 93% of 200-image samples; `calibrate_site.py` default).
Calibration keeps sensitivity but cannot raise specificity on cameras where the model ranks poorly. Site onboarding (MATLAB `calibrateSite` then
`verifySitePolicy` on a separate sample, or the same two steps in `python/calibrate_site.py`) is described in [README_MATLAB](../matlab/README_MATLAB.md).
E_768 also over-refers non-DR retinal disease
(BRSET specificity 0.47 with another labelled disease vs 0.64 without; RFMiD any-DR AUC 0.758 vs 0.844 for legacy), and grade 1 is almost never
predicted. The legacy model remains available (`Variant='legacy'`). See [validation](VALIDATION.md).

## Obtaining the E_768 weights and their terms

The 143 MB ONNX file is not part of the Git history. It is the asset of the GitHub release [`e768-v1`](https://github.com/AbhikRao/Netrai/releases/tag/e768-v1) (research and evaluation use only). `scripts/fetch_e768.sh` (or `fetchE768Weights` in MATLAB) downloads it and verifies the SHA-256 recorded in the committed `matlab/variant_R3_E_768/netrai_bundle.json`, installing nothing on a mismatch. Without the file the public checkout runs the legacy model (`Variant='auto'` in MATLAB, `--variant auto` in Python). Python: `python inference.py --image IMG --variant E_768 [--site-policy site_policy.json]` runs the ONNX graph; its Grad-CAM needs the two PyTorch member files of the same weights (`scripts/fetch_e768.sh --with-gradcam`, hash-checked against `matlab/variant_R3_E_768/torch_members.json`), otherwise the report says the map is unavailable.

**The MIT licence covers this project's original code and documentation, not the weights.** The weights were trained from an ImageNet-pretrained
EfficientNet-B4 backbone on APTOS 2019, EyePACS/EyeQ, IDRiD, DeepDRiD and DDR training images, and were selected on validation images from
those sources plus Messidor-2. Each source has its own access/redistribution terms (see `data/README.md`); the project has **not** verified
that any of them permits redistribution of derived models, so the weights are offered for research and evaluation only, without any warranty of
rights, and anyone who uses them must check the sources' terms. (RETFound weights, CC BY-NC 4.0, were used only for a negative probe and are not part of
E_768.) No patient images or per-image outputs from the credentialed PhysioNet sets (mBRSET, BRSET) are included in any release.

## Round-4 candidate R4_768_s17 (not deployed)

`matlab/variant_R4_768_s17/` continues the round-3 768-pixel model with ODIR and more EyePACS
training data. In one locked test on FDR26 (Morocco, cameras unseen in training) it reached 78.0%
sensitivity / 92.4% specificity (AUC 0.910; deployed model 0.911, round-3 E_768 0.912). About half of
moderate (grade-2) eyes are missed, and ranking is weaker on two rectangular-format image groups (likely camera or capture differences). It is a
research variant; its weights are kept outside Git. See [validation](VALIDATION.md).

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

## Additional locked-test findings (October 22, 2026)

- **Other retinal disease:** on RFMiD 2.0 (any-DR label, 70 DR eyes of 853) the deployed model ranks DR at AUC 0.844 and the 768-pixel research variants at 0.72–0.76; DR versus normal eyes is separated almost perfectly by all (AUC ≥ 0.994), but eyes with other conditions (vascular occlusions, retinitis, detachment) are flagged as DR-like, more so by the research variants. In screening such eyes would be referred anyway, but this is not a DR-specific result. See [validation](VALIDATION.md).
- **Unseen sources:** one decision threshold does not transfer across sites; per-site calibration with about 100–200 labelled clusters restores sensitivity but not always specificity.
- **Learned lesion candidates (opt-in, research):** pixel Dice on a locked test 0.60 (exudates), 0.43 (haemorrhages), 0.26 (microaneurysms); overlays are candidate evidence, not ground truth.
