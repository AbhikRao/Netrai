# Run NetrAI in MATLAB

The primary deployment uses a trained EfficientNet-B4 ONNX model for grading,
native image-processing functions for retinal analysis, and SimEvents for district
capacity simulation. The two prediction heads share a backbone; disagreements
route to human review.

## Requirements

Tested on MATLAB R2026a Update 5 with:

- MATLAB, Deep Learning Toolbox and Image Processing Toolbox.
- Deep Learning Toolbox Converter for ONNX Model Format support package.
- Simulink and SimEvents for the capacity model.

The image pipeline runs serially on CPU. Install the ONNX converter through
MATLAB **Add-Ons**, then check `which importNetworkFromONNX`.
The [recorded environment](../results/benchmarks/environment/matlab_environment.json) lists the tested products.

## Verify deployment

Open this repository's `matlab` folder and run in the **MATLAB Command Window**:

```matlab
fixtureReport = verifyMatlabPackage();
```

The verifier checks dependencies, imports the six-output model and compares a
fixed preprocessed input against reference outputs. If the local reference image
is available, it also checks its preprocessing-path grade. The fixture passes.
The September 28 model-mode comparison also passes on all 733 legacy APTOS
cases: grades and screening routes match exactly, and the maximum probability
difference is 2.82 × 10⁻⁶ under the unchanged 0.05 tolerance. This development
cohort has known training overlap; full-image and clinical validation are separate.
See [validation evidence](../docs/VALIDATION.md).

The schema-2 `models/netrai_bundle.json` binds seven roles: the ONNX model,
deployment configuration, image decoding, preprocessing, grading, calibration
and routing code by content hash. A missing
or changed artifact fails before inference, including same-size replacements.
The loader also checks the callable source bytes selected by MATLAB's current
path for decoding, preprocessing, grading, calibration and routing, including
cached-model reuse. A different implementation shadowing a bound function fails.
Restore the intended artifact if this check fails. After an intentional, reviewed
deployment change, `writeDeploymentBundle` creates a new identity; generating that
identity does not validate the model or authorize an experimental promotion.

The released input contract is `legacy-area-v1`: uint8 RGB, foreground crop,
area resampling, a fixed-point Gaussian with reflected borders, integer
illumination blending and single-precision ImageNet normalization. Synthetic
fixtures check the numerical operations, including enlargement and unequal image
dimensions. The decoder retains uint8 RGB colors, converts grayscale/bilevel
images to RGB, resolves indexed palettes to their colors and ignores transparency
without compositing the color pixels. Direct model preprocessing rejects floating-point/high-bit-depth
inputs instead of silently quantizing them.

## Demonstrate one image

Images are not bundled; obtain a local image under terms that permit your use.
Enter this command in MATLAB's Command Window:

```matlab
result = demo_netrai("/absolute/path/to/fundus.png");
```

Alternatively, from a **Linux terminal** at the repository root:

```bash
./scripts/launch_demo.sh /absolute/path/to/fundus.png
```

This opens MATLAB, runs the demo and opens the capacity model. MATLAB syntax
cannot run directly in Bash. The first model import can take several minutes.

The demo enforces quality assessment and uses a new report directory each run.
If quality is rejected, it returns acquisition feedback and `triageAction="recapture"`.
No grade or grading PDF is produced for that image. For an accepted image, it
shows the analysis and generates a report. Open the report at `result.reportPath`.

For direct control:

```matlab
outputDir = tempname;
result = main_pipeline("/absolute/path/to/fundus.png", ...
    OutputDir=outputDir, Verbose=true, GenerateReport=true, ...
    RunGradCAM=true, EnforceQuality=true);
```

Output includes grade 0–4, grade probabilities, referral-head probability,
head disagreement, triage action, candidate retinal features and explanation
status. Grad-CAM failures are reported explicitly; no synthetic map is substituted.
A QC warning remains visible when attribution concentrates near the image border.

See [the recording guide](../docs/DEMO.md) for a short demonstration sequence.

## Batch validation

Datasets remain local; see [data sources and layout](../data/README.md).
To reproduce the legacy model-only internal evaluation in chunks:

```matlab
for first = 1:100:733
    validatePipeline('../data/aptos2019', ...
        Mode='model', StartIndex=first, MaxImages=100, ...
        SelectionCsv='models/fold0_validation.csv', ...
        OutputDir=fullfile('results','fold0_chunks',sprintf('chunk_%03d',first)));
end
summary = aggregateValidationRuns('results/fold0_chunks', 'results/fold0_combined');
```

Chunk merging validates identities, probabilities and routing fields and rejects
duplicate, missing or differently labelled cases. New runs bind the entire
selection and image-byte hashes before inference, and decode those verified
bytes. Model/config/source changes during a run prevent publication. Use fresh
output directories: existing evidence is never overwritten, and temporary
staging folders are excluded from merging. The validator records per-image
results, confusion matrix, per-grade measures and aggregate metrics. These 733
images are development data with known training overlap, not an independent test.

New schema-3 validation identities additionally preserve the exact configuration
JSON and its decision policy: ordinal grade thresholds, referral-head threshold
and confidence temperature. Merging verifies that the captured JSON matches its
hash, that the policy agrees with it, and that calibrated probabilities and
decisions follow those frozen values. Successful schema-3 rows require complete
raw and calibrated five-grade vectors and derived outputs; the aggregate records
`decision_policy_verified=true` only for this schema.

The [September 28 aggregate](../results/benchmarks/matlab_runtime_2026-09-28/combined_metrics.json)
and [runtime comparison](../results/benchmarks/matlab_runtime_2026-09-28/runtime_comparison.json)
record the complete 733-case rerun. The previous failed runtime comparison is
retained as historical evidence; it is superseded for this model-mode cohort.
Mean CPU time was 2.33 s/image, including byte verification/decoding, quality
assessment and grading, excluding retinal analysis, Grad-CAM, reports, clinician
review and the initial model import.
[Run metrics](../results/benchmarks/matlab_runtime_2026-09-28/matlab_run_metrics.json)

Historical schema-1 chunks cannot prove complete cohort coverage alone. Supply
`ExpectedSelectionCsv='models/fold0_validation.csv'` when merging a compatible
historical cohort; otherwise the aggregate is labelled as observed chunks only.
Unbound old chunks must be rerun. Schema-2 chunks bind cohort/image content but
do not acquire schema-3 policy verification retrospectively. Errors and recaptures
remain unresolved, and metrics with no eligible cases are JSON `null`.

Run the focused MATLAB regression checks with `runtests('tests')`.

For a small full-image check, including quality and retinal analysis:

```matlab
summary = validatePipeline('../data/aptos2019', ...
    Mode='full', MaxImages=5, OutputDir=tempname);
```

For an unlabelled image folder:

```matlab
outputTable = batchProcess('../my_fundus_images', 'results/batch_results.csv');
```

## SimEvents capacity model

```matlab
open_system('models/NetrAI_Telemedicine_Model.slx')
```

The endpoint-v3 model represents acquisition, bandwidth-constrained upload,
inference and clinician review. A rejected patient can return to the same shared
acquisition queue, preserving patient identity and incrementing a bounded attempt
counter. Abandonment and exhausted retries remain unresolved outcomes.

Close any open workflow diagram before these saved-artifact checks; the runner
rejects an already-loaded model to protect in-memory edits. Seven deterministic
fixtures check no rejection, zero retries, a successful return, exhausted retries,
abandonment, a return pending at cutoff and clinician congestion:

```matlab
addpath(genpath('src'))
fixtures = verifyTelemedicineWorkflow(tempname);
scenarioResults = runSimulinkScenarioSweep(tempname, 100000);
```

The saved SLX and its adjacent `.manifest.json` bind the builder and workflow
source. Existing models/results are preserved; a missing or changed identity
fails instead of rebuilding a legacy artifact automatically. For an intentional
rebuild, use a fresh model path:

```matlab
modelPath = setupSimulinkModel(fullfile(tempname, 'NetrAI_Telemedicine_Model.slx'));
oneScenario = runTelemedicineScenario(modelPath);
```

Completion counts disjoint original patients who were automatically routed or
reviewed. Abandoned, quality-exhausted and physically in-system patients remain
unfinished; repeated attempts have separate counters. Queue waiting time is
separate from service time, and reviewer utilization includes occupied service
at cutoff. Staffing is based on expected review demand. The annual horizon uses
working-hour-equivalent seconds; shifts, outages, timed escalation and field
assumptions remain separate work. Simulation outcomes do not establish field benefit.
The September 28 annual sweep completed all 18 native scenarios with zero patient
conservation residuals. The seven deterministic fixtures also match a separate
event-calendar reference; three native seeded smoke runs verify repeatability
and independence of upstream quality events from the routing seed. These are
different checks, not an 18-scenario cross-runtime claim.
[Annual scenario evidence](../results/benchmarks/matlab_workflow_2026-09-28/simulink_scenario_sweep.json) · [Conservation checks](../results/benchmarks/matlab_workflow_2026-09-28/annual_conservation_checks.json) · [Fixture comparison](../results/benchmarks/matlab_workflow_2026-09-28/independent_fixture_comparison.json) · [Repeatability](../results/benchmarks/matlab_workflow_2026-09-28/repeatability_summary.json)

The sweep assumes two 3.5 MB images per patient, 45-second acquisition attempts,
120-second return delay, 3.5-second inference, 30-second review, equal initial/retry
rejection and zero dropout. At one reviewer and 10% rejection, completion was
32.14% at 0.25 Mbps, 99.89% at 1 Mbps and 99.90% at 5 Mbps. The slow-link upload
backlog cannot be resolved by adding a reviewer. Per-scenario seeds differ; small
differences at adequate bandwidth are not paired effects. See the
[validation report](../docs/VALIDATION.md) for denominators and unresolved counts.
The [endpoint-v2 scenarios](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.json)
remain historical and use a different unfinished-recapture definition.

## Quality-model candidate

The separate `quality_model_candidate.onnx` fixture can be checked with
`verifyQualityModel`. That learned candidate is not enabled in `main_pipeline`:
its acquisition-defect stress results and portable-camera evidence are insufficient
for promotion. The default quality route remains the native heuristic assessment.

## Site-adapted grading candidate (opt-in)

The default model is unchanged. `variant_C_site/` holds a research candidate that
was tested once on DeepDRiD (93.3% sensitivity, 89.0% specificity; see
[validation](../docs/VALIDATION.md)). It is tuned for DeepDRiD-like cameras: use
it only where calibration images from the target camera support it. Its ONNX
weights are not distributed in this repository. With a local copy of
`variant_C_site/netrai_fold0.onnx`, run from the `matlab` folder in MATLAB's
Command Window:

```matlab
results = main_pipeline('/absolute/path/to/fundus.jpg', ...
    ModelPath=fullfile(pwd,'variant_C_site','netrai_fold0.onnx'), ...
    ConfigPath=fullfile(pwd,'variant_C_site','netrai_config.json'));
```

The variant's `netrai_bundle.json` binds the weights by SHA-256, so a different
file is refused.

## Default grading variant: E_768 with required site calibration

The project's presented screening model is `variant_R3_E_768/` (two 768 × 768 models averaged inside one ONNX graph). Two fresh Brazilian
tests ranked referable DR better than the legacy model (AUC 0.948 vs 0.918 on smartphone images, 0.957 vs 0.887 on desktop cameras), but no
frozen cut-off met the pre-registered 90% / 85% rule: on Canon desktop images it over-refers (specificity 0.43). **Its referral cut-off must be
calibrated on a labelled sample of each camera/site before screening**; every report says `UNCALIBRATED` until a site policy is loaded
([validation](../docs/VALIDATION.md)). The 143 MB weights exceed GitHub's file limit and are a release asset (`e768-v1`, research use only): run `fetchE768Weights` in MATLAB or `scripts/fetch_e768.sh`; both verify the SHA-256 from `variant_R3_E_768/netrai_bundle.json`. See [the model card](../docs/MODEL_CARD.md) for the weights' terms. The Python pipeline has the same options (`python inference.py --variant E_768 --site-policy FILE`). `main_pipeline` chooses the variant like this:

- `Variant='auto'` (default): E_768 when `matlab/variant_R3_E_768/netrai_fold0.onnx` exists locally, otherwise the legacy model;
- `Variant='E_768'` fails with a clear error when the weights are missing; `Variant='legacy'` always uses `models/netrai_fold0.onnx`;
- explicit `ModelPath`/`ConfigPath` (both, or neither) bypass the choice (variant reported as `custom`);
- a site policy is bound to the model it was fitted for (SHA-256 of model and config); loading it with another model is an error;
- **switching variants inside one MATLAB session needs `clear functions` first** (MATLAB's ONNX import cache).

Site calibration in MATLAB (from a labelled sample of the site; ungradable images removed; 40+ referable cases advised, 150+ non-referable):

```matlab
policy = calibrateSite(imagePaths, icdrGrades, "site_policy.json", SiteName="PHC Nanded", PatientIds=ids);   % E_768, 95% target
check  = verifySitePolicy(otherPaths, otherGrades, "site_policy.json", "site_check.json", PatientIds=otherIds);  % a SEPARATE sample
results = main_pipeline('/absolute/path/to/fundus.jpg', SitePolicy="site_policy.json");
```

Site onboarding: (1) grade a calibration sample of one camera (40–80+ referable eyes plus about 200 others; ungradable images removed);
(2) `calibrateSite` fits the cut-off, binds it to the model and records the calibration sample's identities; (3) `verifySitePolicy` checks it
on a second, separate sample and reports sensitivity and specificity with Wilson 95% intervals; (4) screen with `SitePolicy`.
Reports say `UNCALIBRATED` until step 2 has been done.

`calibrateSite` gives the same cut-off as the Python tool (difference 1.4e-6 on 60 fresh images; E_768 scores agree to 1.5e-6 and every grade and
referral decision matches — [parity](../results/benchmarks/matlab_grading/e768_matlab_python_parity_sustech.json)). The same in Python:

```bash
python python/calibrate_site.py score --images sample.csv --root /path/to/images --out logits.npy     # installed E_768 (scripts/fetch_e768.sh)
python python/calibrate_site.py fit --logits logits.npy --labels labels.csv --site "clinic A" --output site_policy.json
python python/calibrate_site.py score --images check.csv --root /path/to/images --out check.npy
python python/calibrate_site.py verify --policy site_policy.json --logits check.npy --labels check_labels.csv --output site_check.json
```

`score` runs the same ONNX graph, image reading and preprocessing as `inference.py --variant E_768` and writes the SHA-256 of every image next to
the logits (`logits.images.json`; without it, `fit`/`verify` accept `--images-root` plus an `image` column). Both runtimes write and check the same
identity fields: `calibration_image_sha256` (every calibration image; required) and, when patient ids are given (`patient` column /
`PatientIds`), `calibration_patient_sha256` (SHA-256 of each id; ids are never stored). Verification compares identities, not row order: it
refuses a sample that shares an image or a recorded patient with the calibration sample, a policy that records no calibration images, a
sample without image identities (Python) or without patient ids when the policy recorded them, and (Python) logits made by another model.
A policy fitted in Python is accepted and enforced by MATLAB and vice versa (both are bound to the same model SHA-256; cross-runtime test:
`tests/testVerifySitePolicyProvenance.m` with a fixture written by `calibrate_site.py fit`).

```matlab
results = main_pipeline('/absolute/path/to/fundus.jpg', SitePolicy='/path/to/site_policy.json');
disp(results.calibration.calibration_status)   % "site-calibrated (not independently validated)"
```

The policy replaces only the four ordinal boundaries (and the referral-head threshold if present); the bundle-bound configuration is never
edited, and the policy file's SHA-256 is recorded in `results.calibration`. A site policy is not a validation: re-check it on a separate
labelled sample. The default target is 95% sample sensitivity (confirmed on the fresh SUSTech-SYSU test; see [validation](../docs/VALIDATION.md)).
With about 80 referable plus 200 other patients it keeps mean sensitivity near 0.94; specificity then depends on how well the model ranks that camera
(BRSET Nikon 0.92, BRSET Canon 0.83, mBRSET 0.74; [post hoc analysis](../results/benchmarks/site_calibration_rules/enriched_tx.json)). At screening
prevalence (6% referable) 100–200 *patients* contain too few referable cases — collect a sample enriched with referable cases.

Run the legacy model explicitly with `Variant='legacy'`; opt-in alternatives such as `variant_C_site/` still work through
`ModelPath`/`ConfigPath`.

## Interpretation

NetrAI is a research prototype. Lesion outputs are candidates; Grad-CAM is
attribution rather than lesion-boundary evidence. Fresh external grading,
portable-camera robustness, clinician usefulness and review-time evaluation remain
open. See [model documentation](../docs/MODEL_CARD.md) and
[artifact terms](../LICENSE_STATUS.md).

## Learned lesion candidates (opt-in research module)

`matlab/src/m2_lesion/segmentLesionsLearned.m` runs an exported lesion segmenter (ONNX, fixed 1024-pixel window) and
`matlab/demo_lesions.m` writes an EX/HE/MA/SE overlay for one image, and `main_pipeline(img, 'LearnedLesions', true, 'LesionOnnxPath', p, 'LesionThresholdsPath', t)` adds `results.learnedLesions` (counts, field fractions, overlay PNG); the option is off by default and does not change the grade or triage; `matlab/verifyLesionLearned.m` measures parity with the
Python reference. It does not change `main_pipeline.m`. Parity on three validation images: network difference ≤ 7.4e-6
in probability, frozen-threshold mask disagreement ≤ 2.5e-6 of pixels, isolated pixel differences up to 0.19 caused by
1–4 grey-level differences in the Ben Graham blur rounding versus OpenCV. The ONNX file and checkpoint are not distributed
with the repository. Pixel Dice on the locked DDR test: exudates 0.60, haemorrhages 0.43, microaneurysms 0.26.
