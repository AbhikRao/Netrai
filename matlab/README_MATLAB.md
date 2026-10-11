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

## Interpretation

NetrAI is a research prototype. Lesion outputs are candidates; Grad-CAM is
attribution rather than lesion-boundary evidence. Fresh external grading,
portable-camera robustness, clinician usefulness and review-time evaluation remain
open. See [model documentation](../docs/MODEL_CARD.md) and
[artifact terms](../LICENSE_STATUS.md).
