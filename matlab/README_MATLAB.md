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
is available, it also checks its preprocessing-path grade. The fixture passes;
full-cohort numerical/decision equivalence remains unresolved. See
[validation evidence](../docs/VALIDATION.md).

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

Chunk merging checks run identities and rejects duplicate image IDs. Use fresh
output directories. The validator records model/config/data identity, per-image
results, confusion matrix, per-grade measures and aggregate metrics. These 733
images are development data with known training overlap, not an independent test.

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

The model represents acquisition, quality/recapture routing, bandwidth-constrained
upload, inference and clinician review. To rebuild it and run the 18-scenario sweep:

```matlab
addpath(genpath('src'))
setupSimulinkModel(fullfile(pwd, 'models', 'NetrAI_Telemedicine_Model.slx'));
scenarioResults = runSimulinkScenarioSweep(tempname, 100000);
```

The endpoint-v2 definition counts automatic routing and completed clinician
review as completion. Recapture requests remain unfinished. Service times and
routing rates are planning assumptions. The current graph lacks a complete
recapture-return and timed-escalation branch; simulation outcomes do not establish
field benefit. [Recorded scenarios](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.json)

## Quality-model candidate

The separate `quality_model_candidate.onnx` fixture can be checked with
`verifyQualityModel`. That learned candidate is not enabled in `main_pipeline`:
its acquisition-defect stress results and portable-camera evidence are insufficient
for promotion. The default quality route remains the native heuristic assessment.

## Interpretation

NetrAI is a research prototype. Lesion outputs are candidates; Grad-CAM is
attribution rather than lesion-boundary evidence. Fresh external grading,
portable-camera robustness, clinician usefulness and review-time evaluation remain
open. See [model documentation](../docs/MODEL_CARD.md) and
[artifact terms](../LICENSE_STATUS.md).
