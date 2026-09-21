# NetrAI in MATLAB / Simulink

This folder is a first-class MATLAB deployment of the trained NetrAI model. It
does not retrain or imitate the classifier: `models/netrai_fold0.onnx` is an
export of the same checked-in EfficientNet-B4 Fold-0 checkpoint used by the
Python pipeline. MATLAB applies the same black-border crop, Ben Graham
enhancement, ImageNet normalization, softmax, continuous score, and four
ordinal thresholds. Displayed grade confidence uses the same fitted scalar
temperature (`T=0.839077`) as Python; it does not alter the grade decision.
The ONNX output also contains the independently trained referable-DR logit.
MATLAB reports both decisions and sends every disagreement to
`uncertain_human_review` rather than silently preferring one head.

This is an inference/validation deployment, not a second training recipe. The
authoritative training workflow remains `netrai_kaggle_train.ipynb`; the old
MATLAB 224 px hybrid experiment was retired because it did not reproduce the
validated 512 px EfficientNet-B4 checkpoint.

## Installed environment

The local host now runs MATLAB R2026a Update 5 with MATLAB, Simulink, Image
Processing Toolbox, Deep Learning Toolbox, Statistics and Machine Learning
Toolbox, Computer Vision Toolbox, Medical Imaging Toolbox, Optimization
Toolbox, SimEvents, and Parallel Computing Toolbox. Simulink/SimEvents runtime
execution is verified. The exact inventory is saved in
`results/environment_2026-09-18.json`.

The **Deep Learning Toolbox Converter for ONNX Model Format** support package
is installed and `importNetworkFromONNX` resolves locally. The environment
inventory is refreshed by `captureMatlabEnvironment`; full runtime parity and
batch-validation evidence are tracked separately from dependency installation.

## Install on another machine with the IIIT Delhi license

1. Open the official [IIIT Delhi Campus-Wide License portal](https://in.mathworks.com/academia/tah-portal/indraprastha-institute-of-information-technology-delhi-1073736.html).
2. Sign in or create a MathWorks account with the IIIT Delhi email address,
   choose **MATLAB (Individual)**, and download the Linux installer.
3. Install MATLAB R2026a if offered. R2025b/R2025a are also suitable; this
   repository requires R2023b or newer because it uses
   `importNetworkFromONNX`.
4. Select MATLAB, Simulink, Image Processing Toolbox, Deep Learning Toolbox,
   Statistics and Machine Learning Toolbox, Computer Vision Toolbox, Medical
   Imaging Toolbox, Optimization Toolbox, and SimEvents. Parallel Computing
   Toolbox is optional.
5. In MATLAB, open **Add-Ons** and install **Deep Learning Toolbox Converter
   for ONNX Model Format**.

Do not install every MathWorks product just for NetrAI. Reserve roughly 20 GB
so there is room for the selected products, support package, project, and
generated evidence.

After installation, record the environment:

```matlab
ver
license('test','Simulink')
which importNetworkFromONNX
which simevents
```

## Runtime requirements

- MATLAB R2023b or newer (`importNetworkFromONNX` was introduced in R2023b).
- Deep Learning Toolbox.
- Image Processing Toolbox.
- The free **Deep Learning Toolbox Converter for ONNX Model Format** support
package. In MATLAB Online, open **Add-Ons**, search that exact name, and
install it once.
- Simulink and SimEvents are required for the checked-in M5 `.slx` digital
  twin, but not for an M1–M4 inference smoke test.

Official references: [MATLAB Online versions and Basic limits](https://www.mathworks.com/products/matlab-online/matlab-online-versions.html),
[`importNetworkFromONNX`](https://www.mathworks.com/help/deeplearning/ref/importnetworkfromonnx.html),
and the [ONNX converter support package](https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format).

The code runs serially and does not require Parallel Computing Toolbox or a
GPU. Desktop MATLAB is recommended for the full 733-image holdout and M5;
chunked validation remains available for time-limited online sessions.

## Upload and verify

Upload the `Netrai` project (including the 71 MB ONNX file) to MATLAB Drive,
open the `matlab` folder, and run:

```matlab
report = verifyMatlabPackage()
```

The verifier imports the ONNX graph and runs an exact preprocessed parity
fixture. It fails loudly if a toolbox, converter, artifact, or model output is
wrong. The reference ONNX Runtime comparison passed across all six outputs
with maximum absolute error `5.25e-06` on five real APTOS images. Desktop
MATLAB execution uses the installed ONNX converter support package. On
2026-09-21, all six raw fixture outputs passed with maximum error `7.63e-06`
and the end-to-end grade matched. Measured bicubic/uint8 preprocessing
alignment reduced the fixture's maximum preprocessing-path logit difference
from `1.0163` to `0.2210`.

The complete aligned MATLAB Fold-0 run then processed 733/733 images with zero
errors: QWK `0.9297`, referable sensitivity `94.97%`, specificity `93.33%`, AUC
`0.9841`, and average model time `1.55 s/image`. Row-for-row comparison with
Python matches 717/733 strict grades, 729/733 grade-derived referable decisions,
727/733 independent referable-head decisions, and 724/733 final triage actions.
The maximum comparable probability difference is `0.2990`; exact screening,
strict-grade, and `0.05` probability-tolerance gates therefore remain failed.
Do not describe the runtimes as exactly equivalent. The MATLAB pipeline is
executable and independently exceeds the internal screening target, while
cross-library preprocessing remains a bounded deployment difference.

The deployment config also contains the fitted confidence temperature. On a
disjoint post-hoc half of Fold 0 it reduced deployed-policy ECE from 4.59% to
2.58%, NLL from 0.3932 to 0.3837, and Brier from 0.2101 to 0.2079. This is not
external validation; the full artifact and limitation statement are in
`../python/weights/calibration.json`.

## Learned M1 quality candidate (non-default)

`models/quality_model_candidate.onnx` is the evaluated frozen
MobileNetV3-Small EyeQ candidate. Its input is the complete RGB frame resized
to 224 x 224 with explicit half-pixel bilinear coordinates, float32 ImageNet
normalization, no crop, no enhancement, and no antialiasing. The class policy
is Good/Usable argmax unless calibrated Reject probability is at least 0.375.

Verify the shared Python/ONNX/MATLAB contract with:

```matlab
qualityReport = verifyQualityModel();
```

The fixed real-image fixture passes with exact preprocessed-input equality,
the same decision, and maximum MATLAB/Python logit error `2.65e-05`. The
command writes `results/quality_model_candidate_parity.json` for the validation
passport.

The candidate is intentionally not called by `main_pipeline` yet. Although it
passes the global EyeQ engineering gate (macro-F1 `0.8289`, Reject recall
`86.37%`), the paired synthetic audit hard-rejects only `0.4%` of defocus,
`7.6%` low light, `6.4%` uneven illumination, `1.6%` shifted FOV, and `0%` JPEG
Q25 cases. It also lacks real portable-camera and factor-specific validation.
Promoting it would turn a benchmark improvement into an unsupported field
claim, so the existing research route remains unchanged for now.

## Run one image

```matlab
result = main_pipeline('../data/aptos2019/train_images/000c1434d8d7.png', ...
    OutputDir='results/demo', Verbose=true, GenerateReport=true);
```

`result.grade` is 0–4 and `result.isReferable` is true for Grade 2 or higher.
`result.referableHeadProbability`, `result.headDisagreement`, and
`result.triageAction` expose the independent safety evidence and four-way
routing action.
The classifier always receives the original image; M1 enhancement is confined
to the M2 analysis branch so the model input stays consistent with training.

For a preconfigured real-image run:

```matlab
demo_netrai
```

## Batch validation and CSV export

Model-only validation on the fixed 733-image Fold-0 holdout is reproducible in
chunks that fit the free-session compute limit:

```matlab
for first = 1:100:733
    validatePipeline('../data/aptos2019', ...
        Mode='model', StartIndex=first, MaxImages=100, ...
        SelectionCsv='models/fold0_validation.csv', ...
        OutputDir=fullfile('results','fold0_chunks',sprintf('chunk_%03d',first)));
end
summary = aggregateValidationRuns('results/fold0_chunks', 'results/fold0_combined');
```

Full MATLAB M1–M3/M2 processing on a small subset (recommended first in the
time-limited free online session):

```matlab
summary = validatePipeline('../data/aptos2019', ...
    Mode='full', MaxImages=25, OutputDir='results/full_smoke');
```

The validator writes:

- `per_image_results.csv`
- `metrics_summary.csv`
- `aggregate_metrics.json`
- `confusion_matrix.csv`
- `per_grade_metrics.csv`
- `validation_results.mat`

Per-image and aggregate evidence includes both referable decisions,
disagreement, triage, concordant coverage, and retained sensitivity and
specificity. The matching Python Fold-0 run found 9/733 internal head
disagreements (1.23%); the aligned MATLAB run finds 11/733 (1.50%). This
difference is retained as evidence and routed to human review rather than
normalized away.

Compare a MATLAB CSV with the Python evidence using the layered audit:

```bash
python3 ../python/compare_runtime_outputs.py \
  --python-csv ../results/verification_2026-09-18/fold0_dual_head/per_image_results.csv \
  --matlab-csv results/fold0_combined/per_image_results.csv \
  --output-dir ../results/matlab_python_comparison
```

The audit reports screening-decision parity, strict five-grade parity,
threshold-edge mismatches, and numerical tolerances separately. Do not use a
screening parity pass as a claim of exact logits or exact five-grade parity.

For an unlabeled folder, use:

```matlab
table = batchProcess('../my_fundus_images', 'results/batch_results.csv');
```

## SimEvents district digital twin (M5)

`models/NetrAI_Telemedicine_Model.slx` is an executable SimEvents model built
by `src/m5_simulink/setupSimulinkModel.m`. It represents 100,000 annual
arrivals, image acquisition, quality/recapture routing, bandwidth-constrained
upload, AI inference, auto-clear/refer/uncertain routing, and clinician review.
The queue capacities retain the complete annual cohort; an earlier 10,000
entity cap was removed after the independent Python model exposed hidden
back-pressure in the 0.25 Mbps scenarios.

```matlab
% Rebuild the model after changing its programmatic definition:
setupSimulinkModel(fullfile(pwd, 'models', 'NetrAI_Telemedicine_Model.slx'));

% Run 18 bandwidth/staffing/quality scenarios at 100,000 patients/year:
results = runSimulinkScenarioSweep( ...
    fullfile(pwd, 'results', 'm5_scenario_sweep'), 100000);
```

The sweep exports CSV/JSON with throughput, completion, utilization, running
average wait, queue maxima, recapture load, and required review staffing at an
80% utilization planning limit. `../python/simulate_district_workflow.py`
provides the independent patient-level FCFS reference and patient-level
P50/P90/P95 waits:

```bash
python/.venv/bin/python python/simulate_district_workflow.py \
  --simulink-csv matlab/results/m5_scenario_sweep/simulink_scenario_sweep.csv \
  --output results/verification_2026-09-18/m5_python_reference
```

Routing rates are planning assumptions until EyeQ and external validation are
complete. Review escalation is quantified by the Python reference at a
declared 24-hour planning threshold; the current SimEvents graph does not yet
contain a timed abandonment/escalation branch.

## Important interpretation limits

- M2 output is a set of classical-CV lesion candidates. It is not a clinically
  validated lesion boundary. The Python `evaluate_lesion_masks.py` runner is
  provided to quantify precision, recall, Dice, and IoU when pixel annotations
  are available.
- MATLAB attempts genuine Grad-CAM. Some MATLAB releases cannot differentiate
  every imported ONNX graph; in that case the report explicitly says
  **Grad-CAM unavailable**. It never substitutes synthetic attention.
- When Grad-CAM is available, MATLAB applies the same border/corner/FOV QC as
  Python and labels suspicious maps **QC FLAG** for manual review.
- This is a research screening prototype, not a medical device. An
  ophthalmologist must review referrals and uncertain cases.
- M5 is an operational planning model, not evidence of clinical safety or a
  prospective district deployment. Its routing assumptions and continuous
  operating calendar must be replaced with measured field inputs.
