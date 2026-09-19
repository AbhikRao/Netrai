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
2026-09-19, all six raw fixture outputs passed with maximum error `7.63e-06`
and the end-to-end grade matched. The verifier also reported a `1.0163`
preprocessing-path logit difference, so MATLAB/OpenCV resize and Gaussian
boundary alignment remains an explicit follow-up before the full holdout.

The deployment config also contains the fitted confidence temperature. On a
disjoint post-hoc half of Fold 0 it reduced deployed-policy ECE from 4.59% to
2.58%, NLL from 0.3932 to 0.3837, and Brier from 0.2101 to 0.2079. This is not
external validation; the full artifact and limitation statement are in
`../python/weights/calibration.json`.

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
specificity. The matching Python Fold-0 run found 9/733 disagreements (1.23%);
the MATLAB run must reproduce this after installation.

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
