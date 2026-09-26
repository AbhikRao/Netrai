# Supporting research and reference implementation

The primary demonstration entry point is [MATLAB](../matlab/README_MATLAB.md).
This directory preserves model definitions, export tools, dataset-dependent
benchmarks and a supporting reference pipeline. It provides training provenance
and engineering cross-checks; its results must not be attributed to MATLAB unless
that runtime was evaluated separately.

| Area | Entry points |
| --- | --- |
| Reference screening and reports | `inference.py`, `batch_validate.py` |
| Grading model and MATLAB export | `models/netrai_model.py`, `export_matlab_model.py`, `verify_matlab_export.py` |
| Quality candidate | `train_quality_model.py`, `export_quality_model.py`, `audit_learned_quality_robustness.py` |
| Annotated retinal benchmarks | `evaluate_lesion_masks.py`, `evaluate_drive_vessels.py`, `evaluate_idrid_landmarks.py` |
| Calibration and interpretation | `calibrate_model.py`, `audit_explainability.py`, `analyze_selective_prediction.py` |
| Capacity reference and runtime checks | `simulate_district_workflow.py`, `compare_runtime_outputs.py` |
| Dataset integrity | `audit_datasets.py`, `audit_eyeq_duplicates.py` |

## Environment and execution

Use a separate Python environment with the dependencies in
[`requirements.txt`](requirements.txt). The requirements specify supported minimum
versions, not a reproducible environment lock. MATLAB users do not need this
reference environment to run the ONNX deployment.

From this directory, inspect an entry point's options with `python script.py --help`.
For example:

```bash
python inference.py --help
python batch_validate.py --help
python -m unittest discover -s tests
```

Obtain the required datasets under their [source terms](../data/README.md).
Data-dependent scripts write raw outputs to local dated directories. Some analyses
expect per-image outputs from earlier evaluation stages; regenerate those inputs
or supply explicit input/output arguments. The public
[benchmark collection](../results/README.md) contains curated evaluation evidence.

The optional learned quality checkpoint and config are named
`weights/quality_model_candidate.pth` and `weights/quality_model_candidate.json`.
They remain research candidates; relocating them does not enable or promote them.
The historical grading checkpoint and export fixtures retain their recorded
identity. Do not combine weights, calibration and preprocessing from different runs.
