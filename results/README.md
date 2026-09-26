# Benchmark evidence

Start with the [validation report](../docs/VALIDATION.md) for measured results and
acceptance gaps. The artifacts below are organized by evaluation purpose. They
record development experiments, not clinical validation.

| Evidence | Runtime and scope | Key limitation |
| --- | --- | --- |
| [MATLAB grading](benchmarks/matlab_grading/aggregate_metrics.json) | 733-image, model-only internal evaluation | 35 evaluation images have exact training counterparts; cohort used for selection |
| [Split integrity](benchmarks/split_integrity/legacy_aptos_overlap.json) | Legacy APTOS content audit | File grouping does not establish patient independence |
| [Runtime comparison](benchmarks/runtime_comparison/runtime_comparison.json) | MATLAB and reference outputs on 733 cases | Exact-decision and numerical gates failed |
| [Capacity simulation](benchmarks/capacity_simulation/simulink_scenario_sweep.csv) | MATLAB SimEvents; 18 endpoint-v2 scenarios | Assumed service/routing rates; recapture requests remain unresolved |
| [Simulation cross-check](benchmarks/capacity_simulation/reference_crosscheck.json) | Separate reference implementation | Rate agreement does not validate field assumptions |
| [Reference grading](benchmarks/reference_grading/aggregate_metrics.json) | 103 IDRiD development-test images; supporting reference runtime | Below screening targets; not a MATLAB external-cohort result |
| [Grading candidate](benchmarks/grading_candidate/summary.json) | Separately trained, non-default candidate | Failed targets; no MATLAB candidate export; test previously exposed |
| [Retinal analysis](benchmarks/retinal_analysis/idrid_native_mask_metrics.csv) | Native-mask evaluation in the supporting reference runtime | Weak lesion baselines; not measured MATLAB segmentation accuracy |
| [Image-quality candidate](benchmarks/image_quality/quality_model_metrics.json) | EyeQ learned candidate, non-default | Same-retina split overlap and poor synthetic-defect rejection |
| [Quality stress audit](benchmarks/image_quality/learned_quality_robustness.json) | 250 paired images per corruption | Synthetic stress is not portable-camera validation |
| [Attribution QC](benchmarks/explanation_qc/gradcam_qc_summary.json) | Supporting reference audit of 25 maps | Heuristic flags; clinician usefulness remains unmeasured |
| [Confidence calibration](benchmarks/confidence_calibration/calibration_metrics.json) | Post-hoc legacy internal calibration | Cohort also informed model/policy selection |
| [MATLAB environment](benchmarks/environment/matlab_environment.json) | Installed products and converter evidence | Environment record, not end-to-end acceptance |

The [provenance manifest](benchmarks/provenance.json) records source dates, scope
and source/published SHA-256 hashes. Machine-specific paths have been made portable;
numeric evaluation values are unchanged. Raw experiment outputs are retained locally.
Historical published outputs also remain in Git history.

Model weights and fixtures reside beside their runtime implementations. Generated
reports, medical-image derivatives, caches and intermediate smoke runs are excluded
from the public evidence collection. Reproduction scripts write fresh outputs to
their documented output directories; obtain the required datasets separately.
