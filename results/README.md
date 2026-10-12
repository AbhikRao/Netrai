# Benchmark evidence

Start with the [validation report](../docs/VALIDATION.md) for measured results and
acceptance gaps. The artifacts below are organized by evaluation purpose. They
record development experiments, not clinical validation.

| Evidence | Runtime and scope | Key limitation |
| --- | --- | --- |
| [MATLAB grading — September 28](benchmarks/matlab_runtime_2026-09-28/combined_metrics.json) | 733-image model-mode rerun; complete expected cohort, zero errors | 35 evaluation images have exact training counterparts; cohort used for selection |
| [Split integrity](benchmarks/split_integrity/legacy_aptos_overlap.json) | Legacy APTOS content audit | File grouping does not establish patient independence |
| [Runtime comparison — September 28](benchmarks/matlab_runtime_2026-09-28/runtime_comparison.json) | All 733 grades and screening routes match; probability gate passes | Model-mode consistency on development images, not clinical or full-image validation |
| [MATLAB run timing](benchmarks/matlab_runtime_2026-09-28/matlab_run_metrics.json) | Mean 2.33 s/image on the tested CPU host | Model mode includes decoding, quality and grading; excludes retinal analysis, Grad-CAM, reporting and clinician review |
| [Preprocessing verification](benchmarks/matlab_runtime_2026-09-28/operator_verification.json) | 41 synthetic cases and two identical decoded real arrays match reference operations | Numerical contract check; model and policy unchanged |
| [Historical MATLAB grading](benchmarks/matlab_grading/aggregate_metrics.json) | Earlier 733-image model-mode evaluation | Superseded by the September 28 preprocessing repair; retained for provenance |
| [Historical failed runtime comparison](benchmarks/runtime_comparison/runtime_comparison.json) | Earlier comparison: 717/733 grades and 724/733 triage actions match | Failed gates are historical; current model-mode rerun passes |
| [SimEvents annual sweep — September 28](benchmarks/matlab_workflow_2026-09-28/simulink_scenario_sweep.csv) | 18 native endpoint-v3 bandwidth/staffing/quality scenarios with bounded recapture | Unfitted service/routing/return assumptions; zero dropout; no annual independent-runtime comparison |
| [Patient conservation](benchmarks/matlab_workflow_2026-09-28/annual_conservation_checks.json) | All 18 native scenarios; queue/server occupancy and rates checked | Completion does not measure diagnostic correctness |
| [Workflow fixture comparison](benchmarks/matlab_workflow_2026-09-28/independent_fixture_comparison.json) | Seven deterministic fixtures match a separate event-calendar reference | Engineering cases; not stochastic population or field validation |
| [Workflow repeatability](benchmarks/matlab_workflow_2026-09-28/repeatability_summary.json) | Three native seeded smoke runs; repeatability and routing-seed independence | No uncertainty estimates across multiple annual seeds |
| [Historical capacity simulation](benchmarks/capacity_simulation/simulink_scenario_sweep.csv) | 18 endpoint-v2 scenarios | Earlier unfinished-recapture definition; superseded for current workflow behavior |
| [Historical simulation cross-check](benchmarks/capacity_simulation/reference_crosscheck.json) | Separate endpoint-v2 reference implementation | Rate agreement does not validate field assumptions |
| [Round 3 fresh tests — October 12](benchmarks/round3_fresh_tests/report.json) | 11 locked variants scored once on ODIR-5K (4,408 gradable eyes, cameras unseen in training) and the EyePACS test remainder (4,000); 768-px primary candidate 90.5% / 84.5% on ODIR (AUC 0.948), 87.6% / 92.0% on EyePACS | Specificity 0.5 points under target on ODIR; keyword-derived labels; other-disease eyes outside the endpoint; both tests now exposed |
| [Round 2 fresh tests — October 12](benchmarks/round2_fresh_tests/report.json) | 16 locked variants scored once on Messidor-2 (874 exams) and DDR test (4,105 images); primary candidate 93.9% / 69.3% on Messidor-2, 86.1% / 94.3% on DDR | Thresholds do not transfer to Messidor-2 (specificity below target); DDR is in-dataset for the candidate; both tests now exposed |
| [DeepDRiD locked evaluation — October 11](benchmarks/deepdrid_acceptance/acceptance_report.json) | Five locked variants on 392 primary images / 98 patients in MATLAB; site-adapted candidate 93.3% sensitivity, 89.0% specificity | Retrospective; candidate needs target-camera calibration; evaluation now exposed |
| [Grading adaptation ladder](benchmarks/model_adaptation/full_v1_results.json) | Frozen-feature multi-source adaptation, site policies, Grad-CAM QC | Development selection on DeepDRiD validation and an EyePACS subset |
| [DeepDRiD development validation](benchmarks/deepdrid_development/summary.json) | MATLAB, 396 of 400 validation images (4 predeclared identity exclusions), zero errors; sensitivity 63.3%, specificity 96.8%, AUC 0.910 | Development partition, image-level, no intervals; now exposed; not the reserved evaluation set |
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
