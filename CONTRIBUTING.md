# Contributing

NetrAI focuses on MATLAB screening, explainable reporting and SimEvents capacity
analysis for PS26038. Keep changes focused and include reproducible evidence.

For MATLAB changes, run `verifyMatlabPackage` and a quality-gated image/report
check where appropriate. For simulation changes, verify completed, recapture and
unfinished counts against endpoint version 2. A fixture pass does not establish
whole-cohort equivalence. Use fresh output directories and preserve historical results.

For model changes, freeze data splits, seeds, configuration and artifact hashes.
Select models on selection data, fit policies on calibration data, and disclose
prior test exposure. Report unresolved cases and failures alongside performance.
Update the README, model card and validation report when behavior or evidence changes.

Do not submit datasets, patient reports, retinal-image derivatives, credentials
or generated caches. See [licensing scope](LICENSE_STATUS.md) for separate model,
image and dependency terms. Original source/documentation use [MIT](LICENSE).
