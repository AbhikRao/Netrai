# DeepDRiD Challenge-2 quality lock checklist (O5 §3.4), 2026-10-22
(a) full suite 345 tests OK. (b) `exposure.py status deepdrid_challenge2_quality` = reserved. (c) Development gate PASSED on DeepDRiD validation (TD): stacked AUC 0.812 vs EyeQ candidate 0.724 (+0.087, CI 0.050-0.125; gate +0.02) (`experiments/X9/dev_result.json`).
(d) frozen: candidate encoder/head weights, DeepDRiD training-only fit (`train_quality_x9.fit_pipeline`), operating threshold from out-of-fold training scores (<= 5% false-reject), heuristic IQS comparator; primary subset 392 images (patients 420, 454 excluded), protocol X9_PREREG.md bound.
(e) static scan: no project code names the workbook (static test passes); `locked_quality_eval.read_workbook` takes the path from the exposure registry and is called only after `assert_label_read_allowed`; the workbook has been opened by no code (headers/rows only were read in Batch 1).
(f) git tree clean after commit "Locked quality evaluation (freeze/evaluate)...".
