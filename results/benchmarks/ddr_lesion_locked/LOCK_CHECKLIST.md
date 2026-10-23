# DDR lesion locked-evaluation checklist (O3 §3.1), 2026-10-21 23:13
(a) full test suite: 314 tests OK.
(b) `exposure.py status ddr_lesion_seg_test`: reserved.
(c) exclusion file sha256 = 5cdd424e93918da7039743accd6d4ec8f023df0be0ef799b92be9cf6ae64355d (matches LESION_MODEL_PLAN.md).
(d) x8_lock_candidate/best.pt sha256 783a3fbeb0809e2a... equals weights_sha256 in validation_native.json.
(e) grep of python/*.py (excluding tests): the only reader of the DDR test labels is `lesion_data.ddr_records('test', lock_dir)` called from `locked_segmentation_eval.evaluate` (line 195); image listing (line 44) touches only `test/image`. No code references `lesion_detection/test`.
(f) git working tree clean (HEAD committed).

## Lock v1 voided, re-frozen as v2 (23:30)
Before any label was read, an evaluation memory estimate (about 18 GB vs 14 GB free) exposed a defect in `locked_segmentation_eval.evaluate` (all probability maps retained). Fixed (streaming per-image statistics; heuristic failures excluded from paired comparisons, not counted as empty predictions), tests 9/9, dry-run of the learned model and heuristic on 12 spread test images (images only) OK (3-4 s/image). v1 moved to `x8_lock_void_v1/` (VOID.md). Checklist items (a)-(f) re-checked below for v2:
(a) full suite 316 tests OK (23:36). (b) status reserved (registry reset after voiding v1). (c) exclusion sha256 5cdd424e... unchanged. (d) best.pt sha256 783a3fbe... unchanged (weights, thresholds.json and exclusions are the v1 files). (e) grep unchanged: the only reader of the test labels is `ddr_records('test', lock)` in `evaluate`. (f) git tree clean after commit "Registry: DDR lesion lock v1 voided before any label read".
