# RFMiD 2.0 lock checklist (O3 §3.3), 2026-10-22
(a) full suite 324 tests OK. (b) `exposure.py status rfmid2` = reserved; `exposure.py verify --strict` OK.
(c) preregistration RFMID2_PREREG.md sha256 827a04491ae6b28a... (bound in the lock); exclusions `rfmid2_identity_v2/exclusions_v1.txt` (541.jpg); clusters `rfmid2_internal_clusters_0.6/clusters.csv` (bound).
(d) models frozen: A_legacy (comparator), R5_768_s17 (primary; MATLAB variant check at freeze), E_768 = mean of R3_768_s2026 + R3_768_s17 (secondary); each with its frozen policy JSON (thresholds, head threshold, temperature); no TTA; no lesion fusion.
(e) static scan: `locked_rfmid.py` takes label paths from the exposure registry and never names the files; the only reader is `read_labels` after `exposure.assert_label_read_allowed`; the static test `test_no_project_code_reads_reserved_labels` passes. No project code has opened the RFMiD label CSVs (identity screens used images only).
(f) git tree clean after commit "RFMiD locked evaluation reads label paths from the registry...".
Update 02:15: logits for R5_768_s17, R3_768_s2026 and R3_768_s17 were computed on the Mac (MPS) from the 859 images only (no label files on the Mac) and enter the lock as precomputed files with an 8-image CPU spot check (tolerance 5e-3); A_legacy and the E_768 mean are computed on the laptop at freeze. Tests: 328 pass (re-run before freezing).

## Lock v2 (2026-10-22 03:05)
v1 voided before any value was seen (VOID.md). (a) 330 tests OK; (b) registry reserved, verify --strict OK; (c) same preregistration/exclusions/clusters; (d) same four models; logits re-bound from the same files (A_legacy from v1 + the Mac files; CPU spot check again); (e) label paths only via the registry in `read_labels` after `assert_label_read_allowed`; (f) tree committed.
