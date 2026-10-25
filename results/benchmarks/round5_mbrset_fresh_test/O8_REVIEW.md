# O8 pre-lock review — mBRSET (Opus, 2026-10-23 ~17:20; BINDING for Batch 7 steps 5-6)

Checked: identity screen (114,644 references; 98 pairs >= 0.6 reviewed on 9 sheets, all different retinas; agreed), exclusions
(none; agreed), within-test structure, inventory/logits order, policies and temperatures of all 8 variants, A_legacy and R5 logits
(5,164 x 6, finite), quality scores (finite, same order, X9 threshold reproduced), no byte duplicates inside mBRSET, label file never
read beyond the declared header/spellings, locked code + 11 tests.

**Finding (fixed before lock):** patient + vessel(>=0.6) single linkage produced a 920-image cluster. Evidence shows 0.6 is not an
identity signal in this cohort (0 same-patient pairs >= 0.6; strongest cross-patient pairs are different retinas). Clusters are now the
patient prefix only (Addendum 3a; `locked_mbrset.CLUSTERS = None`; 1,291 clusters, max 4 images).

**Decision: APPROVED to freeze and evaluate once**, when ALL of these hold:
1. `mbrset_precomputed/` has logits_{A_legacy, R5_768_s17, R4_768_s17, R3_768_s17, R3_768_s2026}.npy, each (5164, 6) and finite.
2. The full test suite passes (or only pre-existing, unrelated failures, listed in the report).
3. Freeze with the BATCH7_PLAN step-5 command (no `--clusters`), then evaluate once. Any failure before a number is printed ->
   VOID.md and STOP. Never re-freeze with edits without Opus.
After the evaluation: BATCH7_PLAN step 6, then STOP for O9. BRSET identity screen may run after the evaluation.
