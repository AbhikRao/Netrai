# O10 pre-lock review — BRSET confirmation (Opus, 2026-10-25 00:05; BINDING)

Checked on the real data (labels still reserved; only the Addendum-4 columns were read): inventory 16,266 images, 0 exclusions,
8,524 patient clusters (max 4 images), logits of A_legacy / R4_768_s17 / R3_768_s2026 / R3_768_s17 each (16266, 6), finite, order =
inventory; quality scores finite, same order, X9 threshold reproduced; policies/temperatures of all 5 variants as frozen (E_768 = round-3
balanced); identity review (85 pairs >= 0.6, all different retinas; agreed); BRSET and mBRSET independent by site/years/camera
(public descriptors); locked_brset.py + 10 tests; full suite 367 OK. Registry `brset` = reserved.
Note: the "any factor unsatisfactory" quality target relies on spellings starting with "unsat"; the report prints the factor spellings,
so if they differ that secondary is reported as not evaluable (no re-run). It has no role in the primary or the verdict.

**APPROVED:** freeze, then evaluate once:
  python locked_brset.py freeze --variants ../results/model_improvement/brset_eval/variants.json --output ../results/model_improvement/brset_lock --precomputed ../results/model_improvement/brset_precomputed
  python locked_brset.py evaluate --variants ../results/model_improvement/brset_eval/variants.json --output ../results/model_improvement/brset_lock
Any failure before a number is printed -> VOID.md and STOP. Afterwards: curated copy in results/benchmarks/round5_brset_confirmation/ (report without
per-image file-name lists, lock, variants, IDENTITY_DECISION.md, README with PhysioNet citations: BRSET doi 10.13026/mysn-8b26 + Nakayama 2024 PLOS Digit Health
doi 10.1371/journal.pdig.0000454 + PhysioNet platform), `.project-notes/experiments/BRSET_RESULT.md` (facts only, tiers), AGENTS.md; commit locally.
Do NOT edit README/EVIDENCE_CARD/VALIDATION claims or any default -> STOP for O11 (Opus) with the owner's default-model decision.
Owner (2026-10-25): the DUA permits using the owner's own Mac (not needed for this lock; logits were computed on the laptop CPU).
