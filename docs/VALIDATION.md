# Validation and prototype evidence

NetrAI has implementations for acquisition quality, retinal analysis, grading,
explanation reporting and capacity simulation. This describes component coverage,
not a percentage of clinical readiness or completion of all challenge requirements.

## Two more fresh tests and a confirmed calibration rule (October 30, 2026)

Two public graded datasets were registered as reserved before download, extracted blind (neutral image names, labels sealed), identity-reviewed
against 136,074 training, validation and exposed-test images (no exclusion), locked, and evaluated once (tier **TF**): Paraguay (Hospital de
Clínicas, Asunción; Zeiss Visucam 500; 705 images, 519 referable; [report](../results/benchmarks/round6_fresh_paraguay_hc/report.json)) and
SUSTech-SYSU (Guangdong, China; Topcon TRC-50DX; 1,145 images, 490 referable; [report](../results/benchmarks/round6_fresh_sustech_sysu/report.json)).
Each model at its own frozen policy (95% cluster-bootstrap intervals):

| Model | Paraguay sensitivity / specificity | AUC | SUSTech-SYSU sensitivity / specificity | AUC |
| --- | --- | --- | --- | --- |
| E_768 (presented default) | 96.7% (94.8–98.3) / 96.8% (94.1–99.0) | 0.989 | 99.4% (98.6–100) / 95.6% (93.9–97.0) | 0.998 |
| Legacy model | 64.7% (60.2–69.3) / 100% | 0.995 | 85.9% (82.6–88.9) / 99.4% (98.8–99.9) | 0.996 |

- **E_768's frozen cut-off met both screening targets (≥ 90% / ≥ 85%) on both sets** — the first pre-registered fresh tests on which it did. The
  legacy cut-off missed 35% and 14% of referable eyes. All 68 laser-treated SUSTech-SYSU eyes were referred by both models.
- **Both sets are easier than a screening population.** Paraguay excluded eyes with other retinal disease and is severity-enriched (74% referable;
  461 severe NPDR or PDR); SUSTech-SYSU is case-control (its 631 grade-0 images come from non-diabetic volunteers). Both models rank them almost
  perfectly (AUC ≥ 0.99). They show that the cut-off can carry over to Zeiss and Topcon cameras, not that it holds for every camera: on mBRSET, BRSET
  and ODIR the frozen cut-off was borderline or over-referred.
- **The site-calibration rule was changed and the change confirmed.** The pre-registered procedure (re-fit the cut-off to 92% sample sensitivity)
  lowered sensitivity where the frozen cut-off already worked (both sets: mean 0.91; both targets met in 65–72% of 100–200-image samples). A 95% target
  (rule R2), selected on exposed sites before the SUSTech-SYSU lock ([development](../results/benchmarks/site_calibration_rules/development_tx.json), TX),
  was confirmed there as a pre-registered secondary endpoint: both targets met in 93% vs 67% of 200-image samples, mean sensitivity 0.94 vs 0.91
  ([secondary report](../results/benchmarks/round6_fresh_sustech_sysu/secondary_report.json), TF). `calibrate_site.py` now uses 95% by default.
- **Calibration cannot repair weak ranking** ([enriched samples](../results/benchmarks/site_calibration_rules/enriched_tx.json), TX): with 80 referable
  plus 200 other patients and the 95% target, mean sensitivity is 0.94–0.95 on every site, but specificity is 1.00 (SUSTech-SYSU) and 0.92 (BRSET Nikon)
  versus 0.83 (BRSET Canon), 0.74 (mBRSET), 0.69 (ODIR) and 0.42 (FDR26).
- **What the whole pipeline does with ungradable images** ([routing](../results/benchmarks/operational_routing/routing_tx.json), TX): the deployed
  heuristic quality gate is weak on these cameras — on mBRSET it sends 13% of usable smartphone images back for recapture yet passes 73% of the
  ungradable ones, and the pipeline then auto-clears 39% (mBRSET) and 45% (BRSET) of all images graders marked ungradable. None of the 85 ungradable
  images that graders still graded referable was auto-cleared. Grade/referral-head disagreements route 7% (mBRSET) and 19% (BRSET) of gradable images
  to human review. A label-free candidate (send each site's worst 10% by the learned quality score to human review instead of auto-clear) would prevent
  24–40% of ungradable auto-clears for about 3% extra reviews; it is not deployed (post hoc only, and the learned quality model failed its stress audit).
- Citations: Paraguay dataset (Data in Brief 2021, PMC8257963; Zenodo doi:10.5281/zenodo.4647952, CC BY 4.0); Lin L et al., Sci Data 2020,
  doi:10.1038/s41597-020-00755-0 (figshare doi:10.6084/m9.figshare.12570770, CC BY 4.0). Only aggregates are published.

## Desktop-camera confirmation: BRSET, and the default-model decision (October 25, 2026)

BRSET 1.0.2 (PhysioNet, credentialed; three São Paulo centres 2010–2020; Canon CR and Nikon NF5050 desktop cameras) was locked before its labels
were read and evaluated once ([report](../results/benchmarks/round5_brset_confirmation/report.json), tier **TF**, a confirmation of a variant
chosen on mBRSET). Referable DR = `DR_ICDR` ≥ 2; 14,280 adequate-quality graded images (841 referable) from 8,070 patients; 1,986 inadequate images
are outside the DR metrics. Each model at its own frozen policy:

| Model | Sensitivity | Specificity | AUC | Sensitivity at 85% specificity |
| --- | --- | --- | --- | --- |
| Deployed model (legacy) | 76.8% (73.6–79.9) | 84.6% (83.9–85.3) | 0.887 (0.874–0.899) | 75.9% |
| E_768 (pre-registered primary) | 98.5% (97.5–99.3) | 56.0% (55.1–57.0) | 0.957 (0.949–0.963) | 95.1% |
| R4_768_s17 | 98.9% (98.1–99.6) | 44.3% (43.3–45.3) | 0.957 (0.950–0.964) | 95.6% |

- **The pre-registered pass rule was not met** (specificity 56.0% < 85%). The paired AUC gain of E_768 over the deployed model is +0.070
  (+0.060 to +0.081) and sensitivity at 85% specificity is +19.3 points; at the frozen cut-off specificity is −28.6 points.
- **Ranking transfers, the cut-off does not:** specificity is 43% on Canon images and 82% on Nikon images. The deployed model's own cut-off sits near its
  design point here (76.8% / 84.6%) but misses 23% of referable eyes. Over-referral is also concentrated in eyes with another retinal disease
  (E_768 specificity 0.47 with another labelled disease vs 0.64 without).
- **Image quality:** the DeepDRiD-aware score ranks inadequate images worse than the EyeQ-trained candidate here (AUC 0.678 vs 0.752); neither operating
  point fires (reject recall 5% / 9%). Quality-score ranking is camera-dependent and no threshold transfers.
- Five-grade output is unreliable at 93% grade-0 prevalence (QWK 0.27 for E_768; grade-1 recall 2%).

**Decision (owner, October 25, 2026):** the presented screening model is **E_768 with a required per-site calibration of its referral cut-off**; the legacy
model stays available. This is a product decision informed by the results, not a promotion by the pre-registered rule. The calibration step is supported only by
post hoc and leave-dataset-out evidence ([curves](../results/benchmarks/site_calibration_brazil/site_calibration_e768.json), tier **TX**): with the cut-off re-fitted
on a sample containing 80 referable plus 200 other patients, mean specificity rose to 0.89 on Canon and 0.96 on Nikon images at mean sensitivity 0.91–0.92, but
meeting 90% sensitivity and 85% specificity together in one draw had probability only 0.5–0.7 on BRSET and about 0.2 on smartphone images (the smartphone ceiling for any
single cut-off is 91.8% / 85.0%). At screening prevalence (6% referable) a sample of 100–200 *patients* holds only 6–12 referable cases and is too small; the number of
referable cases matters (40–80 or more advised). A site policy is not a validation. (Superseded on October 30: the target is now 95%,
confirmed on a fresh test; see above.) Citations: Nakayama LF et al., BRSET (v1.0.2), PhysioNet 2026,
<https://doi.org/10.13026/mysn-8b26>; Nakayama LF et al., PLOS Digit Health 2024, <https://doi.org/10.1371/journal.pdig.0000454>; Pollard T et al., Nature Health 2026,
<https://doi.org/10.1038/s44360-026-00096-z>. Only aggregates are published.

## Non-fundus input guard (October 31, 2026; tier TX)

Before this guard, a portrait photograph passed the heuristic quality check (IQS 0.77) and was graded "referable" by both runtimes.
Both pipelines now score every image that passes the quality check: the Mahalanobis distance of its 576-D embedding (the frozen
ImageNet MobileNetV3-Small encoder inside the quality candidate, full-frame 224-pixel input) to fundus images of the six datasets that
trained or validated the published weights (APTOS, DDR, DeepDRiD, EyeQ, IDRiD, Messidor-2; 50,391 images). The threshold is 1.25 × the
99.9th percentile of those images' scores; the margin was chosen by leaving each of the six datasets out in turn, on fundus images only
(largest false rejection 0.013%). The rule for rejecting rather than flagging was written before any score was computed.

- **Unseen fundus cameras** (gradable images over the threshold): ODIR 7 / 6,367 (0.11%), FDR26 1 / 2,833, RFMiD 2.0 1 / 853,
  Paraguay 0 / 740, SUSTech-SYSU 0 / 1,219, BRSET 9 / 14,280 (all Canon), but **mBRSET (smartphone) 55 / 4,872 (1.13%)**, including
  7 of 211 grade-4 eyes; these are brighter smartphone captures (mean grey level 102 vs 73). Grade-4 images pooled over all sets:
  11 / 3,147 (0.35%).
- **Non-fundus images:** 2,562 of 2,579 local MATLAB and matplotlib sample images are over the threshold (99.3%), and all 10,000
  digit images. The lowest-scoring ones show a bright object on black, like a fundus photograph (the moon 1.07×, a galaxy 1.36×,
  one car photograph 0.96×). Near-miss medical images (external eye, OCT, slit lamp) were not available; the MATLAB "medical"
  folder holds app icons and screenshots, not scans.
- **Decision (pre-registered):** the smartphone result misses the 0.5% bar, so the guard ships as a **visible warning**, not a
  rejection: the image is still graded, and the result and the report carry "This image does not look like a fundus (retina)
  photograph". Rejecting at this threshold would send about 1 in 90 smartphone retinas, and 1 in 30 grade-4 smartphone eyes, back to
  recapture.
- **Post hoc** (exposed sets): no fundus image of 84,250 scores above 1.79× the threshold (the maximum is a severely atrophic ODIR
  retina; mBRSET maximum 1.36×); the portrait scores 3.05×. A hard rejection above 2.5× is pre-registered as a secondary for the next
  fresh test (Indian Remidio images) and is not deployed.
- Python and MATLAB use byte-identical parameter files (SHA-256 checked) and agree on the score (fixture checked by both test suites).
  [Input guard](../results/benchmarks/input_guard/fundus_guard_tx.json)

## Python and MATLAB: same decisions end to end (October 31, 2026)

Both full pipelines (default settings: quality gate, input guard, E_768) ran on 68 images: five seeded random images from each of 13 fundus
datasets plus a portrait, MATLAB's peppers image and a black frame. No errors in either runtime; the same status for all 68 (3 rejected by
the quality gate, including one smartphone image at IQS 0.291 next to the 0.3 cut-off), the same grade for all 65 graded images, the same
routing for all 68 and the same input warning (2 non-fundus images). Largest differences: quality score 0.0007, guard score 1.1e-5
(relative), continuous model score 5.3e-6. Earlier parity evidence covered the model step only; this includes the separately implemented
quality gate and the guard. It is runtime agreement, not an accuracy result.
[End-to-end parity](../results/benchmarks/runtime_comparison/e2e_parity.json) (`python/compare_runtimes_e2e.py`)

## Upload compression and referral decisions (November 1, 2026; tier TX)

The capacity model below assumes two 3.5 MB images per patient, and a 0.25 Mbps link then falls behind. Can a site send smaller files
without changing E_768's referral decision? 300 gradable images (25 referable and 25 non-referable from each of ODIR, FDR26, Paraguay,
SUSTech-SYSU, BRSET and mBRSET: exposed sets that never trained or selected E_768) were re-encoded in memory: JPEG quality 90, 75, 50
and 30 at native size, and the long side downscaled to 1024 or 768 pixels, then JPEG 90, 75 and 50. The stored files are all JPEG
already (median 423 KB, 177 KB to 1.04 MB per set), so every condition is a recompression; nothing here describes raw camera output.
**Rule (written before scoring):** recommend the smallest setting that changes at most 1% of E_768 decisions (3 of 300) and adds no
quality-gate rejection or guard flag.

| Upload | Median size | E_768 decisions changed (upper 95%) | Legacy decisions changed | E_768 AUC (descriptive) |
| --- | ---: | ---: | ---: | ---: |
| Stored file (reference) | 423 KB | — | — | 0.957 |
| JPEG 90 | 294 KB | 6 (≤ 4.3%) | 3 | 0.955 |
| JPEG 75 | 187 KB | 4 (≤ 3.4%) | 7 | 0.954 |
| JPEG 50 | 131 KB | 16 (≤ 8.5%) | 10 | 0.949 |
| JPEG 30 | 95 KB | 18 (≤ 9.3%) | 20 | 0.940 |
| 1024 px, JPEG 90 / 75 / 50 | 79 / 42 / 28 KB | 11 / 17 / 26 | 10 / 13 / 22 | 0.950 / 0.936 / 0.928 |
| 768 px, JPEG 90 / 75 / 50 | 49 / 27 / 18 KB | 10 / 27 / 28 | 11 / 15 / 21 | 0.950 / 0.937 / 0.934 |

- **No setting meets the rule**, so no smaller upload is recommended: send images as the camera saved them. No setting added a
  quality-gate rejection or a guard flag.
- 14 of the 300 stored-file scores lie within 0.1 of the cut-off, so the sample can show flips at the cut-off.
- **Post hoc (descriptive):** at JPEG 90 and 75 the changed decisions belong to images whose stored-file score lay next to the
  referral cut-off (JPEG 75: all 4 within 0.04 on the 0–4 score; JPEG 90: 5 of 6 within 0.1), in both directions, and sensitivity /
  specificity on this balanced sample hardly moved (stored 92.7% / 86.0%, JPEG 75 92.0% / 86.7%, JPEG 90 92.7% / 87.3%). Downscaling
  then JPEG 90 changed 10–11 decisions, up to 0.44 from the cut-off, with sensitivity / specificity close to the stored file
  (91.3–92.0% / 87.3–88.7%). At JPEG 50 and 30, and downscaling then JPEG 75 or 50, the changes reach images far from the cut-off
  (0.7–1.6) and sensitivity falls to 89–90% (JPEG 50 and 30, downscaled JPEG 50) or specificity to 79–83% (downscaled JPEG 75 and 50).
- The capacity scenario for a smaller upload was planned only for a setting that met the rule, so it was not run; the SimEvents
  results keep 3.5 MB per image. The sampled stored files are much smaller than that, but a screening camera's file size was not
  measured. This concerns the AI's decisions only, not human grading of compressed images.
  [Upload compression](../results/benchmarks/upload_compression/compression_budget_tx.json) (`python/compression_budget_study.py`)

## Explanations and camera behaviour of the presented default (October 30, 2026; post hoc, tier TX)

- **Grad-CAM of E_768 aligns with lesions** (the map both runtimes compute: each member's last activation, after batch norm and SiLU, averaged over
  the two members): energy inside annotated lesions 6.1× chance on 202 DDR lesion-test images and 3.0× on 27 IDRiD images; the map's maximum lies on a
  lesion in 86% / 89% of images (chance 17% / 35%); masking pixels in Grad-CAM order lowers the referral score much faster than random masking
  ([shipped layer](../results/benchmarks/development_studies/x11_gradcam_faithfulness_bn2.json)). Python computes it from the two PyTorch member files
  (same weights; outputs agree with the ONNX graph to 6e-6; `fetch_e768.sh --with-gradcam`), MATLAB in closed form from the ONNX graph; the two maps
  agree to 0.01 on a test image (a fixture both test suites check). **Correction (2026-10-31):** earlier versions said both runtimes produced the same
  map. They did not: Python used each member's convolution before batch norm (6.8× / 3.5×, 91% / 93%;
  [history](../results/benchmarks/development_studies/x11_gradcam_faithfulness_e768.json)), which an ONNX export cannot reproduce because it folds batch
  norm into that convolution, and MATLAB's built-in `gradCAM` explained only one of the two members. The shipped layer was chosen by a rule written before
  its numbers were read; per image it aligns slightly less than the earlier layer for E_768 (DDR energy −0.7, pointing −4.5 points) and clearly better for
  the legacy model (DDR energy +3.9, pointing +8.9 points). Since 2026-10-31 (owner decision, made after these numbers were read) the legacy model
  also uses this layer in both runtimes: 7.2× / 3.7×, maximum on a lesion in 73% / 89% (its earlier Python map: 3.3× / 2.4×, 64% / 93%); the Python
  and MATLAB legacy maps agree to 1e-4 per pixel on a test image (a second fixture both suites check). Both runtimes now differentiate only the network head
  (Python: bit-identical to `pytorch_grad_cam` at that layer; MATLAB: closed form), which makes Grad-CAM 3–4× faster in Python (legacy 1.8 → 0.5 s,
  E_768 6.9 → 2.2 s per image); in the MATLAB legacy pipeline the Grad-CAM step fell from 11–36 s to 2–5 s and the full report from
  23–32 s to 16–26 s per image (3 images, laptop CPU). On 2026-11-01 reading the network's layer list once (it was rebuilt for every head
  layer) cut the MATLAB Grad-CAM step again, with identical maps: E_768 9.1 → 5.3 s, legacy 2.8 → 1.5 s per image (6 images, paired).
  On 2026-11-17 grading and Grad-CAM began sharing one network pass in MATLAB (the grading call keeps each member's last activation):
  outputs and maps are bit-identical on the real models, and the full report took 14–21 s per image for E_768 and 8–16 s for legacy
  (three images, two repeats; 19–34 s and 9–17 s with the two-pass code, measured first in the same session;
  [timing](../results/benchmarks/matlab_runtime_2026-11-17/shared_forward_pass_timing.json)).
- **Why E_768 over-refers on BRSET Canon images** ([diagnosis](../results/benchmarks/development_studies/canon_shift_diagnosis.json)): healthy grade-0 eyes
  without any other finding score a median 1.10 on Canon vs 0.66 on Nikon (50% vs 10% referred at the frozen cut-off), in every age band, while ranking
  stays good (AUC 0.951). It is not the field-of-view format, sensor noise, fine detail, batch-normalisation statistics (label-free AdaBN: specificity 0.39–0.56
  instead of 0.43) or the camera's static dust specks (147 found; inpainting them does not help). The shift sits in coarse image appearance and is not removed
  by any label-free correction tried; per-site labelled calibration of the cut-off is the mitigation.

## Smartphone-camera fresh test: mBRSET (October 23, 2026; nothing here changes the deployed model)

mBRSET (PhysioNet, credentialed; Phelcom Eyer handheld smartphone camera, Itabuna, Brazil, November 2022) was locked before
its labels were read and evaluated once ([report](../results/benchmarks/round5_mbrset_fresh_test/report.json), tier **TF**).
Referable DR = ICDR ≥ 2; 4,859 gradable graded images (861 referable) from 1,285 patients; 292 ungradable (5.7%) and 13 ungraded
images are outside the DR metrics. Each model at its own frozen policy:

| Model | Sensitivity | Specificity | AUC | Sensitivity at 85% specificity |
| --- | --- | --- | --- | --- |
| Deployed model (legacy) | 52.6% (47.3–57.4) | 96.6% (95.8–97.3) | 0.918 | 84.4% |
| R5_768_s17 (pre-registered primary) | 84.1% (80.8–87.0) | 89.6% (88.4–90.8) | 0.928 | 87.8% |
| R4_768_s17 | 85.9% (83.0–88.7) | 91.5% (90.4–92.6) | 0.945 | 90.7% |
| E_768 (two-model mean) | 89.2% (86.7–91.6) | 89.3% (88.1–90.4) | 0.948 | 91.8% |

- **The pre-registered pass rule was not met** (sensitivity > 90% and specificity > 85% at the primary policy): no model is promoted
  and the deployed default is unchanged.
- **The deployed model's frozen threshold misses about 47% of referable eyes on this smartphone data** (specificity 96.6%).
  All 768-pixel research variants cut that to 11–16% at 89–92% specificity (paired sensitivity gain of R5 over the deployed model
  +31.5 points, CI +27.2 to +36.1; specificity −7.0 points). The ranking gap (AUC) is small; the difference is mainly the threshold.
- R5_768_s17 ranked below R4_768_s17 on every measure here (AUC −0.016, CI −0.021 to −0.012); R5 changed several things at once
  (FDR26 data, more epochs, glare augmentation), so the cause is not attributed. Five-grade quality is low: Grade-1 recall 7–16%.
- **Image quality:** the DeepDRiD-aware score ranks ungradable smartphone images with AUC 0.855 (0.828–0.880) vs 0.838 for the
  EyeQ-trained candidate and 0.610 for the heuristic score. Only the ranking carries over: at its frozen operating point it rejected
  23.9% of usable images (designed for 5%), so a smartphone deployment needs its own threshold.
- Limits: one site, one device and one country; the vessel-pattern identity screen cannot detect same-retina re-captures in this
  cohort (no same-patient pair reached its copy threshold), so independence from the training sources rests on provenance
  (none of them is Brazilian); the 768-px variants' weakness against other retinal disease (RFMiD) is unchanged.
- Citation: Nakayama LF et al., mBRSET, a Mobile Brazilian Retinal Dataset (version 1.0), PhysioNet, 2024,
  <https://doi.org/10.13026/qxpd-1y65>; Pollard T et al., PhysioNet as a global platform for biomedical research, Nature Health 2026,
  <https://doi.org/10.1038/s44360-026-00096-z>. Only aggregate results are published; images and per-image outputs stay local
  under the PhysioNet data use agreement. A confirmation test of E_768 on BRSET (three São Paulo centres) is pre-registered but not run.

## Research results of October 20–22, 2026 (branch `research/q4-2026`; nothing here changes the deployed model)

Evidence tiers: **TF** fresh locked test (single use, lock cited) · **TX** exposed data, post hoc · **TL** leave-dataset-out
(sources never trained on or selected on; previously exposed data) · **TD** development/selection data. The registry of
every cohort's exposure status is [exposure_registry.json](../results/benchmarks/exposure_registry.json).

| Question | Result | Tier and evidence |
| --- | --- | --- |
| Do the 768-px research models beat the deployed model on a fresh any-DR test? | **No.** RFMiD 2.0 (853 labelled eyes, 70 with DR): any-DR AUC deployed 0.844 (0.816–0.871), R5_768_s17 0.716 (0.674–0.757), E_768 0.758; paired differences −0.128 and −0.086, CIs exclude 0. DR vs normal eyes AUC ≥ 0.994 for all; the deficit is against eyes with other retinal disease (post hoc). At each policy's grade ≥ 1 (any-DR) boundary, which is not the grade ≥ 2 referral decision, every model flags 97–100% of DR eyes at about 50% specificity. | TF, [RFMiD 2.0](../results/benchmarks/rfmid2_locked/report.json) |
| Does a learned lesion segmenter beat the hand-crafted detectors? | **Yes, by an order of magnitude, with moderate absolute quality.** DDR lesion test, 202 images after removing 23 same-retina near-copies: Dice EX 0.60 (0.55–0.65), HE 0.43 (0.31–0.55), MA 0.26 (0.21–0.30) vs 0.16 / 0.01 / 0.001. MA lesion-level sensitivity 61.8% at 14.2 false-positive components per image. Validation numbers were optimistic (HE 0.67, MA 0.36). | TF, [DDR lesion](../results/benchmarks/ddr_lesion_locked/report.json) |
| Does a DeepDRiD-aware quality score improve on the EyeQ-trained one? | **Yes in ranking.** Challenge-2 (392 images, 54% ungradable): AUC 0.822 (0.775–0.864) vs 0.733 vs heuristic IQS 0.550; at ≤ 5% false-reject only 41% of ungradable images are rejected. Research candidate; deployed M1 unchanged. | TF, [quality](../results/benchmarks/quality_locked/report.json) |
| How well does grading transfer to unseen sources? | Referable-DR AUC 0.907–0.962 (Messidor-2 0.929/0.943, FDR26 0.907, ODIR 0.937, DeepDRiD 0.962); one fixed threshold does not transfer (sensitivity 0.75–0.97, specificity 0.68–0.95). Seed noise about 0.01–0.015 AUC. | TL, [LODO baseline](../results/benchmarks/lodo_baseline) |
| Can a per-site calibration set fix the thresholds? | Sensitivity ≥ 90% from about 100–200 labelled clusters; 90% sensitivity with 85% specificity is reachable only on some sites (DeepDRiD yes; Messidor-2, ODIR, FDR26 no). | TL, [site calibration](../results/benchmarks/lodo_baseline/site_calibration_lodo.json) |
| Did lesion evidence, flip averaging, a retina foundation model or decoding rules improve grading? | No pre-registered rule was met: lesion fusion (3 of 4 sites +0.007…+0.013, ODIR −0.021; with per-site standardisation mean +0.006 < +0.010), flip averaging (+0.003 AUC, below +0.005), frozen RETFound probe (fails to transfer; −0.05…−0.35 AUC), grade-decoding rules (Grade-1 recall 0.08 → 0.2 at a Grade-3/QWK cost). | TL, [development studies](../results/benchmarks/development_studies) |

Limits that apply to all rows: RFMiD's `DR` label is any-DR, not referable DR, and the set shares an institute family with
IDRiD; DDR has no patient ids (identity evidence only); Grade-1 recall is 1–11% on unseen sources; locks and the two
voided first attempts (before any value was seen) are documented next to each report. RETFound weights (CC BY-NC 4.0)
were used for a negative research probe only and are not part of any deliverable.

## MATLAB grading

The September 28, 2026 model-mode evaluation processed all 733 legacy APTOS
Fold-0 images with zero errors and verified complete expected-cohort coverage:

| Metric | Value |
| --- | ---: |
| Referable-DR sensitivity | 94.63% |
| Referable-DR specificity | 94.02% |
| Quadratic weighted kappa | 0.9330 |
| Referable AUC | 0.9842 |
| Mean model-mode time on CPU | 2.33 s/image |

[Aggregate metrics](../results/benchmarks/matlab_runtime_2026-09-28/combined_metrics.json)

The per-image timer includes image-byte verification/decoding, native quality
assessment, preprocessing, inference, calibration and routing. It excludes
retinal analysis, Grad-CAM, report generation, clinician review and the initial
model import. Timing is from this development run on the tested CPU host.
[Run metrics](../results/benchmarks/matlab_runtime_2026-09-28/matlab_run_metrics.json)

The cohort informed checkpoint/policy selection. A later exact-content audit found
35 evaluation images with training counterparts; these figures are development
evidence and cannot establish independent accuracy. No amount of metric inflation
has been estimated. [Overlap audit](../results/benchmarks/split_integrity/legacy_aptos_overlap.json)

The matched comparison passes all three declared gates: exact five-grade outputs,
screening decisions and probability tolerance. All 733 grades, grade-derived
referral decisions, referral-head decisions, disagreement flags and triage actions
match. The maximum absolute probability difference is 2.82 × 10⁻⁶, below the
unchanged 0.05 tolerance; probabilities are not bit-identical.
[Current runtime comparison](../results/benchmarks/matlab_runtime_2026-09-28/runtime_comparison.json)

The repair reproduces `legacy-area-v1` preprocessing through area resampling,
fixed-point uint8 Gaussian filtering and integer illumination blending. All 41
synthetic operator cases and two identical decoded real arrays match the reference
operations exactly. Model weights, calibration and routing policy are unchanged.
[Operator verification](../results/benchmarks/matlab_runtime_2026-09-28/operator_verification.json)

This closes model-mode runtime consistency on the recorded cohort. It does not
establish quality-gate, retinal-analysis or report equivalence, independent
accuracy, portable-camera robustness or clinical safety. The earlier comparison
with 717/733 grades and 724/733 triage actions remains
[historical failed evidence](../results/benchmarks/runtime_comparison/runtime_comparison.json).

## External-domain development evidence

The legacy reference pipeline evaluated 103 IDRiD grading-test images: sensitivity
79.69%, specificity 82.05%, QWK 0.6652 and AUC 0.8746.
[Recorded reference metrics](../results/benchmarks/reference_grading/aggregate_metrics.json)
These are reference-runtime results, not a MATLAB external-cohort measurement.
They miss the challenge's >90% sensitivity and >85% specificity targets.

A separately trained corrected candidate scored 84.38% sensitivity, 69.23%
specificity and QWK 0.5843 on the same 103 images; Grade-4 recall was 2/13.
[Candidate summary](../results/benchmarks/grading_candidate/summary.json)
It also fails screening acceptance and has not replaced the deployed MATLAB model.
The workflows differ, so the change cannot isolate a loss-function effect.
IDRiD test outcomes were already inspected during development. Neither result
constitutes fresh external or prospective clinical validation.

### DeepDRiD development validation (MATLAB, September 28)

The deployed MATLAB pipeline processed the DeepDRiD v1.1 regular-fundus
**validation** partition: 400 requested images from 100 patients, four images of
one patient excluded before prediction for an identity overlap with the reserved
evaluation set, 396 processed and zero model errors. This is the first result on
a source new to the project, but it is a development partition, not the reserved
evaluation set, and it is now exposed.

| Model-only endpoint (396 images) | Value |
| --- | ---: |
| Referable sensitivity | 63.33% (114/180) |
| Referable specificity | 96.76% (209/216) |
| Referable AUC | 0.910 |
| Quadratic weighted kappa | 0.749 |
| Grade-2 / Grade-4 recall | 32/92 / 6/20 |

With the quality gate, 61 of 180 known-referable images were auto-cleared and 20
remained unresolved (recapture or head disagreement). The model ranks referable
disease reasonably (AUC 0.91) but systematically under-grades this camera domain,
so the screening sensitivity target fails by a wide margin. Metrics are
image-level without confidence intervals; views of one patient are correlated.
[Summary](../results/benchmarks/deepdrid_development/summary.json),
[confusion matrix](../results/benchmarks/deepdrid_development/confusion_matrix.csv),
[per-grade metrics](../results/benchmarks/deepdrid_development/per_grade_metrics.csv)

## Round 4: added camera diversity, tested on FDR26 (October 13)

The two 768-pixel models continued training on 36,096 images, adding 3,296 ODIR eyes (several
Chinese camera brands) and 4,000 more EyePACS images. 1,112 ODIR eyes from other patients joined
validation. The primary candidate, `R4_768_s17`, was chosen on validation by the predeclared rule.
It is a single model; the two-model mean was within the tie margin. Everything was then locked
(42 artifacts) and a new public test was scored once:

- **FDR26**, published on Zenodo in October 2026 (CC BY 4.0). Fundus photographs from Mohammed VI
  University Hospital, Oujda, Morocco: a region and cameras absent from training. Three graders
  and a senior grader assigned five-stage ALFEDIAM grades; referable means stage ≥ 2.
- **Identity review:** no image matches any training or validation image
  ([review](../results/benchmarks/round4_fresh_test/fdr26_identity_review.json)).
- **Repeated photographs:** the release repeats many photographs, often under different names, so
  167 repeats were removed before any grade was read: 2,833 images remain. Name groups were merged
  through repeated or same-eye photographs into 1,817 clusters, which are used for the intervals.
- **Two halted attempts:** the first two lock attempts stopped at the label-join step before any
  result was computed, once on a release file-name duplicate and once on the grade spelling. Both
  are recorded in the protocol.

| FDR26 (2,833 images, 1,687 referable) | Sensitivity (95% CI) | Specificity (95% CI) | AUC | Sens. at 85% spec. |
| --- | --- | --- | ---: | ---: |
| Deployed model | 50.3% (47.5–53.1) | 99.7% (99.3–99.9) | 0.911 | 81.9% |
| Round-2 `K3` | 76.6% (74.2–78.9) | 91.7% (90.0–93.2) | 0.904 | 81.1% |
| Round-3 `E_768` | 77.4% (75.0–79.6) | 92.7% (91.1–94.2) | 0.912 | 84.2% |
| **Round-4 `R4_768_s17` primary operating point** | **78.0% (75.7–80.2)** | **92.4% (90.8–93.9)** | **0.910** | **83.5%** |
| `R4_768_s17` balanced (Youden) point | 74.8% (72.2–77.0) | 94.7% (93.3–95.9) | 0.910 | 83.5% |

[Full report](../results/benchmarks/round4_fresh_test/report.json)

What this shows:

- **Round 4 did not improve on this test.**
  - Threshold-free ranking is the same as the deployed model's: paired AUC difference −0.001
    (95% CI −0.011 to +0.009).
  - It is also the same as round 3's: −0.002 (−0.007 to +0.003).
  - The newer models refer far more referable eyes at their thresholds: +27.7 sensitivity points
    against the deployed model (+25.2 to +30.1), at a cost of 7.2 specificity points.
  - [Paired analysis](../results/benchmarks/round4_fresh_test/threshold_free_paired.json)
    (paired intervals added after the lock).
- **The targets are not met** (sensitivity 78% < 90%), so `R4_768_s17` stays a research variant.
- **Misses are concentrated in moderate disease.**
  - About half of grade-2 eyes are referred, against 82% of grade 3 and 97% of grade 4.
  - The deployed model refers 12%, 50% and 88% of these grades.
  - The test set is enriched for disease (60% referable), so it is not a screening population.
- **Performance depends on the image format, post hoc.**
  - AUC is 0.968 on the square-format images but 0.894 and 0.768 on two rectangular formats.
  - It is not an aspect-ratio artefact: the cropped retinal field is square in all three formats
    (median width/height 1.000).
  - The difference is more likely due to the camera or capture, for example the bright glare
    crescents common in the rectangular-format images. It cannot be tested on FDR26 again.
  - [Post-hoc analysis](../results/benchmarks/round4_fresh_test/posthoc_by_grade_and_camera.json)
- **Limits.**
  - ALFEDIAM grades are close to, but not the same as, ICDR.
  - Clusters are inferred from file names and image similarity, not documented patient IDs.
  - Most test logits were computed on an Apple M4 GPU (CPU spot checks within 4.1e-3); the
    deployed model's were computed on the CPU.
- **MATLAB parity:** `R4_768_s17` runs in MATLAB with the locked thresholds. Its grades and
  referral decisions match Python on 40 validation images (maximum logit difference 1.7e-5).
  [Parity](../results/benchmarks/round4_fresh_test/matlab_python_R4_parity.json)
- **Exposure:** FDR26 is now exposed.

## Round 3: 768-pixel models on two fresh tests (October 12)

The round-2 recipe was retrained at **768 × 768 pixels** instead of 512 (same
`legacy-area-v1` preprocessing, architecture and six-output ONNX schema) on 28,800
images: the round-2 sources plus DDR test and DeepDRiD evaluation, both already exposed.
Messidor-2, now exposed, joined validation. Two seeds were trained on Kaggle GPUs. The
primary candidate, **`E_768`**, is the mean of the two models. It was chosen by a rule
fixed before training (validation selection value 0.885 vs 0.883 and 0.881 for the
single models); thresholds were fitted on validation data only.

Everything was then locked (40 artifacts) and two never-used tests were scored once:

- **ODIR-5K** (primary): several camera brands, none used in training. Its per-eye
  diagnostic keywords were mapped to ICDR grades by a mapping fixed in advance. Eyes
  with other diseases only, unspecified DR or quality problems are outside the DR
  endpoint. 108 patients whose photographs also appear in DDR were removed before any
  label was read ([identity review](../results/benchmarks/round3_fresh_tests/odir_identity_review.json)).
- **EyePACS test remainder** (secondary): 4,000 unused images from 2,919 patients,
  in-domain.

| ODIR (4,408 gradable eyes, 2,570 patients) | Sensitivity (95% CI) | Specificity (95% CI) | AUC | Sens. at 85% spec. |
| --- | --- | --- | ---: | ---: |
| Deployed model | 32.1% (28.9–35.4) | 98.3% (97.8–98.7) | 0.885 | 77.4% |
| Round-2 `K3` (512 px) | 89.8% (87.9–91.6) | 81.3% (79.9–82.7) | 0.935 | 87.7% |
| **`E_768` primary operating point** | **90.5% (88.7–92.3)** | **84.5% (83.1–85.8)** | **0.948** | **90.3%** |
| `E_768` balanced (Youden) point | 84.6% (82.3–86.8) | 92.6% (91.7–93.5) | 0.948 | 90.3% |

| EyePACS remainder (4,000 images) | Sensitivity (95% CI) | Specificity (95% CI) | AUC |
| --- | --- | --- | ---: |
| Deployed model | 31.3% (27.8–35.0) | 98.0% (97.5–98.5) | 0.838 |
| Round-2 `K3` | 85.3% (82.6–87.8) | 88.1% (86.9–89.2) | 0.928 |
| **`E_768`** | **87.6% (85.1–89.9)** | **92.0% (91.0–93.0)** | **0.952** |

Intervals are patient-cluster bootstrap (2,000 replicates); AUCs use the expected-grade
score. [Full report](../results/benchmarks/round3_fresh_tests/report.json)

What this shows:

- **Higher resolution improves ranking on unseen data.** The paired difference against
  round-2 `K3` excludes zero on both tests: ODIR AUC +0.013 (95% CI +0.008 to +0.019) and
  sensitivity at 85% specificity +2.5 points (+0.6 to +4.5); EyePACS AUC +0.024 (+0.016 to
  +0.034). Against the deployed model the ODIR AUC gain is +0.063 (+0.053 to +0.074).
  These paired intervals were added after the lock; the summaries were predeclared.
  [Paired analysis](../results/benchmarks/round3_fresh_tests/threshold_free_paired.json)
- **Thresholds now transfer much better.** On a camera never seen in training, `E_768`
  is at 90.5% sensitivity / 84.5% specificity at thresholds fitted on other data. Round 2
  got 69% specificity on Messidor-2, and `K3` gets 81% here.
- **The predeclared promotion rule is still narrowly not met.** It requires >85%
  specificity, and `E_768` reaches 84.5% (interval 83.1–85.8). `E_768` is therefore a
  **research variant, not the default**.
- **Severe disease is rarely missed.** On ODIR, `E_768` missed 1 of 151 grade-3 eyes and
  none of 31 grade-4 eyes; the deployed model missed 37 and 6. On EyePACS it missed 3 of
  118 and 0 of 67 (deployed: 36 and 15). Almost all misses are grade 2. About half of the
  false positives are grade-1 (mild) eyes; grade 1 is almost never predicted (recall 3%).
- **Limits.**
  - ODIR labels are diagnostic keywords from a hospital record, not adjudicated
    research grades.
  - The non-referable class contains only normal and mild-DR eyes. Eyes with other
    diseases are outside the endpoint, so specificity in a real clinic, where other
    diseases appear, may be lower.
  - The test logits were computed on an Apple M4 GPU, and spot checks against the CPU
    agree within 2.4e-3. The ODIR copy is a public Kaggle mirror.
- MATLAB runs `E_768` as one 768-pixel, two-member ONNX graph with locked thresholds.
  On 40 Messidor-2 validation images its grades and referral decisions match Python
  (maximum logit difference 9.1e-6).
  [Parity](../results/benchmarks/round3_fresh_tests/matlab_python_E768_parity.json)
  ODIR and the EyePACS test DR grades are now exposed and cannot serve as fresh tests
  again.

## Round 2: full-network models on two fresh tests (October 12)

Four EfficientNet-B4 models were fine-tuned end to end on GPUs (Kaggle and an
Apple M4) from the deployed weights, with the architecture, input contract and
six-output ONNX schema unchanged. Training used up to 24,641 images (APTOS, EyePACS,
IDRiD, DeepDRiD, and for two models DDR train/valid), acquisition-style
augmentation (rotation, field-of-view shift, exposure, colour, blur, JPEG) and
source- and grade-balanced sampling. The primary candidate, `K3` (first four
backbone stages frozen, DDR added, seed 17), was chosen on development validation by
a rule fixed in advance; thresholds were fitted on validation data only.

Everything was then locked (53 artifacts: predictions, checkpoints, policies, code,
protocol, label-file hashes) and two never-used tests were scored once:
**Messidor-2** with adjudicated grades (Krause et al., 2018; a camera no model saw,
874 exams) and the **DDR** test split (4,105 images; in-dataset for `K3`, which trained
on DDR train/valid). Identity reviews found no overlap between Messidor-2 and any
training image. The DDR release itself repeats 12 images between its train/valid and
test splits; excluding them leaves the DDR results unchanged
([addendum](../results/benchmarks/round2_fresh_tests/ddr_trainvalid_overlap_addendum.json)).

| Messidor-2 (1,744 gradable images) | Sensitivity (95% CI) | Specificity (95% CI) | AUC | Sens. at 85% spec. |
| --- | --- | --- | ---: | ---: |
| Deployed model | 54.1% (48.7–59.4) | 96.3% (95.1–97.4) | 0.909 | 82.3% |
| **`K3` primary operating point** | **93.9% (91.7–95.9)** | **69.3% (66.5–72.2)** | **0.937** | **88.2%** |
| `K3` balanced (Youden) point | 91.9% (89.4–94.4) | 76.5% (73.8–79.1) | 0.937 | 88.2% |

| DDR test (3,759 gradable images) | Sensitivity | Specificity | AUC |
| --- | --- | --- | ---: |
| Deployed model | 34.4% | 99.2% | 0.891 |
| **`K3`** | **86.1%** (84.5–87.7) | **94.3%** (93.3–95.3) | **0.962** |

Messidor-2 intervals are exam-cluster bootstrap; DDR has no patient identifiers, so
its intervals are image-level and likely optimistic. AUCs use the expected-grade score.

What this shows:

- On a camera never seen in training, `K3` ranks referable disease genuinely better
  than the deployed model: AUC +0.028 (paired exam-cluster 95% CI +0.015 to +0.041) and
  sensitivity at 85% specificity +5.9 points (+1.7 to +9.1). These paired intervals were
  added after the lock; the summaries themselves were predeclared.
  [Paired analysis](../results/benchmarks/round2_fresh_tests/messidor_threshold_free_paired.json)
- At its fixed thresholds `K3` is far more sensitive (+39.8 points, 34.8–45.0) at a cost
  of 27.0 specificity points; this part is largely a shift in operating point.
- The thresholds, fitted on other cameras, are too low for Messidor-2: specificity
  is 69%, below the 85% target. Even at the best possible threshold, sensitivity at 85%
  specificity is 88.2%, so both targets are narrowly missed. Many of the false
  positives are mild (grade 1) eyes graded as referable; `K3` never predicts grade 1.
- Per the predeclared rule, `K3` is therefore a **research variant, not the default**.
  A single global threshold still does not transfer across cameras; per-camera threshold
  calibration with a small labelled set remains the practical deployment requirement.
- Post-hoc and exploratory (Messidor-2 already exposed): even with the threshold
  calibrated on half of the Messidor-2 exams and tested on the other half (40 splits),
  `K3` reaches about 90% sensitivity at 81% specificity (deployed model: 72%). The
  remaining gap on this camera is the model's ranking, not only calibration.
  [Post-hoc analysis](../results/benchmarks/round2_fresh_tests/posthoc_messidor_site_calibration.json)
- MATLAB reproduces `K3` exactly at the locked thresholds: on all 1,748 Messidor-2
  images its grades and referral decisions match the Python results (maximum logit
  difference 2.1e-5). [Parity](../results/benchmarks/round2_fresh_tests/matlab_python_k3_messidor_parity.json) Messidor-2
  and DDR test are now exposed and cannot serve as fresh tests again.

[Locked report](../results/benchmarks/round2_fresh_tests/report.json),
[variants and policies](../results/benchmarks/round2_fresh_tests/),
[identity reviews](../results/benchmarks/round2_fresh_tests/messidor_identity_review.json)

## Locked DeepDRiD evaluation of site-adapted candidates (October 11)

A new grading candidate was developed without the reserved data and then tested
once on the DeepDRiD v1.1 regular-fundus **evaluation** set, under an acceptance
lock that bound the protocol, the five deployment bundles, the decision policies
and the 50-file MATLAB evaluation source list before any label was read. The
evaluation ran in MATLAB, and an independent Python recomputation from the
per-image outputs matches every confusion count.

How the candidates were built: the deployed EfficientNet-B4 architecture, input
contract and six-output ONNX schema are unchanged. The backbone stays frozen;
its head layers (`C`) or last stage plus heads (`D`) were retrained on EyePACS
(EyeQ train, gradable), IDRiD train and DeepDRiD train images. Decision
thresholds were then calibrated on DeepDRiD **validation** (site calibration).
The candidate `C_site` was chosen by a rule fixed before `D` finished training.
The first lock halted at label attachment because the workbook column is named
`DR_Levels` rather than the documented `DR_Level`; only the header and image IDs
were inspected, and the parser was corrected and re-locked before any grade value
was read.

Primary population: 392 images from 98 patients, all gradable (four images each
from two patients are kept in coverage but excluded for identity review). Model
only; intervals are 95% patient-cluster bootstrap.

| Variant | Sensitivity | Specificity | AUC | QWK |
| --- | --- | --- | ---: | ---: |
| Deployed legacy model | 62.8% (52.2–73.1) | 96.9% (94.6–99.0) | 0.951 | 0.812 |
| Legacy + site thresholds | 92.7% (87.5–97.1) | 82.9% (75.6–89.2) | 0.951 | 0.809 |
| **`C_site` primary candidate** | **93.3% (88.4–97.5)** | **89.0% (83.6–93.9)** | **0.971** | **0.865** |
| `C_site` recipe without DeepDRiD training | 93.9% (89.2–97.7) | 79.4% (72.5–85.7) | 0.946 | 0.787 |
| `D_site` secondary (information only) | 95.1% (90.3–98.9) | 90.8% (85.4–95.3) | 0.975 | 0.891 |

`C_site` meets the >90% sensitivity and >85% specificity targets on its point
estimates; the lower interval bounds (88.4% / 83.6%) do not exclude missing them.
Against the deployed model it gains 30.5 sensitivity points (paired interval
21.5–39.7) and loses 7.9 specificity points (3.3–12.9). With the quality gate,
only 7 of 164 referable images are auto-cleared (deployed model: 58). Grade-4
recall remains weak (4/20). AUCs use the expected-grade score.

What this shows and what it does not:

- The gain needs labelled images from the target camera: the same recipe without
  DeepDRiD training images fails specificity, and thresholds fitted on other
  sources did not transfer during development. A deployment would need a
  calibration set from each camera.
- It is a retrospective public benchmark, not prospective or rural-India
  validation, and the DeepDRiD evaluation set is now exposed.
- The deployed MATLAB model is unchanged; promoting a candidate is a separate
  decision. On the same 25 audit images and the current QC code, Grad-CAM
  shortcut flags fall from 80% (deployed model) to 64% (`C_site`); flags remain
  common. The earlier 13/25 (52%) figure below used the previous QC rules, so the
  two audits are not directly comparable.

[Acceptance report](../results/benchmarks/deepdrid_acceptance/acceptance_report.json),
[independent recomputation](../results/benchmarks/deepdrid_acceptance/python_crosscheck.json),
[development ladder and policies](../results/benchmarks/model_adaptation/)

## SimEvents capacity scenarios

The September 28 endpoint-v3 sweep completed all 18 combinations of 0.25/1/5 Mbps,
one/two reviewers and 5%/10%/20% assumed quality rejection. A rejected patient
returns to the same acquisition queue with the same identity, with at most two
recaptures. Completion counts disjoint original patients automatically routed or
reviewed; abandonment, exhausted retries and patients still in the system remain
unfinished. This completion endpoint does not measure correct diagnosis.

All 18 native MATLAB scenarios satisfy
`generated = auto_cleared + reviewed + abandoned + quality_exhausted + in_system`,
with in-system counts checked against independent queue/server occupancy sums.
Attempt and return counters have separate denominators.
[Conservation checks](../results/benchmarks/matlab_workflow_2026-09-28/annual_conservation_checks.json)

The annual horizon is 250 × 8 working-hour-equivalent hours, with a 72-second
arrival interval: 100,000 target patients and 100,001 observed arrivals including
the start and cutoff. Each patient uploads two 3.5 MB images (56 Mb total); recompressing uploads to shrink them changes some
AI referral decisions (see [Upload compression](#upload-compression-and-referral-decisions-november-1-2026-tier-tx)).
Acquisition takes 45 seconds per attempt, the return delay is 120 seconds,
inference takes 3.5 seconds and review takes 30 seconds. Initial and retry
rejection probabilities are equal; dropout is explicitly zero. Referral and
uncertain routing probabilities are 8% and 1.23%, respectively. These are
configured assumptions, not fitted field measurements.

| One reviewer, 10% assumed rejection | Completed / generated | Quality-exhausted | In system at cutoff | Completion |
| --- | ---: | ---: | ---: | ---: |
| 0.25 Mbps | 32,142 / 100,001 | 87 | 67,772 | 32.14% |
| 1 Mbps | 99,893 / 100,001 | 105 | 3 | 99.89% |
| 5 Mbps | 99,904 / 100,001 | 95 | 2 | 99.90% |

[Scenario JSON](../results/benchmarks/matlab_workflow_2026-09-28/simulink_scenario_sweep.json) · [Scenario CSV](../results/benchmarks/matlab_workflow_2026-09-28/simulink_scenario_sweep.csv)

At 0.25 Mbps, the 224-second upload service exceeds the 72-second arrival interval;
the upload backlog dominates, and adding a second reviewer does not increase
completed throughput. Under these assumptions keeping pace needs each patient's upload to take less than the arrival interval, that
is under about 1.1 MB per image (arithmetic, not simulated). `matlab/runPayloadWhatIf.m` reruns the three headline scenarios at a
site's own file size; its 3.5 MB reference run reproduces the published rows exactly. Under the assumed 30-second review and 9.23% review demand,
reviewer utilization is low. It cannot establish real staffing requirements.
Each annual scenario uses different seeds, so small differences at adequate
bandwidth are not paired causal effects or estimates of uncertainty across seeds.

Verification has three distinct scopes. Seven deterministic native fixtures cover
no rejection, zero retries, successful return, retry exhaustion, abandonment,
a return pending at cutoff and review congestion. Their counts, return occupancy,
mean queue wait and cutoff utilization match a separate event-calendar reference.
Three native stochastic smoke runs verify identical-seed repeatability and that
changing only the routing seed preserves upstream quality events. The 18 annual
scenarios were checked for native conservation; they were not compared against
an independent annual runtime.
[Fixture summary](../results/benchmarks/matlab_workflow_2026-09-28/fixture_summary.json) · [Independent fixture comparison](../results/benchmarks/matlab_workflow_2026-09-28/independent_fixture_comparison.json) · [Repeatability summary](../results/benchmarks/matlab_workflow_2026-09-28/repeatability_summary.json)

Mean review queue wait covers patients who started review and is separate from
service time; it is not a patient wait percentile. Censored waiting patients stay
in the in-system count. The model does not implement closed hours, shifts, outages
or timed escalation, and does not demonstrate field benefit or cost savings.
The [endpoint-v2 sweep](../results/benchmarks/capacity_simulation/simulink_scenario_sweep.json)
and its [earlier reference comparison](../results/benchmarks/capacity_simulation/reference_crosscheck.json)
remain historical; their unresolved-recapture definition is different.

## Retinal analysis and image quality

The supporting reference runtime was evaluated against native IDRiD annotation
masks on 27 test images. Recorded Dice was 0.0134 for microaneurysms, 0.0170 for
haemorrhage, 0.2037 for exudates and 0.8603 for optic disc.
[Native-mask metrics](../results/benchmarks/retinal_analysis/idrid_native_mask_metrics.csv)
These weak lesion baselines support retaining the candidate label on overlays.
They are not measured MATLAB segmentation accuracy.

The non-default learned quality model recorded macro-F1 0.8289 and Reject recall
86.37% on 16,249 EyeQ test images.
[Quality metrics](../results/benchmarks/image_quality/quality_model_metrics.json)
Five confirmed same-retina pairs cross that official split, and its test outcomes
were inspected during development.
[Overlap audit](../results/benchmarks/image_quality/overlap_impact.json)

On a 250-image paired synthetic stress audit, that candidate rejected only 0.4%
of defocused images, 7.6% of low-light images and 0% of JPEG-Q25 images. Its quality
decision remains disabled in the default pipelines (only its frozen encoder runs, as the
feature of the non-fundus input guard); global quality-label performance does
not establish reliable acquisition-defect detection or portable-camera robustness.
[Stress results](../results/benchmarks/image_quality/learned_quality_robustness.json)

The reference attribution audit flagged 13 of 25 grade-stratified Grad-CAM maps
for possible border/FOV/corner attention. This is a heuristic review signal,
not an established error rate. With the current QC rules on the same 25 images
(October 31), the earlier legacy Python map is flagged 20/25, the map both
runtimes now produce 7/25 (none in grades 2–4) and E_768's map 8/25
([layer comparison](../results/benchmarks/explanation_qc/gradcam_qc_layers.json)). MATLAB reports also display their per-image QC
warnings; clinician explanation usefulness remains unmeasured.
[Attribution QC](../results/benchmarks/explanation_qc/gradcam_qc_summary.json)

## Acceptance work remaining

- Portable-camera quality validation and factor-specific robustness.
- Lesion evaluation beyond DDR (the locked DDR test is in-distribution), including subtype/NV labels; microaneurysm quality is weak (Dice 0.26).
- Prospective, camera-specific validation of the site-adapted candidate and severe-grade (Grade-4) recall.
- For the 768-pixel candidates: specificity just below target on ODIR, grade-2 sensitivity and rectangular-format cameras on FDR26, grade-1 recognition, and a fresh adjudicated test with other eye diseases included.
- Broader full-image runtime checks, including quality and retinal analysis.
- Clinician explanation usefulness and measured review time.
- Field-derived recapture-return assumptions, shift/outage modeling and timed referral escalation.

Candidate masks are not confirmed diagnoses. Grad-CAM is model attribution,
not validated lesion localization. Disagreement-to-review is an implemented rule;
its clinical safety benefit has not been established.
