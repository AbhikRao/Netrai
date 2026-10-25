# NetrAI evidence card (October 2026)

One page for reviewers. Every line names its evidence tier and its source. **TF** = fresh test, locked and used once ·
**TL** = leave-dataset-out (sources never trained on; previously exposed) · **TX** = exposed data, analysed after the fact.
Research results; not clinical or regulatory validation. Details: [validation](VALIDATION.md).

## What the evidence supports
| # | Claim | Tier and source |
|---|---|---|
| 1 | The deployed model separates diabetic retinopathy from normal eyes almost perfectly on a fresh Indian-clinic set (AUC ≥ 0.99 against normal eyes) and ranks DR against all eyes at AUC 0.844 (0.816–0.871; 70 DR eyes of 853). RFMiD labels some DR-type lesions as other conditions, which lowers every model's any-DR AUC. | TF, [RFMiD 2.0](../results/benchmarks/rfmid2_locked/report.json) |
| 2 | The 768-pixel research models detect referable DR well on two fresh sets: 90.5% / 84.5% sensitivity / specificity on ODIR-5K (AUC 0.948) and 78.0% / 92.4% on FDR26 (AUC 0.910). | TF, [round 3](../results/benchmarks/round3_fresh_tests), [round 4](../results/benchmarks/round4_fresh_test) |
| 3 | On sources a model never trained on, referable-DR AUC is 0.907–0.962, but a single decision threshold does not transfer between sites. | TL, [leave-dataset-out](../results/benchmarks/lodo_baseline) |
| 4 | A per-site calibration set of about 100–200 labelled clusters brings sensitivity to ≥ 90%; reaching 85% specificity as well depends on the site. | TL, [site calibration](../results/benchmarks/lodo_baseline/site_calibration_lodo.json) |
| 5 | A learned lesion segmenter (opt-in) improves on the hand-crafted detectors by an order of magnitude on a fresh locked test: Dice 0.60 exudates, 0.43 haemorrhages, 0.26 microaneurysms (vs 0.16 / 0.01 / 0.001). | TF, [DDR lesion test](../results/benchmarks/ddr_lesion_locked/report.json) |
| 6 | The deployed model's referral Grad-CAM concentrates on annotated lesions well above chance (2.4–3.3× the chance energy; strongest pixel near a lesion in 64–93% of images vs 17–36% by chance). Alignment, not proof of mechanism. | TX, [Grad-CAM vs lesions](../results/benchmarks/development_studies/x11_gradcam_faithfulness.json) |
| 7 | A DeepDRiD-aware quality score (research) ranks ungradable images better than the EyeQ-trained one: AUC 0.822 vs 0.733 vs 0.550 for the heuristic score; it does not yet respond to synthetic blur or compression, so it is not deployed. | TF, [quality](../results/benchmarks/quality_locked/report.json) |
| 8 | At each policy's any-DR (grade ≥ 1) boundary, every model flags 97–100% of DR eyes on RFMiD 2.0 at about 50% specificity. This is not the grade ≥ 2 referral decision. | TF, [RFMiD 2.0](../results/benchmarks/rfmid2_locked/report.json) |
| 9 | **Smartphone camera (mBRSET, Brazil):** at its frozen threshold the deployed model finds 52.6% of referable-DR eyes (specificity 96.6%); the 768-pixel research models find 84–92% at 83–92% specificity (E_768 89.2% / 89.3%, AUC 0.948). No model met the pre-registered pass rule (> 90% / > 85%). The DeepDRiD-aware quality score ranks ungradable smartphone images at AUC 0.855 (vs 0.838, 0.610); its threshold does not transfer. | TF, [mBRSET](../results/benchmarks/round5_mbrset_fresh_test/report.json) |
| 10 | **Desktop cameras (BRSET, Brazil):** E_768 ranks referable DR better than the deployed model (AUC 0.957 vs 0.887; sensitivity at 85% specificity 95% vs 76%), but at its frozen cut-off specificity is only 56% (Canon 43%, Nikon 82%); the deployed model is 76.8% / 84.6%. Hence the presented default is **E_768 with a required per-site calibration of the cut-off**; the calibration step has post hoc evidence only (needs 40–80+ referable cases; meeting 90% / 85% together in one sample is likely only on some sites). | TF, [BRSET](../results/benchmarks/round5_brset_confirmation/report.json); TX, [calibration curves](../results/benchmarks/site_calibration_brazil/site_calibration_e768.json) |

## What we do not claim
- That the 768-pixel research models are better than the deployed model in general: on RFMiD 2.0 any-DR they are significantly worse (AUC 0.72–0.76), mostly because they flag other retinal disease as DR-like.
- That the screening targets (≥ 90% sensitivity and ≥ 85% specificity) are met at one fixed threshold on unseen sites.
- Five-grade accuracy: mild DR (grade 1) is rarely recognised on unseen sources.
- Clinical or regulatory validation, patient-level independence of DDR images (no patient ids), or that RFMiD's any-DR label equals referable DR.
- Any benefit from lesion-evidence fusion, flip averaging or a frozen RETFound probe: each failed its pre-registered rule.
- That any pre-registered promotion rule was met (none was on mBRSET or BRSET); E_768 + site calibration is the owner's product decision, and the calibration step has not had a fresh test.
- That a quality-score threshold transfers between cameras (ranking AUC 0.68–0.86 depends on the camera), or that results beyond Brazil, one smartphone and two desktop cameras are known.
