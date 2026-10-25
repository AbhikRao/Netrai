# BRSET identity screen (Batch 8, 2026-10-23; no label values read except the declared pre-lock columns)
Reference = 119,808 images: manifest-v6 exports (all training/validation incl. FDR26), DDR, all EyeQ, DeepDRiD Challenge-2 evaluation images, IDRiD grading, ODIR (+ preprocessed copies),
RFMiD 2.0, Messidor-2, APTOS train+test, **and all mBRSET images**. 16,266 BRSET images; max vessel correlation (128 px, also mirrored) histogram:
>=0.45: 2,808; >=0.5: 1,185; **>=0.6: 85; >=0.7: 7; >=0.8: 2 (0.829 and 0.811, both vs EyeQ frames)**. Best-match source of the 85: EyeQ 24, training export 20, ODIR 19, DDR 10, APTOS 6, mBRSET 5, round-4 export 1.
All 85 pairs >= 0.6 reviewed visually (`review/sheet_1..8.jpg`, test left — mirrored when the match was mirrored — reference right): all are **clearly different retinas** (different disc, vessel layout,
field shape and camera look); the highest matches come from a few low-information reference images (one washed-out EyeQ frame matches seven BRSET images; one fully saturated white frame matches one BRSET image).
The five mBRSET matches are different eyes. No byte-identical image pair inside BRSET (SHA256SUMS). -> **no image excluded** (`exclusions_v1.txt` empty).
Limits: the screen cannot detect same-retina re-captures across desktop/smartphone cameras or inside this cohort; independence rests on provenance (three São Paulo centres 2010–2020, Nikon/Canon; training sources are not Brazilian; mBRSET is Itabuna 2022, Eyer).
Bootstrap clusters = `patient_id` (8,524 patients; patients.csv bound in the lock). Vessel clustering is NOT used (O8 finding for mBRSET).
