# mBRSET identity screen (Batch 7, 2026-10-23; image-only, no label read)
Reference = 114,644 images: manifest-v6 exports (train_export_768, _r4, _r5 = every training/validation image incl. FDR26), DDR (train/valid/test),
all EyeQ, DeepDRiD Challenge-2 evaluation images, IDRiD grading, ODIR (incl. preprocessed copies), RFMiD 2.0, Messidor-2, APTOS train+test.
Vessel-correlation screen (128 px, also mirrored): 5,164 mBRSET images; max correlation histogram: >=0.45: 1,621; >=0.5: 815; **>=0.6: 98; >=0.7: 2 (0.719, 0.715); >=0.8: 0**.
Visual review of **all 98 pairs >= 0.6** (`review/sheet_1..9.jpg`, test image left, mirrored when the best match was mirrored, reference right):
all 98 are clearly **different retinas** (different optic-disc position/shape, vessel branching, fovea and field; the matches come from shared
low-information structure and the phone vs desk-camera differences). The two highest (0.72 each, one against an ODIR image and one against an EyeQ image) are
different eyes. No clear copy and no ambiguous pair -> **no image excluded** (`exclusions_v1.txt` has no entries). Pairs 0.45–0.6 (1,523 images) were not reviewed (RFMiD precedent: copies cluster >= 0.6).
Reference hub effect: the same ODIR/EyeQ images recur as best match for many mBRSET images (e.g. ODIR 1258_left, EyeQ 8546_left, 9261_left).
Internal clusters (vessel >= 0.6 single linkage, merged with patient file-name prefixes): 5,164 images -> 4,870 clusters; 381 images share a cluster.
**Caveat for O8:** one cluster holds 71 images from 70 patients (chain through low-information "hub" images, as seen at 0.5 in RFMiD); 87 clusters contain >1 patient
(259 patients merged). A 1.4%-of-data cluster widens bootstrap intervals (conservative); alternatives are not applied without O8.
Exact-file duplicates against references were not separately searched (the vessel screen would have given correlation ~1.0; max observed 0.72).
