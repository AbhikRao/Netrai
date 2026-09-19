# NetrAI Dataset Inventory and Layout

Large medical-image datasets stay under `data/` and must not be committed or
redistributed. The extracted trees were normalized and re-audited on
2026-09-19. Machine-readable status is in
`../results/dataset_audit/dataset_audit.json`. The user deleted the redundant
download archives after confirming extraction; the normalized datasets and
their SHA-256 manifests remain local under `data/`.

## 1. EyeQ / EyePACS — M1 quality calibration

Purpose: calibrate **Good / Usable / Reject** behavior instead of using the
current heuristic `0.30/0.70` IQS cutoffs.

- Official EyeQ repository and labels: <https://github.com/HzFu/EyeQ>
- Original images: follow the EyeQ repository link to the EyePACS/Kaggle 2015
  Diabetic Retinopathy Detection data and accept its access rules.
- Required set: all **28,792 EyeQ-listed images**. Plan for a large download;
  the source EyePACS archive is tens of GB compressed and substantially larger
  after extraction.
- EyeQ quality label mapping: `0=Good`, `1=Usable`, `2=Reject`.

Current layout:

```text
data/eyeq/
├── Label_EyeQ_train.csv
├── Label_EyeQ_test.csv
└── images/
    ├── train/  # 12,543 images
    └── test/   # 16,249 images
```

Then run:

```bash
cd python
python3 calibrate_quality.py
```

The runner refuses to fit or report test metrics if any listed image is
missing/unreadable. It fits thresholds only on the official train split and
evaluates once on the official test split.

**Current status:** ready for calibration. Both official label files and all
28,792 corresponding EyePACS images are present in the split directories. The
2026-09-19 strict dataset audit matched every expected image and label count.

## 2. IDRiD — lesions, landmarks, and external Indian-domain grading

Purpose: pixel/lesion validation, optic-disc/fovea localization, sub-pixel MA
evaluation, and an India-specific grading check.

- Official page: <https://idrid.grand-challenge.org/Data/>
- Download target: the complete IEEE DataPort release.
- Contents: **516** graded 4288×2848 images; **81** images with pixel masks for
  microaneurysms/hard exudates and optic disc, 80 hemorrhage masks, 40 soft
  exudate masks, plus optic-disc/fovea coordinates.
- Approximate image payload from the official description: about 0.4 GB before
  masks/metadata and archive overhead.

The complete release is present under `data/idrid/`: 413/103 grading images,
54/27 segmentation images, 81 MA masks, 80 haemorrhage masks, 81 hard-exudate
masks, 40 soft-exudate masks, and 81 optic-disc masks. The normalized
`segmentation_manifest.csv` contains 81 cases. Current evaluation evidence is
under `../results/verification_2026-09-18/idrid_lesion_validation/` and
`idrid_landmark_validation/`.

## 3. DRIVE — vessel segmentation benchmark

Purpose: report vessel Dice, sensitivity, and specificity against manual masks.

- Official page: <https://drive.grand-challenge.org/>
- Download the complete training and test image/mask packages after registering.
- Store the extraction under `data/drive/`.

**Current status:** the 20 annotated training images, FOV masks and first
manual vessel masks are ready. The downloaded official test package contains
20 images/FOV masks but no public `1st_manual` masks, so the current benchmark
explicitly uses the annotated training split. Evidence is under
`../results/verification_2026-09-18/drive_vessel_validation/`.

## 4. External grading data

Messidor-2 can support a camera/domain robustness study, but its official
release contains **1,748 images and no DR ground-truth labels**. Downloading the
official images alone is therefore not sufficient for severity validation.
Only use it with a separately sourced, provenance-checked label set and keep
that label source/version explicit.

- Official page: <https://www.adcis.net/en/third-party/messidor2/>

**Current status:** all 1,748 images and 874 eye-pair metadata rows are under
`data/messidor2/`. There are no grading labels, so this set is not used to
claim external DR accuracy. IDRiD's independent 103-image official test split
is currently the external grading check.

## Integrity manifest

Each normalized dataset root contains `SHA256SUMS.txt`. Regenerate a manifest
after an intentional layout change without hashing the manifest into itself:

```bash
find data/<dataset> -type f ! -name SHA256SUMS.txt -print0 | sort -z | xargs -0 sha256sum \
  > data/<dataset>/SHA256SUMS.txt
```

Audit all layouts with:

```bash
python/.venv/bin/python python/audit_datasets.py \
  --strict --output results/dataset_audit/dataset_audit.json
```

Do not put credentials, Kaggle API tokens, patient identifiers beyond the
official research IDs, or dataset images into source control.
