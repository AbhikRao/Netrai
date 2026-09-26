# Dataset sources and local layout

Medical-image datasets are obtained separately and are excluded from this repository.
Follow each source's access and redistribution terms. Original source-code licensing
does not grant rights to dataset images, labels or derived model artifacts.

| Dataset | Project use | Local layout | Source |
| --- | --- | --- | --- |
| APTOS 2019 | Grading development and legacy internal holdout | `data/aptos2019/train.csv`, `train_images/` | [APTOS challenge](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data) |
| EyeQ / EyePACS | Good, Usable and Reject quality assessment | `data/eyeq/data/Label_EyeQ_train.csv`, `Label_EyeQ_test.csv`, `images/train/`, `images/test/` | [EyeQ labels and acquisition instructions](https://github.com/HzFu/EyeQ) |
| IDRiD | Indian-domain grading, lesions and landmarks | `data/idrid/` with grading/segmentation manifests | [IDRiD data](https://idrid.grand-challenge.org/Data/) |
| DRIVE | Vessel-mask benchmark | `data/drive/training/`, `test/` | [DRIVE dataset](https://drive.grand-challenge.org/) |
| Messidor-2 | Potential camera/domain robustness research | `data/messidor2/` | [Official Messidor-2 source](https://www.adcis.net/en/third-party/messidor2/) |

## Evaluation boundaries

- **APTOS:** 3,662 source images. The content audit found 30 conflicting-grade
  groups covering 62 images and 35 legacy validation images with exact training
  counterparts. Corrected development manifests quarantine conflicting labels
  and group exact duplicates. Filename identities do not establish patient independence.
- **EyeQ:** 12,543 official training and 16,249 official test images;
  quality labels are `0=Good`, `1=Usable`, `2=Reject`. Five transformed same-retina
  pairs cross the official split despite distinct filename identities. Test
  outcomes were inspected during development. The learned candidate remains non-default.
- **IDRiD:** 413/103 grading images and 54/27 segmentation images. Score native
  annotation masks at their original resolution; a missing mask is not a negative
  annotation. Project test exposure prevents fresh external-validation claims.
- **DRIVE:** the recorded benchmark uses the 20 annotated training images.
  The acquired public test package has images/FOV masks without manual vessel labels.
- **Messidor-2:** the acquired 1,748-image release and eye-pair metadata contain
  no official DR grades. Images alone cannot establish grading accuracy.

See the [benchmark index](../results/README.md) for split-integrity audits,
quality robustness, native-mask metrics and grading evidence.

## Reproduction

The MATLAB demo accepts your own permitted RGB fundus image. Local
`patient_sample.jpg` and dataset images are not bundled; a fresh clone requires
an explicit image path. For internal APTOS validation, use the checked-in
`matlab/models/fold0_validation.csv` with the local APTOS dataset.

Dataset-dependent reference utilities can audit the normalized local layout:

```bash
python python/audit_datasets.py --strict --output results/dataset_audit/dataset_audit.json
```

The generated audit and checksums remain local. Exclude credentials, personal
identifiers, medical images and generated patient reports from commits.
