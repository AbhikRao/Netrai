# Netrai model card — research only

EfficientNet-B4 with a shared representation and five-grade/binary-referral heads.
The legacy checkpoint was trained on APTOS Fold 0 using the archived notebook;
its smoothed loss did not apply class weights as intended. The new shared trainer
fixes this. Five corrected 30-epoch runs are complete; their selected candidate
failed the external-domain development screening targets and remains non-default.
It has no MATLAB candidate export; MATLAB still uses the legacy ONNX model.

The selected corrected candidate scored sensitivity 84.38%, specificity 69.23%,
QWK 0.5843 and AUC 0.8654 on all 103 IDRiD grading-test images, with Grade-4
recall 2/13. Test labels had prior project exposure, so these are retrospective
development results. The 720-image APTOS calibration result (99.66%/90.07%)
uses the partition that fitted the policy and is not independent validation.
See `results/verification_2026-09-26/grading_study_report.md`.

The 733-image legacy APTOS evaluation was used for model/threshold selection and
contains 35 images with byte-identical counterparts in training. Its QWK 0.9330
and 94.63%/94.02% referral sensitivity/specificity are not independent validation.
Default IDRiD results remain 79.69%/82.05% on 103 development-exposed test images.
The opt-in target-domain probe is score-only (85.94%/87.18%), not default triage.

Legacy inference uses legacy-area-v1, thresholds and confidence calibration bound
by `python/weights/bundle.json`. The corrected training contract is
training-linear-v2. An inference-only contract switch is not promoted: a paired
103-image IDRiD diagnostic reduced sensitivity from 79.69% to 76.56% while raising
specificity from 82.05% to 92.31%; five-grade QWK fell from 0.6652 to 0.5976.

Intended use: reproducible research, not diagnosis, treatment selection or autonomous
patient clearance. Lesion outputs are unverified candidates. Grad-CAM targets the
referral logit and is not lesion localization proof. Confidence is post-hoc internal
calibration, not demonstrated cross-site reliability. Portable-camera, prospective
and clinician usefulness validation remain absent.

Use only trusted model artifacts. Hash binding detects unintended mixing, not
malicious replacement of a model together with its manifest. Legacy optional
scikit-learn/joblib candidates must never be loaded from untrusted sources.

Original source and documentation use [MIT](../LICENSE). Training data,
pretrained-backbone, checkpoint and medical-image fixture redistribution
permissions remain separate review items; see [LICENSE_STATUS.md](../LICENSE_STATUS.md).
