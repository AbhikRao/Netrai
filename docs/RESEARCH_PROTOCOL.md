# Research protocol and artifact boundaries

## Legacy evidence

APTOS Fold 0 was used for epoch and threshold selection. The later calibration
split does not make it an independent test. An exact-content audit additionally
finds 35 Fold-0 validation images duplicated in training; 30 exact-content
groups contain conflicting grades (62 images). The corrected manifest
quarantines those groups and keeps exact duplicates together, but lacks patient
identity. EyeQ and IDRiD official tests have
been repeatedly inspected; five transformed same-retina EyeQ pairs cross the
official split. These sources remain development evidence. A fresh identity-reviewed
external cohort and clinician study are required for final generalization/usefulness claims.

Raw historical metrics are retained locally. The public [benchmark index](../results/README.md)
provides curated evidence with source hashes and runtime scope. New evaluations
use separate output directories.

## Corrected training

The corrected local manifest builder creates five grouped development folds: three fit,
one selection, one calibration. Content duplicates are grouped transitively with
provided identities; absent verified identity, grouping is file-only and patient
independence is unknown. Perceptual review is still required.

The shared trainer uses training-linear-v2 (RGB uint8, foreground crop >7,
OpenCV linear resize to 512, Gaussian sigma 10 Ben Graham enhancement, ImageNet
normalization). Legacy inference remains legacy-area-v1. Never silently pair new
preprocessing/model weights with old thresholds or calibration.

Compare weighted CE, corrected focal and CE plus cumulative ordinal error on one
manifest. Use seed 42 for screening; repeat finalists with seeds 17, 42, 2026.
Use the same initialization, augmentation, epoch and batch budgets for comparisons.
Selection NLL chooses the checkpoint; calibration rows are reserved for later
calibration, not read by the trainer. Smoke mode is mechanics-only.

Promotion additionally requires development-only policy selection (sensitivity
maximized subject to specificity >85%), per-grade recall, paired comparisons,
three-seed reporting, quality/coverage analysis and frozen external evaluation.
Crossing point-estimate targets does not establish clinical safety.

## Metric/report conventions

Undefined metrics are null in JSON. Conditional model accuracy is separate from
all-cohort resolution/coverage. Rejects, errors and pending human reviews are not
correct predictions. Lesion candidates and attention maps are not confirmed lesions
or diagnoses; missing anatomy is unknown. Grad-CAM targets the referral logit and
is drawn in classifier coordinates. M5 endpoint v2 excludes recapture requests
from completed screening; a recapture return/retry loop remains future work.
Native annotation masks stay at sensor resolution for pixel/lesion scoring; a
prediction is mapped back from the 512 analysis frame. Historical fixed-proposal
MA FROC is superseded by actual detector reruns. Batch evaluation checkpoints
bind selected image bytes, labels, source, environment, settings and model bundle;
resume rejects identity changes.

## Release gates still open

Fresh external grading, reviewed portable-camera quality, validated lesion evidence,
full cross-runtime decision parity, clinician usefulness/review-time evaluation,
and model/fixture redistribution review. Original source and documentation use
[MIT](../LICENSE); other artifact terms remain separate.
