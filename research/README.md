# Training provenance

`legacy_training.ipynb` preserves the training recipe associated with the
checked-in legacy model exported for MATLAB deployment. The deployment uses
only Fold 0, not a five-model ensemble.

Its internal split later proved to contain exact-content training/evaluation
overlap, and its metrics are development evidence. The notebook is historical
source, not the recipe for the separately trained corrected candidate.
See the [model card](../docs/MODEL_CARD.md) and [validation report](../docs/VALIDATION.md)
for current acceptance limits. Supporting model/evaluation utilities reside in `python/`; MATLAB remains the
primary demonstration entry point.
