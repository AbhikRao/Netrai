"""Strict, content-bound legacy artifact loading without unrestricted pickle."""
import json
from pathlib import Path

import numpy as np

from utils.artifacts import sha256_file
from utils.validation import ordinal_thresholds


def verify_bundle(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('preprocessing_version') != 'legacy-area-v1':
        raise ValueError('Unsupported legacy grading bundle schema or preprocessing')
    required = {'netrai_fold0_best.pth', 'fold0_thresholds.npy', 'calibration.json', 'safety_policy.json'}
    if set(manifest.get('files', {})) != required:
        raise ValueError('Bundle must bind checkpoint, thresholds, calibration and policy')
    if manifest.get('class_order') != [0, 1, 2, 3, 4]:
        raise ValueError('Unexpected grade class order')
    for name, digest in manifest['files'].items():
        if sha256_file(path.parent / name) != digest:
            raise ValueError(f'Bundle integrity failure: {name}; do not mix model artifacts')
    ordinal_thresholds(np.load(path.parent / 'fold0_thresholds.npy', allow_pickle=False))
    return manifest


def load_legacy_checkpoint(path, device='cpu'):
    """Allow only the NumPy scalar types present in the known historical file.

    New training checkpoints contain tensors and primitive metadata and load
    directly with weights_only=True. Never fall back to unrestricted pickle.
    """
    import torch
    with torch.serialization.safe_globals([
            np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get('model_state_dict')
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Checkpoint must contain a tensor-only model_state_dict')
    return checkpoint
