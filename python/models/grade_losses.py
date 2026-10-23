"""Grade-head objectives for experiments (manifest-v6 trainer).

All losses act on the five grade logits and keep the deployed contract: the model still outputs five
logits decoded by softmax and expected grade, so any variant exports to the existing ONNX/MATLAB
schema. (CORN/CORAL-style ordinal heads change that decoding and are deliberately not offered here.)

* ``ce``            cross-entropy with label smoothing 0.1 (the recipe of rounds 2-5)
* ``ce_nosmooth``   plain cross-entropy
* ``emd``           squared earth-mover distance between predicted and one-hot CDFs (ordinal)
* ``ordinal_ls``    cross-entropy against Gaussian soft labels over neighbouring grades (ordinal)
* ``gce``           generalised cross-entropy (1 - p_y^q)/q, robust to noisy labels
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

GRADES = 5
LOSSES = ('ce', 'ce_nosmooth', 'emd', 'ordinal_ls', 'gce')


def soft_labels(targets: torch.Tensor, sigma: float, classes: int = GRADES) -> torch.Tensor:
    """Gaussian distribution over grades centred on the true grade, normalised to sum to one."""
    grades = torch.arange(classes, device=targets.device, dtype=torch.float32)
    weights = torch.exp(-0.5 * ((grades[None, :] - targets[:, None].float()) / sigma) ** 2)
    return weights / weights.sum(dim=1, keepdim=True)


def ordinal_label_smoothing(logits: torch.Tensor, targets: torch.Tensor, sigma: float = 0.7) -> torch.Tensor:
    return -(soft_labels(targets, sigma, logits.shape[1]) * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def squared_emd(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared difference of cumulative distributions (squared EMD for ordered classes)."""
    cdf = logits.softmax(dim=1).cumsum(dim=1)
    target_cdf = F.one_hot(targets, logits.shape[1]).float().cumsum(dim=1)
    return ((cdf - target_cdf) ** 2).sum(dim=1).mean()


def generalized_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, q: float = 0.7) -> torch.Tensor:
    if not 0 < q <= 1:
        raise ValueError('q must lie in (0, 1]')
    probability = logits.softmax(dim=1).gather(1, targets[:, None]).squeeze(1).clamp_min(1e-7)
    return ((1.0 - probability ** q) / q).mean()


def build_grade_loss(name: str, *, sigma: float = 0.7, q: float = 0.7):
    """Return ``loss(logits, targets) -> scalar``."""
    if name == 'ce':
        return lambda logits, targets: F.cross_entropy(logits, targets, label_smoothing=0.1)
    if name == 'ce_nosmooth':
        return lambda logits, targets: F.cross_entropy(logits, targets)
    if name == 'emd':
        return squared_emd
    if name == 'ordinal_ls':
        return lambda logits, targets: ordinal_label_smoothing(logits, targets, sigma)
    if name == 'gce':
        return lambda logits, targets: generalized_cross_entropy(logits, targets, q)
    raise ValueError(f'Unknown loss {name!r}; choose from {LOSSES}')
