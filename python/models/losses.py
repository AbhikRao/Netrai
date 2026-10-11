"""Shared, explicitly weighted training objectives for the five-grade head."""
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Weighted smoothed CE modulated by the *unweighted* true-class p_t.

    Class weights multiply every term in the smoothed target distribution,
    matching PyTorch's weighted label-smoothed cross entropy. Reduction is the
    batch mean of per-example losses. gamma=0 gives this CE exactly.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        if gamma < 0 or not 0 <= label_smoothing < 1:
            raise ValueError('gamma must be nonnegative; smoothing must be in [0, 1)')
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            if weight.ndim != 1 or not torch.isfinite(weight).all() or (weight <= 0).any():
                raise ValueError('Class weights must be finite and positive')
        self.register_buffer('weight', weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = inputs.softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
        return ((1 - pt).pow(self.gamma) * ce).mean()


class OrdinalLoss(FocalLoss):
    """Weighted smoothed CE plus squared cumulative-distribution error."""
    def __init__(self, weight=None, label_smoothing=0.1, ordinal_weight=1.0):
        super().__init__(weight, gamma=0, label_smoothing=label_smoothing)
        self.ordinal_weight = ordinal_weight

    def forward(self, inputs, targets):
        target = F.one_hot(targets, inputs.shape[1]).to(inputs.dtype)
        error = (inputs.softmax(1).cumsum(1) - target.cumsum(1)).square().mean()
        return super().forward(inputs, targets) + self.ordinal_weight * error
