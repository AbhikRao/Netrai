"""Lesion segmentation networks (U-Net and FPN decoders on a timm encoder) and their loss.

Design for MATLAB/ONNX portability: only Conv, BatchNorm, ReLU, bilinear Resize, Concat and Add (no
transposed convolutions). Input height/width must be divisible by 32. Output is one logit map per lesion
type (EX, HE, MA, SE) at the input resolution.
"""
from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F

LESIONS = ('EX', 'HE', 'MA', 'SE')


def conv_block(inputs: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.ReLU(inplace=True))


def up(x: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class UNetDecoder(nn.Module):
    def __init__(self, channels: list[int], decoder_channels=(128, 64, 32, 16)):
        super().__init__()
        skips = channels[:-1][::-1]  # strides 16, 8, 4, 2
        inputs = channels[-1]
        blocks = []
        for skip, out in zip(skips, decoder_channels):
            blocks.append(conv_block(inputs + skip, out))
            inputs = out
        self.blocks = nn.ModuleList(blocks)
        self.out_channels = inputs

    def forward(self, features: list[torch.Tensor], size) -> torch.Tensor:
        x = features[-1]
        for block, skip in zip(self.blocks, features[:-1][::-1]):
            x = block(torch.cat([up(x, skip.shape[-2:]), skip], dim=1))
        return up(x, size)


class FPNDecoder(nn.Module):
    def __init__(self, channels: list[int], width: int = 96):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, width, 1) for c in channels[1:]])  # strides 4, 8, 16, 32
        self.smooth = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 3, padding=1, bias=False),
                                                   nn.BatchNorm2d(width), nn.ReLU(inplace=True)) for _ in channels[1:]])
        self.merge = conv_block(width * len(channels[1:]), width)
        self.out_channels = width

    def forward(self, features: list[torch.Tensor], size) -> torch.Tensor:
        pyramid = [lat(f) for lat, f in zip(self.lateral, features[1:])]
        for index in range(len(pyramid) - 1, 0, -1):
            pyramid[index - 1] = pyramid[index - 1] + up(pyramid[index], pyramid[index - 1].shape[-2:])
        target = pyramid[0].shape[-2:]
        merged = torch.cat([up(s(p), target) if p.shape[-2:] != target else s(p) for s, p in zip(self.smooth, pyramid)], dim=1)
        return up(self.merge(merged), size)


class LesionNet(nn.Module):
    def __init__(self, arch: str = 'unet', encoder: str = 'efficientnet_b2', pretrained: bool = True,
                 num_classes: int = len(LESIONS)):
        super().__init__()
        if arch not in ('unet', 'fpn'):
            raise ValueError("arch must be 'unet' or 'fpn'")
        self.encoder = timm.create_model(encoder, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        channels = self.encoder.feature_info.channels()
        self.decoder = UNetDecoder(channels) if arch == 'unet' else FPNDecoder(channels)
        self.head = nn.Conv2d(self.decoder.out_channels, num_classes, 1)
        self.arch, self.encoder_name = arch, encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError('input height and width must be divisible by 32')
        features = self.encoder(x)
        return self.head(self.decoder(features, x.shape[-2:]))


def lesion_loss(logits: torch.Tensor, targets: torch.Tensor, annotated: torch.Tensor | None = None,
                pos_weight: torch.Tensor | None = None, dice_weight: float = 1.0, eps: float = 1.0) -> torch.Tensor:
    """BCE (optionally positive-weighted) + soft Dice, per lesion channel, ignoring unannotated channels.

    logits/targets: (N, L, H, W); annotated: (N, L) bool, False where a lesion type was not annotated for that image
    (a missing mask is not a negative annotation).
    """
    if annotated is None:
        annotated = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
    weight = annotated.float()[:, :, None, None]
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=None if pos_weight is None else pos_weight[None, :, None, None],
                                             reduction='none')
    bce = (bce * weight).sum() / (weight.sum() * logits.shape[-1] * logits.shape[-2]).clamp_min(1.0)
    prob = torch.sigmoid(logits)
    inter = (prob * targets).sum(dim=(2, 3))
    denominator = prob.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice = 1.0 - (2 * inter + eps) / (denominator + eps)
    dice = (dice * annotated.float()).sum() / annotated.float().sum().clamp_min(1.0)
    return bce + dice_weight * dice
