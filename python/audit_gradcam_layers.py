#!/usr/bin/env python3
"""Grad-CAM shortcut QC of the previous and the current referral maps on the fixed audit sample.

Same sample rule as ``audit_explainability.py`` / ``audit_variant_gradcam.py`` (APTOS training images, seeded draw per true grade)
and the current ``assess_gradcam_quality`` rules, for three maps: the legacy model at ``conv_head`` (the Python map before
2026-10-31), the legacy model at ``bn2`` and E_768 at each member's ``bn2`` (the maps both runtimes now produce). Heuristic
attribution QC on exposed training images; a flag is a review signal, not proof of a wrong prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import inference
from modules import variant_e768
from modules.explainability import assess_gradcam_quality, generate_gradcam
from utils.artifacts import atomic_json
from utils.preprocessing import model_space_fov_mask, preprocess_for_model

ROOT = Path(__file__).resolve().parents[1]


def conv_head_map(model, rgb) -> np.ndarray:
    """The previous Python legacy map: pytorch_grad_cam at ``backbone.conv_head`` (before batch norm)."""
    import torch
    from pytorch_grad_cam import GradCAM

    class BothHeads(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, inputs):
            logits, referral = self.network(inputs)
            return torch.cat([logits, referral[:, None]], dim=1)

    x = torch.from_numpy(preprocess_for_model(rgb, version='legacy-area-v1').transpose(2, 0, 1)[None]).float()
    with GradCAM(model=BothHeads(model).eval(), target_layers=[model.backbone.conv_head]) as cam:
        return cam(input_tensor=x, targets=[lambda output: output[5]])[0]


def summarize(table: pd.DataFrame) -> dict:
    return {'images': len(table), 'flagged': int(table.shortcut_flag.sum()), 'flag_rate': float(table.shortcut_flag.mean()),
            'flag_rate_by_grade': {str(g): float(v) for g, v in table.groupby('grade').shortcut_flag.mean().items()},
            'median_outside_fov_attention_fraction': float(table.outside_fov_attention_fraction.median()),
            'median_border_enrichment': float(table.border_enrichment.median())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-grade', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(ROOT / 'data/aptos2019/train.csv')
    sample = pd.concat([g.sample(n=min(args.per_grade, len(g)), random_state=args.seed)
                        for _, g in frame.groupby('diagnosis', sort=True)]).sort_values(['diagnosis', 'id_code'])
    model, _, _ = inference._get_model()
    maps = {'legacy_conv_head': lambda rgb: conv_head_map(model, rgb),
            'legacy_bn2': lambda rgb: generate_gradcam(model, rgb, target_class='referral')}
    if variant_e768.weights_present() and variant_e768.members_present():
        maps['E_768_bn2'] = variant_e768.E768Runtime().gradcam
    rows = []
    for row in sample.itertuples(index=False):
        rgb = cv2.cvtColor(cv2.imread(str(ROOT / 'data/aptos2019/train_images' / f'{row.id_code}.png')), cv2.COLOR_BGR2RGB)
        for name, make in maps.items():
            heatmap = make(rgb)
            rows.append({'map': name, 'grade': int(row.diagnosis),
                         **assess_gradcam_quality(heatmap, model_space_fov_mask(rgb, size=heatmap.shape[0]))})
    table = pd.DataFrame(rows)
    result = {name: summarize(table[table['map'] == name]) for name in maps}
    before = table[table['map'] == 'legacy_conv_head'].shortcut_flag.to_numpy(bool)
    after = table[table['map'] == 'legacy_bn2'].shortcut_flag.to_numpy(bool)
    result['legacy_paired'] = {'flagged_conv_head_only': int((before & ~after).sum()), 'flagged_bn2_only': int((~before & after).sum()),
                               'flagged_both': int((before & after).sum())}
    print(json.dumps(result, indent=1))
    atomic_json(args.output, {'schema_version': 1, 'tier': 'TX (APTOS training images; heuristic QC)',
                              'sample': f'APTOS grade-stratified, {args.per_grade} per grade, seed {args.seed}',
                              'target': 'referral logit', 'qc': 'modules.explainability.assess_gradcam_quality (current rules)',
                              'maps': result,
                              'interpretation': 'Heuristic shortcut-attention QC; a flag asks for visual review. Not lesion '
                                                'localisation and not an error rate.'})


if __name__ == '__main__':
    main()
