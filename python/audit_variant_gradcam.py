#!/usr/bin/env python3
"""Repeat the fixed grade-stratified Grad-CAM shortcut QC for adapted variants.

Uses the same APTOS sample rule as ``audit_explainability.py`` (seeded draw per
true grade), the referral-logit target and ``assess_gradcam_quality``. Heuristic
attribution QC only; a flag is a review signal, not proof of a wrong prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from export_adapted_variant import adapted_model
from modules.explainability import assess_gradcam_quality, generate_gradcam
from utils.artifacts import atomic_json
from utils.preprocessing import model_space_fov_mask

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', nargs=2, action='append', metavar=('ID', 'RUNG_CHECKPOINT_OR_LEGACY'),
                        required=True)
    parser.add_argument('--per-grade', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(ROOT / 'data/aptos2019/train.csv')
    sample = pd.concat([g.sample(n=min(args.per_grade, len(g)), random_state=args.seed)
                        for _, g in frame.groupby('diagnosis', sort=True)]).sort_values(['diagnosis', 'id_code'])
    summary = {}
    for variant_id, checkpoint in args.variant:
        model = adapted_model(None if checkpoint == 'legacy' else Path(checkpoint))
        flags = []
        for row in sample.itertuples(index=False):
            bgr = cv2.imread(str(ROOT / 'data/aptos2019/train_images' / f'{row.id_code}.png'))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            heatmap = generate_gradcam(model, rgb, target_class='referral')
            metrics = assess_gradcam_quality(heatmap, model_space_fov_mask(rgb, size=heatmap.shape[0]))
            flags.append({'id_code': row.id_code, 'grade': int(row.diagnosis), **metrics})
        table = pd.DataFrame(flags)
        summary[variant_id] = {
            'images': len(table), 'flag_rate': float(table.shortcut_flag.mean()),
            'flag_rate_by_grade': {str(g): float(v) for g, v in table.groupby('grade').shortcut_flag.mean().items()},
            'median_outside_fov_attention_fraction': float(table.outside_fov_attention_fraction.median()),
            'median_border_enrichment': float(table.border_enrichment.median())}
        print(variant_id, json.dumps(summary[variant_id]), flush=True)
    atomic_json(args.output, {'schema_version': 1, 'sample': 'APTOS grade-stratified, seed 42',
                              'target': 'referral logit', 'variants': summary,
                              'interpretation': 'Heuristic shortcut-attention QC; not lesion localisation.'})


if __name__ == '__main__':
    main()
