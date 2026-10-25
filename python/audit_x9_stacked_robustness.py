#!/usr/bin/env python3
"""Paired synthetic stress audit of the X9 stacked quality score (same protocol as audit_learned_quality_robustness.py).

250 EyeQ test images graded Good (seed 42) are corrupted (defocus blur, low light, uneven illumination, FOV shift,
JPEG Q25). For each condition: the reject rate of the stacked DeepDRiD-aware score at its frozen operating threshold
(<= 5% false-reject on DeepDRiD training, out-of-fold) next to the EyeQ candidate at its 0.375 threshold, and the median
paired change of each score. A promotion prerequisite, not camera validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import train_quality_x9 as x9
from audit_quality_robustness import apply_corruption
from calibrate_quality import resolve_eyeq_image
from models.quality_model import quality_input

ROOT = x9.ROOT
CORRUPTIONS = ('baseline', 'defocus_blur', 'low_light', 'uneven_illumination', 'fov_misalignment', 'jpeg_q25')
CANDIDATE_THRESHOLD = 0.375


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/eyeq/data/Label_EyeQ_test.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/eyeq/images/test')
    parser.add_argument('--dev-cache', type=Path, default=ROOT / 'results/model_improvement/experiments/X9/embeddings_dev.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-size', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    cfg = json.loads(x9.CANDIDATE_CFG.read_text())
    qmodel = x9.load_candidate()
    predict, info = x9.fit_pipeline(np.load(args.dev_cache)['train'], x9.load_split('training'), qmodel, float(cfg['confidence_temperature']))
    threshold = info['operating_threshold_5pct_false_reject_oof']
    cand_logit = float(np.log(CANDIDATE_THRESHOLD / (1 - CANDIDATE_THRESHOLD)))
    labels = pd.read_csv(args.labels)
    selected = labels.loc[labels.quality == 0].sample(n=args.sample_size, random_state=args.seed)
    rows = []
    for item in selected.itertuples(index=False):
        rgb = cv2.cvtColor(cv2.imread(str(resolve_eyeq_image(args.images, str(item.image)))), cv2.COLOR_BGR2RGB)
        batch = np.stack([quality_input(apply_corruption(rgb, c)) for c in CORRUPTIONS])
        with torch.inference_mode():
            features = qmodel.encode(torch.from_numpy(batch)).numpy()
        scores = predict(features)
        for k, c in enumerate(CORRUPTIONS):
            rows.append({'image': str(item.image), 'corruption': c, 'stacked': float(scores['final'][k]),
                         'eyeq_logit': float(scores['eyeq_reject_logit'][k])})
    frame = pd.DataFrame(rows)
    base = frame[frame.corruption == 'baseline'].set_index('image')
    summary = {}
    for c, group in frame.groupby('corruption', sort=False):
        g = group.set_index('image')
        summary[c] = {'images': int(len(g)),
                      'stacked_reject_rate': float((g.stacked >= threshold).mean()),
                      'candidate_reject_rate': float((g.eyeq_logit >= cand_logit).mean()),
                      'stacked_median_change': float((g.stacked - base.stacked).median()),
                      'candidate_median_logit_change': float((g.eyeq_logit - base.eyeq_logit).median()),
                      'stacked_fraction_increased': float((g.stacked > base.stacked).mean())}
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / 'per_image.csv', index=False)
    report = {'status': 'synthetic stress audit; not camera validation', 'tier': 'TX (EyeQ test images, exposed)',
              'sample': {'source': 'EyeQ official test, Good class', 'images': args.sample_size, 'seed': args.seed},
              'stacked_threshold': threshold, 'candidate_threshold': CANDIDATE_THRESHOLD, 'conditions': summary}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    for c, s in summary.items():
        print(f"{c:20s} stacked reject {s['stacked_reject_rate']:.3f}  candidate reject {s['candidate_reject_rate']:.3f}  "
              f"stacked median change {s['stacked_median_change']:+.2f}  increased {s['stacked_fraction_increased']:.2f}")


if __name__ == '__main__':
    main()
