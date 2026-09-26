#!/usr/bin/env python3
"""Stress-test the evaluated learned M1 candidate on paired EyeQ corruptions."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from audit_quality_robustness import apply_corruption
from calibrate_quality import resolve_eyeq_image
from models.quality_model import QualityModel, quality_decisions, quality_input


ROOT = Path(__file__).resolve().parents[1]
CORRUPTIONS = ('baseline', 'defocus_blur', 'low_light', 'uneven_illumination',
               'fov_misalignment', 'jpeg_q25')


def run(args):
    config = json.loads(args.config.read_text(encoding='utf-8'))
    model = QualityModel(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    labels = pd.read_csv(args.labels)
    selected = labels.loc[labels.quality == 0].sample(
        n=args.sample_size, random_state=args.seed)
    rows, pending, metadata = [], [], []
    start = time.perf_counter()

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            probabilities = softmax(
                model(torch.from_numpy(np.stack(pending))).numpy()
                / config['confidence_temperature'], axis=1)
        predictions = quality_decisions(probabilities, config['reject_threshold'])
        for values, probs, prediction in zip(metadata, probabilities, predictions):
            rows.append({**values, 'prediction': int(prediction),
                         'probability_good': float(probs[0]),
                         'probability_usable': float(probs[1]),
                         'probability_reject': float(probs[2])})
        pending.clear(); metadata.clear()

    for item in selected.itertuples(index=False):
        path = resolve_eyeq_image(args.images, str(item.image))
        if path is None:
            raise FileNotFoundError(item.image)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f'Cannot decode {path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for corruption in CORRUPTIONS:
            pending.append(quality_input(apply_corruption(rgb, corruption)))
            metadata.append({'image': str(item.image), 'corruption': corruption})
            if len(pending) == args.batch_size:
                flush()
    flush()
    elapsed = time.perf_counter() - start
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.corruption == 'baseline'].set_index('image')
    summaries = {}
    for corruption, group in frame.groupby('corruption', sort=False):
        paired = group.join(baseline[['probability_reject']], on='image', rsuffix='_baseline')
        counts = group.prediction.value_counts()
        summaries[corruption] = {
            'images': len(group),
            'prediction_counts': {name: int(counts.get(index, 0))
                                  for index, name in enumerate(config['classes'])},
            'reject_rate': float(np.mean(group.prediction == 2)),
            'median_reject_probability': float(group.probability_reject.median()),
            'median_reject_probability_change': float(np.median(
                paired.probability_reject - paired.probability_reject_baseline)),
            'fraction_reject_probability_increased': float(np.mean(
                paired.probability_reject > paired.probability_reject_baseline)),
        }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'per_image_learned_quality_robustness.csv', index=False)
    report = {
        'schema_version': 1,
        'status': 'research_candidate_synthetic_stress_not_camera_validation',
        'sample': {'source': 'EyeQ official test Good class', 'images': args.sample_size,
                   'seed': args.seed},
        'elapsed_seconds': elapsed, 'images_per_second': len(frame) / elapsed,
        'corruptions': summaries,
        'limitations': ['Synthetic corruptions are not portable-camera validation.',
                        'EyeQ labels are global quality labels, not defect-cause labels.'],
    }
    (output / 'learned_quality_robustness.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    result = ROOT / 'results/verification_2026-09-21/eyeq_learned_quality'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'python/weights/quality_model_candidate.pth')
    parser.add_argument('--config', type=Path, default=ROOT / 'python/weights/quality_model_candidate.json')
    parser.add_argument('--labels', type=Path, default=ROOT / 'data/eyeq/data/Label_EyeQ_test.csv')
    parser.add_argument('--images', type=Path, default=ROOT / 'data/eyeq/images/test')
    parser.add_argument('--output', type=Path, default=result / 'robustness')
    parser.add_argument('--sample-size', type=int, default=250)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
