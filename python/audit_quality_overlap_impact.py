#!/usr/bin/env python3
"""Quantify confirmed EyeQ cross-split image overlap without hiding it.

The exclusion is a sensitivity analysis on already-exposed benchmark data,
not a newly independent holdout or a retrained model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibrate_quality import file_sha256
from compare_quality_baselines import metrics


ROOT = Path(__file__).resolve().parents[1]


def endpoints(frame):
    summary, _ = metrics(frame.quality.to_numpy(int), frame.prediction.to_numpy(int))
    nonreject = frame.quality.to_numpy(int) != 2
    predictions = frame.prediction.to_numpy(int)
    return {'images': len(frame), 'metrics': summary,
            'false_reject_rate_good_or_usable': float(np.mean(predictions[nonreject] == 2))}


def run(args):
    pairs = pd.read_csv(args.confirmed)
    if not {'source_a', 'source_b', 'image_a', 'image_b', 'review_decision'}.issubset(pairs):
        raise ValueError('Confirmed overlap file lacks required columns')
    if (pairs.empty or pairs.image_b.duplicated().any()
            or not pairs.source_a.eq('eyeq_train').all()
            or not pairs.source_b.eq('eyeq_test').all()
            or not pairs.review_decision.eq('same_retina').all()):
        raise ValueError('Confirmed pairs must uniquely map EyeQ train to test images')
    geometry = pd.read_csv(args.geometry)
    for item in pairs.itertuples(index=False):
        match = geometry.loc[(geometry.image_a == item.image_a)
                             & (geometry.image_b == item.image_b)]
        if len(match) != 1 or not bool(match.iloc[0].geometrically_consistent):
            raise ValueError(f'Visual confirmation lacks geometry evidence: {item.image_b}')
    frame = pd.read_csv(args.predictions)
    if frame.image.duplicated().any() or not {'image', 'quality', 'prediction'}.issubset(frame):
        raise ValueError('Invalid learned-M1 prediction rows')
    if not set(pairs.image_b).issubset(frame.image):
        raise ValueError('Confirmed test images are not in the benchmark predictions')
    original = endpoints(frame)
    filtered = endpoints(frame.loc[~frame.image.isin(pairs.image_b)])
    evidence = json.loads(args.metrics.read_text(encoding='utf-8'))
    if abs(original['metrics']['macro_f1'] - evidence['official_test_metrics']['macro_f1']) > 1e-10:
        raise ValueError('Predictions do not reproduce the official metric artifact')
    train_labels = pd.read_csv(args.train_labels).set_index('image')
    pair_details = []
    for item in pairs.itertuples(index=False):
        test = frame.loc[frame.image == item.image_b].iloc[0]
        pair_details.append({'train_image': item.image_a, 'test_image': item.image_b,
                             'train_quality_label': int(train_labels.loc[item.image_a, 'quality']),
                             'test_quality_label': int(test.quality),
                             'test_prediction': int(test.prediction)})
    result = {'schema_version': 1,
              'status': 'confirmed_overlap_diagnostic_exclusion_not_independent_test',
              'confirmed_cross_split_same_retina_pairs': len(pairs),
              'sources_sha256': {
                  'confirmed_pairs': file_sha256(args.confirmed),
                  'geometric_review': file_sha256(args.geometry),
                  'test_predictions': file_sha256(args.predictions),
                  'training_labels': file_sha256(args.train_labels),
                  'test_labels': file_sha256(args.test_labels),
              },
              'official_full_test': original,
              'after_excluding_confirmed_test_images': filtered,
              'confirmed_pair_details': pair_details,
              'limitations': ['Five same-retina pairs were visually adjudicated after pHash+ORB triage.',
                              'The pHash radius and ORB gate may miss other transformed duplicates.',
                              'The official test set was previously exposed during development.',
                              'Excluding five images post hoc does not create an independent external cohort.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'confirmed_pairs': len(pairs), 'full': original,
                      'excluded': filtered}, indent=2))


if __name__ == '__main__':
    folder = ROOT/'results/verification_2026-09-24/perceptual_overlap'
    quality = ROOT/'results/verification_2026-09-21/eyeq_learned_quality'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed', type=Path, default=folder/'visually_confirmed_pairs.csv')
    parser.add_argument('--geometry', type=Path, default=folder/'geometric_candidate_review.csv')
    parser.add_argument('--predictions', type=Path, default=quality/'test_predictions.csv')
    parser.add_argument('--metrics', type=Path, default=quality/'quality_model_metrics.json')
    parser.add_argument('--train-labels', type=Path,
                        default=ROOT/'data/eyeq/data/Label_EyeQ_train.csv')
    parser.add_argument('--test-labels', type=Path,
                        default=ROOT/'data/eyeq/data/Label_EyeQ_test.csv')
    parser.add_argument('--output', type=Path, default=folder/'overlap_impact.json')
    run(parser.parse_args())
