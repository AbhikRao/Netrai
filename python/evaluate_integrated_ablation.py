#!/usr/bin/env python3
"""Same-cohort development ablation of grade, M1 candidate, and dual-head review.

Recapture and review are unresolved outcomes, never silently counted as correct
screening decisions. The EyeQ-trained M1 candidate is evaluated on APTOS only
to measure routing/workload, not to establish APTOS image gradability.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax

from calibrate_quality import file_sha256
from models.quality_model import QualityModel, quality_decisions, quality_input


ROOT = Path(__file__).resolve().parents[1]


def ablation_row(truth, grades, abstain, name):
    truth = np.asarray(truth, dtype=int) >= 2
    grades = np.asarray(grades, dtype=int) >= 2
    abstain = np.asarray(abstain, dtype=int)
    auto = abstain == 0
    tp = int(np.sum(auto & truth & grades))
    tn = int(np.sum(auto & ~truth & ~grades))
    fp = int(np.sum(auto & ~truth & grades))
    fn = int(np.sum(auto & truth & ~grades))
    return {'variant': name, 'images': len(truth),
            'auto_decisions': int(auto.sum()),
            'auto_coverage': float(auto.mean()),
            'recaptures': int(np.sum(abstain == 1)),
            'human_reviews': int(np.sum(abstain == 2)),
            'unresolved_referable': int(np.sum(~auto & truth)),
            'unresolved_nonreferable': int(np.sum(~auto & ~truth)),
            'auto_tp': tp, 'auto_tn': tn, 'auto_fp': fp, 'auto_fn': fn,
            'auto_sensitivity_conditional': tp/(tp+fn) if tp+fn else None,
            'auto_specificity_conditional': tn/(tn+fp) if tn+fp else None}


def quality_predictions(frame, checkpoint, config, batch_size):
    settings = json.loads(config.read_text(encoding='utf-8'))
    if settings.get('deployment_status') != 'research_candidate_not_deployed':
        raise ValueError('Expected the non-deployed learned M1 candidate')
    if settings.get('checkpoint_sha256') != file_sha256(checkpoint):
        raise ValueError('Quality checkpoint does not match its evaluated config')
    model = QualityModel(pretrained=False)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    predictions, pending = [], []

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            logits = model(torch.from_numpy(np.stack(pending))).numpy()
        probs = softmax(logits/settings['confidence_temperature'], axis=1)
        predictions.extend(quality_decisions(probs, settings['reject_threshold']).tolist())
        pending.clear()

    for path in frame.image_path:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        pending.append(quality_input(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        if len(pending) >= batch_size:
            flush()
    flush()
    return np.asarray(predictions, dtype=int)


def run(args):
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    frame = pd.read_csv(args.grading_csv)
    required = {'id_code', 'status', 'true_grade', 'predicted_grade',
                'head_disagreement', 'image_path'} | {f'prob_grade_{i}' for i in range(5)}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required-set(frame))}')
    if not frame.status.eq('success').all() or frame.id_code.duplicated().any():
        raise ValueError('Ablation requires unique successful grading rows')
    if not frame.image_path.map(lambda path: Path(path).is_file()).all():
        raise FileNotFoundError('One or more grading image paths are missing')
    grades = frame.predicted_grade.to_numpy(int)
    argmax = frame[[f'prob_grade_{i}' for i in range(5)]].to_numpy(float).argmax(axis=1)
    quality = quality_predictions(frame, args.quality_checkpoint, args.quality_config,
                                  args.batch_size)
    reject = quality == 2
    disagree = frame.head_disagreement.astype(bool).to_numpy()
    # 0=auto decision; 1=recapture; 2=human review. Recapture takes precedence.
    quality_route = np.where(reject, 1, 0)
    dual_route = np.where(disagree, 2, 0)
    combined_route = np.where(reject, 1, dual_route)
    variants = [('grade_argmax_only', argmax, np.zeros(len(frame), dtype=int)),
                ('ordinal_grade_only', grades, np.zeros(len(frame), dtype=int)),
                ('ordinal_plus_quality_candidate', grades, quality_route),
                ('ordinal_plus_dual_head', grades, dual_route),
                ('ordinal_quality_dual_head', grades, combined_route)]
    summary = pd.DataFrame([ablation_row(frame.true_grade, predictions, routing, name)
                            for name, predictions, routing in variants])
    per_image = frame[['id_code', 'true_grade', 'predicted_grade', 'head_disagreement']].copy()
    per_image['argmax_grade'] = argmax
    per_image['eyeq_quality_candidate_prediction'] = quality
    per_image['quality_reject'] = reject
    per_image['combined_route'] = np.choose(combined_route,
                                             ['auto_decision', 'recapture', 'human_review'])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/'integrated_ablation.csv', index=False)
    per_image.to_csv(output/'integrated_ablation_per_image.csv', index=False)
    protocol = {
        'schema_version': 1, 'cohort': 'APTOS Fold-0 internal holdout, 733 images',
        'status': 'post_hoc_development_ablation_not_independent_validation',
        'm1_status': 'EyeQ-trained candidate, not default/deployed',
        'grading_csv_sha256': file_sha256(args.grading_csv),
        'quality_checkpoint_sha256': file_sha256(args.quality_checkpoint),
        'quality_config_sha256': file_sha256(args.quality_config),
        'routing_precedence': 'M1 Reject -> recapture; otherwise dual-head disagreement -> human review',
        'denominators': 'All input images retained; recaptures/reviews unresolved, not assumed correct.',
        'limitations': ['APTOS has no independent quality labels; M1 recaptures cannot be called correct.',
                        'Internal Fold-0 and EyeQ test have been exposed during development.',
                        'No lesion/Grad-CAM clinical-concordance variant: evidence is not validated.',
                        'Conditional sensitivity/specificity exclude unresolved images; inspect counts and coverage.'],
    }
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    result = ROOT/'results/verification_2026-09-21/eyeq_learned_quality'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grading-csv', type=Path,
                        default=ROOT/'results/verification_2026-09-18/fold0_dual_head/per_image_results.csv')
    parser.add_argument('--quality-checkpoint', type=Path,
                        default=result/'quality_mobilenet_v3_small.pth')
    parser.add_argument('--quality-config', type=Path, default=result/'quality_model_config.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'results/verification_2026-09-24/integrated_ablation')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())
