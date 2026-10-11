#!/usr/bin/env python3
"""Export an adaptation-ladder rung as a MATLAB deployment variant directory.

The variant keeps the deployed architecture, ``legacy-area-v1`` input contract,
config schema and six-output ONNX graph. Only weights (rungs C/D) and the
decision policy (grade thresholds, referral-head threshold, temperature) change.
The directory is created next to ``matlab/src`` so the MATLAB deployment
bundle's ``../src`` role paths resolve; run ``writeDeploymentBundle`` there and
``verifyVariantParity`` afterwards. Nothing replaces the active deployment.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_matlab_model import DualHeadOutputs
from models.netrai_model import NetrAIModel
from utils.artifacts import atomic_json, sha256_file
from utils.bundle import load_legacy_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / 'matlab/models'
PREFIX = {'block6.': 'backbone.blocks.6.', 'conv_head.': 'backbone.conv_head.', 'bn2.': 'backbone.bn2.',
          'head.': 'head.', 'grade_head.': 'grade_head.', 'ref_head.': 'ref_head.'}


def adapted_model(rung_checkpoint: Path | None) -> NetrAIModel:
    model = NetrAIModel('efficientnet_b4', 5, pretrained=False)
    model.load_state_dict(load_legacy_checkpoint(ROOT / 'python/weights/netrai_fold0_best.pth')['model_state_dict'])
    if rung_checkpoint is not None:
        state = torch.load(rung_checkpoint, map_location='cpu', weights_only=True)['state_dict']
        target = model.state_dict()
        for key, value in state.items():
            mapped = next(PREFIX[p] + key[len(p):] for p in PREFIX if key.startswith(p))
            if target[mapped].shape != value.shape:
                raise ValueError(f'Shape mismatch for {mapped}')
            target[mapped] = value
        model.load_state_dict(target)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant-id', required=True)
    parser.add_argument('--results', type=Path, required=True, help='train_grading_adapter results.json')
    parser.add_argument('--rung', required=True, choices=('B', 'C', 'D'),
                        help='B = legacy weights; C/D = adapted weights from the run')
    parser.add_argument('--policy', type=Path, required=True,
                        help='site policy JSON from calibrate_site_policy.py')
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    policy = json.loads(args.policy.read_text())
    expected_rung = 'A' if args.rung == 'B' else args.rung
    if policy['rung'] != expected_rung:
        raise ValueError(f'Policy was fitted on rung {policy["rung"]}, not {expected_rung}')
    thresholds = np.asarray(policy['thresholds'], dtype=float)
    if thresholds.shape != (4,) or not np.all(np.diff(thresholds) > 0) or thresholds[0] <= 0 or thresholds[-1] >= 4:
        raise ValueError('Policy thresholds must be strictly increasing inside (0, 4)')
    output = ROOT / 'matlab' / f'variant_{args.variant_id}'
    if output.exists():
        raise FileExistsError(f'{output} exists; variants are immutable')
    output.mkdir()
    config = json.loads((ACTIVE / 'netrai_config.json').read_text())
    config['grade_thresholds'] = [float(v) for v in policy['thresholds']]
    config['referable_head_threshold'] = float(policy['head_threshold'])
    if policy['temperature'] is not None:
        config['confidence_temperature'] = float(policy['temperature'])
    if args.rung == 'B':
        shutil.copy2(ACTIVE / 'netrai_fold0.onnx', output / 'netrai_fold0.onnx')
        model = adapted_model(None)
    else:
        checkpoint = args.results.parent / f'rung_{args.rung}.pt'
        model = adapted_model(checkpoint)
        torch.onnx.export(DualHeadOutputs(model).eval(), torch.zeros(1, 3, 512, 512), output / 'netrai_fold0.onnx',
                          export_params=True, opset_version=17, do_constant_folding=True,
                          input_names=['input'], output_names=['model_outputs'], dynamo=False)
    (output / 'netrai_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    # PyTorch/ONNX parity on a fixed random tensor and the bundled MATLAB fixture input.
    session = ort.InferenceSession(str(output / 'netrai_fold0.onnx'), providers=['CPUExecutionProvider'])
    probe = np.random.default_rng(7).normal(size=(1, 3, 512, 512)).astype(np.float32)
    with torch.inference_mode():
        expected = DualHeadOutputs(model)(torch.from_numpy(probe)).numpy()
    error = float(np.abs(session.run(['model_outputs'], {'input': probe})[0] - expected).max())
    if error > 2e-4:
        raise RuntimeError(f'ONNX parity failed: {error}')
    atomic_json(output / 'variant_provenance.json', {
        'schema_version': 1, 'variant_id': args.variant_id, 'rung': args.rung,
        'status': 'research_candidate_not_deployed',
        'adaptation_results_sha256': sha256_file(args.results),
        'adaptation_code_sha256': (json.loads((args.results.parent / 'code_provenance.json').read_text())
                                   ['adapter_sha256'] if (args.results.parent / 'code_provenance.json').is_file()
                                   else results['code_sha256']),
        'rung_checkpoint_sha256': None if args.rung == 'B' else sha256_file(args.results.parent / f'rung_{args.rung}.pt'),
        'onnx_sha256': sha256_file(output / 'netrai_fold0.onnx'),
        'pytorch_onnx_max_abs_error_random_probe': error,
        'policy_sha256': sha256_file(args.policy), 'policy': {k: policy[k] for k in
            ('thresholds', 'head_threshold', 'temperature', 'calibration_cohort', 'target_sensitivity')},
        'training_pool': results['pool'],
        'selection_cohorts': results['selection'],
        'note': 'Run writeDeploymentBundle(modelPath, configPath) in MATLAB to bind this variant.'})
    print(json.dumps({'variant': str(output.relative_to(ROOT)), 'onnx_parity': error, 'policy': policy}, indent=2))


if __name__ == '__main__':
    main()
