#!/usr/bin/env python3
"""Export the evaluated EyeQ quality candidate and fixed MATLAB parity fixtures."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from scipy.io import savemat
from scipy.special import softmax

from calibrate_quality import file_sha256
from models.quality_model import QualityModel, quality_decisions, quality_input


ROOT = Path(__file__).resolve().parents[1]


def run(args):
    metrics = json.loads(args.metrics.read_text(encoding='utf-8'))
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if metrics['status'] != 'research_candidate_not_deployed':
        raise ValueError('Unexpected candidate status')
    if not metrics['held_out_gate']['passed']:
        raise ValueError('Refusing to export a quality candidate that failed its gate')
    if metrics['checkpoint_sha256'] != file_sha256(args.checkpoint):
        raise ValueError('Checkpoint hash does not match evaluation evidence')
    model = QualityModel(pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
    torch.onnx.export(
        model, example, args.output, export_params=True, opset_version=17,
        do_constant_folding=True, input_names=['quality_input'],
        output_names=['quality_logits'], dynamo=False)
    onnx.checker.check_model(onnx.load(args.output))
    session = ort.InferenceSession(str(args.output), providers=['CPUExecutionProvider'])
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)
    model_input = quality_input(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(model_input[None])).numpy()[0]
    onnx_logits = session.run(
        ['quality_logits'], {'quality_input': model_input[None]})[0][0]
    error = float(np.max(np.abs(torch_logits - onnx_logits)))
    if error > args.tolerance:
        raise ValueError(f'Quality ONNX parity error {error} exceeds {args.tolerance}')
    probabilities = softmax(onnx_logits / config['confidence_temperature'])
    prediction = int(quality_decisions(
        probabilities[None], config['reject_threshold'])[0])
    exported = {
        **config,
        'model_file': args.output.name,
        'input_name': 'quality_input', 'output_name': 'quality_logits',
        'input_layout': 'NCHW', 'onnx_sha256': file_sha256(args.output),
        'onnx_torch_max_logit_error': error, 'onnx_tolerance': args.tolerance,
    }
    config_path = args.output.with_name('quality_model_candidate.json')
    config_path.write_text(json.dumps(exported, indent=2) + '\n', encoding='utf-8')
    savemat(args.output.with_name('quality_parity_fixture.mat'), {
        'model_input': model_input.transpose(1, 2, 0),
        'expected_logits': onnx_logits,
        'expected_probabilities': probabilities,
        'expected_prediction': prediction,
    })
    reference = {
        'image_relative_to_project': args.image.resolve().relative_to(ROOT).as_posix(),
        'expected_logits': onnx_logits.tolist(),
        'expected_probabilities': probabilities.tolist(),
        'expected_prediction': prediction,
        'class_names': config['classes'],
        'reject_threshold': config['reject_threshold'],
        'pytorch_onnx_max_logit_error': error,
    }
    args.output.with_name('quality_parity_reference.json').write_text(
        json.dumps(reference, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'onnx': str(args.output), 'max_logit_error': error,
                      'prediction': prediction}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'python/weights/quality_model_candidate.pth')
    parser.add_argument('--metrics', type=Path, default=ROOT / 'results/benchmarks/image_quality/quality_model_metrics.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'python/weights/quality_model_candidate.json')
    parser.add_argument('--image', type=Path, default=ROOT / 'data/eyeq/images/test/1_right.jpeg')
    parser.add_argument('--output', type=Path, default=ROOT / 'matlab/models/quality_model_candidate.onnx')
    parser.add_argument('--tolerance', type=float, default=1e-4)
    run(parser.parse_args())
