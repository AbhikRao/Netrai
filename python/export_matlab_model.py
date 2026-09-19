#!/usr/bin/env python3
"""Export the trained NetrAI dual-head model to ONNX for MATLAB.

The exported graph accepts a normalized NCHW tensor named ``input`` with the
fixed shape 1x3x512x512 and returns one six-value tensor: five uncalibrated
grade logits followed by the independent binary referable-DR logit. MATLAB
performs the same softmax/sigmoid, ordinal thresholding, and safety routing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch

from models.netrai_model import NetrAIModel
from utils.calibration import load_calibration_temperature
from utils.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from utils.safety import load_safety_policy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "python" / "weights" / "netrai_fold0_best.pth"
DEFAULT_THRESHOLDS = ROOT / "python" / "weights" / "fold0_thresholds.npy"
DEFAULT_OUTPUT = ROOT / "matlab" / "models" / "netrai_fold0.onnx"
DEFAULT_CALIBRATION = ROOT / "python" / "weights" / "calibration.json"
DEFAULT_SAFETY_POLICY = ROOT / "python" / "weights" / "safety_policy.json"


class DualHeadOutputs(torch.nn.Module):
    """Expose five grade logits plus one referable logit in one tensor."""

    def __init__(self, model: NetrAIModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grade_logits, referable_logits = self.model(image)
        return torch.cat((grade_logits, referable_logits.unsqueeze(1)), dim=1)


def export_model(
        checkpoint_path: Path, thresholds_path: Path, calibration_path: Path,
        safety_policy_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = NetrAIModel("efficientnet_b4", 5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapper = DualHeadOutputs(model.eval()).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        example,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["model_outputs"],
        dynamo=False,
    )
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)

    thresholds = np.load(thresholds_path).astype(float)
    confidence_temperature = load_calibration_temperature(calibration_path)
    safety_policy = load_safety_policy(safety_policy_path)
    config = {
        "schema_version": 2,
        "model_file": output_path.name,
        "architecture": "EfficientNet-B4 dual-head (grade + referable heads exported)",
        "input_name": "input",
        "output_name": "model_outputs",
        "output_schema": [
            "grade_logit_0", "grade_logit_1", "grade_logit_2",
            "grade_logit_3", "grade_logit_4", "referable_logit"
        ],
        "input_layout": "NCHW",
        "input_size": 512,
        "black_crop_threshold": 7,
        "ben_graham_sigma": 10,
        "imagenet_mean": IMAGENET_MEAN.tolist(),
        "imagenet_std": IMAGENET_STD.tolist(),
        "grade_thresholds": thresholds.tolist(),
        "fold": int(checkpoint.get("fold", 0)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_qwk": float(checkpoint.get("qwk", float("nan"))),
        "referable_grade_minimum": 2,
        "referable_head_threshold": safety_policy["referable_head_threshold"],
        "head_disagreement_action": safety_policy["head_disagreement_action"],
        "confidence_temperature": confidence_temperature,
        "confidence_calibration": {
            "method": "scalar_temperature_scaling",
            "artifact": str(calibration_path.relative_to(ROOT)),
            "applies_to": "grade_logits_for_confidence_only",
        },
    }
    config_path = output_path.with_name("netrai_config.json")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Exported: {output_path}")
    print(f"Config:   {config_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--safety-policy", type=Path, default=DEFAULT_SAFETY_POLICY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_model(
        args.checkpoint.resolve(), args.thresholds.resolve(),
        args.calibration.resolve(), args.safety_policy.resolve(),
        args.output.resolve())
