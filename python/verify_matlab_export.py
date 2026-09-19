#!/usr/bin/env python3
"""Verify ONNX/PyTorch parity and write MATLAB reference predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.io import savemat
from sklearn.model_selection import StratifiedKFold

from models.netrai_model import NetrAIModel
from utils.preprocessing import preprocess_for_model


ROOT = Path(__file__).resolve().parents[1]


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def main(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.csv)
    folds = np.full(len(frame), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_number, (_, validation_indices) in enumerate(
            splitter.split(frame, frame["diagnosis"].astype(int))):
        folds[validation_indices] = fold_number
    fold0 = frame.loc[folds == 0, ["id_code", "diagnosis"]]
    fold0.to_csv(args.onnx.parent / "fold0_validation.csv", index=False)
    selected = frame.groupby("diagnosis", sort=True).head(1).sort_values("diagnosis")
    image_ids = selected["id_code"].tolist()
    labels = selected["diagnosis"].astype(int).tolist()

    arrays = []
    for image_id in image_ids:
        image_path = args.images / f"{image_id}.png"
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        arrays.append(preprocess_for_model(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    batch = np.stack(arrays).transpose(0, 3, 1, 2).astype(np.float32)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NetrAIModel("efficientnet_b4", 5, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.inference_mode():
        torch_grade_logits, torch_referable_logits = model(torch.from_numpy(batch))
        torch_outputs = np.concatenate((
            torch_grade_logits.numpy(), torch_referable_logits.numpy()[:, None]), axis=1)

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    onnx_rows = []
    for item in batch:
        onnx_rows.append(session.run(["model_outputs"], {"input": item[None]})[0][0])
    onnx_outputs = np.stack(onnx_rows)
    onnx_logits = onnx_outputs[:, :5]
    referable_logits = onnx_outputs[:, 5]
    referable_probabilities = 1.0 / (1.0 + np.exp(-referable_logits))

    thresholds = np.load(args.thresholds)
    probabilities = softmax(onnx_logits)
    continuous = probabilities @ np.arange(5, dtype=np.float32)
    grades = np.digitize(continuous, thresholds).clip(0, 4)
    max_abs_error = float(np.max(np.abs(torch_outputs - onnx_outputs)))
    if max_abs_error > args.tolerance:
        raise RuntimeError(
            f"ONNX parity failed: max output |PyTorch-ONNX|={max_abs_error:.6g} "
            f"> {args.tolerance:.6g}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for idx, image_id in enumerate(image_ids):
        record = {
            "image_id": image_id,
            "ground_truth": labels[idx],
            "predicted_grade": int(grades[idx]),
            "continuous_score": float(continuous[idx]),
            "referable_logit": float(referable_logits[idx]),
            "referable_head_probability": float(referable_probabilities[idx]),
            "referable_head_prediction": bool(referable_probabilities[idx] >= 0.5),
            "head_disagreement": bool((grades[idx] >= 2) != (referable_probabilities[idx] >= 0.5)),
        }
        record.update({f"logit_{i}": float(onnx_logits[idx, i]) for i in range(5)})
        record.update({f"probability_{i}": float(probabilities[idx, i]) for i in range(5)})
        records.append(record)

    pd.DataFrame(records).to_csv(args.output / "parity_predictions.csv", index=False)
    summary = {
        "sample_count": len(records),
        "max_absolute_output_error": max_abs_error,
        "tolerance": args.tolerance,
        "onnxruntime_version": ort.__version__,
        "passed": True,
    }
    (args.output / "parity_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    reference = {
        "description": "Reference outputs for MATLAB Online package verification",
        "logit_tolerance": 0.02,
        "samples": records,
    }
    (args.onnx.parent / "parity_reference.json").write_text(
        json.dumps(reference, indent=2) + "\n", encoding="utf-8"
    )
    smoke_bgr = cv2.imread(str(args.smoke_image), cv2.IMREAD_COLOR)
    if smoke_bgr is None:
        raise FileNotFoundError(args.smoke_image)
    smoke_input = preprocess_for_model(cv2.cvtColor(smoke_bgr, cv2.COLOR_BGR2RGB))
    smoke_outputs = session.run(
        ["model_outputs"], {"input": smoke_input.transpose(2, 0, 1)[None].astype(np.float32)}
    )[0][0]
    smoke_logits = smoke_outputs[:5]
    smoke_referable_probability = float(1.0 / (1.0 + np.exp(-smoke_outputs[5])))
    smoke_probabilities = softmax(smoke_logits[None])[0]
    smoke_score = float(smoke_probabilities @ np.arange(5, dtype=np.float32))
    smoke_grade = int(np.digitize(smoke_score, thresholds).clip(0, 4))
    preprocessing_reference = {
        "image_relative_to_project": str(args.smoke_image.resolve().relative_to(ROOT)),
        "expected_grade": smoke_grade,
        "expected_continuous_score": smoke_score,
        "expected_logits": smoke_logits.astype(float).tolist(),
        "expected_referable_logit": float(smoke_outputs[5]),
        "expected_referable_head_probability": smoke_referable_probability,
        "note": "End-to-end cross-language preprocessing check; small resize/blur differences may change logits.",
    }
    (args.onnx.parent / "preprocessing_reference.json").write_text(
        json.dumps(preprocessing_reference, indent=2) + "\n", encoding="utf-8"
    )
    # Exact preprocessed tensor for a MATLAB-side ONNX graph smoke test.  This
    # isolates importer/runtime parity from small library-specific resize
    # differences in end-to-end image preprocessing.
    savemat(
        args.onnx.parent / "parity_fixture.mat",
        {
            "model_input": batch[0].transpose(1, 2, 0),
            "expected_logits": onnx_logits[0],
            "expected_referable_logit": np.array([[referable_logits[0]]], dtype=np.float32),
            "expected_referable_head_probability": np.array(
                [[referable_probabilities[0]]], dtype=np.float32),
            "expected_grade": np.array([[grades[0]]], dtype=np.int32),
            "expected_continuous_score": np.array([[continuous[0]]], dtype=np.float32),
        },
        do_compression=True,
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=ROOT / "matlab/models/netrai_fold0.onnx")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "python/weights/netrai_fold0_best.pth")
    parser.add_argument("--thresholds", type=Path, default=ROOT / "python/weights/fold0_thresholds.npy")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/aptos2019/train.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/aptos2019/train_images")
    parser.add_argument("--output", type=Path, default=ROOT / "results/verification_2026-09-17/matlab_onnx")
    parser.add_argument("--smoke-image", type=Path, default=ROOT / "data/patient_sample.jpg")
    parser.add_argument("--tolerance", type=float, default=2e-4)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
