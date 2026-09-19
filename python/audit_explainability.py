#!/usr/bin/env python3
"""Audit Grad-CAM maps for border/FOV shortcut-attention indicators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from inference import (
    _model_display_image, generate_real_gradcam, get_calibration_temperature,
    predict_with_model_details,
)
from modules.explainability import assess_gradcam_quality, calibrate_confidence
from utils.preprocessing import model_space_fov_mask


ROOT = Path(__file__).resolve().parents[1]


def main(args: argparse.Namespace) -> None:
    frame = pd.read_csv(args.csv)
    selected = pd.concat([
        group.sample(n=min(args.per_grade, len(group)), random_state=args.seed)
        for _, group in frame.groupby("diagnosis", sort=True)
    ]).sort_values(["diagnosis", "id_code"])
    args.output.mkdir(parents=True, exist_ok=True)
    temperature = get_calibration_temperature()
    records = []
    for row in selected.itertuples(index=False):
        path = args.images / f"{row.id_code}.png"
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            records.append({"id_code": row.id_code, "status": "error", "error": "decode failed"})
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        prediction = predict_with_model_details(rgb)
        grade = int(prediction['grades'])
        probabilities = prediction['grade_probabilities']
        calibrated = calibrate_confidence(probabilities, temperature)
        heatmap = generate_real_gradcam(rgb)
        if heatmap is None:
            records.append({"id_code": row.id_code, "status": "error", "error": "Grad-CAM failed"})
            continue
        fov = model_space_fov_mask(rgb, size=heatmap.shape[0])
        metrics = assess_gradcam_quality(heatmap, fov)
        artifact_path = ''
        if args.save_maps and (args.save_all_maps or metrics['shortcut_flag']):
            artifact_dir = args.output / ('flagged_maps' if metrics['shortcut_flag'] else 'review_maps')
            artifact_dir.mkdir(parents=True, exist_ok=True)
            display = _model_display_image(rgb)
            heatmap_u8 = np.uint8(np.clip(heatmap, 0, 1) * 255)
            colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            if colored.shape[:2] != display_bgr.shape[:2]:
                colored = cv2.resize(
                    colored, (display_bgr.shape[1], display_bgr.shape[0]))
            overlay = cv2.addWeighted(display_bgr, 0.55, colored, 0.45, 0)
            artifact_path = str(artifact_dir / f'{row.id_code}_gradcam_qc.png')
            cv2.imwrite(artifact_path, overlay)
        records.append({
            "id_code": row.id_code,
            "true_grade": int(row.diagnosis),
            "predicted_grade": int(grade),
            "raw_confidence": float(probabilities[grade]),
            "confidence": float(calibrated[grade]),
            "calibration_temperature": temperature,
            "referable_head_probability": float(
                prediction['referable_head_probabilities']),
            "head_disagreement": bool(prediction['head_disagreements']),
            "triage_action": str(prediction['triage_actions']),
            "qc_overlay_path": artifact_path,
            "status": "success",
            "error": "",
            **metrics,
        })

    results = pd.DataFrame(records)
    results.to_csv(args.output / "gradcam_qc_per_image.csv", index=False)
    valid = results[results["status"] == "success"]
    summary = {
        "requested": int(len(results)),
        "successful": int(len(valid)),
        "errors": int((results["status"] == "error").sum()),
        "flagged": int(valid["shortcut_flag"].sum()) if len(valid) else 0,
        "flag_rate": float(valid["shortcut_flag"].mean()) if len(valid) else None,
        "median_border_enrichment": float(valid["border_enrichment"].median()) if len(valid) else None,
        "maximum_corner_enrichment": float(valid["max_corner_enrichment"].max()) if len(valid) else None,
        "maximum_top5_border_fraction": float(valid["top5_border_fraction"].max()) if len(valid) else None,
        "median_outside_fov_attention_fraction": float(valid["outside_fov_attention_fraction"].median()) if len(valid) else None,
        "flag_rate_by_true_grade": {
            str(int(grade)): float(group['shortcut_flag'].mean())
            for grade, group in valid.groupby('true_grade')
        } if len(valid) else {},
        "interpretation": (
            "Heuristic QC only. A flag indicates possible shortcut attention and requires visual/clinical review; "
            "it is not evidence that a model prediction is incorrect."
        ),
    }
    (args.output / "gradcam_qc_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/aptos2019/train.csv")
    parser.add_argument("--images", type=Path, default=ROOT / "data/aptos2019/train_images")
    parser.add_argument("--output", type=Path, default=ROOT / "results/verification_2026-09-17/gradcam_qc")
    parser.add_argument("--per-grade", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-maps", action=argparse.BooleanOptionalAction, default=True,
        help="Save QC overlays for flagged maps (default: true)")
    parser.add_argument(
        "--save-all-maps", action="store_true",
        help="Save overlays for non-flagged maps too")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
