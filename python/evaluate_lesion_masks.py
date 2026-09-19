#!/usr/bin/env python3
"""Evaluate M2 lesion candidates against pixel-level annotation masks.

The input manifest must contain ``image_path`` and one or more supported mask
columns. Hard/soft exudate masks are unioned at runtime because the classical
detector produces one combined exudate candidate mask. Blank cells are skipped.
Paths may be absolute or relative to the manifest. The script exports
per-image, aggregate, and bootstrap-confidence-interval evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from modules.segmentation import (
    detect_exudates, detect_hemorrhages, detect_microaneurysms,
    localize_fovea, localize_optic_disc, segment_vessels,
)
from utils.preprocessing import preprocess_fundus


MASK_GROUPS = {
    "ma": ("ma_mask",),
    "hem": ("hem_mask",),
    "exudate": ("exudate_mask", "hard_exudate_mask", "soft_exudate_mask"),
    "optic_disc": ("optic_disc_mask",),
}


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def transform_mask(mask: np.ndarray, target_size: int = 512) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = target_size / max(h, w)
    resized = cv2.resize(mask, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_NEAREST)
    top = (target_size-resized.shape[0])//2
    bottom = target_size-resized.shape[0]-top
    left = (target_size-resized.shape[1])//2
    right = target_size-resized.shape[1]-left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT) > 0


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool); gt = truth.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pixel_metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "pixel_precision": tp/max(tp+fp, 1),
        "pixel_recall": tp/max(tp+fn, 1),
        "pixel_dice": 2*tp/max(2*tp+fp+fn, 1),
        "pixel_iou": tp/max(tp+fp+fn, 1),
    }
    return {**pixel_metrics, **component_metrics(pred, gt)}


def component_metrics(prediction: np.ndarray, truth: np.ndarray, iou_threshold=0.10) -> dict[str, float]:
    """One-to-one lesion-component matching using an IoU threshold."""
    pred_count, pred_labels = cv2.connectedComponents(prediction.astype(np.uint8), 8)
    truth_count, truth_labels = cv2.connectedComponents(truth.astype(np.uint8), 8)
    pred_ids = list(range(1, pred_count)); truth_ids = list(range(1, truth_count))
    candidates = []
    for pred_id in pred_ids:
        pred_mask = pred_labels == pred_id
        overlapping = np.unique(truth_labels[pred_mask])
        for truth_id in overlapping[overlapping > 0]:
            truth_mask = truth_labels == truth_id
            intersection = np.logical_and(pred_mask, truth_mask).sum()
            union = np.logical_or(pred_mask, truth_mask).sum()
            iou = intersection/max(union, 1)
            if iou >= iou_threshold:
                candidates.append((iou, pred_id, int(truth_id)))
    matched_pred, matched_truth = set(), set()
    for _, pred_id, truth_id in sorted(candidates, reverse=True):
        if pred_id not in matched_pred and truth_id not in matched_truth:
            matched_pred.add(pred_id); matched_truth.add(truth_id)
    tp = len(matched_truth); fp = len(pred_ids)-tp; fn = len(truth_ids)-tp
    return {
        "lesion_tp": tp, "lesion_fp": fp, "lesion_fn": fn,
        "lesion_precision": tp/max(tp+fp, 1),
        "lesion_recall": tp/max(tp+fn, 1),
        "lesion_f1": 2*tp/max(2*tp+fp+fn, 1),
    }


def load_union_mask(base: Path, item: dict, columns: tuple[str, ...]) -> np.ndarray | None:
    masks = []
    for column in columns:
        value = item.get(column)
        if value is None or pd.isna(value) or str(value).strip() == "":
            continue
        path = resolve(base, str(value))
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(path)
        masks.append(transform_mask(raw))
    if not masks:
        return None
    return np.logical_or.reduce(masks)


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1))


def aggregate_counts(frame: pd.DataFrame) -> dict[str, float]:
    totals = frame[["tp", "fp", "fn", "lesion_tp", "lesion_fp", "lesion_fn"]].sum()
    return {
        "pixel_precision": ratio(totals.tp, totals.tp + totals.fp),
        "pixel_recall": ratio(totals.tp, totals.tp + totals.fn),
        "pixel_dice": ratio(2 * totals.tp, 2 * totals.tp + totals.fp + totals.fn),
        "pixel_iou": ratio(totals.tp, totals.tp + totals.fp + totals.fn),
        "lesion_precision": ratio(
            totals.lesion_tp, totals.lesion_tp + totals.lesion_fp),
        "lesion_recall": ratio(
            totals.lesion_tp, totals.lesion_tp + totals.lesion_fn),
        "lesion_f1": ratio(
            2 * totals.lesion_tp,
            2 * totals.lesion_tp + totals.lesion_fp + totals.lesion_fn),
        "false_positives_per_image": ratio(totals.lesion_fp, len(frame)),
    }


def bootstrap_intervals(
        output: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    metric_names = list(aggregate_counts(output.iloc[:1]).keys())
    for lesion, group in output.groupby("lesion", sort=True):
        samples = {metric: [] for metric in metric_names}
        for _ in range(replicates):
            indices = rng.integers(0, len(group), size=len(group))
            metrics_row = aggregate_counts(group.iloc[indices])
            for metric in metric_names:
                samples[metric].append(metrics_row[metric])
        point = aggregate_counts(group)
        row = {"lesion": lesion, "images": len(group)}
        for metric in metric_names:
            low, high = np.percentile(samples[metric], [2.5, 97.5])
            row[metric] = point[metric]
            row[f"{metric}_ci95_low"] = float(low)
            row[f"{metric}_ci95_high"] = float(high)
        rows.append(row)
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    if "image_path" not in manifest:
        raise ValueError("manifest requires image_path")
    available = {
        lesion: tuple(column for column in columns if column in manifest.columns)
        for lesion, columns in MASK_GROUPS.items()
    }
    available = {lesion: columns for lesion, columns in available.items() if columns}
    if not available:
        supported = sorted({column for columns in MASK_GROUPS.values() for column in columns})
        raise ValueError(f"manifest needs at least one mask column: {supported}")
    rows = []
    base = args.manifest.resolve().parent
    for item in tqdm(
            manifest.itertuples(index=False), total=len(manifest),
            desc="IDRiD lesion evaluation"):
        image_path = resolve(base, str(item.image_path))
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        proc, green, fov = preprocess_fundus(rgb)
        od_center, od_radius = localize_optic_disc(proc)
        fovea = localize_fovea(green, od_center, od_radius)
        vessels = segment_vessels(green, fov)
        ma, _, _ = detect_microaneurysms(green, vessels, fov, od_center, od_radius)
        hem, _, _ = detect_hemorrhages(green, vessels, ma, fov, od_center, od_radius)
        exudate, _, _, _ = detect_exudates(proc, green, fovea, od_center, od_radius, fov)
        optic_disc = np.zeros_like(fov, dtype=np.uint8)
        cv2.circle(optic_disc, tuple(map(int, od_center)), int(od_radius), 255, -1)
        predictions = {
            "ma": ma > 0,
            "hem": hem > 0,
            "exudate": exudate > 0,
            "optic_disc": optic_disc > 0,
        }
        item_dict = item._asdict()
        for lesion, columns in available.items():
            truth = load_union_mask(base, item_dict, columns)
            if truth is None:
                continue
            rows.append({
                "image_path": str(image_path),
                "annotation_columns": "+".join(columns),
                "lesion": lesion,
                **metrics(predictions[lesion], truth),
            })
    output = pd.DataFrame(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "lesion_mask_per_image.csv", index=False)
    aggregate = bootstrap_intervals(
        output, replicates=args.bootstrap_replicates, seed=args.seed)
    aggregate.to_csv(args.output / "lesion_mask_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/lesion_mask_validation"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
