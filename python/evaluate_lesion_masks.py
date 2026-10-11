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


def metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool); gt = truth.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    pixel_metrics = {
        "tp": tp, "fp": fp, "fn": fn,
        "pixel_precision": ratio(tp, tp+fp),
        "pixel_recall": ratio(tp, tp+fn),
        "pixel_dice": ratio(2*tp, 2*tp+fp+fn),
        "pixel_iou": ratio(tp, tp+fp+fn),
    }
    return {**pixel_metrics, **component_metrics(pred, gt)}


def component_metrics(prediction: np.ndarray, truth: np.ndarray, iou_threshold=0.10) -> dict[str, float]:
    """Maximum one-to-one IoU matching, using a vectorized overlap table."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    if prediction.shape != truth.shape:
        raise ValueError('Prediction and annotation coordinates must match')
    pred_count, pred_labels = cv2.connectedComponents(prediction.astype(np.uint8), 8)
    truth_count, truth_labels = cv2.connectedComponents(truth.astype(np.uint8), 8)
    pred_area = np.bincount(pred_labels.ravel(), minlength=pred_count)
    truth_area = np.bincount(truth_labels.ravel(), minlength=truth_count)
    overlap = np.bincount((pred_labels.astype(np.int64)*truth_count+truth_labels).ravel(),
                          minlength=pred_count*truth_count).reshape(pred_count, truth_count)
    p, t = np.nonzero(overlap[1:, 1:])
    union = pred_area[p+1]+truth_area[t+1]-overlap[p+1, t+1]
    accepted = overlap[p+1, t+1]/np.maximum(union, 1) >= iou_threshold
    if pred_count > 1 and truth_count > 1:
        graph = csr_matrix((np.ones(int(accepted.sum())), (p[accepted], t[accepted])),
                           shape=(pred_count-1, truth_count-1))
        tp = int(np.sum(maximum_bipartite_matching(graph, perm_type='column') >= 0))
    else:
        tp = 0
    fp, fn = pred_count-1-tp, truth_count-1-tp
    return {'lesion_tp': tp, 'lesion_fp': fp, 'lesion_fn': fn,
            'lesion_precision': ratio(tp, tp+fp), 'lesion_recall': ratio(tp, tp+fn),
            'lesion_f1': ratio(2*tp, 2*tp+fp+fn)}


def restore_native_mask(prediction, shape):
    """Undo analysis padding and integer resize; never downsample tiny truth lesions."""
    height, width = shape
    size = prediction.shape[0]
    scale = size/max(height, width)
    resized_width, resized_height = int(width*scale), int(height*scale)
    left, top = (size-resized_width)//2, (size-resized_height)//2
    cropped = prediction[top:top+resized_height, left:left+resized_width].astype(np.uint8)
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


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
        masks.append(raw > 0)
    if not masks:
        return None
    return np.logical_or.reduce(masks)


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float('nan')


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
    for (split, lesion), group in output.groupby(["split", "lesion"], sort=True):
        samples = {metric: [] for metric in metric_names}
        for _ in range(replicates):
            indices = rng.integers(0, len(group), size=len(group))
            metrics_row = aggregate_counts(group.iloc[indices])
            for metric in metric_names:
                samples[metric].append(metrics_row[metric])
        point = aggregate_counts(group)
        row = {"split": split, "lesion": lesion, "images": len(group)}
        for metric in metric_names:
            finite = np.asarray(samples[metric], dtype=float)
            finite = finite[np.isfinite(finite)]
            low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (np.nan, np.nan)
            row[metric] = point[metric]
            row[f"{metric}_ci95_low"] = float(low)
            row[f"{metric}_ci95_high"] = float(high)
            row[f"{metric}_bootstrap_defined"] = len(finite)
        rows.append(row)
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest)
    if not {'image_path', 'split'}.issubset(manifest):
        raise ValueError('manifest requires image_path and explicit split')
    if manifest['split'].isna().any() or manifest.image_path.duplicated().any():
        raise ValueError('Missing split or duplicate image path')
    if args.split != 'all':
        manifest = manifest.loc[manifest['split'].eq(args.split)].copy()
    if manifest.empty:
        raise ValueError('No images selected for requested split')
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
            if truth.shape != rgb.shape[:2]:
                raise ValueError(f'Annotation/image size mismatch: {image_path}')
            rows.append({
                "split": item.split,
                "image_path": str(image_path),
                "annotation_columns": "+".join(columns),
                "lesion": lesion,
                "evaluation_space": "native_annotation_v2",
                **metrics(restore_native_mask(predictions[lesion], truth.shape), truth),
            })
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError('No annotated cases; missing annotations are not negatives')
    args.output.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output / "lesion_mask_per_image.csv", index=False)
    aggregate = bootstrap_intervals(
        output, replicates=args.bootstrap_replicates, seed=args.seed)
    aggregate.to_csv(args.output / "lesion_mask_aggregate.csv", index=False)
    print(aggregate.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'test', 'all'), default='test')
    parser.add_argument("--output", type=Path, default=Path("results/lesion_mask_validation"))
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
