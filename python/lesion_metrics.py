"""Lesion segmentation metrics from per-image probability histograms (exact up to the bin width).

Each image contributes, for each lesion type, a histogram of predicted probabilities over its positive
(lesion) pixels and over its negative pixels. From the summed histograms we get pixel-level AUPR, AUC and Dice at
any threshold; resampling images and summing their histograms gives image-level bootstrap intervals without
keeping full-resolution maps in memory. Lesion-level (connected component) detection with false positives per image
is computed separately.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

LESIONS = ('EX', 'HE', 'MA', 'SE')
BINS = 1000


def histograms(prob: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None, bins: int = BINS) -> np.ndarray:
    """(2, bins) counts: row 0 = lesion pixels, row 1 = non-lesion pixels, indexed by probability bin."""
    prob = np.clip(np.asarray(prob, np.float64), 0.0, 1.0)
    index = np.minimum((prob * bins).astype(np.int64), bins - 1)
    mask = np.asarray(mask) > 0
    keep = np.ones(mask.shape, bool) if valid is None else np.asarray(valid, bool)
    out = np.zeros((2, bins))
    out[0] = np.bincount(index[mask & keep], minlength=bins)
    out[1] = np.bincount(index[~mask & keep], minlength=bins)
    return out


def _tail(hist: np.ndarray):
    """Counts of positives/negatives with probability bin >= b, for every b (descending-threshold sweep)."""
    pos = np.cumsum(hist[0][::-1])[::-1]
    neg = np.cumsum(hist[1][::-1])[::-1]
    return pos, neg


def aupr(hist: np.ndarray) -> float:
    """Average precision (step-wise, as scikit-learn) from a summed (2, bins) histogram."""
    pos, neg = _tail(hist)
    total = hist[0].sum()
    if total == 0:
        return float('nan')
    recall = pos / total
    precision = np.divide(pos, pos + neg, out=np.zeros_like(pos), where=(pos + neg) > 0)
    # thresholds from the highest bin downwards; only bins that hold at least one pixel create a new operating point
    occupied = (hist[0] + hist[1]) > 0
    order = np.arange(len(recall))[::-1]
    order = order[occupied[order]]
    r = np.concatenate([[0.0], recall[order]])
    p = precision[order]
    return float(np.sum(np.diff(r) * p))


def roc_auc(hist: np.ndarray) -> float:
    pos_total, neg_total = hist[0].sum(), hist[1].sum()
    if pos_total == 0 or neg_total == 0:
        return float('nan')
    below_neg = np.cumsum(hist[1]) - hist[1]  # negatives strictly below each bin
    return float((hist[0] * (below_neg + 0.5 * hist[1])).sum() / (pos_total * neg_total))


def dice_at(hist: np.ndarray, threshold: float, bins: int = BINS) -> float:
    cut = int(np.ceil(threshold * bins))
    tp, fp = hist[0][cut:].sum(), hist[1][cut:].sum()
    fn = hist[0][:cut].sum()
    return float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float('nan')


def best_dice_threshold(hist: np.ndarray, bins: int = BINS) -> tuple[float, float]:
    """Threshold and Dice maximising Dice on this histogram (choose it on validation, apply it to test)."""
    best = (float('nan'), -1.0)
    for cut in range(1, bins):
        d = dice_at(hist, cut / bins, bins)
        if not np.isnan(d) and d > best[1]:
            best = (cut / bins, d)
    return best


def iou_at(hist: np.ndarray, threshold: float, bins: int = BINS) -> float:
    d = dice_at(hist, threshold, bins)
    return float(d / (2 - d)) if not np.isnan(d) else float('nan')


def lesion_level(prob: np.ndarray, mask: np.ndarray, threshold: float, min_pixels: int = 1) -> dict:
    """Connected-component detection: a ground-truth lesion is found if any predicted pixel overlaps it; a
    predicted component with no ground-truth overlap is a false positive."""
    truth = np.asarray(mask) > 0
    pred = np.asarray(prob) >= threshold
    gt_labels, n_gt = ndimage.label(truth)
    pr_labels, n_pr = ndimage.label(pred)
    found = int(np.sum(ndimage.maximum(pred, gt_labels, np.arange(1, n_gt + 1)) > 0)) if n_gt else 0
    false_positive = 0
    if n_pr:  # vectorised: a predicted component is a false positive if it overlaps no truth pixel (and is large enough)
        overlaps = np.asarray(ndimage.maximum(truth, pr_labels, np.arange(1, n_pr + 1))) > 0
        sizes = np.bincount(pr_labels.ravel(), minlength=n_pr + 1)[1:]
        false_positive = int(np.sum(~overlaps & (sizes >= min_pixels)))
    return {'gt_lesions': int(n_gt), 'detected': int(found), 'false_positives': int(false_positive)}


class LesionSet:
    """Per-image histograms for all lesion types: build with ``add`` then summarise or bootstrap."""

    def __init__(self, lesions=LESIONS, bins: int = BINS):
        self.lesions, self.bins = tuple(lesions), bins
        self.per_image: list[np.ndarray] = []  # each (lesions, 2, bins)
        self.max_prob: list[np.ndarray] = []
        self.has_lesion: list[np.ndarray] = []

    def add(self, probs: np.ndarray, masks: np.ndarray, valid: np.ndarray | None = None) -> None:
        """probs/masks: (H, W, L)."""
        h = np.stack([histograms(probs[..., i], masks[..., i], valid, self.bins) for i in range(len(self.lesions))])
        self.per_image.append(h)
        self.max_prob.append(np.array([probs[..., i].max() for i in range(len(self.lesions))]))
        self.has_lesion.append(np.array([(masks[..., i] > 0).any() for i in range(len(self.lesions))]))

    def total(self, rows=None) -> np.ndarray:
        stack = np.stack(self.per_image)
        return (stack if rows is None else stack[rows]).sum(axis=0)

    def summary(self, thresholds: dict | None = None, rows=None) -> dict:
        total = self.total(rows)
        out = {}
        for i, name in enumerate(self.lesions):
            hist = total[i]
            entry = {'aupr': aupr(hist), 'auc': roc_auc(hist), 'positive_pixels': int(hist[0].sum())}
            threshold = (thresholds or {}).get(name)
            if threshold is None:
                threshold, entry['dice_best_on_this_set'] = best_dice_threshold(hist, self.bins)
                entry['threshold_chosen_on_this_set'] = threshold
            else:
                entry['dice'] = dice_at(hist, threshold, self.bins)
                entry['iou'] = iou_at(hist, threshold, self.bins)
                entry['threshold'] = threshold
            out[name] = entry
        return out

    def bootstrap(self, replicates: int = 1000, seed: int = 0, thresholds: dict | None = None) -> dict:
        rng = np.random.default_rng(seed)
        n = len(self.per_image)
        samples = {name: {'aupr': [], 'dice': []} for name in self.lesions}
        for _ in range(replicates):
            rows = rng.integers(n, size=n)
            total = self.total(rows)
            for i, name in enumerate(self.lesions):
                samples[name]['aupr'].append(aupr(total[i]))
                if thresholds and thresholds.get(name) is not None:
                    samples[name]['dice'].append(dice_at(total[i], thresholds[name], self.bins))
        out = {}
        for name in self.lesions:
            out[name] = {}
            for metric, values in samples[name].items():
                values = np.array(values, float)
                values = values[~np.isnan(values)]
                if len(values):
                    out[name][metric + '_ci95'] = [float(v) for v in np.percentile(values, [2.5, 97.5])]
        return out
