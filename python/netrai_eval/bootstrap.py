"""Cluster (patient / exam) bootstrap with explicit seeds. Rows of one cluster are resampled together."""
from __future__ import annotations

from typing import Callable

import numpy as np


def cluster_members(clusters) -> list[np.ndarray]:
    clusters = np.asarray(clusters)
    _, inverse = np.unique(clusters, return_inverse=True)
    order = np.argsort(inverse, kind='stable')
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def resample(members: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    chosen = rng.integers(len(members), size=len(members))
    return np.concatenate([members[k] for k in chosen])


def percentile_ci(samples, level: float = 0.95) -> list[float]:
    samples = np.asarray(samples, float)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return [float('nan'), float('nan')]
    low, high = 100 * (1 - level) / 2, 100 * (1 + level) / 2
    return [float(v) for v in np.percentile(samples, [low, high])]


def bootstrap(statistic: Callable[[np.ndarray], dict], clusters, replicates: int = 1000, seed: int = 0) -> dict:
    """Estimate and 95% interval for every key of ``statistic(indices) -> {name: float}``.

    ``statistic`` receives row indices (all rows for the point estimate, resampled rows otherwise).
    """
    members = cluster_members(clusters)
    everything = np.arange(len(np.asarray(clusters)))
    estimate = statistic(everything)
    draws = {name: np.empty(replicates) for name in estimate}
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        values = statistic(resample(members, rng))
        for name in draws:
            draws[name][r] = values[name]
    return {name: {'estimate': float(estimate[name]), 'ci95': percentile_ci(draws[name]),
                   'valid_replicates': int(np.sum(~np.isnan(draws[name])))} for name in estimate}


def paired_bootstrap(statistic_a: Callable[[np.ndarray], dict], statistic_b: Callable[[np.ndarray], dict],
                     clusters, replicates: int = 1000, seed: int = 0) -> dict:
    """Paired differences ``a - b`` of equally named statistics on shared cluster resamples."""
    members = cluster_members(clusters)
    everything = np.arange(len(np.asarray(clusters)))
    est_a, est_b = statistic_a(everything), statistic_b(everything)
    if set(est_a) != set(est_b):
        raise ValueError('Paired statistics must have identical keys')
    diffs = {name: np.empty(replicates) for name in est_a}
    rng = np.random.default_rng(seed)
    for r in range(replicates):
        rows = resample(members, rng)
        a, b = statistic_a(rows), statistic_b(rows)
        for name in diffs:
            diffs[name][r] = a[name] - b[name]
    return {name: {'a': float(est_a[name]), 'b': float(est_b[name]), 'difference': float(est_a[name] - est_b[name]),
                   'difference_ci95': percentile_ci(diffs[name]),
                   'valid_replicates': int(np.sum(~np.isnan(diffs[name])))} for name in est_a}
