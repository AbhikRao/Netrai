"""Strict shared contracts: do not silently coerce invalid ML evidence."""
import numpy as np


def grades(values, name='grades'):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite 1-D array')
    if ((array < 0) | (array > 4) | (array != np.floor(array))).any():
        raise ValueError(f'{name} must contain integer grades 0–4')
    return array.astype(np.int64)


def probabilities(values, classes=5, name='probabilities'):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != classes:
        raise ValueError(f'{name} must have shape (N, {classes})')
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError(f'{name} must be finite and in [0, 1]')
    if not np.allclose(array.sum(axis=1), 1, atol=1e-5, rtol=0):
        raise ValueError(f'{name} rows must sum to one')
    return array


def ordinal_thresholds(values):
    array = np.asarray(values, dtype=float)
    if (array.shape != (4,) or not np.isfinite(array).all()
            or not np.all(np.diff(array) > 0) or array[0] < 0 or array[-1] > 4):
        raise ValueError('Expected four finite increasing thresholds in [0, 4]')
    return array


def image_ids(values):
    import pandas as pd
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError('Image IDs cannot be missing')
    series = series.astype(str).str.strip()
    if series.eq('').any() or series.duplicated().any():
        raise ValueError('Image IDs must be non-empty and unique')
    return series
