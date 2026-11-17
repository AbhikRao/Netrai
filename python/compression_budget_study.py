#!/usr/bin/env python3
"""X16: how small can an upload be before the referral decision changes? (rule in the private ledger, 2026-10-31)

300 gradable images (25 referable + 25 non-referable from each of ODIR, FDR26, Paraguay, SUSTech-SYSU, BRSET and mBRSET: exposed sets that
never trained or selected E_768) are re-encoded in memory: JPEG quality 90/75/50/30 at native size, and the long side downscaled to 1024 or
768 pixels (area interpolation, only if larger) then JPEG 90/75/50. The stored file is the reference (mostly JPEG already, so every
condition is a recompression). Per condition: E_768 referral flips against the stored file at its frozen cut-off (primary), legacy flips,
extra quality-gate rejections (heuristic IQS < 0.3) and extra input-guard flags, encoded size. Nothing compressed is written to disk;
the per-image table stays local and only aggregates are published (the sample includes PhysioNet DUA images).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
SETS = ('odir', 'fdr26', 'paraguay_hc', 'sustech_sysu', 'brset', 'mbrset')
SEED, PER_CLASS = 20261101, 25
CONDITIONS = [('stored', None, None)] + [(f'q{q}', None, q) for q in (90, 75, 50, 30)] + \
             [(f'{side}px_q{q}', side, q) for side in (1024, 768) for q in (90, 75, 50)]
FLIP_LIMIT, NEAR_BAND = 0.01, 0.1


def draw_sample() -> pd.DataFrame:
    import fundus_guard_study as study
    rng = np.random.default_rng(SEED)
    parts = []
    for source in SETS:
        table = study.inventory(source)
        table = table.loc[table.gradable & (table.grade >= 0)]
        for referable in (True, False):
            pool = table.loc[(table.grade >= 2) == referable]
            parts.append(pool.iloc[rng.choice(len(pool), PER_CLASS, replace=False)])
    return pd.concat(parts, ignore_index=True)[['source', 'image_id', 'path', 'grade']]


def encode(bgr: np.ndarray, side: int | None, quality: int | None) -> tuple[np.ndarray, int]:
    """Re-encoded image (decoded back to RGB) and its encoded size in bytes; never written to disk."""
    if side is not None and max(bgr.shape[:2]) > side:
        scale = side / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError('JPEG encoding failed')
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), int(data.size)


def score_all(sample: pd.DataFrame) -> pd.DataFrame:
    import inference
    from modules import variant_e768
    from modules.fundus_guard import FundusGuard
    from modules.quality import assess_image_quality
    runtime, guard = variant_e768.E768Runtime(), FundusGuard()
    rows = []
    for k, item in enumerate(sample.itertuples(index=False)):
        stored = Path(item.path).read_bytes()
        bgr = cv2.imdecode(np.frombuffer(stored, np.uint8), cv2.IMREAD_COLOR)
        for name, side, quality in CONDITIONS:
            rgb, size = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), len(stored)) if quality is None else encode(bgr, side, quality)
            e768 = runtime.predict([rgb])
            legacy = inference.predict_batch_with_details([rgb])
            rows.append({'k': k, 'source': item.source, 'grade': int(item.grade), 'condition': name, 'bytes': size,
                         'e768_score': float(e768['continuous_scores'][0]), 'e768_refer': bool(e768['grades'][0] >= 2),
                         'legacy_score': float(legacy['continuous_scores'][0]), 'legacy_refer': bool(legacy['grades'][0] >= 2),
                         'iqs': float(assess_image_quality(rgb)[0]), 'guard_flag': not guard.check(rgb)['is_fundus']})
        print(f'{k + 1}/{len(sample)}', flush=True)
    return pd.DataFrame(rows)


def upper95(k: int, n: int) -> float:
    return 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))


def auc(score: np.ndarray, label: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, score))


def summarize(table: pd.DataFrame, sample: pd.DataFrame, cut: float) -> dict:
    ref = table.loc[table.condition == 'stored'].set_index('k')
    label = ref.grade >= 2
    out = {'images': int(len(ref)), 'sets': list(SETS), 'per_set': {}, 'conditions': {}, 'cv2_version': cv2.__version__,
           'e768_referral_cut': cut, 'near_cut_images': int((np.abs(ref.e768_score - cut) < NEAR_BAND).sum()),
           'near_cut_band': NEAR_BAND}
    for source in SETS:
        paths = sample.loc[sample.source == source, 'path']
        out['per_set'][source] = {'formats': sorted({Path(p).suffix.lower() for p in paths}),
                                  'median_stored_kb': round(float(ref.loc[ref.source == source, 'bytes'].median()) / 1024, 1)}
    for name, _, _ in CONDITIONS:
        rows = table.loc[table.condition == name].set_index('k').loc[ref.index]
        flips = int((rows.e768_refer != ref.e768_refer).sum())
        change = np.abs(rows.e768_score - ref.e768_score)
        out['conditions'][name] = {
            'median_kb': round(float(rows.bytes.median()) / 1024, 1),
            'e768_flips': flips, 'e768_flip_upper95': round(upper95(flips, len(rows)), 4),
            'e768_score_change_median': float(change.median()), 'e768_score_change_p95': float(change.quantile(0.95)),
            'legacy_flips': int((rows.legacy_refer != ref.legacy_refer).sum()),
            'extra_quality_rejections': int(((rows.iqs < 0.3) & ~(ref.iqs < 0.3)).sum()),
            'extra_guard_flags': int((rows.guard_flag & ~ref.guard_flag).sum()),
            'e768_auc': auc(rows.e768_score.to_numpy(), label.to_numpy()),
            'legacy_auc': auc(rows.legacy_score.to_numpy(), label.to_numpy())}
    eligible = [(c['median_kb'], -int(name.split('q')[-1]), name) for name, c in out['conditions'].items() if name != 'stored'
                and c['e768_flips'] <= FLIP_LIMIT * out['images'] and c['extra_quality_rejections'] == 0 and c['extra_guard_flags'] == 0]
    out['recommended'] = min(eligible)[2] if eligible else None
    out['rule'] = (f'smallest median size with E_768 flips <= {FLIP_LIMIT:.0%} of images and no extra quality rejection or guard flag '
                   'versus the stored file (written before scoring)')
    out['capacity_what_if'] = ('run for the recommended size only' if out['recommended'] else
                               'not run: the plan adds a capacity scenario only for a recommended size')
    out['post_hoc'] = describe_flips(table, ref, label, cut)
    out['scope'] = 'AI decision stability under recompression only; nothing about human grading of compressed images'
    return out


def describe_flips(table: pd.DataFrame, ref: pd.DataFrame, label: pd.Series, cut: float) -> dict:
    """Where the E_768 flips fall (direction, distance of the stored-file score from the cut-off, set) and the sensitivity/specificity
    on this balanced sample. Added after the rule result was seen: descriptive only, it does not change the result."""
    distance = (ref.e768_score - cut).abs()
    out = {'note': 'descriptive, added after the rule result was seen; sensitivity/specificity on the balanced 150/150 sample'}
    for name in table.condition.unique():
        rows = table.loc[table.condition == name].set_index('k').loc[ref.index]
        flip = rows.e768_refer != ref.e768_refer
        out[name] = {'referrals_lost': int((flip & ref.e768_refer).sum()), 'referrals_gained': int((flip & ~ref.e768_refer).sum()),
                     'flips_within_near_band': int((flip & (distance < NEAR_BAND)).sum()),
                     'max_stored_distance_from_cut': round(float(distance[flip].max()), 3) if flip.any() else None,
                     'flips_per_set': {s: int(n) for s, n in rows.loc[flip, 'source'].value_counts().items()},
                     'sensitivity': round(float((rows.e768_refer & label).sum() / label.sum()), 3),
                     'specificity': round(float((~rows.e768_refer & ~label).sum() / (~label).sum()), 3)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', type=Path, default=ROOT / 'results/model_improvement/x16_compression', help='local per-image table')
    parser.add_argument('--out', type=Path, required=True, help='aggregate JSON (no file names)')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    sample = draw_sample()
    table_path = args.work / 'scores.csv'
    table = pd.read_csv(table_path) if table_path.is_file() else score_all(sample)
    table.to_csv(table_path, index=False)
    from modules import variant_e768
    cut = float(variant_e768.E768Runtime().config['grade_thresholds'][1])
    summary = summarize(table, sample, cut)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: summary[k] for k in ('recommended', 'near_cut_images', 'per_set')}, indent=1))
    for name, c in summary['conditions'].items():
        print(f"{name:12s} {c['median_kb']:8.1f} KB  flips {c['e768_flips']:2d} (<= {c['e768_flip_upper95']:.3f})  legacy {c['legacy_flips']:2d}  "
              f"+rej {c['extra_quality_rejections']}  +flag {c['extra_guard_flags']}  AUC {c['e768_auc']:.3f}")


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
