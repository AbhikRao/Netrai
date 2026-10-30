#!/usr/bin/env python3
"""Per-site referral calibration for a NetrAI grading variant (default: E_768).

A frozen model's referral cut-off does not transfer between cameras and sites (leave-dataset-out study, mBRSET, BRSET). This tool fits the
site's own grade >= 2 boundary from a small labelled sample of that site and writes a policy file the MATLAB pipeline loads with
``main_pipeline(..., 'SitePolicy', file)``.

Inputs: a ``logits.npy`` (n x 6, rows in the order of the labels file; see ``score`` below to create it) and a labels CSV with a ``grade`` column
(ICDR/ALFEDIAM 0-4, ungradable images removed) and an optional ``patient`` column. The boundary is the highest score cut-off at which the
sample sensitivity for grade >= 2 reaches ``--target-sensitivity`` (fewest false referrals), default 0.95 (rule R2). R2 was selected over the
earlier 0.92 target on exposed sites and confirmed on the fresh SUSTech-SYSU test (tier TF; 200-image samples: both screening targets met in 93%
of draws vs 67%, mean sensitivity 0.94 vs 0.91; results/benchmarks/round6_fresh_sustech_sysu). A sample's own sensitivity is optimistic, the
number of referable cases matters more than the total (40-80+ advised), and calibration cannot fix a camera on which the model ranks poorly: with
R2 sensitivity stays near 0.94 but specificity can fall below 0.85 (results/benchmarks/site_calibration_rules).

    python calibrate_site.py fit --logits logits.npy --labels labels.csv --base-policy POLICY.json --output site_policy.json
    python calibrate_site.py verify --policy site_policy.json --logits logits_B.npy --labels labels_B.csv --output check.json   (a SEPARATE sample)
    python calibrate_site.py score --images images.csv --root IMAGES --out logits.npy     (E_768 members; needs the local research checkpoints)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MIN_REFERABLE_HARD, MIN_REFERABLE_ADVISED, MIN_NON_REFERABLE_ADVISED = 5, 40, 150
E768_MEMBERS = {'R3_768_s2026': 'results/model_improvement/kaggle_output_768_s2026/run/best.pt',
                'R3_768_s17': 'results/model_improvement/kaggle_output_768_s17/run/best.pt'}


def expected_grade(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, float)[:, :5]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return p @ np.arange(5)


def fit_boundary(score: np.ndarray, referable: np.ndarray, target: float) -> float:
    """Highest cut-off with sample sensitivity >= target (call everything >= the (k+1)-th lowest referable score)."""
    pos = np.sort(score[referable])
    allowed_misses = int(np.floor((1 - target) * len(pos)))
    return float(pos[allowed_misses]) - 1e-9


THRESHOLD_GAP = 1e-6


def monotone_thresholds(base: list[float], cut: float) -> list[float]:
    """Four strictly increasing ordinal boundaries whose grade >= 2 decision is exactly ``score >= cut``.

    Neighbouring boundaries are moved just past the cut when needed (MATLAB's grading code requires strictly increasing boundaries).
    """
    t0 = min(float(base[0]), cut - THRESHOLD_GAP)
    t2 = max(float(base[2]), cut + THRESHOLD_GAP)
    t3 = max(float(base[3]), t2 + THRESHOLD_GAP)
    return [t0, float(cut), t2, t3]


def model_identity(base_policy: Path, bundle: Path | None) -> dict:
    """SHA-256 of the model and config the policy is fitted for, read from the variant's deployment bundle."""
    bundle = Path(bundle) if bundle else Path(base_policy).with_name('netrai_bundle.json')
    if not bundle.is_file():
        raise FileNotFoundError(f'Deployment bundle not found: {bundle} (pass --bundle: a site policy must name the model it belongs to)')
    roles = {a['role']: a['sha256'] for a in json.loads(bundle.read_text())['artifacts']}
    if 'model' not in roles or 'config' not in roles:
        raise ValueError('The bundle lists no model/config artifact')
    return {'model_sha256': roles['model'], 'config_sha256': roles['config']}


def fit(args) -> dict:
    logits = np.load(args.logits)
    labels = pd.read_csv(args.labels)
    if len(labels) != len(logits):
        raise ValueError(f'{len(logits)} logit rows but {len(labels)} label rows')
    if 'grade' not in labels or not set(labels.grade.dropna().astype(int)) <= {0, 1, 2, 3, 4}:
        raise ValueError('labels need a grade column with values 0-4')
    grade = labels.grade.astype(int).to_numpy()
    referable = grade >= 2
    n_pos, n_neg = int(referable.sum()), int((~referable).sum())
    if n_pos < MIN_REFERABLE_HARD or n_neg < MIN_REFERABLE_HARD:
        raise ValueError(f'Too few cases: {n_pos} referable / {n_neg} non-referable (need at least {MIN_REFERABLE_HARD} of each; {MIN_REFERABLE_ADVISED}+ referable advised)')
    warnings = []
    if n_pos < MIN_REFERABLE_ADVISED:
        warnings.append(f'Only {n_pos} referable cases: the fitted boundary is unstable (advised: at least {MIN_REFERABLE_ADVISED}).')
    if n_neg < MIN_NON_REFERABLE_ADVISED:
        warnings.append(f'Only {n_neg} non-referable cases: specificity cannot be judged (advised: at least {MIN_NON_REFERABLE_ADVISED}).')
    score = expected_grade(logits)
    cut = fit_boundary(score, referable, args.target_sensitivity)
    identity = model_identity(args.base_policy, getattr(args, 'bundle', None))
    base = json.loads(Path(args.base_policy).read_text())
    base_thresholds = base.get('thresholds', base.get('grade_thresholds'))
    if base_thresholds is None or len(base_thresholds) != 4:
        raise ValueError('The base policy needs four ordinal boundaries (thresholds or grade_thresholds)')
    called = score >= cut
    sample = {'sensitivity': float(called[referable].mean()), 'specificity': float((~called[~referable]).mean())}
    old = float(base_thresholds[1])
    old_called = score >= old
    policy = {**{k: v for k, v in base.items() if k not in ('thresholds', 'grade_thresholds')}, 'thresholds': monotone_thresholds(base_thresholds, cut),
              'status': 'site_calibrated', 'model_variant': args.variant, **identity, 'site': {
                  'name': args.site, 'created_utc': datetime.now(timezone.utc).isoformat(), 'target_sample_sensitivity': args.target_sensitivity,
                  'images': int(len(labels)), 'patients': int(labels.patient.nunique()) if 'patient' in labels else None,
                  'referable': n_pos, 'non_referable': n_neg, 'fitted_boundary': cut, 'base_boundary': old,
                  'sample_operating_point_fitted': sample,
                  'sample_operating_point_base': {'sensitivity': float(old_called[referable].mean()), 'specificity': float((~old_called[~referable]).mean())},
                  'logits_sha256': hashlib.sha256(Path(args.logits).read_bytes()).hexdigest(),
                  'warnings': warnings,
                  'validation': 'NONE: sample numbers are optimistic (fitted on the same cases). Re-check on a separate labelled sample before screening.'}}
    Path(args.output).write_text(json.dumps(policy, indent=2))
    for w in warnings:
        print('WARNING:', w)
    print(f"Site policy {args.output}: boundary {old:.3f} -> {cut:.3f}; sample sensitivity {sample['sensitivity']:.3f}, specificity {sample['specificity']:.3f} "
          f"({n_pos} referable / {n_neg} non-referable)")
    return policy


def wilson(successes: int, total: int, z: float = 1.959964) -> list[float]:
    """Wilson score 95% interval for a proportion."""
    if total == 0:
        return [float('nan'), float('nan')]
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(centre - half), float(centre + half)]


def verification_summary(called: np.ndarray, referable: np.ndarray, target_sensitivity: float = 0.90, target_specificity: float = 0.85) -> dict:
    """Sensitivity/specificity of a site policy on a separate labelled sample, with Wilson 95% intervals and pass flags."""
    called, referable = np.asarray(called, bool), np.asarray(referable, bool)
    tp, fn = int((called & referable).sum()), int((~called & referable).sum())
    tn, fp = int((~called & ~referable).sum()), int((called & ~referable).sum())
    sens = tp / (tp + fn) if tp + fn else float('nan')
    spec = tn / (tn + fp) if tn + fp else float('nan')
    sens_ci, spec_ci = wilson(tp, tp + fn), wilson(tn, tn + fp)
    return {'referable': tp + fn, 'non_referable': tn + fp, 'tp': tp, 'fn': fn, 'tn': tn, 'fp': fp,
            'sensitivity': sens, 'sensitivity_ci95': sens_ci, 'specificity': spec, 'specificity_ci95': spec_ci,
            'meets_targets_point': bool(sens >= target_sensitivity and spec >= target_specificity),
            'meets_targets_lower_bound': bool(sens_ci[0] >= target_sensitivity and spec_ci[0] >= target_specificity),
            'targets': {'sensitivity': target_sensitivity, 'specificity': target_specificity}}


def verify(args) -> dict:
    """Check a site policy on a SEPARATE labelled sample of the same site (never the calibration sample)."""
    policy = json.loads(Path(args.policy).read_text())
    if policy.get('status') != 'site_calibrated':
        raise ValueError('Not a site-calibrated policy')
    identity = model_identity(args.base_policy, getattr(args, 'bundle', None))
    if policy.get('model_sha256') != identity['model_sha256'] or policy.get('config_sha256') != identity['config_sha256']:
        raise ValueError('The policy was fitted for a different model than the one whose bundle was given')
    digest = hashlib.sha256(Path(args.logits).read_bytes()).hexdigest()
    if digest == policy.get('site', {}).get('logits_sha256'):
        raise ValueError('These are the calibration logits: verification needs a separate labelled sample')
    logits = np.load(args.logits)
    labels = pd.read_csv(args.labels)
    if len(labels) != len(logits):
        raise ValueError(f'{len(logits)} logit rows but {len(labels)} label rows')
    grade = labels.grade.astype(int).to_numpy()
    if not set(grade) <= {0, 1, 2, 3, 4}:
        raise ValueError('labels need grades 0-4')
    if getattr(args, 'calibration_labels', None) and 'patient' in labels:
        overlap = set(pd.read_csv(args.calibration_labels).patient) & set(labels.patient)
        if overlap:
            raise ValueError(f'{len(overlap)} patients appear in both the calibration and the verification sample')
    called = expected_grade(logits) >= float(policy['thresholds'][1])
    summary = verification_summary(called, grade >= 2)
    warnings = []
    if summary['referable'] < 40:
        warnings.append(f"Only {summary['referable']} referable cases: the sensitivity interval is wide (advised: at least 40).")
    result = {'policy_sha256': hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(), 'verification_logits_sha256': digest,
              'site': policy.get('site', {}).get('name'), 'checked_utc': datetime.now(timezone.utc).isoformat(), **summary, 'warnings': warnings}
    Path(args.output).write_text(json.dumps(result, indent=2))
    for w in warnings:
        print('WARNING:', w)
    print(f"Verification: sensitivity {summary['sensitivity']:.3f} ({summary['sensitivity_ci95'][0]:.3f}-{summary['sensitivity_ci95'][1]:.3f}), "
          f"specificity {summary['specificity']:.3f} ({summary['specificity_ci95'][0]:.3f}-{summary['specificity_ci95'][1]:.3f}); "
          f"targets met: point {summary['meets_targets_point']}, lower bound {summary['meets_targets_lower_bound']}")
    return result


def score(args) -> None:
    """E_768 logits (mean of its two members) for a CSV with an ``image`` column. Needs the local research checkpoints."""
    import torch
    from locked_evaluation import ROOT
    from locked_rfmid import logits_on
    frame = pd.read_csv(args.images).rename(columns={'image': 'image_id'})
    folder = Path(args.root)
    device = torch.device(args.device)
    outputs = []
    for name, checkpoint in E768_MEMBERS.items():
        path = ROOT / checkpoint
        if not path.is_file():
            raise FileNotFoundError(f'{path} not found: the E_768 research checkpoints are local-only artifacts')
        outputs.append(logits_on({'id': name, 'kind': 'full', 'size': 768}, frame[['image_id']], folder, device, path))
    np.save(args.out, np.mean(outputs, axis=0))
    print(f'Wrote {args.out} ({len(frame)} images)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--logits', type=Path, required=True)
    f.add_argument('--labels', type=Path, required=True)
    f.add_argument('--base-policy', type=Path, default=Path(__file__).resolve().parents[1] / 'matlab/variant_R3_E_768/netrai_config.json')
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--bundle', type=Path, default=None, help='netrai_bundle.json of the model (default: next to the base config)')
    f.add_argument('--variant', default='E_768', help='model variant name recorded in the policy')
    f.add_argument('--site', default='unnamed site')
    f.add_argument('--target-sensitivity', type=float, default=0.95, help='sample sensitivity to reach (0.95 = rule R2, confirmed on a fresh test)')
    v = sub.add_parser('verify', help='check a site policy on a separate labelled sample of the same site')
    v.add_argument('--policy', type=Path, required=True)
    v.add_argument('--logits', type=Path, required=True)
    v.add_argument('--labels', type=Path, required=True)
    v.add_argument('--output', type=Path, required=True)
    v.add_argument('--calibration-labels', type=Path, default=None, help='labels CSV of the calibration sample (patient overlap check)')
    v.add_argument('--base-policy', type=Path, default=Path(__file__).resolve().parents[1] / 'matlab/variant_R3_E_768/netrai_config.json')
    v.add_argument('--bundle', type=Path, default=None)
    s = sub.add_parser('score')
    s.add_argument('--images', type=Path, required=True)
    s.add_argument('--root', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    {'fit': fit, 'verify': verify, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
