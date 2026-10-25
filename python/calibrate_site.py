#!/usr/bin/env python3
"""Per-site referral calibration for a NetrAI grading variant (default: E_768).

A frozen model's referral cut-off does not transfer between cameras and sites (leave-dataset-out study, mBRSET, BRSET). This tool fits the
site's own grade >= 2 boundary from a small labelled sample of that site and writes a policy file the MATLAB pipeline loads with
``main_pipeline(..., 'SitePolicy', file)``.

Inputs: a ``logits.npy`` (n x 6, rows in the order of the labels file; see ``score`` below to create it) and a labels CSV with a ``grade`` column
(ICDR/ALFEDIAM 0-4, ungradable images removed) and an optional ``patient`` column. The boundary is the highest score cut-off at which the
sample sensitivity for grade >= 2 reaches ``--target-sensitivity`` (fewest false referrals). It is NOT validated: the sample sensitivity is
optimistic, and the number of referable patients matters more than the total (post hoc evidence: results/benchmarks/site_calibration_brazil).

    python calibrate_site.py fit --logits logits.npy --labels labels.csv --base-policy POLICY.json --output site_policy.json
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


def monotone_thresholds(base: list[float], cut: float) -> list[float]:
    """Four ordinal boundaries whose grade >= 2 decision is exactly ``score >= cut`` (t0 <= cut = t1 <= t2 <= t3)."""
    t = list(base)
    return [min(t[0], cut), cut, max(t[2], cut), max(t[3], cut)]


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
    f.add_argument('--target-sensitivity', type=float, default=0.92)
    s = sub.add_parser('score')
    s.add_argument('--images', type=Path, required=True)
    s.add_argument('--root', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    {'fit': fit, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
