#!/usr/bin/env python3
"""Per-site referral calibration for a NetrAI grading variant (default: E_768).

A frozen model's referral cut-off does not transfer between cameras and sites (leave-dataset-out study, mBRSET, BRSET). This tool fits the
site's own grade >= 2 boundary from a small labelled sample of that site and writes a policy file the MATLAB pipeline loads with
``main_pipeline(..., 'SitePolicy', file)``.

Inputs: a ``logits.npy`` (n x 6, rows in the order of the labels file; see ``score`` below to create it) and a labels CSV with a ``grade`` column
(ICDR/ALFEDIAM 0-4, ungradable images removed), an optional ``patient`` column and an optional ``image`` column. The boundary is the highest score cut-off at which the
sample sensitivity for grade >= 2 reaches ``--target-sensitivity`` (fewest false referrals), default 0.95 (rule R2). R2 was selected over the
earlier 0.92 target on exposed sites and confirmed on the fresh SUSTech-SYSU test (tier TF; 200-image samples: both screening targets met in 93%
of draws vs 67%, mean sensitivity 0.94 vs 0.91; results/benchmarks/round6_fresh_sustech_sysu). A sample's own sensitivity is optimistic, the
number of referable cases matters more than the total (40-80+ advised), and calibration cannot fix a camera on which the model ranks poorly: with
R2 sensitivity stays near 0.94 but specificity can fall below 0.85 (results/benchmarks/site_calibration_rules).

    python calibrate_site.py score --images images.csv --root IMAGES --out logits.npy     (the installed E_768 runtime; scripts/fetch_e768.sh)
    python calibrate_site.py fit --logits logits.npy --labels labels.csv --output site_policy.json
    python calibrate_site.py verify --policy site_policy.json --logits logits_B.npy --labels labels_B.csv --output check.json   (a SEPARATE sample)

``score`` runs the same ONNX graph and preprocessing as ``inference.py --variant E_768`` and writes ``logits.images.json`` next to the logits
(SHA-256 of every image, in row order, and the model identity). Every policy records the identities of its calibration sample in the format
MATLAB's calibrateSite/verifySitePolicy share: ``calibration_image_sha256`` (SHA-256 of each image file; required) and, when the labels have a
``patient`` column, ``calibration_patient_sha256`` (SHA-256 of each patient id's text; ids are never stored in clear). Image identities come from
``logits.images.json`` or, without it, from ``--images-root`` plus an ``image`` column. ``verify`` compares identities, not row order: it refuses
a sample that shares an image (or a recorded patient) with the calibration sample, a sample or policy whose identities are missing, and logits
made by another model.
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


def read_grades(labels: pd.DataFrame) -> np.ndarray:
    """ICDR grades 0-4 of every row (ungradable or unlabelled images must be removed first, not left blank)."""
    if 'grade' not in labels:
        raise ValueError('labels need a grade column (ICDR 0-4)')
    grade = pd.to_numeric(labels.grade, errors='coerce').to_numpy(float)
    if not np.isin(grade, [0, 1, 2, 3, 4]).all():
        raise ValueError('every label row needs a grade 0-4: remove ungradable or unlabelled images from the sample first')
    return grade.astype(int)


def sidecar_path(logits: Path) -> Path:
    """``logits.npy`` -> ``logits.images.json`` (written by ``score``)."""
    return Path(logits).with_suffix('.images.json')


def read_sidecar(logits: Path, rows: int, identity: dict) -> dict | None:
    """Image list of a logits file made by ``score`` (None when absent); refuses a wrong length or another model's logits."""
    path = sidecar_path(logits)
    if not path.is_file():
        return None
    meta = json.loads(path.read_text())
    if len(meta.get('sha256', [])) != rows:
        raise ValueError(f'{path.name} lists {len(meta.get("sha256", []))} images but the logits have {rows} rows')
    if meta.get('model_sha256') and meta['model_sha256'].lower() != identity['model_sha256'].lower():
        raise ValueError(f'{Path(logits).name} was produced by a different model than the one the policy belongs to')
    return meta


def text_sha256(value) -> str:
    """SHA-256 of an identifier's UTF-8 text (MATLAB textSHA256 gives the same digest)."""
    return hashlib.sha256(str(value).encode('utf-8')).hexdigest()


def read_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={'patient': str, 'image': str})     # '007' stays '007' (identities are hashed as text)


def sample_identities(logits: Path, labels: pd.DataFrame, rows: int, identity: dict, images_root: Path | None = None) -> dict:
    """Image identities (required) and hashed patient identities (None without a patient column) of a calibration or verification sample."""
    meta = read_sidecar(logits, rows, identity)
    if meta is not None:
        if 'image' in labels and list(labels.image) != list(meta.get('images', [])):
            raise ValueError('The image column of the labels does not match the scored images row by row')
        images = list(meta['sha256'])
    elif images_root is not None and 'image' in labels:
        images = [file_sha256(Path(images_root) / name) for name in labels.image]
    else:
        raise ValueError('The sample has no image identities: create the logits with "calibrate_site.py score" (it writes '
                         f'{sidecar_path(logits).name}) or give --images-root and an image column in the labels. Policies record their '
                         'calibration images so that a verification sample sharing any of them is refused.')
    patients = None
    if 'patient' in labels:
        if labels.patient.isna().any() or (labels.patient.str.strip() == '').any():
            raise ValueError('The patient column has blank values')
        patients = sorted({text_sha256(p) for p in labels.patient})
    return {'images': images, 'patients': patients}


def fit(args) -> dict:
    logits = np.load(args.logits)
    labels = read_labels(args.labels)
    if len(labels) != len(logits):
        raise ValueError(f'{len(logits)} logit rows but {len(labels)} label rows')
    grade = read_grades(labels)
    referable = grade >= 2
    n_pos, n_neg = int(referable.sum()), int((~referable).sum())
    if n_pos < MIN_REFERABLE_HARD or n_neg < MIN_REFERABLE_HARD:
        raise ValueError(f'Too few cases: {n_pos} referable / {n_neg} non-referable (need at least {MIN_REFERABLE_HARD} of each; {MIN_REFERABLE_ADVISED}+ referable advised)')
    warnings = []
    if n_pos < MIN_REFERABLE_ADVISED:
        warnings.append(f'Only {n_pos} referable cases: the fitted boundary is unstable (advised: at least {MIN_REFERABLE_ADVISED}).')
    if n_neg < MIN_NON_REFERABLE_ADVISED:
        warnings.append(f'Only {n_neg} non-referable cases: specificity cannot be judged (advised: at least {MIN_NON_REFERABLE_ADVISED}).')
    identity = model_identity(args.base_policy, getattr(args, 'bundle', None))
    ids = sample_identities(args.logits, labels, len(logits), identity, getattr(args, 'images_root', None))
    score = expected_grade(logits)
    cut = fit_boundary(score, referable, args.target_sensitivity)
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
    policy['calibration_image_sha256'] = sorted(set(ids['images']))       # shared with MATLAB calibrateSite / verifySitePolicy
    if ids['patients'] is not None:
        policy['calibration_patient_sha256'] = ids['patients']
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
    labels = read_labels(args.labels)
    if len(labels) != len(logits):
        raise ValueError(f'{len(logits)} logit rows but {len(labels)} label rows')
    grade = read_grades(labels)
    recorded = policy.get('calibration_image_sha256') or []
    if not recorded:
        raise ValueError('The policy does not record its calibration images, so overlap with a verification sample cannot be ruled out: '
                         're-create it with "calibrate_site.py score" + "fit" (or MATLAB calibrateSite)')
    ids = sample_identities(args.logits, labels, len(logits), identity, getattr(args, 'images_root', None))
    shared = set(ids['images']) & set(recorded)
    if shared:
        raise ValueError(f'{len(shared)} images were in the calibration sample: verification needs a separate sample')
    if policy.get('calibration_patient_sha256'):
        if ids['patients'] is None:
            raise ValueError('The calibration sample recorded patient identities: give the verification labels a patient column '
                             'so that patients in both samples can be refused')
        shared = set(ids['patients']) & set(policy['calibration_patient_sha256'])
        if shared:
            raise ValueError(f'{len(shared)} patients appear in both the calibration and the verification sample')
    if getattr(args, 'calibration_labels', None) and 'patient' in labels:
        overlap = set(read_labels(args.calibration_labels).patient) & set(labels.patient)
        if overlap:
            raise ValueError(f'{len(overlap)} patients appear in both the calibration and the verification sample')
    warnings = []
    called = expected_grade(logits) >= float(policy['thresholds'][1])
    summary = verification_summary(called, grade >= 2)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def score(args) -> None:
    """Logits of the installed E_768 runtime for a CSV with an ``image`` column, plus the image list ``fit``/``verify`` use.

    Same ONNX graph (hash-checked against the bundle), image reading and preprocessing as ``inference.py --variant E_768``, so the fitted
    cut-off applies to exactly the scores the screening pipeline computes.
    """
    import cv2
    from modules.variant_e768 import E768Runtime
    out = Path(args.out)
    if out.suffix != '.npy':
        raise ValueError('--out must end in .npy')
    frame = pd.read_csv(args.images)
    if 'image' not in frame:
        raise ValueError('the images CSV needs an image column (file names relative to --root)')
    runtime = E768Runtime()
    names, rows, digests = list(frame.image.astype(str)), [], []
    for i, name in enumerate(names):
        path = Path(args.root) / name
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f'Unreadable image: {path}')
        rows.append(runtime.predict([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)])['logits'][0])
        digests.append(file_sha256(path))
        if (i + 1) % 25 == 0:
            print(f'scored {i + 1}/{len(names)}', flush=True)
    np.save(out, np.asarray(rows, float))
    sidecar_path(out).write_text(json.dumps({'schema_version': 1, 'runtime': 'E_768 ONNX (modules/variant_e768.py)', **runtime.identity,
                                             'images': names, 'sha256': digests}, indent=1))
    print(f'Wrote {out} and {sidecar_path(out).name} ({len(names)} images)')


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
    f.add_argument('--images-root', type=Path, default=None, help='folder of the images named in an image column (when the logits have no image list)')
    v = sub.add_parser('verify', help='check a site policy on a separate labelled sample of the same site')
    v.add_argument('--policy', type=Path, required=True)
    v.add_argument('--logits', type=Path, required=True)
    v.add_argument('--labels', type=Path, required=True)
    v.add_argument('--output', type=Path, required=True)
    v.add_argument('--calibration-labels', type=Path, default=None, help='labels CSV of the calibration sample (extra check on raw patient ids)')
    v.add_argument('--images-root', type=Path, default=None, help='folder of the images named in an image column (when the logits have no image list)')
    v.add_argument('--base-policy', type=Path, default=Path(__file__).resolve().parents[1] / 'matlab/variant_R3_E_768/netrai_config.json')
    v.add_argument('--bundle', type=Path, default=None)
    s = sub.add_parser('score')
    s.add_argument('--images', type=Path, required=True)
    s.add_argument('--root', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True, help='logits file (.npy); the image list is written next to it')
    args = parser.parse_args(argv)
    {'fit': fit, 'verify': verify, 'score': score}[args.command](args)


if __name__ == '__main__':
    main()
