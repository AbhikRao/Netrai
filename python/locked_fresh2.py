#!/usr/bin/env python3
"""Locked, single-use evaluations on the round-6 fresh sets: Paraguay (Zeiss Visucam) and SUSTech-SYSU (Topcon), plan v6.

Same two-stage pattern as ``locked_mbrset.py``. Images were extracted blind (``blind_extract.py``: neutral content-hash names); the labels sit
in the sealed folder registered as reserved, and only ``evaluate`` opens it, once, after the lock.

``logits``   label-free: logits of the listed variants on every image (resumable chunks).
``freeze``   binds inventory (identity exclusions flagged, near-copy clusters), logits (CPU spot check), optional AdaBN logits and adaptation
             list, models/policies/code and the sealed label files' hashes into ``lock.json`` (``labels_read: false``); registry -> locked.
``evaluate`` verifies the bindings, reads the sealed labels once and reports (plan v6 section 3):
  1. E_768 at its frozen policy vs A_legacy: referable (ICDR >= 2) sensitivity/specificity, AUC, sensitivity at 85% specificity;
  2. the labelled site-calibration procedure (``calibrate_site.fit_boundary``, target sample sensitivity 0.92) on random calibration samples of
     100 and 200 images scored on the remaining images;
  3. AdaBN-adapted E_768 at the frozen cut-off, paired with frozen E_768 (only when its logits were locked).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import exposure
import locked_mbrset as lm
from calibrate_site import fit_boundary
from locked_evaluation import ROOT, logits_for, sha, thresholds_for
from locked_rfmid import logits_on
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m
from utils.artifacts import atomic_json

PROTOCOL = ROOT / '.project-notes/docs/MODEL_IMPROVEMENT_PLAN_v6.md'
IDENTITY = ROOT / 'results/model_improvement/fresh2_identity'
COHORTS = {'paraguay_hc': 'paraguay_dr', 'sustech_sysu': 'sustech_sysu'}
PARAGUAY_FOLDERS = {'1. No DR signs': 0, '2. Mild (or early) NPDR': 1, '3. Moderate NPDR': 2, '4. Severe NPDR': 3,
                    '5. Very Severe NPDR': 3, '6. PDR': 4, '7. Advanced PDR': 4}
SUSTECH_ICDR = 'DR_grade(International_Clinical_DR_Severity_Scale)'
CODE = ['python/locked_fresh2.py', 'python/locked_mbrset.py', 'python/locked_evaluation.py', 'python/locked_rfmid.py', 'python/exposure.py',
        'python/calibrate_site.py', 'python/adabn.py', 'python/blind_extract.py', 'python/utils/preprocessing.py', 'python/models/netrai_model.py',
        'python/export_adapted_variant.py', 'python/netrai_eval/metrics.py', 'python/netrai_eval/bootstrap.py']
SPOT_CHECK_IMAGES, SPOT_CHECK_TOLERANCE, CHUNK = 8, 5e-3, 256
CALIBRATION_SIZES, CALIBRATION_DRAWS, CALIBRATION_TARGET, MIN_CASES = (100, 200), 500, 0.92, 5
TARGET_SENSITIVITY, TARGET_SPECIFICITY = 0.90, 0.85


def data_dir(cohort: str) -> Path:
    return ROOT / 'data' / COHORTS[cohort]


def identity_dir(cohort: str) -> Path:
    return IDENTITY / COHORTS[cohort]


def inventory(cohort: str, exclusions: Path | None = None, clusters: Path | None = None, images: Path | None = None) -> pd.DataFrame:
    """Every image (neutral name), cluster (image unless merged as a near-copy) and identity-excluded flag. No label is opened."""
    folder = images or data_dir(cohort) / 'images'
    files = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif'))
    frame = pd.DataFrame({'image_id': files})
    frame['cluster'] = frame.image_id
    if clusters is not None and Path(clusters).is_file():
        table = pd.read_csv(clusters)
        table['file'] = table.image.astype(str).str.replace(r'.*/', '', regex=True)
        mapping = dict(zip(table.file, table.cluster.astype(str)))
        frame['cluster'] = [f'c{mapping[f]}' if f in mapping else f for f in files]
    excluded = set()
    if exclusions is not None and Path(exclusions).is_file():
        excluded = {line.strip() for line in Path(exclusions).read_text().splitlines() if line.strip() and not line.startswith('#')}
    unknown = excluded - set(files)
    if unknown:
        raise ValueError(f'Exclusions name unknown images: {sorted(unknown)[:5]}')
    frame['excluded'] = frame.image_id.isin(excluded)
    return frame


def sealed_files(cohort: str) -> list[Path]:
    """The label-bearing files (hash-bound at freeze, opened only at evaluate)."""
    sealed = data_dir(cohort) / 'sealed'
    if cohort == 'paraguay_hc':
        return [sealed / 'mapping.csv', sealed / 'Annotations of the classifications.xlsx']
    return [sealed / 'mapping.csv', sealed / 'originalImages/drLabels.csv', sealed / 'originalImages/c5_DR_reclassified.csv']


def read_labels(cohort: str, frame: pd.DataFrame, sealed: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Labels for the inventory rows. Callers must have passed ``assert_label_read_allowed``. Returns (frame + grade/primary/secondary, notes)."""
    sealed = sealed or data_dir(cohort) / 'sealed'
    mapping = pd.read_csv(sealed / 'mapping.csv', dtype=str)
    out, notes = frame.copy(), {}
    if cohort == 'paraguay_hc':
        mapping['folder'] = mapping.archive_path.str.split('/').str[0]
        unknown = sorted(set(mapping.folder) - set(PARAGUAY_FOLDERS))
        if unknown:
            raise ValueError(f'Unexpected Paraguay folders: {unknown}')
        mapping['grade'] = mapping.folder.map(PARAGUAY_FOLDERS)
        grades = mapping.groupby('image_id').grade.agg(lambda g: sorted(set(g)))
        conflict = {i for i, g in grades.items() if len(g) > 1}
        notes['duplicate_copies_same_category'] = int(sum(len(v) - 1 for i, v in mapping.groupby('image_id').grade if i not in conflict))
        notes['images_with_conflicting_categories'] = sorted(conflict)
        out['grade'] = [grades[i][0] if i in grades.index and i not in conflict else -1 for i in out.image_id]
        out['primary'] = (out.grade >= 0) & ~out.excluded
        out['secondary'] = False
        try:  # sheet check: Status vs folder, reported only
            import openpyxl
            sheet = openpyxl.load_workbook(sealed / 'Annotations of the classifications.xlsx', read_only=True).worksheets[0]
            rows = list(sheet.iter_rows(values_only=True))[1:]
            by_name = {str(r[0]).strip(): str(r[2]).strip() for r in rows if r and r[0] is not None}
            folder_of = {Path(p).name: Path(p).parent.name for p in mapping.archive_path}
            pairs = [(folder_of[n], by_name.get(n, by_name.get(Path(n).stem))) for n in folder_of]
            notes['sheet_status_rows_matched'] = int(sum(b is not None for _, b in pairs))
            notes['sheet_status_equal_to_folder_text'] = int(sum(b is not None and b.lower() in a.lower() for a, b in pairs))
        except Exception as error:  # the sheet is a cross-check only
            notes['sheet_check_error'] = str(error)
    else:
        mapping['file'] = mapping.archive_path.str.replace(r'.*/', '', regex=True)
        table = pd.read_csv(sealed / 'originalImages/drLabels.csv', dtype=str, encoding='utf-8-sig')
        c5 = pd.read_csv(sealed / 'originalImages/c5_DR_reclassified.csv', dtype=str, encoding='utf-8-sig')

        def key(name: str) -> str:
            return Path(str(name).strip()).stem

        grade_of = {key(n): int(g) for n, g in zip(table.Fundus_images, table[SUSTECH_ICDR])}
        c5_of = {key(n): int(g) for n, g in zip(c5.Fundus_images, c5[SUSTECH_ICDR])}
        if not set(grade_of.values()) <= {0, 1, 2, 3, 4, 5} or not set(c5_of.values()) <= {2, 3, 4}:
            raise ValueError('Unexpected SUSTech-SYSU grade values')
        neutral = dict(zip(mapping.image_id, mapping.file))
        raw = [grade_of.get(key(neutral[i]), -1) for i in out.image_id]
        out['grade'] = [c5_of.get(key(neutral[i]), -1) if g == 5 else g for g, i in zip(raw, out.image_id)]
        out['category5'] = [g == 5 for g in raw]
        notes['images_without_label_row'] = int(sum(g == -1 for g in raw))
        notes['category5_images'] = int(sum(out.category5))
        notes['category5_without_reclassification'] = int(sum(c and g < 0 for c, g in zip(out.category5, out.grade)))
        out['primary'] = (out.grade >= 0) & ~out.category5 & ~out.excluded
        out['secondary'] = out.category5 & (out.grade >= 0) & ~out.excluded
    return out, notes


def exclude_conflicting_near_copies(labelled: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Both images of a pair (columns a, b) leave the primary/secondary populations when their grades differ."""
    grade = dict(zip(labelled.image_id, labelled.grade))
    conflicted = sorted({x for a, b in zip(pairs.a, pairs.b) if a in grade and b in grade and grade[a] >= 0 and grade[b] >= 0
                         and grade[a] != grade[b] for x in (a, b)})
    out = labelled.copy()
    hit = out.image_id.isin(conflicted)
    out.loc[hit, 'primary'] = False
    if 'secondary' in out:
        out.loc[hit, 'secondary'] = False
    return out, conflicted


def calibration_draws(score: np.ndarray, referable: np.ndarray, clusters: np.ndarray, frozen: float, size: int, draws: int, seed: int) -> dict:
    """Plan v6 endpoint 2: fit the referral boundary on a random calibration sample (whole clusters), score the remaining images."""
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in unique}
    results = []
    for _ in range(draws):
        order = rng.permutation(unique)
        take, count = [], 0
        for c in order:
            if count >= size:
                break
            take.extend(members[c]); count += len(members[c])
        take = np.array(take)
        rest = np.setdiff1d(np.arange(len(score)), take)
        sample_pos = referable[take]
        if sample_pos.sum() >= MIN_CASES and (~sample_pos).sum() >= MIN_CASES:
            cut = fit_boundary(score[take], sample_pos, CALIBRATION_TARGET)
        else:
            cut = frozen
        called = score[rest] >= cut
        sens, spec = float(called[referable[rest]].mean()), float((~called[~referable[rest]]).mean())
        results.append((sens, spec, sens >= TARGET_SENSITIVITY and spec >= TARGET_SPECIFICITY))
    r = np.array(results, float)
    frozen_called = score >= frozen
    return {'calibration_images': size, 'draws': draws, 'p_both_targets': float(r[:, 2].mean()),
            'sensitivity_mean': float(r[:, 0].mean()), 'sensitivity_p10': float(np.percentile(r[:, 0], 10)),
            'specificity_mean': float(r[:, 1].mean()), 'specificity_p10': float(np.percentile(r[:, 1], 10)),
            'frozen_sensitivity': float(frozen_called[referable].mean()), 'frozen_specificity': float((~frozen_called[~referable]).mean())}


def logits_command(args) -> None:
    torch.set_num_threads(args.threads)
    config = json.loads(args.variants.read_text())
    frame = inventory(args.cohort)
    folder = data_dir(args.cohort) / 'images'
    args.out.mkdir(parents=True, exist_ok=True)
    frame[['image_id']].to_csv(args.out / 'logits_order.csv', index=False)
    for spec in config['variants']:
        if spec['id'] not in args.ids or spec['kind'] in ('ensemble', 'policy_of'):
            continue
        target = args.out / f"logits_{spec['id']}.npy"
        if target.exists():
            continue
        parts = args.out / f"parts_{spec['id']}"
        parts.mkdir(exist_ok=True)
        for start in range(0, len(frame), CHUNK):
            part = parts / f'{start:06d}.npy'
            if not part.exists():
                np.save(part, logits_on(spec, frame.iloc[start:start + CHUNK].reset_index(drop=True), folder, torch.device(args.device)))
                print(f"{args.cohort} {spec['id']}: {min(start + CHUNK, len(frame))}/{len(frame)}", flush=True)
        np.save(target, np.concatenate([np.load(parts / f'{s:06d}.npy') for s in range(0, len(frame), CHUNK)]))
        shutil.rmtree(parts)


def freeze(args) -> None:
    config = json.loads(args.variants.read_text())
    out = args.output
    lock_path = out / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    exposure.assert_fresh_allowed(args.cohort)
    out.mkdir(parents=True, exist_ok=True)
    folder = args.images or data_dir(args.cohort) / 'images'
    frame = inventory(args.cohort, args.exclusions, args.clusters, args.images)
    frame['sha256'] = [sha(folder / i) for i in frame.image_id]
    inv = out / 'inventory.csv'
    frame.to_csv(inv, index=False)
    order = pd.read_csv(args.precomputed / 'logits_order.csv').image_id.tolist()
    if order != frame.image_id.tolist():
        raise ValueError('precomputed logits: row order differs from the inventory')
    cache, spot = {}, {}
    for spec in config['variants']:
        if spec['kind'] in ('ensemble', 'policy_of'):
            continue
        saved = np.load(args.precomputed / f"logits_{spec['id']}.npy")
        if saved.shape != (len(frame), 6) or not np.isfinite(saved).all():
            raise ValueError(f"logits_{spec['id']}: shape {saved.shape} or non-finite values")
        rows = np.unique(np.linspace(0, len(frame) - 1, SPOT_CHECK_IMAGES).round().astype(int))
        error = float(np.abs(logits_for(spec, frame.iloc[rows].reset_index(drop=True), folder=folder) - saved[rows]).max())
        if error > SPOT_CHECK_TOLERANCE:
            raise ValueError(f"logits_{spec['id']}: CPU spot check differs by {error}")
        spot[spec['id']] = error
        path = out / f"logits_{spec['id']}.npy"
        shutil.copy2(args.precomputed / path.name, path)
        cache[spec['id']] = path
    for spec in config['variants']:
        if spec['kind'] == 'ensemble':
            path = out / f"logits_{spec['id']}.npy"
            np.save(path, np.mean([np.load(cache[mid]) for mid in spec['members']], axis=0))
            cache[spec['id']] = path
    adabn = []
    if args.adabn_logits:
        saved = np.load(args.adabn_logits)
        if saved.shape != (len(frame), 6) or not np.isfinite(saved).all():
            raise ValueError('AdaBN logits: wrong shape or non-finite')
        shutil.copy2(args.adabn_logits, out / 'logits_E_768_adabn.npy')
        shutil.copy2(args.adabn_list, out / 'adabn_adaptation_images.txt')
        adabn = [out / 'logits_E_768_adabn.npy', out / 'adabn_adaptation_images.txt']
    near = []
    if getattr(args, 'near_identical', None):
        pairs = pd.read_csv(args.near_identical)
        pairs = pairs[pairs['corr'] >= 0.95][['a', 'b', 'corr']]
        pairs.to_csv(out / 'near_identical_pairs.csv', index=False)
        near = [out / 'near_identical_pairs.csv']
    files = [args.variants, inv, args.protocol, *sealed_files(args.cohort), *[ROOT / c for c in CODE], *cache.values(), *adabn, *near]
    files += [p for p in (args.clusters, args.exclusions) if p is not None and Path(p).is_file()]
    files += [ROOT / s['checkpoint'] for s in config['variants'] if s.get('checkpoint')]
    files += [ROOT / s['policy'] for s in config['variants'] if s.get('policy')]
    bindings = {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in files}  # sealed label files: hash only
    atomic_json(lock_path, {
        'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
        'tests': {args.cohort: f'{len(frame)} images, {int(frame.excluded.sum())} identity-excluded; referable DR (ICDR >= 2)'},
        'fresh': True, 'labels_read': False, 'primary_variant': config['primary'], 'reference_variant': config['reference'],
        'bootstrap_seed': config['bootstrap_seed'], 'bootstrap_replicates': config['bootstrap_replicates'],
        'calibration': {'sizes': list(CALIBRATION_SIZES), 'draws': CALIBRATION_DRAWS, 'target_sample_sensitivity': CALIBRATION_TARGET},
        'adabn_endpoint': bool(adabn), 'precomputed_logits_cpu_spot_check_max_abs': spot, 'bindings': bindings})
    exposure.record_locked(args.cohort, lock_path)
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def evaluate(args) -> None:
    out = args.output
    lock = json.loads((out / 'lock.json').read_text())
    exposure.assert_label_read_allowed(args.cohort, out)
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    config = json.loads(args.variants.read_text())
    frame = pd.read_csv(out / 'inventory.csv', dtype={'cluster': str})
    labelled, notes = read_labels(args.cohort, frame)
    near = out / 'near_identical_pairs.csv'
    if near.is_file():   # addendum 6.2: practically identical images (vessel >= 0.95) with different labels are excluded
        labelled, notes['near_identical_conflicts'] = exclude_conflicting_near_copies(labelled, pd.read_csv(near))
    primary = labelled.primary.to_numpy(bool)
    d = labelled[primary].reset_index(drop=True)
    grade, clusters = d.grade.to_numpy(int), d.cluster.to_numpy()
    seed, reps = lock['bootstrap_seed'], lock['bootstrap_replicates']
    specs = {s['id']: s for s in config['variants']}
    logits = {name: np.load(out / f"logits_{s.get('logits_from', name)}.npy")[primary] for name, s in specs.items()}
    thresholds = {name: thresholds_for(s)[0] for name, s in specs.items()}
    if lock.get('adabn_endpoint'):
        logits['E_768_adabn'] = np.load(out / 'logits_E_768_adabn.npy')[primary]
        thresholds['E_768_adabn'] = thresholds[lock['primary_variant']]
    stat = lambda name: (lambda rows: lm.referable_stats(grade[rows], logits[name][rows], thresholds[name]))
    results = {name: bs.bootstrap(stat(name), clusters, reps, seed) for name in logits}
    summaries = {name: lm.grade_summary(grade, logits[name], thresholds[name]) for name in logits}
    primary_name, reference = lock['primary_variant'], lock['reference_variant']
    comparisons = {f'{primary_name}_vs_{reference}': bs.paired_bootstrap(stat(primary_name), stat(reference), clusters, reps, seed)}
    if 'E_768_adabn' in logits:
        comparisons[f'E_768_adabn_vs_{primary_name}'] = bs.paired_bootstrap(stat('E_768_adabn'), stat(primary_name), clusters, reps, seed)
    score = m.expected_grade_score(logits[primary_name])
    frozen = float(thresholds[primary_name][1])
    calibration = {str(n): calibration_draws(score, grade >= 2, clusters, frozen, n, CALIBRATION_DRAWS, seed + n) for n in CALIBRATION_SIZES}
    secondary = {}
    if labelled.secondary.any():
        sec = labelled.secondary.to_numpy(bool)
        for name in (primary_name, reference):
            all_logits = np.load(out / f"logits_{specs[name].get('logits_from', name)}.npy")
            called = m.decode_thresholds(m.expected_grade_score(all_logits[sec]), thresholds[name]) >= 2
            secondary[name] = {'images': int(sec.sum()), 'called_referable': float(called.mean())}
    targets = {name: bool(results[name]['sensitivity']['estimate'] > TARGET_SENSITIVITY and results[name]['specificity']['estimate'] > TARGET_SPECIFICITY)
               for name in results}
    report = {'schema_version': 1, 'tier': 'TF', 'cohort': args.cohort, 'lock_sha256': sha(out / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'primary_variant': primary_name, 'reference_variant': reference,
              'population': {'images_in_lock': int(len(labelled)), 'identity_excluded': int(labelled.excluded.sum()),
                             'primary_images': int(len(d)), 'referable': int((grade >= 2).sum()), 'clusters': int(len(np.unique(clusters))),
                             'grade_support': np.bincount(grade, minlength=5).tolist(), 'label_notes': notes},
              'results': results, 'decision_summaries': summaries, 'comparisons': comparisons, 'meets_90_85_at_frozen_policy': targets,
              'site_calibration_procedure': calibration, 'secondary_population': secondary}
    atomic_json(out / 'report.json', report)
    exposure.record_exposure(args.cohort, out / 'lock.json', out / 'report.json')
    for name, block in results.items():
        print(args.cohort, name, {k: round(v['estimate'], 4) for k, v in block.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'evaluate'):
        s = sub.add_parser(name)
        s.add_argument('--cohort', choices=sorted(COHORTS), required=True)
        s.add_argument('--variants', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--protocol', type=Path, default=PROTOCOL)
        if name == 'freeze':
            s.add_argument('--precomputed', type=Path, required=True)
            s.add_argument('--exclusions', type=Path, default=None)
            s.add_argument('--clusters', type=Path, default=None)
            s.add_argument('--images', type=Path, default=None)
            s.add_argument('--adabn-logits', type=Path, default=None)
            s.add_argument('--adabn-list', type=Path, default=None)
            s.add_argument('--near-identical', type=Path, default=None, help='within-set vessel pairs (a, b, corr); pairs >= 0.95 are bound')
    g = sub.add_parser('logits')
    g.add_argument('--cohort', choices=sorted(COHORTS), required=True)
    g.add_argument('--variants', type=Path, required=True)
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--ids', nargs='+', required=True)
    g.add_argument('--device', default='cpu')
    g.add_argument('--threads', type=int, default=6)
    args = parser.parse_args(argv)
    {'freeze': freeze, 'evaluate': evaluate, 'logits': logits_command}[args.command](args)


if __name__ == '__main__':
    main()
