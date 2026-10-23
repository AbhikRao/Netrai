#!/usr/bin/env python3
"""Locked, single-use evaluation of frozen grading models on RFMiD 2.0 (any-DR endpoint; pre-registered in RFMID2_PREREG.md).

``freeze``   inventories the 859 identity-reviewed images (file names only), binds image hashes, the bootstrap clusters,
             models/policies/code and the CPU logits of every model into ``lock.json`` with ``labels_read: false``, and records
             the cohort as *locked* in the exposure registry. No label CSV is opened.
``evaluate`` verifies every binding, opens the label CSVs once through ``exposure.assert_label_read_allowed``, computes the
             pre-registered endpoints and paired comparisons with a cluster bootstrap, writes ``report.json`` and records the
             cohort as *exposed*. A second evaluation is refused.

The ``DR`` column is the presence of ANY diabetic retinopathy: it is never equated with referable DR. Endpoints:
  primary    AUC for DR of the score 1 - p(grade 0), p = softmax(logits / T) with the model's deployed temperature;
  secondary  AUC of the referral-head probability and of the expected-grade score; sensitivity/specificity at the frozen
             policy's grade >= 1 boundary; specificity separately for WNL negatives and other-disease negatives.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import exposure
from locked_evaluation import ROOT, check_matlab_variant, load_model, logits_for, sha, thresholds_for  # noqa: F401
from netrai_eval import bootstrap as bs
from netrai_eval import metrics as m
from models.netrai_model import NetrAIModel
from utils.artifacts import atomic_json
from utils.preprocessing import preprocess_for_model

COHORT = 'rfmid2'
DATA = ROOT / 'data/rfmid2/extracted'
SPLITS = ('Training_set', 'Validation_set', 'Test_set')
EXCLUSIONS = ROOT / 'results/model_improvement/rfmid2_identity_v2/exclusions_v1.txt'
CLUSTERS = ROOT / 'results/model_improvement/rfmid2_internal_clusters_0.6/clusters.csv'
PROTOCOL = ROOT / '.project-notes/docs/RFMID2_PREREG.md'
CODE = ['python/locked_rfmid.py', 'python/locked_evaluation.py', 'python/exposure.py', 'python/utils/preprocessing.py',
        'python/models/netrai_model.py', 'python/export_adapted_variant.py', 'python/utils/metrics.py',
        'python/netrai_eval/metrics.py', 'python/netrai_eval/bootstrap.py']
LEGACY_TEMPERATURE = ROOT / 'python/weights/calibration.json'
SPOT_CHECK_IMAGES, SPOT_CHECK_TOLERANCE = 8, 5e-3


def inventory(data: Path = DATA, exclusions: Path = EXCLUSIONS, clusters: Path = CLUSTERS) -> pd.DataFrame:
    """image_id (``<split>/<file>``), file name, cluster id. File names only; no label file is opened."""
    excluded = {line.strip() for line in exclusions.read_text().splitlines() if line.strip()}
    rows = []
    for split in SPLITS:
        for path in sorted((data / split).iterdir()):
            if path.suffix.lower() == '.jpg' and path.name not in excluded:
                rows.append({'image_id': f'{split}/{path.name}', 'file': path.name})
    frame = pd.DataFrame(rows)
    if frame.file.duplicated().any():
        raise ValueError('File names repeat across splits')
    cluster = pd.read_csv(clusters)
    cluster['file'] = cluster.image.str.replace(r'.*/', '', regex=True)
    mapping = dict(zip(cluster.file, cluster.cluster))
    missing = sorted(set(frame.file) - set(mapping))
    if missing:
        raise ValueError(f'Images without a cluster: {missing[:5]}')
    frame['cluster'] = [int(mapping[f]) for f in frame.file]
    return frame


def label_files() -> dict:
    """split -> label CSV path, taken from the exposure registry (this module never names the files itself)."""
    out = {}
    for item in exposure.entry(COHORT)['label_paths']:
        path = exposure._resolved(item)
        out[path.parent.name] = path
    return out


def read_labels(frame: pd.DataFrame, data: Path = DATA) -> pd.DataFrame:
    """Open the three label CSVs (latin-1). Callers must have passed ``assert_label_read_allowed`` first."""
    parts = []
    for split, path in label_files().items():
        table = pd.read_csv(path, encoding='latin-1')
        table.columns = [c.strip() for c in table.columns]
        table['split'] = split
        parts.append(table[['ID', 'DR', 'WNL', 'split']])
    labels = pd.concat(parts, ignore_index=True)
    out = frame.copy()
    stems = out.file.str.replace(r'\.[^.]+$', '', regex=True)
    if labels.duplicated(['split', 'ID']).any():
        raise ValueError('Duplicate label rows inside one split')
    key = {(row.split, str(row.ID)): row for row in labels.itertuples()}
    chosen = [key.get((image.split('/')[0], stem)) for image, stem in zip(out.image_id, stems)]
    out['has_label'] = [c is not None for c in chosen]  # images without a label row are reported and left out, never guessed
    out['DR'] = [int(c.DR) if c is not None else -1 for c in chosen]
    out['WNL'] = [int(c.WNL) if c is not None else -1 for c in chosen]
    return out


def temperature_for(spec: dict) -> float:
    if spec.get('policy'):
        return float(json.loads((ROOT / spec['policy']).read_text())['temperature'])
    return float(json.loads(LEGACY_TEMPERATURE.read_text())['temperature'])


def scores_for(logits: np.ndarray, temperature: float) -> dict:
    z = logits[:, :5] / temperature
    z = z - z.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return {'any_dr': 1 - p[:, 0], 'referral_head': 1 / (1 + np.exp(-logits[:, 5])), 'expected_grade': m.expected_grade_score(logits)}


def logits_on(spec: dict, frame: pd.DataFrame, folder: Path, device, checkpoint: Path | None = None, batch: int = 8) -> np.ndarray:
    """Logits of one ``full``/``legacy`` variant on any device (the Mac MPS path); same preprocessing as ``logits_for``."""
    size = int(spec.get('size', 512))
    if checkpoint is not None:
        model = NetrAIModel('efficientnet_b4', 5, pretrained=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state.get('model_state_dict', state))
        model = model.eval()
    else:
        model = load_model(spec)
    model = model.to(device)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(frame), batch):
            arrays = []
            for name in frame.image_id[start:start + batch]:
                rgb = cv2.cvtColor(cv2.imdecode(np.frombuffer((folder / name).read_bytes(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                arrays.append(preprocess_for_model(rgb, size=size).transpose(2, 0, 1))
            grade, referral = model(torch.from_numpy(np.stack(arrays)).float().to(device))
            outputs.append(torch.cat([grade, referral[:, None]], 1).float().cpu().numpy())
            if start % 200 == 0:
                print(f"{spec['id']}: {start}/{len(frame)}", flush=True)
    return np.concatenate(outputs).astype(np.float64)


def logits_command(args) -> None:
    """Compute logits for the listed variants (run on the Mac); needs only images + the small inventory inputs."""
    config = json.loads(args.variants.read_text())
    frame = inventory(args.data, args.exclusions, args.clusters)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    paths = dict(item.split('=', 1) for item in args.checkpoint)
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in config['variants']:
        if spec['id'] not in args.ids:
            continue
        target = args.out / f"logits_{spec['id']}.npy"
        if target.exists():
            continue
        checkpoint = Path(paths[spec['id']]) if spec['id'] in paths else None
        np.save(target, logits_on(spec, frame, args.data, device, checkpoint))
    frame.to_csv(args.out / 'inventory_rfmid2.csv', index=False)


def freeze(args) -> None:
    config = json.loads(args.variants.read_text())
    lock_path = args.output / 'lock.json'
    if lock_path.exists():
        raise FileExistsError('A lock may not be overwritten')
    exposure.assert_fresh_allowed(COHORT)
    matlab_files = check_matlab_variant(config) if config.get('matlab_primary_variant') else []
    args.output.mkdir(parents=True, exist_ok=True)
    frame = inventory(args.data, args.exclusions, args.clusters)
    frame['sha256'] = [sha(args.data / i) for i in frame.image_id]
    inv = args.output / 'inventory_rfmid2.csv'
    frame.to_csv(inv, index=False)
    cache, spot = {}, {}
    for spec in config['variants']:
        if spec['kind'] in ('ensemble', 'policy_of'):
            continue
        path = args.output / f"logits_{spec['id']}.npy"
        pre = args.precomputed / f"logits_{spec['id']}.npy" if args.precomputed else None
        if not path.exists() and pre is not None and pre.exists():  # computed elsewhere (Mac MPS): shape + CPU spot check
            saved = np.load(pre)
            if saved.shape != (len(frame), 6):
                raise ValueError(f'{pre.name}: shape {saved.shape} != ({len(frame)}, 6)')
            rows = np.unique(np.linspace(0, len(frame) - 1, SPOT_CHECK_IMAGES).round().astype(int))
            head = frame.iloc[rows].reset_index(drop=True)
            error = float(np.abs(logits_for(spec, head, folder=args.data) - saved[rows]).max())
            if error > SPOT_CHECK_TOLERANCE:
                raise ValueError(f'{pre.name}: CPU spot check differs by {error}')
            shutil.copy2(pre, path)
            spot[spec['id']] = error
        if not path.exists():
            np.save(path, logits_for(spec, frame, folder=args.data))
        cache[spec['id']] = path
    for spec in config['variants']:
        if spec['kind'] == 'ensemble':
            path = args.output / f"logits_{spec['id']}.npy"
            np.save(path, np.mean([np.load(cache[mid]) for mid in spec['members']], axis=0))
            cache[spec['id']] = path
    files = [args.variants, inv, args.clusters, args.exclusions, args.protocol, LEGACY_TEMPERATURE, *[ROOT / c for c in CODE],
             *matlab_files, *cache.values()]
    files += [ROOT / s['checkpoint'] for s in config['variants'] if s.get('checkpoint')]
    files += [ROOT / s['policy'] for s in config['variants'] if s.get('policy')]
    bindings = {str(Path(p).resolve().relative_to(ROOT)): sha(p) for p in files}
    atomic_json(lock_path, {
        'schema_version': 1, 'locked_utc': datetime.now(timezone.utc).isoformat(),
        'tests': {COHORT: f'{len(frame)} images after identity review (541.jpg excluded); any-DR endpoint'}, 'fresh': True,
        'labels_read': False, 'primary_variant': config['primary'], 'reference_variant': config['reference'],
        'bootstrap_seed': config['bootstrap_seed'], 'bootstrap_replicates': config['bootstrap_replicates'],
        'logits_provenance': config.get('logits_provenance'), 'precomputed_logits_cpu_spot_check_max_abs': spot, 'bindings': bindings})
    exposure.record_locked(COHORT, lock_path)
    print(f'Locked {len(bindings)} artifacts -> {lock_path}')


def endpoint_stats(dr: np.ndarray, wnl: np.ndarray, scores: dict, thresholds: np.ndarray) -> dict:
    predicted = scores['expected_grade'] >= thresholds[0]  # frozen policy's grade >= 1 boundary
    other = (dr == 0) & (wnl == 0)
    return {'auc_any_dr': m.auc(dr == 1, scores['any_dr']), 'auc_referral_head': m.auc(dr == 1, scores['referral_head']),
            'auc_expected_grade': m.auc(dr == 1, scores['expected_grade']),
            'sensitivity_grade_ge1': float(predicted[dr == 1].mean()),
            'specificity_grade_ge1': float((~predicted[dr == 0]).mean()),
            'specificity_wnl': float((~predicted[wnl == 1]).mean()) if (wnl == 1).any() else float('nan'),
            'specificity_other_disease': float((~predicted[other]).mean()) if other.any() else float('nan')}


def evaluate(args) -> None:
    lock = json.loads((args.output / 'lock.json').read_text())
    exposure.assert_label_read_allowed(COHORT, args.output)  # raises without a lock / after a report
    for path, digest in lock['bindings'].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f'Locked artifact changed: {path}')
    config = json.loads(args.variants.read_text())
    frame = pd.read_csv(args.output / 'inventory_rfmid2.csv')
    everything = read_labels(frame, args.data)
    keep = everything.has_label.to_numpy()
    unlabelled = list(everything.image_id[~keep])
    labeled = everything[keep].reset_index(drop=True)
    dr, wnl, clusters = labeled.DR.to_numpy(), labeled.WNL.to_numpy(), labeled.cluster.to_numpy()
    specs = {s['id']: s for s in config['variants']}
    scores, thresholds = {}, {}
    for spec in config['variants']:
        source = spec.get('logits_from', spec['id'])
        logits = np.load(args.output / f'logits_{source}.npy')[keep]
        scores[spec['id']] = scores_for(logits, temperature_for(spec))
        thresholds[spec['id']] = thresholds_for(spec)[0]
    reference = lock['reference_variant']
    seed, reps = lock['bootstrap_seed'], lock['bootstrap_replicates']
    results = {}
    for name in scores:
        def stat(index, name=name):
            return endpoint_stats(dr[index], wnl[index], {k: v[index] for k, v in scores[name].items()}, thresholds[name])
        results[name] = bs.bootstrap(stat, clusters, reps, seed)
    comparisons = {}
    for name in scores:
        if name == reference:
            continue
        def stat_a(index, name=name):
            return endpoint_stats(dr[index], wnl[index], {k: v[index] for k, v in scores[name].items()}, thresholds[name])
        def stat_b(index):
            return endpoint_stats(dr[index], wnl[index], {k: v[index] for k, v in scores[reference].items()}, thresholds[reference])
        comparisons[f'{name}_vs_{reference}'] = bs.paired_bootstrap(stat_a, stat_b, clusters, reps, seed)
    report = {'schema_version': 1, 'tier': 'TF', 'lock_sha256': sha(args.output / 'lock.json'),
              'evaluated_utc': datetime.now(timezone.utc).isoformat(), 'primary_variant': lock['primary_variant'],
              'reference_variant': reference,
              'population': {'images_in_lock': int(len(everything)), 'unlabelled_in_release': unlabelled, 'images': int(len(labeled)), 'dr_positive': int(dr.sum()), 'wnl': int(wnl.sum()),
                             'other_disease_negatives': int(((dr == 0) & (wnl == 0)).sum()), 'clusters': int(len(np.unique(clusters)))},
              'results': results, 'comparisons': comparisons,
              'limitations': ['DR is presence of ANY diabetic retinopathy, not referable DR or an ICDR grade.',
                              'RFMiD 2.0 comes from the same Nanded institute family as IDRiD (a training source): fresh images from a partly seen site, not an unseen domain.',
                              'Annotation adjudication procedure not verified (data descriptor not retrievable).',
                              'No patient ids: bootstrap clusters are vessel-correlation clusters at 0.6 (839 clusters); 541.jpg excluded as an ambiguous near-copy.']}
    report_path = args.output / 'report.json'
    atomic_json(report_path, report)
    exposure.record_exposure(COHORT, args.output / 'lock.json', report_path)
    for name, block in results.items():
        print(name, {k: round(v['estimate'], 4) for k, v in block.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('freeze', 'evaluate'):
        s = sub.add_parser(name)
        s.add_argument('--variants', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        s.add_argument('--data', type=Path, default=DATA)
        s.add_argument('--exclusions', type=Path, default=EXCLUSIONS)
        s.add_argument('--clusters', type=Path, default=CLUSTERS)
        s.add_argument('--protocol', type=Path, default=PROTOCOL)
        if name == 'freeze':
            s.add_argument('--precomputed', type=Path, help='folder with logits_<id>.npy computed elsewhere (spot-checked on CPU)')
    g = sub.add_parser('logits')
    g.add_argument('--variants', type=Path, required=True)
    g.add_argument('--out', type=Path, required=True)
    g.add_argument('--data', type=Path, required=True)
    g.add_argument('--exclusions', type=Path, required=True)
    g.add_argument('--clusters', type=Path, required=True)
    g.add_argument('--ids', nargs='+', required=True)
    g.add_argument('--checkpoint', nargs='*', default=[], help='id=path overrides for the checkpoint files')
    g.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    {'freeze': freeze, 'evaluate': evaluate, 'logits': logits_command}[args.command](args)


if __name__ == '__main__':
    main()
