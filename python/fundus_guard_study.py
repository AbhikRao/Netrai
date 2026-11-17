#!/usr/bin/env python3
"""X15: a guard against inputs that are not fundus photographs (study; rule in the private ledger, 2026-10-31).

Feature: the 576-D embedding of the frozen ImageNet MobileNetV3-Small encoder inside the quality candidate
(``python/weights/quality_model_candidate.pth``; input contract ``rgb_half_pixel_bilinear_float32_224_v1``, full frame).
Score: Mahalanobis distance to the embeddings of the datasets that trained or validated the published weights (APTOS, DDR,
DeepDRiD, EyeQ, IDRiD, Messidor-2), Ledoit-Wolf covariance. Threshold: margin x the 99.9th percentile of the fit scores.

  embed     cache embeddings per source (local only; DUA sets never leave the machine)
  select    leave-one-fit-source-out choice of the margin, on fundus images only
  evaluate  final fit on all fit sources; one evaluation on the external fundus sets and the local non-fundus images; writes the
            guard parameters (``--params``) and an aggregate report
"""
from __future__ import annotations

import argparse
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import shutil

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/model_improvement/x15_guard'
CANDIDATE = ROOT / 'python/weights/quality_model_candidate.pth'
FIT_SOURCES = ('aptos', 'ddr', 'deepdrid', 'eyeq', 'idrid', 'messidor2')
EXTERNAL_SOURCES = ('odir', 'fdr26', 'rfmid2', 'paraguay_hc', 'sustech_sysu', 'mbrset', 'brset')
NEGATIVE_SOURCES = ('matlab_imdata', 'matlab_visiondata', 'matlab_medical', 'matplotlib_samples', 'matlab_digits')
MARGINS = (1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
QUANTILE = 0.999
LODO_LIMIT = 0.002
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def frame(source, ids, paths, grade=None, gradable=None, quality_known=False, camera=None) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame({'source': source, 'image_id': list(ids), 'path': [str(p) for p in paths],
                         'grade': np.asarray(grade if grade is not None else [-1] * n, dtype=int),
                         'gradable': np.asarray(gradable if gradable is not None else [True] * n, dtype=bool),
                         'quality_known': quality_known, 'camera': list(camera) if camera is not None else [''] * n})


def require_readable(cohort: str) -> None:
    import exposure
    if exposure.status(cohort) not in ('exposed', 'training'):
        raise RuntimeError(f'{cohort} labels are {exposure.status(cohort)}; this study reads exposed or training labels only')


def inventory(source: str) -> pd.DataFrame:
    """Every image of one source with its DR grade (-1 unknown) and gradability (True when the source has no quality label)."""
    if source == 'aptos':
        labels = pd.read_csv(ROOT / 'data/aptos2019/train.csv')
        return frame(source, labels.id_code, [ROOT / f'data/aptos2019/train_images/{i}.png' for i in labels.id_code], labels.diagnosis)
    if source == 'eyeq':
        parts = []
        for split in ('train', 'test'):
            labels = pd.read_csv(ROOT / f'data/eyeq/data/Label_EyeQ_{split}.csv')
            paths = []
            for image in labels.image:
                stem = Path(image).stem
                paths.append(next(p for p in (ROOT / f'data/eyeq/images/{split}/{image}', ROOT / f'data/eyeq/images/{split}/{stem}.png',
                                              ROOT / f'data/eyeq/images/{split}/{stem}.jpg') if p.is_file()))
            parts.append(frame(source, split + '/' + labels.image, paths, labels.DR_grade, labels.quality.isin([0, 1]), True))
        return pd.concat(parts, ignore_index=True)
    if source == 'idrid':
        base = ROOT / 'data/idrid/B. Disease Grading'
        parts = []
        for csv, folder in (('a. IDRiD_Disease Grading_Training Labels.csv', 'a. Training Set'),
                            ('b. IDRiD_Disease Grading_Testing Labels.csv', 'b. Testing Set')):
            labels = pd.read_csv(base / '2. Groundtruths' / csv)
            ids = labels['Image name'].astype(str).str.strip()
            parts.append(frame(source, folder[:1] + '/' + ids, [base / '1. Original Images' / folder / f'{i}.jpg' for i in ids],
                               labels['Retinopathy grade']))
        return pd.concat(parts, ignore_index=True)
    if source == 'deepdrid':
        prep = ROOT / '.project-notes/deepdrid-preparation/reviewed-development-2026-09-28'
        release = ROOT / 'data/DRiD/DeepDRiD-master'
        parts = []
        for split in ('training', 'validation', 'evaluation'):
            table = pd.read_csv(prep / f'{split}.csv', dtype={'patient_id': str, 'id_code': str})
            if split == 'evaluation':
                labelled = pd.read_csv(ROOT / 'results/model_improvement/acceptance/labelled/evaluation_labelled.csv',
                                       dtype={'patient_id': str, 'id_code': str})
                table = table.drop(columns=['true_grade', 'grading_eligible'], errors='ignore').merge(
                    labelled[['id_code', 'true_grade', 'grading_eligible']], on='id_code', validate='one_to_one')
            grade = pd.to_numeric(table.true_grade, errors='coerce').fillna(-1)
            parts.append(frame(source, split + '/' + table.id_code, [release / p for p in table.image_relative_path], grade,
                               table.grading_eligible.astype(bool), True))
        return pd.concat(parts, ignore_index=True)
    if source == 'ddr':
        base = ROOT / 'data/DDR/extracted/DDR-dataset/DR_grading'
        parts = []
        for split in ('train', 'valid', 'test'):
            labels = pd.read_csv(base / f'{split}.txt', sep=' ', header=None, names=['file', 'grade'])
            parts.append(frame(source, split + '/' + labels.file, [base / split / f for f in labels.file],
                               np.where(labels.grade <= 4, labels.grade, -1), labels.grade <= 4, True))
        return pd.concat(parts, ignore_index=True)
    if source == 'messidor2':
        import locked_evaluation as le
        require_readable('messidor2')
        listing = le.inventory('messidor2')
        listing, gradable, grade = le.cohort_labels('messidor2', listing)
        return frame(source, listing.image_id, [ROOT / 'data/messidor2/images' / i for i in listing.image_id],
                     np.where(gradable, grade, -1), gradable, True)
    if source in ('odir', 'fdr26'):
        import locked_evaluation as le
        require_readable(source)
        listing = le.inventory(source)
        listing, has_grade, grade = le.cohort_labels(source, listing)
        folder = le.ODIR / 'Training Images' if source == 'odir' else le.FDR26 / 'extracted/images'
        if source == 'odir':     # gradable = not marked with a quality term; other-disease-only eyes are gradable fundus images
            gradable = (listing.odir_status != 'ungradable').to_numpy()
            return frame(source, listing.image_id, [folder / i for i in listing.image_id], grade.astype(int), gradable, True)
        return frame(source, listing.image_id, [folder / i for i in listing.image_id], grade.astype(int))
    if source == 'rfmid2':
        import locked_rfmid as lr
        require_readable('rfmid2')
        listing = lr.read_labels(lr.inventory())
        listing = listing.loc[listing.has_label]
        return frame(source, listing.image_id, [lr.DATA / i for i in listing.image_id])
    if source in ('paraguay_hc', 'sustech_sysu'):
        import locked_fresh2 as lf
        require_readable(source)
        listing, _ = lf.read_labels(source, lf.inventory(source))
        listing = listing.loc[~listing.excluded]
        return frame(source, listing.image_id, [lf.data_dir(source) / 'images' / i for i in listing.image_id], listing.grade)
    if source == 'mbrset':
        import locked_mbrset as lm
        require_readable('mbrset')
        listing = lm.read_labels(lm.inventory())
        listing = listing.loc[listing.has_row]
        return frame(source, listing.image_id, [lm.DATA / i for i in listing.image_id], listing.icdr, listing.gradable, True)
    if source == 'brset':
        import locked_brset as lb
        require_readable('brset')
        listing, _ = lb.read_labels(lb.inventory())
        listing = listing.loc[listing.has_row]
        return frame(source, listing.image_id, [lb.DATA / i for i in listing.image_id], listing.icdr,
                     listing.gradable, True, listing.camera)
    matlab_root = matlab_installation()
    folders = {'matlab_imdata': [matlab_root / 'toolbox/images/imdata'], 'matlab_visiondata': [matlab_root / 'toolbox/vision/visiondata'],
               'matlab_medical': [matlab_root / 'toolbox/medical'], 'matlab_digits': [matlab_root / 'toolbox/nnet/nndemos/nndatasets'],
               'matplotlib_samples': [Path(__import__('matplotlib').get_data_path()) / 'sample_data']}
    if source in folders:
        paths = sorted(p for folder in folders[source] for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        return frame(source, [str(p.relative_to(folders[source][0])) for p in paths], paths)
    raise ValueError(f'Unknown source {source}')


def matlab_installation() -> Path:
    """MATLAB root from the ``matlab`` command on the PATH (its sample images are the local negatives; never redistributed)."""
    command = shutil.which('matlab')
    if command is None:
        raise RuntimeError('MATLAB is not on the PATH; its sample images are needed as non-fundus negatives')
    return Path(command).resolve().parents[1]


def encoder():
    import torch
    from models.quality_model import QualityModel
    expected = json.loads((ROOT / 'python/weights/quality_model_candidate.json').read_text())['checkpoint_sha256']
    if hashlib.sha256(CANDIDATE.read_bytes()).hexdigest() != expected:
        raise ValueError('quality_model_candidate.pth does not match its recorded SHA-256')
    model = QualityModel(pretrained=False)
    model.load_state_dict(torch.load(CANDIDATE, map_location='cpu', weights_only=True))
    return model.eval()


def read_rgb(path: str):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def embed(model, paths, batch: int = 32, threads: int = 6) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from models.quality_model import quality_input
    features = np.full((len(paths), 576), np.nan, np.float32)
    readable = np.zeros(len(paths), bool)
    with ThreadPoolExecutor(threads) as pool:
        for start in range(0, len(paths), batch):
            images = list(pool.map(read_rgb, paths[start:start + batch]))
            arrays = [quality_input(i) if i is not None and min(i.shape[:2]) > 1 else None for i in images]
            index = [k for k, a in enumerate(arrays) if a is not None]
            if index:
                with torch.no_grad():
                    out = model.encode(torch.from_numpy(np.stack([arrays[k] for k in index])))
                features[[start + k for k in index]] = out.numpy()
                readable[[start + k for k in index]] = True
    return features, readable


def embed_command(args) -> None:
    import torch
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    model = encoder()
    for source in args.sources:
        target = OUT / f'emb_{source}.npz'
        if target.is_file() and not args.force:
            print(f'{source}: cached', flush=True)
            continue
        table = inventory(source)
        features, readable = embed(model, table.path.tolist())
        np.savez(target, features=features, readable=readable, image_id=table.image_id.to_numpy(str),
                 grade=table.grade.to_numpy(), gradable=table.gradable.to_numpy(), quality_known=table.quality_known.to_numpy(),
                 camera=table.camera.to_numpy(str))
        print(f'{source}: {len(table)} images, {int(readable.sum())} readable', flush=True)


def load(source: str) -> dict:
    data = dict(np.load(OUT / f'emb_{source}.npz'))
    keep = data['readable']
    return {k: v[keep] for k, v in data.items()} | {'unreadable': int((~keep).sum())}


class Guard:
    """Ledoit-Wolf Mahalanobis distance to fundus embeddings."""

    def __init__(self, features: np.ndarray):
        from sklearn.covariance import LedoitWolf
        estimate = LedoitWolf().fit(features.astype(np.float64))
        self.mean = estimate.location_
        self.precision = estimate.precision_
        self.shrinkage = float(estimate.shrinkage_)
        self.reference = float(np.quantile(self.score(features), QUANTILE))

    def score(self, features: np.ndarray) -> np.ndarray:
        centred = features.astype(np.float64) - self.mean
        return np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', centred, self.precision, centred), 0))


def rejection(scores: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    n = int(mask.sum())
    k = int((scores[mask] > threshold).sum())
    return {'n': n, 'rejected': k, 'rate': k / n if n else None}


def select_command(args) -> None:
    fit = {s: load(s) for s in FIT_SOURCES}
    folds = {}
    for held in FIT_SOURCES:
        guard = Guard(np.concatenate([fit[s]['features'] for s in FIT_SOURCES if s != held]))
        scores = guard.score(fit[held]['features'])
        folds[held] = {'reference_q999': guard.reference, 'scores': scores}
    table = {}
    for margin in MARGINS:
        table[str(margin)] = {held: rejection(f['scores'], fit[held]['gradable'], margin * f['reference_q999']) for held, f in folds.items()}
    worst = {m: max(r['rate'] for r in rows.values()) for m, rows in table.items()}
    chosen = next((float(m) for m in map(str, MARGINS) if worst[m] <= LODO_LIMIT), MARGINS[-1])
    result = {'rule': f'smallest margin in {list(MARGINS)} with max LODO false rejection <= {LODO_LIMIT}; else {MARGINS[-1]}',
              'quantile': QUANTILE, 'lodo_false_rejection_by_margin': table, 'worst_by_margin': worst, 'chosen_margin': chosen,
              'fit_images': {s: int(len(fit[s]['features'])) for s in FIT_SOURCES}}
    (OUT / 'selection.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({'worst_by_margin': worst, 'chosen_margin': chosen}, indent=1))


def summarize(source: str, data: dict, scores: np.ndarray, threshold: float) -> dict:
    gradable = data['gradable']
    out = {'images': int(len(scores)), 'unreadable': data['unreadable'],
           'gradable': rejection(scores, gradable, threshold),
           'ungradable': rejection(scores, ~gradable, threshold) if data['quality_known'].any() else None,
           'by_grade': {str(g): rejection(scores, gradable & (data['grade'] == g), threshold) for g in range(5)
                        if (gradable & (data['grade'] == g)).any()},
           'score_median': float(np.median(scores)), 'score_p99': float(np.quantile(scores, 0.99)),
           'score_over_threshold_p99': float(np.quantile(scores, 0.99) / threshold)}
    cameras = sorted(set(data['camera']) - {''})
    if cameras:
        out['by_camera'] = {c: rejection(scores, gradable & (data['camera'] == c), threshold) for c in cameras}
    return out


def evaluate_command(args) -> None:
    selection = json.loads((OUT / 'selection.json').read_text())
    margin = selection['chosen_margin']
    fit = {s: load(s) for s in FIT_SOURCES}
    guard = Guard(np.concatenate([fit[s]['features'] for s in FIT_SOURCES]))
    threshold = margin * guard.reference
    report = {'tier': 'TX (external fundus sets exposed; negatives local, not redistributed)', 'margin': margin,
              'reference_q999': guard.reference, 'threshold': threshold, 'shrinkage': guard.shrinkage,
              'fit_images': selection['fit_images'], 'fit': {}, 'external': {}, 'negatives': {}}
    for source in FIT_SOURCES:
        report['fit'][source] = summarize(source, fit[source], guard.score(fit[source]['features']), threshold)
    pooled_grade4 = [0, 0]
    for source in EXTERNAL_SOURCES:
        data = load(source)
        summary = summarize(source, data, guard.score(data['features']), threshold)
        report['external'][source] = summary
        if '4' in summary['by_grade']:
            pooled_grade4[0] += summary['by_grade']['4']['rejected']
            pooled_grade4[1] += summary['by_grade']['4']['n']
    for held, rows in selection['lodo_false_rejection_by_margin'][str(margin)].items():
        report.setdefault('lodo_at_chosen_margin', {})[held] = rows
    lodo4 = [0, 0]
    for held in FIT_SOURCES:     # grade-4 images of each fit source scored by the guard fitted without it
        other = Guard(np.concatenate([fit[s]['features'] for s in FIT_SOURCES if s != held]))
        scores = other.score(fit[held]['features'])
        mask = fit[held]['gradable'] & (fit[held]['grade'] == 4)
        lodo4[0] += int((scores[mask] > margin * other.reference).sum())
        lodo4[1] += int(mask.sum())
    report['grade4_pooled'] = {'external': {'rejected': pooled_grade4[0], 'n': pooled_grade4[1]},
                               'lodo': {'rejected': lodo4[0], 'n': lodo4[1]},
                               'rate': (pooled_grade4[0] + lodo4[0]) / max(pooled_grade4[1] + lodo4[1], 1)}
    natural = [0, 0]
    for source in NEGATIVE_SOURCES:
        data = load(source)
        scores = guard.score(data['features'])
        summary = {'images': int(len(scores)), 'unreadable': data['unreadable'], 'rejected': int((scores > threshold).sum()),
                   'rate': float((scores > threshold).mean()) if len(scores) else None,
                   'score_min_over_threshold': float(scores.min() / threshold) if len(scores) else None}
        report['negatives'][source] = summary
        if source != 'matlab_digits':
            natural[0] += summary['rejected']
            natural[1] += summary['images']
    report['negatives_natural_and_medical'] = {'rejected': natural[0], 'n': natural[1], 'rate': natural[0] / max(natural[1], 1)}
    external_worst = max(r['gradable']['rate'] for r in report['external'].values())
    lodo_worst = max(r['rate'] for r in report['lodo_at_chosen_margin'].values())
    report['acceptance'] = {
        'i_false_rejection_le_0.5pct_every_external_and_lodo': bool(external_worst <= 0.005 and lodo_worst <= 0.005),
        'ii_grade4_pooled_le_1pct': bool(report['grade4_pooled']['rate'] <= 0.01),
        'iii_negatives_ge_95pct': bool(report['negatives_natural_and_medical']['rate'] >= 0.95),
        'external_worst': external_worst, 'lodo_worst': lodo_worst}
    passed = report['acceptance']['i_false_rejection_le_0.5pct_every_external_and_lodo'] and report['acceptance']['ii_grade4_pooled_le_1pct']
    report['decision'] = 'hard reject' if passed else 'warning flag only'
    (OUT / 'evaluation.json').write_text(json.dumps(report, indent=1) + '\n')
    if args.params:
        write_params(guard, threshold, margin, selection, Path(args.params), 'reject' if passed else 'flag')
    print(json.dumps({k: report[k] for k in ('threshold', 'acceptance', 'decision', 'grade4_pooled', 'negatives_natural_and_medical')}, indent=1))


def write_params(guard: Guard, threshold: float, margin: float, selection: dict, path: Path, action: str) -> None:
    """Guard parameters for both runtimes: JSON metadata plus little-endian float32 mean (576) and precision (576 x 576, row-major)."""
    blob = np.concatenate([guard.mean.astype('<f4'), guard.precision.astype('<f4').ravel()]).tobytes()
    binary = path.with_suffix('.f32')
    binary.write_bytes(blob)
    path.write_text(json.dumps({
        'schema_version': 1, 'purpose': 'reject inputs that are not fundus photographs (X15)',
        'encoder': 'quality candidate MobileNetV3-Small (frozen ImageNet), 576-D global-average-pooled features',
        'encoder_checkpoint_sha256': json.loads((ROOT / 'python/weights/quality_model_candidate.json').read_text())['checkpoint_sha256'],
        'input_contract': 'rgb_half_pixel_bilinear_float32_224_v1', 'dimension': 576,
        'score': 'Mahalanobis distance sqrt((x - mean)^T P (x - mean)), Ledoit-Wolf precision P, float32 parameters',
        'threshold': float(threshold), 'action': action, 'margin': margin, 'reference_quantile': QUANTILE,
        'fit_sources': list(FIT_SOURCES), 'fit_images': selection['fit_images'],
        'parameters_file': binary.name, 'parameters_sha256': hashlib.sha256(blob).hexdigest()}, indent=1) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    e = commands.add_parser('embed')
    e.add_argument('--sources', nargs='+', default=list(FIT_SOURCES + EXTERNAL_SOURCES + NEGATIVE_SOURCES))
    e.add_argument('--threads', type=int, default=4)
    e.add_argument('--force', action='store_true')
    commands.add_parser('select')
    v = commands.add_parser('evaluate')
    v.add_argument('--params', help='write the guard parameters to this JSON path (plus a .f32 file next to it)')
    args = parser.parse_args(argv)
    {'embed': embed_command, 'select': select_command, 'evaluate': evaluate_command}[args.command](args)


if __name__ == '__main__':
    main()
