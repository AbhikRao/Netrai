#!/usr/bin/env python3
"""Cache frozen legacy-model features for CPU-feasible grading adaptation.

For every image the deployed ``legacy-area-v1`` input is passed through the
frozen EfficientNet-B4 backbone. Stored per image:

* ``block5``: float16 output of ``backbone.blocks[5]`` (272x16x16), the input of
  the trainable last stage used by ``train_grading_adapter.py``;
* ``pooled``: float32 1792-D pooled backbone feature;
* ``logits``: float32 legacy six-value output (5 grade logits + referral logit).

Training cohorts optionally also store the horizontally flipped view. Each
shard is bound to image SHA-256s, checkpoint, preprocessing and code hashes, so
an interrupted run resumes and a changed input is refused instead of reused.
No reserved DeepDRiD evaluation image is accepted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from utils.artifacts import atomic_json, sha256_file
from utils.preprocessing import preprocess_for_model

ROOT = Path(__file__).resolve().parents[1]
DEEPDRID = ROOT / '.project-notes/deepdrid-preparation/reviewed-development-2026-09-28'
DEEPDRID_ROOT = ROOT / 'data/DRiD/DeepDRiD-master'
IDRID = ROOT / 'data/idrid/B. Disease Grading'
EYEQ = ROOT / 'data/eyeq'
CHECKPOINT = ROOT / 'python/weights/netrai_fold0_best.pth'
SHARD = 128


def cohort(name: str, eyeq_test_limit: int, seed: int,
           eyeq_train_normal_patients: int = 0) -> pd.DataFrame:
    """Return image_id, path, grade, group_id and source for one named cohort."""
    if name in ('deepdrid_training', 'deepdrid_validation'):
        frame = pd.read_csv(DEEPDRID / f"{name.split('_')[1]}.csv", dtype={'patient_id': str})
        frame = frame.loc[(frame.development_excluded == 0) & (frame.grading_eligible == 1)]
        return pd.DataFrame({
            'image_id': frame.id_code, 'path': [str(DEEPDRID_ROOT / p) for p in frame.image_relative_path],
            'grade': frame.true_grade.astype(int), 'group_id': 'deepdrid_' + frame.patient_id,
            'expected_sha256': frame.image_sha256, 'source': 'deepdrid'})
    if name in ('idrid_train', 'idrid_test'):
        split = 'a. IDRiD_Disease Grading_Training Labels.csv' if name == 'idrid_train' else \
            'b. IDRiD_Disease Grading_Testing Labels.csv'
        folder = 'a. Training Set' if name == 'idrid_train' else 'b. Testing Set'
        labels = pd.read_csv(IDRID / '2. Groundtruths' / split)
        ids = labels['Image name'].astype(str).str.strip()
        return pd.DataFrame({
            'image_id': ids, 'path': [str(IDRID / '1. Original Images' / folder / f'{i}.jpg') for i in ids],
            'grade': labels['Retinopathy grade'].astype(int), 'group_id': 'idrid_' + ids,
            'expected_sha256': '', 'source': 'idrid'})
    if name in ('eyeq_train', 'eyeq_test'):
        split = name.split('_')[1]
        labels = pd.read_csv(EYEQ / f'data/Label_EyeQ_{split}.csv')
        labels = labels.loc[labels.quality.isin([0, 1])]  # Reject images lack reliable grades
        labels['patient'] = labels.image.str.split('_').str[0]
        if name == 'eyeq_train' and eyeq_train_normal_patients:
            # Keep every patient with any DR grade; sample all-normal patients.
            worst = labels.groupby('patient').DR_grade.max()
            normal = worst.index[worst == 0].to_series().sample(frac=1, random_state=seed)
            keep = set(worst.index[worst > 0]) | set(normal.iloc[:eyeq_train_normal_patients])
            labels = labels.loc[labels.patient.isin(keep)]
        if name == 'eyeq_test':
            overlap = ROOT / 'results/verification_2026-09-24/perceptual_overlap/visually_confirmed_pairs.csv'
            if overlap.is_file():  # same-retina copies of training images are not test data
                labels = labels.loc[~labels.image.isin(pd.read_csv(overlap).image_b)]
            if eyeq_test_limit:
                patients = labels.patient.drop_duplicates().sample(frac=1, random_state=seed)
                chosen, total = [], 0
                counts = labels.groupby('patient').size()
                for patient in patients:
                    if total >= eyeq_test_limit:
                        break
                    chosen.append(patient)
                    total += int(counts[patient])
                labels = labels.loc[labels.patient.isin(chosen)]
        paths = []
        for image in labels.image:
            stem = Path(image).stem
            found = next((p for p in (EYEQ / f'images/{split}/{image}', EYEQ / f'images/{split}/{stem}.png',
                                      EYEQ / f'images/{split}/{stem}.jpg') if p.is_file()), None)
            if found is None:
                raise FileNotFoundError(image)
            paths.append(str(found))
        return pd.DataFrame({
            'image_id': labels.image.values, 'path': paths, 'grade': labels.DR_grade.astype(int).values,
            'group_id': ('eyeq_' + labels.patient).values, 'expected_sha256': '', 'source': 'eyeq'})
    raise ValueError(f'Unsupported cohort {name}; the DeepDRiD evaluation set is reserved')


class Images(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        digest = sha256_file(row.path)
        if row.expected_sha256 and digest != row.expected_sha256:
            raise ValueError(f'Image bytes differ from manifest: {row.image_id}')
        data = np.frombuffer(Path(row.path).read_bytes(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)  # decode the hashed bytes
        if bgr is None:
            raise ValueError(f'Cannot decode {row.path}')
        array = preprocess_for_model(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return torch.from_numpy(array.transpose(2, 0, 1).copy()), digest


def forward(model, batch):
    backbone = model.backbone
    hidden = backbone.bn1(backbone.conv_stem(batch))
    for index, block in enumerate(backbone.blocks):
        hidden = block(hidden)
        if index == 5:
            block5 = hidden
    pooled = backbone.global_pool(backbone.bn2(backbone.conv_head(hidden)))
    features = model.head(pooled)
    logits = torch.cat([model.grade_head(features), model.ref_head(features)], 1)
    return block5, pooled, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohorts', nargs='+')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/model_improvement/feature_cache')
    parser.add_argument('--flip', nargs='*', default=['deepdrid_training', 'idrid_train', 'eyeq_train'])
    parser.add_argument('--eyeq-test-limit', type=int, default=3000)
    parser.add_argument('--eyeq-train-normal-patients', type=int, default=1300)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    from inference import _get_model
    model, _, _ = _get_model()
    model.eval()
    code = {name: sha256_file(Path(__file__).parent / name) for name in
            ('extract_backbone_cache.py', 'utils/preprocessing.py', 'models/netrai_model.py')}
    for name in args.cohorts:
        frame = cohort(name, args.eyeq_test_limit, args.seed,
                       args.eyeq_train_normal_patients).reset_index(drop=True)
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        identity = {'cohort': name, 'rows': len(frame), 'checkpoint_sha256': sha256_file(CHECKPOINT),
                    'preprocessing': 'legacy-area-v1', 'code_sha256': code,
                    'flip': name in args.flip, 'torch': str(torch.__version__), 'opencv': cv2.__version__,
                    'image_list_sha256': hashlib.sha256('\n'.join(frame.path).encode()).hexdigest()}
        identity_path = folder / 'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError(f'{folder} belongs to a different extraction; use a new output folder')
        atomic_json(identity_path, identity)
        frame.drop(columns='expected_sha256').to_csv(folder / 'rows.csv', index=False)
        for start in range(0, len(frame), SHARD):
            shard = folder / f'shard_{start // SHARD:05d}.npz'
            if shard.exists():
                continue
            part = frame.iloc[start:start + SHARD]
            loader = DataLoader(Images(part), batch_size=args.batch_size, num_workers=args.workers)
            outputs = {key: [] for key in ('block5', 'pooled', 'logits', 'block5_flip', 'pooled_flip', 'logits_flip')}
            digests = []
            with torch.inference_mode():
                for batch, batch_digests in loader:
                    digests.extend(batch_digests)
                    views = [('', batch)] + ([('_flip', torch.flip(batch, dims=[3]))] if identity['flip'] else [])
                    for suffix, view in views:
                        block5, pooled, logits = forward(model, view)
                        outputs['block5' + suffix].append(block5.numpy().astype(np.float16))
                        outputs['pooled' + suffix].append(pooled.numpy().astype(np.float32))
                        outputs['logits' + suffix].append(logits.numpy().astype(np.float32))
            arrays = {key: np.concatenate(value) for key, value in outputs.items() if value}
            if any(not np.isfinite(value.astype(np.float32)).all() for value in arrays.values()):
                raise ValueError(f'Nonfinite features in {shard}')
            temporary = shard.with_suffix('.tmp.npz')
            np.savez(temporary, image_id=part.image_id.to_numpy(str), sha256=np.asarray(digests), **arrays)
            temporary.replace(shard)
            print(f'{name}: {min(start + SHARD, len(frame))}/{len(frame)}', flush=True)
        print(f'{name}: complete', flush=True)


if __name__ == '__main__':
    main()
