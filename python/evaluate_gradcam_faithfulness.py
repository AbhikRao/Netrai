#!/usr/bin/env python3
"""X11: is the referral Grad-CAM faithful to lesions? (post hoc, tier TX: the lesion masks of two exposed test sets)

Images: the DDR lesion-segmentation test (the 202 primary images of the locked evaluation) and the IDRiD segmentation
test (27). Models: the deployed legacy model (512 px) and R5_768_s17 (768 px). For each image the referral Grad-CAM
(``model.backbone.conv_head``, target = referral logit) is compared with the union of the expert lesion masks
(EX, HE, MA, SE), mapped into the classifier's crop/resize space:

* energy-in-lesion ratio  = (Grad-CAM mass inside lesions / mass in the field of view) / (lesion area / field area);
  1 = chance, > 1 = attention concentrates on lesions;
* pointing game           = the Grad-CAM maximum falls inside lesions dilated by 15 px (scaled to the input size),
  against the chance level of that dilated area fraction;
* deletion curve          = referral probability (relative to the unmasked image) after masking the top 10/25/50% of
  field pixels in Grad-CAM order, against random order (2 draws); only images the model calls referable (p >= 0.5).

Image-level bootstrap CIs. This measures how well the explanation aligns with annotated lesions; it does not prove
the model uses them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import lesion_data as ld
from utils.preprocessing import model_space_fov_mask, preprocess_for_model

FRACTIONS = (0.10, 0.25, 0.50)
POINTING_RADIUS_AT_512 = 15


def lesion_in_model_space(masks: np.ndarray, annotated: np.ndarray, size: int) -> np.ndarray:
    """Union of the annotated lesion masks (cropped to the field), mapped to the square model space (any coverage = lesion)."""
    union = np.zeros(masks.shape[:2], np.float32)
    for index in np.flatnonzero(annotated):
        union = np.maximum(union, (masks[..., index] > 0).astype(np.float32))
    return cv2.resize(union, (size, size), interpolation=cv2.INTER_AREA) > 0


def energy_ratio(cam: np.ndarray, lesion: np.ndarray, fov: np.ndarray) -> float:
    inside = lesion & fov
    if not inside.any() or cam[fov].sum() <= 0:
        return float('nan')
    return float((cam[inside].sum() / cam[fov].sum()) / (inside.sum() / fov.sum()))


def pointing_hit(cam: np.ndarray, lesion: np.ndarray, fov: np.ndarray, size: int) -> tuple[bool, float]:
    """(top Grad-CAM pixel inside dilated lesions, chance level = dilated area fraction of the field)."""
    radius = max(1, round(POINTING_RADIUS_AT_512 * size / 512))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(lesion.astype(np.uint8), kernel) > 0
    masked = np.where(fov, cam, -1.0)
    y, x = np.unravel_index(int(masked.argmax()), masked.shape)
    return bool(dilated[y, x]), float((dilated & fov).sum() / fov.sum())


def masked_inputs(tensor: torch.Tensor, order: np.ndarray, fov_count: int, fractions=FRACTIONS) -> torch.Tensor:
    """Copies of ``tensor`` (1, 3, H, W) with the first fraction*fov_count pixels of ``order`` set to the mean colour (0)."""
    out = []
    height, width = tensor.shape[-2:]
    for fraction in fractions:
        copy = tensor.clone()
        pixels = order[:int(round(fraction * fov_count))]
        copy[0, :, pixels // width, pixels % width] = 0.0
        out.append(copy)
    return torch.cat(out)


@torch.inference_mode()
def referral_probability(model, batch: torch.Tensor, chunk: int = 5) -> np.ndarray:
    outputs = []
    for start in range(0, len(batch), chunk):
        _, referral = model(batch[start:start + chunk])
        outputs.append(torch.sigmoid(referral).float().cpu().numpy())
    return np.concatenate(outputs)


def deletion_curves(model, tensor: torch.Tensor, cam: np.ndarray, fov: np.ndarray, rng, random_draws: int = 2) -> dict:
    size = cam.shape[0]
    fov_flat = np.flatnonzero(fov.ravel())
    cam_order = fov_flat[np.argsort(-cam.ravel()[fov_flat], kind='stable')]
    base = float(referral_probability(model, tensor)[0])
    batches = [masked_inputs(tensor, cam_order, len(fov_flat))]
    for _ in range(random_draws):
        batches.append(masked_inputs(tensor, rng.permutation(fov_flat), len(fov_flat)))
    probs = referral_probability(model, torch.cat(batches)).reshape(1 + random_draws, len(FRACTIONS))
    return {'base': base, 'cam': probs[0], 'random': probs[1:].mean(0)}


def curve_auc(base: float, values: np.ndarray) -> float:
    """Mean relative referral probability over the deleted fractions (trapezoid, 0 -> 0.5); lower = faster drop."""
    xs = np.r_[0.0, FRACTIONS]
    ys = np.r_[1.0, np.asarray(values) / max(base, 1e-9)]
    return float(np.trapezoid(ys, xs) / xs[-1])


def bootstrap_mean(values: np.ndarray, replicates=2000, seed=0) -> list:
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(len(values), size=len(values))].mean() for _ in range(replicates)]
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def load_model_spec(name: str):
    from locked_evaluation import ROOT, load_model
    if name == 'legacy':
        return load_model({'kind': 'legacy'}), 512
    if name == 'R5_768_s17':
        return load_model({'kind': 'full', 'checkpoint': 'results/model_improvement/kaggle_output_r5_s17/run/best.pt'}), 768
    raise ValueError(name)


def cam_for(model, tensor: torch.Tensor) -> np.ndarray:
    from pytorch_grad_cam import GradCAM

    class BothHeads(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, inputs):
            logits, referral = self.network(inputs)
            return torch.cat([logits, referral[:, None]], dim=1)

    with GradCAM(model=BothHeads(model).eval(), target_layers=[model.backbone.conv_head]) as cam:
        return cam(input_tensor=tensor, targets=[lambda output: output[5]])[0]


def records(dataset: str, exclusions: Path):
    if dataset == 'ddr':
        keep = set(pd.read_csv(exclusions).query("decision == 'keep'").image_id.astype(str))
        return [r for r in ld.ddr_test_records_exposed() if r.image_id.removeprefix('ddr_') in keep]
    if dataset == 'idrid':
        return ld.idrid_test_records()
    raise ValueError(dataset)


def run(model_name: str, dataset: str, exclusions: Path, limit: int, seed: int) -> pd.DataFrame:
    model, size = load_model_spec(model_name)
    rng = np.random.default_rng(seed)
    rows = []
    for record in records(dataset, exclusions)[:limit or None]:
        rgb = cv2.cvtColor(cv2.imread(str(record.image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        masks, annotated = ld.read_masks(record, rgb.shape[:2])
        top, bottom, left, right = ld.foreground_box(rgb)
        lesion = lesion_in_model_space(masks[top:bottom, left:right], annotated, size)
        fov = model_space_fov_mask(rgb, size)
        if not (lesion & fov).any():
            rows.append({'image': record.image_id, 'has_lesion': False})
            continue
        tensor = torch.from_numpy(preprocess_for_model(rgb, size=size).transpose(2, 0, 1)[None]).float()
        cam = cam_for(model, tensor)
        hit, chance = pointing_hit(cam, lesion, fov, size)
        curves = deletion_curves(model, tensor, cam, fov, rng)
        rows.append({'image': record.image_id, 'has_lesion': True, 'energy_ratio': energy_ratio(cam, lesion, fov),
                     'lesion_fraction': float((lesion & fov).sum() / fov.sum()), 'pointing_hit': hit, 'pointing_chance': chance,
                     'referral_p': curves['base'], 'del_auc_cam': curve_auc(curves['base'], curves['cam']),
                     'del_auc_random': curve_auc(curves['base'], curves['random'])})
        print(f'{model_name}/{dataset}: {len(rows)}', flush=True)
    return pd.DataFrame(rows)


def summarise(frame: pd.DataFrame, seed: int = 0) -> dict:
    lesion = frame[frame.has_lesion]
    out = {'images': int(len(frame)), 'images_with_lesions': int(len(lesion))}
    er = lesion.energy_ratio.to_numpy(float)
    out['energy_ratio'] = {'mean': float(np.nanmean(er)), 'median': float(np.nanmedian(er)), 'ci95_mean': bootstrap_mean(er, seed=seed),
                           'fraction_above_1': float(np.nanmean(er > 1))}
    hit = lesion.pointing_hit.to_numpy(float)
    chance = lesion.pointing_chance.to_numpy(float)
    out['pointing_game'] = {'hit_rate': float(hit.mean()), 'ci95': bootstrap_mean(hit, seed=seed), 'chance': float(chance.mean()),
                            'hit_minus_chance': float((hit - chance).mean()), 'ci95_diff': bootstrap_mean(hit - chance, seed=seed)}
    referable = lesion[lesion.referral_p >= 0.5]
    if len(referable):
        diff = (referable.del_auc_cam - referable.del_auc_random).to_numpy(float)
        out['deletion'] = {'images_called_referable': int(len(referable)), 'auc_cam': float(referable.del_auc_cam.mean()),
                           'auc_random': float(referable.del_auc_random.mean()), 'cam_minus_random': float(diff.mean()),
                           'ci95': bootstrap_mean(diff, seed=seed), 'note': 'negative = Grad-CAM order lowers the referral score faster than random'}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', nargs='+', default=['legacy', 'R5_768_s17'])
    parser.add_argument('--datasets', nargs='+', default=['idrid', 'ddr'])
    parser.add_argument('--exclusions', type=Path, default=Path('../results/model_improvement/ddr_lesion_test_exclusions_v1.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'tier': 'TX (post hoc; lesion masks of exposed test sets)'}
    for model in args.models:
        for dataset in args.datasets:
            path = args.output / f'{model}_{dataset}.csv'
            frame = pd.read_csv(path) if path.exists() else run(model, dataset, args.exclusions, args.limit, args.seed)
            frame.to_csv(path, index=False)
            result[f'{model}/{dataset}'] = summarise(frame, args.seed)
            print(model, dataset, json.dumps(result[f'{model}/{dataset}'])[:400], flush=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
