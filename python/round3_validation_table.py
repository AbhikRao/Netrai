#!/usr/bin/env python3
"""Round-3 candidate selection table on the round-3 validation set (no fresh-test data).

Each candidate is one or more training runs (several = mean-logit ensemble). The
round-2 K3 reference is reassembled on the same rows from its saved validation
outputs (EyeQ selection, IDRiD test) and its saved Messidor-2 logits, so every
variant is compared on identical images. Selection value = mean over validation
sources of mean sensitivity at 80/85/90% specificity (as in the trainer).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_grading_adapter import scores, selection_value
from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MI = ROOT / 'results/model_improvement'


def run_outputs(run: Path):
    return pd.read_csv(run / 'validation_rows.csv'), np.load(run / 'best_validation_outputs.npy')


def k3_on(rows: pd.DataFrame) -> np.ndarray:
    """Round-2 K3 logits aligned to round-3 validation rows."""
    old_rows, old = run_outputs(MI / 'kaggle_output_v3/run')
    table = dict(zip(old_rows.image_id, old))
    inventory = pd.read_csv(MI / 'round2_eval/inventory_messidor2.csv')
    messidor = np.load(MI / 'round2_eval/logits_messidor2_K3_freeze4_ddr_s17.npy')
    table.update(zip('messidor_' + inventory.image_id.str.replace('.', '_', regex=False), messidor))
    missing = [i for i in rows.image_id if i not in table]
    if missing:
        raise ValueError(f'K3 lacks {len(missing)} validation rows, e.g. {missing[:3]}')
    return np.stack([table[i] for i in rows.image_id])


def summary(logits, rows):
    score, grade, source = scores(logits), rows.grade.to_numpy(), rows.source.to_numpy()
    per = {s: {'selection': selection_value(score[source == s], grade[source == s]),
               'auc': float(roc_auc_score(grade[source == s] >= 2, score[source == s]))} for s in np.unique(source)}
    return {'selection_value': float(np.mean([v['selection'] for v in per.values()])), 'per_source': per}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', nargs='+', metavar=('ID', 'RUN'), required=True,
                        help='ID followed by run directories, or the token K3 for the round-2 reference')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference_rows = None
    table = {}
    for name, *members in args.candidate:
        logits = []
        for member in members:
            if member == 'K3':
                if reference_rows is None:
                    raise ValueError('List a trained run before K3 so its rows define the validation set')
                logits.append(k3_on(reference_rows))
                continue
            rows, out = run_outputs(Path(member))
            if reference_rows is None:
                reference_rows = rows
            elif not rows.image_id.equals(reference_rows.image_id):
                raise ValueError(f'{member}: validation rows differ')
            logits.append(out)
        table[name] = {'members': members, **summary(np.mean(logits, axis=0), reference_rows)}
    atomic_json(args.output, {'schema_version': 1, 'status': 'round3_selection_validation_only',
                              'validation_images': int(len(reference_rows)), 'candidates': table})
    for name, v in table.items():
        print(f"{name:22s} {v['selection_value']:.4f} " + ' '.join(
            f"{s}:{p['selection']:.3f}/{p['auc']:.3f}" for s, p in v['per_source'].items()))


if __name__ == '__main__':
    main()
