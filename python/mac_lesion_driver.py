#!/usr/bin/env python3
"""Run ``lesion_features.py`` on the Mac (MPS) in disk-safe chunks of native images (key-based SSH, temp files only).

For each LODO fold's held-out rows the native image files are rsynced in chunks, summarised on the Mac, the per-image
``.npz`` summaries copied back, and the temporary images removed. Resumable: images with an existing local ``.npz`` are
skipped. Results: ``<out>/<image_id>.npz`` (all folds together).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

import lesion_features as lf
from mac_tta_driver import HOST, KEY, REMOTE, remote, run

SOURCE = {'p_messidor_s2': 'messidor', 'p_fdr26_s1': 'fdr26', 'p_odir_s1': 'odir', 'p_deepdrid_s1': 'deepdrid'}


def main(argv=None):
    if not HOST:
        raise SystemExit('Set NETRAI_REMOTE_HOST (user@host) and NETRAI_REMOTE_KEY before using the remote driver')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=300)
    parser.add_argument('--limit', type=int, default=0, help='per fold, for smoke tests')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    remote(f'mkdir -p {REMOTE}/code {REMOTE}/nat {REMOTE}/les')
    run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -i {KEY}', '--exclude=.venv/', '--exclude=tests/', '--exclude=__pycache__/',
         '--include=*/', '--include=*.py', '--exclude=*', './', f'{HOST}:{REMOTE}/code/'])
    run(['scp', '-q', '-i', KEY, str(args.weights), f'{HOST}:{REMOTE}/les/lesion.pt'])
    for run_dir in args.runs:
        rows = pd.read_csv(run_dir / 'heldout_rows.csv')
        if args.limit:
            rows = rows.iloc[:args.limit]
        todo = rows[[not (args.out / f'{i}.npz').exists() for i in rows.image_id]].reset_index(drop=True)
        for start in range(0, len(todo), args.chunk):
            chunk = todo.iloc[start:start + args.chunk].reset_index(drop=True)
            files = [lf.local_path(r.image_id, r.source).resolve() for r in chunk.itertuples()]
            remote(f'rm -rf {REMOTE}/nat {REMOTE}/les/out; mkdir -p {REMOTE}/nat')
            run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -i {KEY}', *[str(f) for f in files], f'{HOST}:{REMOTE}/nat/'])
            ids = args.out / '_chunk_ids.csv'
            chunk.to_csv(ids, index=False)
            run(['scp', '-q', '-i', KEY, str(ids), f'{HOST}:{REMOTE}/les/ids.csv'])
            remote(f'cd {REMOTE}/code && ~/netrai_gpu/.venv/bin/python lesion_features.py --weights {REMOTE}/les/lesion.pt '
                   f'--ids {REMOTE}/les/ids.csv --images-dir {REMOTE}/nat --output {REMOTE}/les/out --device mps')
            run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -i {KEY}', f'{HOST}:{REMOTE}/les/out/', f'{args.out}/'])
            remote(f'rm -rf {REMOTE}/nat {REMOTE}/les/out')
            print(f'{run_dir.parent.name}: {start + len(chunk)}/{len(todo)} done', flush=True)


if __name__ == '__main__':
    main()
