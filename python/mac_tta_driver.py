#!/usr/bin/env python3
"""Drive ``tta_eval.py score`` on the Mac (MPS) in disk-safe chunks.

The Mac has little free disk, so for every LODO fold the held-out images are copied in chunks (rsync of the needed
768-px PNGs only), scored on the Mac, the logits copied back, and the temporary images removed (only files this script
created under ``~/netrai_b3``). Key-based SSH only. Results land in ``<out>/<fold>/logits_{identity,hflip}.npy`` and
``rows.csv`` (the same layout as a local ``tta_eval.py score``), so ``tta_eval.py compare`` works on them.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HOST = os.environ.get('NETRAI_REMOTE_HOST', '')          # user@host of the remote MPS machine (set locally, never committed)
KEY = str(Path(os.environ.get('NETRAI_REMOTE_KEY', '~/.ssh/id_ed25519')).expanduser())
REMOTE = '~/netrai_b3'
ROOTS = [Path('../results/model_improvement/train_export_768/images'),
         Path('../results/model_improvement/train_export_768_r4/images'),
         Path('../results/model_improvement/train_export_768_r5/images')]
SSH = ['ssh', '-o', 'BatchMode=yes', '-i', KEY, HOST]


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def remote(command: str):
    return run(SSH + [command])


def rsync(src, dst, files_from=None):
    cmd = ['rsync', '-a', '-e', f'ssh -o BatchMode=yes -i {KEY}']
    if files_from:
        cmd += ['--files-from', str(files_from)]
    run(cmd + [str(src), str(dst)])


def image_path(image_id: str) -> Path:
    for root in ROOTS:
        if (root / f'{image_id}.png').is_file():
            return root / f'{image_id}.png'
    raise FileNotFoundError(image_id)


def main(argv=None):
    if not HOST:
        raise SystemExit('Set NETRAI_REMOTE_HOST (user@host) and NETRAI_REMOTE_KEY before using the remote driver')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True, help='LODO run folders (best.pt + heldout_rows.csv)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--chunk', type=int, default=1000)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    remote(f'mkdir -p {REMOTE}/code {REMOTE}/imgs {REMOTE}/work')
    run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -i {KEY}', '--exclude=.venv/', '--exclude=tests/', '--exclude=__pycache__/',
         '--include=*/', '--include=*.py', '--exclude=*', './', f'{HOST}:{REMOTE}/code/'])
    for run_dir in args.runs:
        fold = run_dir.parent.name
        rows = pd.read_csv(run_dir / 'heldout_rows.csv')
        if args.limit:
            rows = rows.iloc[:args.limit]
        fold_out = args.out / fold
        fold_out.mkdir(parents=True, exist_ok=True)
        if (fold_out / 'logits_hflip.npy').exists():
            print(fold, 'already done', flush=True)
            continue
        remote(f'mkdir -p {REMOTE}/work/{fold}')
        run(['scp', '-q', '-i', KEY, str(run_dir / 'best.pt'), f'{HOST}:{REMOTE}/work/{fold}/best.pt'])
        parts = {'identity': [], 'hflip': []}
        for start in range(0, len(rows), args.chunk):
            chunk = rows.iloc[start:start + args.chunk].reset_index(drop=True)
            tag = f'{fold}_{start}'
            local_list = fold_out / f'files_{start}.txt'
            local_list.write_text('\n'.join(str(image_path(i).resolve()) for i in chunk.image_id))
            # rsync with absolute paths: copy each file flat into imgs/
            remote(f'rm -rf {REMOTE}/imgs/* ; mkdir -p {REMOTE}/imgs {REMOTE}/work/{tag}')
            listed = [image_path(i).resolve() for i in chunk.image_id]
            run(['rsync', '-a', '-e', f'ssh -o BatchMode=yes -i {KEY}', *[str(p) for p in listed], f'{HOST}:{REMOTE}/imgs/'])
            chunk_dir = fold_out / f'chunk_{start}'
            chunk_dir.mkdir(exist_ok=True)
            chunk.to_csv(chunk_dir / 'heldout_rows.csv', index=False)
            run(['scp', '-q', '-i', KEY, str(chunk_dir / 'heldout_rows.csv'), f'{HOST}:{REMOTE}/work/{tag}/heldout_rows.csv'])
            remote(f'cp {REMOTE}/work/{fold}/best.pt {REMOTE}/work/{tag}/best.pt')
            remote(f'cd {REMOTE}/code && ~/netrai_gpu/.venv/bin/python tta_eval.py score --run {REMOTE}/work/{tag} '
                   f'--roots {REMOTE}/imgs --output {REMOTE}/work/{tag}/out --device mps --workers 2')
            for view in parts:
                local = chunk_dir / f'logits_{view}.npy'
                run(['scp', '-q', '-i', KEY, f'{HOST}:{REMOTE}/work/{tag}/out/logits_{view}.npy', str(local)])
                parts[view].append(np.load(local))
            remote(f'rm -rf {REMOTE}/imgs/* {REMOTE}/work/{tag}')
            print(f'{fold}: chunk {start}-{start + len(chunk)} done', flush=True)
        for view, arrays in parts.items():
            np.save(fold_out / f'logits_{view}.npy', np.concatenate(arrays))
        rows.to_csv(fold_out / 'rows.csv', index=False)
        remote(f'rm -rf {REMOTE}/work/{fold}')


if __name__ == '__main__':
    main()
