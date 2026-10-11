#!/usr/bin/env python3
"""Generate a self-contained Kaggle notebook for the corrected grading study.

Only the source allowlist from build_gpu_bundle is embedded. No images,
checkpoints, credentials, or local paths enter the notebook.
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import textwrap
from zipfile import ZipFile, ZIP_DEFLATED

from build_gpu_bundle import FILES, ROOT


MARKER = 'python/build_kaggle_notebook.py'


def notebook_cell(kind: str, name: str, source: str) -> dict:
    cell = {'cell_type': kind, 'id': name, 'metadata': {},
            'source': source.strip('\n').splitlines(keepends=True)}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    return cell


def source_archive() -> bytes:
    stream = BytesIO()
    with ZipFile(stream, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            path = ROOT / name
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, name)
    return stream.getvalue()


def notebook_payload(archive: bytes) -> dict:
    import hashlib
    digest = hashlib.sha256(archive).hexdigest()
    encoded = base64.b64encode(archive).decode('ascii')
    chunks = '\n'.join(f'    {piece!r}' for piece in textwrap.wrap(encoded, 100))
    cells = [
        notebook_cell('markdown', 'overview', """
# Netrai corrected DR grading study — Kaggle GPU

This notebook contains the exact small source bundle. **Upload only this notebook**;
then attach the Kaggle **APTOS 2019 Blindness Detection** competition data with
**Add Input**. You do not need to upload your local 88 GB collection. In
**Settings**, enable one **GPU** and **Internet** for public ImageNet weights.

Run all cells once. By default this performs **one of five 30-epoch training
runs**, then stops safely. Run the training cell again to do the next run.
After every completed run, use **Save Version → Quick Save**, with its
advanced setting to **save output files**. Do not use **Save & Run All** to
preserve a live session: that starts a separate clean run.
If you start a fresh session, add the previous Notebook's output as an Input;
the setup cell below restores its `study` folder automatically. Do not change
the source, epoch count, or training settings between sessions.

The trainer prints an early per-epoch timing estimate; the first completed
run prints its elapsed time and a rough remaining-time estimate.
Only after all five runs does this notebook fit the APTOS calibration policy
and create a single download ZIP. After the fifth training-cell run, execute
the next code cell once. This is development evidence, **not** a
clinical validation or an automatic replacement for the current pipeline.
"""),
        notebook_cell('code', 'check-gpu-and-deps', """
from pathlib import Path
import subprocess, sys, time, json, csv, shutil
import torch

assert torch.cuda.is_available(), 'Enable GPU in Kaggle Settings, then restart the session.'
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'timm==1.0.29'], check=True)
assert torch.cuda.is_available(), 'CUDA became unavailable after package installation.'
"""),
        notebook_cell('code', 'embedded-source', f"""
# Embedded, hash-checked source. Do not edit this cell.
import base64, hashlib, io, zipfile
SOURCE_SHA256 = {digest!r}
SOURCE_B64 = (
{chunks}
)
source_bytes = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SHA256
work = Path('/kaggle/working/netrai')
work.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(source_bytes)) as package:
    assert all(not Path(item).is_absolute() and '..' not in Path(item).parts
               for item in package.namelist())
    package.extractall(work)
print('Netrai source ready:', work)
"""),
        notebook_cell('code', 'dataset-and-resume', """
input_root = Path('/kaggle/input')
aptos_candidates = []
for labels_csv in input_root.rglob('train.csv'):
    if not (labels_csv.parent / 'train_images').is_dir():
        continue
    with labels_csv.open(newline='') as stream:
        header = next(csv.reader(stream), [])
    if {'id_code', 'diagnosis'}.issubset(header):
        aptos_candidates.append(labels_csv.parent)
assert len(aptos_candidates) == 1, (
    'Attach exactly one APTOS 2019 competition input. Found: ' + str(aptos_candidates))
aptos = aptos_candidates[0]
print('APTOS input:', aptos)

study = work / 'study'
previous = [path.parent for path in input_root.rglob('study_protocol.json')]
if not (study / 'study_protocol.json').exists() and previous:
    assert len(previous) == 1, 'Attach only one previous Netrai Notebook output.'
    shutil.copytree(previous[0], study)
    print('Restored previous study:', previous[0])

manifest = work / 'manifests/aptos_development.csv'
if not manifest.exists():
    subprocess.run([sys.executable, str(work / 'python/prepare_training_manifest.py'),
        '--csv', str(aptos / 'train.csv'), '--image-dir', str(aptos / 'train_images'),
        '--output', str(manifest), '--conflicting-labels', 'quarantine'], check=True)
print('Development manifest:', manifest)
"""),
        notebook_cell('code', 'one-training-run', """
# Run this cell once per new model run. Five completed runs finish the study.
EPOCHS = 30
RUNS_THIS_INVOCATION = 1
started = time.monotonic()
subprocess.run([sys.executable, str(work / 'python/run_grading_study.py'),
    '--manifest', str(manifest), '--output', str(study),
    '--epochs', str(EPOCHS), '--batch-size', '4', '--workers', '2',
    '--max-new-runs', str(RUNS_THIS_INVOCATION)], check=True)
elapsed_hours = (time.monotonic() - started) / 3600
complete = []
for history_path in study.glob('*/history.json'):
    history = json.loads(history_path.read_text())
    if len(history) == EPOCHS and (history_path.parent / 'best.pt').is_file():
        complete.append(history_path.parent.name)
print(f'Finished runs: {len(complete)}/5 | This invocation: {elapsed_hours:.2f} hours')
if 0 < len(complete) < 5:
    print(f'Rough remaining GPU time if runs are similar: '
          f'{(5 - len(complete)) * elapsed_hours:.1f} hours. This is only an estimate.')
    print('Quick Save WITH OUTPUT FILES, then run this training cell again.')
else:
    print('Study selection ready:', (study / 'study_selection.json').is_file())
    print('Run the next code cell to fit the policy and create your download ZIP.')
"""),
        notebook_cell('code', 'fit-and-package-policy', """
# Run this cell after the fifth training run; Run All reaches it automatically.
import zipfile
selection_path = study / 'study_selection.json'
if selection_path.is_file():
    chosen = json.loads(selection_path.read_text())['selected_checkpoint']
    best = Path(chosen['output']) / 'best.pt'
    policy_dir = work / 'candidate_policy'
    if not (policy_dir / 'policy.json').is_file():
        subprocess.run([sys.executable, str(work / 'python/evaluate_grading_candidate.py'),
            'fit-policy', '--manifest', str(manifest), '--checkpoint', str(best),
            '--output', str(policy_dir), '--device', 'cuda', '--batch-size', '4'],
            check=True)
    handoff = work / 'Netrai_Grading_GPU_Results.zip'
    items = {
        'best.pt': best,
        'policy.json': policy_dir / 'policy.json',
        'calibration_summary.json': policy_dir / 'summary.json',
        'study_selection.json': selection_path,
        'study_protocol.json': study / 'study_protocol.json',
        'manifest_protocol.json': manifest.with_suffix('.protocol.json'),
    }
    with zipfile.ZipFile(handoff, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for name, path in items.items():
            assert path.is_file(), f'Missing required output: {path}'
            package.write(path, name)
    print('DOWNLOAD THIS FILE and send it to me:', handoff)
    print('APTOS calibration metrics:', json.loads(items['calibration_summary.json'].read_text())['metrics'])
else:
    print('Study incomplete. Run the training cell again; no policy has been fitted yet.')
"""),
        notebook_cell('markdown', 'interpretation', """
## What the outputs mean

The five runs compare corrected CE, focal, and ordinal training without using
the calibration labels for model selection. The chosen checkpoint is calibrated
on its separate APTOS calibration split. The ZIP is **not a clinical result**:
Netrai still needs a frozen local IDRiD comparison and, ultimately, a fresh
identity-reviewed external cohort. Send `Netrai_Grading_GPU_Results.zip` to the
Netrai project workspace for that next evaluation.
"""),
    ]
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'netrai_generated_by': MARKER,
            'netrai_source_sha256': digest,
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'Netrai_Kaggle_Grading_Study.ipynb')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        existing = json.loads(output.read_text(encoding='utf-8'))
        if existing.get('metadata', {}).get('netrai_generated_by') != MARKER:
            raise FileExistsError(f'Refusing to replace a non-generated notebook: {output}')
    payload = notebook_payload(source_archive())
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.netrai-notebook-', dir=output.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=1, ensure_ascii=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        os.chmod(output, 0o644)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f'Generated {output} ({output.stat().st_size} bytes); source SHA-256 '
          f'{payload["metadata"]["netrai_source_sha256"]}')


if __name__ == '__main__':
    main()
