#!/usr/bin/env python3
"""Evidence-exposure registry and guards.

``results/benchmarks/exposure_registry.json`` records, for every cohort, whether its labels
were used for training, read in a locked evaluation (exposed), are still reserved, or are
planned. This module is the only code that interprets it.

Guards:
* ``assert_fresh_allowed`` -- a lock may only freeze cohorts that are reserved, locked or
  planned (never ``exposed`` or ``training``) unless the caller declares a development run.
* ``is_reserved_label_path`` / ``assert_label_read_allowed`` -- reserved label files may only
  be read after a valid lock exists for that cohort and before it has been evaluated.
* ``record_locked`` / ``record_exposure`` -- keep the registry in step with locks and reports.

Statuses: training, exposed, reserved, locked, planned.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from utils.artifacts import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / 'results/benchmarks/exposure_registry.json'
NOT_FRESH = ('exposed', 'training')
LABEL_READABLE_BEFORE_LOCK = ('exposed', 'training')


class ExposureError(RuntimeError):
    """A cohort would be misused as fresh evidence, or a reserved label read without a lock."""


def load_registry(path: Path | None = None) -> dict:
    path = Path(path or REGISTRY_PATH)
    if not path.is_file():
        raise ExposureError(f'Exposure registry missing: {path}')
    registry = json.loads(path.read_text(encoding='utf-8'))
    if registry.get('schema_version') != 1 or 'cohorts' not in registry:
        raise ExposureError('Unsupported exposure registry schema')
    return registry


def entry(cohort: str) -> dict:
    cohorts = load_registry()['cohorts']
    if cohort not in cohorts:
        raise ExposureError(f'Cohort {cohort!r} is not in the exposure registry; register it before locking or reading it')
    return cohorts[cohort]


def status(cohort: str) -> str:
    return entry(cohort)['status']


def assert_fresh_allowed(cohort: str, development: bool = False) -> None:
    """Refuse to freeze a cohort whose labels are already exposed or used for training."""
    current = status(cohort)
    if current in NOT_FRESH and not development:
        raise ExposureError(
            f'{cohort!r} is {current}: it cannot be locked as a fresh test. '
            'Pass development=True (--development) to freeze it as a non-fresh development run.')


def _resolved(path: str | Path) -> Path:
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def reserved_label_paths() -> list[tuple[str, Path]]:
    """(cohort, path) for every label path of a cohort whose labels are not yet exposed."""
    out = []
    for name, item in load_registry()['cohorts'].items():
        if item['status'] in ('reserved', 'locked'):
            out.extend((name, _resolved(p)) for p in item.get('label_paths', []))
    return out


def is_reserved_label_path(path: str | Path) -> str | None:
    """Name of the reserved cohort the path belongs to (itself or inside a reserved path), else None."""
    target = _resolved(path)
    for name, reserved in reserved_label_paths():
        if target == reserved or reserved in target.parents:
            return name
    return None


def assert_not_reserved_path(path: str | Path) -> None:
    name = is_reserved_label_path(path)
    if name:
        raise ExposureError(f'{path} holds reserved labels of {name!r}; use assert_label_read_allowed with its lock')


def assert_label_read_allowed(cohort: str, lock_dir: str | Path) -> None:
    """Reserved labels may be read once: after a lock listing the cohort exists, before any report."""
    lock_dir = Path(lock_dir)
    lock_path = lock_dir / 'lock.json'
    if not lock_path.is_file():
        raise ExposureError(f'No lock at {lock_path}; reserved labels of {cohort!r} cannot be read')
    lock = json.loads(lock_path.read_text(encoding='utf-8'))
    if cohort not in lock.get('tests', {}):
        raise ExposureError(f'{lock_path} does not list cohort {cohort!r}')
    if lock.get('labels_read') is not False:
        raise ExposureError(f'{lock_path} does not record labels_read = false')
    if (lock_dir / 'report.json').exists():
        raise ExposureError(f'{lock_dir} was already evaluated; labels of {cohort!r} are exposed')
    if status(cohort) in ('exposed',):
        raise ExposureError(f'{cohort!r} is already exposed')


def _update(cohort: str, **fields) -> None:
    registry = load_registry()
    if cohort not in registry['cohorts']:
        raise ExposureError(f'Cohort {cohort!r} is not in the exposure registry')
    registry['cohorts'][cohort].update(fields)
    registry['updated'] = date.today().isoformat()
    atomic_json(REGISTRY_PATH, registry)


def _relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def record_locked(cohort: str, lock_path: str | Path) -> None:
    if status(cohort) in ('reserved', 'planned', 'locked'):
        _update(cohort, status='locked', lock=_relative(lock_path))


def record_exposure(cohort: str, lock_path: str | Path, report_path: str | Path) -> None:
    if status(cohort) != 'exposed':
        _update(cohort, status='exposed', exposed_on=date.today().isoformat(),
                lock=_relative(lock_path), report=_relative(report_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list', help='list every cohort and its status')
    one = sub.add_parser('status', help='status of one cohort')
    one.add_argument('cohort')
    check = sub.add_parser('verify', help='check that locks, reports and markers named by the registry exist')
    check.add_argument('--strict', action='store_true', help='fail on missing local files (default: report them)')
    args = parser.parse_args()
    if args.command == 'list':
        for name, item in load_registry()['cohorts'].items():
            print(f"{name:32s} {item['status']:10s} {item.get('kind', '')}")
    elif args.command == 'status':
        print(status(args.cohort))
    else:
        problems = []
        for name, item in load_registry()['cohorts'].items():
            for key in ('lock', 'report', 'marker'):
                if item.get(key) and not _resolved(item[key]).exists():
                    problems.append(f'{name}: {key} {item[key]} not found')
            if item['status'] == 'exposed' and not (item.get('lock') or item.get('marker') or item.get('notes')):
                problems.append(f'{name}: exposed without any evidence pointer')
        print('\n'.join(problems) or 'registry verified')
        if problems and args.strict:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
