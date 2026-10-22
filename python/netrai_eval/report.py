"""Evidence-tier-stamped experiment reports.

Every reported result carries one of five tiers so development, validation, exposed and fresh
evidence can never be mixed silently:
  T0 training        data the model was fitted on
  TD development     patient-level holdouts / validation used for selection
  TL leave-dataset-out  a whole source withheld from training (not fresh evidence)
  TX exposed         previously locked and read cohorts; post hoc description only
  TF fresh           never-read labels evaluated once under a lock
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from utils.artifacts import atomic_json

SCHEMA_VERSION = 1
TIERS = {
    'T0': 'Training data: not evidence of generalization.',
    'TD': 'Development/selection data: used to choose recipes, epochs and policies; optimistic by construction.',
    'TL': 'Leave-dataset-out estimate on previously exposed data: informative about cross-domain transfer, not fresh independent evidence.',
    'TX': 'Previously exposed test cohort: post hoc description only; never used for selection and never a fresh test.',
    'TF': 'Fresh independent test: labels read once under a lock with the model and policy frozen.',
}
REQUIRED = ('schema_version', 'tier', 'tier_statement', 'cohort', 'model', 'created_utc', 'n_images', 'n_clusters',
            'metrics', 'limitations')


def build_report(*, tier: str, cohort: str, model: str, metrics: dict, n_images: int, n_clusters: int,
                 limitations: list[str], lock_sha256: str | None = None, experiment: str | None = None,
                 notes: str | None = None) -> dict:
    if tier not in TIERS:
        raise ValueError(f'Unknown evidence tier {tier!r}; expected one of {sorted(TIERS)}')
    if tier == 'TF' and not lock_sha256:
        raise ValueError('A tier-TF (fresh) report must cite the SHA-256 of its lock')
    if tier != 'TF' and lock_sha256:
        raise ValueError(f'Only tier-TF reports cite a lock; this one is {tier}')
    report = {'schema_version': SCHEMA_VERSION, 'tier': tier, 'tier_statement': TIERS[tier], 'cohort': cohort,
              'model': model, 'created_utc': datetime.now(timezone.utc).isoformat(), 'n_images': int(n_images),
              'n_clusters': int(n_clusters), 'metrics': metrics, 'limitations': list(limitations)}
    if lock_sha256:
        report['lock_sha256'] = lock_sha256
    if experiment:
        report['experiment'] = experiment
    if notes:
        report['notes'] = notes
    validate_report(report)
    return report


def validate_report(report: dict) -> None:
    missing = [key for key in REQUIRED if key not in report]
    if missing:
        raise ValueError(f'Report is missing required fields: {missing}')
    if report['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported report schema version')
    if report['tier'] not in TIERS:
        raise ValueError(f"Unknown evidence tier {report['tier']!r}")
    if report['tier_statement'] != TIERS[report['tier']]:
        raise ValueError('tier_statement does not match the tier')
    if report['tier'] == 'TF' and not report.get('lock_sha256'):
        raise ValueError('A tier-TF report must cite its lock')
    if not report['limitations']:
        raise ValueError('A report must state at least one limitation')


def write_report(path: Path, report: dict) -> None:
    """Validate and write atomically; never overwrites an existing report."""
    validate_report(report)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; reports are immutable')
    atomic_json(path, report)


def read_report(path: Path) -> dict:
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_report(report)
    return report
