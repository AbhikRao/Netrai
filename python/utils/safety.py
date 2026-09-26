"""Versioned screening-triage policy for NetrAI's shared-backbone grade and referral outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


TRIAGE_AUTO_CLEAR = 'auto_clear'
TRIAGE_REFER = 'refer'
TRIAGE_RECAPTURE = 'recapture'
TRIAGE_HUMAN_REVIEW = 'uncertain_human_review'


def load_safety_policy(path: str | Path) -> dict:
    """Load a strictly validated deployment safety policy."""
    policy_path = Path(path)
    if not policy_path.is_file():
        raise FileNotFoundError(f'safety policy not found: {policy_path}')
    try:
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid safety policy {policy_path}: {exc}') from exc
    if policy.get('schema_version') != 1:
        raise ValueError(f'unsupported safety-policy schema: {policy_path}')
    threshold = float(policy.get('referable_head_threshold', float('nan')))
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('referable_head_threshold must be strictly between 0 and 1')
    if policy.get('head_disagreement_action') != TRIAGE_HUMAN_REVIEW:
        raise ValueError('head disagreement must route to uncertain_human_review')
    policy['referable_head_threshold'] = threshold
    return policy


def assign_triage(
        *, quality_rejected: bool, grade_referable: bool,
        referable_head_positive: bool) -> str:
    """Apply conservative four-way research screening routing."""
    if quality_rejected:
        return TRIAGE_RECAPTURE
    if bool(grade_referable) != bool(referable_head_positive):
        return TRIAGE_HUMAN_REVIEW
    return TRIAGE_REFER if grade_referable else TRIAGE_AUTO_CLEAR
