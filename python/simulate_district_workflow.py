#!/usr/bin/env python3
"""Independent patient-level reference for the M5 SimEvents digital twin.

The implementation deliberately does not call MATLAB.  It uses deterministic
FCFS queue equations and seeded routing draws so the SimEvents scenario sweep
can be checked against a second implementation.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANNUAL_SECONDS = 250 * 8 * 3600


@dataclass(frozen=True)
class Scenario:
    bandwidth_mbps: float
    clinicians: int
    quality_reject_rate: float
    total_patients: int = 100_000
    acquisition_seconds: float = 45.0
    recapture_seconds: float = 120.0
    ai_seconds: float = 3.5
    review_seconds: float = 30.0
    refer_rate: float = 0.08
    uncertain_rate: float = 0.0123
    escalation_threshold_seconds: float = 24 * 3600


def _single_server(arrivals: np.ndarray, service_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return FCFS start and finish times for one deterministic server."""
    arrivals = np.asarray(arrivals, dtype=float)
    starts = np.empty(arrivals.size, dtype=float)
    finishes = np.empty(arrivals.size, dtype=float)
    available = 0.0
    for index, arrival in enumerate(arrivals):
        start = max(float(arrival), available)
        finish = start + service_seconds
        starts[index] = start
        finishes[index] = finish
        available = finish
    return starts, finishes


def _multi_server(
    arrivals: np.ndarray, service_seconds: float, servers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return FCFS start and finish times for identical parallel servers."""
    if servers < 1:
        raise ValueError('servers must be at least one')
    arrivals = np.asarray(arrivals, dtype=float)
    starts = np.empty(arrivals.size, dtype=float)
    finishes = np.empty(arrivals.size, dtype=float)
    available = [0.0] * servers
    heapq.heapify(available)
    for index, arrival in enumerate(arrivals):
        server_available = heapq.heappop(available)
        start = max(float(arrival), server_available)
        finish = start + service_seconds
        starts[index] = start
        finishes[index] = finish
        heapq.heappush(available, finish)
    return starts, finishes


def _max_waiting(arrivals: np.ndarray, starts: np.ndarray) -> int:
    """Maximum number waiting after arrivals and simultaneous starts settle."""
    if arrivals.size == 0:
        return 0
    arrivals = np.sort(np.asarray(arrivals, dtype=float))
    starts = np.sort(np.asarray(starts, dtype=float))
    arrived = np.arange(1, arrivals.size + 1)
    started = np.searchsorted(starts, arrivals, side='right')
    return int(np.maximum(arrived - started, 0).max(initial=0))


def _utilization(
    starts: np.ndarray, finishes: np.ndarray, horizon: float, servers: int = 1,
) -> float:
    busy = np.clip(np.minimum(finishes, horizon) - starts, 0.0, None).sum()
    return float(busy / (horizon * servers))


def simulate_scenario(scenario: Scenario, seed: int) -> dict:
    """Simulate one annual scenario and return patient-level queue metrics."""
    if scenario.total_patients < 1:
        raise ValueError('total_patients must be positive')
    if scenario.bandwidth_mbps <= 0:
        raise ValueError('bandwidth_mbps must be positive')
    if not 0 <= scenario.quality_reject_rate <= 1:
        raise ValueError('quality_reject_rate must be between zero and one')
    if scenario.refer_rate + scenario.uncertain_rate > 1:
        raise ValueError('refer_rate + uncertain_rate cannot exceed one')

    # SimEvents generates one entity at t=0 and another at the StopTime.
    arrivals = np.linspace(
        0.0, ANNUAL_SECONDS, scenario.total_patients + 1, dtype=float)
    expected_arrivals = int(arrivals.size)
    rng = np.random.default_rng(seed)

    acquisition_start, acquisition_finish = _single_server(
        arrivals, scenario.acquisition_seconds)
    acquisition_reached = acquisition_finish <= ANNUAL_SECONDS
    acquired_finish = acquisition_finish[acquisition_reached]
    quality_rejected = rng.random(acquired_finish.size) < scenario.quality_reject_rate

    recapture_arrivals = acquired_finish[quality_rejected]
    recapture_start, recapture_finish = _single_server(
        recapture_arrivals, scenario.recapture_seconds)
    recaptures = int((recapture_finish <= ANNUAL_SECONDS).sum())

    upload_arrivals = acquired_finish[~quality_rejected]
    upload_seconds = 2 * 3.5 * 8 / scenario.bandwidth_mbps
    upload_start, upload_finish = _single_server(upload_arrivals, upload_seconds)
    upload_reached = upload_finish <= ANNUAL_SECONDS

    ai_arrivals = upload_finish[upload_reached]
    ai_start, ai_finish = _single_server(ai_arrivals, scenario.ai_seconds)
    ai_completed = ai_finish <= ANNUAL_SECONDS
    triage_times = ai_finish[ai_completed]
    routing = rng.random(triage_times.size)
    auto_clear_rate = 1 - scenario.refer_rate - scenario.uncertain_rate
    auto_mask = routing < auto_clear_rate
    refer_mask = (routing >= auto_clear_rate) & (
        routing < auto_clear_rate + scenario.refer_rate)
    uncertain_mask = ~(auto_mask | refer_mask)
    auto_cleared = int(auto_mask.sum())

    review_arrivals = triage_times[refer_mask | uncertain_mask]
    review_start, review_finish = _multi_server(
        review_arrivals, scenario.review_seconds, scenario.clinicians)
    review_started = review_start <= ANNUAL_SECONDS
    review_completed = review_finish <= ANNUAL_SECONDS
    reviewed = int(review_completed.sum())
    review_waits = review_start[review_started] - review_arrivals[review_started]
    censored_wait = ANNUAL_SECONDS - review_arrivals[~review_started]
    escalated_started = review_waits > scenario.escalation_threshold_seconds
    escalated_censored = censored_wait > scenario.escalation_threshold_seconds
    escalations = int(escalated_started.sum() + escalated_censored.sum())

    completed = auto_cleared + reviewed + recaptures
    expected_review_demand = expected_arrivals * (
        1 - scenario.quality_reject_rate
    ) * (scenario.refer_rate + scenario.uncertain_rate)
    required_clinicians = max(1, math.ceil(
        expected_review_demand * scenario.review_seconds
        / (ANNUAL_SECONDS * 0.80)
    ))

    result = {
        **asdict(scenario),
        'seed': seed,
        'expected_arrivals': expected_arrivals,
        'upload_seconds_per_patient': upload_seconds,
        'auto_cleared': auto_cleared,
        'review_routed': int(review_arrivals.size),
        'reviewed': reviewed,
        'recaptures': recaptures,
        'completed': completed,
        'unfinished_at_stop': expected_arrivals - completed,
        'completion_rate': completed / expected_arrivals,
        'review_rate': reviewed / max(completed, 1),
        'recapture_rate_observed': recaptures / max(completed, 1),
        'clinician_utilization': _utilization(
            review_start, review_finish, ANNUAL_SECONDS, scenario.clinicians),
        'acquisition_utilization': _utilization(
            acquisition_start, acquisition_finish, ANNUAL_SECONDS),
        'upload_utilization': _utilization(
            upload_start, upload_finish, ANNUAL_SECONDS),
        'ai_utilization': _utilization(ai_start, ai_finish, ANNUAL_SECONDS),
        'average_review_wait_seconds': (
            float(review_waits.mean()) if review_waits.size else 0.0),
        'p50_review_wait_seconds': (
            float(np.percentile(review_waits, 50)) if review_waits.size else 0.0),
        'p90_review_wait_seconds': (
            float(np.percentile(review_waits, 90)) if review_waits.size else 0.0),
        'p95_review_wait_seconds': (
            float(np.percentile(review_waits, 95)) if review_waits.size else 0.0),
        'max_acquisition_queue': _max_waiting(arrivals, acquisition_start),
        'max_upload_queue': _max_waiting(upload_arrivals, upload_start),
        'max_ai_queue': _max_waiting(ai_arrivals, ai_start),
        'max_review_queue': _max_waiting(review_arrivals, review_start),
        'review_escalations_over_24h': escalations,
        'required_review_clinicians_at_80pct_utilization': required_clinicians,
        'throughput_patients_per_hour': completed / (250 * 8),
    }
    return result


def run_sweep(total_patients: int = 100_000) -> list[dict]:
    """Run the same 18 scenarios as the checked-in SimEvents sweep."""
    rows = []
    scenario_id = 0
    for bandwidth in (0.25, 1.0, 5.0):
        for clinicians in (1, 2):
            for reject_rate in (0.05, 0.10, 0.20):
                scenario_id += 1
                scenario = Scenario(
                    bandwidth_mbps=bandwidth,
                    clinicians=clinicians,
                    quality_reject_rate=reject_rate,
                    total_patients=total_patients,
                )
                row = simulate_scenario(scenario, seed=4200 + scenario_id)
                row['scenario_id'] = scenario_id
                rows.append(row)
                print(
                    f"Python M5 {scenario_id:02d}/18: {bandwidth:.2f} Mbps, "
                    f"{clinicians} clinician(s), reject {100*reject_rate:.0f}% -> "
                    f"completion {row['completion_rate']:.3f}, "
                    f"max upload queue {row['max_upload_queue']}"
                )
    return rows


def compare_with_simulink(
    python_rows: list[dict], simulink_csv: Path, tolerance: float = 0.015,
) -> dict:
    """Compare routing/completion rates without assuming identical RNG streams."""
    with simulink_csv.open(newline='', encoding='utf-8-sig') as handle:
        matlab_rows = {int(row['scenario_id']): row for row in csv.DictReader(handle)}
    comparisons = []
    rate_fields = ('completion_rate', 'review_rate', 'recapture_rate_observed')
    for python_row in python_rows:
        scenario_id = int(python_row['scenario_id'])
        matlab_row = matlab_rows.get(scenario_id)
        if matlab_row is None:
            raise ValueError(f'SimEvents result missing scenario {scenario_id}')
        comparison = {'scenario_id': scenario_id}
        passed = True
        for field in rate_fields:
            error = abs(float(python_row[field]) - float(matlab_row[field]))
            comparison[f'{field}_absolute_error'] = error
            passed &= error <= tolerance
        comparison['passed'] = bool(passed)
        comparisons.append(comparison)
    return {
        'tolerance_absolute_rate': tolerance,
        'scenarios_compared': len(comparisons),
        'scenarios_passed': sum(row['passed'] for row in comparisons),
        'all_passed': all(row['passed'] for row in comparisons),
        'comparisons': comparisons,
        'interpretation': (
            'Independent implementations use different random-number streams; '
            'the gate compares annual rates, not patient identity.'
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--total-patients', type=int, default=100_000)
    parser.add_argument(
        '--output', type=Path,
        default=ROOT / 'results/verification_2026-09-18/m5_python_reference')
    parser.add_argument('--simulink-csv', type=Path)
    parser.add_argument('--comparison-tolerance', type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_sweep(args.total_patients)
    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / 'python_scenario_sweep.csv', rows)
    payload = {
        'schema_version': 1,
        'implementation': 'independent Python patient-level deterministic-FCFS reference',
        'scenario_count': len(rows),
        'results': rows,
        'limitations': [
            'Routing probabilities are planning assumptions pending external validation.',
            'Review escalation is reported at a 24-hour threshold; it is not a clinical SLA.',
            'Continuous annual operating time abstracts away nights, weekends, and outages.',
        ],
    }
    if args.simulink_csv:
        payload['simulink_cross_check'] = compare_with_simulink(
            rows, args.simulink_csv, args.comparison_tolerance)
    with (args.output / 'python_scenario_sweep.json').open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write('\n')
    if args.simulink_csv and not payload['simulink_cross_check']['all_passed']:
        raise SystemExit('SimEvents/Python cross-check exceeded the configured tolerance')


if __name__ == '__main__':
    main()
