"""Freeze strict stopping before flight with unchanged T1 identity and motion caps."""
import argparse
from dataclasses import asdict
from pathlib import Path

from prepare_t1_identity_flight_v6p1 import prepare_async as prepare_v6p1
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified
from t1_control_settlement import SettlementConfig
from t1_settlement_plan_v1 import ARCHITECTURE, TIMING_POLICY, validate_plan

SOURCES = ('python/t1_control_settlement.py', 'python/test_t1_control_settlement.py',
    'python/t1_feedback_flight.py', 'python/t1_async_control_v1p2.py',
    'python/test_t1_async_control_v1p2.py', 'python/t1_settlement_plan_v1.py',
    'python/test_t1_settlement_plan.py', 'python/t1_identity_flight_v7.py',
    'python/prepare_t1_identity_flight_v7.py', 'scripts/run-t1-identity-flight-v7.ps1',
    'python/analyze_t1_settlement.py', 'python/test_analyze_t1_settlement.py')


def prepare_async(output, mode, session_id, device='cuda'):
    if any(not (ROOT/name).is_file() for name in SOURCES):
        raise ValueError('V7_DECLARED_SOURCE_MISSING')
    plan = prepare_v6p1(output, mode, session_id, device)
    output = Path(output).resolve()
    for name in SOURCES:
        file = ROOT/name
        sha = digest(file)
        snapshot = output/'source_bundle'/name
        copy_verified(file, snapshot, sha)
        plan['source_inventory'].append(dict(relative_path=name, path=str(file),
            sha256=sha, snapshot_path=str(snapshot)))
        plan['source_sha256'][name] = sha
    plan.update(control_architecture=ARCHITECTURE,
        own_state_settlement=asdict(SettlementConfig()), completion_timing_policy=TIMING_POLICY,
        settled_endpoint_policy=dict(initial_wait_timeout_s=2.5, final_wait_timeout_s=2.5,
            endpoint='actual own state after strict sampled stop; observed tail retained'),
        comparison_scope='new feedback-stop version; old full-completion deadline and motion-scale threshold retained')
    validate_plan(plan)
    save(output/'plan.json', plan)
    return plan


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('zero', 'short'), required=True)
    p.add_argument('--session-id', required=True)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    plan = prepare_async(args.output, args.mode, args.session_id, args.device)
    print(plan['status'], plan['control_architecture'], len(plan['planned_slots']))
