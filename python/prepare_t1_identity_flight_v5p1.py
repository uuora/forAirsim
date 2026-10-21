"""Freeze one loss-aware review-handoff development flight."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v5 import prepare_async as prepare_v5
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified

SOURCES = (
    'python/t1_review_handoff_v1p1.py', 'python/test_t1_review_handoff_v1p1.py',
    'python/t1_review_handoff_observer_v1p1.py', 'python/t1_identity_flight_v5p1.py',
    'python/prepare_t1_identity_flight_v5p1.py', 'scripts/run-t1-identity-flight-v5p1.ps1',
)


def prepare_async(output, mode, session_id, device='cuda', active_stall=False):
    if any(not (ROOT / name).is_file() for name in SOURCES):
        raise ValueError('V5P1_DECLARED_SOURCE_MISSING')
    plan = prepare_v5(output, mode, session_id, device, active_stall)
    output = Path(output).resolve()
    for name in SOURCES:
        file = ROOT / name
        sha = digest(file)
        snapshot = output / 'source_bundle' / name
        copy_verified(file, snapshot, sha)
        plan['source_inventory'].append(dict(relative_path=name, path=str(file),
            sha256=sha, snapshot_path=str(snapshot)))
        plan['source_sha256'][name] = sha
    plan.update(control_architecture='INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5P1',
        recovery_submission_policy=dict(
            source='strictly newer actual readback after observed loss',
            submission='strictly after observed loss time',
            rejected_source='no cooldown, request serial or transport slot consumption',
            repeated_old_reply='diagnostic rejection retains original loss watermark',
            current_frame_confirmation='unchanged adoption plus two newer actual frames'),
        owned_engine_manifest='fresh verified PID, UTC start and executable path required')
    save(output / 'plan.json', plan)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('zero', 'short'), required=True)
    parser.add_argument('--session-id', required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--active-stall', action='store_true')
    args = parser.parse_args()
    plan = prepare_async(args.output, args.mode, args.session_id, args.device, args.active_stall)
    print(plan['status'], plan['control_architecture'], plan['active_lease_stall']['enabled'])
