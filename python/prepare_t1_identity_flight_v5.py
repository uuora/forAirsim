"""Freeze background review handoff and optionally one active-dispatch stall."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v4 import prepare_async as prepare_v4
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified

SOURCES = (
    'python/t1_identity_flight_v5.py', 'python/prepare_t1_identity_flight_v5.py',
    'python/t1_review_handoff.py', 'python/test_t1_review_handoff.py',
    'python/t1_review_handoff_observer.py', 'python/t1_active_stall.py',
    'python/test_t1_active_stall.py', 'scripts/run-t1-identity-flight-v5.ps1',
)


def prepare_async(output, mode, session_id, device='cuda', active_stall=False):
    if active_stall and mode != 'short':
        raise ValueError('ACTIVE_STALL_REQUIRES_NONZERO_SHORT_PILOT')
    if any(not (ROOT / name).is_file() for name in SOURCES):
        raise ValueError('V5_DECLARED_SOURCE_MISSING')
    # The old scheduled injections are disabled in this independent plan.
    plan = prepare_v4(output, mode, session_id, device, 0.)
    output = Path(output).resolve()
    for name in SOURCES:
        file = ROOT / name; sha = digest(file); snapshot = output / 'source_bundle' / name
        copy_verified(file, snapshot, sha)
        plan['source_inventory'].append(dict(relative_path=name, path=str(file), sha256=sha, snapshot_path=str(snapshot)))
        plan['source_sha256'][name] = sha
    plan.update(control_architecture='INDEPENDENT_CONTROL_REVIEW_HANDOFF_V5',
        semantic_strategy='CONFIRM_NEW_CHAIN_IN_BACKGROUND_HANDOFF_ONLY_SAME_CURRENT_CANDIDATE',
        review_handoff=dict(prior_source_deadline_extended=False,
            confirmation='adoption plus two actual newer frames for each source',
            conflict='clear active output; fresh acquisition required',
            loss='invalidate pending; no loss-preceding source revival'),
        active_lease_stall=dict(enabled=bool(active_stall), duration_s=.65 if active_stall else 0.,
            maximum_injections=1, minimum_remaining_actual_dispatch_lease_s=.04,
            trigger='first eligible measured actual authorized nonzero target dispatch',
            trigger_inputs='control dispatch receipts only; no truth, phase or actor label',
            required='real event replay must show active nonzero overlap and independent explicit clear during main-thread stall',
            not_triggered='FAIL_PREDECLARED_ACTIVE_STALL_NOT_TRIGGERED'),
        figure_language='zh-CN', formal_efficacy_results=False)
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
