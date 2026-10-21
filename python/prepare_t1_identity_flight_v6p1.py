"""Freeze idle-zero scheduling validation at unchanged motion budgets."""
import argparse
from pathlib import Path
from prepare_t1_identity_flight_v5p1 import prepare_async as prepare_v5p1
from prepare_t1_identity_flight import ROOT, digest, save, copy_verified, phase_grid

SOURCES = (
    'python/t1_async_control_v1p1.py', 'python/test_t1_async_control_v1p1.py',
    'python/test_t1_async_control.py', 'python/test_t1_identity_motion_policy.py',
    'python/t1_identity_flight_v6.py', 'python/prepare_t1_identity_flight_v6.py',
    'scripts/run-t1-identity-flight-v6.ps1',
    'python/t1_motion_plan_contract_v1.py', 'python/test_t1_motion_plan_contract.py',
    'python/t1_identity_flight_v6p1.py', 'python/prepare_t1_identity_flight_v6p1.py',
    'scripts/run-t1-identity-flight-v6p1.ps1',
)


def prepare_async(output, mode, session_id, device='cuda'):
    if any(not (ROOT / name).is_file() for name in SOURCES):
        raise ValueError('V6P1_DECLARED_SOURCE_MISSING')
    plan = prepare_v5p1(output, mode, session_id, device, False)
    output = Path(output).resolve()
    for name in SOURCES:
        file = ROOT / name
        sha = digest(file)
        snapshot = output / 'source_bundle' / name
        copy_verified(file, snapshot, sha)
        plan['source_inventory'].append(dict(relative_path=name, path=str(file),
            sha256=sha, snapshot_path=str(snapshot)))
        plan['source_sha256'][name] = sha
    duration = 8. if mode == 'zero' else 20.
    phases = [dict(phase_id='visible', duration_s=duration)]
    measured_slots = phase_grid(phases, .4)
    plan.update(control_architecture='INDEPENDENT_CONTROL_IDLE_ZERO_SUPPRESSION_V6P1',
        phases=phases, planned_duration_s=duration, planned_slots=measured_slots,
        flight_scope='One continuously-visible fixed-height development run; initialization and takeoff/landing separate',
        idle_zero_scheduling=dict(refresh_s=.30, monitor_period_s=.05,
            omit_only='actual prior zero plus fresh own-state sampled stopping',
            actual_command_clock_preserved=True, maximum_previous_command_interval_s=.4,
            moving_protection_expiry_and_close='actual braking and explicit zero retained'),
        motion_scale_validation=dict(scope='DEVELOPMENT_ENDPOINT_DISPLACEMENT_NOT_FORMAL_TRAJECTORY_EFFICACY',
            continuous_visible_duration_s=duration, planned_measured_slots=len(measured_slots),
            minimum_endpoint_net_xy_displacement_m=.020 if mode=='short' else None,
            maximum_endpoint_drift_m=.005 if mode=='zero' else None,
            minimum_visible_target_mask_pixels=500,
            zero_gate_requires_no_nonzero_target_commands=(mode=='zero'),
            endpoint='explicit measured start and end own-state XY; takeoff/landing excluded',
            qualification='independent frozen-input analysis; original runner status is retained separately',
            first_last_observation_window_s=1.,
            response_diagnostics='target integral direction projection; displacement and speed outside active leases; no extra post-hoc success threshold',
            all_started_and_missing='retain all planned measured and acquisition slots'))
    from t1_motion_plan_contract_v1 import validate_plan
    validate_plan(plan)
    save(output / 'plan.json',plan)
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('zero','short'),required=True)
    parser.add_argument('--session-id',required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    plan=prepare_async(args.output,args.mode,args.session_id,args.device)
    print(plan['status'],plan['control_architecture'],len(plan['planned_slots']))
