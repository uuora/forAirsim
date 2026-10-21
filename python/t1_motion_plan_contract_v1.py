"""Validate the complete motion-scale plan before starting a simulator.

No AirSim client, model, API configuration or simulator process is created.
The fixed 20 mm endpoint criterion and the original command budgets are kept.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ARCHITECTURE = 'INDEPENDENT_CONTROL_IDLE_ZERO_SUPPRESSION_V6P1'
DIAGNOSTICS = 'target integral direction projection; displacement and speed outside active leases; no extra post-hoc success threshold'


def grid(mode):
    if mode not in ('ZERO_GATE', 'SHORT_XY'):
        raise ValueError('MOTION_PLAN_MODE_INVALID')
    count = 20 if mode == 'ZERO_GATE' else 50
    return [dict(slot_index=i, phase_id='visible', scheduled_start_s=round(i*.4, 9),
                 scheduled_end_s=round((i+1)*.4, 9)) for i in range(count)]


def validate_plan(plan):
    mode = plan.get('mode')
    slots = grid(mode)
    duration = 8. if mode == 'ZERO_GATE' else 20.
    errors = []
    def check(ok, name):
        if not ok: errors.append(name)
    check(plan.get('control_architecture') == ARCHITECTURE, 'ARCHITECTURE')
    check(plan.get('status') == 'FROZEN_BEFORE_FLIGHT' and
          plan.get('description') == '带白色横条的红色气球', 'FROZEN_TARGET')
    check(plan.get('formal_efficacy_results') is False, 'DEVELOPMENT_SCOPE')
    check(plan.get('phases') == [dict(phase_id='visible', duration_s=duration)] and
          plan.get('planned_duration_s') == duration, 'CONTINUOUS_VISIBLE_DURATION')
    check(plan.get('planned_slots') == slots, 'FULL_PLANNED_SLOT_GRID')
    for key, wanted in dict(period_s=.4, world_z_ned_m=-3., command_speed_cap_m_s=.03,
                            command_lease_s=.12, self_monitor_period_s=.05).items():
        check(type(plan.get(key)) in (int, float) and math.isfinite(plan[key]) and
              plan[key] == wanted, 'BUDGET_' + key.upper())
    config = plan.get('motion_scale_validation', {})
    check(config.get('planned_measured_slots') == len(slots), 'DECLARED_SLOT_COUNT')
    check(config.get('continuous_visible_duration_s') == duration, 'DECLARED_DURATION')
    check(config.get('minimum_endpoint_net_xy_displacement_m') ==
          (None if mode == 'ZERO_GATE' else .020), 'ENDPOINT_MINIMUM')
    check(config.get('maximum_endpoint_drift_m') ==
          (.005 if mode == 'ZERO_GATE' else None), 'ENDPOINT_DRIFT')
    check(config.get('zero_gate_requires_no_nonzero_target_commands') is
          (mode == 'ZERO_GATE'), 'ZERO_COMMAND_RULE')
    check(config.get('minimum_visible_target_mask_pixels') == 500 and
          config.get('first_last_observation_window_s') == 1. and
          config.get('response_diagnostics') == DIAGNOSTICS, 'FROZEN_DIAGNOSTICS')
    idle = plan.get('idle_zero_scheduling', {})
    check(idle.get('refresh_s') == .30 and idle.get('monitor_period_s') == .05 and
          idle.get('maximum_previous_command_interval_s') == .4 and
          idle.get('actual_command_clock_preserved') is True, 'IDLE_ACTUAL_COMMAND_CLOCK')
    check(plan.get('active_lease_stall', {}).get('enabled') is False and
          not plan.get('injected_delays'), 'NO_INJECTIONS')
    initial = plan.get('initial_identity_acquisition', {})
    check(initial.get('maximum_duration_s') == 20. and initial.get('period_s') == .4
          and initial.get('maximum_planned_slots') == 50 and
          initial.get('desired_xy_m_s') == [0., 0.], 'SEPARATE_ZERO_ACQUISITION')
    if errors:
        raise ValueError('MOTION_PLAN_CONTRACT_MISMATCH:' + ','.join(errors))
    return dict(status='PASS_COMPLETE_MOTION_PLAN_CONTRACT', mode=mode,
                expected_main_slots=len(slots), expected_main_duration_s=duration,
                simulator_calls=False, new_flights=0)


def validate_file(path, output):
    path = Path(path).resolve()
    receipt = dict(plan_path=str(path), plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   simulator_calls=False, new_flights=0)
    try:
        receipt.update(validate_plan(json.loads(path.read_text(encoding='utf-8-sig'))))
    except (ValueError, KeyError, TypeError) as error:
        receipt.update(status='FAIL_MOTION_PLAN_CONTRACT', error=str(error))
    Path(output).write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate_file(args.plan, args.output)
    print(result['status'])
    raise SystemExit(0 if result['status'].startswith('PASS') else 2)
