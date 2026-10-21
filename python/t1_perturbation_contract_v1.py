"""Predeclared source perturbations with unchanged V7P3 execution limits."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from prepare_t1_identity_flight import phase_grid
from t1_settlement_plan_v1p3 import validate_plan as validate_execution

ARCHITECTURE = 'INDEPENDENT_CONTROL_SOURCE_PERTURBATIONS_V8'
CONDITIONS = ('baseline', 'delivery_delay', 'occlusion', 'glint')


def specification(condition, mode):
    if condition not in CONDITIONS or mode not in ('ZERO_GATE', 'SHORT_XY'):
        raise ValueError('UNKNOWN_PERTURBATION_CONDITION_OR_MODE')
    if mode == 'ZERO_GATE' and condition != 'baseline':
        raise ValueError('ZERO_GATE_MUST_PRECEDE_SOURCE_PERTURBATIONS')
    if mode == 'ZERO_GATE':
        phases = [dict(phase_id='visible', duration_s=8.)]
    elif condition == 'occlusion':
        phases = [dict(phase_id='visible', duration_s=20.),
                  dict(phase_id='hidden', duration_s=4.), dict(phase_id='return', duration_s=36.)]
    elif condition == 'glint':
        phases = [dict(phase_id='visible', duration_s=20.),
                  dict(phase_id='glint', duration_s=20.), dict(phase_id='return', duration_s=20.)]
    else:
        phases = [dict(phase_id='visible', duration_s=60.)]
    return dict(condition=condition, phases=phases,
        reply_delivery_fault=dict(enabled=condition == 'delivery_delay',
            durations_s=[3., 9.] if condition == 'delivery_delay' else [],
            trigger='first two actual requests whose original RGB receipt is at or after measurement epoch',
            boundary='AFTER_REAL_BACKEND_RETURN_BEFORE_POLICY_FUTURE_COMPLETION',
            service_reply_clock_and_content_unchanged=True, natural_service_delay=False),
        scene_fault=dict(occluder_phase='hidden' if condition == 'occlusion' else None,
            distractor_texture_phase='glint' if condition == 'glint' else None,
            texture_profile='small_white_patches_on_red' if condition == 'glint' else 'plain_red',
            target_pose_or_texture_changed=False, planner_receives_constructor_or_segmentation=False),
        interpretation='controlled development conditions; no formal matched efficacy estimate',
        all_planned_slots_retained=True, scale_threshold_m=.020 if mode == 'SHORT_XY' else None,
        scale_is_separate_from_fault_behavior=True)


def validate_plan(plan):
    if plan.get('control_architecture') != ARCHITECTURE:
        raise ValueError('V8_SOURCE_PERTURBATION_ARCHITECTURE_REQUIRED')
    declared = plan.get('source_perturbation', {})
    expected = specification(declared.get('condition'), plan.get('mode'))
    if declared != expected or plan.get('phases') != expected['phases']:
        raise ValueError('FROZEN_SOURCE_PERTURBATION_SPECIFICATION_REQUIRED')
    duration = sum(p['duration_s'] for p in expected['phases'])
    slots = phase_grid(expected['phases'], .4)
    scale = plan.get('motion_scale_validation', {})
    visible_duration = None if declared['condition'] == 'occlusion' else duration
    if (plan.get('planned_slots') != slots or plan.get('planned_duration_s') != duration
            or scale.get('planned_measured_slots') != len(slots)
            or scale.get('continuous_visible_duration_s') != visible_duration):
        raise ValueError('COMPLETE_DECLARED_PERTURBATION_GRID_REQUIRED')
    if plan.get('active_lease_stall', {}).get('enabled') is not False or plan.get('injected_perception_delays'):
        raise ValueError('NO_UNDECLARED_MAIN_THREAD_INJECTIONS_ALLOWED')
    # Reuse execution validation for limits only. Actual phases and fault
    # declarations above are checked independently and are never relabeled.
    normalized = deepcopy(plan)
    normalized.update(control_architecture='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P3',
        phases=[dict(phase_id='visible', duration_s=duration)],
        planned_slots=phase_grid([dict(phase_id='visible', duration_s=duration)], .4))
    normalized['motion_scale_validation']['continuous_visible_duration_s'] = duration
    validate_execution(normalized)
    return dict(status='PASS_FROZEN_SOURCE_PERTURBATION_PLAN', architecture=ARCHITECTURE,
        condition=declared['condition'], expected_main_slots=len(slots), expected_main_duration_s=duration,
        phases=expected['phases'], execution_limits_unchanged=True,
        normalization_scope='V7P3 execution limits only; actual phase grid and faults checked separately',
        simulator_calls=False, new_flights=0)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('NEW_PREFLIGHT_RECEIPT_REQUIRED')
    result=dict(plan_path=str(args.plan.resolve()),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest())
    try:result.update(validate_plan(json.loads(args.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError,TypeError,KeyError) as error:result.update(status='FAIL_FROZEN_SOURCE_PERTURBATION_PLAN',error=str(error))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result['status']);raise SystemExit(0 if result['status'].startswith('PASS') else 2)
