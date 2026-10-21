"""Validate declared 20/60-second motion grids and actual-command receipt semantics."""
import argparse
import hashlib
import json
from pathlib import Path
from t1_settlement_plan_v1p1 import validate_plan as validate_prior, TRANSITION

ARCHITECTURE='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P2'


def validate_plan(plan):
    if plan.get('control_architecture')!=ARCHITECTURE:
        raise ValueError('V7P2_ACTUAL_COMMAND_RECEIPT_ARCHITECTURE_REQUIRED')
    duration=plan.get('planned_duration_s'); mode=plan.get('mode')
    if duration not in ((8.,) if mode=='ZERO_GATE' else (20.,60.)):
        raise ValueError('PREDECLARED_DURATION_REQUIRED')
    count=round(duration/.4)
    slots=[dict(slot_index=i,phase_id='visible',scheduled_start_s=round(i*.4,9),
        scheduled_end_s=round((i+1)*.4,9)) for i in range(count)]
    if (plan.get('planned_slots')!=slots or plan.get('phases')!=[dict(phase_id='visible',duration_s=duration)]
            or plan.get('motion_scale_validation',{}).get('planned_measured_slots')!=count
            or plan['motion_scale_validation'].get('continuous_visible_duration_s')!=duration):
        raise ValueError('COMPLETE_NEW_DURATION_GRID_REQUIRED')
    if plan.get('state_command_receipt_policy')!='actual last command before SELF_STATE query; later feedback cannot be relabeled as prior':
        raise ValueError('ACTUAL_PRE_QUERY_COMMAND_RECEIPT_POLICY_REQUIRED')
    normalized=dict(plan,control_architecture='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P1')
    if duration==60.:
        normalized.update(planned_duration_s=20.,phases=[dict(phase_id='visible',duration_s=20.)],planned_slots=slots[:50],
            motion_scale_validation=dict(plan['motion_scale_validation'],planned_measured_slots=50,continuous_visible_duration_s=20.))
    result=validate_prior(normalized)
    return dict(result,status='PASS_ACTUAL_COMMAND_RECEIPT_AND_DURATION_PLAN',architecture=ARCHITECTURE,
        expected_main_duration_s=duration,expected_main_slots=count,
        longer_duration_is_a_separate_condition=(duration==60.))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    if args.output.exists(): raise FileExistsError('NEW_PREFLIGHT_RECEIPT_REQUIRED')
    result=dict(plan_path=str(args.plan.resolve()),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        simulator_calls=False,new_flights=0)
    try: result.update(validate_plan(json.loads(args.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError,TypeError,KeyError) as error: result.update(status='FAIL_ACTUAL_COMMAND_RECEIPT_PLAN',error=str(error))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status']); raise SystemExit(0 if result['status'].startswith('PASS') else 2)
