"""V10 adds bounded persistence and clocks; all V9 flight limits are inherited."""
import argparse
from copy import deepcopy
import hashlib,json
from pathlib import Path
from t1_scaled_motion_plan_v1 import validate_plan as prior_validate

ARCHITECTURE='INDEPENDENT_CONTROL_OWN_STATE_BOUNDED_PERSISTENCE_V10'
PERSISTENCE=dict(max_inflight_frames=3,event_capacity=4096,initial_media_wait_s=.5,final_media_wait_s=2.,
    media_completion_scope='actual media, evaluation and immutable record flush; queue admission is not completion',
    historical_full_clock_preserved=True,source_clocks_and_command_limits_unchanged=True,
    stable_interval_zero_hold_receipt=True,python_flush_not_fsync=True)


def validate_plan(plan):
    if plan.get('control_architecture')!=ARCHITECTURE or plan.get('bounded_persistence')!=PERSISTENCE:
        raise ValueError('EXACT_V10_BOUNDED_PERSISTENCE_DECLARATION_REQUIRED')
    adapted=deepcopy(plan);adapted['control_architecture']='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V9'
    result=prior_validate(adapted)
    result.update(status='PASS_PREDECLARED_V10_PIPELINE_PLAN',architecture=ARCHITECTURE,bounded_persistence=PERSISTENCE,
        simulator_calls=False,new_flights=0)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('NEW_PIPELINE_PREFLIGHT_RECEIPT_REQUIRED')
    receipt=dict(plan_path=str(a.plan.resolve()),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),simulator_calls=False,new_flights=0)
    try:receipt.update(validate_plan(json.loads(a.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError,KeyError,TypeError) as error:receipt.update(status='FAIL_V10_PIPELINE_PLAN',error=str(error))
    a.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(receipt['status']);raise SystemExit(0 if receipt['status'].startswith('PASS') else 2)
