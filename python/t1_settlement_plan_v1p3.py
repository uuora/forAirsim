"""Freeze actual-clear reserve under the original target-source deadline."""
import argparse,hashlib,json
from pathlib import Path
from t1_settlement_plan_v1p2 import validate_plan as validate_prior

ARCHITECTURE='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P3'


def validate_plan(plan):
    if plan.get('control_architecture')!=ARCHITECTURE or plan.get('target_dispatch_clear_budget')!=dict(
            actual_clear_reserve_s=.12,observation_end_deadline_unchanged=True,shorten_or_withdraw_target=True):
        raise ValueError('FROZEN_ACTUAL_CLEAR_ADMISSION_RESERVE_REQUIRED')
    return dict(validate_prior(dict(plan,control_architecture='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P2')),
        status='PASS_ACTUAL_CLEAR_RESERVE_PLAN',architecture=ARCHITECTURE)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError('NEW_PREFLIGHT_RECEIPT_REQUIRED')
    result=dict(plan_path=str(args.plan.resolve()),plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),simulator_calls=False,new_flights=0)
    try:result.update(validate_plan(json.loads(args.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError,TypeError,KeyError) as error:result.update(status='FAIL_ACTUAL_CLEAR_RESERVE_PLAN',error=str(error))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(result['status'])
    raise SystemExit(0 if result['status'].startswith('PASS') else 2)
