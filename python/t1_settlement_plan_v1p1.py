"""Strict-stopped acquisition with continuous observations before measurement."""
import argparse
import hashlib
import json
from pathlib import Path

from t1_settlement_plan_v1 import validate_plan as validate_v7

ARCHITECTURE = 'INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7P1'
TRANSITION = dict(require='same current tracked RGB/surface frame plus strict stationary zero hold',
    blocking_wait_between_last_acquisition_and_first_measurement=False,
    repeated_zero='original policy validation; no new command or fabricated dispatch timestamp')


def validate_plan(plan):
    if plan.get('control_architecture') != ARCHITECTURE or plan.get('acquisition_transition_policy') != TRANSITION:
        raise ValueError('CONTINUOUS_OBSERVATION_STRICT_ACQUISITION_TRANSITION_REQUIRED')
    return dict(validate_v7(dict(plan,control_architecture='INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7')),
        status='PASS_CONTINUOUS_ACQUISITION_STRICT_SETTLEMENT_PLAN', architecture=ARCHITECTURE)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError('NEW_PREFLIGHT_RECEIPT_REQUIRED')
    result=dict(plan_path=str(args.plan.resolve()), plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        simulator_calls=False,new_flights=0)
    try: result.update(validate_plan(json.loads(args.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError,TypeError,KeyError) as error: result.update(status='FAIL_CONTINUOUS_ACQUISITION_PLAN',error=str(error))
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status']); raise SystemExit(0 if result['status'].startswith('PASS') else 2)
