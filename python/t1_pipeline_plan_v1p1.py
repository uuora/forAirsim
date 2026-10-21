"""V10P1 declines unavailable zero-hold proof; V10 motion limits are inherited."""
from copy import deepcopy
from t1_pipeline_plan_v1 import ARCHITECTURE as PRIOR,PERSISTENCE as PRIOR_PERSISTENCE,validate_plan as prior_validate
ARCHITECTURE='INDEPENDENT_CONTROL_OWN_STATE_BOUNDED_PERSISTENCE_V10P1'
PERSISTENCE=dict(PRIOR_PERSISTENCE,unavailable_zero_hold='refuse current proposal without motion or rewritten clocks')
def validate_plan(plan):
    if plan.get('control_architecture')!=ARCHITECTURE or plan.get('bounded_persistence')!=PERSISTENCE:
        raise ValueError('EXACT_V10P1_ZERO_HOLD_REFUSAL_REQUIRED')
    adapted=deepcopy(plan);adapted.update(control_architecture=PRIOR,bounded_persistence=dict(PRIOR_PERSISTENCE))
    result=prior_validate(adapted);result.update(status='PASS_PREDECLARED_V10P1_PIPELINE_PLAN',architecture=ARCHITECTURE,bounded_persistence=PERSISTENCE)
    return result
if __name__=='__main__':
    import argparse,json,hashlib
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('NEW_PREFLIGHT_REQUIRED')
    r=dict(plan_path=str(a.plan.resolve()),plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),simulator_calls=False,new_flights=0)
    try:r.update(validate_plan(json.loads(a.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError,KeyError,TypeError) as e:r.update(status='FAIL_V10P1_PLAN',error=str(e))
    a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(r['status']);raise SystemExit(0 if r['status'].startswith('PASS') else 2)
