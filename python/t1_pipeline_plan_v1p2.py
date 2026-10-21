"""V10P2 measures dispatch boundaries and allows one bounded own-state retry."""
from copy import deepcopy
from t1_pipeline_plan_v1p1 import ARCHITECTURE as PRIOR, PERSISTENCE as PRIOR_PERSISTENCE, validate_plan as prior_validate

ARCHITECTURE = 'INDEPENDENT_CONTROL_OWN_STATE_BOUNDED_PERSISTENCE_V10P2'
PERSISTENCE = dict(PRIOR_PERSISTENCE, feedback_dispatch='actual local deadline clock; zero then at most one fresh own-state retry')
DISPATCH = dict(self_state_max_age_s=.12, maximum_fresh_retries_per_feedback_tick=1,
    expired_vector='never dispatch', recovery_first_command='actual unconditional own-state zero',
    fresh_query='same owning RPC thread; measured host receipt; protection checked',
    repeated_expiry='fatal with actual zero', target_lease_renewal=False,
    original_stop_deadline_preserved=True, original_source_image_clock_preserved=True,
    hard_real_time_established=False)


def validate_plan(plan):
    if (plan.get('control_architecture') != ARCHITECTURE or plan.get('bounded_persistence') != PERSISTENCE
            or plan.get('feedback_dispatch_recovery') != DISPATCH):
        raise ValueError('EXACT_V10P2_DISPATCH_RECOVERY_REQUIRED')
    adapted = deepcopy(plan)
    adapted.update(control_architecture=PRIOR, bounded_persistence=dict(PRIOR_PERSISTENCE))
    adapted.pop('feedback_dispatch_recovery')
    result = prior_validate(adapted)
    result.update(status='PASS_PREDECLARED_V10P2_DISPATCH_PLAN', architecture=ARCHITECTURE,
        bounded_persistence=PERSISTENCE, feedback_dispatch_recovery=DISPATCH)
    return result


if __name__ == '__main__':
    import argparse, json, hashlib
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('NEW_PREFLIGHT_REQUIRED')
    r = dict(plan_path=str(a.plan.resolve()), plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(), simulator_calls=False, new_flights=0)
    try: r.update(validate_plan(json.loads(a.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError, KeyError, TypeError) as e: r.update(status='FAIL_V10P2_PLAN', error=str(e))
    a.output.write_text(json.dumps(r, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(r['status']); raise SystemExit(0 if r['status'].startswith('PASS') else 2)
