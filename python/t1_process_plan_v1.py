"""V11 process ownership contract; prior source, lease and stop limits retained."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from t1_pipeline_plan_v1p2 import ARCHITECTURE as PRIOR, PERSISTENCE as PRIOR_PERSISTENCE, validate_plan as prior_validate

ARCHITECTURE = 'SPAWN_PROCESS_OWN_STATE_BOUNDED_CONTROL_V11'
PERSISTENCE = dict(PRIOR_PERSISTENCE,
    control_events='child-owned bounded journal; verified post-close replay preserves original event clocks')
PROCESS = dict(start_method='spawn', proposal_capacity=1, response_capacity=8,
    proposal_max_bytes=65536, response_max_bytes=131072, latest_state_max_bytes=16384,
    maximum_tickets=1024, maximum_active_runtime_s=380., bridge_period_s=.01,
    client_owner='child controller thread constructs client and handles every RPC/callback',
    parent_loss='spawn parent process handle; actual zero and strict sampled stop',
    overflow='explicit rejection or fatal closure; original child proof retained',
    rpc_handoff='child exited plus independent byte/state/stop audit; otherwise preserve simulator',
    latest_mailbox='nonblocking lock; skipped publications never replace child event evidence',
    source_clock_rebased=False, target_lease_renewed=False, hard_real_time_established=False)


def validate_plan(plan):
    if (plan.get('control_architecture') != ARCHITECTURE or plan.get('bounded_persistence') != PERSISTENCE
            or plan.get('process_control') != PROCESS):
        raise ValueError('EXACT_V11_PROCESS_CONTROL_PLAN_REQUIRED')
    gate=plan.get('process_offline_gate',{})
    gate_path=Path(gate.get('path',''))
    if (not gate_path.is_file() or hashlib.sha256(gate_path.read_bytes()).hexdigest()!=gate.get('sha256')
            or gate.get('qualification')!='PASS_OFFLINE_SPAWN_INJECTIONS_ONLY'):
        raise ValueError('V11_SOURCE_BOUND_OFFLINE_GATE_REQUIRED')
    source=json.loads(gate_path.read_text(encoding='utf-8'))
    names={'t1_process_control_v1.py','t1_process_control_factories_v1.py','t1_process_control_fixture_v1.py',
           'audit_t1_process_control_v1.py','t1_process_trace_v1.py','run_t1_process_offline_v1.py'}
    if (source.get('status')!='PASS_OFFLINE_SPAWN_INJECTIONS' or set(source.get('source_sha256',{}))!=names
            or source.get('source_unchanged_after_execution') is not True or len(source.get('conditions',[]))!=5):
        raise ValueError('V11_OFFLINE_GATE_CONTENT_MISMATCH')
    for name,value in source['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=value:
            raise ValueError('V11_OFFLINE_TESTED_SOURCE_CHANGED')
    for row in source['conditions']:
        if row['status']!='PASS':raise ValueError('V11_OFFLINE_CONDITION_FAILED')
        for key in ('final_receipt','audit'):
            if hashlib.sha256(Path(row[key]['path']).read_bytes()).hexdigest()!=row[key]['sha256']:
                raise ValueError('V11_OFFLINE_EVIDENCE_CHANGED')
    adapted = deepcopy(plan)
    adapted.update(control_architecture=PRIOR, bounded_persistence=dict(PRIOR_PERSISTENCE))
    adapted.pop('process_control')
    result = prior_validate(adapted)
    result.update(status='PASS_PREDECLARED_V11_PROCESS_PLAN', architecture=ARCHITECTURE,
        bounded_persistence=PERSISTENCE, process_control=PROCESS)
    return result


if __name__ == '__main__':
    import argparse, json, hashlib
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('NEW_PROCESS_PREFLIGHT_REQUIRED')
    r = dict(plan_path=str(a.plan.resolve()), plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
             simulator_calls=False, new_flights=0)
    try: r.update(validate_plan(json.loads(a.plan.read_text(encoding='utf-8-sig'))))
    except (ValueError, KeyError, TypeError) as e: r.update(status='FAIL_V11_PROCESS_PLAN', error=str(e))
    a.output.write_text(json.dumps(r, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(r['status']); raise SystemExit(0 if r['status'].startswith('PASS') else 2)
