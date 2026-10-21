"""New strict-stop/two-clock plan preflight; historical V6P1 thresholds retained."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from t1_control_settlement import SettlementConfig
from t1_motion_plan_contract_v1 import validate_plan as validate_old

ARCHITECTURE = 'INDEPENDENT_CONTROL_OWN_STATE_FEEDBACK_V7'
TIMING_POLICY = dict(frame_processing_deadline_s=.4, maximum_stop_after_lease_s=2.,
    historical_full_clock='max(frame_processing_completed, measured_control_stop_completed)',
    historical_full_deadline_s=.4, clocks_reported_separately=True,
    source_time_or_lease_extended=False)


def validate_plan(plan):
    if plan.get('control_architecture') != ARCHITECTURE:
        raise ValueError('V7_SETTLEMENT_ARCHITECTURE_REQUIRED')
    result = validate_old(dict(plan, control_architecture='INDEPENDENT_CONTROL_IDLE_ZERO_SUPPRESSION_V6P1'))
    if plan.get('own_state_settlement') != asdict(SettlementConfig()):
        raise ValueError('FROZEN_STRICT_SETTLEMENT_CONFIG_REQUIRED')
    if plan.get('completion_timing_policy') != TIMING_POLICY:
        raise ValueError('FROZEN_SEPARATE_TIMING_CLOCKS_REQUIRED')
    if plan.get('settled_endpoint_policy') != dict(initial_wait_timeout_s=2.5,
            final_wait_timeout_s=2.5, endpoint='actual own state after strict sampled stop; observed tail retained'):
        raise ValueError('FROZEN_SETTLED_ENDPOINT_POLICY_REQUIRED')
    return dict(result, status='PASS_STRICT_SETTLEMENT_PLAN', architecture=ARCHITECTURE,
        source_time_or_lease_extended=False)


def validate_file(path, output):
    path = Path(path).resolve()
    result = dict(plan_path=str(path), plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        simulator_calls=False, new_flights=0)
    try:
        result.update(validate_plan(json.loads(path.read_text(encoding='utf-8-sig'))))
    except (ValueError, KeyError, TypeError) as error:
        result.update(status='FAIL_STRICT_SETTLEMENT_PLAN', error=str(error))
    output = Path(output)
    if output.exists(): raise ValueError('PREFLIGHT_OUTPUT_ALREADY_EXISTS')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = validate_file(args.plan, args.output)
    print(result['status'])
    raise SystemExit(0 if result['status'].startswith('PASS') else 2)
