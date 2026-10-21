"""Extend the sealed V7 auditor for current-observation stationary zero holds."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

import analyze_t1_settlement as prior
from t1_control_settlement import SettlementConfig
from t1_settlement_plan_v1p1 import ARCHITECTURE, validate_plan


def zero_hold_audit(report, events):
    issues=[]; count=0; c=SettlementConfig()
    states=[e for e in events if e.get('kind')=='SELF_STATE' and e.get('stage')=='PERCEPTION_XY']
    commands=[e for e in events if e.get('kind')=='COMMAND' and e.get('stage')=='PERCEPTION_XY']
    for f in report.get('acquisition_frames',[])+report.get('frames',[]):
        proof=f.get('zero_held_stop_proof')
        if proof is None: continue
        count+=1; bad=[]; current=proof.get('current_self_state',{})
        samples=proof.get('samples',[])
        if (f.get('command') is not None or f.get('lease_stop') is not None
                or proof.get('command_timestamp_changed') is not False
                or proof.get('new_target_lease') is not False
                or current!=f.get('command_self_state')):
            bad.append('ZERO_HOLD_CREATED_COMMAND_LEASE_OR_MISMATCHED_CURRENT_STATE')
        if not any(all(s.get(k)==current.get(k) for k in ('host_receipt_monotonic_s','state_timestamp_ns',
                'position_world_ned_m','velocity_ned_m_s')) for s in states):
            bad.append('ZERO_HOLD_CURRENT_STATE_NOT_ACTUALLY_OBSERVED')
        matching=[e for e in commands if e['dispatch_monotonic_s']==proof.get('actual_zero_dispatch_monotonic_s')
            and math.hypot(*e['velocity_xy_m_s'])<=1e-12]
        now=f['motion_gate']['decision_monotonic_s']; receipt=current.get('host_receipt_monotonic_s',now-1.)
        if (not matching or not 0<=now-receipt<=.12+1e-9
                or not 0<=now-proof.get('actual_zero_dispatch_monotonic_s',now-1.)<.30+1e-9
                or math.hypot(*current.get('velocity_ned_m_s',[1.,1.])[:2])>c.stationary_xy_threshold_m_s):
            bad.append('ZERO_HOLD_CURRENT_STATE_OR_REAL_ZERO_CLOCK_INVALID')
        if len({s.get('state_timestamp_ns') for s in samples})<3 or len(samples)<3:
            bad.append('ZERO_HOLD_DISTINCT_STATE_PROOF_MISSING')
        elif (samples[-1]['receipt_monotonic_s']-samples[0]['receipt_monotonic_s']<.10-1e-9
                or receipt-proof['target_setpoint_cleared_monotonic_s']<.30-1e-9):
            bad.append('ZERO_HOLD_DWELL_NOT_ESTABLISHED')
        for s in samples:
            if (math.hypot(*s['velocity_xy_m_s'])>c.stationary_xy_threshold_m_s
                    or math.hypot(*s['actual_last_command_xy_m_s'])>1e-12
                    or not any(r['host_receipt_monotonic_s']==s['receipt_monotonic_s']
                        and r['state_timestamp_ns']==s['state_timestamp_ns']
                        and r['velocity_ned_m_s'][:2]==s['velocity_xy_m_s'] for r in states)):
                bad.append('ZERO_HOLD_HISTORY_NOT_STATIONARY_ACTUAL_SAMPLES')
        issues.extend(dict(frame_id=f['frame_id'], issue=x) for x in sorted(set(bad)))
    return dict(status='PASS_ACTUAL_STATIONARY_ZERO_HOLD_AUDIT' if not issues and count else
        'FAIL_OR_MISSING_STATIONARY_ZERO_HOLD_PROOF', zero_hold_frame_count=count, issues=issues)


def analyze_run(root):
    result=prior.analyze_run(root)
    if result['architecture']!=ARCHITECTURE: return result
    root=Path(root).resolve(); report=prior.read_json(root/'flight/report.json')
    events=[json.loads(s) for s in (root/'flight/execution.jsonl').read_text(encoding='utf-8').splitlines()]
    result['plan_check']=validate_plan(report['configuration'])
    result['feedback_stop_audit']=prior.feedback_audit(report['configuration'],report,events)
    result['zero_hold_audit']=zero_hold_audit(report,events)
    result['frame_timing_rows']=prior.timing_rows(dict(report,control_architecture=prior.ARCHITECTURE))
    result['timing_normalization_scope']='V7P1 uses identical separate timing rules; raw source architecture remains unchanged'
    return result


def analyze(roots, output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('NEW_ANALYSIS_DIRECTORY_REQUIRED')
    runs=[analyze_run(p) for p in roots]; output.mkdir(parents=True)
    for name in ('analyze_t1_settlement_v1p1.py','analyze_t1_settlement.py','t1_control_settlement.py',
            't1_settlement_plan_v1.py','t1_settlement_plan_v1p1.py','t1_async_control_v1p3.py',
            'analyze_t1_delayed_identity.py','analyze_t1_identity_flight.py','analyze_t1_motion_scale.py','analyze_t1_async_timing.py'):
        file=Path(__file__).parent/name; dest=output/'source_snapshots'/name
        dest.parent.mkdir(exist_ok=True); shutil.copyfile(file,dest)
    result=dict(kind='T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs,new_flights_started_by_analysis=0,formal_efficacy_results=False,
        limitations=['Old full 0.4-second deadline preserved; split clocks are a predeclared new policy.',
            'Strict sampled stopping/own-state feedback are engineering evidence, not formal trajectory optimization results.'])
    prior.save(output/'report.json',result)
    prior.save(output/'artifact_receipt.json',dict(files=[dict(path=str(f),sha256=prior.digest(f)) for f in output.rglob('*') if f.is_file()],
        input_receipts=[x for r in runs for x in r['input_receipts']]))
    for r in runs:
        print(r['session_id'], r['base_evidence']['independent_integration_qualification'],r['feedback_stop_audit']['status'],
            r.get('zero_hold_audit',{}).get('status'),r['endpoint_scale_status'],r['endpoint']['net_xy_displacement_m'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args(); analyze(args.runs,args.output)
