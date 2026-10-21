"""Audit V7P3 actual pre-query commands and complete 20/60-second measured grids."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import analyze_t1_settlement_v1p1 as old
from t1_settlement_plan_v1p3 import ARCHITECTURE, validate_plan


def admission_audit(report):
    issues=[]; count=0
    for f in report['frames']+report.get('acquisition_frames',[]):
        command=f.get('command',{}); gate=f.get('motion_gate',{})
        if command.get('purpose')!='CURRENT_TARGET_TRAJECTORY' or sum(x*x for x in command['velocity_xy_m_s'])<=1e-12:
            continue
        count+=1
        deadline=gate.get('command_end_deadline_monotonic_s'); limit=gate.get('latest_declared_lease_end_monotonic_s')
        if (gate.get('actual_clear_reserve_s')!=.12 or deadline is None or limit is None
                or abs(deadline-limit-.12)>1e-8
                or gate.get('original_observation_end_deadline_monotonic_s')!=deadline
                or command['dispatch_monotonic_s']+command['duration_s']>limit+1e-8):
            issues.append(dict(frame_id=f['frame_id'],issue='ACTUAL_CLEAR_RESERVE_ADMISSION_NOT_VERIFIED'))
    return dict(status='PASS_DECLARED_ACTUAL_CLEAR_RESERVE_AUDIT' if not issues else 'FAIL_ACTUAL_CLEAR_RESERVE_AUDIT',
        nonzero_target_count=count,issues=issues,original_observation_deadline_preserved=True)


def analyze_run(root):
    r=old.analyze_run(root)
    if r['architecture']!=ARCHITECTURE: return r
    root=Path(root).resolve(); report=old.prior.read_json(root/'flight/report.json')
    events=[json.loads(s) for s in (root/'flight/execution.jsonl').read_text(encoding='utf-8').splitlines()]
    check=validate_plan(report['configuration']); count=check['expected_main_slots']
    frames=report['frames']; slots=report['slots']
    grid_ok=(len(frames)==len(slots)==count and {f['slot_index'] for f in frames}==set(range(count))
        and len({f['frame_id'] for f in frames})==count
        and all(s.get('status')=='CAPTURED' for s in slots)
        and all(f.get('evaluation_truth',{}).get('target',{}).get('visible_pixel_count',0)>=500 for f in frames))
    r.update(plan_check=check,measured_grid_check=dict(status='PASS_COMPLETE_SAMPLED_VISIBLE_GRID' if grid_ok else 'FAIL_MEASURED_GRID',
        planned_slots=count,recorded_frames=len(frames),duration_s=check['expected_main_duration_s']),
        feedback_stop_audit=old.prior.feedback_audit(report['configuration'],report,events),
        zero_hold_audit=old.zero_hold_audit(report,events),
        frame_timing_rows=old.prior.timing_rows(dict(report,control_architecture=old.prior.ARCHITECTURE)),
        timing_normalization_scope='V7P2 retains original completion clocks; only architecture adapters are normalized')
    r['actual_clear_admission_audit']=admission_audit(report)
    return r


def analyze(roots, output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('NEW_ANALYSIS_DIRECTORY_REQUIRED')
    runs=[analyze_run(p) for p in roots]; output.mkdir(parents=True)
    source_names=('analyze_t1_settlement_v1p3.py','analyze_t1_settlement_v1p2.py','analyze_t1_settlement_v1p1.py','analyze_t1_settlement.py',
        't1_settlement_plan_v1p3.py','t1_identity_motion_policy_v1p1.py','t1_settlement_plan_v1p2.py','t1_settlement_plan_v1p1.py','t1_settlement_plan_v1.py','t1_control_settlement.py',
        't1_async_control_v1p4.py','t1_async_control_v1p3.py','analyze_t1_delayed_identity.py',
        'analyze_t1_identity_flight.py','analyze_t1_motion_scale.py','analyze_t1_async_timing.py')
    for name in source_names:
        file=Path(__file__).parent/name; dest=output/'source_snapshots'/name
        dest.parent.mkdir(exist_ok=True); shutil.copyfile(file,dest)
    result=dict(kind='T1_STRICT_SETTLEMENT_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs,new_flights_started_by_analysis=0,formal_efficacy_results=False,
        limitations=['Failed earlier versions remain in their sealed reports.',
            'Longer duration is a separate condition and does not requalify failed 20-second motion.',
            'Strict sampled stop is not continuous physical safety or formal algorithm efficacy.'])
    old.prior.save(output/'report.json',result)
    old.prior.save(output/'artifact_receipt.json',dict(files=[dict(path=str(f),sha256=old.prior.digest(f)) for f in output.rglob('*') if f.is_file()],
        input_receipts=[x for r in runs for x in r['input_receipts']]))
    for r in runs:
        print(r['session_id'],r['base_evidence']['independent_integration_qualification'],r['feedback_stop_audit']['status'],
            r.get('zero_hold_audit',{}).get('status'),r['endpoint_scale_status'],r['endpoint']['net_xy_displacement_m'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args(); analyze(args.runs,args.output)
