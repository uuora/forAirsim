"""Sealed V3 acquisition and measured-flight evidence, read-only/no cloud."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import cv2
import numpy as np

from analyze_t1_identity_flight import (digest, read_json, verify_file, source_checks,
    identity_score, mask_truth, binding_issues, command_audit, lease_clear_audit,
    recovery_metrics, current_phase, stage_metrics, finite, canonical_sha)
from analyze_t1_async_timing import timing_metrics, is_nonzero_target, distribution
from t1_semantic_perception import Candidate, SemanticFrame
from t1_delayed_semantic_bridge import verify_delayed_selection


def scoped_events(events, start, end):
    if not finite(start) or not finite(end):return []
    selected=[]
    for row in events:
        key='host_receipt_monotonic_s' if row.get('kind')=='SELF_STATE' else 'dispatch_monotonic_s'
        if row.get('kind') not in {'SELF_STATE','COMMAND'} or (
                finite(row.get(key)) and start<=row[key]<=end):
            selected.append(row)
    return selected


def phase_evidence(report, events, *, acquisition=False):
    prefix='acquisition_stage_' if acquisition else 'low_speed_stage_'
    start=report.get(prefix+'start_state',{}).get('host_receipt_monotonic_s')
    end=report.get(prefix+'end_state',{}).get('host_receipt_monotonic_s')
    if not finite(start) or not finite(end):return None,None
    phase_report=dict(report, frames=report.get('acquisition_frames' if acquisition else 'frames',[]),
        slots=report.get('acquisition_slots' if acquisition else 'slots',[]),
        low_speed_stage_start_state=report[prefix+'start_state'],
        low_speed_stage_end_state=report[prefix+'end_state'])
    phase_report['injected_delays']=[row for row in report.get('injected_delays',[]) if
        finite(row.get('started_monotonic_s')) and finite(row.get('completed_monotonic_s')) and
        row['started_monotonic_s']<end and row['completed_monotonic_s']>start]
    subset=scoped_events(events,start,end)
    return timing_metrics(phase_report,subset),stage_metrics(phase_report,subset)


def frame_from_dict(value):
    return SemanticFrame(value['session_id'], value['target_generation'], value['frame_id'],
        value['frame_sequence'], value['image_sha256'], *value['image_size_wh'],
        value['readback_receipt_monotonic_s'], value['candidates_ready_monotonic_s'],
        tuple(Candidate(c['candidate_id'], tuple(c['bbox_xyxy_pixels']), c['detector_score'],
            c['source_class_name'], c['truncated']) for c in value['candidates']),
        detector_report_json=json.dumps(value['detector_report'], allow_nan=False))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def analyze_run(root):
    root = Path(root).resolve()
    manifest = read_json(root/'manifest.json')
    path = Path(manifest.get('report', root/'flight/report.json'))
    report = read_json(path)
    plan = read_json(report['plan'])
    checks = [verify_file(path, manifest.get('report_sha256'), 'launcher_report'),
        verify_file(report['plan'], manifest.get('plan_sha256'), 'launcher_plan')]
    checks += source_checks(plan, root, report)
    execution = report['execution_log']
    checks.append(verify_file(execution['path'], execution['sha256'], 'execution'))
    if manifest.get('fresh_ground_closure_path'):
        checks.append(verify_file(manifest['fresh_ground_closure_path'],
            manifest.get('fresh_ground_closure_sha256'), 'fresh_ground_closure'))
    if any(row['status'] != 'PASS' for row in checks):
        return dict(run=str(root), status='SEALED_INPUT_FAILURE', checks=checks, formal_efficacy_results=False)
    events = [json.loads(line) for line in Path(execution['path']).read_text(encoding='utf-8-sig').splitlines()]
    all_frames = report.get('acquisition_frames', []) + report.get('frames', [])
    issues, frame_rows, seen = [], [], set()
    if any(report.get('safe_end',{}).get(key) is not True for key in ('grounded_and_released','paused')):
        issues.append(dict(issue='RUNNER_SAFE_END_NOT_VERIFIED'))
    closure=read_json(manifest['fresh_ground_closure_path']) if manifest.get('fresh_ground_closure_path') else {}
    if (manifest.get('fresh_ground_closure_verified') is not True or
            manifest.get('engine_closed_after_safe_completion') is not True or
            closure.get('status')!='READY' or closure.get('grounded_and_released') is not True or closure.get('paused') is not True):
        issues.append(dict(issue='FRESH_GROUND_CLOSURE_NOT_VERIFIED'))
    saved_semantic={row['frame_id']:row.get('observer_result',{}).get('semantic_frame') for row in all_frames}
    terminal=report.get('observer_close',{}).get('request_terminal_trace',{}).get('terminal_request_trace',[])
    accepted_requests={row['request_binding']['request_id']:row for row in terminal if row.get('status')=='ACCEPTED_CHAIN'}
    previous_sequence = previous_receipt = None
    for frame in all_frames:
        row_issues = binding_issues(frame, report)
        frame_id = frame['frame_id']
        sequence = frame['frame_sequence']
        acquiring = frame.get('experiment_scope') == 'INITIAL_IDENTITY_ACQUISITION'
        base = 0 if acquiring else report.get('measured_frame_sequence_base')
        if not isinstance(base, int) or isinstance(base, bool) or sequence != base + frame['slot_index']:
            row_issues.append('GLOBAL_SEQUENCE_DIFFERS_FROM_DECLARED_STAGE_SLOT')
        if frame_id != f"{plan['session_id']}_f{sequence:03d}":
            row_issues.append('FRAME_ID_DIFFERS_FROM_GLOBAL_SEQUENCE')
        receipt = frame.get('capture_received_host_s')
        if frame_id in seen:
            row_issues.append('DUPLICATE_FRAME_ID')
        seen.add(frame_id)
        if previous_sequence is not None and (sequence <= previous_sequence or
                not finite(receipt) or receipt <= previous_receipt):
            row_issues.append('FRAME_SEQUENCE_OR_RECEIPT_REGRESSION')
        if finite(receipt):
            previous_sequence, previous_receipt = sequence, receipt
        truth = None
        media = frame.get('media', {})
        media_checks = [verify_file(media.get(key, {}).get('path', '__missing__'),
            media.get(key, {}).get('sha256'), 'frame_'+key) for key in ('rgb','depth','segmentation')]
        checks += media_checks
        if all(row['status'] == 'PASS' for row in media_checks):
            rgb = cv2.imread(media['rgb']['path'])
            seg = cv2.imread(media['segmentation']['path'])
            depth = np.load(media['depth']['path'], allow_pickle=False)
            if rgb is None or seg is None or rgb.shape != seg.shape or rgb.shape[:2] != depth.shape:
                row_issues.append('PAIRED_MEDIA_DIMENSIONS_INVALID')
            else:
                truth = mask_truth(seg, report['segmentation_palette_rgb'])
        selection = frame.get('observer_result', {}).get('temporal_identity', {})
        proof_audit = None
        if selection.get('status') == 'TEMPORAL_LOCAL_ASSOCIATION_VALID':
            try:
                proof_audit = verify_delayed_selection(
                    frame_from_dict(frame['observer_result']['semantic_frame']), selection, plan['description'],
                    frame['observer_result']['process_completed_host_monotonic_s'])
                proof=selection['delayed_chain_proof']
                if any(saved_semantic.get(node['frame']['frame_id'])!=node['frame'] for node in proof['nodes']):
                    row_issues.append('CHAIN_NODE_DIFFERS_FROM_ACTUAL_SAVED_OBSERVATION')
                anchor_trace=accepted_requests.get(proof['request_binding']['request_id'],{})
                if (anchor_trace.get('request_binding')!=proof['request_binding'] or any(
                        anchor_trace.get(key)!=proof[key] for key in ('request_sha256','semantic_reply_sha256','reply_result_sha256'))):
                    row_issues.append('CHAIN_ANCHOR_NOT_BOUND_TO_SAVED_TERMINAL_REQUEST')
            except (ValueError, KeyError, TypeError):
                row_issues.append('INDEPENDENT_DELAYED_CHAIN_PROOF_REJECTED')
        identity = identity_score(selection, truth)
        phase = ('initial_acquisition' if acquiring else
            current_phase(report, frame.get('capture_started_host_s'), receipt))
        row = dict(frame_id=frame_id, frame_sequence=sequence, slot_index=frame['slot_index'],
            actual_phase=phase, readback_s=receipt, saved_status=frame.get('status'),
            observer_completed_s=frame.get('observer_result', {}).get('process_completed_host_monotonic_s'),
            decision_completed_s=frame.get('decision_completed_monotonic_s'),
            temporal_status=selection.get('status'), identity=identity, proof_audit=proof_audit,
            surface_status=frame.get('surface', {}).get('status'), issues=row_issues)
        frame_rows.append(row)
        issues += [dict(frame_id=frame_id, issue=issue) for issue in row_issues]
    # Both ledgers are checked against their own saved final arrays; acquisition
    # observations never disappear from the denominator after successful entry.
    for name, field in (('frame_results.jsonl','frames'), ('acquisition_frame_results.jsonl','acquisition_frames')):
        ledger = path.parent/name
        if ledger.is_file():
            saved = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
            checks.append(dict(label=name, path=str(ledger), actual_sha256=digest(ledger),
                status='PASS' if canonical_sha(saved)==canonical_sha(report.get(field, [])) else 'FAIL'))
        else:
            checks.append(dict(label=name, status='MISSING'))
    by_id = {row['frame_id']:row for row in all_frames}
    scored = {row['frame_id']:row for row in frame_rows}
    commands = [row for row in events if row.get('kind')=='COMMAND' and row.get('stage')=='PERCEPTION_XY']
    command_rows = []
    for command in commands:
        target_frame = by_id.get(command.get('target_frame_id'))
        command_issues = command_audit(command, target_frame, report)
        clear = lease_clear_audit(command, target_frame, events)
        command_issues += clear['issues']
        if is_nonzero_target(command) and scored.get(command.get('target_frame_id'),{}).get('identity',{}).get('status')!='CORRECT':
            command_issues.append('NONZERO_TARGET_COMMAND_WITHOUT_INDEPENDENT_CORRECT_IDENTITY')
        command_rows.append(dict(command, issues=sorted(set(command_issues)), lease_clear=clear))
        issues += [dict(target_frame_id=command.get('target_frame_id'), issue=issue) for issue in command_issues]
    measured = [row for row in frame_rows if row['actual_phase']!='initial_acquisition']
    acquisition = [row for row in frame_rows if row['actual_phase']=='initial_acquisition']
    timing,motion=phase_evidence(report,events)
    acquisition_timing,acquisition_motion=phase_evidence(report,events,acquisition=True)
    controller_report=dict(report,frames=all_frames,
        low_speed_stage_start_state=report.get('acquisition_stage_start_state',{}),
        low_speed_stage_end_state=report.get('low_speed_stage_end_state',report.get('acquisition_stage_end_state',{})))
    controller_timing=timing_metrics(controller_report,events)
    controller_timing['frame_timing']={'scope':'separate phase grids; overlapping local slot indexes never merged',
        'acquisition':acquisition_timing.get('frame_timing') if acquisition_timing else None,
        'measured':timing.get('frame_timing') if timing else None}
    traces = report.get('observer_close', {}).get('sanitized_transport_trace', [])
    latencies = [row['backend_completed_monotonic_s']-row['backend_started_monotonic_s']
        for row in traces if finite(row.get('backend_completed_monotonic_s'))]
    wrong=sum(row['identity']['status']=='WRONG_DISTRACTOR' for row in frame_rows)
    nonzero=sum(is_nonzero_target(row) for row in commands)
    qualification=('FAIL_WRONG_IDENTITY' if wrong else 'FAIL_EVIDENCE_ISSUES' if issues or any(c['status']!='PASS' for c in checks)
        else 'FAIL_NO_VALID_MEASURED_OUTPUT' if not any(row['identity']['status']=='CORRECT' for row in measured)
        else 'FAIL_NO_NONZERO_TARGET_COMMAND' if plan['mode']=='SHORT_XY' and not nonzero
        else 'FAIL_RUNNER_STATUS' if report['status']!='PASS' else 'PASS_CONTROLLED_DEVELOPMENT_SCOPE')
    return dict(run=str(root), session_id=plan['session_id'], mode=plan['mode'],
        status='SEALED_EVIDENCE_ANALYZED' if not issues and all(c['status']=='PASS' for c in checks) else 'EVIDENCE_ISSUES_RETAINED',
        acquisition_epoch_monotonic_s=report.get('acquisition_epoch_monotonic_s'),
        measured_epoch_monotonic_s=report.get('epoch_monotonic_s'),
        phase_transitions=report.get('phase_transitions',[]),
        original_runner_status=report['status'], errors_preserved=report['errors'],
        independent_integration_qualification=qualification,
        launcher_status=manifest['status'], safe_end=report['safe_end'],
        fresh_ground_closure_verified=manifest.get('fresh_ground_closure_verified'),
        engine_closed_after_safe_completion=manifest.get('engine_closed_after_safe_completion'),
        checks=checks, issues=issues, initial_acquisition={key:report.get(key) for key in (
            'acquisition_status','acquisition_epoch_monotonic_s','acquisition_completed_monotonic_s',
            'acquisition_elapsed_s','acquisition_selected_frame_id','measurement_entered')},
        acquisition_slot_status_counts=dict(Counter(row.get('status') for row in report.get('acquisition_slots',[]))),
        measured_slot_status_counts=dict(Counter(row.get('status') for row in report.get('slots',[]))),
        acquisition_frame_count=len(acquisition), measured_frame_count=len(measured),
        acquisition_identity_counts=dict(Counter(row['identity']['status'] for row in acquisition)),
        measured_identity_counts=dict(Counter(row['identity']['status'] for row in measured)),
        nonzero_target_command_count=sum(is_nonzero_target(row) for row in commands),
        nonzero_acquisition_target_commands=sum(is_nonzero_target(row) and row.get('target_frame_id') in
            {frame['frame_id'] for frame in acquisition} for row in commands),
        frame_rows=frame_rows, target_command_rows=[row for row in command_rows if is_nonzero_target(row)],
        recovery=recovery_metrics(report, measured), timing=timing,
        acquisition_timing=acquisition_timing,controller_timing=controller_timing,
        actual_motion_stage=motion,acquisition_motion_stage=acquisition_motion,
        sanitized_transport_trace=traces,
        backend_wall_distribution=distribution(latencies),
        transport_status_counts=dict(Counter(row.get('backend_status','PENDING') for row in traces)),
        source_attribute_status_counts=dict(Counter(row.get('source_attribute_guard_status','NOT_CONFIGURED') for row in traces)),
        transport_error_counts=dict(Counter(row.get('transport_error',row.get('error_category')) for row in traces if
            row.get('transport_error') or row.get('error_category'))), formal_efficacy_results=False)


def plot_identity(runs, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    usable=[run for run in runs if run.get('frame_rows')]
    if not usable:return []
    fig,axes=plt.subplots(len(usable),2,figsize=(13,3.2*len(usable)),squeeze=False)
    for index,run in enumerate(usable):
        left,right=axes[index]; origin=run.get('acquisition_epoch_monotonic_s')
        if not finite(origin):origin=min(row['readback_s'] for row in run['frame_rows'] if finite(row['readback_s']))
        groups={'CORRECT':(1,'#167c50'),'NO_OUTPUT':(0,'#576879'),
            'WRONG_DISTRACTOR':(-1,'#c23d37'),'AMBIGUOUS_OR_UNMATCHED':(-.5,'#ce872c')}
        for status,(level,color) in groups.items():
            rows=[row for row in run['frame_rows'] if row['identity']['status']==status and finite(row['readback_s'])]
            if rows:left.scatter([row['readback_s']-origin for row in rows],[level]*len(rows),s=18,c=color,label=status)
        left.set_yticks([-1,-.5,0,1],['Wrong','Ambiguous','No output','Correct'])
        left.set_ylim(-1.25,1.25);left.set_xlabel('Seconds from initial acquisition');left.grid(alpha=.2)
        tag=run['session_id'].replace('t1_identity_chain_','').replace('t1_identity_attribute_','attribute_')
        qualified='PASS (controlled scope)' if run['independent_integration_qualification'].startswith('PASS') else 'FAIL: wrong identity'
        left.set_title(tag+'\nRunner '+run['original_runner_status']+' / independent '+qualified,fontsize=10)
        if finite(run.get('measured_epoch_monotonic_s')):
            left.axvline(run['measured_epoch_monotonic_s']-origin,c='#246b9e',ls='--',label='Measured phase starts')
        transitions={row['phase_id']:row['completed_monotonic_s'] for row in run['phase_transitions']}
        if 'hidden' in transitions and 'return' in transitions:
            left.axvspan(transitions['hidden']-origin,transitions['return']-origin,color='#ad5f42',alpha=.12,label='Actual occlusion')
        left.legend(fontsize=7,loc='lower right')
        for request_index,trace in enumerate(run['sanitized_transport_trace']):
            start=trace['backend_started_monotonic_s']-origin
            end=trace.get('backend_completed_monotonic_s')
            if finite(end):
                label=trace.get('transport_error',trace.get('backend_status','UNKNOWN'))
                color='#167c50' if trace.get('backend_status')=='REPLY_RETURNED' else '#c23d37'
                if trace.get('source_attribute_guard_status')=='REJECTED':
                    color='#d58d29';label='SOURCE_ATTRIBUTE_REJECTED'
                right.plot([start,end-origin],[request_index+1]*2,c=color,lw=4)
                right.text(end-origin+.15,request_index+1,label,fontsize=7,va='center')
        right.set_ylim(0,len(run['sanitized_transport_trace'])+1)
        right.set_ylabel('Semantic request index');right.set_xlabel('Seconds from initial acquisition')
        right.set_title('Backend intervals (reply return does not prove acceptance)');right.grid(alpha=.2)
        right.margins(x=.22)
    fig.tight_layout();files=[]
    for suffix in ('png','pdf'):
        file=output/('chain_identity_transport.'+suffix);fig.savefig(file,dpi=170,bbox_inches='tight');files.append(file)
    plt.close(fig);return files


def analyze(roots, output):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    runs=[analyze_run(root) for root in roots]
    report=dict(kind='T1_DELAYED_CHAIN_DEVELOPMENT_ANALYSIS', created_utc=datetime.now(timezone.utc).isoformat(),
        runs=runs, formal_efficacy_results=False,
        limitations=['Original source time remains unchanged; chain uniqueness is geometric, not guaranteed instance identity.',
            'Acquisition successes and failures are retained; these chronological pilots are not formal matched trials.',
            'Saved host clocks do not establish sensor exposure, hard real time or continuous safety.'])
    save(output/'report.json',report)
    lines=['# 任务一逐帧语义关联与初始身份获取结果','',
        '| 试次 | 原运行器 / 独立资格 | 获取帧 / 主测量帧 | 主测量正确 / 错误输出 | 非零目标指令 | 控制查询最大间隔(ms) |',
        '|---|---|---:|---:|---:|---:|']
    for run in runs:
        maximum=(run.get('controller_timing') or {}).get('monitor_event_receipt_cadence',{}).get('gap_distribution',{}).get('max_s')
        lines.append(f"| {run.get('session_id')} | {run.get('original_runner_status')} / {run.get('independent_integration_qualification')} | {run.get('acquisition_frame_count')} / {run.get('measured_frame_count')} | {run.get('measured_identity_counts',{}).get('CORRECT',0)} / {run.get('measured_identity_counts',{}).get('WRONG_DISTRACTOR',0)} | {run.get('nonzero_target_command_count',0)} | {maximum*1000:.1f} |" if maximum is not None else f"| {run.get('session_id')} | 输入未通过 | 未知 | 未知 | 未知 | 未知 |")
    lines+=['','初始身份获取与遮挡后的恢复分别统计。全部失败、未采集和多余获取槽状态保留在JSON中。',
        'AI辅助编写分析代码与报告；数值来自封存数据，方法与论文结论需作者审定。','']
    (output/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    artifacts=[output/'report.json', output/'summary.md']+plot_identity(runs,output)
    source=Path(__file__).resolve(); snapshot_dir=output/'source_snapshots';snapshot_dir.mkdir()
    sources=[]
    for name in (source.name,'analyze_t1_identity_flight.py','analyze_t1_async_timing.py','t1_delayed_semantic_bridge.py','t1_semantic_perception.py','t1_temporal_identity.py'):
        item=source.with_name(name);snapshot=snapshot_dir/name;shutil.copyfile(item,snapshot)
        sources.append(dict(path=str(item),snapshot_path=str(snapshot),sha256=digest(item)))
    save(output/'artifact_receipt.json',dict(sources=sources,
        artifacts=[dict(path=str(p),sha256=digest(p)) for p in artifacts],formal_efficacy_results=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.runs,args.output)
    print(json.dumps(dict(runs=len(result['runs']),output=str(args.output.resolve()))))
