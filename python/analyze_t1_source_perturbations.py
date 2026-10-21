"""Read-only source-bound fault behavior audit with independent media scoring."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

import analyze_t1_settlement as timing
from analyze_t1_settlement_v1p1 import zero_hold_audit
from analyze_t1_settlement_v1p4 import admission_audit
from audit_t1_feedback_stop_v1p1 import feedback_audit
from analyze_t1_review_handoff_v5p1 import handoff_evidence
from analyze_t1_identity_flight import read_json, digest, mask_truth
from t1_perturbation_contract_v1 import validate_plan, ARCHITECTURE


def delivery_audit(report, trace):
    declaration=report['configuration']['source_perturbation']['reply_delivery_fault']
    terminal=report.get('observer_close',{}).get('request_terminal_trace',{}).get('terminal_request_trace',[])
    terminal_by_id={t['request_binding']['request_id']:t for t in terminal}
    issues=[];holds=[];service=[]
    for record in trace:
        start=record.get('backend_started_monotonic_s');end=record.get('backend_completed_monotonic_s')
        if start is not None and end is not None:service.append(end-start)
        delivery=record.get('controlled_reply_delivery',{})
        hold=delivery.get('declared_hold_s',0.)
        if not hold:continue
        rid=record['request_binding']['request_id'];term=terminal_by_id.get(rid,{})
        completed=delivery.get('hold_completed_monotonic_s');began=delivery.get('hold_started_monotonic_s')
        source=record['request_binding']['readback_receipt_monotonic_s']
        if (delivery.get('request_binding')!=record['request_binding']
                or delivery.get('original_reply_binding')!=record['request_binding']
                or delivery.get('original_reply_completed_monotonic_s')!=record.get('reply_completed_monotonic_s')
                or delivery.get('natural_service_delay') is not False
                or delivery.get('service_reply_clock_and_content_unchanged') is not True
                or delivery.get('delivery_status')!='DELIVERED'
                or began is None or completed is None or completed-began<hold-1e-6
                or source<report['epoch_monotonic_s']):
            issues.append(dict(request_id=rid,issue='ACTUAL_DELIVERY_HOLD_OR_ORIGINAL_SOURCE_NOT_VERIFIED'))
        expired=hold>8.
        if expired and (not term or term.get('status')=='ACCEPTED_CHAIN'
                or not term.get('transport_done') or term.get('reason') not in
                    ('SOURCE_CHAIN_AGE_EXPIRED','RESPONSE_TIMEOUT','SOURCE_CHAIN_AGE_EXPIRED_AT_COMPLETION')):
            issues.append(dict(request_id=rid,issue='EXPIRED_HELD_REPLY_REJECTION_NOT_VERIFIED',terminal_status=term.get('status'),reason=term.get('reason')))
        holds.append(dict(request_id=rid,declared_hold_s=hold,hold_started_monotonic_s=began,
            hold_completed_monotonic_s=completed,real_service_wall_s=end-start if start is not None and end is not None else None,
            original_source_readback_s=source,original_reply_completed_s=record.get('reply_completed_monotonic_s'),
            delivery_completed_s=delivery.get('delivered_to_policy_monotonic_s'),
            terminal_status=term.get('status'),terminal_reason=term.get('reason'),expired_expected=expired))
    expected=declaration['durations_s']
    if [r['declared_hold_s'] for r in holds]!=expected:
        issues.append(dict(issue='PREDECLARED_HOLD_COUNT_OR_ORDER_NOT_OBSERVED',expected=expected,actual=[r['declared_hold_s'] for r in holds]))
    return dict(status='PASS_SOURCE_BOUND_CONTROLLED_DELIVERY_AUDIT' if not issues else 'FAIL_CONTROLLED_DELIVERY_AUDIT',
        holds=holds,issues=issues,actual_service_wall_distribution=timing.distribution(service),
        natural_service_over_two_s_count=sum(s>2. for s in service),natural_service_delay_was_injected=False,
        controlled_delays_count=len(holds),terminal_scope='full future drain with original semantic source clock retained')


def analyze_run(root):
    root=Path(root).resolve();raw=read_json(root/'flight/report.json');plan=raw['configuration']
    check=validate_plan(plan)
    r=timing.analyze_run(root)
    if r['base_evidence'].get('status')=='SEALED_INPUT_FAILURE':return r
    events=[json.loads(s) for s in (root/'flight/execution.jsonl').read_text(encoding='utf-8').splitlines()]
    base=r['base_evidence'];spec=plan['source_perturbation'];by_frame={f['frame_id']:f for f in base['frame_rows']}
    frames=[];scene_issues=[];expected_count=check['expected_main_slots']
    grid_ok=(len(raw['frames'])==expected_count and len(raw['slots'])==expected_count
        and {f['slot_index'] for f in raw['frames']}==set(range(expected_count))
        and all(s.get('status')=='CAPTURED' for s in raw['slots']))
    for frame in raw['frames']:
        fid=frame['frame_id']
        if any(k not in frame.get('media',{}) for k in ('rgb','segmentation')):
            scene_issues.append(dict(frame_id=fid,issue='PAIRED_FRAME_MEDIA_MISSING'))
            continue
        seg=cv2.imread(frame['media']['segmentation']['path']);rgb=cv2.imread(frame['media']['rgb']['path'])
        if seg is None or rgb is None or seg.shape!=rgb.shape:
            scene_issues.append(dict(frame_id=fid,issue='PAIRED_FRAME_MEDIA_UNREADABLE'))
            continue
        truth=mask_truth(seg,raw['segmentation_palette_rgb'])
        phase=frame['phase_id'];identity=by_frame[fid]['identity']['status'];target_pixels=truth['target']['visible_pixel_count']
        expected_hidden=spec['condition']=='occlusion' and phase=='hidden'
        if (target_pixels!=0 if expected_hidden else target_pixels<500):
            scene_issues.append(dict(frame_id=fid,issue='DECLARED_TARGET_VISIBILITY_NOT_OBSERVED',pixels=target_pixels,phase=phase))
        if truth['distractor']['visible_pixel_count']<500:
            scene_issues.append(dict(frame_id=fid,issue='DECLARED_DISTRACTOR_NOT_VISIBLE'))
        colour=np.asarray(raw['segmentation_palette_rgb']['43'][::-1],np.uint8)
        mask=np.all(seg==colour,axis=2)
        white=(rgb.min(axis=2)>=200)&(rgb.max(axis=2)-rgb.min(axis=2)<=35)
        white_count=int(np.count_nonzero(mask&white))
        if spec['condition']=='glint' and phase=='glint' and white_count<5:
            scene_issues.append(dict(frame_id=fid,issue='SMALL_WHITE_PATCH_NOT_VISIBLE_IN_REAL_RGB',pixels=white_count))
        frames.append(dict(frame_id=fid,slot_index=frame['slot_index'],phase_id=phase,
            relative_readback_s=frame['capture_received_host_s']-raw['epoch_monotonic_s'],identity=identity,
            target_visible_pixels=target_pixels,distractor_visible_pixels=truth['distractor']['visible_pixel_count'],
            distractor_white_pixels=white_count,temporal_status=by_frame[fid]['temporal_status'],
            semantic_poll_reason=frame.get('observer_result',{}).get('semantic_poll',{}).get('reason'),
            nonzero_target_dispatch=bool(frame.get('command',{}).get('purpose')=='CURRENT_TARGET_TRAJECTORY'
                and np.linalg.norm(frame.get('command',{}).get('velocity_xy_m_s',[0.,0.]))>1e-6)))
    state_audit=feedback_audit(plan,raw,events);zero=zero_hold_audit(raw,events);admission=admission_audit(raw)
    handoff=handoff_evidence(raw)
    for f in raw.get('acquisition_frames',[])+raw['frames']:
        if not isinstance(f.get('observer_result',{}).get('temporal_identity',{}).get('review_handoff'),dict):
            handoff['issues'].append(dict(frame_id=f['frame_id'],issue='V8_HANDOFF_RECEIPT_MISSING'))
    delivery=delivery_audit(raw,base['sanitized_transport_trace'])
    counts=Counter(f['identity'] for f in frames)
    hidden=[f for f in frames if f['phase_id']=='hidden'];returned=[f for f in frames if f['phase_id']=='return']
    hidden_wrong=sum(f['identity']!='NO_OUTPUT' or f['nonzero_target_dispatch'] for f in hidden)
    recovered=next((f for f in returned if f['identity']=='CORRECT'),None)
    behavior_issues=[]
    if any(k not in ('CORRECT','NO_OUTPUT') and n for k,n in counts.items()):
        behavior_issues.append('WRONG_OR_UNMATCHED_TARGET_OUTPUT_OBSERVED')
    if spec['condition']=='occlusion' and (len(hidden)!=10 or hidden_wrong or recovered is None):
        behavior_issues.append('OCCLUSION_REFUSAL_OR_RETURN_RECOVERY_NOT_ESTABLISHED')
    if spec['condition']=='baseline' and counts.get('CORRECT',0)!=expected_count:
        behavior_issues.append('BASELINE_CURRENT_IDENTITY_NOT_ESTABLISHED_ON_ALL_MAIN_FRAMES')
    evidence_ok=(base['independent_integration_qualification']=='PASS_CONTROLLED_DEVELOPMENT_SCOPE'
        and state_audit['status'].startswith('PASS') and zero['status'].startswith('PASS')
        and admission['status'].startswith('PASS') and not handoff['issues']
        and delivery['status'].startswith('PASS') and grid_ok and not scene_issues)
    r.update(plan_check=check,source_perturbation=spec,feedback_stop_audit=state_audit,zero_hold_audit=zero,
        actual_clear_admission_audit=admission,review_handoff=handoff,reply_delivery_audit=delivery,
        frame_timing_rows=timing.timing_rows(raw),source_fault_frame_rows=frames,
        measured_grid_check=dict(status='PASS_COMPLETE_DECLARED_PERTURBATION_GRID' if grid_ok else 'FAIL_PERTURBATION_GRID',
            planned_slots=expected_count,recorded_frames=len(frames)),
        physical_scene_audit=dict(status='PASS_MEASURED_DECLARED_SCENE' if not scene_issues else 'FAIL_DECLARED_SCENE_NOT_ESTABLISHED',issues=scene_issues),
        source_fault_behavior=dict(identity_counts=dict(counts),issues=behavior_issues,
            hidden_frames=len(hidden),hidden_refusal_failures=hidden_wrong,
            first_correct_return_frame=recovered['frame_id'] if recovered else None,
            return_first_correct_relative_s=recovered['relative_readback_s']-24. if recovered and spec['condition']=='occlusion' else None),
        independent_fault_qualification=('PASS_BOUNDED_SOURCE_FAULT_DEVELOPMENT' if evidence_ok and not behavior_issues
            else 'FAIL_SOURCE_FAULT_BEHAVIOR' if evidence_ok else 'FAIL_SOURCE_FAULT_EVIDENCE'),
        timing_normalization_scope='original complete timing and 0.4s deadline retained; source faults have distinct ledgers',
        actual_fault_behavior_and_motion_scale_are_separate=True)
    return r


def analyze(roots,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_SOURCE_FAULT_ANALYSIS_REQUIRED')
    runs=[analyze_run(p) for p in roots];output.mkdir(parents=True)
    names=('analyze_t1_source_perturbations.py','t1_perturbation_contract_v1.py','t1_reply_delivery_fault_v1.py','t1_scene_fault_v1.py',
        'analyze_t1_settlement.py','analyze_t1_settlement_v1p1.py','analyze_t1_settlement_v1p4.py','audit_t1_feedback_stop_v1p1.py',
        't1_stop_receipt_match.py','analyze_t1_delayed_identity.py','analyze_t1_identity_flight.py','analyze_t1_motion_scale.py',
        'analyze_t1_async_timing.py','analyze_t1_review_handoff_v5p1.py')
    for name in names:
        p=Path(__file__).with_name(name);dest=output/'source_snapshots'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
    result=dict(kind='T1_SOURCE_FAULT_DEVELOPMENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),runs=runs,
        new_flights_started_by_analysis=0,formal_efficacy_results=False,
        limitations=['Artificial reply holds and actual service latency are distinct.','Independent session units; frames are temporally correlated.',
            'Source-fault behavior does not establish formal trajectory efficacy or continuous safety.','All missed, failed and no-output slots retained.'])
    timing.save(output/'report.json',result)
    timing.save(output/'artifact_receipt.json',dict(files=[dict(path=str(p),sha256=digest(p)) for p in output.rglob('*') if p.is_file()],
        input_receipts=[x for r in runs for x in r['input_receipts']]))
    for r in runs:print(r['session_id'],r['independent_fault_qualification'],r['source_fault_behavior']['identity_counts'],r['endpoint_scale_status'])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.runs,a.output)
