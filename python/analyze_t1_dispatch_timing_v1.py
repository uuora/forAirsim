"""Source-bound host timing, with every attempt and stage denominator retained."""
import argparse, hashlib, json, math, shutil, statistics
from pathlib import Path
from datetime import datetime, timezone


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stats(values):
    values=sorted(values)
    return dict(count=len(values),median_ms=1000*statistics.median(values) if values else None,
        p95_ms=1000*values[math.ceil(.95*len(values))-1] if values else None,
        maximum_ms=1000*max(values) if values else None)


def analyze(analysis,output):
    analysis,output=Path(analysis).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_DISPATCH_TIMING_OUTPUT_REQUIRED')
    data=json.loads(analysis.read_text(encoding='utf-8'));rows=[];inputs=[dict(path=str(analysis),sha256=sha(analysis))]
    for source in data['runs']:
        if not source.get('architecture','').endswith('V10P2'):continue
        for r in source['input_receipts']:
            if sha(r['path'])!=r['sha256']:raise ValueError('INPUT_CHANGED')
            inputs.append(r)
        root=Path(source['run']);raw=json.loads((root/'flight/report.json').read_text(encoding='utf-8'))
        events=[json.loads(s) for s in (root/'flight/execution.jsonl').read_text(encoding='utf-8').splitlines()]
        epoch=source['base_evidence'].get('measured_epoch_monotonic_s')
        end=epoch+raw['configuration']['planned_duration_s'] if epoch is not None else None
        def stage(t):return 'ACQUISITION' if epoch is None or t<epoch else 'MEASURED_PHASE' if t<=end else 'FINALIZATION'
        reads=[];feedback=[];refusals=[]
        for e in events:
            if e.get('kind')=='OWN_STATE_READ_TIMING':
                t=e['timing']
                reads.append(dict(stage=stage(t['query_started_monotonic_s']),state_timestamp_ns=t['state_timestamp_ns'],
                    query_s=t['query_completed_monotonic_s']-t['query_started_monotonic_s'],
                    validation_s=t['validation_completed_monotonic_s']-t['query_completed_monotonic_s'],
                    condition_wait_s=t['condition_acquired_monotonic_s']-t['condition_entered_monotonic_s'],
                    condition_inside_s=t['condition_released_monotonic_s']-t['condition_acquired_monotonic_s'],
                    state_event_admission_s=t['state_event_admission_completed_monotonic_s']-t['state_event_admission_started_monotonic_s']))
            elif e.get('kind')=='OWN_STATE_FEEDBACK_PROOF':
                t=e['timing'];s=e['command_self_state']
                feedback.append(dict(stage=stage(t['preparation_started_monotonic_s']),sequence=e['feedback_sequence'],attempt=e['attempt'],
                    state_receipt_monotonic_s=s['host_receipt_monotonic_s'],state_timestamp_ns=s['state_timestamp_ns'],
                    preparation_s=t['preparation_completed_monotonic_s']-t['preparation_started_monotonic_s'],
                    dispatch_check_monotonic_s=t['deadline_check_monotonic_s'],
                    receipt_to_preparation_s=t['preparation_started_monotonic_s']-s['host_receipt_monotonic_s'],
                    preparation_to_dispatch_s=t['deadline_check_monotonic_s']-t['preparation_started_monotonic_s'],
                    state_age_at_dispatch_s=t['deadline_check_monotonic_s']-s['host_receipt_monotonic_s']))
            elif e.get('kind')=='FEEDBACK_DISPATCH_REFUSAL':
                t=e['timing'];s=e['rejected_self_state']
                refusals.append(dict(stage=stage(t['preparation_started_monotonic_s']),sequence=e['feedback_sequence'],attempt=e['attempt'],
                    state_receipt_monotonic_s=s['host_receipt_monotonic_s'],state_timestamp_ns=s['state_timestamp_ns'],
                    rejection_checked_monotonic_s=t['rejection_checked_monotonic_s'],
                    state_age_at_rejection_s=t['rejection_checked_monotonic_s']-s['host_receipt_monotonic_s'],
                    rejection_boundary=t['rejection_boundary'],actual_zero_dispatch_monotonic_s=e['actual_zero_dispatch_monotonic_s'],
                    computed_rejected_vector=e['rejected_feedback'].get('command_velocity_xy_m_s') if e['rejected_feedback'] else None,
                    saved_read_timing=t.get('read_timing'),target_lease_renewed=e['target_lease_renewed']))
        groups={}
        for name in ('ALL_PERCEPTION_XY','ACQUISITION','MEASURED_PHASE','FINALIZATION'):
            chosen=lambda r:name=='ALL_PERCEPTION_XY' or r['stage']==name
            rr=[r for r in reads if chosen(r)];ff=[r for r in feedback if chosen(r)];xx=[r for r in refusals if chosen(r)]
            groups[name]=dict(read_count=len(rr),feedback_count=len(ff),refused_attempts=len(xx),recovered_ticks=sum(f['attempt']==1 for f in ff),
                read_components={k:stats([r[k] for r in rr]) for k in ('query_s','validation_s','condition_wait_s','condition_inside_s','state_event_admission_s')},
                feedback_components={k:stats([r[k] for r in ff]) for k in ('preparation_s','receipt_to_preparation_s','preparation_to_dispatch_s','state_age_at_dispatch_s')},
                rejection_age=stats([r['state_age_at_rejection_s'] for r in xx]))
        rows.append(dict(session_id=source['session_id'],independent_status=source['independent_scale_qualification'],
            dispatch_audit=source['feedback_dispatch_audit'],main_started=len(raw['frames']),
            main_completed=sum(r.get('status')=='FRAME_COMPLETED' for r in raw['frames']),planned=len(raw['slots']),
            counts=source['base_evidence']['measured_identity_counts'],net_mm=source['endpoint']['net_xy_displacement_m']*1000
                if source['endpoint']['net_xy_displacement_m'] is not None else None,
            errors=raw['errors'],safe_closed=source['base_evidence']['fresh_ground_closure_verified'] and source['base_evidence']['engine_closed_after_safe_completion'],
            groups=groups,read_rows=reads,feedback_rows=feedback,refusal_rows=refusals,
            full_timing_counts=source['timing_summary']['historical_full']['counts'],
            processing_timing_counts=source['timing_summary']['frame_processing']['counts'],
            natural_recovery_status='OBSERVED_BOUND_FRESH_RETRY' if any(r['attempt']==1 for r in feedback) else 'NOT_TRIGGERED_IN_THIS_FLIGHT'))
    output.mkdir(parents=True)
    result=dict(kind='T1_FEEDBACK_DISPATCH_COMPONENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),runs=rows,input_receipts=inputs,
        limits=dict(freshness_s=.12,maximum_fresh_retries=1),new_flights=0,cloud_requests=False,formal_efficacy_results=False,
        limitations=['Host-local dispatch check precedes RPC submission; no server application clock or hard realtime guarantee.',
            'Groups distinguish acquisition, declared measured time window and finalization; all attempts remain.',
            'Intervals do not establish CPU/GIL/disk/RPC root cause; percentile sums are not latency decompositions.',
            'Absent natural refusals mean recovery was not triggered, even if synthetic tests passed.'])
    save(output/'report.json',result);shutil.copyfile(__file__,output/Path(__file__).name)
    save(output/'artifact_receipt.json',dict(inputs=inputs,files=[dict(path=str(p),sha256=sha(p)) for p in output.iterdir() if p.is_file()]))
    print([(r['session_id'],r['groups']['ALL_PERCEPTION_XY']['refused_attempts'],r['groups']['ALL_PERCEPTION_XY']['recovered_ticks']) for r in rows])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.analysis,a.output)
