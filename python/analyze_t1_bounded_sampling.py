"""Saved V10 component intervals; overlap remains explicit and failures retained."""
import argparse,json,hashlib,math,statistics,shutil
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def stat(values):
    ordered=sorted(values)
    return dict(count=len(values),median_ms=statistics.median(ordered)*1000 if values else None,
        p95_ms=ordered[math.ceil(.95*len(ordered))-1]*1000 if values else None,maximum_ms=max(ordered)*1000 if values else None)
def analyze(analysis,output):
    analysis=Path(analysis).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('NEW_SAMPLING_ANALYSIS_REQUIRED')
    data=json.loads(analysis.read_text(encoding='utf-8'));runs=[];inputs=[dict(path=str(analysis),sha256=sha(analysis))]
    for source in data['runs']:
        if not source['architecture'].startswith('INDEPENDENT_CONTROL_OWN_STATE_BOUNDED_PERSISTENCE'):continue
        path=Path(source['run'])/'flight/report.json'
        expected=next(x['sha256'] for x in source['input_receipts'] if Path(x['path']).resolve()==path.resolve())
        if sha(path)!=expected:raise ValueError('SOURCE_REPORT_CHANGED')
        raw=json.loads(path.read_text(encoding='utf-8'));components={};rows=[]
        for row in raw['frames']:
            intervals={}
            for label,start,end in [('CAPTURE','capture_started_host_s','capture_received_host_s'),
                ('RGB_SAVE_HASH','rgb_persistence_started_monotonic_s','rgb_persistence_completed_monotonic_s'),
                ('SURFACE','surface_started_monotonic_s','surface_completed_monotonic_s'),
                ('OWN_STATE_FETCH','planning_self_state_started_monotonic_s','planning_self_state_completed_monotonic_s'),
                ('PLANNING','planning_started_monotonic_s','planning_completed_monotonic_s'),
                ('CONTROL_SUBMIT_AND_AWAIT','control_submit_started_monotonic_s','motion_decision_available_monotonic_s'),
                ('EVIDENCE_ADMISSION','evidence_enqueue_started_monotonic_s','evidence_enqueue_completed_monotonic_s'),
                ('WORKER_FROM_ADMISSION','evidence_enqueue_started_monotonic_s','evidence_worker_started_monotonic_s'),
                ('TRUTH_EVALUATION','truth_evaluation_started_monotonic_s','truth_evaluation_completed_monotonic_s'),
                ('DEPTH_SEG_SAVE_HASH','depth_segmentation_persistence_started_monotonic_s','depth_segmentation_persistence_completed_monotonic_s'),
                ('RECORD_SERIALIZATION','evidence_record_serialization_started_monotonic_s','evidence_record_serialization_completed_monotonic_s'),
                ('PRODUCER_SPAN','capture_started_host_s','producer_pipeline_completed_monotonic_s'),
                ('ACTUAL_MEDIA_COMPLETION_SPAN','capture_started_host_s','frame_processing_completed_monotonic_s')]:
                if isinstance(row.get(start),(int,float)) and isinstance(row.get(end),(int,float)):
                    value=row[end]-row[start]
                    if value< -1e-9:raise ValueError('NEGATIVE_COMPONENT_INTERVAL')
                    intervals[label]=max(0.,value)
            for span in row.get('observer_result',{}).get('main_thread_component_timings',[]):
                intervals[span['component']]=intervals.get(span['component'],0.)+span['completed_monotonic_s']-span['started_monotonic_s']
            for label,value in intervals.items():components.setdefault(label,[]).append(value)
            rows.append(dict(frame_id=row['frame_id'],slot_index=row['slot_index'],intervals_s=intervals,
                timing=row.get('completion_timing'),start_offset_s=row['capture_started_host_s']-row['due_monotonic_s']))
        traces=raw.get('observer_close',{}).get('sanitized_transport_trace',[])
        http=[t['http_completed_monotonic_s']-t['http_started_monotonic_s'] for t in traces
            if t.get('backend_status')=='REPLY_RETURNED' and t.get('http_completed_monotonic_s') is not None]
        guard=[t['source_attribute_guard_completed_monotonic_s']-t['source_attribute_guard_started_monotonic_s'] for t in traces
            if t.get('source_attribute_guard_completed_monotonic_s') is not None]
        runs.append(dict(session_id=source['session_id'],architecture=source['architecture'],runner_status=raw['status'],
            independent_status=source['independent_scale_qualification'],main_frames=len(rows),planned_main_slots=len(raw['slots']),
            acquisition_frames=len(raw['acquisition_frames']),measurement_entered=raw['measurement_entered'],
            components={key:stat(v) for key,v in components.items()},start_offset=stat([r['start_offset_s'] for r in rows]),
            main_rows=rows,completed_main_frames=sum(f.get('status')=='FRAME_COMPLETED' for f in raw['frames']),
            incomplete_started_main_frames=sum(f.get('status')!='FRAME_COMPLETED' for f in raw['frames']),
            missing_main_slots=[s['slot_index'] for s in raw['slots'] if s.get('status')!='CAPTURED'],
            actual_http_interval=stat(http),actual_http_over_two_s=sum(v>2 for v in http),source_guard=stat(guard),
            current_zero_hold_refusal_count=sum('zero_held_stop_refusal' in f for f in raw['frames']+raw['acquisition_frames']),
            persistence_audit=source.get('bounded_persistence_audit'),evidence_writer=raw.get('evidence_writer'),
            net_mm=source['endpoint']['net_xy_displacement_m']*1000 if source['endpoint']['net_xy_displacement_m'] is not None else None,
            timing_summary=source['timing_summary']))
        inputs.append(dict(path=str(path),sha256=expected))
    output.mkdir(parents=True)
    result=dict(kind='T1_BOUNDED_SAMPLING_COMPONENT_ANALYSIS',created_utc=datetime.now(timezone.utc).isoformat(),runs=runs,input_receipts=inputs,
        limitations=['Nested detection/YOLO intervals and overlapping producer/worker spans are not additive.',
            'Host receipt clocks are not exposure clocks; tied clock readings do not establish zero operation cost.',
            'Python flush does not establish power-loss durability.',
            'These sessions qualify development gates, not formal algorithm efficacy or hard realtime.'],new_flights=0,cloud_requests=False,formal_efficacy_results=False)
    save(output/'report.json',result);shutil.copyfile(__file__,output/Path(__file__).name)
    save(output/'artifact_receipt.json',dict(inputs=inputs,files=[dict(path=str(p),sha256=sha(p)) for p in output.iterdir() if p.is_file()]))
    print([(r['session_id'],r['main_frames'],r['net_mm']) for r in runs]);return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.analysis,a.output)
