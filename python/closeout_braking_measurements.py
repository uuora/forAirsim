"""Reconcile this development campaign; update task registry after held-out gates."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def relative(path):return str(Path(path).resolve().relative_to(ROOT)).replace('\\','/')


def closeout(validation,output):
    validation=validation.resolve();summary=read(validation/'summary.json')
    if summary['status']!='VERIFIED' or not summary['gate_passed'] or summary['planned_independent_sessions']!=2 or summary['measured_stable_pulse_count']!=16:
        raise ValueError('Two held-out sessions and all sixteen measured pulses required')
    if any(read(s['configuration'])['protocol_id']!='braking_response_v1p4_saved_query_gate_20261003' for s in summary['sessions']):
        raise ValueError('Do not mix calibration with held-out validation')
    core=read(ROOT/'configs/trajectory_timed_pilot_v6p2/center.json')['flight_source_sha256']
    if any(sha(ROOT/'python'/name)!=digest for name,digest in core.items()):raise ValueError('Frozen V6.2 source changed')
    attempts=[]
    for path in sorted((ROOT/'logs').glob('braking_campaign_20261003_*/manifest.json')):
        manifest=read(path)
        for session in manifest['sessions']:
            row=dict(manifest=relative(path),manifest_sha256=sha(path),configuration=relative(session['configuration']),
                engine_process_id=session.get('engine_process_id'),engine_started=session.get('engine_started'),flight_dispatch_started=False,
                terminal_report_present=False,safe_landing=False,verified_not_started=False,unknown_start=False,engine_closed=False)
            config=read(session['configuration'])
            if not config['protocol_id'].startswith('braking_response_'):raise ValueError('Unexpected protocol in selected campaign')
            if sha(session['configuration'])!=session['configuration_sha256']:raise ValueError('Attempt config changed')
            if session.get('report_path'):
                raw=read(session['report_path']);row.update(terminal_report_present=True,report=relative(session['report_path']),report_sha256=sha(session['report_path']))
                if row['report_sha256']!=session['report_sha256']:raise ValueError('Terminal bytes changed')
                row['flight_dispatch_started']=raw.get('flight_dispatch_started') is True
                landing=raw.get('landing_evidence',{})
                row['safe_landing']=bool(row['flight_dispatch_started'] and raw['status']=='PASS' and raw.get('paused') is True
                    and landing.get('landed_flag') is True and landing.get('api_control_released') is True)
                row['verified_not_started']=bool(raw.get('flight_dispatch_started') is False and raw.get('failed_phase')=='PREFLIGHT'
                    and raw.get('manual_recovery_required') is False)
                row['engine_closed']=session.get('engine_closed_after_safe_completion') is True
            closure_path=path.parent/'prearm_closure_receipt.json'
            if closure_path.exists():
                closure=read(closure_path)
                if closure['status']!='VERIFIED' or closure['engine_process_id']!=row['engine_process_id']:raise ValueError('Pre-arm closure identity mismatch')
                row.update(engine_closed=True,closure=relative(closure_path),closure_sha256=sha(closure_path))
                if closure.get('flight_trial_dispatched') is False:row['verified_not_started']=True
            row['unknown_start']=not row['flight_dispatch_started'] and not row['verified_not_started']
            if not row['engine_closed'] or row['unknown_start']:raise ValueError('Attempt remains unclosed or unknown: '+str(path))
            attempts.append(row)
    if output.exists():raise ValueError('Closeout output must be new')
    output.mkdir(parents=True)
    all_pulses=[p for s in summary['sessions'] for p in s['pulses']]
    record=dict(status='LOW_SPEED_XY_CANDIDATE_HELD_OUT_VALIDATION_COMPLETED',formal_thesis_effect_results=False,
        version='V1.4 workflow; one h=1.0 s candidate chosen from V1.2 development data',
        measurement_methods=1,model_horizon_s=1.0,request_amplitudes_m_s=[.03,.06],source_module_count=19,
        related_tests=58,test_log='logs/braking_response_final_related_tests.txt',
        independent_validation_flights=2,nested_pulses=16,stable_measured_pulses=16,sampled_allowance_pass_pulses=16,
        validation_analysis=relative(validation/'summary.json'),validation_figures=relative(validation/'figure_receipt.json'),
        validation_manifests=[relative(p['path']) for p in summary['manifests']],
        calibration_analysis='logs/braking_campaign_20261003_223639_9536884/analysis/summary.json',
        old_full_plan_disposition='logs/braking_campaign_20261003_223639_9536884/full_plan_disposition.json',
        report='docs/research-plan-20261003/独立制动与响应实测结果.md',
        next_plan='docs/research-plan-20261003/下一轮三维与姿态接口验收计划.md',
        minimum_actual_entry_forward_speed_m_s=min(p['entry_forward_speed_m_s'] for p in all_pulses),
        maximum_actual_entry_forward_speed_m_s=max(p['entry_forward_speed_m_s'] for p in all_pulses),
        all_development_accounting=dict(engine_attempts=len(attempts),known_started_flights=sum(a['flight_dispatch_started'] for a in attempts),
            safe_landing_flights=sum(a['safe_landing'] for a in attempts),verified_not_started_attempts=sum(a['verified_not_started'] for a in attempts),
            unknown_start_count=sum(a['unknown_start'] for a in attempts)),
        limits=['Two independent validation sessions, nested pulses; no formal method efficacy inference',
            'Sampled low-speed XY only; higher amplitude, Z, attitude and continuous safety unverified',
            'Equivalent allowance window is not measured actuation delay or certified constant physical deceleration'])
    registry_path=ROOT/'configs/thesis_simulation_tasks.json';registry=read(registry_path)
    (output/'task_registry_before.json').write_bytes(registry_path.read_bytes())
    registry['active_stage']='task_1_expanded_braking_range_and_camera_pose_scoring_gate'
    task=next(t for t in registry['tasks'] if t['id']=='T1');task['independent_braking_response_measurement']=record
    task['next_gate']='higher_range_Z_response_and_full_3D_pose_identity_visibility_before_formal_T1'
    registry_path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    receipt=dict(kind='BRAKING_DEVELOPMENT_CLOSEOUT',status='VERIFIED',recorded_local=datetime.now().astimezone().isoformat(),
        all_development_accounting=record['all_development_accounting'],attempts=attempts,validation=record,
        unchanged_v6p2_source_count=len(core),task_registry_sha256=sha(registry_path),ai_assistance='AI-assisted code and evidence reconciliation; no thesis acceptance decision')
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(output)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--validation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();closeout(args.validation,args.output)
