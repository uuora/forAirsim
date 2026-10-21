"""Close the recorded five-pilot development milestone, preserving history.

This consumes saved receipts only. It neither connects to AirSim/cloud nor
reads credentials. It updates the current thesis task index with bounded
development evidence; all formal research acceptance gates remain separate.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import re

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def utc_tick_string(value):
    # Preserve all seven Windows UTC fractional digits (100 ns). A serializer
    # may omit insignificant trailing zeros without changing the same instant.
    match=re.fullmatch(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,7}))?Z',value)
    if not match:raise ValueError('UNSUPPORTED_OWNER_UTC_FORMAT')
    return match[1]+'.'+(match[2] or '').ljust(7,'0')+'Z'


def closeout(analysis_path,output):
    output=output.resolve()
    if output.exists():raise ValueError('NEW_CLOSEOUT_DIRECTORY_REQUIRED')
    analysis=read(analysis_path);runs=analysis['runs']
    if len(runs)!=5 or analysis['formal_efficacy_results'] is not False:
        raise ValueError('FIVE_DEVELOPMENT_ATTEMPTS_REQUIRED')
    artifact_receipt=analysis_path.parent/'artifact_receipt.json'
    for item in read(artifact_receipt)['artifacts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('FINAL_ANALYSIS_ARTIFACT_MISMATCH')
    passed=[r for r in runs if r['evidence_gate_status']=='PASS_RECORDED_DEVELOPMENT_INTEGRATION']
    if len(passed)!=2 or not all(r['observed_airborne_from_execution'] for r in runs):
        raise ValueError('ACTUAL_START_OR_INTEGRATION_GATE_MISMATCH')
    latest=next(r for r in runs if r['session_id']=='t1_identity_flight_short_v1p3_20261004')
    if latest not in passed or latest['audit_issues'] or latest['captured_frames']!=40:
        raise ValueError('LATEST_DEVELOPMENT_EVIDENCE_REJECTED')
    closures=[];known_cloud=0;unknown_cloud_sessions=[]
    for run in runs:
        folder=Path(run['run']);report_path=Path(run['report_path'])
        if folder!=report_path.parent.parent:raise ValueError('RUN_LOCATION_MISMATCH')
        manifest=read(folder/'manifest.json');raw=read(report_path)
        if manifest.get('engine_closed_after_safe_completion') is True:
            closure_path=Path(manifest['fresh_ground_closure_path']);proof=read(closure_path)
            assert proof['status']=='READY' and proof['grounded_and_released'] and proof['paused']
            assert digest(closure_path)==manifest['fresh_ground_closure_sha256']
        else:
            closure_path=folder/'supplemental_owned_closure.json';proof=read(closure_path)
            assert proof['status']=='PASS' and proof['engine_closed_after_supplemental_ground_recovery']
            assert proof['engine_process_id']==manifest['engine_process_id']
            assert utc_tick_string(proof['verified_started_utc'])==utc_tick_string(manifest['engine_started_utc'])
            recoveries=[p for p in folder.glob('supplemental_ground_recovery*.json') if read(p)['status']=='PASS']
            assert recoveries and all(read(p)['grounded_and_released'] and read(p)['paused'] for p in recoveries)
        closures.append(dict(session_id=run['session_id'],original_status=manifest['status'],
            verified_closure=str(closure_path),sha256=digest(closure_path)))
        closed=raw.get('observer_close')
        if closed:known_cloud+=closed['request_submissions']
        else:unknown_cloud_sessions.append(run['session_id'])
    legacy=read(ROOT/'configs/trajectory_timed_pilot_v6p2/center.json')['flight_source_sha256']
    assert len(legacy)==12
    assert all(digest(ROOT/'python'/name)==value for name,value in legacy.items())
    output.mkdir(parents=True)
    index_path=ROOT/'configs/thesis_simulation_tasks.json'
    protocol_path=ROOT/'configs/t1_completion_protocol_v1.json'
    for path in (index_path,protocol_path,ROOT/'README.md'):
        shutil.copyfile(path,output/(path.name+'.before'))
    index=read(index_path);other_tasks=index['tasks'][1:]
    protocol=read(protocol_path)
    evidence_path=str(analysis_path.relative_to(ROOT)).replace('\\','/')
    summary_path=str((analysis_path.parent/'summary.md').relative_to(ROOT)).replace('\\','/')
    stage=latest['low_speed_stage']
    block=dict(status='MICRO_XY_IDENTITY_DEPTH_INTEGRATION_RECORDED_FORMAL_TIMING_PENDING',
        data_date='2026-10-04',review_update_date='2026-10-05',analysis=evidence_path,
        closeout_receipt=str((output/'receipt.json').relative_to(ROOT)).replace('\\','/'),
        observed_airborne_attempts_all=5,development_gate_pass_attempts=2,retained_failed_attempts=3,
        planned_slots_all=sum(r['planned_slots'] for r in runs),captured_groups_all=sum(r['captured_frames'] for r in runs),
        media_original_hash_verified_all=sum(r['media_files_original_hash_verified'] for r in runs),
        media_analysis_locked_without_original_seal_all=sum(r['media_files_locked_at_analysis_without_original_seal'] for r in runs),
        known_cloud_submissions_all=known_cloud,cloud_submission_unknown_sessions=unknown_cloud_sessions,
        latest_planned_slots=latest['planned_slots'],latest_captured_frames=latest['captured_frames'],
        latest_identity_point_counts=latest['identity_point_counts'],
        latest_target_motion_commands=latest['command_counts']['target_commands_nonzero'],
        latest_max_command_xy_m_s=latest['command_counts']['max_commanded_xy_speed_m_s'],
        latest_max_actual_xy_m_s=stage['max_actual_xy_speed_m_s'],
        latest_sampled_xy_path_m=stage['sampled_xy_path_length_m'],
        latest_max_fixed_height_error_m=stage['max_fixed_height_error_m'],
        latest_full_frame_late=latest['late_completed_decisions'],latest_observer_late=latest['late_observer_returns'],
        latest_gate_decisions_late=latest['gate_decisions_late'],latest_gate_decisions_timely=latest['gate_decisions_timely'],
        latest_missed_or_unknown_gate_slots=latest['gate_decision_unknown_slots'],
        latest_transition_recovery=latest['actual_transition_recovery'],
        latest_lease_clear=latest['target_lease_clear_summary'],
        specialist_weights_sha256='92b0dc0f4e17ff14e9c1e110b9cad9ef249274136ee9a28d99b7885dfb396844',
        all_attempts_finally_grounded_released_paused_owned_closed=True,
        formal_method_effect_complete=False,whole_thesis_complete=False,
        development_report='docs/research-plan-20261004/任务一机载身份深度与低速轨迹集成结果.md',
        next_gate='decouple_own_state_controller_and_blocking_perception_then_latency_and_four_method_pilots')
    index.update(updated='2026-10-05',active_stage='task_1_perception_control_scheduling_and_four_method_development')
    t1=index['tasks'][0]
    t1.update(status='GROUND_AND_MICRO_XY_IDENTITY_DEPTH_DEVELOPMENT_VERIFIED_FORMAL_T1_PENDING',
              next_gate=block['next_gate'],airborne_identity_depth_development_20261004=block)
    assert index['tasks'][1:]==other_tasks
    for item in protocol['acceptance_map']:
        if item['id']=='A5':
            item.update(status='GROUND_AND_MICRO_XY_SEMANTIC_INTERFACE_VERIFIED_FORMAL_LOOP_TIMING_PENDING',
                airborne_development_evidence=evidence_path,
                scope_limit='Saved-frame binding and attribute-cued micro-XY recovery verified; full-cycle .4s timing still pending; no instance guarantee')
        elif item['id']=='A6':
            item.update(status='GROUND_AND_MICRO_XY_SAVED_SLOT_SCORING_VERIFIED_FORMAL_FLIGHT_TRUTH_PENDING',
                airborne_development_evidence=evidence_path,
                scope_limit='Latest 40/41 slots; point labels, late/unknown and actual gate/clear times kept separately; formal fixed-tick identity coverage pending')
        elif item['id']=='A3':
            item.update(development_surface_replay='logs/t1_temporal_surface_replay_20261004/summary.json',
                        airborne_development_evidence=evidence_path)
    protocol['latest_development_review_date']='2026-10-05'
    protocol['latest_airborne_development']=block
    assert protocol['formal_method_effect_results_completed'] is False
    save(index_path,index);save(protocol_path,protocol)
    receipt=dict(status='PASS_RECORDED_DEVELOPMENT_CLOSEOUT',review_update_date='2026-10-05',
        created_utc=datetime.now(timezone.utc).isoformat(),scope='five development attempts; no formal effect or thesis-completion claim',
        analysis=str(analysis_path.resolve()),analysis_sha256=digest(analysis_path),artifact_receipt_sha256=digest(artifact_receipt),
        flight_summary=block,closures=closures,tasks_2_and_3_semantically_unchanged=True,
        legacy_twelve_sources_unchanged=legacy,unit_test_counts=dict(interface_geometry_and_policy=85,independent_analysis=18),
        backups=[dict(path=str(output/(p.name+'.before')),sha256=digest(output/(p.name+'.before'))) for p in (index_path,protocol_path,ROOT/'README.md')],
        updated_config_hashes={str(p.relative_to(ROOT)):digest(p) for p in (index_path,protocol_path)},
        source_sha256=digest(Path(__file__)),ai_assistance='AI assisted implementation, analysis, figures and evidence mapping')
    save(output/'receipt.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    r=closeout(args.analysis.resolve(),args.output)
    print(json.dumps(dict(status=r['status'],flights=r['flight_summary']['observed_airborne_attempts_all'],
                         planned=r['flight_summary']['planned_slots_all']),ensure_ascii=False))
