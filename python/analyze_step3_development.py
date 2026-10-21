"""Postflight audit only. Never imported by perception or flight control.

Original frame timestamps are used to AUDIT saved decisions, not to make
recorded images fresh. No target Actor queries or fabricated trajectories.
"""
import bisect,csv,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
from fixed_map_perception import read,save,sha,utc,ROOT,HIST
from fixed_map_perception_geometry import validate_pair,make_proposal,preview_task,perception_safety_parameters
from fixed_map_perception_execution import static_geometry_v2
from fixed_map_mission import Geometry

def lines(path):
    if not path.exists():return []
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def csv_save(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def norm(value):return float(np.linalg.norm(value))

def trajectory(folder,out):
    events=lines(folder/'timestamped.jsonl');states=[];commands=[];phase='observation_preparation';plan=None;binding=None;last_state=None;previous_command=None;acceleration=[];stops=[];holds=[];first_stop=None
    task=read(folder/'task.json') if (folder/'task.json').exists() else read(ROOT/'reports/fixed_map_flight_20261009/fixed_version/task.json')
    goal=task['goal_xyz_m'];planning_status=[]
    for r in events:
        kind=r['kind']
        if kind=='PERCEPTION_START_ACCEPTED':phase='perception_guidance';plan=r['plan_version'];goal=r['task']['goal_xyz_m']
        if kind=='PERCEPTION_MOTION_EVIDENCE':binding=r['binding']
        if kind in ['PERCEPTION_PLAN_REJECTED','PERCEPTION_TASK_REJECTED']:planning_status.append(r['reason'])
        if kind=='HOVER_COMMAND':
            first_stop=dict(host=r['host_monotonic_s'],reason=r['reason'],first_below=None,first_timestamp=None,maximum_drift=0.,hold_origin=None)
            stops.append(first_stop)
        if kind=='HOVER_VERIFIED':holds.append(dict(stage=r['stage'],stable_sim_s=r['stable_hold_sim_s'],drift_m=r['displacement_m']))
        if kind=='state':
            p=r['position_xyz_m'];gap=0. if last_state is None else r['host_monotonic_s']-last_state['host_monotonic_s'];sim_dt=None if last_state is None else (r['state_timestamp_ns']-last_state['state_timestamp_ns'])/1e9
            acc=None if not sim_dt or sim_dt<=0 else norm(np.array(r['velocity_xyz_m_s'])-last_state['velocity_xyz_m_s'])/sim_dt
            row=dict(utc=r['utc'],host_monotonic_s=r['host_monotonic_s'],state_timestamp_ns=r['state_timestamp_ns'],stage=r['stage'],task_phase=phase,plan_version=plan,x_m=p[0],y_m=p[1],z_m=p[2],vx_m_s=r['velocity_xyz_m_s'][0],vy_m_s=r['velocity_xyz_m_s'][1],vz_m_s=r['velocity_xyz_m_s'][2],speed_recomputed_m_s=norm(r['velocity_xyz_m_s']),acceleration_state_difference_m_s2=acc,goal_x_m=goal[0],goal_y_m=goal[1],goal_z_m=goal[2],goal_error_recomputed_m=norm(np.array(p)-goal),envelope_clearance_m=r['envelope_clearance_calculated_m'],state_gap_host_s=gap,landed=r['landed'],DroneB_landed=r['DroneB_landed'],DroneB_api_control=r['DroneB_api_control'],collision=r['collision_has_collided'],stop_reason=r['stop_reason'])
            states.append(row)
            if first_stop:
                if first_stop['first_below'] is None and row['speed_recomputed_m_s']<=task['arrival_speed_m_s']:
                    first_stop['first_below']=r['host_monotonic_s']-first_stop['host'];first_stop['first_timestamp']=r['state_timestamp_ns'];first_stop['hold_origin']=p
                if first_stop['hold_origin'] and r['stage'] in ['STOP_HOVER','FAILED_HOVER','ARRIVAL_HOVER','ABORTED','ARRIVED']:
                    first_stop['maximum_drift']=max(first_stop['maximum_drift'],norm(np.array(p)-first_stop['hold_origin']))
            last_state=r
        if kind=='velocity_command_requested':
            value=r['world_xyz_m_s'];derivative=None
            if previous_command and r['host_monotonic_s']>previous_command['host_monotonic_s'] and r['stage']==previous_command['stage']:
                derivative=norm(np.array(value)-previous_command['world_xyz_m_s'])/(r['host_monotonic_s']-previous_command['host_monotonic_s']);acceleration.append(derivative)
            use_binding=binding if phase=='perception_guidance' and r['stage']=='APPROACH' else None
            commands.append(dict(host_monotonic_s=r['host_monotonic_s'],stage=r['stage'],task_phase=phase,source_state_timestamp_ns=r.get('source_state_timestamp_ns'),vx_m_s=value[0],vy_m_s=value[1],vz_m_s=value[2],command_speed_m_s=norm(value),requested_time_acceleration_difference_m_s2=derivative,duration_sim_s=r.get('duration_sim_s'),plan_version=plan,source_frame_id=use_binding.get('frame_id') if use_binding else None,source_image_sha256=use_binding.get('image_sha256') if use_binding else None,target_generation=use_binding.get('target_generation') if use_binding else None,source_evidence_age_s=use_binding.get('evidence_age_s') if use_binding else None))
            previous_command=r
    stem=folder.parent.name+'_'+folder.name;csv_save(out/(stem+'_trajectory.csv'),states);csv_save(out/(stem+'_commands.csv'),commands)
    moving=[r for r in states if r['stage'] in ['APPROACH','RETURN']]
    pathlength=sum(norm(np.array(b['position_xyz_m'])-a['position_xyz_m']) for a,b in zip([r for r in events if r['kind']=='state'],[r for r in events if r['kind']=='state'][1:]))
    result=dict(trial=stem,raw=str(folder/'timestamped.jsonl'),state_count=len(states),command_count=len(commands),actual_trajectory_length_m=pathlength,max_speed_m_s=max([r['speed_recomputed_m_s'] for r in states],default=0),max_command_speed_m_s=max([r['command_speed_m_s'] for r in commands],default=0),minimum_approach_return_envelope_clearance_m=min([r['envelope_clearance_m'] for r in moving],default=None),max_state_gap_s=max([r['state_gap_host_s'] for r in states],default=0),max_requested_time_acceleration_difference_m_s2=max(acceleration,default=None),command_acceleration_interpretation='requested-event-time derivative includes logging/transport jitter; original0.4 control ramp retained by source; exact generator dt not logged, so strict realized ramp validation is NOT established',configured_command_acceleration_m_s2=task['max_command_accel_m_s2'],perception_command_count=sum(bool(r['source_frame_id']) for r in commands),hover_proofs=holds,stop_response=stops,planning_rejections=planning_status,returned_release_event=any(r['kind']=='RETURN_LANDED_RELEASED' for r in events),failures=[r['reason'] for r in events if r['kind']=='EXECUTION_FAILED'])
    active={'TAKEOFF','APPROACH','RETURN','LANDING'}
    result['max_continuous_motion_state_gap_s']=max([b['host_monotonic_s']-a['host_monotonic_s'] for a,b in zip(states,states[1:]) if a['stage'] in active and b['stage'] in active],default=0.)
    result['clearance_interpretation']='planned static path threshold1.25m; actual net clearance reported independently, not rounded up; actual stop threshold0.75m unchanged'
    return result

def frames(folder,out):
    cfg=read(folder/'version_manifest.json')['config'];rows=[];reject=Counter();valid=[];candidates=0;fresh_candidates=0;surfaces=0;certificate_failures=[];index=[];attribute_counts=Counter()
    from t1_delayed_semantic_bridge import _frame_from_dict,verify_delayed_selection
    for path in sorted((folder/'frames').glob('*/perception.json')):
        r=read(path);m=r['capture'];estimate=r.get('position_estimate',{});sf=r.get('semantic_frame',{});p=estimate.get('surface_world_xyz_m',[None]*3);ts=r.get('processed_monotonic_s',m['received_monotonic_s']);pair_errors=validate_pair(m,cfg,ts)
        count=len(sf.get('candidates',[]));candidates+=count>0;fresh_candidates+=count>0 and not pair_errors
        surfaces+=sum(c['position'].get('valid',False) for c in r.get('candidate_positions',[]));reject.update(r.get('rejection_reasons',[]))
        certificate='NOT_TRACKING'
        if estimate.get('valid'):
            try:
                if cfg.get('semantic_policy_version')=='delayed_chain_attributes_v3':verify_delayed_selection(_frame_from_dict(sf),r['temporal_identity'],cfg['target_description'],ts)
                certificate='SAVED_CERTIFICATE_VALID_AT_ORIGINAL_PROCESSING_TIME';valid.append(path)
            except ValueError as ex:certificate=str(ex);certificate_failures.append(dict(frame_id=r['frame_id'],reason=str(ex)))
        attr=r.get('current_attribute_audit',{});attribute_counts[attr.get('reason','NONE')]+=1
        item=dict(session=folder.name,frame_id=r['frame_id'],mode='online_original_record',target_generation=m['target_generation'],rgb=str(path.parent/'rgb.png'),depth=str(path.parent/'depth_ray_range_m.npy'),pose=str(path.parent/'capture.json'),perception=str(path),image_sha256=m['image_sha256'],depth_sha256=m['depth_sha256'],rgb_timestamp_ns=m['rgb_timestamp_ns'],depth_timestamp_ns=m['depth_timestamp_ns'],readback_delta_ms=m['rgb_depth_delta_ms'],source_received_monotonic_s=m['received_monotonic_s'],processed_monotonic_s=ts,source_age_at_original_processing_s=ts-m['received_monotonic_s'],inference_wall_s=r.get('inference_wall_s'),candidate_count=count,valid_candidate_surface_count=sum(c['position'].get('valid',False) for c in r.get('candidate_positions',[])),target_state=r.get('temporal_identity',{}).get('state'),local_track_id=r.get('temporal_identity',{}).get('track_id'),estimate_valid_at_original_record=r.get('valid',False),surface_x_m=p[0],surface_y_m=p[1],surface_z_m=p[2],attribute_status=attr.get('status'),semantic_anchor=estimate.get('semantic_anchor'),semantic_status=r.get('semantic_poll',{}).get('status'),certificate_audit=certificate,rejection_reasons=';'.join(r.get('rejection_reasons',[])),pair_audit_at_original_time=';'.join(pair_errors))
        rows.append(item);index.append({k:item[k] for k in ['session','frame_id','rgb','depth','pose','perception','image_sha256','depth_sha256','semantic_anchor','certificate_audit']})
    csv_save(out/(folder.name+'_frames.csv'),rows)
    ev=lines(folder/'events.jsonl');requests={r['binding']['request_id']:r for r in ev if r['kind']=='VLX_REQUEST'};transport={r['binding']['request_id']:r for r in ev if r['kind']=='VLX_TRANSPORT_IMAGE'};delays=[]
    for r in ev:
        if r['kind']!='VLX_REPLY':continue
        b=r['binding'];req=requests[b['request_id']];tr=transport.get(b['request_id'],{})
        delays.append(dict(session=folder.name,request_id=b['request_id'],source_frame_id=b['frame_id'],target_generation=b['target_generation'],description_sha256=b['description_sha256'],source_image_sha256=b['image_sha256'],transmitted_sha256=tr.get('transmitted_image_sha256'),transmitted_path=tr.get('path'),request_monotonic_s=b['submitted_monotonic_s'],reply_monotonic_s=r['completed_monotonic_s'],response_elapsed_s=r['completed_monotonic_s']-b['submitted_monotonic_s'],source_age_at_reply_s=r['completed_monotonic_s']-b['readback_receipt_monotonic_s'],response_budget_s=cfg['vlx_response_timeout_s'],reply_within_transport_budget=r['completed_monotonic_s']-b['submitted_monotonic_s']<=cfg['vlx_response_timeout_s'],reply_box_count=len(r['result'].get('result_bbox_list',[])),binding_scope='client_assigned_not_server_echo'))
    csv_save(out/(folder.name+'_vlx_latency.csv'),delays)
    summary=dict(session=folder.name,processed_frames=len(rows),candidate_frames=candidates,fresh_candidate_frames=fresh_candidates,valid_candidate_surfaces=surfaces,valid_designated_frames=len(valid),certificate_failures=certificate_failures,rejections=dict(reject),attribute_reasons=dict(attribute_counts),vlx_completed=len(delays),vlx_min_s=min([d['response_elapsed_s'] for d in delays],default=None),vlx_max_s=max([d['response_elapsed_s'] for d in delays],default=None),vlx_mean_s=float(np.mean([d['response_elapsed_s'] for d in delays])) if delays else None)
    return summary,index,valid

def main():
    root=Path(sys.argv[1]).resolve();out=root/'audit';out.mkdir(exist_ok=True)
    flight_summaries=[]
    for folder in sorted(root.glob('*runtime*/flight_*')):flight_summaries.append(trajectory(folder,out))
    for folder in sorted(root.glob('*runtime*/safety_recovery_*')):flight_summaries.append(trajectory(folder,out))
    sessions=[];fullindex=[];selected=[]
    for folder in sorted(root.glob('online_*')):
        if not folder.is_dir() or not (folder/'version_manifest.json').exists():continue
        summary,index,valid=frames(folder,out);sessions.append(summary);fullindex+=index
        if valid:
            # Few linked examples; all complete originals remain in their own
            # frame directories. References, no duplicating large raw files.
            for p in valid[-3:]:selected.append(dict(mode='genuine_online_at_original_time',perception=str(p),capture=str(p.parent/'capture.json'),rgb=str(p.parent/'rgb.png'),depth=str(p.parent/'depth_ray_range_m.npy'),overlay=str(p.parent/'overlay.png')))
    (out/'frame_evidence_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in fullindex),encoding='utf-8')
    save(out/'online_sample_references.json',selected)
    flat=[]
    for r in flight_summaries:flat.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
    csv_save(out/'trial_metrics.csv',flat);save(out/'trial_metrics.json',flight_summaries);save(out/'perception_metrics.json',sessions)
    # Diagnostic replay preview, clearly separate from failed LIVE submission.
    # The original processing time is an offline audit parameter, not now.
    replay=None
    positive=root/'online_designated_positive_v4.json'
    if positive.exists():
        data=read(positive);r=data['report'];cfg=read(root/'base_trial_v4_config.json');task=perception_safety_parameters(read(HIST/'fixed_version/task.json'));parts,_=static_geometry_v2(read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives']);g=Geometry(parts,.75)
        original_now=r['processed_monotonic_s'];estimate=r['position_estimate'];start=data['actual_self']['position_xyz_m']
        try:
            p=make_proposal(estimate,start,task,cfg,original_now);preview=preview_task(p,estimate,start,g,task,cfg,original_now)
            replay=dict(mode='REPLAY_PLANNING_DIAGNOSTIC_NOT_LIVE_OR_EXECUTED',original_frame_id=r['frame_id'],source_rgb_sha256=r['capture']['image_sha256'],original_processing_time_s=original_now,actual_current_time_s=time.monotonic(),result=preview,position_meaning='visible_surface_NOT_center',safety_scope='audited static geometry only; excludes semantic bodies, no complete unseen-object proof')
        except ValueError as ex:replay=dict(mode='REPLAY_PLANNING_DIAGNOSTIC_NOT_LIVE',reason=str(ex))
        save(out/'replay_planning_preview.json',replay)
    latest=read(root/'base_trial_v4_result.json');decision=dict(utc=utc(),status='BLOCKED_LIVE_PLANNING_FRESHNESS_NOT_READY_FOR_FOUR_METHOD_PILOT',aircraft_landed=latest['final_state']['landed'],api_control_released=latest['api_control_released'],DroneB_landed=latest['final_state']['DroneB_landed'],DroneB_control=latest['final_state']['DroneB_api_control'],blocking_reason=latest['issue'],online_positive_exists=latest['online_designated_confirmed'],perception_guidance_started=latest['perception_started'],perception_arrival=False,disturbances_A_B_C='NOT_RUN_BASE_NOT_PASSED',no_formal_algorithm_benefit_claim=True)
    save(root/'STEP3_DECISION.json',decision)
    save(out/'summary.json',dict(flights=flight_summaries,perception=sessions,decision=decision,replay=replay))
    print(json.dumps(dict(decision=decision,perception=sessions,flights=[{k:r[k] for k in ['trial','max_speed_m_s','max_command_speed_m_s','minimum_approach_return_envelope_clearance_m','max_state_gap_s','perception_command_count','returned_release_event']} for r in flight_summaries]),ensure_ascii=False))

if __name__=='__main__':main()
