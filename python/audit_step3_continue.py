"""Postflight development audit. Saved inputs never become online evidence."""
import sys,time,json,copy,threading,statistics,platform
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
from analyze_step3_development import trajectory,frames,lines,csv_save
from fixed_map_perception_task import build_preview
from fixed_map_perception_geometry import perception_safety_parameters
from fixed_map_perception_execution import static_geometry_v2,TrackingMarginGeometry

def main():
    root=Path(sys.argv[1]).resolve();out=root/'realtime_planning_audit';out.mkdir(exist_ok=False)
    flights=[]
    for folder in sorted(root.rglob('timestamped.jsonl')):
        if 'flight_' not in folder.parent.name and 'safety_recovery_' not in folder.parent.name:continue
        metric=trajectory(folder.parent,out);events=lines(folder);ramp=[];bad_dt=0
        for r in events:
            if r['kind']!='velocity_command_requested' or 'command_ramp_dt_s' not in r:continue
            dt=r['command_ramp_dt_s'];delta=float(np.linalg.norm(np.array(r['world_xyz_m_s'])-r['previous_command_xyz_m_s']))
            value=delta/dt if dt>0 else None
            bad_dt+=dt<=0
            ramp.append(dict(host_monotonic_s=r['host_monotonic_s'],dt_s=dt,command_delta_m_s=delta,recomputed_ramp_accel_m_s2=value,zero_dt=dt<=0,raw_logged_acceleration_finite=bool(np.isfinite(r.get('command_ramp_accel_m_s2'))) if r.get('command_ramp_accel_m_s2') is not None else False))
        if ramp:
            csv_save(out/(folder.parent.name+'_command_ramp.csv'),ramp)
            metric.update(command_ramp_max_recomputed_m_s2=max([r['recomputed_ramp_accel_m_s2'] for r in ramp if r['dt_s']>0],default=None),zero_dt_command_records=bad_dt,command_acceleration_interpretation='Recomputed from exact generation dt and previous command; zero dt undefined and preserved, not counted as measured acceleration')
        flights.append(metric)
    summaries=[];index=[]
    for folder in sorted(root.glob('*')):
        if not (folder/'frames').is_dir() or not (folder/'version_manifest.json').exists():continue
        summary,ix,valid=frames(folder,out);summaries.append(summary);index+=ix
        rr=[read(p) for p in sorted((folder/'frames').glob('*/perception.json'))]
        durations=[r['inference_wall_s'] for r in rr if r.get('inference_wall_s') is not None]
        summary['inference_wall_median_s']=statistics.median(durations) if durations else None
        summary['inference_wall_max_s']=max(durations,default=None)
    (out/'frame_evidence_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in index),encoding='utf-8')
    save(out/'perception_metrics.json',summaries);save(out/'trial_metrics.json',flights)
    csv_save(out/'trial_metrics.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in flights])
    recovery=read(root/'interruption_recovery_v6/unowned_recovery_result.json')
    events=lines(root/'base_runtime_v6/flight_20261010_155648/timestamped.jsonl')
    interruption=dict(utc=utc(),status='INTERRUPTED_INCOMPLETE_NOT_PASSED',final_runner_receipt_missing=not (root/'base_trial_v6_result.json').exists(),owner_absent_observed=True,owner_exit_cause='not established; execution session disappeared during conversation interruption',last_logged_state=next(r for r in reversed(events) if r['kind']=='state'),perception_starts=sum(r['kind']=='PERCEPTION_START_ACCEPTED' for r in events),new_takeoffs=sum(r['kind']=='TAKEOFF_REQUESTED' for r in events),separate_safety_recovery=recovery,does_not_claim_trial_completed=True)
    interruption['new_takeoffs']=int(any(r['kind']=='state' and r['stage']=='TAKEOFF' for r in events))
    interruption['takeoff_evidence']='OBSERVATION_START_ACCEPTED + actual TAKEOFF states + TAKEOFF_REACHED; no additional takeoff in recovery'
    save(root/'base_trial_v6_interruption_audit.json',interruption)
    # Historic sensor input, original processing clock, independent current RPC
    # reads. This benchmark issues no flight/scene/camera change commands.
    src=ROOT/'reports/step3_development_20261010_101819/online_designated_positive_v4.json'
    recorded=read(src);row=recorded['report'];original=row['processed_monotonic_s'];cfg=read(ROOT/'configs/fixed_map_perception_v6.json')
    safety=perception_safety_parameters(read(HIST/'fixed_version/task.json'))
    parts,_=static_geometry_v2(read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives']);g=TrackingMarginGeometry(parts,.75,.15)
    import airsim
    c=airsim.MultirotorClient(port=41454,timeout_value=3)
    states=[];timings=[];plans=[];done=threading.Event();error=[]
    def state_reader():
        try:
            while not done.is_set():
                start=time.monotonic();a=c.getMultirotorState(vehicle_name='DroneA');b=c.getMultirotorState(vehicle_name='DroneB')
                if a.landed_state!=b.landed_state or a.landed_state!=airsim.LandedState.Landed:raise ValueError('BENCHMARK_REQUIRES_BOTH_GROUND')
                states.append(dict(host_monotonic_s=start,latency_s=time.monotonic()-start,timestamp_ns=a.timestamp,DroneA_landed=True,DroneB_landed=True));done.wait(.1)
        except Exception as ex:error.append(type(ex).__name__+':'+str(ex))
    t=threading.Thread(target=state_reader);t.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as worker:
            for i in range(5):
                begin=time.monotonic();p=worker.submit(build_preview,copy.deepcopy(row),recorded['actual_self']['position_xyz_m'],g,safety,cfg,original).result()
                timings.append(dict(iteration=i,wall_s=time.monotonic()-begin,mode='HISTORICAL_RGB_DEPTH_REPLAY_ORIGINAL_TIME_NOT_ONLINE',source_frame_id=row['frame_id'],source_image_sha256=row['capture']['image_sha256'],plan_vertices=len(p['path'])));plans.append(p)
    finally:done.set();t.join(timeout=4)
    csv_save(out/'replay_planning_latency.csv',timings);csv_save(out/'ground_rpc_during_replay_planning.csv',states)
    benchmark=dict(mode='GROUND_RPC_PLUS_ASYNC_HISTORICAL_PLANNER_BENCHMARK_NOT_FLIGHT_VALIDATION',source=str(src),source_sha256=sha(src),source_original_processing_monotonic_s=original,actual_benchmark_monotonic_s=time.monotonic(),planning_runs=len(timings),median_wall_s=statistics.median(r['wall_s'] for r in timings),max_wall_s=max(r['wall_s'] for r in timings),current_evidence_deadline_s=.5,fresh_handoff_budget_s=2.,ground_rpc_count=len(states),ground_rpc_max_latency_s=max([r['latency_s'] for r in states],default=None),ground_rpc_max_gap_s=max([b['host_monotonic_s']-a['host_monotonic_s'] for a,b in zip(states,states[1:])],default=None),errors=error,flight_commands_issued=False,truth_used=False,scope='timing and thread separation only; no current randomized target confirmation or execution proof',last_replay_plan=plans[-1])
    save(out/'replay_benchmark.json',benchmark)
    decision=dict(utc=utc(),status='BLOCKED_CURRENT_RANDOM_SCENE_DESIGNATED_DETECTION',current_scene=read(ROOT/'configs/fixed_map_scene_state.json'),online_perception=summaries,perception_execution_starts=interruption['perception_starts'],flight_trial_status=interruption['status'],both_aircraft_ground=recovery['state']['landed'] and recovery['state']['DroneB_landed'],api_released=not recovery['api_control'] and not recovery['state']['DroneB_api_control'],realtime_planning_scope='asynchronous guarded interfaces and replay timing only; online sensor-to-trajectory capability not established',disturbance_trials='NOT_RUN_BASE_NOT_PASSED',four_method_pilot_ready=False)
    save(root/'STEP3_DECISION.json',decision)
    paths=list((ROOT/'python').glob('fixed_map_*.py'))+[ROOT/'python/audit_step3_continue.py',ROOT/'python/step3_recover_unowned.py',ROOT/'python/run_fixed_map_perception_base.py',ROOT/'01_固定地图实验.cmd']+list((ROOT/'configs').glob('fixed_map_*v*.json'))
    detector=read(ROOT/'configs/step3_balloon_cuda1280_v2.json');paths+=[ROOT/'configs/step3_balloon_cuda1280_v2.json',Path(detector['weights'])]
    save(root/'delivery_version_manifest.json',dict(utc=utc(),python=platform.python_version(),files=[dict(path=str(p),sha256=sha(p)) for p in paths if p.is_file()],source_during_actual_trial=str(root/'source_snapshot_v6/hashes.json'),frozen_map_sha256=sha(ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual.umap'),historical_evidence=[str(ROOT/'reports/step1_quick_review_20261010_083715'),str(ROOT/'reports/step2_perception_20261010'),str(ROOT/'reports/step3_development_20261010_101819')],scene_truth_isolation='scenario truth left unread by this audit; no control imports'))
    print(json.dumps(dict(decision=decision,benchmark={k:v for k,v in benchmark.items() if k!='last_replay_plan'},flights=flights),ensure_ascii=False))

if __name__=='__main__':main()
