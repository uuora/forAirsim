"""One bounded prospective base flight; disturbances are separate later trials."""
import copy,json,sys,time,shutil
from pathlib import Path
import numpy as np
import fixed_map_mission as mission
from fixed_map_runtime import configure_session
from fixed_map_perception import read,save,sha,utc,ROOT
from fixed_map_perception_process import PerceptionProcess
from fixed_map_perception_execution import PerceptionExecutor,static_geometry_v2
from fixed_map_perception_task import live_estimate

def main():
    out=Path(sys.argv[1]).resolve()
    version=sys.argv[2] if len(sys.argv)>2 else 'v3'
    if (out/('base_trial_'+version+'_declaration.json')).exists() or (out/('source_snapshot_'+version)).exists():raise FileExistsError('Trial output already used; original evidence must remain unchanged')
    ground_name='ground_v5_recheck_result.json' if version in ['v5','v6'] else 'ground_recheck_v3_result.json'
    driver_name='ground_driver_audit_v5.json' if version in ['v5','v6'] else 'ground_driver_audit_v3.json'
    if read(out/ground_name)['issue'] is not None or read(out/driver_name)['status']!='PASS':raise ValueError('GROUND_RUNTIME_NOT_PASSED')
    runtime=configure_session(out/('base_runtime_'+version));mission.ensure_scene()
    cfg=read(ROOT/('configs/fixed_map_perception_'+version+'.json'));cfg['collection_context']='step3_base_flight'
    if version in ['v5','v6']:cfg['runtime_scene_version']=read(ROOT/'configs/fixed_map_scene_state.json')['scene_version']
    declaration=dict(created_utc=utc(),version='base_trial_v3',observation_goal_xyz_m=[30.,-9.,3.1],heading_yaw_deg=0.,selection_basis='same predeclared static side corridor, independent of target truth',stage='observation_preparation_NOT_perception_guidance',target_description=cfg['target_description'],target_generation=cfg['target_generation'],policy_declaration=str(out/'policy_runtime_v3_declaration.json'),source_kind='online_RGB_depth_bound_white_band_VLX_continuous_chain',surface_meaning='visible_surface_NOT_center',standoff_m=3.5,observed_surface_exclusion_m=3.25,exclusion_basis='original envelope .75 + clearance1.25 + declared location error .5 + unknown balloon/tether horizontal extent .75',limitations=['surface error/extent bounds are engineering restrictions, not measured confidence','only observed objects and audited static geometry: unseen obstacle coverage is NOT proved','fixed-map controller yaw0; strong viewpoint change outside current attribute profile'],deadline_host_s=300.,online_confirmation_wait_host_s=45.,maximum_reacquisition_cycles=3,success='sensor-bound task accepted; actual arrival error<=.2 and speed<=.12 and drift<=.25 for3sim seconds; returned landed/disarmed/API released; original limits retained',failure='missing identity/depth, expired evidence, unsafe route, accelerator error, state gap>.75, collision, original controller gate',disturbances='NOT_STARTED_UNTIL_BASE_PASS',no_reset=True,no_teleport=True,no_target_actor_reads=True)
    declaration['version']='base_trial_'+version
    if version in ['v5','v6']:declaration.update(policy_declaration=str(out/'declaration_v5.json'),runtime_scene_version=cfg['runtime_scene_version'],planning_margin_m=.15,maximum_base_flights=1)
    if version=='v4':declaration['publication_repair']=str(out/'publication_v4_declaration.json')
    save(out/('base_trial_'+version+'_declaration.json'),declaration)
    snapshot=out/('source_snapshot_'+version);snapshot.mkdir(exist_ok=False)
    modules=['fixed_map_mission.py','fixed_map_perception.py','fixed_map_perception_execution.py','fixed_map_perception_task.py','fixed_map_perception_process.py','fixed_map_delayed_identity.py','fixed_map_dashboard.py','run_fixed_map_perception_base.py','fixed_map_perception_geometry.py','camera_geometry.py','t1_pose_geometry.py','specialist_balloon_detector.py','t1_delayed_semantic_bridge.py','t1_source_attribute_guard.py','t1_semantic_perception.py','t1_temporal_identity.py']
    for name in modules:shutil.copyfile(ROOT/'python'/name,snapshot/name)
    save(snapshot/'hashes.json',{p.name:sha(p) for p in snapshot.glob('*.py')});save(out/('base_trial_'+version+'_config.json'),cfg)
    service=PerceptionProcess(out/('online_base_'+version),cfg);service.start();e=PerceptionExecutor(declaration,service);e.start()
    started=False;observation=None;confirmed=False;preview_requested=False;preview_tries=0;execute_requested=False;perception_started=False;arrival=False;issue=None;return_requested=False;deadline=time.monotonic()+300
    try:
        ready_until=time.monotonic()+45
        while (e.stage=='CHECKING' or not service.warm_ready.is_set()) and time.monotonic()<ready_until:
            if e.stage in ['FAILED','FAILED_HOVER']:raise ValueError('PREFLIGHT_'+e.reason)
            if service.process.poll() is not None:raise ValueError('PERCEPTION_PROCESS_NOT_READY')
            time.sleep(.1)
        if e.stage!='READY' or not service.warm_ready.is_set():raise ValueError('GROUND_WARMUP_DEADLINE')
        e.command('start','user_authorized_observation_preparation');started=True
        while time.monotonic()<deadline:
            if e.stage in ['FAILED','FAILED_HOVER']:issue=e.reason;break
            if any('INFERENCE_BLOCKED' in r for r in service.latest.get('rejection_reasons',[])):issue='INFERENCE_BLOCKED';break
            if e.stage=='ARRIVED' and not e.perception_mode:
                if observation is None:
                    observation=time.monotonic();save(out/('base_observation_arrived_'+version+'.json'),dict(utc=utc(),actual_state=e.state,run_dir=str(e.run_dir),not_perception_guidance=True))
                try:
                    estimate=live_estimate(service.latest,cfg,time.monotonic())
                except (ValueError,KeyError,TypeError):
                    if time.monotonic()-observation>45:issue='DESIGNATED_ONLINE_CONFIRMATION_NOT_OBTAINED';break
                else:
                    if not confirmed:
                        confirmed=True;save(out/('online_designated_positive_'+version+'.json'),dict(utc=utc(),report=service.latest,actual_self=e.state,scope='current-map online; visible surface, NOT center'))
                    if not preview_requested:
                        e.plan_perception(False,'base_trial_sensor_preview');preview_requested=True;preview_tries+=1
                    elif e.preview_result and e.preview_result.get('status')=='FRESH_REVALIDATED_PERCEPTION_PLAN' and not execute_requested:
                        e.plan_perception(True,'base_trial_sensor_execute');execute_requested=True
                    elif e.preview_result and e.preview_result.get('status')=='REJECTED':
                        reason=e.preview_result['reason']
                        if version=='v3' and preview_tries<2 and reason in ['CURRENT_FRAME_EXPIRED','NO_VALID_ONLINE_DESIGNATED_ESTIMATE','TARGET_NOT_TRACKING']:
                            preview_requested=False;execute_requested=False
                        else:issue='PERCEPTION_PLAN_REJECTED_'+reason;break
            if e.perception_mode:
                perception_started=True
                if e.stage=='ARRIVED':arrival=True;save(out/('perception_arrival_'+version+'.json'),dict(utc=utc(),actual_state=e.state,active_plan=e.active_plan,current_report=service.latest));e.command('return','base_trial_complete');return_requested=True
                if e.plan_version>8:issue='EXCESSIVE_REACQUISITION_CYCLES';break
                if e.preview_result and e.preview_result.get('status')=='REJECTED' and e.stage=='ABORTED':issue=e.reason;break
            if return_requested and e.stage=='READY':break
            time.sleep(.1)
        else:issue='BASE_TRIAL_DEADLINE'
    except Exception as ex:issue=type(ex).__name__+':'+str(ex)
    finally:
        service.stop(wait=False)
        if started and e.stage!='READY':
            if e.stage in ['TAKEOFF','APPROACH','RETURN','LANDING']:e.command('stop','mandatory_base_trial_safe_hold')
            sent=False;recovery_limit=time.monotonic()+220
            while e.stage!='READY':
                if e.stage in ['PAUSED','ARRIVED','ABORTED','STOP_HOVER','ARRIVAL_HOVER'] and not sent:
                    e.command('return','mandatory_base_trial_return');sent=True
                if not e.is_alive():raise RuntimeError('AIRBORNE_OWNER_EXITED_RECOVERY_REQUIRED')
                if time.monotonic()>recovery_limit:
                    print('AIRBORNE_RECOVERY_REQUIRES_ATTENTION',flush=True);recovery_limit=time.monotonic()+60
                time.sleep(.2)
        service.stop(wait=True)
        if e.stage=='READY':e.exit_requested=True;e.join(timeout=5)
        e.sample();receipt=dict(utc=utc(),issue=issue,observation_arrived=observation is not None,online_designated_confirmed=confirmed,perception_started=perception_started,perception_arrival=arrival,final_stage=e.stage,final_state=e.state,api_control_released=not e.c.isApiControlEnabled(vehicle_name='DroneA'),run_dir=str(e.run_dir),processed_count=service.process_count,source_snapshot=str(snapshot),control_owner_alive=e.is_alive(),no_disturbance_trials_yet=True)
        save(out/('base_trial_'+version+'_result.json'),receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
