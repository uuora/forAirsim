"""One bounded real observation-preparation flight, with mandatory recovery.
No reset/teleport/target Actor input. Does not claim perception-guided arrival.
"""
import sys,time,json
from pathlib import Path
import numpy as np
import fixed_map_mission as mission
from fixed_map_runtime import configure_session
from fixed_map_perception import PerceptionService,read,save,sha,utc
from fixed_map_perception_execution import PerceptionExecutor,static_geometry_v2

def main():
    output=Path(sys.argv[1]).resolve();version=sys.argv[2] if len(sys.argv)>2 else 'v1'
    if (output/('observation_declaration_'+version+'.json')).exists():raise FileExistsError('Trial output already used; original evidence must remain unchanged')
    runtime=configure_session(output/('observation_runtime' if version=='v1' else 'observation_runtime_'+version));mission.ensure_scene()
    cfg=read(mission.ROOT/'configs/fixed_map_perception_v1.json')
    dwell=22.
    if version=='v2':
        cfg.update(schema='step3_online_perception_v2',detector_config='configs/step3_balloon_cuda1280_v2.json',static_geometry_version='v2',camera_reference_version='frozen_self_origin_v2',current_attribute_guard=True,torch_threads=1)
        dwell=35.
        save(output/'online_recheck_v2_declaration.json',dict(utc=utc(),scope='one bounded recheck after GPU1280 repair; observation preparation NOT perception guidance',config=cfg,old_failures_preserved=True,strict_vlx_response_timeout_s=2.,current_max_age_s=.5,camera_conversion='constant Step1 self origin and actual response poses; bracketed motion mount diagnostic, not exposure sync',description_ambiguity='two visible red candidates cause rejection; no autonomous instance refinement'))
    task=read(runtime/'fixed_version/task.json');parts,removed=static_geometry_v2(read(runtime/'fixed_version/editor_audit/planning_geometry.json')['primitives']);g=mission.Geometry(parts,task['envelope_radius_config_m'])
    # Static workspace progress and a wide side corridor, declared before flight.
    goal=[30.,-9.,task['height_world_m']]
    declaration=dict(created_utc=utc(),observation_goal_xyz_m=goal,heading_yaw_deg=0.,selection_basis='predeclared far-middle workspace side corridor; no balloon positions, actors or baseline target used',stage='observation_preparation_NOT_perception_guidance',target_description=cfg['target_description'],deadline_host_s=210.,arrival_and_hover='unchanged frozen task thresholds',maximum_online_dwell_s=dwell,recovery='return via audited static geometry; landed/disarmed/released; never abandon airborne state',static_geometry_version='v2 physical platforms included, all semantic labels anonymized',adapter_sha256=sha(mission.ROOT/'python/fixed_map_perception_execution.py'))
    start=np.array(read(mission.ROOT/'reports/step1_quick_review_20261010_083715/runtime_geometry.json')['DroneA_local_origin_xyz_m']);start[2]=task['height_world_m'];path=g.plan(start,goal,task['path_clearance_m'])
    declaration.update(preflight_path=path,segments=[g.segment(a,b,task['path_clearance_m'])[1] for a,b in zip(path,path[1:])]);save(output/('observation_declaration_'+version+'.json'),declaration)
    service=PerceptionService(output/('online_observation_'+version),cfg);service.start();executor=PerceptionExecutor(declaration,service);executor.start();started=False;arrived=None;return_requested=False;issue=None;deadline=time.monotonic()+210
    try:
        ready_deadline=time.monotonic()+40
        while executor.stage=='CHECKING' and time.monotonic()<ready_deadline:time.sleep(.2)
        if executor.stage!='READY':raise RuntimeError('OBSERVATION_PREFLIGHT_'+executor.reason)
        # Load/warm the model on ground, then use the same independent worker.
        while not service.warm_ready.is_set() and time.monotonic()<ready_deadline:
            if not service.threads[1].is_alive():raise RuntimeError('PERCEPTION_MODEL_NOT_READY')
            time.sleep(.2)
        if not service.warm_ready.is_set():raise RuntimeError('MODEL_WARMUP_DEADLINE')
        executor.command('start','user_authorized_observation_preparation');started=True
        while time.monotonic()<deadline:
            if executor.stage=='ARRIVED':
                if arrived is None:arrived=time.monotonic();save(output/('observation_arrived.json' if version=='v1' else 'observation_arrived_'+version+'.json'),dict(utc=utc(),actual_state=executor.state,run_dir=str(executor.run_dir),scope='observation preparation, not perception-guided success'))
                if time.monotonic()-arrived>=dwell and not return_requested:executor.command('return','bounded_observation_dwell_complete');return_requested=True
            if executor.stage in ['FAILED','FAILED_HOVER']:
                issue=executor.reason;break
            if return_requested and executor.stage=='READY':break
            time.sleep(.2)
        else:issue='OBSERVATION_DEADLINE'
    except Exception as ex:issue=type(ex).__name__+':'+str(ex)
    finally:
        # Stop heavy rendering/inference FIRST before safety recovery.
        service.stop(wait=False)
        if started and executor.stage!='READY':
            if executor.stage in ['TAKEOFF','APPROACH','RETURN','LANDING']:
                executor.command('stop','observation_finally_safe_hold')
            recovery_deadline=time.monotonic()+200
            while time.monotonic()<recovery_deadline and executor.is_alive() and executor.stage!='READY':
                if executor.stage in ['PAUSED','ARRIVED','ABORTED','STOP_HOVER','ARRIVAL_HOVER','FAILED_HOVER']:
                    # Existing return handler permits the normal stable-hold phases.
                    if executor.stage!='FAILED_HOVER':executor.command('return','safety_recovery_after_observation')
                    time.sleep(1)
                time.sleep(.2)
            if executor.stage!='READY':issue=(issue or '')+';RECOVERY_REQUIRES_ATTENTION'
        service.stop(wait=True)
        if executor.stage=='READY':executor.exit_requested=True;executor.join(timeout=4)
        save(output/('observation_result_'+version+'.json'),dict(utc=utc(),issue=issue,observation_arrived=arrived is not None,flight_started=started,final_stage=executor.stage,final_state=executor.state,online_frames=service.process_count,perception_guidance_success=False,run_dir=str(executor.run_dir),control_thread_alive=executor.is_alive(),api_control_released=not executor.c.isApiControlEnabled(vehicle_name='DroneA')))
        print(json.dumps(dict(output=str(output),issue=issue,final_stage=executor.stage,frames=service.process_count),ensure_ascii=False))
if __name__=='__main__':main()
