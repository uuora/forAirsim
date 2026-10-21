"""One persistent simulator trial; output directories never reused.

Launch detached on Windows so stopping a tool turn cannot reap a flight owner.
stop.request requests a return; it never terminates the airborne owner.
"""
import argparse,sys,time,copy,traceback,os
from pathlib import Path
from fixed_map_cpu_runtime import configure_from_argv
configure_from_argv()
from fixed_map_runtime import configure_session
from fixed_map_perception import ROOT,read,save,utc
from fixed_map_storage import check_output_storage,storage_policy,CHECK_PERIOD_HOST_S

def check_interface_binding(request,bound_config):
    if request is not None and (request.get('engine_identity')!=bound_config.get('runtime_engine_identity') or request.get('scene_version')!=bound_config.get('runtime_scene_version')):
        raise ValueError('INTERFACE_PREDECLARED_SCENE_CHANGED_NO_TAKEOFF')

def run(output,text,config_path='configs/fixed_map_perception_yolo_cpu_v11.json',stop_request_path=None,interface_request_path=None,interface_request_sha256=None):
    out=Path(output).resolve()
    if out.exists():raise FileExistsError('TRIAL_OUTPUT_ALREADY_USED')
    configure_session(out)
    external_stop=Path(stop_request_path).resolve() if stop_request_path else None
    if external_stop is not None and external_stop.parent!=out.parent:
        raise ValueError('EXTERNAL_STOP_REQUEST_MUST_SHARE_TRIAL_PARENT')
    def stop_requested():
        return (out/'stop.request').exists() or (external_stop is not None and external_stop.exists())
    storage_preflight=check_output_storage(out,'preflight')
    try:save(out/'storage_preflight.json',storage_preflight)
    except OSError as ex:
        print('STORAGE_PREFLIGHT_RECEIPT_WRITE_FAILED',storage_preflight,type(ex).__name__,str(ex),flush=True)
        raise ValueError('STORAGE_PREFLIGHT_RECEIPT_WRITE_FAILED_NO_TAKEOFF') from ex
    if storage_preflight['status']!='PASS':
        raise ValueError('STORAGE_PREFLIGHT_'+storage_preflight['status']+'_NO_TAKEOFF')
    import fixed_map_mission as mission
    from fixed_map_perception_execution import PerceptionExecutor,TrackingMarginGeometry
    from fixed_map_scene_binding import bind_scene_config
    from fixed_map_perception_process import PerceptionProcess
    from fixed_map_sensor_language import interpret
    from fixed_map_sensor_mission import SensorMission,OBSERVATION
    from fixed_map_resources import memory_headroom
    cfg=read(ROOT/config_path);cfg.update(ground_development_only=False,flight_authorized=True,collection_context='one_development_flight_'+cfg['schema'])
    interface_request=None
    if interface_request_path is not None:
        request_path=Path(interface_request_path).resolve()
        from fixed_map_perception import sha
        if request_path.parent!=out.parent or not interface_request_sha256 or sha(request_path)!=interface_request_sha256:
            raise ValueError('INTERFACE_REQUEST_PATH_OR_DIGEST_CHANGED_NO_TAKEOFF')
        interface_request=read(request_path)
        if interface_request.get('config_sha256')!=sha(ROOT/config_path) or interface_request.get('maximum_new_takeoffs')!=1 or interface_request.get('actor')!='manual_v24_desktop':
            raise ValueError('INTERFACE_REQUEST_CONFIG_OR_AUTHORIZATION_CHANGED_NO_TAKEOFF')
        if not interface_request.get('engine_identity') or not interface_request.get('scene_version'):
            raise ValueError('INTERFACE_REQUEST_SCENE_DECLARATION_MISSING_NO_TAKEOFF')
    mission.ensure_scene();e=PerceptionExecutor(copy.deepcopy(OBSERVATION));coordinator=SensorMission(e);service=None;issue=None;flight_started=False;record=None;resource_trace=None
    if cfg.get('resource_trace_v20'):
        from fixed_map_resources import ResourceTrace
        engine=read(out/'active_engine.json')
        resource_trace=ResourceTrace(out/'resource_trace.jsonl',lambda:dict(stage=e.stage,mission=coordinator.phase,latest_frame=service.latest.get('frame_id') if service else None),lambda:dict(driver=os.getpid(),sensor=service.process.pid if service is not None and hasattr(service,'process') else None,engine=engine['pid']))
        resource_trace.start()
    save(out/'declaration.json',dict(utc=utc(),instruction=text,config=cfg,observation=OBSERVATION,maximum_new_takeoffs=1,untethered_balloon_region=[42,49,-7,-1.5,2.4,5.0],flight_plane_m=3.1,scene_truth_never_read=True,scope='one development closed loop trial, not four-method evidence',owner_pid=os.getpid(),persistent_owner=True,stop_file_requests_safe_return=True,storage_policy=storage_policy()))
    try:
        if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_MODEL_LOAD_NO_TAKEOFF')
        from fixed_map_target_profiles import instruction_target
        target=instruction_target(text)
        if target:
            cfg['target_description']=target
            cfg['target_generation']+=1
        save(out/'effective_target_config.json',dict(config=cfg,source_instruction=text,
            source_config_path=str(ROOT/config_path),scope='per-task descriptor binding; no scene target coordinates'))
        record=interpret(text,e.safety,cfg);save(out/'language.json',record)
        if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
        if not record.get('validated') or record['proposal']['action'] not in ['start','preview']:raise ValueError('LANGUAGE_START_OR_PREVIEW_NOT_ACCEPTED')
        declaration=read(out/'declaration.json')
        declaration.update(config=cfg,parsed_action=record['proposal']['action'],
                           target_description=cfg['target_description'],
                           maximum_new_takeoffs=0 if record['proposal']['action']=='preview' else 1)
        save(out/'declaration.json',declaration)
        cfg_bound=bind_scene_config(cfg,read(ROOT/'configs/fixed_map_scene_state.json'))
        check_interface_binding(interface_request,cfg_bound)
        deadline=time.monotonic()+20
        while min(memory_headroom()['physical_available_bytes'],memory_headroom()['commit_available_bytes'])<3*2**30 and time.monotonic()<deadline:
            if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
            time.sleep(.25)
        if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
        service=PerceptionProcess(out/'online',cfg_bound);service.start();e.service=service;e.static_geometry=TrackingMarginGeometry(e.static_geometry.parts,.75,.15);e.g=e.static_geometry;e.start()
        # Cold model loading must not consume the first-image warmup budget.
        # These are ground-only waits; flight evidence age limits are unchanged.
        deadline=time.monotonic()+90
        while not service.model_ready.is_set() and time.monotonic()<deadline:
            if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
            if e.stage in ['FAILED','FAILED_HOVER'] or service.process.poll() is not None:raise ValueError('GROUND_INITIALIZATION_FAILED')
            time.sleep(.1)
        if not service.model_ready.is_set():raise ValueError('GROUND_MODEL_LOAD_DEADLINE')
        deadline=time.monotonic()+35
        while (e.stage=='CHECKING' or not service.warm_ready.is_set()) and time.monotonic()<deadline:
            if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
            if e.stage in ['FAILED','FAILED_HOVER'] or service.process.poll() is not None:raise ValueError('GROUND_INITIALIZATION_FAILED')
            time.sleep(.1)
        if e.stage!='READY' or not service.warm_ready.is_set():raise ValueError('GROUND_INITIALIZATION_DEADLINE')
        if record['proposal']['action']=='preview':
            # Preview samples current ground images; it never arms or takes off.
            deadline=time.monotonic()+10
            while service.process_count<8 and time.monotonic()<deadline:
                if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
                if service.process.poll() is not None:raise ValueError('GROUND_PREVIEW_SENSOR_EXITED')
                time.sleep(.1)
            coordinator.submit(record,cfg)
            coordinator.outcome='GROUND_PREVIEW_COMPLETED_NO_TAKEOFF'
            save(out/'ground_preview.json',dict(latest=service.latest,preview=e.preview_result,
                flight_authorized=False,scope='ground image preview; target may be outside view'))
            return None
        storage_launch_gate=check_output_storage(out,'preflight');save(out/'storage_launch_gate.json',storage_launch_gate)
        if storage_launch_gate['status']!='PASS':raise ValueError('STORAGE_RESERVE_'+storage_launch_gate['status']+'_GROUND_NO_TAKEOFF')
        if interface_request is not None:
            check_interface_binding(interface_request,bind_scene_config(cfg,read(ROOT/'configs/fixed_map_scene_state.json')))
        if stop_requested():raise ValueError('OPERATOR_STOP_BEFORE_TAKEOFF')
        coordinator.submit(record,cfg);flight_started=True;deadline=time.monotonic()+410;stop_sent=False;next_storage_check=0.
        while coordinator.active or e.stage!='READY':
            now=time.monotonic()
            if not stop_sent and now>=next_storage_check:
                next_storage_check=now+CHECK_PERIOD_HOST_S
                storage=check_output_storage(out,'active')
                if storage['status']!='PASS':
                    issue='STORAGE_RESERVE_LOW_SAFE_RETURN' if storage['status']=='LOW_SPACE' else 'STORAGE_CHECK_UNAVAILABLE_SAFE_RETURN'
                    # Queue recovery before any write that could fail on a full
                    # disk. The owner drains these commands before its next
                    # control cycle; return clears perception plans/mode.
                    coordinator.active=False;coordinator.phase='STORAGE_SAFE_RETURN'
                    e.command('stop','storage_reserve_guard');e.command('return','storage_reserve_guard');stop_sent=True
                    try:e.event('STORAGE_SAFE_RETURN_REQUESTED',reason=issue,storage=storage)
                    except OSError as ex:print('STORAGE_EVENT_WRITE_FAILED',type(ex).__name__,str(ex),flush=True)
                    try:save(out/'storage_safe_return.json',dict(reason=issue,storage=storage,stop_and_return_enqueued=True))
                    except OSError as ex:print('STORAGE_RECEIPT_WRITE_FAILED',type(ex).__name__,str(ex),flush=True)
                    # Only request normal perception-worker shutdown. Return
                    # uses own-state feedback/static geometry independently.
                    try:service.stop(wait=False)
                    except OSError as ex:print('STORAGE_SENSOR_STOP_REQUEST_WRITE_FAILED',type(ex).__name__,str(ex),flush=True)
            coordinator.tick();save(out/'trial_status.json',dict(utc=utc(),owner_pid=os.getpid(),stage=e.stage,state=e.state,mission=coordinator.status(),reason=e.reason,worker_alive=service.process.poll() is None,latest_frame=service.latest.get('frame_id'),latest_valid=service.latest.get('valid'),preview=e.preview_result))
            if (stop_requested() or time.monotonic()>deadline or service.process.poll() is not None) and not stop_sent:
                issue='STOP_REQUEST_OR_TRIAL_DEADLINE_OR_SENSOR_WORKER_EXIT';e.event('TRIAL_SAFE_RETURN_REQUESTED',reason=issue);coordinator.active=False;e.command('stop','persistent_trial_recovery');e.command('return','persistent_trial_recovery');stop_sent=True
            if not e.is_alive() and e.stage!='READY':raise ValueError('CONTROL_OWNER_THREAD_EXITED')
            time.sleep(.05)
        if coordinator.phase!='LANDED_RELEASED':issue=issue or 'MISSION_NOT_COMPLETED'
    except Exception as ex:
        issue=issue or str(ex)
        try:save(out/'failure.json',dict(utc=utc(),reason=issue,failure_error=str(ex),traceback=traceback.format_exc(),stage=e.stage,new_takeoff_requested=flight_started))
        except OSError as write_error:print('FAILURE_RECEIPT_WRITE_FAILED',type(write_error).__name__,str(write_error),flush=True)
    finally:
        # An exception in the orchestration loop must not abandon a live owner.
        if e.is_alive() and e.stage not in ['READY','CHECKING']:
            e.command('stop','mandatory_driver_cleanup');e.command('return','mandatory_driver_cleanup')
            while e.is_alive() and e.stage!='READY':
                try:save(out/'recovery_status.json',dict(utc=utc(),stage=e.stage,state=e.state,reason=e.reason))
                except OSError as ex:print('RECOVERY_RECEIPT_WRITE_FAILED_OWNER_RETAINED',type(ex).__name__,str(ex),flush=True)
                time.sleep(.5)
        from fixed_map_trial_cleanup import finalize_trial,wait_for_ground_release
        # The owner is a daemon thread. Keep its process alive if a flight
        # was requested but terminal ground release is not verifiable.
        # This monitor never creates another takeoff or blind landing.
        if flight_started:
            while True:
                try:wait_for_ground_release(out,e,issue);break
                except OSError as ex:
                    # Its pending-ground receipt is useful, but an inability
                    # to write it cannot authorize process exit while airborne.
                    print('GROUND_MONITOR_WRITE_FAILED_OWNER_RETAINED',type(ex).__name__,str(ex),flush=True)
                    if e.is_alive():e.command('stop','storage_ground_monitor');e.command('return','storage_ground_monitor')
                    time.sleep(.5)
        terminal=finalize_trial(out,e,coordinator,service,issue,flight_started);issue=terminal['issue']
        if resource_trace is not None:
            terminal['resource_trace']=resource_trace.stop()
            if terminal['resource_trace']['status']!='PASS':terminal['issue']=issue or 'RESOURCE_TRACE_FAILED';terminal['perception_guidance_hover_passed']=False
            save(out/'result.json',terminal);issue=terminal['issue']
        # The original final result is already durable. Cleanup is permitted
        # only after ground/sensor/owner shutdown; keep all selected raw proof.
        try:
            from fixed_map_retention import apply_postflight_raw_retention
            terminal['raw_retention']=apply_postflight_raw_retention(out,terminal,owner_stage=e.stage,owner_alive=e.is_alive())
        except Exception as ex:
            terminal['raw_retention']=dict(utc=utc(),status='FAILED',error_type=type(ex).__name__,error=str(ex),original_final_result_preserved=True)
            print('POSTFLIGHT_RAW_RETENTION_FAILED',type(ex).__name__,str(ex),flush=True)
        try:save(out/'result.json',terminal)
        except OSError as ex:print('RETENTION_RESULT_UPDATE_FAILED_ORIGINAL_RESULT_PRESERVED',type(ex).__name__,str(ex),flush=True)
    print('TRIAL_FINISHED',issue,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--instruction',default='让单机识别并接近带白色横条的红色气球，然后返航落地');p.add_argument('--config',default='configs/fixed_map_perception_yolo_cpu_v11.json');p.add_argument('--stop-request');p.add_argument('--interface-request');p.add_argument('--interface-request-sha256');p.add_argument('--execute-one-trial',action='store_true');a=p.parse_args()
    if not a.execute_one_trial:raise ValueError('EXPLICIT_ONE_TRIAL_FLAG_REQUIRED')
    run(a.output,a.instruction,a.config,a.stop_request,a.interface_request,a.interface_request_sha256)
