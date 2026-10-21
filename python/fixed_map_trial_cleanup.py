"""Fail-visible terminal persistence independent of sensor shutdown."""
import json
import time
from pathlib import Path

from fixed_map_perception import save, utc


def probe_ground():
    import airsim
    vehicles={};errors=[]
    try:client=airsim.MultirotorClient(port=41454,timeout_value=3)
    except Exception as ex:return dict(utc=utc(),passed=False,vehicles={},errors=[str(ex)])
    for name in ['DroneA','DroneB']:
        try:
            before=client.getMultirotorState(vehicle_name=name);time.sleep(.05)
            after=client.getMultirotorState(vehicle_name=name)
            vehicles[name]=dict(landed=after.landed_state==airsim.LandedState.Landed,
                                api_control=client.isApiControlEnabled(vehicle_name=name),
                                fresh=after.timestamp>before.timestamp,
                                timestamp_before=before.timestamp,timestamp_after=after.timestamp)
        except Exception as ex:
            vehicles[name]=dict(landed=None,api_control=None,fresh=False,error=str(ex));errors.append(name+': '+str(ex))
    passed=not errors and len(vehicles)==2 and all(s['landed'] is True and s['api_control'] is False and s['fresh'] for s in vehicles.values())
    return dict(utc=utc(),passed=passed,vehicles=vehicles,errors=errors)


def wait_for_ground_release(output,executor,issue,ground_probe=probe_ground,sleeper=time.sleep):
    """Retain the driver rather than exit while its requested flight is unknown."""
    receipt=ground_probe()
    while not receipt['passed']:
        save(Path(output)/'terminal_recovery_pending.json',dict(utc=utc(),stage=executor.stage,owner_alive=executor.is_alive(),issue=issue or 'GROUND_RELEASE_NOT_VERIFIABLE',ground=receipt,driver_process_retained=True))
        if executor.is_alive():
            executor.command('stop','retained_terminal_ground_guard');executor.command('return','retained_terminal_ground_guard')
        sleeper(.5);receipt=ground_probe()
    return receipt


def finalize_trial(output,executor,coordinator,service,issue,flight_started,ground_probe=probe_ground):
    output=Path(output);errors=[]
    def ground_check():
        try:
            receipt=ground_probe();vehicles=receipt.get('vehicles',{})
            receipt['passed']=bool(receipt.get('passed') and set(vehicles)=={'DroneA','DroneB'} and all(v.get('landed') is True and v.get('api_control') is False and v.get('fresh') is True for v in vehicles.values()))
            return receipt
        except Exception as ex:return dict(utc=utc(),passed=False,vehicles={},errors=[str(ex)])
    ground=ground_check();save(output/'final_ground_receipt.json',ground)
    result=dict(utc=utc(),issue=issue,mission=coordinator.status(),state=executor.state,
                final_ground=ground['vehicles'],flight_start_requested=flight_started,
                perception_guidance_hover_passed=False,terminal_phase='POSTFLIGHT_SENSOR_CLEANUP_PENDING',
                ground_verification_passed=ground['passed'],cleanup_errors=errors,
                result_record_provenance='ORIGINAL_DRIVER_OUTPUT_V18')
    # This failure-visible record exists before any worker wait or teardown.
    save(output/'result.json',result)
    if service is not None and hasattr(service,'process'):
        try:result['sensor_cleanup']=service.stop(wait=True)
        except Exception as ex:
            errors.append(dict(component='sensor_stop',error=str(ex)));result['issue']=result['issue'] or str(ex)
            result['sensor_cleanup']=getattr(service,'stop_receipt',dict(status='ERROR',error=str(ex)))
            try:result['sensor_reap']=service.reap_after_ground_release(ground_check)
            except Exception as reap_error:errors.append(dict(component='sensor_reap',error=str(reap_error)))
    if ground['passed'] and executor.is_alive():
        try:
            executor.exit_requested=True;executor.join(timeout=3)
            if executor.is_alive():errors.append(dict(component='control_owner',error='CONTROL_OWNER_EXIT_TIMEOUT'))
        except Exception as ex:errors.append(dict(component='control_owner',error=str(ex)))
    ground=ground_check();save(output/'final_ground_receipt.json',ground)
    result.update(utc=utc(),final_ground=ground['vehicles'],ground_verification_passed=ground['passed'],state=executor.state)
    if not ground['passed']:result['issue']=result['issue'] or 'FINAL_GROUND_RPC_UNVERIFIED'
    if errors:result['issue']=result['issue'] or 'TERMINAL_CLEANUP_FAILED'
    result['cleanup_errors']=errors
    event_path=output/'session_events.jsonl'
    hover=False
    try:hover=any(json.loads(line).get('kind')=='LANGUAGE_SENSOR_APPROACH_HOVER_PASSED' for line in event_path.read_text(encoding='utf-8').splitlines()) if event_path.exists() else False
    except (ValueError,OSError) as ex:
        errors.append(dict(component='event_read',error=str(ex)));result['issue']=result['issue'] or 'TERMINAL_EVENT_RECORD_UNVERIFIED'
    result['perception_guidance_hover_passed']=bool(coordinator.phase=='LANDED_RELEASED' and hover and ground['passed'] and result['issue'] is None and not errors)
    result['terminal_phase']='FINAL' if ground['passed'] else 'FINAL_UNVERIFIED_GROUND'
    save(output/'result.json',result)
    return result
