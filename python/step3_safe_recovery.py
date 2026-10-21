"""Recover an already airborne, stopped development flight. Never take off.

Uses the recorded HOME and same static planner/controller; no target input.
"""
import sys,time,json
from pathlib import Path
import numpy as np
import airsim
import fixed_map_mission as mission
from fixed_map_runtime import configure_session
from fixed_map_perception_execution import PerceptionExecutor

def main():
    output=Path(sys.argv[1]).resolve();runtime=configure_session(output/'observation_runtime_v2')
    declaration=mission.read(output/'observation_declaration_v2.json')
    original=Path(mission.read(output/'observation_result_v2.json')['run_dir'])
    plan=mission.read(original/'plan.json')
    e=PerceptionExecutor(declaration);e.api=airsim;e.c=airsim.MultirotorClient(port=41454,timeout_value=2)
    e.calibrate();e.sample();e.home=np.array(plan['home_xyz_m']);e.ground_b=e.state['DroneB_position_xyz_m']
    e.run_dir=runtime/('safety_recovery_'+time.strftime('%Y%m%d_%H%M%S'));e.run_dir.mkdir();e.log=(e.run_dir/'timestamped.jsonl').open('w',encoding='utf-8')
    mission.save(e.run_dir/'source_binding.json',dict(original_flight=str(original),original_failure='STATE_READ_GAP_EXCEEDED',home_source=str(original/'plan.json'),home_xyz_m=e.home.tolist(),goal_source='recorded_takeoff_HOME_only',no_new_takeoff=True,perception_guidance_success=False))
    if e.state['landed']:
        e.c.armDisarm(False,vehicle_name='DroneA');e.c.enableApiControl(False,vehicle_name='DroneA');return
    e.hold('MANDATORY_RECOVERY_AFTER_ORIGINAL_OWNER_EXIT','STOP_HOVER');e.sample()
    e.collision_base=int(e.c.simGetCollisionInfo(vehicle_name='DroneA').time_stamp)
    # The route planner requires a common height within .03m. Reach the
    # ORIGINAL 3.1m plane through real dynamics, rather than relax this gate.
    plane=e.p.copy();plane[2]=e.task['height_world_m'];last=None;vcmd=np.zeros(3);until=time.monotonic()+25
    if not e.g.segment(e.p,plane,e.task['path_clearance_m'])[0]:raise ValueError('RECOVERY_HEIGHT_COLUMN_UNSAFE')
    e.event('RECOVERY_HEIGHT_ALIGNMENT',actual_xyz_m=e.p.tolist(),goal_xyz_m=plane.tolist(),reason='original constant-height route gate retained')
    stable=None
    while True:
        tick=time.monotonic();e.sample()
        if last is not None and tick-last>e.task['state_gap_stop_host_s']:raise ValueError('RECOVERY_ALIGNMENT_STATE_GAP')
        if time.monotonic()>until:raise ValueError('RECOVERY_HEIGHT_ALIGNMENT_TIMEOUT')
        last=tick;delta=plane-e.p;desired=.7*delta-.25*e.v
        norm=np.linalg.norm(desired)
        if norm>.25:desired*=.25/norm
        dv=desired-vcmd;norm=np.linalg.norm(dv)
        if norm>.04:dv*=.04/norm
        vcmd+=dv
        if not e.g.segment(e.p,e.p+vcmd*.7,e.task['clearance_stop_m'])[0]:raise ValueError('RECOVERY_ALIGNMENT_PREDICTED_UNSAFE')
        e.c.moveByVelocityAsync(float(vcmd[0]),float(vcmd[1]),float(-vcmd[2]),e.task['command_duration_sim_s'],drivetrain=e.api.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=e.api.YawMode(False,0),vehicle_name='DroneA')
        e.event('velocity_command_requested',source_state_timestamp_ns=e.state['state_timestamp_ns'],world_xyz_m_s=vcmd.tolist(),duration_sim_s=e.task['command_duration_sim_s'],recovery_phase='RECOVERY_HEIGHT_ALIGNMENT')
        good=abs(e.p[2]-plane[2])<=.02 and e.state['speed_m_s']<=e.task['arrival_speed_m_s']
        if good:
            if stable is None:stable=int(e.raw_state.timestamp)
            if (int(e.raw_state.timestamp)-stable)/1e9>=e.task['arrival_dwell_s']:break
        else:stable=None
        time.sleep(max(0,.1-(time.monotonic()-tick)))
    e.hold('RECOVERY_HEIGHT_ALIGNED','STOP_HOVER');e.sample()
    e.process_request(dict(action='return',actor='user_authorized_mandatory_safety_recovery'))
    last=None;deadline=time.monotonic()+200;issue=None
    while e.stage!='READY':
        tick=time.monotonic()
        try:
            e.sample()
            if last is not None and tick-last>e.task['state_gap_stop_host_s']:raise ValueError('RECOVERY_STATE_READ_GAP_EXCEEDED')
            if time.monotonic()>deadline:raise ValueError('RECOVERY_RETURN_TIMEOUT')
            last=tick;e.control()
        except Exception as ex:
            issue=str(ex);e.event('SAFETY_RECOVERY_FAILED',reason=issue);e.hold(issue,'FAILED_HOVER')
            print('SAFETY_RECOVERY_REQUIRES_ATTENTION:'+issue,flush=True)
            while not e.state['landed']:
                try:e.sample()
                except Exception:pass
                time.sleep(.1)
            e.c.armDisarm(False,vehicle_name='DroneA');e.c.enableApiControl(False,vehicle_name='DroneA');e.stage='READY'
        time.sleep(max(0,.1-(time.monotonic()-tick)))
    e.sample()
    receipt=dict(utc=mission.utc(),issue=issue,final_stage=e.stage,final_state=e.state,api_control_released=not e.c.isApiControlEnabled(vehicle_name='DroneA'),original_flight=str(original),recovery_log=str(e.run_dir/'timestamped.jsonl'),no_new_takeoff=True)
    mission.save(output/'safety_recovery_result_v2.json',receipt);print(json.dumps(receipt,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
