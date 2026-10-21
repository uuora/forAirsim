"""Ground-only startup readiness; no arm, control acquisition or motion requests."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import airsim
import numpy as np


def vec(value):return np.array([value.x_val,value.y_val,value.z_val])


def vehicle_sample(client,name):
    state=client.getMultirotorState(vehicle_name=name)
    # Object pose for the two vehicle actors is world NED; VehiclePose is local
    # to that vehicle's startup frame and cannot be compared to world ground Z.
    pose=client.simGetObjectPose(name)
    position=vec(pose.position);speed=float(np.linalg.norm(vec(state.kinematics_estimated.linear_velocity)))
    acc=float(np.linalg.norm(vec(state.kinematics_estimated.linear_acceleration)))
    controlled=client.isApiControlEnabled(vehicle_name=name);ground=client.simGetCollisionInfo(vehicle_name=name).object_name
    grounded=bool(np.isfinite(position).all() and not controlled and state.landed_state==airsim.LandedState.Landed
        and .65<position[2]<1.1 and speed<=.01 and acc<=.05 and ground.startswith('Ground'))
    return dict(world_ned_m=position.tolist(),speed_m_s=speed,acceleration_m_s2=acc,api_control_enabled=controlled,
        landed_flag=state.landed_state==airsim.LandedState.Landed,ground_actor=ground,ground_readiness=grounded,state_timestamp_ns=int(state.timestamp))


def probe(output,failure=None,pause_after=False):
    if output.exists():raise ValueError('Readiness receipt must be new')
    client=airsim.MultirotorClient(timeout_value=5);client.ping();rows=[];stable=None;started=time.monotonic()
    receipt=dict(kind='GROUND_ONLY_STARTUP_READINESS',status='RUNNING',flight_motion_requests=False,control_acquisition=False,arm_requests=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),position_source='world_object_pose_of_vehicle_actors_only',samples=rows)
    try:
        if failure:
            raw=json.loads(failure.read_text(encoding='utf-8'))
            if raw.get('flight_dispatch_started') is not False or raw.get('failed_phase')!='PREFLIGHT' or raw.get('manual_recovery_required') is not False:
                raise ValueError('Not a verified pre-arm preflight failure')
            if any(client.isApiControlEnabled(vehicle_name=name) for name in ('DroneA','DroneB')):
                raise ValueError('Preserve externally controlled vehicle')
            receipt.update(preflight_failure_report=str(failure.resolve()),preflight_failure_sha256=hashlib.sha256(failure.read_bytes()).hexdigest(),unpause_for_ground_settling=True)
            client.simPause(False)
        elif client.simIsPause():raise ValueError('Preserve already paused simulator')
        while time.monotonic()-started<15:
            row=dict(host_monotonic_s=time.monotonic(),vehicles={});ready=True
            for name in ('DroneA','DroneB'):
                row['vehicles'][name]=vehicle_sample(client,name)
                ready=ready and row['vehicles'][name]['ground_readiness']
            rows.append(row)
            stable=(row['host_monotonic_s'] if stable is None else stable) if ready else None
            if stable is not None and row['host_monotonic_s']-stable>=.5:
                receipt.update(status='READY',stable_ground_host_span_s=row['host_monotonic_s']-stable);break
            time.sleep(.1)
        else:receipt.update(status='NOT_READY',reason='Ground readiness timeout')
    except Exception as error:receipt.update(status='NOT_READY',reason=str(error))
    finally:
        if failure or pause_after:client.simPause(True);receipt['paused_after_probe']=True
        output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='samples'},indent=2));return receipt['status']=='READY'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-failure-report',type=Path);parser.add_argument('--pause-after',action='store_true');args=parser.parse_args()
    raise SystemExit(0 if probe(args.output,args.preflight_failure_report,args.pause_after) else 2)
