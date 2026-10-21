"""Ground-contact recovery for a caller-verified owned pilot process only.

Never takes off, moves or unpauses. Refuses unless both vehicle actors are
stationary on the known ground; disarms/releases only the pilot DroneA.
"""
import argparse
from pathlib import Path
import time
import json
import airsim
import numpy as np
from setup_scene import vector
from t1_ground_perception_probe import ground_states
from braking_startup_readiness import vehicle_sample


def recover(output):
    if output.exists():raise ValueError('NEW_RECOVERY_RECEIPT_REQUIRED')
    c=airsim.MultirotorClient(timeout_value=5);c.ping();rows=[];stable=None
    receipt=dict(kind='OWNED_T1_GROUND_CONTACT_RECOVERY',status='FAIL',takeoff_requests=False,
        motion_requests=False,unpause_requests=False,samples=rows)
    try:
        end=time.monotonic()+8
        while time.monotonic()<end:
            row={'host_monotonic_s':time.monotonic(),'vehicles':{}};ok=True
            for name in ('DroneA','DroneB'):
                s=c.getMultirotorState(vehicle_name=name);p=vector(c.simGetObjectPose(name).position)
                speed=float(np.linalg.norm(vector(s.kinematics_estimated.linear_velocity)))
                collision=c.simGetCollisionInfo(vehicle_name=name)
                controlled=bool(c.isApiControlEnabled(vehicle_name=name))
                supported=.65<p[2]<1.1 and speed<=.02 and collision.object_name.startswith('Ground')
                if name=='DroneB':supported=supported and not controlled
                row['vehicles'][name]=dict(world_ned_m=p,speed_m_s=speed,ground_actor=collision.object_name,
                                          api_control_enabled=controlled,ground_contact_supported=supported)
                ok=ok and supported
            rows.append(row);stable=(time.monotonic() if stable is None else stable) if ok else None
            if stable is not None and time.monotonic()-stable>=1.:break
            time.sleep(.1)
        else:raise RuntimeError('GROUND_CONTACT_NOT_CONFIRMED')
        if c.isApiControlEnabled(vehicle_name='DroneA'):
            c.cancelLastTask(vehicle_name='DroneA')
            if not c.armDisarm(False,vehicle_name='DroneA'):raise RuntimeError('DISARM_REJECTED')
            c.enableApiControl(False,vehicle_name='DroneA')
        end=time.monotonic()+5
        while True:
            try:
                receipt['ground_after']=ground_states(c);break
            except ValueError:
                if time.monotonic()>=end:raise
                time.sleep(.1)
        c.simPause(True)
        receipt['fresh_ground_samples']=[]
        for _ in range(6):
            row={n:vehicle_sample(c,n) for n in ('DroneA','DroneB')}
            receipt['fresh_ground_samples'].append(row)
            if not c.simIsPause() or not all(v['ground_readiness'] for v in row.values()):
                raise RuntimeError('FRESH_GROUND_PROOF_FAILED')
            time.sleep(.1)
        receipt.update(status='PASS',grounded_and_released=True,paused=True)
    except Exception as error:receipt['error_type']=type(error).__name__
    output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('samples','ground_after','fresh_ground_samples')}))
    return receipt['status']=='PASS'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();raise SystemExit(0 if recover(args.output) else 2)
