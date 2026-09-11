"""Bounded altitude response diagnosis from a paused airborne scene."""
import json
import time
from datetime import datetime
import airsim
from setup_scene import ROOT, vector


def main():
    c=airsim.MultirotorClient(timeout_value=5)
    if not c.simIsPause() or not c.isApiControlEnabled(vehicle_name='DroneA'):
        raise RuntimeError('Requires paused controlled airborne scene')
    initial=c.simGetVehiclePose(vehicle_name='DroneA').position.z_val
    folder=ROOT/'logs'/datetime.now().strftime('altitude_probe_%Y%m%d_%H%M%S')
    folder.mkdir()
    rows=[]
    try:
        c.simPause(False)
        for phase,z in [('hold',initial),('up',initial-0.3),('restore',initial)]:
            for _ in range(20):
                p=c.simGetVehiclePose(vehicle_name='DroneA').position
                if abs(p.z_val-initial)>0.6:
                    raise RuntimeError('Altitude bound exceeded')
                c.moveByVelocityZAsync(0,0,z,0.4,vehicle_name='DroneA')
                rows.append({'phase':phase,'requested_local_z':z,'local_z':p.z_val,
                             'world':vector(c.simGetObjectPose('DroneA').position)})
                time.sleep(0.2)
    finally:
        c.simPause(True)
        (folder/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(folder)
        for phase in ['hold','up','restore']:
            seq=[r['local_z'] for r in rows if r['phase']==phase]
            if seq: print(phase,seq[0],seq[-1])


if __name__=='__main__': main()
