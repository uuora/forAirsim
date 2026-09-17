"""Read-only landed stability sampling. Run twice across a real process restart."""
import argparse
import json
import time
from pathlib import Path
import airsim
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--baseline')
a=p.parse_args()
c=airsim.MultirotorClient(port=41452,timeout_value=15)
names=['DroneA','DroneB']
rows={n:[] for n in names}
pad=c.simGetObjectPose('MH2_ObservationPad').position
def vec(v): return [v.x_val,v.y_val,v.z_val]
for i in range(31):
    for n in names:
        pose=c.simGetVehiclePose(vehicle_name=n)
        hit=c.simGetCollisionInfo(vehicle_name=n)
        state=c.getMultirotorState(vehicle_name=n)
        rows[n].append(dict(position=vec(pose.position),landed=int(state.landed_state),collision=hit.has_collided,object=hit.object_name))
    if i<30: time.sleep(1)
report={'flight_commands_issued':False,'duration_seconds':30,'samples':rows,'checks':{},'status':'PASS'}
for n,rs in rows.items():
    xyz=np.array([r['position'] for r in rs]); h=pad.z_val-.2-xyz[:,2]
    # The flag is reset by AirSim's physics update; object_name retains the last hit.
    # The H marking is part of the platform landing surface in this generated map.
    support_names={'MH2_ObservationPad','MH2_HBar_0','MH2_HBar_1'}
    checks=dict(finite=bool(np.isfinite(xyz).all()),drift_m=float(np.linalg.norm(xyz-xyz[0],axis=1).max()),min_origin_height_above_deck_m=float(h.min()),max_origin_height_above_deck_m=float(h.max()),within_pad=bool((np.abs(xyz[:,0]-pad.x_val)<5.5).all() and (np.abs(xyz[:,1]-pad.y_val)<5.5).all()),landed=all(r['landed']==0 for r in rs),platform_contact=any(r['object'] in support_names for r in rs),contact_evidence='last_collision_object_not_continuous_contact_sensor')
    passed=checks['finite'] and checks['drift_m']<.05 and 0<=h.min()<=.6 and h.max()<=.6 and checks['within_pad'] and checks['landed'] and checks['platform_contact']
    checks['pass']=bool(passed)
    report['checks'][n]=checks
    if not passed: report['status']='FAIL'
if a.baseline:
    old=json.loads(Path(a.baseline).read_text())
    report['restart_position_delta_m']={n:float(np.linalg.norm(np.array(rows[n][-1]['position'])-np.array(old['samples'][n][-1]['position']))) for n in names}
    if old['status']!='PASS' or any(v>.05 for v in report['restart_position_delta_m'].values()): report['status']='FAIL'
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
raise SystemExit(0 if report['status']=='PASS' else 1)
