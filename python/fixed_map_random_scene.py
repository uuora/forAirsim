"""Ground-only scenario constructor. Truth stays here and NEVER feeds perception.

Runtime scene-object pose edits are reversible. No vehicle movement/reset API.
"""
import argparse,random,sys,time,uuid
from pathlib import Path
import numpy as np,airsim
from fixed_map_perception import ROOT,HIST,read,save,sha,utc
from fixed_map_perception_execution import static_geometry_v2
from fixed_map_mission import Geometry

REGION=dict(x_m=[42.,49.],y_m=[-7.,-1.5],center_height_m=3.1)
NAMES=['EL1_Balloon_Target','EL1_BalloonTether_0','EL1_BalloonAnchor_0']
def pose_json(p):return dict(position_ned_m=[p.position.x_val,p.position.y_val,p.position.z_val],orientation_xyzw=[p.orientation.x_val,p.orientation.y_val,p.orientation.z_val,p.orientation.w_val])
def from_json(p):return airsim.Pose(airsim.Vector3r(*p['position_ned_m']),airsim.Quaternionr(*p['orientation_xyzw']))
def ground(c):
    for n in ['DroneA','DroneB']:
        if c.getMultirotorState(vehicle_name=n).landed_state!=airsim.LandedState.Landed or c.isApiControlEnabled(vehicle_name=n):raise ValueError('SCENE_CHANGE_REQUIRES_LANDED_RELEASED_BOTH_DRONES')
def sample(seed,parts):
    static,_=static_geometry_v2(parts);g=Geometry(static,1.)
    distractors=[np.array(p['center_xyz_m']) for p in parts if p.get('display_label') in ['EL1_Balloon_RedDistractor','EL1_Balloon_YellowDistractor']]
    rng=random.Random(seed)
    for i in range(1000):
        p=np.array([rng.uniform(*REGION['x_m']),rng.uniform(*REGION['y_m']),REGION['center_height_m']])
        if g.clearance(p)[0]>=.25 and all(np.linalg.norm(p-d)>=3. for d in distractors):return p,i+1
    raise ValueError('BOUNDED_RANDOM_REGION_HAS_NO_VALID_SAMPLE')
def apply(output,seed=None,restore=False):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('SCENE_EVIDENCE_DIRECTORY_ALREADY_USED')
    output.mkdir(parents=True);c=airsim.MultirotorClient(port=41454,timeout_value=3);ground(c)
    parts=read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives'];by={p['id']:p for p in parts}
    version='runtime_region40_'+uuid.uuid4().hex[:12]
    originals={}
    for n in NAMES:
        pp=c.simGetObjectPose(n);ss=c.simGetObjectScale(n)
        originals[n]=dict(pose=pose_json(pp),scale=[ss.x_val,ss.y_val,ss.z_val])
        if not np.isfinite(originals[n]['pose']['position_ned_m']).all():raise ValueError('SCENE_ACTOR_UNAVAILABLE_'+n)
    truth=dict(utc=utc(),version=version,seed=seed,region=REGION,original_runtime=originals,isolated_scoring_only=True,map_file_unchanged=True,aircraft_never_teleported=True)
    save(output/'scenario_truth.json',truth)
    changes={}
    if restore:
        for n in NAMES:
            p=by[n];xyz=p['center_xyz_m'];changes[n]=dict(pose=dict(position_ned_m=[xyz[0],xyz[1],1.-xyz[2]],orientation_xyzw=originals[n]['pose']['orientation_xyzw']),scale=p.get('local_size_xyz_m',originals[n]['scale']))
    else:
        center,draws=sample(seed,parts);x,y,z=center
        platform=by['EL1_TargetPlatform'];half=np.array(platform['local_size_xyz_m'])/2;support=.6 if abs(x-platform['center_xyz_m'][0])<half[0]-.15 and abs(y-platform['center_xyz_m'][1])<half[1]-.15 else 0.
        top=z-by[NAMES[0]]['radius_m'];bottom=support+.08
        changes[NAMES[0]]=dict(pose=dict(position_ned_m=[float(x),float(y),1.-float(z)],orientation_xyzw=originals[NAMES[0]]['pose']['orientation_xyzw']),scale=originals[NAMES[0]]['scale'])
        changes[NAMES[1]]=dict(pose=dict(position_ned_m=[float(x),float(y),1.-(top+bottom)/2],orientation_xyzw=originals[NAMES[1]]['pose']['orientation_xyzw']),scale=[.025,.025,top-bottom])
        changes[NAMES[2]]=dict(pose=dict(position_ned_m=[float(x),float(y),1.-(support+.06)],orientation_xyzw=originals[NAMES[2]]['pose']['orientation_xyzw']),scale=[.22,.22,.12])
        truth.update(designated_center_world_xyz_m=center.tolist(),sample_draws=draws,support_height_m=support)
    truth['requested']=changes;save(output/'scenario_truth.json',truth)
    observed={};changed=[]
    try:
        for n,row in changes.items():
            ground(c)
            if not c.simSetObjectPose(n,from_json(row['pose']),teleport=True):raise ValueError('RUNTIME_SCENE_POSE_REJECTED_'+n)
            changed.append(n)
            if not c.simSetObjectScale(n,airsim.Vector3r(*row['scale'])):raise ValueError('RUNTIME_SCENE_SCALE_REJECTED_'+n)
            observed[n]=pose_json(c.simGetObjectPose(n))
            if np.linalg.norm(np.array(observed[n]['position_ned_m'])-row['pose']['position_ned_m'])>.02:raise ValueError('RUNTIME_SCENE_POSE_NOT_APPLIED_'+n)
        truth['observed']=observed;save(output/'scenario_truth.json',truth)
        from fixed_map_scene_binding import current_engine_identity
        receipt=dict(utc=utc(),status='PASS_RUNTIME_SCENE_RESTORED' if restore else 'PASS_RUNTIME_RANDOM_SCENE',scene_version=version,region=REGION,truth_used_as_control_input=False,map_modified=False,original_baseline_available=restore,engine_identity=current_engine_identity())
        save(output/'public_receipt.json',receipt);save(ROOT/'configs/fixed_map_scene_state.json',receipt)
        print(__import__('json').dumps(receipt),flush=True);return receipt
    except Exception as ex:
        rollback=[]
        for n in changed:
            rollback.append(dict(actor=n,pose_restored=c.simSetObjectPose(n,from_json(originals[n]['pose']),teleport=True),scale_restored=c.simSetObjectScale(n,airsim.Vector3r(*originals[n]['scale']))))
        save(output/'failure.json',dict(utc=utc(),reason=str(ex),rollback=rollback));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=24101001);p.add_argument('--restore',action='store_true');a=p.parse_args();apply(a.output,a.seed,a.restore)
