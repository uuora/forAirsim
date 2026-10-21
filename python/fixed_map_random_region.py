"""Ground-only 3D balloon scene generator. Private truth never enters mission."""
import argparse,random,uuid
from pathlib import Path
import numpy as np,airsim
from fixed_map_perception import ROOT,HIST,read,save,utc
from fixed_map_random_scene import ground,pose_json,from_json
from fixed_map_perception_execution import static_geometry_v2
from fixed_map_mission import Geometry
from fixed_map_scene_binding import current_engine_identity
REGION=dict(x_m=[42.,49.],y_m=[-7.,-1.5],z_m=[2.4,5.0])
NAMES=['EL1_Balloon_Target','EL1_Balloon_RedDistractor','EL1_Balloon_YellowDistractor']

def sample(seed,primitives,count=3):
    if type(count)!=int or not 1<=count<=3:raise ValueError('COUNT_OUTSIDE_1_TO_3')
    g=Geometry(static_geometry_v2(primitives)[0],1.);rng=random.Random(seed);points=[]
    for _ in range(1000):
        p=np.array([rng.uniform(*REGION[k]) for k in ['x_m','y_m','z_m']])
        if g.clearance(p)[0]>=.25 and all(np.linalg.norm(p-q)>=3.5 for q in points):points.append(p)
        if len(points)==count:return [p.tolist() for p in points]
    raise ValueError('BOUNDED_REGION_SAMPLING_FAILED')

def apply(output,seed,count=3,restore=False):
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    c=airsim.MultirotorClient(port=41454,timeout_value=3);ground(c)
    scene=read(ROOT/'configs/fixed_map_scene_runtime_v2.json')
    if scene['runtime_map']!='/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_RuntimeRandom_v2':raise ValueError('UNTETHERED_MAP_VERSION_REQUIRED')
    import psutil
    engine=current_engine_identity()
    if scene['runtime_map'].lower() not in ' '.join(psutil.Process(engine['pid']).cmdline()).lower():raise ValueError('ACTIVE_MAP_NOT_UNTETHERED_V2')
    parts=read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives'];original={n:pose_json(c.simGetObjectPose(n)) for n in NAMES}
    if not all(np.isfinite(p['position_ned_m']).all() for p in original.values()):raise ValueError('SCENE_BALLOON_UNAVAILABLE')
    points=sample(seed,parts,3) if not restore else [next(p['center_xyz_m'] for p in parts if p['id']==n) for n in NAMES]
    private=dict(utc=utc(),seed=seed,region=REGION,original=original,requested_centers_world_xyz_m=dict(zip(NAMES,points)),isolation='scenario construction and scoring only; never control',count=count,aircraft_motion_commands=0)
    save(out/'scenario_truth.json',private);changed=[]
    try:
        for i,(n,p) in enumerate(zip(NAMES,points)):
            ground(c);wanted=dict(position_ned_m=[p[0],p[1],1.-p[2]],orientation_xyzw=original[n]['orientation_xyzw'])
            # Optional fewer balloons are parked below the world, not reused as detections.
            if i>=count and not restore:wanted['position_ned_m'][2]=51.
            before=pose_json(c.simGetObjectPose(n))
            already_at_pose=(np.allclose(before['position_ned_m'],wanted['position_ned_m'],atol=1e-5,rtol=0)
                and np.allclose(before['orientation_xyzw'],wanted['orientation_xyzw'],atol=1e-6,rtol=0))
            # UE may return false for an unchanged transform. Skip only a
            # verified no-op; real moves still require the RPC success flag.
            if not already_at_pose:
                if not c.simSetObjectPose(n,from_json(wanted),teleport=True):raise ValueError('BALLOON_OBJECT_MOVE_REJECTED_'+n)
                changed.append(n)
            actual=pose_json(c.simGetObjectPose(n))
            if np.linalg.norm(np.array(actual['position_ned_m'])-wanted['position_ned_m'])>.02:raise ValueError('BALLOON_OBJECT_MOVE_NOT_APPLIED_'+n)
            private.setdefault('observed',{})[n]=actual
        save(out/'scenario_truth.json',private)
        receipt=dict(utc=utc(),status='PASS_RUNTIME_UNTETHERED_3D_REGION',scene_version='untethered_3d_'+uuid.uuid4().hex[:12],region=REGION,balloon_count=3 if restore else count,engine_identity=engine,truth_used_as_control_input=False,map_modified=False,original_baseline_available=False,restored_initial_balloon_positions=restore,layout_seed=None if restore else seed,layout_mode='restored_fixed' if restore else 'seeded_random')
        save(out/'public_receipt.json',receipt);save(ROOT/'configs/fixed_map_scene_state.json',receipt);return receipt
    except Exception as ex:
        rollback=[dict(actor=n,restored=c.simSetObjectPose(n,from_json(original[n]),teleport=True)) for n in changed]
        save(out/'failure.json',dict(reason=str(ex),rollback=rollback));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=24101008);p.add_argument('--count',type=int,default=3);p.add_argument('--restore',action='store_true');a=p.parse_args();print(__import__('json').dumps(apply(a.output,a.seed,a.count,a.restore),ensure_ascii=False))
