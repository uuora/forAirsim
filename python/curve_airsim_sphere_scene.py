"""Fixed runtime Sphere scene using the existing setup_scene mesh mechanism.

Geometry is local z-up metres, converted to AirSim world NED with a measured
origin offset. Contact validation is a swept simulated Pawn query, not flight.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def scene_hash(scene):
    return hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_scene(scene):
    if scene['coordinate_frame']!='local_zup_m' or not scene['obstacles']:
        raise ValueError('nonempty local-zup sphere scene required')
    names=[]
    for item in scene['obstacles']:
        p=np.asarray(item['centre'],dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(item['radius']) or item['radius']<=0:
            raise ValueError('invalid sphere geometry')
        if (not item['name'].startswith('CurveSphere_') and not scene.get('preplaced_inspected_actor')) or not item['name'].replace('_','').isalnum():
            raise ValueError('owned sphere name required')
        names.append(item['name'])
    if len(set(names))!=len(names):raise ValueError('duplicate sphere names')
    for name in ('vehicle_envelope_radius_m','safety_margin_m'):
        if not np.isfinite(scene[name]) or scene[name]<0:raise ValueError('invalid clearance envelope')
    return scene


def swept_clearance(positions,scene):
    """Exact minimum distance of measured-state chords, not continuous flight."""
    p=np.asarray(positions,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)==0 or not np.isfinite(p).all():raise ValueError('finite path required')
    a=p[:-1] if len(p)>1 else p
    d=p[1:]-a if len(p)>1 else np.zeros_like(a)
    denom=np.sum(d*d,axis=1)
    minimum=float('inf')
    for obstacle in scene['obstacles']:
        c=np.asarray(obstacle['centre'])
        u=np.clip(np.sum((c-a)*d,axis=1)/np.maximum(denom,1e-300),0,1)
        distance=np.linalg.norm(a+u[:,None]*d-c,axis=1)-obstacle['radius']-scene['vehicle_envelope_radius_m']
        minimum=min(minimum,float(distance.min()))
    return dict(measured_chord_min_envelope_clearance_m=minimum,
        clearance_margin_passed=minimum>=scene['safety_margin_m'],
        clearance_scope='Exact measured-state straight-chord distance minus configured sphere envelope; not continuous-time vehicle-body proof')


def setup_scene(client,config,airsim,output):
    from setup_scene import ensure_mesh, vector
    scene=validate_scene(config['sphere_scene'])
    vehicle=config['vehicle']
    local=np.asarray(vector(client.getMultirotorState(vehicle_name=vehicle).kinematics_estimated.position))
    world=np.asarray(vector(client.simGetObjectPose(vehicle).position))
    offset=world-local
    all_assets=client.simListAssets()
    output=Path(output);output.mkdir(exist_ok=True)
    (output/'runtime_assets.json').write_text(json.dumps(all_assets,indent=2),encoding='utf-8')
    assets=[x for x in all_assets if x.casefold()==scene['mesh_asset'].casefold()]
    if not scene.get('preplaced_inspected_actor') and len(assets)!=1:raise RuntimeError('explicit inspected sphere asset missing or ambiguous')
    asset_receipt=json.loads((Path(__file__).resolve().parents[1]/scene['asset_receipt']).read_text(encoding='utf-8'))
    if asset_receipt['status']!='PASS' or len(asset_receipt['collision_radii_cm'])!=1 or not np.allclose(asset_receipt['collision_radii_cm'],[50.],atol=.01):
        raise RuntimeError('explicit sphere collision geometry not certified by editor inspection')
    client.simFlushPersistentMarkers()
    operations=[]
    for item in scene['obstacles']:
        if scene.get('preplaced_inspected_actor'):
            if item['name']!=asset_receipt['actor_name'] or item['name'] not in client.simListSceneObjects():
                raise RuntimeError('inspected preplaced actor absent; refusing fallback spawn')
        centre=np.asarray(item['centre'])*[1,1,-1]+offset
        diameter=2*item['radius'];scale=[diameter]*3
        observed=ensure_mesh(client,item['name'],scene['mesh_asset'],centre.tolist(),scale)
        if not np.allclose(vector(client.simGetObjectScale(item['name'])),scale,atol=.001):
            raise RuntimeError('sphere scale mismatch')
        # Independent blocking query around this isolated ball, not a visual test.
        voxel=output/(item['name']+'.binvox')
        side=int(np.ceil(max(2.,diameter+1.)))
        if not client.simCreateVoxelGrid(airsim.Vector3r(*centre),side,side,side,.05,str(voxel)):
            raise RuntimeError('sphere blocking voxel query failed')
        raw=voxel.read_bytes().split(b'data\n',1)[1]
        occupied=sum(count for value,count in zip(raw[::2],raw[1::2]) if value)
        if occupied<=0:raise RuntimeError('sphere region has no blocking voxels')
        maximum=4*np.pi/3*(item['radius']+np.sqrt(3)*.05)**3/.05**3
        if occupied>maximum:raise RuntimeError('blocking volume exceeds configured sphere plus voxel tolerance')
        # Sweep the actual simulated Pawn from free space into this sphere.
        # Restore the ground pose afterwards; no arm or velocity command here.
        original=client.simGetVehiclePose(vehicle_name=vehicle)
        before=client.simGetCollisionInfo(vehicle_name=vehicle)
        contacts=[];attempts=[]
        try:
            for i,x in enumerate(np.linspace(-item['radius']-1.2,0.,45)):
                position=centre-offset+np.array([x,0,0])
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*position),airsim.Quaternionr()),i==0,vehicle_name=vehicle)
                # Multirotor setPose is RenderPending until a simulation tick;
                # querying a paused world immediately would test no sweep.
                client.simContinueForTime(.01)
                contact=client.simGetCollisionInfo(vehicle_name=vehicle)
                attempts.append(dict(index=i,requested_local_ned_m=position.tolist(),has_collided=bool(contact.has_collided),
                    object_name=contact.object_name,time_stamp=int(contact.time_stamp)))
                if contact.has_collided and contact.object_name==item['name'] and (contact.time_stamp!=before.time_stamp or contact.object_name!=before.object_name):
                    contacts.append(dict(object_name=contact.object_name,time_stamp=int(contact.time_stamp),
                        penetration_depth=float(contact.penetration_depth),requested_local_ned_m=position.tolist()))
                    break
        finally:
            client.simSetVehiclePose(original,True,vehicle_name=vehicle)
            client.simContinueForTime(.01)
            (output/(item['name']+'_contact_probe.json')).write_text(json.dumps(attempts,indent=2),encoding='utf-8')
        if not contacts:raise RuntimeError('sphere-specific swept Pawn contact not observed: '+item['name'])
        operations.append(dict(name=item['name'],position_world_ned_m=observed,scale=scale,
            occupied_region_voxels=occupied,contact=contacts[0],contact_test='simulation-stepped kinematic Pawn sweep; not powered flight'))
    receipt=dict(status='PASS',scene=scene,scene_sha256=scene_hash(scene),world_origin_offset_ned_m=offset.tolist(),
                 objects=operations,mesh_asset=scene['mesh_asset'],movement_commands_sent=0)
    (output/'scene_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def show_reference(client,curve,plan,config,airsim,offset=None):
    if not config.get('visualization',{}).get('enabled'):return
    client.simFlushPersistentMarkers()
    offset=np.zeros(3) if offset is None else np.asarray(offset)
    points=[airsim.Vector3r(*(p*[1,1,-1]+offset)) for p in curve.sample(np.linspace(0,curve.duration_s,201))]
    client.simPlotLineStrip(points,[0.,.7,1.,1.],3.,is_persistent=True)
    client.simPlotPoints([points[0],points[-1]],[1.,.8,0.,1.],18.,is_persistent=True)
    client.simPlotStrings(['START','GOAL'],[points[0],points[-1]],1.,[1.,.8,0.,1.],60.)
    client.simPrintLogMessage(plan['group_id'],'reference CYAN / actual ORANGE',0)


def show_step(client,previous,current,config,airsim,offset=None):
    if not config.get('visualization',{}).get('enabled'):return
    offset=np.zeros(3) if offset is None else np.asarray(offset)
    convert=lambda p:airsim.Vector3r(*(np.asarray(p)*[1,1,-1]+offset))
    client.simPlotLineStrip([convert(previous['position_zup_m']),convert(current['position_zup_m'])],
                            [1.,.3,.05,1.],4.,is_persistent=True)
    client.simPlotPoints([convert(current['reference_position_zup_m'])],[0.,1.,1.,1.],10.,duration=.1)


def capture_overview(client,config,airsim,path):
    """Actual simulator camera rendering; temporarily set a world overview."""
    from setup_scene import rotate
    name=config['vehicle'];camera='front_center'
    vehicle=client.simGetVehiclePose(vehicle_name=name)
    original_world=client.simGetCameraInfo(camera,vehicle_name=name).pose
    inv=vehicle.orientation.inverse()
    original=airsim.Pose(rotate(original_world.position-vehicle.position,inv),inv*original_world.orientation)
    eye=airsim.Vector3r(-3.,-10.,-11.)
    import math
    direction=airsim.Vector3r(4.,0.,-5.)-eye
    orientation=airsim.to_quaternion(math.atan2(-direction.z_val,math.hypot(direction.x_val,direction.y_val)),0,math.atan2(direction.y_val,direction.x_val))
    pose=airsim.Pose(rotate(eye-vehicle.position,inv),inv*orientation)
    try:
        client.simSetCameraPose(camera,pose,vehicle_name=name)
        client.simGetImage(camera,airsim.ImageType.Scene,vehicle_name=name)
        data=client.simGetImage(camera,airsim.ImageType.Scene,vehicle_name=name)
        if not data:raise RuntimeError('empty rendered overview')
        Path(path).write_bytes(data)
    finally:
        client.simSetCameraPose(camera,original,vehicle_name=name)
