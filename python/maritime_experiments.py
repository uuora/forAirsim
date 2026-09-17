"""Explicit scene profiles for stationary capture; resets to baseline on completion."""
import hashlib
import json
import math
import re
import time
from pathlib import Path
import airsim

ROOT=Path(__file__).resolve().parents[1]
def unique(client, pattern):
    names=[n for n in client.simListSceneObjects('MH2_.*') if re.fullmatch(pattern,n)]
    if len(names)!=1: raise RuntimeError('Expected unique actor: '+pattern)
    return names[0]

def apply(client, name):
    raw=(ROOT/'configs/maritime_experiments.json').read_bytes()
    config=json.loads(raw); p=config['profiles'][name]
    applied=[]
    for pattern in ('MH2_OceanGrid(_[0-9]+)?','MH2_DistantOcean(_[0-9]+)?'):
        actor=unique(client,pattern)
        material='/Game/MaritimeHarborV2/Materials/'+p['wave_material']+'.'+p['wave_material']
        if not client.simSetObjectMaterial(actor,material): raise RuntimeError('Material assignment failed: '+actor)
        applied.append({'actor':actor,'material':material,'api_acknowledged':True})
    sun=unique(client,'MH2_Sun(_[0-9]+)?')
    sunpose=client.simGetObjectPose(sun)
    sunpose.orientation=airsim.to_quaternion(math.radians(p['sun_pitch_deg']),0,math.radians(p['sun_yaw_deg']))
    client.simSetObjectPose(sun,sunpose,True)
    hull=client.simGetObjectPose(unique(client,'MH2_Vessel_Hull(_[0-9]+)?')).position
    # Coordinates relative to the original hull origin. Move balloon AND string.
    locations=[(4,9.5,-1.5),(11,-10,-1.7),(-11,6.5,-1.45)]
    if p['occlusion']: locations[1]=(5,0,-1.7)
    positions={}
    for i,(x,y,z) in enumerate(locations):
        for prefix,z_offset in [('Balloon',z),('String',z/2)]:
            actor=unique(client,'MH2_'+prefix+'_'+str(i)+'(_[0-9]+)?')
            pose=client.simGetObjectPose(actor)
            pose.position=airsim.Vector3r(hull.x_val+x,hull.y_val+y,hull.z_val+z_offset)
            # UE can return false when an identical transform needs no movement.
            # Readback, rather than the movement flag, verifies application.
            client.simSetObjectPose(actor,pose,True)
            measured=client.simGetObjectPose(actor).position
            error=(measured-pose.position).get_length()
            if error>.001: raise RuntimeError('Scene position readback mismatch')
            positions[actor]=[measured.x_val,measured.y_val,measured.z_val]
    actual=client.simGetObjectPose(sun).orientation
    dot=abs(actual.w_val*sunpose.orientation.w_val+actual.x_val*sunpose.orientation.x_val+actual.y_val*sunpose.orientation.y_val+actual.z_val*sunpose.orientation.z_val)
    if abs(dot-1)>1e-4: raise RuntimeError('Sun rotation readback mismatch')
    time.sleep(1)
    return {'name':name,'configuration':p,'configuration_sha256':hashlib.sha256(raw).hexdigest(),'seed':config['seed'],'seed_scope':config['seed_scope'],'applied_materials':applied,'target_positions_global_ned_m':positions,'sun_orientation_wxyz':[actual.w_val,actual.x_val,actual.y_val,actual.z_val],'repeatability':'parameter_repeatable_not_pixel_deterministic; shader time and sequential captures vary','restore_policy':'baseline after run'}
