"""Inspect saved harbor and capture the fixed overview; no flight or target logic."""
import json
from datetime import datetime
from pathlib import Path
import time
import re
import airsim
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
client=airsim.MultirotorClient(port=41452,timeout_value=12)
output=ROOT/'logs'/datetime.now().strftime('harbor_%Y%m%d_%H%M%S')
output.mkdir(parents=True,exist_ok=True)
report={'status':'RUNNING','ocean':'visual_shader_waves_only','flight_commands_issued':False}
try:
    report['vehicles']=client.listVehicles()
    resolved={}
    scene=client.simListSceneObjects('MH2_.*')
    for name in ('MH2_Vessel_Hull','MH2_OceanGrid','MH2_ObservationPad','MH2_Balloon_0','MH2_Balloon_1','MH2_Balloon_2'):
        matches=[n for n in scene if re.fullmatch(re.escape(name)+r'(?:_\d+)?',n)]
        if len(matches)!=1: raise RuntimeError('Expected one harbor actor: '+name+'; found '+repr(matches))
        resolved[name]=matches[0]
        p=client.simGetObjectPose(matches[0]).position
        if not np.isfinite([p.x_val,p.y_val,p.z_val]).all():
            raise RuntimeError('Missing harbor actor tag: '+name)
    report['resolved_actor_names']=resolved
    sea=client.simGetObjectPose(resolved['MH2_OceanGrid']).position.z_val
    deck=client.simGetObjectPose(resolved['MH2_ObservationPad']).position.z_val
    report['platform_underside_clearance_above_wave_bound_m']=float(sea-deck-.2-.43)
    if report['platform_underside_clearance_above_wave_bound_m'] < 2:
        raise RuntimeError('Platform clearance below the configured requirement')
    report['vehicle_height_above_deck_m']={name:float(deck-.2-client.simGetVehiclePose(vehicle_name=name).position.z_val) for name in report['vehicles']}
    if any(h < -.05 for h in report['vehicle_height_above_deck_m'].values()):
        raise RuntimeError('Vehicle is below the platform deck')
    frames=[]
    for i in range(2):
        data=client.simGetImage('harbor_overview',airsim.ImageType.Scene,external=True)
        image=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR) if data else None
        if image is None: raise RuntimeError('Overview capture unavailable')
        frames.append(image)
        (output/('overview_'+str(i)+'.png')).write_bytes(data)
        if i==0: time.sleep(2)
    report['image_difference_mean']=float(np.abs(frames[1].astype(float)-frames[0].astype(float)).mean())
    report['note']='Image difference is a visual-change check, not wave-height validation.'
    report['status']='PASS'
    (ROOT/'logs'/'maritime_harbor_latest.json').write_text(json.dumps({'output':str(output),'preview':str(output/'overview_1.png')}),encoding='utf-8')
except Exception as exc:
    report.update(status='FAIL',error=str(exc))
    raise
finally:
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(output)
