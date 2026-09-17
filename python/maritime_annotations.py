"""Simulator evaluation annotations only; never an inference tool."""
import json
import re
import time
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def setup(client, output):
    registry=json.loads((ROOT/'configs/maritime_targets.json').read_text())
    scene=client.simListSceneObjects('MH2_.*')
    resolved={}
    for target in registry['targets']:
        matches=[n for n in scene if re.fullmatch(target['actor_pattern'],n)]
        if len(matches)!=target['expected_components']: raise RuntimeError('Target registry mismatch: '+target['id'])
        resolved[target['id']]=matches
    for vehicle in ('DroneA','DroneB'):
        client.simClearDetectionMeshNames('front_center',0,vehicle_name=vehicle)
        client.simSetDetectionFilterRadius('front_center',0,20000,vehicle_name=vehicle)
        for names in resolved.values():
            for name in names: client.simAddDetectionFilterMeshName('front_center',0,name,vehicle_name=vehicle)
    folder=output/'evaluation_ground_truth'; folder.mkdir()
    (folder/'registry.json').write_text(json.dumps({'registry':registry,'resolved_names':resolved},indent=2))
    return registry,resolved,folder

def annotate(client, context, row, frame):
    registry,resolved,folder=context
    start=time.time_ns()
    detections=client.simGetDetections(row['camera'],0,vehicle_name=row['vehicle'])
    boxes={d.name:[d.box2D.min.x_val,d.box2D.min.y_val,d.box2D.max.x_val,d.box2D.max.y_val] for d in detections}
    targets=[]
    preview=frame.copy()
    for target in registry['targets']:
        names=resolved[target['id']]
        parts=[boxes[n] for n in names if n in boxes]
        poses={}
        for name in names:
            p=client.simGetObjectPose(name).position
            xyz=[p.x_val,p.y_val,p.z_val]
            if not np.isfinite(xyz).all(): raise RuntimeError('Invalid target pose: '+name)
            poses[name]=xyz
        item={'target_id':target['id'],'category_id':target['category_id'],'status':'returned' if parts else 'not_returned_visibility_unknown','global_ned_component_positions_m':poses,'returned_components':[n for n in names if n in boxes]}
        if parts:
            x1=max(0,min(b[0] for b in parts)); y1=max(0,min(b[1] for b in parts))
            x2=min(row['width'],max(b[2] for b in parts)); y2=min(row['height'],max(b[3] for b in parts))
            if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1: raise RuntimeError('Invalid annotation box')
            item['bbox_xyxy_pixels']=[x1,y1,x2,y2]
            item['box_semantics']='union_of_returned_component_projected_bounds_not_visible_mask'
            cv2.rectangle(preview,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),2)
            cv2.putText(preview,target['id'],(int(x1),max(20,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
        targets.append(item)
    result={'image':row['image'],'pair_id':row['pair_id'],'vehicle':row['vehicle'],'image_timestamp_ns':row['timestamp_ns'],'annotation_host_start_ns':start,'annotation_host_end_ns':time.time_ns(),'source':'AirSim simulator API; not YOLO prediction','temporal_alignment':'separate RPC after image; static targets in these profiles','review_status':'REQUIRES_REVIEW_BEFORE_TRAINING','targets':targets}
    with (folder/'annotations.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(result,allow_nan=False)+'\n')
    if row['pair_id']==0: cv2.imwrite(str(folder/(row['vehicle']+'_review.jpg')),preview)

def cleanup(client):
    for vehicle in ('DroneA','DroneB'): client.simClearDetectionMeshNames('front_center',0,vehicle_name=vehicle)
