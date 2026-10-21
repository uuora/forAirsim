"""Configured static buildings, exact segment checks and a visibility-graph route.

World NED metres. The map is operator-supplied, not inferred by YOLO. This
planner is for fixed-height simulation development, not dynamic avoidance.
"""
import copy
import hashlib
import heapq
import json
import math
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DESCRIPTION='带白色横条的红色气球'

def vector(v,n=3):
    if not isinstance(v,(list,tuple)) or len(v)!=n or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v):
        raise ValueError('FINITE_SCENE_VECTOR_REQUIRED')
    return [float(x) for x in v]

def box(building,margin=0.):
    c=vector(building['center_ned_m']);s=vector(building['size_m'])
    return ([c[i]-s[i]/2-margin for i in range(3)],[c[i]+s[i]/2+margin for i in range(3)])

def intersects(start,end,low,high):
    """Closed segment vs closed AABB; touching a boundary counts as blocked."""
    a=vector(start);b=vector(end);lo=vector(low);hi=vector(high);left,right=0.,1.
    for i in range(3):
        delta=b[i]-a[i]
        if abs(delta)<1e-12:
            if a[i]<lo[i] or a[i]>hi[i]:return False
        else:
            x,y=(lo[i]-a[i])/delta,(hi[i]-a[i])/delta
            if x>y:x,y=y,x
            left=max(left,x);right=min(right,y)
            if left>right:return False
    return True

def segment_clear(start,end,scene):
    start=vector(start);end=vector(end);bounds=scene['workspace']
    if any(not bounds['low_ned_m'][i]<=p[i]<=bounds['high_ned_m'][i] for p in (start,end) for i in range(3)):return False
    return not any(intersects(start,end,*box(b,scene['clearance_m'])) for b in scene['buildings'])

def route(scene,start=None,goal=None):
    start=vector(start or scene['start_hover_ned_m']);goal=vector(goal or scene['observation_goal_ned_m'])
    if abs(start[2]-scene['fixed_height_ned_m'])>.12 or abs(goal[2]-start[2])>1e-9:raise ValueError('FIXED_HEIGHT_ROUTE_REQUIRED')
    if not segment_clear(start,start,scene) or not segment_clear(goal,goal,scene):raise ValueError('START_OR_GOAL_BLOCKED')
    nodes=[start,goal];eps=.025
    for b in scene['buildings']:
        lo,hi=box(b,scene['clearance_m'])
        if lo[2]<=start[2]<=hi[2]:
            for x in (lo[0]-eps,hi[0]+eps):
                for y in (lo[1]-eps,hi[1]+eps):
                    p=[x,y,start[2]]
                    if segment_clear(p,p,scene):nodes.append(p)
    distances={0:0.};previous={};pending=[(0.,0)]
    while pending:
        distance,i=heapq.heappop(pending)
        if distance!=distances[i]:continue
        if i==1:
            indices=[1]
            while indices[-1]!=0:indices.append(previous[indices[-1]])
            points=[nodes[j] for j in reversed(indices)]
            return dict(status='STATIC_ROUTE_AVAILABLE',waypoints_ned_m=points,length_m=distance,
                        direct_path_blocked=not segment_clear(start,goal,scene),source='operator configured static map',flight_authorized=False)
        for j,p in enumerate(nodes):
            if j==i or not segment_clear(nodes[i],p,scene):continue
            trial=distance+math.dist(nodes[i],p)
            if trial<distances.get(j,float('inf')):
                distances[j]=trial;previous[j]=i;heapq.heappush(pending,(trial,j))
    raise ValueError('NO_COLLISION_FREE_STATIC_ROUTE')

def validate_scene(value):
    scene=copy.deepcopy(value)
    expected={'schema_version','scene_id','description','fixed_height_ned_m','start_hover_ned_m','observation_goal_ned_m','workspace','clearance_m','target','distractor','buildings'}
    if set(scene)!=expected or scene['schema_version']!=1 or scene['description']!=DESCRIPTION or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',scene['scene_id']):raise ValueError('SCENE_SCHEMA_OR_TARGET_INVALID')
    if scene['fixed_height_ned_m']!=-3.:raise ValueError('DEVELOPMENT_HEIGHT_MUST_BE_MINUS_THREE')
    if isinstance(scene['clearance_m'],bool) or not isinstance(scene['clearance_m'],(int,float)) or not .3<=scene['clearance_m']<=2.:raise ValueError('CLEARANCE_03_TO_2_METRES_REQUIRED')
    if set(scene['workspace'])!={'low_ned_m','high_ned_m'}:raise ValueError('WORKSPACE_REQUIRED')
    lo=vector(scene['workspace']['low_ned_m']);hi=vector(scene['workspace']['high_ned_m'])
    if any(a>=b for a,b in zip(lo,hi)) or any(b-a>40 for a,b in zip(lo,hi)):raise ValueError('BOUNDED_WORKSPACE_REQUIRED')
    for key in ('start_hover_ned_m','observation_goal_ned_m'):scene[key]=vector(scene[key])
    for key in ('target','distractor'):
        if set(scene[key])!={'position_ned_m','scale_xyz'}:raise ValueError('TARGET_SCENE_GEOMETRY_REQUIRED')
        vector(scene[key]['position_ned_m']);size=vector(scene[key]['scale_xyz'])
        if any(not .2<=s<=3 for s in size):raise ValueError('BALLOON_SCALE_OUT_OF_RANGE')
    if not isinstance(scene['buildings'],list) or not 1<=len(scene['buildings'])<=20:raise ValueError('ONE_TO_TWENTY_BUILDINGS_REQUIRED')
    names=set()
    for b in scene['buildings']:
        if set(b)!={'name','center_ned_m','size_m'} or not re.fullmatch(r'T1Building_[A-Za-z0-9_]{1,48}',b['name']) or b['name'] in names:raise ValueError('UNIQUE_BUILDING_NAME_REQUIRED')
        names.add(b['name']);vector(b['center_ned_m']);size=vector(b['size_m'])
        if any(not .2<=s<=15 for s in size):raise ValueError('BUILDING_DIMENSIONS_OUT_OF_RANGE')
    # Vertical lifecycle corridor must not overlap any inflated building.
    start=scene['start_hover_ned_m'];ground=[start[0],start[1],.8]
    if any(intersects(ground,start,*box(b,scene['clearance_m'])) for b in scene['buildings']):raise ValueError('TAKEOFF_LANDING_COLUMN_BLOCKED')
    route(scene)
    return scene

def load_scene(path):return validate_scene(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def command_guard(position,velocity,scene):
    position=vector(position);velocity=vector(velocity,2)
    # Conservative 1s forward envelope plus measured stopping reserve.
    endpoint=[position[0]+velocity[0]*1.2,position[1]+velocity[1]*1.2,position[2]]
    return dict(allowed=segment_clear(position,endpoint,scene),position_ned_m=position,
                predicted_endpoint_ned_m=endpoint,clearance_m=scene['clearance_m'],
                map_source='configured static building AABBs',continuous_physical_safety_established=False)

def route_velocity(position,scene,speed=.03):
    path=route(scene,start=[*vector(position)[:2],scene['fixed_height_ned_m']])
    distance=math.dist(position,scene['observation_goal_ned_m'])
    if distance<.06:return [0.,0.],dict(path,goal_reached=True)
    waypoint=path['waypoints_ned_m'][1];dx,dy=waypoint[0]-position[0],waypoint[1]-position[1];n=math.hypot(dx,dy)
    return [dx/n*min(speed,n),dy/n*min(speed,n)],dict(path,goal_reached=False)

def scene_flight_geometry(scene):
    return dict(target_world_ned_m=scene['target']['position_ned_m'],distractor_world_ned_m=scene['distractor']['position_ned_m'],
                target_scale_xyz=scene['target']['scale_xyz'],occluder_world_ned_m=[30.,100.,-3.],
                inactive_world_ned_m=[30.,100.,-3.],occluder_scale_xyz=[.2,1.,2.],
                target_texture='red_64x64_white_horizontal_rows_24_to_39',distractor_texture='plain_red_64x64')

def build_buildings(client,registry,assets,scene,output):
    import airsim
    import cv2
    import numpy as np
    cube=assets.get('1m_cube_chamfer') or assets.get('cube')
    if not cube:raise RuntimeError('BUILDING_CUBE_ASSET_MISSING')
    operations=[]
    for index,b in enumerate(scene['buildings']):
        registry.ensure(client,b['name'],cube,b['center_ned_m'],b['size_m'])
        observed_scale=client.simGetObjectScale(b['name'])
        values=[observed_scale.x_val,observed_scale.y_val,observed_scale.z_val]
        if not np.allclose(values,b['size_m'],atol=.01):raise RuntimeError('BUILDING_SCALE_READBACK_MISMATCH')
        # A simple facade texture; geometry and collision derive from the mesh.
        texture=np.full((128,128,3),[145,150,155],np.uint8)
        for y in range(12,116,24):
            for x in range(10,120,24):texture[y:y+12,x:x+12]=[85,70,55]
        image=Path(output)/(b['name']+'.png');cv2.imwrite(str(image),texture)
        if not client.simSetObjectMaterialFromTexture(b['name'],str(image)):raise RuntimeError('BUILDING_TEXTURE_FAILED')
        client.simSetSegmentationObjectID(b['name'],60+index)
        operations.append(dict(name=b['name'],position_scale_readback=True,configured_aabb=box(b),physical_collision_contact_tested=False))
    return operations

def compile_task(text,scene):
    """Closed local intent parser. LLMs can later emit this same task schema."""
    if not isinstance(text,str) or not text.strip() or len(text)>500:raise ValueError('请输入500字以内的目标指令。')
    if not ('白' in text and '气球' in text):raise ValueError('当前版本仅支持带白色横条的红色气球。')
    if any(word in text for word in ('忽略障碍','撞','取消限速','关闭避障','起飞','takeoff','exec','import','http')):raise ValueError('指令包含当前不支持的动作；请使用界面的独立飞行按钮。')
    if re.search(r'(?<![\d.])(?!3(?:\.0)?\s*米)\d+(?:\.\d+)?\s*米',text):raise ValueError('当前开发飞行高度固定为3米。')
    task=dict(schema_version=1,kind='T1_OBSERVE_TARGET',scene_id=scene['scene_id'],target_description=DESCRIPTION,
              observation_goal_ned_m=scene['observation_goal_ned_m'],fixed_height_ned_m=-3.,avoid_buildings=True,
              speed_cap_m_s=.03,source='local constrained parser; LLM provider not connected',
              instruction_text=text.strip(),execution_state='PREVIEW_ONLY',flight_authorized=False)
    return task

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=ROOT/'configs/t1_building_scene_v1.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();scene=load_scene(a.scene);r=dict(scene=scene,route=route(scene),new_flights=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(r['route']['status'])
if __name__=='__main__':main()
