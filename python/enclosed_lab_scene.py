"""Declarative ground preview; engineering coordinates use metres, Z upwards."""
import json,math,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/enclosed_uav_lab_v1.json'
SETTINGS=ROOT/'configs/enclosed_uav_lab_preview_settings_v1.json'

def scene_definition():
    parts=[]
    def actor_id(name):
        return 'EL1_'+re.sub(r'[^A-Za-z0-9_]',lambda m:'m' if m.group(0)=='-' else 'p',name)
    def box(name,group,center,size,color,role='obstacle',collides=True):
        parts.append(dict(id=actor_id(name),group=group,mesh='Box',center_xyz_m=center,size_xyz_m=size,
            material=color,role=role,collision_shape='box' if collides else 'none',
            planner_shape='aabb' if collides else 'none',collides=collides))
    def sphere(name,group,center,diameter,color):
        parts.append(dict(id=actor_id(name),group=group,mesh='Sphere',center_xyz_m=center,size_xyz_m=[diameter]*3,
            material=color,role='target',collision_shape='sphere',planner_shape='sphere',collides=True))
    box('Ground','ground',[24,0,-.125],[60,36,.25],'floor','ground')
    for name,c,s in [('Wall_N',[24,-17.8,1.6],[60,.4,3.2]),('Wall_S',[24,17.8,1.6],[60,.4,3.2]),
                     ('Wall_W',[-5.8,0,1.6],[.4,35.6,3.2]),('Wall_E',[53.8,0,1.6],[.4,35.6,3.2])]:
        box(name,'enclosure',c,s,'wall','boundary')
    for x in range(-5,54,6):
        for side in [-1,1]:box('WallPost_%d_%d'%(x,side),'enclosure',[x,side*17.65,1.85],[.35,.65,3.7],'dark','boundary')
    for name,x in [('Start',0),('Target',48)]:
        box(name+'Platform',name.lower()+'_platform',[x,0,.3],[6,8,.6],'pad','platform')
        for y in [-4,4]:box(name+'Edge_'+str(y),name.lower()+'_platform',[x,y,.63],[6,.12,.06],'yellow','decoration',False)
        for dx in [-3,3]:box(name+'EdgeX_'+str(dx),name.lower()+'_platform',[x+dx,0,.63],[.12,8,.06],'yellow','decoration',False)
    # Two actual AirSim multirotors will be spawned on these bays by settings.
    for i,y in enumerate([-1.8,1.8]):
        box('Bay_%d'%i,'start_platform',[0,y,.608],[2.7,2.7,.016],'teal','decoration',False)
        for dx in [-.5,.5]:box('HSide_%d_%s'%(i,str(dx)),'start_platform',[dx,y,.623],[.13,1.6,.01],'white','decoration',False)
        box('HBar_%d'%i,'start_platform',[0,y,.623],[1.13,.13,.01],'white','decoration',False)
    # Visible centre-line ticks are measurements, not a planned flight route.
    for x in range(4,46,2):box('DistanceTick_%d'%x,'distance_marks',[x,0,.012],[.65,.07,.018],'line','decoration',False)
    # Low / medium / high purpose-built laboratory obstacles.
    box('LowBlock','low_block',[10,-4,.6],[4,2,1.2],'yellow')
    box('TallBlock','tall_block',[17,-1.5,2.75],[3,3,5.5],'purple')
    box('MediumBlock','medium_block',[25,5,1.75],[3.5,3,3.5],'teal')
    # A full-size container surrogate, with a colliding body and painted ribs.
    box('ContainerBody','container',[31,-4.5,1.5],[6,3,3],'blue')
    for x in [28.15+i*.45 for i in range(13)]:
        for y in [-6.012,-2.988]:box('ContainerRib_%s_%s'%(str(round(x,2)),str(y)),'container',[x,y,1.5],[.055,.018,2.8],'blue_light','decoration',False)
    for y in [-5.23,-3.77]:box('ContainerDoor_'+str(y),'container',[34.009,y,1.5],[.015,1.44,2.85],'blue_light','decoration',False)
    # Crate stack: each solid box remains an individual planner primitive.
    for i,(x,y,z) in enumerate([(20,8,.6),(21.3,8,.6),(20.65,8,1.8)]):
        box('Crate_%d'%i,'crate_stack',[x,y,z],[1.2,1.2,1.2],'wood')
        for dz in [-.43,.43]:box('CrateBand_%d_%s'%(i,str(dz)),'crate_stack',[x,y-.608,z+dz],[1.15,.015,.065],'dark','decoration',False)
    # Everyday utility cabinet and park bench, assembled from boxes.
    box('CabinetBody','utility_cabinet',[37,4,1.9],[2,1.8,3.8],'green')
    for x in [36.55,37.45]:box('CabinetDoor_'+str(x),'utility_cabinet',[x,3.091,1.9],[.83,.015,3.5],'green_light','decoration',False)
    box('BenchSeat','bench',[12,8,.52],[2.4,.65,.16],'wood')
    box('BenchBack','bench',[12,8.29,.98],[2.4,.13,.76],'wood')
    for x in [11.15,12.85]:
        for y in [7.79,8.21]:box('BenchLeg_%s_%s'%(str(x),str(y)),'bench',[x,y,.25],[.13,.13,.5],'dark')
    # Two isolation barriers leave the route choices observable.
    for i,(x,y) in enumerate([(22,-9),(38,-.5)]):
        box('BarrierBase_%d'%i,'barrier_'+str(i),[x,y,.15],[3.8,.8,.3],'yellow')
        box('BarrierBeam_%d'%i,'barrier_'+str(i),[x,y,.85],[3.8,.28,.9],'orange')
        for dx in [-1.35,1.35]:box('BarrierFoot_%d_%s'%(i,str(dx)),'barrier_'+str(i),[x+dx,y,.48],[.2,.45,.66],'dark')
    # An overhead frame has a genuinely open gap; no full bounding box fills it.
    for y in [8.5,13.5]:box('FramePost_'+str(y),'overhead_frame',[32,y,2.4],[.45,.45,4.8],'orange')
    box('FrameBeam','overhead_frame',[32,11,4.85],[.45,5.45,.35],'orange')
    sphere('Balloon_Target','balloons',[48,-2,3.1],2.0,'balloon_white_stripe')
    sphere('Balloon_RedDistractor','balloons',[48,1,2.6],1.8,'red')
    sphere('Balloon_YellowDistractor','balloons',[49.2,2.7,3.8],1.6,'balloon_yellow')
    for i,(x,y,top) in enumerate([(48,-2,2.1),(48,1,1.7),(49.2,2.7,3.0)]):
        box('BalloonTether_%d'%i,'balloons',[x,y,(.65+top)/2],[.025,.025,top-.65],'dark','support')
        box('BalloonAnchor_%d'%i,'balloons',[x,y,.66],[.22,.22,.12],'dark','support')
    return dict(schema_version=1,scene_id='enclosed_uav_lab_v1',unreal_map='/Game/EnclosedUAVLabV1/Maps/EnclosedUAVLab',
        purpose=['balloon_identity','obstacle_approach','future_two_drone_handover'],delivery_stage='GROUND_PREVIEW_ONLY',
        coordinate_system='engineering XYZ metres; X forward, Y right, Z up; ground top Z=0',
        airsim_global_origin_ue_xyz_m=[0,0,1.0],enclosure_size_m=[60,36],platform_center_distance_m=48.,
        start_bays_xyz_m=[[0,-1.8,.6],[0,1.8,.6]],target_platform_center_xyz_m=[48,0,.3],
        flight_authorized=False,target_motion=False,parts=parts)

def bounds(part):
    return [[part['center_xyz_m'][i]+s*part['size_xyz_m'][i]/2 for i in range(3)] for s in [-1,1]]

def camera(pos,aim,width=1600,height=900,fov=70,orthographic=None):
    dx,dy,dz=[aim[i]-pos[i] for i in range(3)]
    capture=dict(ImageType=0,Width=width,Height=height,FOV_Degrees=fov,
                 AutoExposureSpeed=100,AutoExposureBias=0,MotionBlurAmount=0)
    if orthographic:capture.update(ProjectionMode='orthographic',OrthoWidth=orthographic)
    return dict(X=pos[0],Y=pos[1],Z=1-pos[2],Pitch=math.degrees(math.atan2(dz,math.hypot(dx,dy))),
                Yaw=math.degrees(math.atan2(dy,dx)),Roll=0,CaptureSettings=[capture])

def write_definition():
    scene=scene_definition()
    for p in scene['parts']:p['engineering_bounds_xyz_m']=bounds(p)
    CONFIG.write_text(json.dumps(scene,ensure_ascii=False,indent=2),encoding='utf-8')
    cameras={
        'lab_overhead':camera([24,0,62],[24,0,0],1800,1200,65,66),
        'lab_overview':camera([-16,-33,32],[24,0,1.5],1800,1100,68),
        'lab_target_close':camera([42,-7,5],[48,.2,2.7],1600,1000,62),
        'lab_start_close':camera([-4,-8,4],[0,0,.8],1400,900,65)}
    # Align the 66m horizontal orthographic span with the 60m X dimension.
    cameras['lab_overhead']['Yaw']=90
    capture=[dict(ImageType=0,Width=1600,Height=900,FOV_Degrees=70,MotionBlurAmount=0,AutoExposureSpeed=100),
             dict(ImageType=2,Width=640,Height=480,FOV_Degrees=70)]
    settings=dict(SettingsVersion=1.2,SimMode='Multirotor',ApiServerPort=41453,ClockSpeed=1,ViewMode='Manual',
        CameraDirector=dict(X=-16,Y=-33,Z=-31,Pitch=-28,Yaw=39,Roll=0),Vehicles={},ExternalCameras=cameras)
    for name,y in [('DroneA',-1.8),('DroneB',1.8)]:
        settings['Vehicles'][name]=dict(VehicleType='SimpleFlight',X=0,Y=y,Z=0,Yaw=0,
            Cameras=dict(front_center=dict(X=.35,Y=0,Z=-.12,Pitch=0,Yaw=0,Roll=0,CaptureSettings=capture)))
    SETTINGS.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    return scene

if __name__=='__main__':
    s=write_definition();print(json.dumps(dict(scene=str(CONFIG),settings=str(SETTINGS),parts=len(s['parts']),flight_authorized=False)))
