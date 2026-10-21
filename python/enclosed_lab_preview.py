"""Own-session ground preview and actual Unreal camera capture, without flight."""
import argparse,hashlib,json,math,os,re,socket,subprocess,time
from datetime import datetime
from pathlib import Path
import airsim,cv2,numpy as np,psutil
from enclosed_lab_scene import ROOT,CONFIG,SETTINGS

PORT=41453
LATEST=ROOT/'logs/enclosed_lab_preview_latest.json'

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def xyz(p):return [p.x_val,p.y_val,p.z_val]
def client():return airsim.MultirotorClient(port=PORT,timeout_value=15)
def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)
def occupied():
    try:
        with socket.create_connection(('127.0.0.1',PORT),timeout=.3):return True
    except OSError:return False
def identity(pid):
    p=psutil.Process(pid);return dict(pid=pid,exe=p.exe(),creation_time=p.create_time())
def owner(m):
    if identity(m['engine']['pid'])!=m['engine']:raise RuntimeError('ENGINE_IDENTITY_CHANGED')
def enu(ned,origin):return [ned[0]+origin[0],ned[1]+origin[1],origin[2]-ned[2]]
def origin_readback(c,scene,actor_names):
    ground=next(p for p in scene['parts'] if p['id']=='EL1_Ground')
    n=xyz(c.simGetObjectPose(actor_names['EL1_Ground']).position);e=ground['center_xyz_m']
    if not np.isfinite(n).all():raise RuntimeError('GROUND_REFERENCE_MISSING')
    return [e[0]-n[0],e[1]-n[1],e[2]+n[2]]
def ground_samples(c,origin):
    result={}
    for name in ('DroneA','DroneB'):
        state=c.getMultirotorState(vehicle_name=name);p=enu(xyz(c.simGetObjectPose(name).position),origin)
        velocity=xyz(state.kinematics_estimated.linear_velocity);speed=float(np.linalg.norm(velocity))
        acc=float(np.linalg.norm(xyz(state.kinematics_estimated.linear_acceleration)))
        collision=c.simGetCollisionInfo(vehicle_name=name)
        controlled=c.isApiControlEnabled(vehicle_name=name)
        ready=(state.landed_state==airsim.LandedState.Landed and not controlled and speed<.01 and acc<.05
            and -.2<p[0]<.2 and abs(p[1])<3.5 and .64<p[2]<1.05 and 'StartPlatform' in collision.object_name)
        result[name]=dict(engineering_xyz_m=p,speed_m_s=speed,acceleration_m_s2=acc,landed=state.landed_state==airsim.LandedState.Landed,
            api_control_enabled=controlled,contact_actor=collision.object_name,ready=ready,state_timestamp_ns=int(state.timestamp))
    return result
def verify_ground(c,scene,out,actor_names):
    origin=origin_readback(c,scene,actor_names);rows=[];stable=None;deadline=time.monotonic()+25
    while time.monotonic()<deadline:
        row=dict(host_s=time.monotonic(),vehicles=ground_samples(c,origin));rows.append(row)
        if all(v['ready'] for v in row['vehicles'].values()):
            stable=row['host_s'] if stable is None else stable
            if row['host_s']-stable>=.8:
                save(out,dict(status='PASS_GROUND_ONLY',calibrated_global_origin_xyz_m=origin,samples=rows,
                    arm_requests=False,control_acquisition=False,flight_motion_requests=False));return origin
        else:stable=None
        time.sleep(.1)
    save(out,dict(status='FAIL_GROUND_READINESS',calibrated_global_origin_xyz_m=origin,samples=rows))
    raise RuntimeError('GROUND_READINESS_FAILED')

def wait_render_ready(c,path):
    """Avoid publishing UE's grey fallback materials during shader compilation."""
    rows=[];deadline=time.monotonic()+75
    while time.monotonic()<deadline:
        row=dict(host_s=time.monotonic(),red_pixels=0,yellow_pixels=0,purple_pixels=0)
        for name,vehicle,external in [('lab_target_close','',True),('front_center','DroneA',False)]:
            data=c.simGetImage(name,airsim.ImageType.Scene,vehicle_name=vehicle,external=external)
            img=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR) if data else None
            if img is None:continue
            hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            if external:
                row['red_pixels']=int(((hsv[:,:,1]>80)&(hsv[:,:,2]>60)&((hsv[:,:,0]<12)|(hsv[:,:,0]>170))).sum())
                row['yellow_pixels']=int(((hsv[:,:,0]>=16)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>80)&(hsv[:,:,2]>60)).sum())
            else:row['purple_pixels']=int(((hsv[:,:,0]>=120)&(hsv[:,:,0]<=170)&(hsv[:,:,1]>40)&(hsv[:,:,2]>60)).sum())
        rows.append(row)
        if row['red_pixels']>2000 and row['yellow_pixels']>1000 and row['purple_pixels']>1000:
            save(path,dict(status='PASS_EXPECTED_PALETTE_RENDERED',scope='material readiness only; not a detector score',samples=rows));return
        time.sleep(.5)
    save(path,dict(status='FAIL_RENDER_READINESS',samples=rows));raise RuntimeError('MATERIALS_NOT_RENDERED')

def capture(session):
    m=json.loads(Path(session).read_text(encoding='utf-8'));owner(m)
    folder=Path(m['output'])/('capture_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));folder.mkdir(exist_ok=False)
    scene=json.loads(CONFIG.read_text(encoding='utf-8'))
    if digest(CONFIG)!=m['config_sha256'] or digest(SETTINGS)!=m['settings_sha256']:raise RuntimeError('CONFIG_CHANGED')
    build=json.loads(Path(m['build_report']).read_text(encoding='utf-8'))
    actor_names={p['id']:p['unreal_actor_name'] for p in build['parts']}
    c=client();c.ping();origin=verify_ground(c,scene,folder/'ground_before_capture.json',actor_names)
    wait_render_ready(c,folder/'render_readiness.json')
    report=dict(status='RUNNING',new_flights=0,arm_requests=False,control_acquisition=False,
        flight_motion_requests=False,target_motion_requests=False,render_source='Unreal AirSim Scene cameras',
        origin_calibration_reference='EL1_Ground known engineering centre versus simGetObjectPose global NED',
        calibrated_global_origin_xyz_m=origin,geometry_readbacks=[],views={})
    if len(set(actor_names.values()))!=len(actor_names) or set(actor_names)!=set(p['id'] for p in scene['parts']):
        raise RuntimeError('EDITOR_ACTOR_MAPPING_NOT_ONE_TO_ONE')
    live_names=set(c.simListSceneObjects('EL1_.*'))
    if not set(actor_names.values()).issubset(live_names):raise RuntimeError('EDITOR_RUNTIME_ACTOR_POPULATION_CHANGED')
    for p in scene['parts']:
        actual_name=actor_names[p['id']]
        pose=c.simGetObjectPose(actual_name);position=enu(xyz(pose.position),origin)
        scale=xyz(c.simGetObjectScale(actual_name))
        error=max(abs(position[i]-p['center_xyz_m'][i]) for i in range(3))
        scale_error=max(abs(scale[i]-p['size_xyz_m'][i]) for i in range(3))
        rotation_error=max(abs(v) for v in [pose.orientation.x_val,pose.orientation.y_val,pose.orientation.z_val])
        if not np.isfinite(position+scale).all() or max(error,scale_error)>.02 or rotation_error>1e-5:
            raise RuntimeError('RUNTIME_GEOMETRY_MISMATCH_'+p['id'])
        report['geometry_readbacks'].append(dict(id=p['id'],unreal_actor_name=actual_name,engineering_xyz_m=position,size_xyz_m=scale,
            position_error_m=error,scale_error=scale_error,collision_shape=p['collision_shape'],planner_shape=p['planner_shape']))
    definitions=[('overhead','lab_overhead','',True),('overview','lab_overview','',True),
        ('drone_front','front_center','DroneA',False),('target_close','lab_target_close','',True),('start_close','lab_start_close','',True)]
    for name,camera,vehicle,external in definitions:
        # Warm the real render target before the saved request. No image synthesis.
        c.simGetImage(camera,airsim.ImageType.Scene,vehicle_name=vehicle,external=external)
        time.sleep(.25)
        data=c.simGetImage(camera,airsim.ImageType.Scene,vehicle_name=vehicle,external=external)
        decoded=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR) if data else None
        if decoded is None or float(decoded.std())<3:raise RuntimeError('CAMERA_EMPTY_'+name)
        path=folder/(name+'.png');path.write_bytes(data)
        info=c.simGetCameraInfo(camera,vehicle_name=vehicle,external=external)
        # External camera poses use global NED; vehicle camera poses use its local frame.
        report['views'][name]=dict(path=str(path),sha256=digest(path),width=decoded.shape[1],height=decoded.shape[0],
            source='unmodified simGetImage Scene PNG',camera=camera,vehicle=vehicle,external=external,
            camera_position_reported_ned_m=xyz(info.pose.position),fov_degrees=info.fov,
            camera_orientation_xyzw=[info.pose.orientation.x_val,info.pose.orientation.y_val,info.pose.orientation.z_val,info.pose.orientation.w_val])
        emit(kind='VIEW_SAVED',name=name,path=str(path))
    after=ground_samples(c,origin)
    if not all(v['ready'] for v in after.values()):raise RuntimeError('GROUND_CHANGED_DURING_CAPTURE')
    c.simPause(True)
    report.update(status='PASS_GROUND_PREVIEW_GEOMETRY',ground_after_capture=after,paused=bool(c.simIsPause()),
        parts_verified=len(scene['parts']),target_motion=False,flight_validation='NOT_RUN',two_drone_handover='NOT_RUN')
    save(folder/'preview_report.json',report);m.update(status='GROUND_PREVIEW_READY_PAUSED',report=str(folder/'preview_report.json'))
    save(session,m);save(LATEST,dict(session=str(session),report=str(folder/'preview_report.json'),output=str(folder)))
    emit(kind='GROUND_PREVIEW_READY',session=str(session),output=str(folder));return folder

def start():
    if occupied():raise RuntimeError('PREVIEW_PORT_ALREADY_OCCUPIED')
    build=json.loads((ROOT/'logs/enclosed_lab_build_latest.json').read_text(encoding='utf-8'))
    receipt=json.loads(Path(build['report']).read_text(encoding='utf-8'))
    if receipt['status']!='PASS_EDITOR_GEOMETRY_COLLISION' or receipt['config_sha256']!=digest(CONFIG):raise RuntimeError('BUILD_CHECK_REQUIRED')
    out=ROOT/'logs'/('enclosed_lab_preview_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
    exe='D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'
    project=ROOT/'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
    scene=json.loads(CONFIG.read_text(encoding='utf-8'));env=dict(os.environ);env['UE-LocalDataCachePath']=str(ROOT/'.unreal-cache')
    # Windows hidden launch; images remain available in the delivered preview panel.
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    p=subprocess.Popen([exe,str(project),scene['unreal_map'],'-game','-windowed','-ResX=1280','-ResY=720',
        '-settings='+str(SETTINGS),'-abslog='+str(out/'unreal.log'),'-nosplash'],env=env,startupinfo=startup,
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    m=dict(status='STARTING_GROUND_PREVIEW',engine=identity(p.pid),output=str(out),config=str(CONFIG),config_sha256=digest(CONFIG),
        settings=str(SETTINGS),settings_sha256=digest(SETTINGS),build_report=build['report'],new_flights=0)
    session=out/'scene_session.json';save(session,m);emit(kind='PREVIEW_SESSION',session=str(session))
    deadline=time.monotonic()+150
    while not occupied():
        if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError('UNREAL_STARTUP_FAILED')
        time.sleep(.25)
    try:return capture(session)
    except Exception as error:
        m.update(status='PREVIEW_FAILED_ENGINE_PRESERVED',error_type=type(error).__name__,error=str(error));save(session,m);raise

def close(session):
    m=json.loads(Path(session).read_text(encoding='utf-8'));owner(m);c=client()
    scene=json.loads(CONFIG.read_text(encoding='utf-8'))
    build=json.loads(Path(m['build_report']).read_text(encoding='utf-8'))
    origin=origin_readback(c,scene,{p['id']:p['unreal_actor_name'] for p in build['parts']})
    proof=ground_samples(c,origin)
    if not all(v['ready'] for v in proof.values()):raise RuntimeError('PRESERVE_NOT_GROUNDED_ENGINE')
    c.simPause(True)
    if not c.simIsPause():raise RuntimeError('PAUSE_NOT_CONFIRMED')
    save(Path(m['output'])/'close_ground_proof.json',dict(status='PASS',vehicles=proof,paused=True))
    owner(m);p=psutil.Process(m['engine']['pid']);p.terminate();p.wait(10)
    m['status']='CLOSED_GROUND_ONLY';save(session,m);emit(kind='PREVIEW_CLOSED',session=str(session))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','capture','close']);p.add_argument('--session',type=Path)
    a=p.parse_args()
    if a.action=='start':start()
    elif a.action=='capture':capture(a.session)
    else:close(a.session)
