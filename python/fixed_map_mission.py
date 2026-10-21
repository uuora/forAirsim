"""Fixed-map known-target mission. All trajectories come from live AirSim state."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil, math, heapq, time, csv, queue, threading, traceback, socket, subprocess, os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fixed_map_flight_20261009'
PROJECT=ROOT/'external/AirSim/Unreal/Environments/Blocks'
AUDIT=ROOT/'reports/enclosed_lab_manual_20261009_145302_639883'
LOADED_EXECUTOR_SOURCE=Path(__file__).read_bytes()
LOADED_EXECUTOR_SHA256=hashlib.sha256(LOADED_EXECUTOR_SOURCE).hexdigest()

def utc(): return datetime.now(timezone.utc).isoformat(timespec='milliseconds')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows readers can briefly hold a share lock across an atomic rename.
    # Keep retries bounded to 100 ms; persistent failures remain visible.
    for attempt in range(21):
        try:
            t.replace(p);return
        except PermissionError:
            if attempt==20:raise
            time.sleep(.005)
def xyz(p): return np.array([p.x_val,p.y_val,p.z_val],float)
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

class Geometry:
    def __init__(self,parts,radius=.75):
        self.parts=parts;self.radius=radius
        self.centers=np.array([p['center_xyz_m'] for p in parts])
        self.axes=np.array([p.get('orientation_axes',np.eye(3).tolist()) for p in parts])
        self.halves=np.array([np.array(p.get('local_size_xyz_m',[0,0,0]))/2 for p in parts])
        self.spheres=np.array([p['shape']=='sphere' for p in parts])
        self.radii=np.array([p.get('radius_m',0) for p in parts])
    def distances(self,points):
        points=np.atleast_2d(points);delta=points[:,None,:]-self.centers[None,:,:]
        local=np.einsum('npi,pji->npj',delta,self.axes)
        q=np.abs(local)-self.halves
        d=np.linalg.norm(np.maximum(q,0),axis=2)+np.minimum(np.max(q,axis=2),0)
        d[:,self.spheres]=np.linalg.norm(delta[:,self.spheres],axis=2)-self.radii[self.spheres]
        return d-self.radius
    def clearance(self,p):
        d=self.distances(p)[0];i=int(d.argmin());return float(d[i]),self.parts[i]['id']
    def segment(self,a,b,minimum=1.25,ignore_support=False):
        a=np.array(a);b=np.array(b);length=float(np.linalg.norm(b-a))
        n=max(1,math.ceil(length/.05));ps=np.linspace(a,b,n+1)
        ds=self.distances(ps)
        if ignore_support:
            for i,p in enumerate(self.parts):
                if p['group'] in ('ground','start_platform'):ds[:,i]=1e6
        # Signed distance is 1-Lipschitz: subtract half maximum sample spacing.
        bound=float(ds.min())-length/(2*n)
        return bound>=minimum,dict(lower_bound_clearance_m=bound,samples=n+1,spacing_m=length/n)
    def plan(self,start,goal,minimum=1.25):
        start=np.array(start,float);goal=np.array(goal,float)
        if abs(start[2]-goal[2])>.03: raise ValueError('ROUTE_REQUIRES_CONSTANT_HEIGHT')
        if min(self.clearance(start)[0],self.clearance(goal)[0])<minimum:raise ValueError('START_OR_GOAL_CLEARANCE_FAILED')
        if self.segment(start,goal,minimum)[0]:return [start.tolist(),goal.tolist()]
        # Fixed-height 0.5 m grid; every edge and simplified segment is validated.
        lo=np.array([-4.,-15.]);step=.5;z=goal[2];nx,ny=113,61
        def point(k):return np.array([lo[0]+k[0]*step,lo[1]+k[1]*step,z])
        grid=np.array([point((i,j)) for i in range(nx) for j in range(ny)])
        free=(self.distances(grid).min(axis=1)>=minimum+.05).reshape(nx,ny)
        s=tuple(np.rint((start[:2]-lo)/step).astype(int));g=tuple(np.rint((goal[:2]-lo)/step).astype(int))
        if not free[s] or not free[g] or not self.segment(start,point(s),minimum)[0] or not self.segment(point(g),goal,minimum)[0]:raise ValueError('GRID_ENDPOINT_FAILED')
        frontier=[(0,s)];cost={s:0};parent={};closed=set()
        while frontier:
            _,u=heapq.heappop(frontier)
            if u in closed:continue
            if u==g:break
            closed.add(u)
            for di,dj in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                v=(u[0]+di,u[1]+dj)
                if not (0<=v[0]<nx and 0<=v[1]<ny) or not free[v]:continue
                if di and dj and (not free[u[0]+di,u[1]] or not free[u[0],u[1]+dj]):continue
                c=cost[u]+math.hypot(di,dj)
                if c<cost.get(v,1e9) and self.segment(point(u),point(v),minimum)[0]:
                    cost[v]=c;parent[v]=u;heapq.heappush(frontier,(c+math.dist(v,g),v))
        if g not in cost:raise ValueError('NO_COLLISION_FREE_ROUTE')
        chain=[g]
        while chain[-1]!=s:chain.append(parent[chain[-1]])
        raw=[start]+[point(k) for k in chain[::-1]]+[goal]
        result=[raw[0]];i=0
        while i<len(raw)-1:
            j=len(raw)-1
            while j>i+1 and not self.segment(raw[i],raw[j],minimum)[0]:j-=1
            if not self.segment(raw[i],raw[j],minimum)[0]:raise ValueError('PATH_SEGMENT_FAILED')
            result.append(raw[j]);i=j
        return [p.tolist() for p in result]

def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    if read(AUDIT/'geometry_check.json')['status']!='PASS_SAVED_EDITOR_GEOMETRY':raise ValueError('EDITOR_GEOMETRY_FAILED')
    dest=OUT/'fixed_version';dest.mkdir(exist_ok=False)
    shutil.copytree(AUDIT,dest/'editor_audit')
    inventory=[]
    paths=list((PROJECT/'Content/EnclosedUAVLabManual').rglob('*'))+list((PROJECT/'Config').rglob('*'))
    paths+=[PROJECT/'Blocks.uproject',ROOT/'configs/enclosed_uav_lab_manual.json',ROOT/'configs/enclosed_uav_lab_manual_settings.json']
    paths+=list((PROJECT/'Plugins/AirSim/Content').rglob('*'))
    for p in paths:
        if not p.is_file():continue
        rel=p.relative_to(ROOT);copy=dest/'snapshot'/rel;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,copy)
        digest=sha(p)
        if sha(copy)!=digest:raise ValueError('SNAPSHOT_COPY_HASH_MISMATCH')
        inventory.append(dict(source=str(p),snapshot=str(copy),sha256=digest,bytes=p.stat().st_size))
    env=read(OUT/'drone_envelope_editor.json')
    offsets=np.array(env['visual_bounds_origin_world_m'])-np.array(env['pawn_origin_world_m'])
    bound=float(np.linalg.norm(np.array(env['visual_bounds_half_extent_m'])+abs(offsets)))
    if bound>.75:raise ValueError('CONFIGURED_ENVELOPE_DOES_NOT_CONTAIN_PAWN')
    save(dest/'manifest.json',dict(version='manual_saved_20261009',frozen_at_utc=utc(),inventory=inventory,
        geometry_audit=str(dest/'editor_audit'),visual_envelope_circumsphere_calculated_m=bound,
        configured_envelope_radius_m=.75,scope='map, local assets, AirSim assets, project config, settings; runtime binaries separately hashed',
        binaries={str(p):sha(p) for p in (PROJECT/'Binaries/Win64/UE4Editor-Blocks.dll',PROJECT/'Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll') if p.exists()}))
    target=next(p for p in read(dest/'editor_audit/planning_geometry.json')['primitives'] if p['display_label']=='EL1_Balloon_Target')
    goal=[target['center_xyz_m'][0]-3.5,target['center_xyz_m'][1],3.1]
    save(dest/'task.json',dict(schema='unified_mission_v1',vehicle='DroneA',idle_vehicle='DroneB',
        target_source='known_saved_map',target_actor=target['id'],target_xyz_m=target['center_xyz_m'],goal_xyz_m=goal,
        height_world_m=3.1,max_speed_measured_m_s=.8,max_command_m_s=.65,max_command_accel_m_s2=.4,
        envelope_radius_config_m=.75,clearance_stop_m=.75,path_clearance_m=1.25,
        arrival_error_m=.2,arrival_speed_m_s=.12,arrival_dwell_s=3.,hover_drift_max_m=.25,
        sample_period_host_s=.1,command_duration_sim_s=.25,state_gap_stop_host_s=.75,
        timeout_sim_s=240.,coordinate_system='Unreal engineering metres; Z up; world ground top Z=0',
        takeoff_exception='ground and start platform contact allowed only during vertical departure; other obstacles require .75m',
        value_types={'task_settings':'configured','actual_pose_velocity':'measured AirSim kinematics','clearance_error_path':'calculated from measured pose and fixed audited geometry'},
        adapter_contract='vision/LLM adapters propose this schema; only submit_task validates and the same executor issues vehicle commands'))

def assert_fixed():
    m=read(OUT/'fixed_version/manifest.json')
    for item in m['inventory']:
        if sha(item['source'])!=item['sha256'] or sha(item['snapshot'])!=item['sha256']:raise ValueError('FIXED_VERSION_CHANGED: '+item['source'])
    for p,h in m['binaries'].items():
        if sha(p)!=h:raise ValueError('RUNTIME_BINARY_CHANGED')
    return m

def ensure_scene():
    """Launch only the fixed map; never arm, teleport, reset or issue flight commands."""
    assert_fixed()
    selected_map='/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_Manual'
    scene_config=ROOT/'configs/fixed_map_scene_runtime_v2.json'
    if not scene_config.exists():scene_config=ROOT/'configs/fixed_map_scene_runtime_v1.json'
    if scene_config.exists():
        scene=read(scene_config)
        if scene.get('runtime_map') not in ['/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_RuntimeRandom_v1','/Game/EnclosedUAVLabManual/Maps/EnclosedUAVLab_RuntimeRandom_v2'] or not scene.get('static_geometry_verified_equal'):raise ValueError('RANDOM_RUNTIME_SCENE_NOT_APPROVED')
        if sha(scene['runtime_file'])!=scene['runtime_sha256'] or sha(scene['source_file'])!=scene['source_sha256']:raise ValueError('RANDOM_RUNTIME_SCENE_VERSION_CHANGED')
        selected_map=scene['runtime_map']
    def listening():
        try:
            with socket.create_connection(('127.0.0.1',41454),timeout=.3):return True
        except OSError:return False
    if listening():
        import psutil
        owners={c.pid for c in psutil.net_connections(kind='tcp') if c.laddr and c.laddr.port==41454 and c.status=='LISTEN'}
        if len(owners)!=1:raise ValueError('SCENE_PORT_OWNER_AMBIGUOUS')
        process=psutil.Process(next(iter(owners)));command=' '.join(process.cmdline()).replace('\\','/').lower()
        expected_settings=OUT/'fixed_version/snapshot/configs/enclosed_uav_lab_manual_settings.json'
        active_settings=next((Path(a.split('=',1)[1].strip('"')) for a in process.cmdline() if a.lower().startswith('-settings=')),None)
        settings_match=active_settings is not None and active_settings.is_file() and sha(active_settings)==sha(expected_settings)
        if selected_map.lower() not in command or not settings_match or str(PROJECT/'Blocks.uproject').replace('\\','/').lower() not in command:raise ValueError('PORT_OCCUPIED_BY_DIFFERENT_SCENE')
        save(OUT/'active_engine.json',dict(utc=utc(),pid=process.pid,created=process.create_time(),exe=process.exe(),command_line=process.cmdline()))
        return
    folder=OUT/('scene_'+datetime.now().strftime('%Y%m%d_%H%M%S'));folder.mkdir()
    settings=OUT/'fixed_version/snapshot/configs/enclosed_uav_lab_manual_settings.json'
    exe='D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe'
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    environment=dict(os.environ);environment['UE-LocalDataCachePath']=str(ROOT/'.unreal-cache')
    args=[exe,str(PROJECT/'Blocks.uproject'),selected_map,
        '-game','-windowed','-ResX=1280','-ResY=720','-settings='+str(settings),'-nosplash',
        '-ExecCmds=r.SceneRenderTargetResizeMethod 2','-abslog='+str(folder/'unreal.log')]
    engine=subprocess.Popen(args,startupinfo=startup,env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    import psutil
    process=psutil.Process(engine.pid)
    save(folder/'session.json',dict(started_utc=utc(),pid=engine.pid,process_created=process.create_time(),exe=process.exe(),
        map=args[2],settings_sha256=sha(settings),fixed_manifest_sha256=sha(OUT/'fixed_version/manifest.json'),
        runtime_render_override='r.SceneRenderTargetResizeMethod=2; retain maximum capture target allocation',flight_commands_issued=False))
    deadline=time.monotonic()+150
    while not listening():
        if engine.poll() is not None or time.monotonic()>deadline:raise ValueError('FIXED_SCENE_STARTUP_FAILED')
        time.sleep(.2)
    ensure_scene()

class Executor(threading.Thread):
    """One RPC owner thread; GUI and future adapters use a single task/command queue."""
    def __init__(self):
        super().__init__(daemon=True);self.commands=queue.Queue();self.state={};self.stage='CHECKING';self.reason='';self.error=None
        # SensorMission.tick() runs on the driver thread while control events
        # are emitted by this owner thread. Serialize their shared JSONL append.
        self._event_lock=threading.Lock()
        self.previous=read(OUT/'live_status.json') if (OUT/'live_status.json').exists() else None
        self.path=[];self.trace=[];self.run_dir=None;self.log=None;self.run_done=False;self.hold_since=None;self.last_stamp=None
        self.task=read(OUT/'fixed_version/task.json');self.task_hash=sha(OUT/'fixed_version/task.json')
        self.g=Geometry(read(OUT/'fixed_version/editor_audit/planning_geometry.json')['primitives'],self.task['envelope_radius_config_m'])
        self.vcmd=np.zeros(3);self.prev=None;self.paused_stage=None;self.exit_requested=False
    def submit_task(self,task,actor='operator_interface',preview_context=None):
        if preview_context is not None:
            if actor!='perception_preview':raise ValueError('PREVIEW_REQUIRES_EXPLICIT_SOURCE')
            from fixed_map_perception_geometry import preview_task,perception_safety_parameters
            # Preview validation never alters self.task or enqueues motion.
            return preview_task(task,base_task=perception_safety_parameters(self.task),**preview_context)
        if task!=self.task:raise ValueError('PROPOSED_TASK_REQUIRES_NEW_VALIDATION')
        self.command('start',actor)
    def command(self,action,actor='operator_interface'):
        if action not in ('start','pause','stop','return','resume'):raise ValueError('UNSUPPORTED_ACTION')
        self.commands.put(dict(action=action,actor=actor,requested_at_utc=utc(),request_host_ns=time.time_ns()))
    def event(self,kind,**fields):
        v=dict(utc=utc(),host_monotonic_s=time.monotonic(),kind=kind,stage=self.stage,**fields)
        serialized=json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n'
        with self._event_lock:
            if self.log:self.log.write(serialized);self.log.flush()
            with (OUT/'session_events.jsonl').open('a',encoding='utf-8') as f:f.write(serialized)
    def capture(self,label):
        folder=(self.run_dir or OUT)/'unreal';folder.mkdir(exist_ok=True,parents=True)
        for name in ('lab_overview','lab_overhead'):
            data=self.c.simGetImage(name,self.api.ImageType.Scene,external=True)
            if not data:raise ValueError('REAL_CAMERA_EMPTY')
            (folder/(label+'_'+name+'.png')).write_bytes(data)
    def sample(self):
        sample_started=time.monotonic()
        s=self.c.getMultirotorState(vehicle_name='DroneA');b=self.c.getMultirotorState(vehicle_name='DroneB')
        local=xyz(s.kinematics_estimated.position);p=self.local_origin+local*np.array([1,1,-1]);v=xyz(s.kinematics_estimated.linear_velocity)*np.array([1,1,-1])
        bp=self.b_origin+xyz(b.kinematics_estimated.position)*np.array([1,1,-1]);bv=xyz(b.kinematics_estimated.linear_velocity)
        q=s.kinematics_estimated.orientation;pitch,roll,yaw=self.api.to_eularian_angles(q)
        clearance,nearest=self.g.clearance(p);collision=self.c.simGetCollisionInfo(vehicle_name='DroneA')
        error=float(np.linalg.norm(p-np.array(self.task['goal_xyz_m'])))
        row=dict(utc=utc(),host_time_ns=time.time_ns(),host_monotonic_s=time.monotonic(),state_timestamp_ns=int(s.timestamp),
            position_xyz_m=p.tolist(),velocity_xyz_m_s=v.tolist(),orientation_xyzw=[q.x_val,q.y_val,q.z_val,q.w_val],
            roll_pitch_yaw_rad=[roll,pitch,yaw],speed_m_s=float(np.linalg.norm(v)),goal_error_m=error,
            envelope_clearance_calculated_m=clearance,nearest_obstacle=nearest,stage=self.stage,stop_reason=self.reason,
            command_xyz_m_s=self.vcmd.tolist(),landed=s.landed_state==self.api.LandedState.Landed,
            collision_has_collided=bool(collision.has_collided),collision_timestamp_ns=int(collision.time_stamp),collision_object=collision.object_name,
            DroneB_position_xyz_m=bp.tolist(),DroneB_speed_m_s=float(np.linalg.norm(bv)),DroneB_landed=b.landed_state==self.api.LandedState.Landed,
            DroneB_api_control=self.c.isApiControlEnabled(vehicle_name='DroneB'))
        self.p=p;self.v=v;self.raw_state=s;self.b_state=b;self.state=row
        row['state_read_and_geometry_wall_s']=time.monotonic()-sample_started
        if self.log:self.log.write(json.dumps(dict(kind='state',**row),ensure_ascii=False,allow_nan=False)+'\n');self.log.flush()
        if self.run_dir:self.trace.append(p.tolist())
        save(OUT/'live_status.json',dict(**row,path=self.path,trace=self.trace[-2000:],run_dir=str(self.run_dir) if self.run_dir else None,task=self.task))
        row['state_sample_total_wall_s']=time.monotonic()-sample_started
        return row
    def calibrate(self):
        rows=read(OUT/'fixed_version/editor_audit/scene_geometry.json')['parts'];ground=next(p for p in rows if p['display_label']=='EL1_Ground')
        ned=xyz(self.c.simGetObjectPose(ground['id']).position);center=np.array(ground['center_xyz_m'])
        self.global_origin=center-ned*np.array([1,1,-1]);errors=[]
        for part in rows:
            pose=self.c.simGetObjectPose(part['id']);actual=self.global_origin+xyz(pose.position)*np.array([1,1,-1]);scale=xyz(self.c.simGetObjectScale(part['id']))
            q=pose.orientation;w=q.w_val;x=-q.x_val;y=-q.y_val;z=q.z_val
            rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
            error=max(float(np.max(abs(actual-part['center_xyz_m']))),float(np.max(abs(scale-part['local_size_xyz_m']))),float(np.max(abs(rotation.T-np.array(part['orientation_axes'])))))
            errors.append(dict(id=part['id'],actual_xyz_m=actual.tolist(),scale=scale.tolist(),max_transform_error=error))
            if not np.isfinite(actual).all() or error>.02:raise ValueError('RUNTIME_GEOMETRY_MISMATCH: '+part['id'])
        self.local_origin=self.global_origin+xyz(self.c.simGetObjectPose('DroneA').position)*np.array([1,1,-1])-xyz(self.c.getMultirotorState(vehicle_name='DroneA').kinematics_estimated.position)*np.array([1,1,-1])
        self.b_origin=self.global_origin+xyz(self.c.simGetObjectPose('DroneB').position)*np.array([1,1,-1])-xyz(self.c.getMultirotorState(vehicle_name='DroneB').kinematics_estimated.position)*np.array([1,1,-1])
        save(OUT/'runtime_geometry.json',dict(status='PASS',utc=utc(),actors=errors,global_origin_xyz_m=self.global_origin.tolist(),DroneA_local_origin_xyz_m=self.local_origin.tolist(),DroneB_local_origin_xyz_m=self.b_origin.tolist(),scope='134 pose/scale/rotation readbacks; collision shape from frozen editor audit'))
    def prepare(self):
        assert_fixed();self.c.simPause(False);self.calibrate();self.sample()
        if self.c.isApiControlEnabled(vehicle_name='DroneA') and self.previous and self.previous['stage'] in ('ARRIVED','ABORTED','PAUSED'):
            prior=self.previous;folder=Path(prior['run_dir']).resolve()
            if not folder.is_relative_to(OUT) or read(folder/'task.json')!=self.task:raise ValueError('HOVER_RESUME_TASK_MISMATCH')
            if self.state['speed_m_s']>.12 or np.linalg.norm(self.p-np.array(prior['position_xyz_m']))>.25 or not self.state['DroneB_landed'] or self.state['DroneB_api_control']:raise ValueError('HOVER_RESUME_STATE_MISMATCH')
            original=read(folder/'plan.json');self.home=np.array(original['home_xyz_m']);self.path=prior['path'];self.path_receipt=original
            self.ground_b=self.state['DroneB_position_xyz_m'];self.trace=prior['trace'];self.run_dir=folder
            self.log=(folder/'timestamped.jsonl').open('a',encoding='utf-8');self.stage=prior['stage'];self.reason=prior['stop_reason']
            self.collision_base=int(self.c.simGetCollisionInfo(vehicle_name='DroneA').time_stamp);self.start_stamp=int(self.raw_state.timestamp)
            self.event('HOVER_SESSION_REATTACHED',executor_source_sha256=sha(Path(__file__)),pose_delta_m=float(np.linalg.norm(self.p-np.array(prior['position_xyz_m']))))
            self.last_host=None;self.last_stamp=None;return
        if not self.state['landed'] or not self.state['DroneB_landed'] or self.state['DroneB_api_control']:raise ValueError('GROUND_STATE_NOT_READY')
        self.home=self.p.copy();hover=self.home.copy();hover[2]=self.task['height_world_m']
        ok,vertical=self.g.segment(self.home,hover,.75,True)
        if not ok:raise ValueError('TAKEOFF_COLUMN_BLOCKED')
        self.path=self.g.plan(hover,self.task['goal_xyz_m'],self.task['path_clearance_m'])
        self.ground_b=self.state['DroneB_position_xyz_m'];self.path_receipt=dict(status='PASS',utc=utc(),home_xyz_m=self.home.tolist(),vertical=vertical,
            path=self.path,segments=[self.g.segment(a,b,self.task['path_clearance_m'])[1] for a,b in zip(self.path,self.path[1:])],
            task_sha256=self.task_hash,manifest_sha256=sha(OUT/'fixed_version/manifest.json'),
            direct_path_safe=self.g.segment(hover,self.task['goal_xyz_m'],self.task['path_clearance_m'])[0])
        save(OUT/'preflight.json',self.path_receipt);self.capture('ground');self.stage='READY';self.event('PRECHECK_PASS')
    def hold(self,reason,stage):
        self.c.cancelLastTask(vehicle_name='DroneA');self.c.hoverAsync(vehicle_name='DroneA');self.vcmd=np.zeros(3)
        self.reason=reason;self.stage=stage;self.hold_since=None;self.event('HOVER_COMMAND',reason=reason)
    def start_task(self,request):
        if self.stage!='READY':raise ValueError('START_REQUIRES_READY_GROUND')
        assert_fixed()
        if sha(Path(__file__))!=LOADED_EXECUTOR_SHA256:raise ValueError('EXECUTOR_SOURCE_CHANGED_REOPEN_UI_ON_GROUND')
        if sha(OUT/'fixed_version/task.json')!=self.task_hash:raise ValueError('TASK_CHANGED')
        self.calibrate();self.sample()
        if not self.state['landed'] or self.state['DroneB_api_control'] or not self.state['DroneB_landed']:raise ValueError('START_GROUND_GATE_FAILED')
        hover=self.p.copy();hover[2]=self.task['height_world_m']
        if not self.g.segment(self.p,hover,.75,True)[0]:raise ValueError('LIVE_TAKEOFF_COLUMN_FAILED')
        self.path=self.g.plan(hover,self.task['goal_xyz_m'],self.task['path_clearance_m'])
        self.run_dir=OUT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));self.run_dir.mkdir()
        save(self.run_dir/'task.json',self.task);save(self.run_dir/'plan.json',{**self.path_receipt,'path':self.path})
        source=self.run_dir/'source_snapshot';source.mkdir()
        (source/'fixed_map_mission.py').write_bytes(LOADED_EXECUTOR_SOURCE)
        shutil.copyfile(ROOT/'python/fixed_map_dashboard.py',source/'fixed_map_dashboard.py')
        save(source/'manifest.json',{p.name:sha(p) for p in source.glob('*.py')})
        if self.log:self.log.close()
        self.log=(self.run_dir/'timestamped.jsonl').open('w',encoding='utf-8');self.trace=[];self.run_done=False;self.hold_since=None
        self.reason='';self.stage='TAKEOFF';self.target=hover;self.index=1;self.vcmd=np.zeros(3)
        self.start_stamp=int(self.raw_state.timestamp);self.collision_base=int(self.c.simGetCollisionInfo(vehicle_name='DroneA').time_stamp)
        # Preflight is deliberately performed on the ground. Begin the live watchdog
        # at this fresh read, so planning time is not misclassified as a flight gap.
        self.last_host=time.monotonic();self.last_stamp=None;self.prev=None
        self.c.enableApiControl(True,vehicle_name='DroneA')
        if not self.c.armDisarm(True,vehicle_name='DroneA'):raise ValueError('ARM_FAILED')
        self.event('START_ACCEPTED',request=request,flight_authorized_by='user request; ground and geometry gates passed')
    def process_request(self,r):
        a=r['action'];self.event('INTERFACE_REQUEST',request=r)
        if a=='start':self.start_task(r)
        elif a=='pause' and self.stage in ('TAKEOFF','APPROACH','RETURN'):
            self.paused_stage=self.stage;self.hold('OPERATOR_PAUSE','PAUSED')
        elif a=='resume' and self.stage=='PAUSED':
            if self.paused_stage=='TAKEOFF':self.stage='TAKEOFF'
            else:
                goal=self.task['goal_xyz_m'] if self.paused_stage=='APPROACH' else [self.home[0],self.home[1],self.task['height_world_m']]
                self.path=self.g.plan(self.p,goal,self.task['path_clearance_m']);self.index=1;self.stage=self.paused_stage
            self.sample();self.last_host=time.monotonic();self.last_stamp=None;self.prev=None
            self.reason='';self.vcmd=np.zeros(3)
        elif a=='stop' and self.stage not in ('READY','CHECKING','FAILED'):
            self.hold('OPERATOR_STOP','STOP_HOVER');self.capture('stop_requested')
        elif a=='return' and self.stage in ('PAUSED','ARRIVED','ABORTED','STOP_HOVER','ARRIVAL_HOVER'):
            assert_fixed();goal=[self.home[0],self.home[1],self.task['height_world_m']]
            self.path=self.g.plan(self.p,goal,self.task['path_clearance_m']);self.sample();self.index=1;self.reason='';self.stage='RETURN';self.vcmd=np.zeros(3)
            self.last_host=time.monotonic();self.last_stamp=None;self.prev=None;self.start_stamp=int(self.raw_state.timestamp)
            self.event('RETURN_PLAN',path=self.path)
        else: self.event('REQUEST_REJECTED',action=a,reason='ACTION_INCOMPATIBLE_WITH_PHASE')
    def control(self):
        now=time.monotonic();dt=.1 if self.prev is None else min(.2,now-self.prev);self.prev=now
        r=self.state
        if r['DroneB_api_control'] or not r['DroneB_landed'] or np.linalg.norm(np.array(r['DroneB_position_xyz_m'])-self.ground_b)>.1:raise ValueError('DRONE_B_GROUND_INVARIANT_FAILED')
        if self.stage in ('TAKEOFF','APPROACH','RETURN','LANDING'):
            if r['speed_m_s']>self.task['max_speed_measured_m_s']:raise ValueError('MEASURED_SPEED_LIMIT_EXCEEDED')
            if int(self.raw_state.timestamp)-self.start_stamp>self.task['timeout_sim_s']*1e9 and self.stage!='RETURN':raise ValueError('MISSION_TIMEOUT')
            if r['collision_timestamp_ns']>self.collision_base and r['collision_has_collided'] and self.stage!='LANDING':
                expected_departure=self.stage=='TAKEOFF' and ('StartPlatform' in r['collision_object'] or 'Ground' in r['collision_object']) and self.p[2]<self.home[2]+.15
                if not expected_departure:raise ValueError('NEW_COLLISION: '+r['collision_object'])
                self.collision_base=r['collision_timestamp_ns']
            if self.stage in ('APPROACH','RETURN') and r['envelope_clearance_calculated_m']<self.task['clearance_stop_m']:raise ValueError('CLEARANCE_BELOW_THRESHOLD: '+r['nearest_obstacle'])
            if self.stage=='TAKEOFF':
                if np.linalg.norm(self.p-self.target)<.12 and r['speed_m_s']<.15:self.stage='APPROACH';self.event('TAKEOFF_REACHED')
            elif self.stage in ('APPROACH','RETURN'):
                self.target=np.array(self.path[self.index])
                if np.linalg.norm(self.p-self.target)<.18 and r['speed_m_s']<.2:
                    if self.index<len(self.path)-1:self.index+=1;self.target=np.array(self.path[self.index]);self.event('WAYPOINT_REACHED',index=self.index-1)
                    elif self.stage=='APPROACH':self.hold('GOAL_REACHED_PENDING_DWELL','ARRIVAL_HOVER');return
                    else:
                        self.stage='LANDING';self.target=self.home.copy();self.target[2]-=.6;self.event('HOME_COLUMN_REACHED')
            support_contact=self.stage=='LANDING' and 'StartPlatform' in r['collision_object'] and r['collision_timestamp_ns']>self.collision_base and np.linalg.norm(self.p[:2]-self.home[:2])<.15 and abs(self.p[2]-self.home[2])<.08 and r['speed_m_s']<.04
            if support_contact:
                if self.hold_since is None:self.hold_since=int(self.raw_state.timestamp)
            elif self.stage=='LANDING':self.hold_since=None
            support_stable=support_contact and (int(self.raw_state.timestamp)-self.hold_since)/1e9>=1.
            if self.stage=='LANDING' and (r['landed'] or support_stable) and r['speed_m_s']<.08:
                self.event('TOUCHDOWN_SUPPORT_PROOF',landed_flag=r['landed'],support_stable=bool(support_stable))
                self.c.armDisarm(False,vehicle_name='DroneA');self.c.enableApiControl(False,vehicle_name='DroneA')
                self.event('RETURN_LANDED_RELEASED');self.stage='READY';self.vcmd=np.zeros(3);self.finish('RETURNED');return
            delta=self.target-self.p;desired=.7*delta-.25*self.v
            cap=.25 if self.stage=='LANDING' else self.task['max_command_m_s']
            norm=np.linalg.norm(desired)
            if norm>cap:desired*=cap/norm
            previous_vcmd=self.vcmd.copy();dv=desired-self.vcmd;limit=self.task['max_command_accel_m_s2']*dt
            if np.linalg.norm(dv)>limit:dv*=limit/np.linalg.norm(dv)
            proposed=self.vcmd+dv
            endpoint=self.p+proposed*.7
            if self.stage in ('APPROACH','RETURN') and not self.g.segment(self.p,endpoint,self.task['clearance_stop_m'])[0]:raise ValueError('PREDICTED_CONTROL_SEGMENT_UNSAFE')
            if self.stage=='TAKEOFF' and not self.g.segment(self.p,endpoint,.75,True)[0]:raise ValueError('DEPARTURE_COLUMN_UNSAFE')
            if self.stage=='LANDING' and not self.g.segment(self.p,endpoint,.75,True)[0]:raise ValueError('LANDING_COLUMN_UNSAFE')
            self.vcmd=proposed
            self.c.moveByVelocityAsync(float(proposed[0]),float(proposed[1]),float(-proposed[2]),self.task['command_duration_sim_s'],
                drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=self.api.YawMode(False,0),vehicle_name='DroneA')
            if self.log:
                self.log.write(json.dumps(dict(kind='velocity_command_requested',utc=utc(),host_monotonic_s=time.monotonic(),stage=self.stage,
                    source_state_timestamp_ns=r['state_timestamp_ns'],world_xyz_m_s=proposed.tolist(),duration_sim_s=self.task['command_duration_sim_s'],
                    command_generation_monotonic_s=now,command_ramp_dt_s=dt,previous_command_xyz_m_s=previous_vcmd.tolist(),command_ramp_accel_m_s2=float(np.linalg.norm(proposed-previous_vcmd)/dt) if dt>0 else None,command_ramp_zero_dt=dt==0,
                    vehicle='DroneA',value_type='configured command; actual response is the next measured state'))+'\n');self.log.flush()
        elif self.stage in ('STOP_HOVER','ARRIVAL_HOVER'):
            arrival=self.stage=='ARRIVAL_HOVER'
            good=r['speed_m_s']<=self.task['arrival_speed_m_s'] and (not arrival or r['goal_error_m']<=self.task['arrival_error_m'])
            if good:
                if self.hold_since is None:self.hold_since=int(self.raw_state.timestamp);self.hold_pos=self.p.copy()
                if np.linalg.norm(self.p-self.hold_pos)>self.task['hover_drift_max_m']:self.hold_since=None
                elif (int(self.raw_state.timestamp)-self.hold_since)/1e9>=self.task['arrival_dwell_s']:
                    self.stage='ARRIVED' if arrival else 'ABORTED';self.event('HOVER_VERIFIED',displacement_m=float(np.linalg.norm(self.p-self.hold_pos)),
                        hold_start_timestamp_ns=self.hold_since,hold_end_timestamp_ns=int(self.raw_state.timestamp),
                        stable_hold_sim_s=(int(self.raw_state.timestamp)-self.hold_since)/1e9)
                    self.capture(self.stage.lower());self.finish(self.stage)
            else:self.hold_since=None
    def finish(self,status):
        if not self.run_dir:return
        save(self.run_dir/('result_'+status.lower()+'.json'),dict(status=status,utc=utc(),state=self.state,reason=self.reason,
            task_sha256=self.task_hash,fixed_manifest_sha256=sha(OUT/'fixed_version/manifest.json'),
            evidence='live AirSim state; no reset, teleport, synthetic flight samples or prerecorded trajectory',
            manual_input_provenance='see INTERFACE_REQUEST actor; GUI events are logged separately from automated requests'))
        self.run_done=True
    def run(self):
        import airsim
        self.api=airsim;self.c=airsim.MultirotorClient(port=41454,timeout_value=2)
        try:
            self.c.ping();self.prepare()
            while not self.exit_requested:
                tick=time.monotonic();r=self.sample()
                if self.last_stamp is not None and self.stage in ('TAKEOFF','APPROACH','RETURN','LANDING'):
                    if r['state_timestamp_ns']<=self.last_stamp:raise ValueError('STALE_STATE_TIMESTAMP')
                self.last_stamp=r['state_timestamp_ns']
                if self.last_host is not None and tick-self.last_host>self.task['state_gap_stop_host_s'] and self.stage in ('TAKEOFF','APPROACH','RETURN','LANDING'):raise ValueError('STATE_READ_GAP_EXCEEDED')
                self.last_host=tick
                while not self.commands.empty():
                    try:self.process_request(self.commands.get_nowait())
                    except ValueError as e:
                        if self.stage in ('READY','CHECKING'):self.reason=str(e);self.event('GATE_REJECTED',reason=str(e))
                        else:raise
                self.control();time.sleep(max(0,.1-(time.monotonic()-tick)))
        except Exception as e:
            self.error=traceback.format_exc();self.reason=str(e);self.event('EXECUTION_FAILED',reason=self.reason,traceback=self.error)
            try:
                if self.c.isApiControlEnabled(vehicle_name='DroneA'):self.hold(self.reason,'FAILED_HOVER')
                else:self.stage='FAILED'
            except Exception as stop_error:self.stage='FAILED';self.event('HOVER_RPC_FAILED',reason=str(stop_error))
            self.finish('FAILED')
            # Failure recovery stays read-only except hover; no automatic retry or continuation.
            for _ in range(50):
                try:self.sample()
                except Exception:break
                time.sleep(.1)
        finally:
            if self.log:self.log.flush()
    last_host=None

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','verify']);a=parser.parse_args()
    freeze() if a.action=='freeze' else assert_fixed()
    print(a.action+' PASS')
