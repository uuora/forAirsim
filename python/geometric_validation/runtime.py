"""Independent commissioning protocol. No legacy planner/controller is invoked.

All positions/velocities are world NED, quaternion wxyz is body FRD -> NED.
Preparation/landing use SimpleFlight velocity control, excluded from task timing.
"""
from __future__ import annotations
from collections import deque
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/geometric_control_validation.json'


def filesystem_path(path):
    path=Path(path).resolve()
    if os.name=='nt' and not str(path).startswith('\\\\?\\'):
        value=str(path)
        return Path('\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value)
    return path


def write_json(path, data):
    filesystem_path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return hashlib.sha256(filesystem_path(path).read_bytes()).hexdigest()


def path_at(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    from dual_target_control.machine_config import apply_machine_config
    cfg = apply_machine_config(cfg)
    if cfg['protocol'] not in ('geometric_hybrid_v1', 'geometric_planner_pass_v1') or cfg['visualization']['capture_overview'] is not False:
        raise ValueError('Wrong protocol or capture_overview enabled')
    from .reference import build_reference
    from .core import command
    for case in cfg['cases']:
        ref = build_reference(cfg, case)
        if cfg['protocol'] == 'geometric_planner_pass_v1':
            from .audit import frozen_plan_issues
            issues = frozen_plan_issues(cfg, case)
            if issues: raise ValueError('; '.join(issues))
        for t in np.linspace(0, ref.duration, 301):
            r = ref.sample(float(t))
            # Validation catches nonfinite reference/controller configuration before RPC.
            command(r['p'], r['v'], [1,0,0,0], r, cfg['controller'])
            if np.any(np.array(r['p']) < cfg['limits']['bounds_ned_min']) or np.any(np.array(r['p']) > cfg['limits']['bounds_ned_max']):
                raise ValueError('Reference outside safety box')
    l, e = cfg['limits'], cfg['execution']
    values = [v for v in l.values() if isinstance(v, (int,float))] + [e['hz']]
    if not all(math.isfinite(v) and v > 0 for v in values) or not 10 <= e['hz'] <= 100:
        raise ValueError('Invalid execution/limits')
    return cfg


def local_preflight(cfg):
    a = cfg['airsim']; project = path_at(a['project'])
    files = [path_at(a['editor']), project, path_at(a['settings']),
        project.parent/'Content'/(a['map'].removeprefix('/Game/')+'.umap'),
        project.parent/'Binaries/Win64/UE4Editor-Blocks.dll',
        project.parent/'Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll']
    result = {'files': {str(p): {'exists': p.is_file(), 'sha256': digest(p) if p.is_file() else None} for p in files}}
    settings = json.loads(path_at(a['settings']).read_text(encoding='utf-8-sig'))
    vehicle = settings['Vehicles'][a['vehicle']]
    if vehicle['VehicleType'] != 'SimpleFlight' or settings.get('ClockSpeed',1) != 1:
        raise ValueError('Requires SimpleFlight at ClockSpeed 1')
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing runtime dependencies (no flight started):\n'+'\n'.join(missing))
    result['airsim_commit'] = subprocess.check_output(['git','-C',str(ROOT/'external/AirSim'),'rev-parse','HEAD'], text=True).strip()
    result['source_binary_build_link'] = 'UNVERIFIED: binary hashes and live physics receipt bind runtime; source checkout alone does not prove compiled bytes'
    return result


class Recorder:
    """Disk writes on an independent thread. Queue overflow fails visibly."""
    def __init__(self, directory):
        self.q = queue.Queue(maxsize=20000); self.error = None
        self.stream = (directory/'samples.jsonl').open('x',encoding='utf-8')
        self.thread = threading.Thread(target=self._write, name='geometric-log', daemon=False)
        self.thread.start()

    def _write(self):
        try:
            while True:
                item = self.q.get()
                if item is None: break
                self.stream.write(json.dumps(item, allow_nan=False)+'\n')
                self.stream.flush()
        except BaseException as ex: self.error = repr(ex)
        finally: self.stream.close()

    def put(self,row):
        if self.error: raise RuntimeError('Logging failed: '+self.error)
        self.q.put_nowait(row)

    def close(self):
        if self.thread.is_alive(): self.q.put(None)
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.error: raise RuntimeError('Logger did not finish: '+str(self.error))


def state_row(client, name):
    before = time.perf_counter(); s = client.getMultirotorState(vehicle_name=name); after = time.perf_counter()
    k = s.kinematics_estimated
    def xyz(v): return [float(v.x_val),float(v.y_val),float(v.z_val)]
    q = k.orientation
    row = dict(timestamp_ns=int(s.timestamp), host_read_begin_s=before, host_received_s=after,
        position=xyz(k.position), velocity=xyz(k.linear_velocity), acceleration=xyz(k.linear_acceleration),
        angular_velocity=xyz(k.angular_velocity), quaternion_wxyz=[q.w_val,q.x_val,q.y_val,q.z_val],
        landed_state=int(s.landed_state), collision=bool(s.collision.has_collided), collision_timestamp_ns=int(s.collision.time_stamp))
    if not np.isfinite(row['position']+row['velocity']+row['angular_velocity']+row['quaternion_wxyz']).all():
        raise RuntimeError('Nonfinite state')
    return row


def connect(cfg, directory, launch=False):
    import airsim
    a=cfg['airsim']; process=None
    try:
        with socket.create_connection((a['host'],a['port']),.25): opened=True
    except OSError: opened=False
    if not opened:
        if not launch: raise RuntimeError('AirSim not running. Use --launch explicitly; no flight was started.')
        from dual_target_control.simulator import hidden_process_options
        env=dict(os.environ); env['UE-LocalDataCachePath']=str(ROOT/'.unreal-cache')
        userdir=directory/'ue_user';userdir.mkdir()
        args=[str(path_at(a['editor'])), str(path_at(a['project'])), a['map'],'-game','-windowed','-ResX=1200','-ResY=800',
            '-nosplash','-NoSound','-NoVSync','-settings='+str(path_at(a['settings'])),
            '-abslog='+str(directory/'ue_startup.log'),'-UserDir='+str(userdir),
            '-ExecCmds=t.MaxFPS 120,t.IdleWhenNotForeground 0,Slate.bAllowThrottling 0']
        process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.STDOUT,**hidden_process_options())
        write_json(directory/'launch.json',dict(pid=process.pid,args=args,created_utc_ns=time.time_ns()))
        deadline=time.perf_counter()+60
        while True:
            if process.poll() is not None:raise RuntimeError('UE exited before RPC port opened')
            try:
                with socket.create_connection((a['host'],a['port']),.2):break
            except OSError:
                if time.perf_counter()>deadline:raise TimeoutError('UE RPC port launch deadline exceeded')
                time.sleep(.2)
    c=airsim.MultirotorClient(ip=a['host'],port=a['port'],timeout_value=a['rpc_timeout'])
    end=time.perf_counter()+60
    while True:
        try:
            if c.ping(): break
        except Exception:
            if time.perf_counter()>end or (process and process.poll() is not None): raise
        if time.perf_counter()>end or (process and process.poll() is not None):raise TimeoutError('AirSim ping deadline exceeded')
        time.sleep(.2)
    return c, process


def inspect_live(client,cfg,directory):
    import psutil
    a=cfg['airsim'];name=a['vehicle']
    loaded=json.loads(client.getSettingsString())
    expected=json.loads(path_at(a['settings']).read_text(encoding='utf-8-sig'))
    if loaded != expected: raise RuntimeError('Loaded settings differ from frozen requested settings')
    if client.listVehicles()!=[name] or client.simIsPause(): raise RuntimeError('Requires one vehicle, unpaused physics')
    if client.isApiControlEnabled(vehicle_name=name): raise RuntimeError('Another API owner is active; refusing takeover')
    row=state_row(client,name)
    if row['landed_state']!=0: raise RuntimeError('Vehicle not landed')
    later=state_row(client,name)
    time.sleep(.08); later=state_row(client,name)
    if later['timestamp_ns']<=row['timestamp_ns']: raise RuntimeError('Simulation clock did not advance')
    matches=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        args=p.info['cmdline'] or []
        if any(str(path_at(a['project'])).replace('\\','/').casefold() == x.replace('\\','/').casefold() for x in args):
            logs=[x.split('=',1)[1] for x in args if x.lower().startswith('-abslog=')]
            if len(logs)==1: matches.append((p,Path(logs[0])))
    if len(matches)!=1: raise RuntimeError('Cannot uniquely bind running UE process/startup receipt')
    proc,log=matches[0]
    if a['map'] not in proc.cmdline():raise RuntimeError('Running map does not match obstacle-free commissioning map')
    raw=log.read_bytes();text=raw.decode('utf-8-sig',errors='replace')
    first=text.splitlines()[0]
    opened=datetime.strptime(first,'Log file open, %m/%d/%y %H:%M:%S').timestamp()
    if abs(opened-proc.create_time())>30: raise RuntimeError('Startup receipt process mismatch')
    receipts=[json.loads(line.split('DTC_DYNAMICS ',1)[1]) for line in text.splitlines() if 'DTC_DYNAMICS ' in line]
    receipts=[r for r in receipts if r.get('vehicle')==name]
    if not receipts: raise RuntimeError('No live dynamics receipt')
    from dual_target_control.dynamics import validate_receipt
    profile=expected['Vehicles'][name]['DynamicsProfile']
    physics=validate_receipt(profile,loaded['Vehicles'][name]['DynamicsProfile'],receipts[-1],name)
    if not math.isclose(physics['nominal_hover_throttle'],cfg['controller']['hover_throttle'],abs_tol=1e-5):
        raise RuntimeError('Controller hover mapping does not match physics receipt')
    (directory/'physics_startup.log').write_bytes(raw)
    modules=[m.path for m in proc.memory_maps() if m.path.lower().endswith('ue4editor-airsim.dll')]
    expected_module=(path_at(a['project']).parent/'Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll').resolve()
    if len(set(modules))!=1 or Path(modules[0]).resolve()!=expected_module:
        raise RuntimeError('Loaded AirSim module is not the inspected project binary')
    environment=client.simGetGroundTruthEnvironment(vehicle_name=name)
    return dict(loaded_settings=loaded,server_version=client.getServerVersion(),client_version=client.getClientVersion(),
        pid=proc.pid,process_created=proc.create_time(),dynamics=physics,initial_state=row,
        environment=dict(gravity_ned=[environment.gravity.x_val,environment.gravity.y_val,environment.gravity.z_val],air_density=environment.air_density),
        loaded_modules={str(expected_module):digest(expected_module)})


def run(config_path=DEFAULT_CONFIG, case='hover', output_root=None, telemetry=None, stop_event=None, launch=False):
    from .core import command, quaternion_to_matrix
    from .reference import build_reference
    stop_event=stop_event or threading.Event();cfg=load_config(config_path);ref=build_reference(cfg,case)
    folder=path_at(output_root or 'logs')/('geometric_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+case)
    folder.mkdir(parents=True,exist_ok=False);write_json(folder/'config.json',cfg)
    manifest=dict(protocol=cfg['protocol'],case=case,architecture=cfg['architecture'],sources={},config_sha256=digest(folder/'config.json'),
        created_utc_ns=time.time_ns(),threshold_status=cfg['threshold_status'],scope='AirSim control validation, no images or obstacles; frozen reference only')
    source_paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'python/run_geometric_control_validation.py', ROOT/'python/run_geometric_planner_comparison.py',
        ROOT/'python/dual_target_control/dynamics.py',ROOT/'python/dual_target_control/machine_config.py',
        ROOT/'external/AirSim/PythonClient/airsim/client.py',ROOT/'external/AirSim/PythonClient/airsim/types.py']
    source_base=ROOT/'external/AirSim/AirLib'
    source_paths += [source_base/p for p in ['include/vehicles/multirotor/firmwares/simple_flight/SimpleFlightApi.hpp',
        'include/vehicles/multirotor/firmwares/simple_flight/firmware/AngleLevelController.hpp',
        'include/vehicles/multirotor/firmwares/simple_flight/firmware/AngleRateController.hpp',
        'include/vehicles/multirotor/firmwares/simple_flight/firmware/Mixer.hpp',
        'include/vehicles/multirotor/firmwares/simple_flight/firmware/Params.hpp',
        'include/vehicles/multirotor/firmwares/simple_flight/AirSimSimpleFlightEstimator.hpp',
        'include/vehicles/multirotor/RotorActuator.hpp','include/vehicles/multirotor/RotorParams.hpp',
        'include/vehicles/multirotor/MultiRotorParams.hpp','include/common/VectorMath.hpp']]
    snap=folder/'source';snap.mkdir()
    for p in source_paths:
        source=filesystem_path(p)
        if source.is_file():
            relative=p.relative_to(ROOT);dest=filesystem_path(snap/relative);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
            manifest['sources'][str(relative)]=digest(dest)
    write_json(folder/'manifest.json',manifest)
    recorder=Recorder(folder);events=[];c=None;owned=False;process=None;outcome='FAILED';error=None;cleanup={'landed':False};completion=None
    a=cfg['airsim'];name=a['vehicle']; futures=deque();dt=1/cfg['execution']['hz'];last_stamp=None;initial_collision=0
    def event(kind,**values): events.append(dict(event=kind,host_s=time.perf_counter(),**values))
    def publish(row,phase):
        if telemetry:
            # Contract: callback only copies to a nonblocking mailbox. Never RPC/render here.
            try:telemetry(dict(row,t=row.get('t',0),phase=phase,planned_path_ned=path_preview,task_points_ned=ref.points,
                reference_position=(row.get('reference') or {}).get('p')))
            except Exception as ex:event('DISPLAY_FAILURE',error=repr(ex))
    path_preview=np.asarray(ref.sample_path()).tolist()
    def send_velocity(v):
        import airsim
        futures.append(c.moveByVelocityAsync(*map(float,v),a['command_duration'],yaw_mode=airsim.YawMode(False,math.degrees(cfg['yaw'])),vehicle_name=name))
    def pump():
        for f in list(futures):
            if getattr(f,'_set_flag',False): f.get();futures.remove(f)
        if len(futures)>100: raise RuntimeError('Too many pending command receipts')
    def read(phase,t=None):
        nonlocal last_stamp
        r=state_row(c,name);r['phase']=phase
        if t is not None:r['t']=t
        if last_stamp is not None:r['state_interval_s']=(r['timestamp_ns']-last_stamp)/1e9
        last_stamp=r['timestamp_ns'];return r
    try:
        manifest['local_preflight']=local_preflight(cfg);write_json(folder/'manifest.json',manifest)
        c,process=connect(cfg,folder,launch);live=inspect_live(c,cfg,folder);manifest['live']=live
        initial_collision=live['initial_state']['collision_timestamp_ns']
        manifest['condition_fingerprint']=hashlib.sha256(json.dumps(dict(config=cfg,sources=manifest['sources'],runtime=manifest['local_preflight']['files'],settings=live['loaded_settings']),sort_keys=True).encode()).hexdigest()
        if 'comparison' in cfg:
            from .planners import common_fingerprint
            manifest['comparison_fingerprint']=hashlib.sha256(json.dumps(dict(common=common_fingerprint(cfg),sources=manifest['sources'],
                runtime=manifest['local_preflight']['files'],settings=live['loaded_settings']),sort_keys=True).encode()).hexdigest()
        write_json(folder/'manifest.json',manifest)
        c.enableApiControl(True,vehicle_name=name);owned=True
        if not c.armDisarm(True,vehicle_name=name): raise RuntimeError('Arming rejected')
        event('ARMED');start=np.asarray(ref.start);e=cfg['execution'];l=cfg['limits']
        # Real closed-loop vertical/position preparation, no reset/teleport/pause.
        begin=time.perf_counter();stable=None
        while time.perf_counter()-begin<e['start_timeout_s']:
            tick=time.perf_counter()
            if stop_event.is_set():raise InterruptedError('User stopped during preparation')
            r=read('PREPARING');p=np.array(r['position']);v=np.array(r['velocity'])
            R=quaternion_to_matrix(r['quaternion_wxyz'])
            fault=None
            if r.get('state_interval_s',dt)<=0 or r.get('state_interval_s',dt)>l['max_state_interval_s']:fault='PREPARATION_TIMING'
            if math.acos(float(np.clip(R[2,2],-1,1)))>l['safety_tilt_rad'] or np.linalg.norm(r['angular_velocity'])>l['safety_rate_radps'] or np.linalg.norm(v)>l['safety_speed_mps']:fault='PREPARATION_MOTION'
            lo=np.array(l['bounds_ned_min']);hi=np.array(l['bounds_ned_max']);hi[2]=2 # includes floor during climb only
            if np.any(p<lo) or np.any(p>hi):fault='PREPARATION_BOX'
            if r['collision'] and r['collision_timestamp_ns']>initial_collision and p[2]<-1:fault='PREPARATION_COLLISION'
            if fault:
                r['failure']=fault;recorder.put(r);raise RuntimeError(fault)
            u=1.2*(start-p)-.6*v;u*=min(1,1.5/max(np.linalg.norm(u),1e-9));send_velocity(u);pump()
            nominal=command(start,[0,0,0],r['quaternion_wxyz'],ref.sample(0),cfg['controller'])
            ok=np.linalg.norm(start-p)<=e['start_position_m'] and np.linalg.norm(v)<=e['start_speed_mps'] and nominal['attitude_error_rad']<=e['start_attitude_rad']
            stable=tick if ok and stable is None else stable if ok else None
            recorder.put(r);publish(r,'PREPARING')
            if stable and tick-stable>=e['start_hold_s']:break
            time.sleep(max(0,dt-(time.perf_counter()-tick)))
        else:raise TimeoutError('Unified starting state not reached')
        initial_collision=r['collision_timestamp_ns']
        manifest['airborne_collision_baseline_ns']=initial_collision
        event('AIRBORNE_COLLISION_BASELINE',timestamp_ns=initial_collision)
        plan_begin=time.perf_counter();ref=build_reference(cfg,case);decode_wall=time.perf_counter()-plan_begin
        frozen=cfg['cases'][case].get('frozen_reference')
        manifest['planning_wall_s']=frozen['planning']['wall_s'] if frozen else decode_wall
        manifest['planning_location']='offline_before_flight' if frozen else 'airborne_reference_generation'
        manifest['reference_decode_wall_s']=decode_wall
        write_json(folder/'manifest.json',manifest)
        # Switch to geometric hover and re-establish the same starting state before release.
        phase='GEOMETRIC_SETTLING';begin=time.perf_counter();stable=None;release_stamp=None;last_stamp=None;terminal_stable=None
        while True:
            tick=time.perf_counter();r=read(phase)
            if release_stamp is None: t=0.
            else:t=(r['timestamp_ns']-release_stamp)/1e9
            r['t']=t;reference=ref.sample(t);p=r['position'];v=r['velocity'];q=r['quaternion_wxyz']
            cmd=command(p,v,q,reference,cfg['controller']);nom=command(reference['p'],reference['v'],q,reference,cfg['controller'])
            r.update(reference=reference,command=cmd,R_reference=nom['R_des'],attitude_reference_error_rad=nom['attitude_error_rad'])
            r['host_before_send_s']=time.perf_counter()
            fault=None
            interval=r.get('state_interval_s')
            if interval is not None and (interval<=0 or interval>l['max_state_interval_s']):fault='STATE_TIMING'
            R=np.asarray(quaternion_to_matrix(q));tilt=math.acos(float(np.clip(R[2,2],-1,1)))
            r['actual_tilt_rad']=tilt
            if np.linalg.norm(np.array(p)-reference['p'])>l['safety_position_error_m']:fault='POSITION_DIVERGENCE'
            if tilt>l['safety_tilt_rad'] or np.linalg.norm(r['angular_velocity'])>l['safety_rate_radps'] or np.linalg.norm(v)>l['safety_speed_mps']:fault='SAFETY_MOTION'
            if np.any(np.array(p)<l['bounds_ned_min']) or np.any(np.array(p)>l['bounds_ned_max']):fault='SAFETY_BOX'
            if r['collision'] and r['collision_timestamp_ns']>initial_collision:fault='COLLISION'
            if stop_event.is_set():fault='USER_STOP'
            if fault:
                r['failure']=fault;r['command_sent']=False;recorder.put(r);publish(r,phase)
                raise RuntimeError(fault)
            # Python wrapper negates pitch/yaw: explicitly invert desired NED Euler.
            futures.append(c.moveByRollPitchYawThrottleAsync(cmd['roll'],-cmd['pitch'],-cmd['yaw'],cmd['throttle'],a['command_duration'],vehicle_name=name))
            r['host_send_return_s']=time.perf_counter();r['command_sent']=True;pump()
            recorder.put(r);publish(r,phase)
            pos=float(np.linalg.norm(np.array(p)-reference['p']));speed=float(np.linalg.norm(v))
            if release_stamp is None:
                good=pos<=e['start_position_m'] and speed<=e['start_speed_mps'] and nom['attitude_error_rad']<=e['start_attitude_rad']
                simtime=r['timestamp_ns']/1e9
                stable=simtime if good and stable is None else stable if good else None
                if stable is not None and simtime-stable>=e['start_hold_s']:
                    # Release corresponds to next sampled simulation timestamp, never host-start time.
                    release_stamp=r['timestamp_ns'];event('RELEASE',timestamp_ns=release_stamp);phase='TRACKING';last_stamp=r['timestamp_ns']
                elif tick-begin>e['settle_timeout_s']:raise TimeoutError('Geometric hover did not settle')
            elif t>=ref.duration:
                good=pos<=l['terminal_position_m'] and speed<=l['terminal_speed_mps'] and nom['attitude_error_rad']<=l['terminal_attitude_rad']
                terminal_stable=t if good and terminal_stable is None else terminal_stable if good else None
                if terminal_stable is not None and t-terminal_stable>=e['terminal_hold_s']:
                    completion=t;outcome='COMPLETED';event('TERMINAL_STABLE',t=t);break
                if t>ref.duration+e['settle_timeout_s']:raise TimeoutError('Terminal state did not stabilize')
            time.sleep(max(0,dt-(time.perf_counter()-tick)))
    except BaseException as ex:
        error=repr(ex);event('FAILURE',error=error)
    finally:
        if c is not None and owned:
            try:
                c.cancelLastTask(vehicle_name=name);begin=time.perf_counter();stable=0
                while time.perf_counter()-begin<45:
                    tick=time.perf_counter();r=state_row(c,name);r['phase']='LANDING'
                    contact=c.simGetCollisionInfo(vehicle_name=name)
                    near=np.linalg.norm(np.array(r['position'])-[contact.position.x_val,contact.position.y_val,contact.position.z_val])<.3
                    floor=contact.object_name=='DC_CollisionFloor' and contact.normal.z_val<-.7
                    good=floor and near and np.linalg.norm(r['velocity'])<.12
                    stable=stable+1 if good else 0
                    try:recorder.put(r)
                    except Exception as ex:cleanup['landing_logging_error']=repr(ex)
                    publish(r,'LANDING')
                    if stable>=8:break
                    send_velocity([0,0,.8]);pump();time.sleep(max(0,.04-(time.perf_counter()-tick)))
                else:raise TimeoutError('No verified stable ground contact; control ownership retained')
                c.cancelLastTask(vehicle_name=name)
                if not c.armDisarm(False,vehicle_name=name):raise RuntimeError('Disarm rejected')
                end=time.perf_counter()+3
                while state_row(c,name)['landed_state']!=0:
                    if time.perf_counter()>end:raise RuntimeError('Landed firmware flag not confirmed')
                    time.sleep(.05)
                c.enableApiControl(False,vehicle_name=name);owned=False
                cleanup.update(landed=True,disarmed=True,api_control_released=True)
            except BaseException as ex:cleanup=dict(landed=False,error=repr(ex),ownership_retained=owned)
        try:recorder.close()
        except Exception as ex:cleanup['logger_error']=repr(ex)
        write_json(folder/'events.json',events);write_json(folder/'cleanup.json',cleanup)
        report=dict(protocol=cfg['protocol'],case=case,outcome=outcome,error=error,normal_completion=outcome=='COMPLETED',
            actual_completion_time_s=completion,planning_wall_s=manifest.get('planning_wall_s'),cleanup=cleanup,
            planning_location=manifest.get('planning_location'),reference_decode_wall_s=manifest.get('reference_decode_wall_s'),
            control_constraints_passed=False,data_audit_passed=False,owned_simulator_pid=process.pid if process else None)
        write_json(folder/'report.json',report)
        from .audit import audit
        result=audit(folder);report.update(control_constraints_passed=result['control_constraints_passed'],data_audit_passed=result['data_audit_passed'],metrics=result['metrics'])
        if 'waypoint_passage_passed' in result:
            report.update(waypoint_passage_passed=result['waypoint_passage_passed'],waypoint_passages=result['waypoint_passages'])
        report['accepted']=report['normal_completion'] and report['control_constraints_passed'] and report['data_audit_passed'] and all(cleanup.get(k) is True for k in ('landed','disarmed','api_control_released')) and report.get('waypoint_passage_passed', True)
        write_json(folder/'report.json',report)
    return dict(run_dir=str(folder),report=report)
