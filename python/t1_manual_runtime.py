"""CLI for the manual workbench. Separate scene preparation and flight actions."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
from t1_obstacle_scene import ROOT,load_scene,route,compile_task,scene_flight_geometry,build_buildings,box,intersects

DEFAULT_SCENE=ROOT/'configs/t1_building_scene_v2.json'
RUNTIME_CONFIG=ROOT/'configs/t1_manual_runtime_v1.json'

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)
def stamp():return datetime.now().strftime('%Y%m%d_%H%M%S_%f')
def emit(**value):print(json.dumps(value,ensure_ascii=False),flush=True)
def rpc_busy():
    try:
        with socket.create_connection(('127.0.0.1',41451),timeout=.3):return True
    except OSError:return False
def gate():
    data=json.loads(RUNTIME_CONFIG.read_text(encoding='utf-8'))
    path=Path(data['offline_gate']);r=json.loads(path.read_text(encoding='utf-8'))
    if r['status']!='PASS_OFFLINE_SPAWN_INJECTIONS_V11P1':raise ValueError('新版本离线门控尚未通过。')
    return path
def owner_identity(pid):
    import psutil
    p=psutil.Process(pid)
    return dict(pid=pid,executable=p.exe(),creation_time=p.create_time())
def assert_owner(manifest):
    observed=owner_identity(manifest['engine']['pid'])
    if observed!=manifest['engine']:raise RuntimeError('SCENE_ENGINE_IDENTITY_CHANGED')

def load_operator_scene(path):
    scene=load_scene(path)
    if scene['clearance_m']<.75:raise ValueError('人工场景的建筑净空至少为0.75米。')
    goal=scene['observation_goal_ned_m']
    if any(intersects([goal[0],goal[1],.8],goal,*box(b,scene['clearance_m'])) for b in scene['buildings']):
        raise ValueError('观察点的垂直降落柱被建筑占用。')
    return scene

def snapshot_manual_run(plan_path,plan,folder):
    from prepare_t1_identity_flight import copy_verified,digest
    snapshot=folder/'source_snapshot';snapshot.mkdir(parents=True,exist_ok=False)
    copy_verified(plan_path,snapshot/'plan.json',digest(plan_path))
    for item in plan['source_inventory']:
        copy_verified(item['path'],snapshot/item['relative_path'],item['sha256'])
    for item in plan['legacy_source_inventory']:
        copy_verified(item['path'],snapshot/'legacy_v6p2'/Path(item['path']).name,item['sha256'])
    for key in ('settings','detector_config','training_receipt','weights','legacy_freeze'):
        name='specialist_best.pt' if key=='weights' else Path(plan[key+'_path']).name
        copy_verified(plan[key+'_path'],snapshot/name,plan[key+'_sha256'])

def fresh_ground_closure(client,path):
    from braking_startup_readiness import vehicle_sample
    receipt=dict(kind='FRESH_PAUSED_GROUND_ONLY_CLOSURE_PROOF',status='NOT_READY',
        arm_requests=False,control_acquisition=False,motion_requests=False,unpause_requests=False,samples=[])
    stable=None;deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        row=dict(paused=bool(client.simIsPause()),vehicles={n:vehicle_sample(client,n) for n in ('DroneA','DroneB')},readback_completed_host_s=time.monotonic())
        receipt['samples'].append(row)
        ready=row['paused'] and all(v['ground_readiness'] for v in row['vehicles'].values())
        stable=(row['readback_completed_host_s'] if stable is None else stable) if ready else None
        if stable is not None and row['readback_completed_host_s']-stable>=.5:
            receipt.update(status='READY',grounded_and_released=True,paused=True,stable_query_receipt_span_s=row['readback_completed_host_s']-stable);break
        time.sleep(.1)
    save(path,receipt)
    if receipt['status']!='READY':raise ValueError('无人机尚未完成地面、释放与暂停核查，请等待收尾。')
    return receipt

def scene_start(scene_path):
    if rpc_busy():raise ValueError('已有AirSim占用端口；请先结束原实验。新场景不会覆盖它。')
    scene=load_operator_scene(scene_path);out=ROOT/'logs'/('t1_manual_scene_'+stamp());out.mkdir()
    settings=json.loads((ROOT/'configs/t1_process_zero_v11_20261007/settings.json').read_text(encoding='utf-8'))
    settings['Vehicles']['DroneA'].update(X=scene['start_hover_ned_m'][0],Y=scene['start_hover_ned_m'][1])
    settings_path=out/'settings.json';save(settings_path,settings);save(out/'scene_config.json',scene)
    exe=Path('D:/UE_4.27/Engine/Binaries/Win64/UE4Editor.exe')
    project=ROOT/'external/AirSim/Unreal/Environments/Blocks/Blocks.uproject'
    if not exe.is_file() or not project.is_file():raise ValueError('未找到本机UE4或Blocks工程。')
    cmd=[str(exe),str(project),'/Game/FlyingCPP/Maps/FlyingExampleMap','-game','-windowed','-ResX=1280','-ResY=720',f'-settings={settings_path}',f'-abslog={out/"unreal.log"}','-nosplash']
    engine=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    manifest=dict(status='STARTING_SCENE',engine=owner_identity(engine.pid),scene_snapshot=str(out/'scene_config.json'),settings_path=str(settings_path),settings_sha256=hashlib.sha256(settings_path.read_bytes()).hexdigest(),output=str(out),new_flights=0)
    path=out/'scene_session.json';save(path,manifest);emit(kind='SCENE_SESSION',path=str(path),status=manifest['status'])
    try:
        deadline=time.monotonic()+150
        while not rpc_busy():
            if engine.poll() is not None or time.monotonic()>deadline:raise RuntimeError('SCENE_STARTUP_FAILED')
            time.sleep(.25)
        assert_owner(manifest)
        import airsim
        from t1_ground_perception_probe import ground_states
        from t1_identity_flight_v2 import setup_scene
        client=airsim.MultirotorClient(timeout_value=10);client.ping()
        from braking_startup_readiness import probe
        if not probe(out/'ground_readiness.json'):raise ValueError('无人机地面就绪核查未通过。')
        ground=ground_states(client)
        registry,assets=setup_scene(client,out,scene_flight_geometry(scene))
        buildings=build_buildings(client,registry,assets,scene,out)
        manifest.update(status='SCENE_READY_GROUND_ONLY',ground=ground,buildings=buildings,route=route(scene),new_flights=0)
        save(path,manifest);emit(kind='SCENE_READY',path=str(path),status=manifest['status'])
    except Exception as e:
        manifest.update(status='SCENE_PREPARATION_FAILED',error_type=type(e).__name__);save(path,manifest);raise
    return path

def prepare_manual(condition,scene_path=None,session_path=None,task_path=None):
    from prepare_t1_identity_flight_v11p4 import prepare
    session=None
    if scene_path:
        if not session_path:raise ValueError('请先启动新场景。')
        session=json.loads(Path(session_path).read_text(encoding='utf-8'));assert_owner(session)
        if session['status']!='SCENE_READY_GROUND_ONLY':raise ValueError('场景尚未就绪或已消费。')
        if load_operator_scene(scene_path)!=load_operator_scene(session['scene_snapshot']):raise ValueError('场景配置已修改，请重启新场景。')
        if not task_path:raise ValueError('请先生成任务预览。')
        task=json.loads(Path(task_path).read_text(encoding='utf-8'))
        expected=compile_task(task.get('instruction_text',''),load_scene(scene_path))
        if task!=expected:raise ValueError('任务预览与当前场景不匹配。')
    elif rpc_busy():raise ValueError('已有场景正在运行，基础验收需要独立UE会话。')
    sid='t1_manual_'+condition+'_'+stamp();out=ROOT/'configs'/sid
    runtime=json.loads(RUNTIME_CONFIG.read_text(encoding='utf-8'));device=runtime.get('detector_device','cpu')
    if device not in ('cpu','cuda'):raise ValueError('检测设备配置必须为cpu或cuda。')
    plan=prepare(out,condition,sid,gate(),device=device,scene=scene_path)
    from prepare_t1_identity_flight import digest,copy_verified,save as save_plan
    for name in ('python/t1_manual_runtime.py','python/t1_manual_workbench.py','python/vlx_gui_qt.py'):
        if name in plan['source_sha256']:continue
        path=ROOT/name;sha=digest(path);snapshot=out/'source_bundle'/name;copy_verified(path,snapshot,sha)
        plan['source_inventory'].append(dict(relative_path=name,path=str(path),sha256=sha,snapshot_path=str(snapshot)));plan['source_sha256'][name]=sha
    if session:
        plan['settings_path']=session['settings_path'];plan['settings_sha256']=digest(session['settings_path'])
        plan['manual_scene_session_path']=str(Path(session_path).resolve());plan['manual_task']=task
    save_plan(out/'plan.json',plan)
    emit(kind='PLAN_READY',plan=str(out/'plan.json'),scene_session=str(session_path) if session else None)
    return out/'plan.json'

def run_manual(plan_path,run_output,session_path=None):
    from t1_process_plan_v1p1 import validate_plan
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'));validate_plan(plan)
    if run_output.exists():raise FileExistsError('NEW_RUN_DIRECTORY_REQUIRED')
    stop=run_output.parent/'manual_stop.request'
    if stop.exists():raise ValueError('STOP_SIGNAL_ALREADY_PRESENT')
    if session_path:
        from t1_identity_flight_v11p4 import run
        from prepare_t1_identity_flight import digest
        import airsim
        session=json.loads(Path(session_path).read_text(encoding='utf-8'));assert_owner(session)
        if session['status']!='SCENE_READY_GROUND_ONLY' or plan.get('manual_scene_session_path')!=str(Path(session_path).resolve()):raise ValueError('新场景已消费或与计划不匹配。')
        snapshot_manual_run(plan_path,plan,run_output.parent)
        manifest=dict(kind='T1_MANUAL_SCENE_FLIGHT_ATTEMPT',status='RUNNING',session_id=plan['session_id'],mode=plan['mode'],
            plan=str(Path(plan_path).resolve()),plan_sha256=digest(plan_path),runner_started=True,runner_output=str(run_output),
            formal_efficacy_results=False,engine_closed_after_safe_completion=False,fresh_ground_closure_verified=False,
            engine_process_id=session['engine']['pid'],manual_scene_session=str(Path(session_path).resolve()))
        manifest_path=run_output.parent/'manifest.json';save(manifest_path,manifest)
        session['status']='FLIGHT_RUNNING';save(session_path,session)
        client=airsim.MultirotorClient(timeout_value=5)
        client.simPause(False)
        try:passed=run(plan_path,run_output,stop)
        finally:
            report_path=run_output/'report.json';session['status']='FLIGHT_ENDED_SEE_REPORT';session['last_report']=str(report_path);session['last_manifest']=str(manifest_path);save(session_path,session)
            if report_path.is_file():
                report=json.loads(report_path.read_text(encoding='utf-8'));manifest.update(report=str(report_path),report_sha256=digest(report_path),runner_status=report['status'],safe_end=report['safe_end'],status='COMPLETED_ENGINE_RETAINED')
                save(manifest_path,manifest)
        # Engine stays open for operator review. Safe close is a separate action.
        return passed
    script=ROOT/'scripts/run-t1-identity-flight-v11p4.ps1'
    command=['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script),'-Prepared',str(Path(plan_path).parent),'-Output',str(run_output.parent/'attempt'),'-StopFile',str(stop)]
    return subprocess.run(command,cwd=ROOT).returncode==0

def close_scene(session_path):
    import airsim,psutil
    from t1_ground_perception_probe import ground_states
    m=json.loads(Path(session_path).read_text(encoding='utf-8'));assert_owner(m)
    client=airsim.MultirotorClient(timeout_value=5)
    m['close_ground_readback']=ground_states(client)
    if any(client.isApiControlEnabled(vehicle_name=n) for n in ('DroneA','DroneB')):raise ValueError('控制尚未释放，请先停止飞行并等待落地。')
    client.simPause(True)
    closure=Path(m['output'])/('fresh_ground_closure_'+stamp()+'.json');fresh_ground_closure(client,closure)
    path=Path(m['last_manifest']).resolve() if m.get('last_manifest') else None
    if path and not path.is_relative_to(ROOT/'logs'):raise ValueError('本工具日志目录内的清单路径才可更新。')
    assert_owner(m);owned=psutil.Process(m['engine']['pid']);owned.terminate();owned.wait(timeout=10)
    if path:
        from prepare_t1_identity_flight import digest
        manifest=json.loads(path.read_text(encoding='utf-8'));manifest.update(engine_closed_after_safe_completion=True,
            fresh_ground_closure_verified=True,fresh_ground_closure_path=str(closure),fresh_ground_closure_sha256=digest(closure))
        manifest['status']='PASS' if manifest.get('runner_status')=='PASS' else 'FAIL';save(path,manifest)
    m['status']='CLOSED_AFTER_GROUND_READBACK';save(session_path,m);emit(kind='SCENE_CLOSED',path=str(session_path))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preview','scene-start','scene-close','prepare','run'])
    p.add_argument('--scene',type=Path);p.add_argument('--session',type=Path);p.add_argument('--task',type=Path);p.add_argument('--condition',choices=['zero','scale50','scale100'],default='zero')
    p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    try:
        if a.action=='scene-start':scene_start(a.scene or DEFAULT_SCENE)
        elif a.action=='scene-close':close_scene(a.session)
        elif a.action=='prepare':prepare_manual(a.condition,a.scene,a.session,a.task)
        elif a.action=='run':return 0 if run_manual(a.plan,a.output,a.session) else 2
        elif a.action=='preview':emit(kind='ROUTE_PREVIEW',route=route(load_scene(a.scene or DEFAULT_SCENE)))
        return 0
    except Exception as e:
        emit(kind='MANUAL_ACTION_FAILED',error_type=type(e).__name__,message=str(e) if isinstance(e,ValueError) else '操作失败，原始运行记录已保留。');return 2
if __name__=='__main__':raise SystemExit(main())
