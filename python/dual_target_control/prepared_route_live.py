"""Protocol15: real arrival -> model planning while holding -> reference clock."""
from copy import deepcopy
from datetime import datetime
import hashlib,json,math,os,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .runtime import ControlRuntime,wrap_angle
from .route_live import RouteLiveRuntime
from .curve_live import LiveCurveRuntime
from .simulator import ROOT
from run_airsim_curve_replay_pilot import limit_command
from prepared_route_planner import prepared_spec
from .route_milestones import RouteMilestones


class PreparedRouteRuntime(RouteLiveRuntime):
    protocol_revision=15
    additional_source_paths=RouteLiveRuntime.additional_source_paths+[
        Path(__file__),ROOT/'python/prepared_route_planner.py',ROOT/'python/route_control_experiments.py',
        ROOT/'python/trajectory_research/screened_search.py',ROOT/'python/trajectory_research/search.py',
        ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py',Path(__file__).with_name('route_milestones.py')]

    def __init__(self,config,mode='airsim'):
        self.preparation_events=[];self.preparation_rows=[];self._hold_previous=np.zeros(3)
        self.display_milestones=RouteMilestones(prepared_spec(**config['preparation']['task']))
        super().__init__(config,mode)

    def _task(self):return prepared_spec(**self.config['preparation']['task'])
    def _curve(self):return self.live_curve or SimpleNamespace(spec=self._task())

    def _preview(self):
        self.route_index=0;self.live_curve=None;self.preview_curve=None
        self.display_milestones=RouteMilestones(self._task())
        if self.mode=='airsim' and self._client is not None:
            state=self._client.getMultirotorState(vehicle_name=self.vehicle_name)
            local=state.kinematics_estimated.position;world=self._client.simGetObjectPose(self.vehicle_name).position
            self.marker_origin=np.array([world.x_val-local.x_val,world.y_val-local.y_val,world.z_val-local.z_val])
            self._markers(True)
        self._publish(planned_path=[],trail=[],targets=self._target_queue(),preparation_point=self._task()['start'],route_markers=self.display_milestones.markers,
            marker_world_origin_ned=self.marker_origin.tolist(),route_policy=self._task()['policy_name'],
            message='任务已设置；点击开始后先到准备点，再规划所选方法。')

    def _event(self,stage,**values):
        self.preparation_events.append(dict(stage=stage,host_monotonic_s=time.perf_counter(),utc_ns=time.time_ns(),**values))
        (self._run_dir/'preparation.json').write_text(json.dumps(dict(events=self.preparation_events,
            samples=self.preparation_rows,scope='Preparation/hover excluded from reference tracking time; planning wall time reported separately'),ensure_ascii=False,indent=2),encoding='utf-8')

    def _begin_run(self):
        ControlRuntime._begin_run(self)
        self.live_states=[];self.live_commands=[];self.live_metrics=None;self.live_planning=None;self.live_curve=None
        self.live_marker_receipts=[];self.preparation_events=[];self.preparation_rows=[];self._hold_previous=np.zeros(3)
        self.display_milestones=RouteMilestones(self._task())
        self._publish(planned_path=[],trail=[],control_metrics=None,result_ready=False,route_markers=self.display_milestones.markers)
        self._event('RUN_REQUESTED',task=self._task(),method=self.config['preparation']['method'])
        # Even failures before a trajectory exists remain identifiable as trials.
        sources=[Path(__file__).with_name(n) for n in ('curve_live.py','runtime.py','simulator.py','native_scene.py','overview.py')]
        sources+=self.additional_source_paths+[ROOT/'python/trajectory_research/curves.py',ROOT/'python/trajectory_research/parameterizations.py',ROOT/'python/run_airsim_curve_replay_pilot.py']
        self.preparation_source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        snapshot=self._run_dir/'curve_source';snapshot.mkdir()
        for p in sources:(snapshot/p.name).write_bytes(p.read_bytes())
        conditions=deepcopy(self.config);section=conditions['curve_experiment']
        for key in ('plan','plans_file','plans_sha256','execution_id','batch_id'):section.pop(key,None)
        conditions['preparation'].pop('method',None);conditions['protocol_revision']=15
        manifest=dict(protocol_revision=15,method=self.config['preparation']['method'],mode=self.mode,
            plan=self.config['curve_experiment']['plan'],conditions=conditions,source_sha256=self.preparation_source_hashes,
            effective_duration_s=None,planning_pending=True,scope='Requested task only; no computed reference or flight result yet')
        manifest['condition_fingerprint']=hashlib.sha256(json.dumps(dict(conditions=conditions,sources=self.preparation_source_hashes),sort_keys=True).encode()).hexdigest()
        (self._run_dir/'live_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        if self.mode=='demo':raise RuntimeError('协议15要求真实状态准备；离线验证请使用接口测试，不生成伪AirSim记录')

    def _validate_reference_start(self,p0):
        if not np.allclose(p0,self._task()['start'],atol=1e-9):raise ValueError('规划起点与准备点不一致')

    def _hold_tick(self,goal,stage,previous):
        if self._stop_requested.is_set():raise RuntimeError('用户取消准备/规划；本次保留为中止记录')
        tick=time.perf_counter();p,v,yaw,_=self._read_state();dt=max(.005,min(.1,tick-previous))
        self._check_initial_position(p)
        raw=1.5*(goal-p)-.8*v
        limited=limit_command(raw,self._hold_previous,dt,2.,self.cfg.max_acceleration)
        self._hold_previous=np.array(limited['limited']);self._send(self._hold_previous,0.)
        errors=[float(np.linalg.norm(goal-p)),float(np.linalg.norm(v)),abs(wrap_angle(yaw))]
        self.preparation_rows.append(dict(host_monotonic_s=tick,stage=stage,position=p.tolist(),velocity=v.tolist(),
            yaw_rad=yaw,errors=errors,command=self._hold_previous.tolist()))
        self._publish(position=p.tolist(),velocity=v.tolist(),yaw=yaw,preparation_point=goal.tolist(),
            preparation_error_m=errors[0],planned_path=[])
        if stage=='PLANNING' and errors[0]>.5:raise RuntimeError('悬停规划期间偏离准备点超过0.5米')
        time.sleep(max(0.,1/self.cfg.control_hz-(time.perf_counter()-tick)))
        return tick,errors

    def _settle(self,goal,stage,timeout):
        self._publish(stage,'到准备点并收敛位置、速度与朝向。',running=True)
        begin=previous=time.perf_counter();settled=0
        while time.perf_counter()-begin<timeout:
            previous,errors=self._hold_tick(goal,stage,previous)
            settled=settled+1 if errors[0]<=.02 and errors[1]<=.03 and errors[2]<=.03 else 0
            if settled>=5:
                self._event(stage+'_ARRIVED',position=self.snapshot()['position'],velocity=self.snapshot()['velocity'],
                    tolerances=dict(position_m=.02,velocity_mps=.03,yaw_rad=.03));return
        raise TimeoutError('未在规定时间内收敛到准备点')

    def _rise(self,height):
        ControlRuntime._rise(self,height)  # unchanged original03 real takeoff
        goal=np.array(self._task()['start'])
        self._settle(goal,'MOVING_TO_START',max(30.,float(np.linalg.norm(goal))/1.2+20.))
        output=ROOT/'reports'/f'prepared_route_plan_{datetime.now():%Y%m%d_%H%M%S_%f}'
        request=dict(task=deepcopy(self.config['preparation']['task']),method=self.config['preparation']['method'])
        request_path=self._run_dir/'planning_request.json';request_path.write_text(json.dumps(request,ensure_ascii=False,indent=2),encoding='utf-8')
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONPATH=str(ROOT/'python'))
        self._publish('PLANNING','已抵达准备点，悬停并规划所选方法；参考时钟尚未启动。',running=True)
        begin=time.perf_counter();self._event('PLANNING_STARTED',output=str(output),request=request)
        worker=None
        try:
            with (self._run_dir/'planner.log').open('w',encoding='utf-8') as log:
                worker=subprocess.Popen([str(ROOT/'.venv/python.exe'),str(ROOT/'python/prepared_route_planner.py'),
                    '--request',str(request_path),'--output',str(output)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                previous=time.perf_counter()
                while worker.poll() is None:
                    previous,_=self._hold_tick(goal,'PLANNING',previous)
                    if time.perf_counter()-begin>180:raise TimeoutError('所选方法规划超过180秒，记录超时并降落')
                if worker.returncode:raise RuntimeError('规划失败，详见本次数据 planner.log')
            wall=time.perf_counter()-begin
            path=output/'airsim_replay_plan.json';document=json.loads(path.read_text(encoding='utf-8'))
            plan=document['plans'][0]
            if len(document['plans'])!=1 or plan['group_id']!=request['method'] or plan['route']!=self._task():raise ValueError('规划回执与本次任务不一致')
            self.config['curve_experiment'].update(plan=plan,plans_file=str(path),plans_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self._event('PLANNING_FINISHED',planning_wall_s=wall,method=plan['group_id'],airsim_search_calls=0)
            self._settle(goal,'SETTLING',10.)
            if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in self.preparation_source_hashes.items()):raise RuntimeError('准备/规划期间源码变化，请保留本次失败并重新运行')
            self._initialize_reference()
            self.live_planning.update(parent_total_planning_wall_s=wall,scope='Fresh selected-method model search after real arrival; no AirSim objective calls')
            manifest_path=self._run_dir/'live_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            section=manifest['conditions']['curve_experiment']
            for key in ('plans_file','plans_sha256'):section.pop(key,None)
            # Selected method and output paths are outcomes, not fixed task conditions.
            manifest['conditions']['preparation'].pop('method',None)
            manifest['conditions']['protocol_revision']=15
            manifest['preparation_events']=self.preparation_events
            manifest['condition_fingerprint']=hashlib.sha256(json.dumps(dict(conditions=manifest['conditions'],sources=manifest['source_sha256']),sort_keys=True).encode()).hexdigest()
            manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            (self._run_dir/'config.json').write_text(json.dumps(self.config,ensure_ascii=False,indent=2),encoding='utf-8')
            self._event('REFERENCE_READY',duration_s=self.live_curve.duration_s)
        finally:
            if worker is not None and worker.poll() is None:
                worker.terminate()
                try:worker.wait(timeout=3)
                except subprocess.TimeoutExpired:worker.kill();worker.wait(timeout=3)
                self._event('PLANNING_CANCELLED',returncode=worker.returncode)

    def _start(self):
        super()._start()
        if self._active:
            self._event('REFERENCE_STARTED')
            self._publish(planning_ms=self.live_planning['parent_total_planning_wall_s']*1000,
                message='准备点到达后规划完成；正在执行绿色参考，亮紫线为实测轨迹。')

    def _save(self,outcome):
        super()._save(outcome)
        if self._run_dir and self._summary_written:
            path=self._run_dir/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
            summary.update(preparation_events=self.preparation_events,planning_after_arrival=any(e['stage']=='PLANNING_STARTED' for e in self.preparation_events),
                scope='Protocol15 measured preparation -> fresh model planning -> real-time execution; no AirSim search')
            path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            (self._run_dir/'route_display.json').write_text(json.dumps(dict(markers=self.display_milestones.markers,
                events=self.display_milestones.events,scope='Display landmarks only. Control judgments remain in summary/trace.'),ensure_ascii=False,indent=2),encoding='utf-8')

    def _advance_targets(self,t,p,v):
        super()._advance_targets(t,p,v)
        markers=self.display_milestones.update(t,p,self.route_index,self.live_curve.waypoint_times,self.live_marker_hidden)
        self._publish(route_markers=markers)

    def _markers(self,visible):
        if not self._active:return super()._markers(visible)
        if self.mode!='airsim':return
        points=self._target_queue() if visible else []
        key=(tuple(tuple(p) for p in points),tuple(self.marker_origin))
        if key==self.marker_cache:return
        if self._scene is None:
            from .native_scene import NativeScene
            self._scene=NativeScene(self.config,self)
        self._scene.sync_targets(dict(points=points,origin=self.marker_origin.tolist(),route_index=self.route_index,
            reference_time_s=self.live_elapsed,run_dir=str(self._run_dir)))
        self.marker_cache=key
        self._publish(marker_status='显示线程正在更新红/蓝目标；控制不等待显示RPC。',route_index=self.route_index,
            route_total=len(self._curve().spec['waypoints']))

    def _accept_target_receipts(self,payload,receipts):
        if payload['run_dir']!=str(self._run_dir):return
        self.live_marker_receipts.extend(receipts)
        self._publish(marker_status='红/蓝目标已由独立观察线程更新并读回。')

    def _handle(self,command,value):
        # Reuse the observer on the same connection. A task edit must not hide,
        # recreate and re-centre the camera before every flight.
        keys=('host','port','vehicle','project','map')
        same=command=='configure' and all(self.config.get('airsim',{}).get(k)==value.get('airsim',{}).get(k) for k in keys)
        if same and self._scene is not None and not self._active and not self._flight_owned:
            scene=self._scene;self._scene=None
            self._publish('CONFIGURING','正在应用任务，保留当前UE视角。',execution_id=value.get('curve_experiment',{}).get('execution_id'))
            try:ControlRuntime._handle(self,command,value)
            finally:self._scene=scene
            scene.update_config(self.config);self._preview()
            if self.snapshot().get('connected') and self.config['curve_experiment'].get('auto_show_scene'):self.show_scene()
            return
        super()._handle(command,value)
