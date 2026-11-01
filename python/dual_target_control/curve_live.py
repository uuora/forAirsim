"""Protocol13: frozen curve references on the existing 03 real-time runtime.

Reuses the production simulator launcher, real takeoff/landing, dynamics receipt,
native observer and camera pointing. It never pauses or steps AirSim physics.
This is a new protocol; its results must not be pooled with stepwise replay.
"""
from copy import deepcopy
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from .runtime import ControlRuntime,from_ned,json_value,wrap_angle
from .simulator import ROOT
from run_airsim_curve_replay_pilot import plan_curve,limit_command,metrics_from_trace


class LiveCurveRuntime(ControlRuntime):
    protocol_revision=13

    def _decode_reference(self,plan,factor):
        from trajectory_research.curves import PolynomialSegment
        original=plan_curve(plan)
        return PolynomialSegment(original.coefficients.copy(),original.duration_s/factor)

    def _target_queue(self):return [self.live_curve.sample(self.live_curve.duration_s).tolist()]
    def _target_point(self):return self.live_curve.sample(self.live_curve.duration_s)
    def _completion_hits(self):return 1

    def _advance_targets(self,t,p,v):
        if np.linalg.norm(self._target_point()-p)<=self.cfg.arrival_radius and not self.live_marker_hidden:
            self._publish(targets=[],hits=1);self._markers(False);self.live_marker_hidden=True

    def _compute_metrics(self,config):
        return metrics_from_trace(self.live_curve,self.live_states,self.live_commands[:-1],config)

    def __init__(self,config,mode='airsim'):
        self.live_states=[];self.live_commands=[];self.live_metrics=None
        self.live_curve=None;self.live_elapsed=0.;self.live_previous_stamp=None
        self.live_previous_command=np.zeros(3);self.live_yaw=0.;self.live_last_collision=0.
        self.live_stream=None;self.live_marker_receipts=[];self.live_planning=None
        self.live_marker_hidden=False
        super().__init__(config,mode)

    def _begin_run(self):
        super()._begin_run()
        self._initialize_reference()

    def _validate_reference_start(self,p0):
        centre=np.array([(self.cfg.x_min+self.cfg.x_max)/2,(self.cfg.y_min+self.cfg.y_max)/2])
        if not np.allclose(p0[:2],centre,atol=1e-6):raise ValueError('当前方案起点须在03场地水平中心')
        if abs(p0[2]-self.config['airsim']['initial_height'])>1e-6:raise ValueError('起飞高度与参考起点不一致')

    def _initialize_reference(self):
        self.live_states=[];self.live_commands=[];self.live_metrics=None
        self.live_marker_receipts=[];self.live_planning=None
        section=deepcopy(self.config['curve_experiment'])
        plan=section['plan'];factor=float(section.get('time_scale',1.))
        if not .5<=factor<=2.:raise ValueError('参考时间倍率必须在0.5至2之间')
        effective=deepcopy(plan);effective['parameters'][-1]/=factor
        # The canonical decoder bounds T. Time scaling is applied to the
        # resulting physical polynomial rather than silently changing its box.
        self.live_curve=self._decode_reference(plan,factor)
        p0=self.live_curve.sample(0.)
        self._validate_reference_start(p0)
        sources=[Path(__file__),Path(__file__).with_name('runtime.py'),Path(__file__).with_name('simulator.py'),
                 Path(__file__).with_name('native_scene.py'),Path(__file__).with_name('overview.py'),
                 ROOT/'python/trajectory_research/curves.py',ROOT/'python/trajectory_research/parameterizations.py',
                 ROOT/'python/run_airsim_curve_replay_pilot.py']
        sources+=getattr(self,'additional_source_paths',[])
        snap=self._run_dir/'curve_source';snap.mkdir(exist_ok=True)
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for p in sources:(snap/p.name).write_bytes(p.read_bytes())
        conditions=deepcopy(self.config);conditions['curve_experiment'].pop('plan',None)
        for key in ('execution_id','batch_id'):conditions['curve_experiment'].pop(key,None)
        if self.mode=='airsim':
            conditions['loaded_settings']=json.loads(self._client.getSettingsString())
        project=ROOT/self.config['airsim']['project']
        files=[project.parent/'Content'/(self.config['airsim']['map'].removeprefix('/Game/')+'.umap'),
               project.parent/'Binaries/Win64/UE4Editor-Blocks.dll',
               project.parent/'Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll']
        conditions['runtime_file_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
        manifest=dict(protocol_revision=self.protocol_revision,scope='03 real-time AirSim execution; model search is separate; no image input',
            method=plan['group_id'],plan=plan,effective_duration_s=self.live_curve.duration_s,
            conditions=conditions,source_sha256=hashes,mode=self.mode)
        manifest['condition_fingerprint']=hashlib.sha256(json.dumps(dict(conditions=conditions,sources=hashes),sort_keys=True).encode()).hexdigest()
        (self._run_dir/'live_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        relative=plan.get('selection_provenance',{}).get('result_json')
        if relative:
            package=Path(section['plans_file']).parent.resolve();result=(package/relative).resolve()
            if package not in result.parents:raise ValueError('方案证据路径越界')
            if result.is_file():
                record=json.loads(result.read_text(encoding='utf-8'))
                if record.get('parameters')!=plan['parameters']:raise ValueError('冻结参数与规划记录不一致')
                for key in ('segments','route','boundary','family','representation'):
                    if key in plan and record.get(key)!=plan[key]:raise ValueError('冻结完整方案与规划记录不一致：'+key)
                self.live_planning=dict(counts=record.get('counts'),timings=record.get('timings'),
                    source_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                    scope='Historical model planning; this live flight performs no AirSim search')
        self.live_stream=(self._run_dir/'live_states.jsonl').open('w',encoding='utf-8',buffering=1)

    def _start(self):
        if self._active or self._flight_owned:return
        super()._start()
        if self.mode=='demo':
            self._demo.position=self.live_curve.sample(0.).copy();self._demo.velocity=self.live_curve.sample(0.,derivative=1).copy()
            self._publish(position=self._demo.position.tolist(),velocity=self._demo.velocity.tolist())
        self.live_elapsed=0.;self.live_previous_stamp=None;self.live_previous_command=np.zeros(3)
        self.live_marker_hidden=False
        self.live_yaw=float(self.snapshot()['yaw']);self.live_last_collision=0.
        initial=dict(actual_position=self.snapshot()['position'],actual_velocity=self.snapshot()['velocity'],
            actual_yaw_rad=self.live_yaw,reference_position=self.live_curve.sample(0.).tolist(),
            scope='Original03 real takeoff; measured initial offset retained, no simulation state injection')
        (self._run_dir/'curve_initial_state.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
        points=self.live_curve.sample(np.linspace(0,self.live_curve.duration_s,180)).tolist()
        self._publish(targets=self._target_queue(),planned_path=points,
            hits=0,trajectory_name=self.config['curve_experiment']['plan']['group_id'],reference_elapsed_s=0.,
            planning_ms=(self.live_planning['timings']['search_plus_selected_validation_wall_s']*1000 if self.live_planning and self.live_planning.get('timings') else None),
            message='实时曲线控制；可查看UE4并使用原03鼠标跟随。到达后隐藏目标并保存数据。')
        if self.mode=='airsim' and self.config['curve_experiment'].get('auto_show_scene',True):self.show_scene()

    def _rise(self,height):
        super()._rise(height)
        goal=np.array([(self.cfg.x_min+self.cfg.x_max)/2,(self.cfg.y_min+self.cfg.y_max)/2,height])
        rows=[];begin=previous=time.perf_counter();command=np.zeros(3);settled=0
        try:
            while time.perf_counter()-begin<10:
                if self._stop_requested.is_set():raise RuntimeError('起始状态准备已取消')
                now=time.perf_counter();p,v,yaw,_=self._read_state();dt=max(.005,min(.1,now-previous))
                raw=1.5*(goal-p)-.8*v
                limited=limit_command(raw,command,dt,2.,self.cfg.max_acceleration)
                command=np.array(limited['limited']);self._send(command,0.)
                errors=[float(np.linalg.norm(goal-p)),float(np.linalg.norm(v)),abs(wrap_angle(yaw))]
                rows.append(dict(t=now-begin,position=p.tolist(),velocity=v.tolist(),yaw_rad=yaw,errors=errors))
                settled=settled+1 if errors[0]<=.02 and errors[1]<=.03 and errors[2]<=.03 else 0
                if settled>=5:return
                previous=now;time.sleep(max(0.,1/self.cfg.control_hz-(time.perf_counter()-now)))
            raise TimeoutError('真实起始状态未收敛到共同位置/速度/朝向容差')
        finally:
            (self._run_dir/'curve_start_settle.json').write_text(json.dumps(dict(samples=rows,
                tolerances=dict(position_m=.02,velocity_mps=.03,yaw_rad=.03),rule='same real closed-loop preparation; no state injection')),encoding='utf-8')

    def _resume(self):
        if self._active and self._paused:
            self._paused=False;self._publish('RUNNING','继续实时参考，暂停时间不计入参考时钟。')

    def _markers(self,visible):
        if self.mode!='airsim':return
        point=self.live_curve.sample(self.live_curve.duration_s)
        for name,target in [('DC_TargetRed',point if visible else None),('DC_TargetBlue',None)]:
            xyz=[0.,0.,1000.] if target is None else [float(target[0]),float(target[1]),-float(target[2])]
            if not self._client.simSetObjectPose(name,self._airsim.Pose(self._airsim.Vector3r(*xyz),self._airsim.Quaternionr()),teleport=True):
                raise RuntimeError('03目标实体更新失败：'+name)
            observed=self._client.simGetObjectPose(name).position
            actual=[observed.x_val,observed.y_val,observed.z_val]
            if not np.allclose(actual,xyz,atol=.002):raise RuntimeError('03目标实体位置读回不一致：'+name)
            self.live_marker_receipts.append(dict(name=name,visible=bool(target is not None),requested_ned=xyz,observed_ned=actual))

    def _tick(self,dt):
        begin=time.perf_counter();p,v,yaw,rpc=self._read_state()
        if np.any(p<self.cfg.lower-.1) or np.any(p>self.cfg.upper+.1):raise RuntimeError('实际状态超出03场地范围')
        stamp=(int(self._last_state.timestamp) if self.mode=='airsim' else int((self._start_time+sum(s.get('step_s',0.) for s in self.live_states)+dt)*1e9))
        observed=0. if self.live_previous_stamp is None else (stamp-self.live_previous_stamp)*1e-9
        if observed<0 or observed>.1:raise RuntimeError('实时状态间隔异常：'+str(observed))
        self.live_previous_stamp=stamp
        if not self._paused:self.live_elapsed+=observed
        t=self.live_elapsed;T=self.live_curve.duration_s;u=min(t,T)
        q,qv,qa=[self.live_curve.sample(u,derivative=d) for d in (0,1,2)]
        if t>T:qv=np.zeros(3);qa=np.zeros(3)
        command_raw=qv+2*(q-p)+.8*(qv-v)+.9*qa
        if self._paused:command_raw=np.zeros(3)
        command=limit_command(command_raw,self.live_previous_command,max(observed,.001),self.cfg.max_speed,self.cfg.max_acceleration)
        direction=qv if np.linalg.norm(qv[:2])>.2 else self.live_curve.sample(T)-p
        desired=math.atan2(direction[1],direction[0]) if np.linalg.norm(direction[:2])>.05 else self.live_yaw
        self.live_yaw=wrap_angle(self.live_yaw+np.clip(wrap_angle(desired-self.live_yaw),
            -math.radians(self.cfg.max_yaw_rate_deg)*max(observed,.001),math.radians(self.cfg.max_yaw_rate_deg)*max(observed,.001)))
        self._send(command['limited'],math.degrees(self.live_yaw));self.live_previous_command=np.array(command['limited'])
        if self.mode=='demo':self._demo.step(command['limited'],math.degrees(self.live_yaw),min(dt,.1))
        if not self.live_states:self._markers(True)
        if self.mode=='airsim':
            kin=self._last_state.kinematics_estimated
            acceleration=from_ned(kin.linear_acceleration).tolist()
            rate=[kin.angular_velocity.x_val,kin.angular_velocity.y_val,kin.angular_velocity.z_val]
            if begin-self.live_last_collision>.1:
                collision=self._client.simGetCollisionInfo(vehicle_name=self.vehicle_name);self.live_last_collision=begin
                if self.live_states and collision.has_collided and int(collision.time_stamp)>=self.live_states[0]['sim_timestamp_ns']:
                    raise RuntimeError('参考执行期间观测到新的碰撞')
        else:
            acceleration=((v-np.array(self.live_states[-1]['velocity_zup_mps']))/observed).tolist() if self.live_states and observed>0 else [0.,0.,0.]
            rate=[0.,0.,wrap_angle(yaw-self.live_states[-1]['yaw_rad'])/observed] if self.live_states and observed>0 else [0.,0.,0.]
        state=dict(sim_timestamp_ns=stamp,reference_elapsed_s=t,position_zup_m=p.tolist(),velocity_zup_mps=v.tolist(),
            reference_position_zup_m=q.tolist(),reference_velocity_zup_mps=qv.tolist(),acceleration_zup_mps2=acceleration,
            angular_velocity_body=rate,yaw_rad=float(yaw),step_s=dt,paused=self._paused)
        if not self.live_states:self.live_started_host_s=begin
        state['host_elapsed_s']=begin-self.live_started_host_s
        self.live_states.append(state);self.live_commands.append(dict(command,reference_elapsed_s=t,yaw_command_deg=math.degrees(self.live_yaw)))
        if self.live_stream:self.live_stream.write(json.dumps(dict(state=state,command=self.live_commands[-1]))+'\n')
        self._trail.append(p.tolist());distance=float(np.linalg.norm(self._target_point()-p))
        self._publish(position=p.tolist(),velocity=v.tolist(),yaw=float(yaw),trail=list(self._trail),
            distance=distance,rpc_ms=rpc,loop_ms=(time.perf_counter()-begin)*1000,elapsed=begin-self._start_time,
            reference_elapsed_s=t,yaw_error_deg=math.degrees(wrap_angle(desired-yaw)))
        self._aim_camera(p,self._target_point(),begin)
        self._advance_targets(t,p,v)
        if t>=T and np.linalg.norm(self.live_curve.sample(T)-p)<=.25 and np.linalg.norm(v)<=.5:
            self._publish(targets=[],hits=self._completion_hits())
            if not self.live_marker_hidden:self._markers(False);self.live_marker_hidden=True
            self._active=False;self._stop('COMPLETED')
        elif t>T+10:raise TimeoutError('参考结束后10秒仍未达到终端位置/速度要求')

    def _save(self,outcome):
        if not self._run_dir or self._summary_written:return
        if self.live_stream:self.live_stream.close();self.live_stream=None
        trace=dict(states=self.live_states,commands=self.live_commands)
        (self._run_dir/'trace.json').write_text(json.dumps(json_value(trace),ensure_ascii=False),encoding='utf-8')
        config=dict(protocol_revision=13,execution=dict(terminal_interpolation_max_gap_s=.1),
            control_error_gates=dict(sampled_peak_position_m=.5,terminal_position_m=.25,terminal_velocity_mps=.5))
        error=None
        try:
            # Paused records are retained; a paused trial has a different
            # intervention and is never silently used as an uninterrupted run.
            if any(s['paused'] for s in self.live_states):raise ValueError('用户暂停过，保留数据但不计算连续参考排名')
            self.live_metrics=self._compute_metrics(config)
            self.live_metrics['task_success']=bool(outcome=='COMPLETED' and self.live_metrics['control_error_gates_passed'])
        except Exception as exc:error=str(exc)
        summary=dict(protocol_revision=self.protocol_revision,mode=self.mode,outcome=outcome,error=self._failure,metrics=self.live_metrics,
            metric_error=error,method=self.config['curve_experiment']['plan']['group_id'],dynamics=self._dynamics,
            samples=len(self.live_states),config=self.config,planning_evidence=self.live_planning,
            marker_receipts=self.live_marker_receipts,
            airsim_reference_execution_calls=int(self.mode=='airsim' and bool(self.live_states)),airsim_search_calls=0,
            tracking_host_wall_s=(self.live_states[-1].get('host_elapsed_s') if self.live_states else None),
            scene_view={k:self.snapshot().get(k) for k in ('scene_status','scene_visible','scene_receipts','scene_message')},
            scope='Real-time velocity/yaw control with original03 dynamics and lifecycle; not comparable to protocol12')
        (self._run_dir/'summary.json').write_text(json.dumps(json_value(summary),ensure_ascii=False,indent=2),encoding='utf-8')
        import csv
        with (self._run_dir/'states.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            w=csv.writer(stream);w.writerow(['t_s','px','py','pz','vx','vy','vz','qx','qy','qz','qvx','qvy','qvz','ax','ay','az','cmd_x','cmd_y','cmd_z','yaw_rad','yaw_command_deg','speed_clipped','slew_clipped','paused'])
            for s,c in zip(self.live_states,self.live_commands):
                w.writerow([s['reference_elapsed_s'],*s['position_zup_m'],*s['velocity_zup_mps'],*s['reference_position_zup_m'],
                    *s.get('reference_velocity_zup_mps',[None]*3),*s['acceleration_zup_mps2'],*c['limited'],s['yaw_rad'],c['yaw_command_deg'],c['speed_clipped'],c['slew_clipped'],s['paused']])
        manifest=json.loads((self._run_dir/'live_manifest.json').read_text(encoding='utf-8')) if (self._run_dir/'live_manifest.json').is_file() else {}
        summary['sources_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in manifest.get('source_sha256',{}).items())
        (self._run_dir/'summary.json').write_text(json.dumps(json_value(summary),ensure_ascii=False,indent=2),encoding='utf-8')
        self._summary_written=True
        self._publish(result_ready=True,run_dir=str(self._run_dir),control_metrics=self.live_metrics)
