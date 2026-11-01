"""Protocol16 continuous flight with isolated, deadline-limited model planning.

Reuses original03 lifecycle/map/observer; does not use the known-invalid old
RollingController offline ranking. Search has zero AirSim objective calls.
"""
from copy import deepcopy
import hashlib,json,math,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .runtime import ControlRuntime,wrap_angle,from_ned
from .prepared_route_live import PreparedRouteRuntime
from .online_curve import make_request,fallback,decode,reference_checks,ReferenceTimeline,stream_hash
from .online_worker_client import PlannerClient
from .simulator import ROOT
from run_airsim_curve_replay_pilot import limit_command


class OnlineRuntime(ControlRuntime):
    protocol_revision=16
    _hold_tick=PreparedRouteRuntime._hold_tick
    _settle=PreparedRouteRuntime._settle
    _accept_target_receipts=PreparedRouteRuntime._accept_target_receipts

    def __init__(self,config,mode='airsim'):
        self.planner=None;self.timeline=None;self.online_events=[];self.rows=[];self.pending=None
        self.route_index=0;self.marker_origin=np.zeros(3);self.marker_cache=None;self.live_marker_receipts=[];self.live_elapsed=0.
        self.preparation_rows=[];self.preparation_events=[];self._hold_previous=np.zeros(3)
        super().__init__(config,mode)
        self._preview()

    def _task(self):return self.config['preparation']['task']
    def _curve(self):return SimpleNamespace(spec=dict(waypoints=self.config['online']['targets']))
    def _target_queue(self):return self.config['online']['targets'][self.route_index:self.route_index+2]

    def _event(self,stage,**values):
        row=dict(stage=stage,host_monotonic_s=time.perf_counter(),reference_time_s=self.live_elapsed,**values)
        self.online_events.append(row)
        if getattr(self,'event_stream',None):self.event_stream.write(json.dumps(row,ensure_ascii=False)+'\n')

    def _markers(self,visible):
        if self.mode!='airsim' or self._client is None:return
        from .native_scene import NativeScene
        if self._scene is None:self._scene=NativeScene(self.config,self)
        points=self._target_queue() if visible else []
        key=(tuple(tuple(p) for p in points),tuple(self.marker_origin))
        if key==self.marker_cache:return
        self._scene.sync_targets(dict(points=points,origin=self.marker_origin.tolist(),route_index=self.route_index,
            reference_time_s=self.live_elapsed,run_dir=str(self._run_dir)))
        self.marker_cache=key

    def _landmarks(self):
        start=dict(id='S',kind='start',position=self._task()['start'],state='fixed')
        visited=[dict(id=f'W{i+1}',kind='via',position=p,state='visited') for i,p in enumerate(self.config['online']['targets'][:self.route_index])]
        return [start,*visited,*[dict(id=f'W{self.route_index+i+1}',kind='via',position=p,state='pending') for i,p in enumerate(self._target_queue())]]

    def _preview(self):
        self.route_index=0;self.marker_cache=None
        if self.mode=='airsim' and self._client is not None:
            state=self._client.getMultirotorState(vehicle_name=self.vehicle_name);local=state.kinematics_estimated.position
            world=self._client.simGetObjectPose(self.vehicle_name).position
            self.marker_origin=np.array([world.x_val-local.x_val,world.y_val-local.y_val,world.z_val-local.z_val]);self._markers(True)
        self._publish(planned_path=[],trail=[],targets=self._target_queue(),route_markers=self._landmarks(),marker_world_origin_ned=self.marker_origin.tolist(),
            message='协议16：仅预览当前/下一点；点击开始后连续飞行，后台搜索。')

    def _handle(self,command,value):
        # Same host/map keeps the native observer and manually selected camera.
        same=command=='configure' and all(self.config['airsim'].get(k)==value['airsim'].get(k) for k in ('host','port','vehicle','project','map'))
        if same and self._scene is not None and not self._active and not self._flight_owned:
            scene=self._scene;self._scene=None
            try:ControlRuntime._handle(self,command,value)
            finally:self._scene=scene
            scene.update_config(self.config)
        else:ControlRuntime._handle(self,command,value)
        if command in ('configure','connect','simulator') and not self._active:
            self._preview()
            if self.snapshot().get('connected') and self.config['curve_experiment'].get('auto_show_scene'):self.show_scene()

    def _begin_run(self):
        ControlRuntime._begin_run(self)
        self.route_index=0;self.live_elapsed=0.;self.rows=[];self.online_events=[];self.timeline=None;self.pending=None
        self.next_request_s=0.;self.live_marker_receipts=[];self.preparation_rows=[];self.preparation_events=[];self._hold_previous=np.zeros(3)
        self.previous_stamp=None;self.previous_command=np.zeros(3);self.previous_p=None;self.last_collision_host=0.;self.max_tracking_error=0.
        self.row_stream=(self._run_dir/'online_states.jsonl').open('w',encoding='utf-8',buffering=1)
        self.event_stream=(self._run_dir/'online_events.jsonl').open('w',encoding='utf-8',buffering=1)
        paths=[Path(__file__),Path(__file__).with_name('online_curve.py'),Path(__file__).with_name('online_worker_client.py'),
            Path(__file__).with_name('runtime.py'),Path(__file__).with_name('prepared_route_live.py'),Path(__file__).with_name('native_scene.py'),
            ROOT/'python/online_curve_worker.py',ROOT/'python/trajectory_research/parameterizations.py',ROOT/'python/trajectory_research/curves.py',
            ROOT/'python/trajectory_research/curve_task.py',ROOT/'python/trajectory_research/search.py',ROOT/'python/dual_target_control/route_curve.py',ROOT/'python/run_airsim_curve_replay_pilot.py',ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py']
        paths+=getattr(self,'research_source_paths',[])
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths};folder=self._run_dir/'source_snapshot';folder.mkdir()
        for p in paths:(folder/p.name).write_bytes(p.read_bytes())
        conditions=deepcopy(self.config);method=conditions['online'].pop('method');conditions.pop('curve_experiment',None)
        if self.mode=='airsim':conditions['loaded_settings']=json.loads(self._client.getSettingsString())
        project=ROOT/self.config['airsim']['project']
        binaries=[project.parent/'Content'/(self.config['airsim']['map'].removeprefix('/Game/')+'.umap'),project.parent/'Binaries/Win64/UE4Editor-Blocks.dll',project.parent/'Plugins/AirSim/Binaries/Win64/UE4Editor-AirSim.dll']
        conditions['runtime_file_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in binaries if p.is_file()}
        conditions['protocol_revision']=16
        manifest=dict(protocol_revision=16,method=method,mode=self.mode,conditions=conditions,source_sha256=sources,target_stream_sha256=stream_hash(self.config['online']['targets']),
            scope='Continuous timed references with two-target lookahead; simplified-model search; zero AirSim search calls; no perception')
        manifest['condition_fingerprint']=hashlib.sha256(json.dumps(dict(conditions=conditions,sources=sources),sort_keys=True).encode()).hexdigest()
        (self._run_dir/'online_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        self.planner=PlannerClient(self._run_dir);self._event('WORKER_STARTED',method=method)
        if self.mode!='airsim':raise ValueError('协议16飞行记录要求AirSim；离线验证使用专用测试脚本')

    def _rise(self,height):
        ControlRuntime._rise(self,height)
        self._settle(np.array(self._task()['start']),'MOVING_TO_START',60.)
        # No wait for worker: cold startup may continue while fallback is flying.

    def _request(self,boundary):
        o=self.config['online']
        return make_request(boundary,self._target_queue(),o['method'],seed=int(o['seed'])+self.route_index,
            limits=(self.cfg.max_speed,self.cfg.max_acceleration,30.))

    def _start(self):
        if self._active or self._flight_owned:return
        ControlRuntime._start(self)
        p,v,yaw,_=self._read_state();self.yaw_command=yaw;self.previous_yaw=yaw
        request=self._request([self._task()['start'],[0.,0.,0.],[0.,0.,0.]]);vector,checks=fallback(request)
        self.timeline=ReferenceTimeline(request,vector);self.previous_p=p.copy()
        self._event('REFERENCE_STARTED',checks=checks,actual_start_position=p.tolist(),actual_start_velocity=v.tolist())
        self._show_plan();self._markers(True)
        self._publish('RUNNING','连续飞行：绿色为当前可执行参考；搜索在独立进程进行。',running=True,hits=0)

    def _show_plan(self):
        if self.timeline:
            t=self.live_elapsed;_,w=self.timeline.at(t)
            end=max(t,w['start_s']+w['_curve'].duration_s)
            times=np.linspace(t,end,180);points=np.empty((len(times),3))
            active=[w for w in self.timeline.windows if not w.get('cancelled')]
            starts=np.array([w['start_s'] for w in active]);indices=np.searchsorted(starts,times,side='right')-1
            for i in np.unique(indices):
                selected=indices==i;entry=active[i];curve=entry['_curve']
                points[selected]=curve.sample(np.clip(times[selected]-entry['start_s'],0.,curve.duration_s))
            points=points.tolist()
            self._publish(planned_path=points,targets=self._target_queue(),route_markers=self._landmarks())

    def _schedule_fallback(self,reason):
        t=self.live_elapsed;handoff=t+.35;boundary,_=self.timeline.at(handoff)
        request=self._request(boundary);vector,checks=fallback(request)
        window=self.timeline.add(handoff,request,vector,'fallback',now=t)
        self._event('FALLBACK_SCHEDULED',reason=reason,window_id=window['id'],handoff_s=handoff,C2_errors=window['C2_errors'],checks=checks)
        self.next_request_s=t;self._show_plan()

    def _poll_planner(self,p,q):
        for reply in self.planner.poll():
            if reply['type']=='ready':self._event('WORKER_READY',**{k:v for k,v in reply.items() if k!='type'});continue
            if reply['type']!='result':
                self._event('WORKER_ERROR',reply=reply);self.pending=None;self.next_request_s=self.live_elapsed+2;continue
            pending=self.pending;reason=None;result=reply['result']
            if pending is not None and reply['job_id']==pending['job_id']:self.pending=None
            if pending is None or reply['job_id']!=pending['job_id']:reason='UNKNOWN_JOB'
            elif reply['epoch']!=self.timeline.epoch or pending['target_index']!=self.route_index:reason='STALE_REFERENCE_OR_TARGET'
            elif not result['deadline_met'] or reply['completed_host_s']>pending['deadline_host_s'] or time.perf_counter()>pending['deadline_host_s']:reason='WALL_DEADLINE'
            elif self.live_elapsed>pending['handoff_s']-.05:reason='HANDOFF_PASSED'
            elif not result['validation']['feasible']:reason='MODEL_VALIDATION_FAILED'
            elif np.linalg.norm(p-q)>.5:reason='CURRENT_TRACKING_ERROR'
            else:
                try:
                    check=reference_checks(pending,result['parameters'])
                    if not check['passed']:reason='PARENT_REFERENCE_CHECK_FAILED'
                    elif pending.get('goal_arrival_ceiling_s') is not None and pending['handoff_s']+result['parameters'][3]>pending['goal_arrival_ceiling_s']+1e-8:reason='GOAL_TIME_POSTPONED'
                    elif np.allclose(result['parameters'],pending['anchor'],atol=1e-9):reason='ANCHOR_UNCHANGED'
                    else:
                        window=self.timeline.add(pending['handoff_s'],pending,result['parameters'],'optimizer',now=self.live_elapsed)
                        self._event('CANDIDATE_SCHEDULED',job_id=pending['job_id'],window_id=window['id'],handoff_s=pending['handoff_s'],C2_errors=window['C2_errors'])
                        self._show_plan()
                except Exception as ex:reason='REFERENCE_REJECTED: '+str(ex)
            self._event('SEARCH_RESULT',disposition=reason or 'SCHEDULED',reply=reply)
            self.next_request_s=self.live_elapsed+2.
        if self.pending is not None or self.live_elapsed<self.next_request_s or not self.planner.ready:return
        # Stop/search latency does not freeze the flight/reference clock.
        t=self.live_elapsed;handoff=t+1.3;boundary,_=self.timeline.at(handoff)
        if handoff+.2>=self.config['online']['duration_s']:return
        # No new job while a queued handoff still lies ahead; preserves prefixes.
        if any(not w.get('cancelled') and w['start_s']>t for w in self.timeline.windows):return
        _,active=self.timeline.at(t)
        if active['start_s']+active['_curve'].ends[0]<handoff+.5:return
        request=self._prepare_search_request(self._request(boundary),active,handoff)
        if request is None:
            self.next_request_s=t+.5
            return
        host=time.perf_counter();request.update(job_id=len([e for e in self.online_events if e['stage']=='SEARCH_DISPATCHED']),epoch=self.timeline.epoch,
            handoff_s=handoff,target_index=self.route_index,deadline_host_s=host+self.config['online']['budget_s'],population=self.config['online'].get('population',8),max_candidates=self.config['online'].get('max_candidates',32),
            worker_journal_file=str(self._run_dir/'online_worker_evaluations.jsonl'))
        self.pending=request;self.planner.submit(request);self._event('SEARCH_DISPATCHED',request=request)

    def _prepare_search_request(self,request,active,handoff):
        vector,checks=fallback(request);request['anchor']=vector
        return request

    def _reached(self,p,yaw):
        goal=np.asarray(self._target_queue()[0]);before=self.previous_p;delta=p-before
        alpha=float(np.clip((goal-before)@delta/max(delta@delta,1e-20),0.,1.));nearest=before+alpha*delta
        distance=float(np.linalg.norm(nearest-goal));length_sq=float(delta@delta)
        outside=float((before-goal)@(before-goal)-self.cfg.arrival_radius**2);entry=0.
        if outside>0. and length_sq>1e-12:
            linear=2.*float((before-goal)@delta);discriminant=max(0.,linear**2-4*length_sq*outside)
            entry=float(np.clip((-linear-math.sqrt(discriminant))/(2*length_sq),0.,1.))
        bearing=goal-(before+entry*delta)
        desired=math.atan2(bearing[1],bearing[0]) if np.linalg.norm(bearing[:2])>1e-5 else yaw
        measured=wrap_angle(self.previous_yaw+entry*wrap_angle(yaw-self.previous_yaw))
        error=abs(wrap_angle(desired-measured))
        return distance<=self.cfg.arrival_radius and error<=math.radians(self.cfg.heading_tolerance_deg),distance,error

    def _tick(self,dt):
        begin=time.perf_counter();p,v,yaw,rpc=self._read_state();self._check_initial_position(p)
        stamp=int(self._last_state.timestamp);step=0. if self.previous_stamp is None else (stamp-self.previous_stamp)*1e-9
        if step<0 or step>.1:raise RuntimeError(f'连续控制状态间隔异常：{step:.3f}s')
        self.previous_stamp=stamp;self.live_elapsed+=step;t=self.live_elapsed
        boundary,window=self.timeline.at(t);q,qv,qa=boundary
        reached,distance,heading_error=self._reached(p,yaw)
        if reached:
            self._event('TARGET_REACHED',target_index=self.route_index,position=p.tolist(),velocity=v.tolist(),distance_m=distance,yaw_error_rad=heading_error)
            self.route_index+=1
            if self.route_index+1>=len(self.config['online']['targets']):self._stop('STREAM_COMPLETE');return
            self._schedule_fallback('TARGET_ADVANCED');self._markers(True)
        self._poll_planner(p,q)
        # A missed point is never silently counted as reached. Recover once.
        event_time=window['start_s']+window['_curve'].ends[0]
        if not reached and t>event_time+1.2 and window['request']['targets'][0]==self._target_queue()[0] and not any(not w.get('cancelled') and w['start_s']>t for w in self.timeline.windows):
            self._event('TARGET_MISSED',target_index=self.route_index,distance_m=distance)
            raise RuntimeError('未按时通过当前目标；保留失败并降落，禁止跳过计成功')
        raw=qv+2*(q-p)+.8*(qv-v)+.9*qa
        command=limit_command(raw,self.previous_command,max(step,.001),self.cfg.max_speed,self.cfg.max_acceleration)
        # Original03 aims at the current goal. Tangent-only yaw can turn too
        # late on a displaced optimized curve despite small position error.
        direction=np.asarray(self._target_queue()[0])-p
        desired=math.atan2(direction[1],direction[0]) if np.linalg.norm(direction[:2])>.05 else self.yaw_command
        # Fixed yaw lead compensates the measured goal-bearing servo lag.
        # Kept identical for all six methods; does not relax the 20-degree gate.
        los_rate=(direction[1]*v[0]-direction[0]*v[1])/max(float(direction[:2]@direction[:2]),.04)
        lead=float(self.config['online'].get('yaw_lead_s',.18))*np.clip(los_rate,-math.radians(self.cfg.max_yaw_rate_deg),math.radians(self.cfg.max_yaw_rate_deg))
        yaw_target=wrap_angle(desired+lead)
        rate=math.radians(self.cfg.max_yaw_rate_deg)*max(step,.001);self.yaw_command=wrap_angle(self.yaw_command+np.clip(wrap_angle(yaw_target-self.yaw_command),-rate,rate))
        self._send(command['limited'],math.degrees(self.yaw_command));self.previous_command=np.array(command['limited'])
        if begin-self.last_collision_host>.1:
            collision=self._client.simGetCollisionInfo(vehicle_name=self.vehicle_name);self.last_collision_host=begin
            if self.rows and collision.has_collided and int(collision.time_stamp)>=self.rows[0]['sim_timestamp_ns']:raise RuntimeError('连续飞行观测到新的碰撞')
        error=float(np.linalg.norm(p-q));self.max_tracking_error=max(self.max_tracking_error,error)
        if error>2.:raise RuntimeError('连续参考偏差超过2米紧急终止；0.5米评估门槛不变')
        row=dict(reference_elapsed_s=t,sim_timestamp_ns=stamp,position_zup_m=p.tolist(),velocity_zup_mps=v.tolist(),yaw_rad=float(yaw),
            reference_position_zup_m=q.tolist(),reference_velocity_zup_mps=qv.tolist(),reference_acceleration_zup_mps2=qa.tolist(),
            actual_acceleration_zup_mps2=from_ned(self._last_state.kinematics_estimated.linear_acceleration).tolist(),
            command=command,yaw_command_deg=math.degrees(self.yaw_command),window_id=window['id'],target_index=self.route_index,position_error_m=error,state_gap_s=step,host_monotonic_s=begin)
        self.rows.append(row);self.row_stream.write(json.dumps(row)+'\n');self._trail.append(p.tolist())
        loop=(time.perf_counter()-begin)*1000;self._timings.append((loop,rpc,step*1000))
        self._samples.append(dict(t=t,position=p.tolist(),velocity=v.tolist(),velocity_command=list(command['limited']),yaw=float(yaw),hits=self.route_index))
        self._publish(position=p.tolist(),velocity=v.tolist(),yaw=float(yaw),trail=list(self._trail),targets=self._target_queue(),hits=self.route_index,
            distance=distance,rpc_ms=rpc,loop_ms=loop,elapsed=t,reference_elapsed_s=t,yaw_error_deg=math.degrees(wrap_angle(desired-yaw)),
            route_markers=self._landmarks(),online_search_pending=self.pending is not None,
            message=f'连续飞行 {t:.1f}s · 已到达 {self.route_index} 点 · 后台'+('搜索中' if self.pending else '待机'))
        self.previous_p=p.copy();self.previous_yaw=yaw
        if t>=self.config['online']['duration_s']:self._stop('DURATION_COMPLETE')

    def _pause(self):self._publish(message='连续在线协议不暂停参考时钟；需要中止请点击停止并降落。')
    def _resume(self):pass

    def _save(self,outcome):
        if self._summary_written or self._run_dir is None:return
        if self.planner:
            for reply in self.planner.poll():
                if reply['type']=='result':
                    self._event('SEARCH_RESULT',disposition='FLIGHT_ENDED',reply=reply)
                    if self.pending and self.pending['job_id']==reply['job_id']:self.pending=None
                elif reply['type']=='ready':self._event('WORKER_READY',**{k:v for k,v in reply.items() if k!='type'})
                else:self._event('WORKER_ERROR',reply=reply)
        if self.pending:self._event('SEARCH_ABORTED',job_id=self.pending['job_id'],reason=outcome)
        if self.planner:self.planner.close();self.planner=None
        for name in ('row_stream','event_stream'):
            stream=getattr(self,name,None)
            if stream:stream.close();setattr(self,name,None)
        manifest_path=self._run_dir/'online_manifest.json'
        unchanged=True
        if manifest_path.exists():
            frozen=json.loads(manifest_path.read_text(encoding='utf-8'))
            unchanged=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['source_sha256'].items())
            if not unchanged:self._failure='飞行期间规划/控制源码变化；记录保留，禁止混入公平对比';outcome='FAILED'
        ControlRuntime._save(self,outcome)
        if not (self._run_dir/'online_manifest.json').exists():return
        summary_path=self._run_dir/'summary.json';summary=json.loads(summary_path.read_text(encoding='utf-8'));summary['sources_unchanged']=unchanged
        summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        (self._run_dir/'online_reference_windows.json').write_text(json.dumps(self.timeline.records() if self.timeline else [],ensure_ascii=False,indent=2),encoding='utf-8')
        (self._run_dir/'online_preparation.json').write_text(json.dumps(self.preparation_rows),encoding='utf-8')
        (self._run_dir/'online_marker_receipts.json').write_text(json.dumps(self.live_marker_receipts),encoding='utf-8')
        from .online_results import build_report
        metrics=build_report(self._run_dir)
        path=self._run_dir/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'));summary.update(protocol_revision=16,online_metrics=metrics,
            scope='Protocol16 online two-target proxy search and timed reference tracking; not original03 RollingController; zero AirSim search calls')
        for key in ('trajectory_name','bezier_parameters','planning_ms','limitation'):summary.pop(key,None)
        path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        self._publish(online_metrics=metrics,result_ready=True,result_page=str(self._run_dir/'online_result.html'))
        self._markers(False)
