"""Multi-point protocol14 with connect-time target preview and waypoint rules."""
import json,time
from pathlib import Path
import numpy as np
from .curve_live import LiveCurveRuntime
from .route_curve import decode_route,route_metrics,waypoint_results
from .runtime import from_ned
from .simulator import ROOT


class RouteLiveRuntime(LiveCurveRuntime):
    protocol_revision=14
    additional_source_paths=[Path(__file__),Path(__file__).with_name('route_curve.py')]

    def __init__(self,config,mode='airsim'):
        self.route_index=0;self.preview_curve=None;self.marker_origin=np.zeros(3)
        self.marker_cache=None;self.preview_receipts=[]
        super().__init__(config,mode)

    def _decode_reference(self,plan,factor):
        limits=plan['route'].get('command_limits',dict(max_speed_mps=6.,max_slew_mps2=6.))
        if abs(limits['max_speed_mps']-self.cfg.max_speed)>1e-9 or abs(limits['max_slew_mps2']-self.cfg.max_acceleration)>1e-9:raise ValueError('速度/加速度上限变化，请重新生成多点方案')
        return decode_route(plan,factor)
    def _curve(self):return self.live_curve if self.live_curve is not None else self.preview_curve

    def _target_queue(self):
        return self._curve().spec['waypoints'][self.route_index:self.route_index+2]

    def _target_point(self):
        points=self._target_queue()
        return np.array(points[0]) if points else self._curve().sample(self._curve().duration_s)

    def _markers(self,visible):
        if self.mode!='airsim':return
        points=self._target_queue() if visible else []
        key=(tuple(tuple(p) for p in points),tuple(self.marker_origin))
        if key==self.marker_cache:return
        for index,name in enumerate(('DC_TargetRed','DC_TargetBlue')):
            target=points[index] if index<len(points) else None
            xyz=np.array([0.,0.,1000.]) if target is None else np.array([target[0],target[1],-target[2]])+self.marker_origin
            if target is not None:
                if not self._client.simSetObjectScale(name,self._airsim.Vector3r(1.2,1.2,1.2)):raise RuntimeError('目标缩放设置失败：'+name)
            if not self._client.simSetObjectPose(name,self._airsim.Pose(self._airsim.Vector3r(*xyz),self._airsim.Quaternionr()),teleport=True):raise RuntimeError('目标实体放置失败：'+name)
            actual=self._client.simGetObjectPose(name).position;actual=[actual.x_val,actual.y_val,actual.z_val]
            if not np.allclose(actual,xyz,atol=.002):raise RuntimeError('目标坐标读回不符：'+name)
            scale=self._client.simGetObjectScale(name);scale=[scale.x_val,scale.y_val,scale.z_val]
            if target is not None and not np.allclose(scale,[1.2]*3,atol=.001):raise RuntimeError('目标尺寸读回不符')
            self.live_marker_receipts.append(dict(name=name,visible=target is not None,requested_ned=xyz.tolist(),observed_ned=actual,
                scale=scale,route_index=self.route_index,reference_time_s=self.live_elapsed,
                scope='Entity pose/scale readback; viewport visibility separately observed'))
        self.marker_cache=key
        self._publish(marker_status='红色当前点 / 蓝色下一点：实体位置和尺寸已读回',route_index=self.route_index,
            route_total=len(self._curve().spec['waypoints']))

    def _preview(self):
        plan=self.config.get('curve_experiment',{}).get('plan')
        if not plan or 'route' not in plan:return
        self.preview_curve=decode_route(plan,self.config['curve_experiment']['time_scale']);self.route_index=0
        if not self._active:self.live_curve=None
        if self.mode=='airsim' and self._client is not None:
            state=self._client.getMultirotorState(vehicle_name=self.vehicle_name)
            local=state.kinematics_estimated.position;world=self._client.simGetObjectPose(self.vehicle_name).position
            self.marker_origin=np.array([world.x_val-local.x_val,world.y_val-local.y_val,world.z_val-local.z_val])
            self._markers(True)
            self.preview_receipts=self.live_marker_receipts[-2:]
        self._publish(targets=self._target_queue(),planned_path=self.preview_curve.sample(np.linspace(0,self.preview_curve.duration_s,240)).tolist(),
            marker_world_origin_ned=self.marker_origin.tolist(),route_policy=plan['route']['policy_name'])

    def _handle(self,command,value):
        if command=='configure':self._publish('CONFIGURING','正在准备所选路线',execution_id=value.get('curve_experiment',{}).get('execution_id'))
        super()._handle(command,value)
        if command in ('simulator','connect','configure') and not self._active:
            self._preview()
            if self.mode=='airsim' and self.snapshot().get('connected') and self.config['curve_experiment'].get('auto_show_scene'):self.show_scene()

    def _start(self):
        if self._active or self._flight_owned:return
        self._publish('STARTING','准备执行所选多点方法',execution_id=self.config['curve_experiment'].get('execution_id'))
        self.route_index=0
        super()._start()

    def _begin_run(self):
        preview=list(self.preview_receipts)
        super()._begin_run()
        self.live_marker_receipts.extend(preview)

    def _advance_targets(self,t,p,v):
        curve=self._curve();points=curve.spec['waypoints']
        if self.route_index>=len(points):return
        event=curve.waypoint_times[self.route_index];distance=np.linalg.norm(p-points[self.route_index])
        due=(t>=event['evaluation_time_s'] if curve.spec['policy']=='stop' else t>=max(0.,event['arrival_time_s']-.5))
        reached=due and distance<=curve.spec['waypoint_radius_m']
        missed=t>event['evaluation_time_s']+.5
        if self.route_index<len(points)-1 and (reached or missed):
            self.route_index+=1;self._publish(targets=self._target_queue(),hits=self.route_index);self._markers(True)
        elif self.route_index==len(points)-1 and distance<=self.cfg.arrival_radius and not self.live_marker_hidden:
            self._publish(targets=[],hits=len(points));self._markers(False);self.live_marker_hidden=True

    def _completion_hits(self):
        rows=waypoint_results(self.live_curve,[s['reference_elapsed_s'] for s in self.live_states],
            [s['position_zup_m'] for s in self.live_states],[s['velocity_zup_mps'] for s in self.live_states])
        return sum(r['passed'] for r in rows)

    def _compute_metrics(self,config):return route_metrics(self.live_curve,self.live_states,self.live_commands[:-1])

    def _save(self,outcome):
        super()._save(outcome)
        if self._run_dir and self._summary_written:
            path=self._run_dir/'summary.json';summary=json.loads(path.read_text(encoding='utf-8'))
            summary.update(route=self.config['curve_experiment']['plan']['route'],marker_world_origin_ned=self.marker_origin.tolist(),
                scope='Protocol14 whole-route real-time execution; stop/pass are separate experiments')
            path.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
