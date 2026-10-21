"""Stage3 adapter of the fixed-map Executor; no target truth feeds controls.

Observation preparation uses a declared static free-space waypoint. The fixed
baseline stays in Executor. This subclass reuses its velocity, hover, return
and landing controller with closed safety fields and anonymized STATIC geometry.
"""
import copy,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import fixed_map_mission as mission
from fixed_map_perception_geometry import perception_safety_parameters,digest_json

def current_attribute_audit(frame, description, selected_id=None):
    """Fresh RGB guard, independent of cloud choice and local track IDs.

    Broad 'red balloon' descriptions remain ambiguous when two red candidates
    are visible. An unsupported descriptor cannot silently pass this profile.
    """
    import cv2
    from t1_source_attribute_guard import band_evidence
    from fixed_map_target_profiles import yellow_evidence
    audit=dict(status='REJECTED',reason='CURRENT_ATTRIBUTE_NOT_VERIFIED',frame_id=frame.frame_id,image_sha256=frame.image_sha256,description=description,candidates=[])
    from fixed_map_frame_image import decode_bound_rgb
    try:image=decode_bound_rgb(frame.encoded_image,frame.image_sha256)
    except ValueError:return audit
    if image.shape[:2]!=(frame.height,frame.width):return audit
    matching=[]
    for c in frame.candidates:
        band=band_evidence(image,list(c.bbox_xyxy_pixels))
        red=band.get('core_red_fraction',0)>=.15 and min(c.bbox_xyxy_pixels[2]-c.bbox_xyxy_pixels[0],c.bbox_xyxy_pixels[3]-c.bbox_xyxy_pixels[1])>=24 and not c.truncated
        yellow=yellow_evidence(image,c.bbox_xyxy_pixels,c.truncated)
        match=(red if description=='红色气球' else
               red and band['status']!='ACCEPTED' if description=='普通红色气球' else
               yellow['matched'] if description=='黄色气球' else
               band['status']=='ACCEPTED' if description=='带白色横条的红色气球' else False)
        audit['candidates'].append(dict(candidate_id=c.candidate_id,red=bool(red),yellow=yellow,band=band,matches_description=bool(match)))
        if match:matching.append(c.candidate_id)
    audit['matching_candidate_ids']=matching
    if description not in ['红色气球','普通红色气球','黄色气球','带白色横条的红色气球']:audit['reason']='UNSUPPORTED_CURRENT_ATTRIBUTE_PROFILE'
    elif len(matching)>1:audit['reason']='DESCRIPTOR_AMBIGUOUS_MULTIPLE_MATCHING_CANDIDATES'
    elif len(matching)==0:audit['reason']='CURRENT_DESCRIPTOR_ATTRIBUTE_NOT_VISIBLE'
    elif selected_id is None:audit['reason']='CURRENT_ATTRIBUTE_VISIBLE_BUT_INSTANCE_NOT_CONFIRMED'
    elif selected_id!=matching[0]:audit['reason']='CURRENT_ATTRIBUTE_SELECTION_MISMATCH'
    else:audit.update(status='ACCEPTED',reason='UNIQUE_CURRENT_RGB_ATTRIBUTE_MATCH',selected_candidate_id=selected_id)
    return audit

def static_geometry_v2(primitives):
    kept=[];removed=[]
    for p in primitives:
        name=p.get('id','').lower()
        if p.get('group')=='balloons' or any(k in name for k in ['balloon','tether','anchor']):
            removed.append(p.get('id'));continue
        r=copy.deepcopy(p);r['id']=f'static-{len(kept):03d}';r['display_label']=r['id']
        if r.get('group') not in ['ground','start_platform']:r['group']='static_obstacle'
        kept.append(r)
    return kept,removed

class TrackingMarginGeometry(mission.Geometry):
    """Conservative planning allowance; original execution thresholds stay intact."""
    def __init__(self,parts,radius=.75,margin=.15):
        super().__init__(parts,radius)
        if not 0<=margin<=.3:raise ValueError('PLANNING_MARGIN_OUT_OF_RANGE')
        self.planning_margin=margin
    def plan(self,start,goal,minimum=1.25):
        return super().plan(start,goal,minimum+self.planning_margin)

class PerceptionExecutor(mission.Executor):
    def __init__(self, declaration=None, service=None):
        super().__init__();self.baseline_task=copy.deepcopy(self.task);self.baseline_hash=self.task_hash;self.baseline_mode=declaration is None
        self.safety=perception_safety_parameters(self.task);self.baseline_geometry=self.g
        self.declaration=copy.deepcopy(declaration);self.service=service
        self.source_epoch=mission.sha(__file__);self.active_plan=None;self.perception_mode=False;self.preview_result=None;self.plan_version=0;self.resume_required=False
        self.planner=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sensor-static-planner');self.pending_plan=None
        parts,removed=static_geometry_v2(mission.read(mission.OUT/'fixed_version/editor_audit/planning_geometry.json')['primitives'])
        self.static_geometry=TrackingMarginGeometry(parts,self.task['envelope_radius_config_m'],service.cfg.get('tracking_clearance_margin_m',0.) if service else 0.)
        if declaration is not None:self.set_observation(declaration)
        mission.save(mission.OUT/'static_geometry_v2.json',dict(primitives=parts,removed_semantic_ids=removed,physical_platforms_preserved=True))
    def set_observation(self,declaration):
        goal=np.array(declaration['observation_goal_xyz_m'],float)
        if goal.shape!=(3,) or not np.isfinite(goal).all() or goal[2]!=self.safety['height_world_m'] or np.any(goal<[-3,-14,2.8]) or np.any(goal>[51,14,3.4]):raise ValueError('DECLARED_OBSERVATION_WORKSPACE_FAILED')
        self.declaration=copy.deepcopy(declaration);self.baseline_mode=False;self.perception_mode=False;self.g=self.static_geometry
        self.task=dict(self.safety,schema='observation_preparation_v1',vehicle='DroneA',idle_vehicle='DroneB',goal_xyz_m=goal.tolist(),target_source='predeclared_static_free_space_observation_NOT_perception_guidance')
        self.task_hash=digest_json(self.task)
    def start_baseline(self,actor='native_baseline_button'):
        self.commands.put(dict(action='baseline_start',actor=actor))
    def prepare_observation(self,declaration,actor='native_observation_button'):
        self.commands.put(dict(action='prepare_observation',actor=actor,declaration=copy.deepcopy(declaration)))
    def plan_perception(self,execute=False,actor='native_perception_button'):
        self.commands.put(dict(action='execute_perception' if execute else 'plan_perception',actor=actor))
    def calibrate(self):
        if self.baseline_mode:return super().calibrate()
        # Reuse prior ground reference; do not query target Actor transforms.
        ref=mission.read(mission.ROOT/'reports/step1_quick_review_20261010_083715/runtime_geometry.json')
        self.global_origin=np.array(ref['global_origin_xyz_m']);self.local_origin=np.array(ref['DroneA_local_origin_xyz_m']);self.b_origin=np.array(ref['DroneB_local_origin_xyz_m'])
        for name,origin in [('DroneA',self.local_origin),('DroneB',self.b_origin)]:
            before=self.c.getMultirotorState(vehicle_name=name);obj=self.c.simGetObjectPose(name);after=self.c.getMultirotorState(vehicle_name=name)
            estimated=origin+mission.xyz(before.kinematics_estimated.position)*[1,1,-1];actual=self.global_origin+mission.xyz(obj.position)*[1,1,-1]
            bound=.03+self.task['max_speed_measured_m_s']*abs(int(after.timestamp)-int(before.timestamp))/1e9
            if np.linalg.norm(actual-estimated)>bound:raise ValueError('SELF_WORLD_REFERENCE_MISMATCH')
    def capture(self,label):
        if self.baseline_mode:return super().capture(label)
        self.event('CONTROL_CAPTURE_REFERENCE',label=label,scope='raw onboard images belong to independent perception worker; no blocking render on control owner')
    def start_task(self,request):
        if self.baseline_mode:return super().start_task(request)
        if self.stage!='READY':raise ValueError('OBSERVATION_START_REQUIRES_GROUND')
        self.calibrate();self.sample()
        if not self.state['landed'] or not self.state['DroneB_landed'] or self.state['DroneB_api_control'] or self.c.isApiControlEnabled(vehicle_name='DroneA'):raise ValueError('GROUND_NOT_RELEASED')
        if mission.sha(__file__)!=self.source_epoch:raise ValueError('EXECUTION_ADAPTER_CHANGED')
        self.home=self.p.copy();self.ground_b=self.state['DroneB_position_xyz_m']
        hover=self.p.copy();hover[2]=self.task['height_world_m']
        if not self.g.segment(self.p,hover,self.task['clearance_stop_m'],True)[0]:raise ValueError('LIVE_TAKEOFF_COLUMN_FAILED')
        self.path=self.g.plan(hover,self.task['goal_xyz_m'],self.task['path_clearance_m'])
        self.run_dir=mission.OUT/('flight_'+time.strftime('%Y%m%d_%H%M%S'));self.run_dir.mkdir()
        mission.save(self.run_dir/'task.json',self.task);mission.save(self.run_dir/'declaration.json',self.declaration);mission.save(self.run_dir/'plan.json',dict(home_xyz_m=self.home.tolist(),path=self.path,segments=[self.g.segment(a,b,self.task['path_clearance_m'])[1] for a,b in zip(self.path,self.path[1:])],source='observation_preparation_static_geometry'))
        self.log=(self.run_dir/'timestamped.jsonl').open('w',encoding='utf-8');self.trace=[];self.run_done=False;self.hold_since=None
        self.reason='';self.stage='TAKEOFF';self.target=hover;self.index=1;self.vcmd=np.zeros(3);self.start_stamp=int(self.raw_state.timestamp);self.collision_base=int(self.c.simGetCollisionInfo(vehicle_name='DroneA').time_stamp)
        self.last_host=time.monotonic();self.last_stamp=None;self.prev=None
        self.active_plan=dict(plan_version=1,source='observation_preparation',task_sha256=self.task_hash,goal_xyz_m=self.task['goal_xyz_m'])
        self.c.enableApiControl(True,vehicle_name='DroneA')
        if not self.c.armDisarm(True,vehicle_name='DroneA'):raise ValueError('ARM_FAILED')
        self.event('OBSERVATION_START_ACCEPTED',request=request,plan_binding=self.active_plan,perception_guidance_success=False)
    def control(self):
        # v12 maintains the fixed plane while waiting for fresh recovery data.
        # The stopped XY location, rather than any stale target, is held.
        self.stationary_recovery_hold()
        if self.service and self.service.cfg.get('control_timing_trace_v18'):
            row=self.service.latest;capture=row.get('capture',{});now=time.monotonic()
            self.event('PERCEPTION_CONTROL_CYCLE_TIMING',state_timestamp_ns=self.state.get('state_timestamp_ns'),actual_height_m=float(self.p[2]),height_error_m=abs(float(self.p[2])-self.safety['height_world_m']),actual_speed_m_s=self.state.get('speed_m_s'),frame_id=row.get('frame_id'),frame_valid=row.get('valid'),capture_received_monotonic_s=capture.get('received_monotonic_s'),inference_processed_monotonic_s=row.get('processed_monotonic_s'),parent_publication=getattr(self.service,'last_publication_metric',None),source_age_s=now-capture['received_monotonic_s'] if capture.get('received_monotonic_s') is not None else None,recovery_plane_ready=getattr(self,'recovery_plane_ready',False),plan_pending=self.pending_plan is not None,stationary_height_integral=getattr(self,'stationary_height_integral',0.),command_xyz_m_s=self.vcmd.tolist())
        self.poll_initial_plan_request()
        self.poll_plan()
        if self.perception_mode and self.stage in ('APPROACH','ARRIVAL_HOVER','ARRIVED'):
            from fixed_map_perception_task import execution_evidence
            try:
                if self.service is None:raise ValueError('PERCEPTION_SERVICE_DISCONNECTED')
                service=self.service;row=service.latest
                evidence=execution_evidence(self.active_plan,row,service.cfg,time.monotonic())
                if time.monotonic()-row['capture']['received_monotonic_s']>service.cfg['current_max_age_s']:raise ValueError('EVIDENCE_EXPIRED_DURING_EXECUTION_CHECK')
                if self.service is not service or not service.latest.get('valid'):raise ValueError('NEWER_EVIDENCE_INVALIDATED_DURING_CHECK')
                lo=np.array(self.service.cfg['workspace_min_xyz_m']);hi=np.array(self.service.cfg['workspace_max_xyz_m'])
                if np.any(self.p<lo) or np.any(self.p>hi):raise ValueError('ACTIVE_SELF_OUTSIDE_WORKSPACE')
                self.event('PERCEPTION_MOTION_EVIDENCE',binding=evidence,plan_version=self.active_plan['plan_version'],goal_xyz_m=self.task['goal_xyz_m'],state_timestamp_ns=self.state['state_timestamp_ns'])
            except (ValueError,KeyError,TypeError) as ex:
                self.hold('PERCEPTION_EVIDENCE_INVALID_'+str(ex),'STOP_HOVER');self.resume_required=True
                self.event('PERCEPTION_OLD_ACTION_CANCELLED',reason=str(ex),plan_version=self.active_plan['plan_version'])
                self.recovery_plane_ready=False;self.recovery_plane_since=None;self.recovery_plane_deadline=None
                self.preview_result=dict(status='EVIDENCE_INVALIDATED_HOVER',reason=str(ex),previous_plan_version=self.active_plan['plan_version'])
        if self.perception_mode and self.stage=='ABORTED' and self.resume_required and self.pending_plan is None:
            from fixed_map_perception_task import live_estimate
            try:
                if self.service.cfg.get('recovery_plane_gate_v18') and not self.recovery_plane_gate():raise ValueError('RECOVERY_PLANE_NOT_STABLE')
                live_estimate(self.service.latest,self.service.cfg,time.monotonic())
            except (ValueError,KeyError,TypeError):pass
            else:
                try:self.request_plan(True,'fresh_reconfirmation_replan')
                except (ValueError,KeyError,TypeError) as ex:
                    transient=str(ex) in ['CURRENT_FRAME_EXPIRED','NO_VALID_ONLINE_DESIGNATED_ESTIMATE','TARGET_NOT_TRACKING','SEMANTIC_EVIDENCE_EXPIRED']
                    self.preview_result=dict(status='WAITING_FRESH_RECOVERY' if transient else 'REJECTED',reason=str(ex))
                    self.event('PERCEPTION_RECOVERY_WAITING_FRESH' if transient else 'PERCEPTION_RECOVERY_REJECTED',reason=str(ex),no_old_target_motion=True)
                else:self.resume_required=False
        self.event('ACTIVE_PLAN_BINDING',state_timestamp_ns=self.state.get('state_timestamp_ns'),binding=self.active_plan)
        super().control()
    def poll_initial_plan_request(self):
        pending=getattr(self,'initial_plan_request',None)
        if pending is None:return
        if self.stage!='ARRIVED' or self.perception_mode:
            self.initial_plan_request=None;return
        if time.monotonic()>=pending['deadline']:
            self.initial_plan_request=None;self.preview_result=dict(status='REJECTED',reason='INITIAL_FRESH_PLAN_HANDOFF_DEADLINE')
            self.event('PERCEPTION_INITIAL_PLAN_WAIT_EXPIRED',no_motion_issued=True);return
        try:self.request_plan(pending['execute'],pending['actor'])
        except (ValueError,KeyError,TypeError) as ex:
            if str(ex) in ['CURRENT_FRAME_EXPIRED','NO_VALID_ONLINE_DESIGNATED_ESTIMATE','TARGET_NOT_TRACKING','SEMANTIC_EVIDENCE_EXPIRED']:
                self.preview_result=dict(status='WAITING_FRESH_INITIAL_PLAN',reason=str(ex),deadline_monotonic_s=pending['deadline']);return
            self.initial_plan_request=None;self.preview_result=dict(status='REJECTED',reason=str(ex));self.event('PERCEPTION_TASK_REJECTED',reason=str(ex));return
        self.initial_plan_request=None
    def stationary_recovery_hold(self):
        service=getattr(self,'service',None)
        enabled=service is not None and service.cfg.get('recovery_policy_version')=='fixed_height_stationary_hold_v12'
        active=enabled and self.perception_mode and self.stage=='ABORTED'
        if not active:
            self.stationary_hold_origin=None;self.stationary_hold_previous=None;self.stationary_hold_stamp=None
            self.stationary_height_integral=0.
            return
        now=time.monotonic();previous=getattr(self,'stationary_hold_previous',None)
        if previous is not None and now-previous>self.safety['state_gap_stop_host_s']:raise ValueError('RECOVERY_HOLD_STATE_GAP')
        stamp=self.state['state_timestamp_ns']
        if previous is not None and stamp<=getattr(self,'stationary_hold_stamp',0):raise ValueError('RECOVERY_HOLD_STALE_STATE')
        if self.state['landed'] or not self.state['DroneB_landed'] or self.state['DroneB_api_control']:raise ValueError('RECOVERY_HOLD_SELF_STATE_FAILED')
        if np.linalg.norm(np.array(self.state['DroneB_position_xyz_m'])-self.ground_b)>.1:raise ValueError('DRONE_B_GROUND_INVARIANT_FAILED')
        if self.state['speed_m_s']>self.safety['max_speed_measured_m_s']:raise ValueError('RECOVERY_HOLD_SPEED_LIMIT')
        origin=getattr(self,'stationary_hold_origin',None)
        if origin is None:
            origin=self.p.copy();origin[2]=self.safety['height_world_m'];self.stationary_hold_origin=origin
            self.event('PERCEPTION_STATIONARY_HOLD_STARTED',hold_xyz_m=origin.tolist(),stale_target_motion=False)
        if np.linalg.norm(self.p[:2]-origin[:2])>self.safety['hover_drift_max_m']:raise ValueError('RECOVERY_HOLD_XY_DRIFT')
        plane=self.p.copy();plane[2]=self.safety['height_world_m']
        if not self.g.segment(self.p,plane,self.safety['path_clearance_m'])[0]:raise ValueError('RECOVERY_HOLD_COLUMN_BLOCKED')
        dt=self.safety['sample_period_host_s'] if previous is None else min(.2,max(0,now-previous))
        old=self.vcmd.copy();desired=.7*(origin-self.p)-.25*self.v
        if service.cfg.get('stationary_height_integral_v18'):
            integral=getattr(self,'stationary_height_integral',0.)+(origin[2]-self.p[2])*dt
            self.stationary_height_integral=float(np.clip(integral,-.15,.15))
            desired[2]+=.35*self.stationary_height_integral
        norm=np.linalg.norm(desired)
        if norm>.25:desired*=.25/norm
        dv=desired-old;norm=np.linalg.norm(dv);limit=self.safety['max_command_accel_m_s2']*dt
        if norm>limit:dv*=limit/norm
        proposed=old+dv
        if not self.g.segment(self.p,self.p+proposed*.7,self.safety['clearance_stop_m'])[0]:raise ValueError('RECOVERY_HOLD_PREDICTED_SEGMENT_UNSAFE')
        self.c.moveByVelocityAsync(float(proposed[0]),float(proposed[1]),float(-proposed[2]),self.safety['command_duration_sim_s'],drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=self.api.YawMode(False,0),vehicle_name='DroneA')
        self.vcmd=proposed;self.stationary_hold_previous=now;self.stationary_hold_stamp=stamp
        self.event('velocity_command_requested',world_xyz_m_s=proposed.tolist(),source_state_timestamp_ns=self.state['state_timestamp_ns'],duration_sim_s=self.safety['command_duration_sim_s'],source='stationary_recovery_fixed_plane_NOT_target_guidance',command_generation_monotonic_s=now,command_ramp_dt_s=dt,previous_command_xyz_m_s=old.tolist(),command_ramp_accel_m_s2=float(np.linalg.norm(proposed-old)/dt) if dt>0 else None)
    def recovery_plane_gate(self):
        """Separate bounded fixed-plane recovery from the two-second handoff."""
        now=time.monotonic()
        if getattr(self,'recovery_plane_ready',False):
            if abs(self.p[2]-self.safety['height_world_m'])<=.02 and self.state['speed_m_s']<=self.safety['arrival_speed_m_s']:return True
            self.recovery_plane_ready=False;self.recovery_plane_since=None
        if getattr(self,'recovery_plane_deadline',None) is None:
            self.recovery_plane_deadline=now+25.
            self.event('PERCEPTION_RECOVERY_PLANE_GATE_STARTED',budget_host_s=25.,height_tolerance_m=.02,stable_dwell_sim_s=self.safety['arrival_dwell_s'],plan_handoff_budget_s=self.service.cfg['plan_fresh_handoff_timeout_s'])
        if now>=self.recovery_plane_deadline:
            self.preview_result=dict(status='REJECTED',reason='RECOVERY_PLANE_STABILITY_DEADLINE');self.resume_required=False
            self.event('PERCEPTION_RECOVERY_PLANE_GATE_EXPIRED',actual_height_m=float(self.p[2]),no_target_motion=True)
            return False
        good=abs(self.p[2]-self.safety['height_world_m'])<=.02 and self.state['speed_m_s']<=self.safety['arrival_speed_m_s']
        if not good:self.recovery_plane_since=None;return False
        stamp=self.state['state_timestamp_ns']
        if getattr(self,'recovery_plane_since',None) is None:self.recovery_plane_since=stamp;return False
        if (stamp-self.recovery_plane_since)/1e9<self.safety['arrival_dwell_s']:return False
        self.recovery_plane_ready=True
        self.event('PERCEPTION_RECOVERY_PLANE_GATE_PASSED',actual_height_m=float(self.p[2]),actual_speed_m_s=self.state['speed_m_s'],hold_start_timestamp_ns=self.recovery_plane_since,hold_end_timestamp_ns=stamp,stable_hold_sim_s=(stamp-self.recovery_plane_since)/1e9)
        return True
    def process_request(self,r):
        action=r['action']
        if action=='baseline_start':
            if self.stage!='READY':raise ValueError('BASELINE_REQUIRES_RELEASED_GROUND')
            self.baseline_mode=True;self.perception_mode=False;self.task=copy.deepcopy(self.baseline_task);self.task_hash=self.baseline_hash;self.g=self.baseline_geometry;self.active_plan=None
            return super().start_task(r)
        if action=='prepare_observation':
            if self.stage!='READY':raise ValueError('OBSERVATION_REQUIRES_RELEASED_GROUND')
            self.set_observation(r['declaration']);return self.start_task(r)
        if action in ['plan_perception','execute_perception']:
            try:self.request_plan(action=='execute_perception',r.get('actor'))
            except (ValueError,KeyError,TypeError) as ex:
                transient=str(ex) in ['CURRENT_FRAME_EXPIRED','NO_VALID_ONLINE_DESIGNATED_ESTIMATE','TARGET_NOT_TRACKING','SEMANTIC_EVIDENCE_EXPIRED']
                if action=='execute_perception' and self.service and self.service.cfg.get('initial_plan_fresh_wait_v14') and self.stage=='ARRIVED' and not self.perception_mode and transient:
                    self.initial_plan_request=dict(execute=True,actor=r.get('actor'),deadline=time.monotonic()+self.service.cfg.get('plan_fresh_handoff_timeout_s',0.))
                    self.preview_result=dict(status='WAITING_FRESH_INITIAL_PLAN',reason=str(ex),deadline_monotonic_s=self.initial_plan_request['deadline'])
                    self.event('PERCEPTION_INITIAL_PLAN_WAITING_FRESH',reason=str(ex),no_motion_issued=True);return
                self.reason='PERCEPTION_TASK_REJECTED_'+str(ex);self.event('PERCEPTION_TASK_REJECTED',reason=str(ex));self.preview_result=dict(status='REJECTED',reason=str(ex))
            return
        if action=='resume' and self.perception_mode:
            if self.stage!='PAUSED':return
            self.request_plan(True,'manual_fresh_resume_replan');return
        if action=='return':
            self.perception_mode=False;self.resume_required=False;self.pending_plan=None;self.initial_plan_request=None
            if self.stage in ['PAUSED','ARRIVED','ABORTED','STOP_HOVER','ARRIVAL_HOVER']:self.align_return_height()
        return super().process_request(r)
    def align_return_height(self):
        if abs(self.p[2]-self.safety['height_world_m'])<=.02:return
        plane=self.p.copy();plane[2]=self.safety['height_world_m']
        if not self.g.segment(self.p,plane,self.safety['path_clearance_m'])[0]:raise ValueError('RETURN_HEIGHT_COLUMN_BLOCKED')
        self.hold('ALIGN_ORIGINAL_RETURN_HEIGHT','STOP_HOVER');last=None;vcmd=np.zeros(3);stable=None;deadline=time.monotonic()+25
        self.event('RETURN_HEIGHT_ALIGNMENT',actual_xyz_m=self.p.tolist(),goal_xyz_m=plane.tolist(),original_height_retained=True)
        while time.monotonic()<deadline:
            tick=time.monotonic();self.sample()
            if last is not None and tick-last>self.safety['state_gap_stop_host_s']:raise ValueError('RETURN_ALIGNMENT_STATE_GAP')
            last=tick;desired=.7*(plane-self.p)-.25*self.v;norm=np.linalg.norm(desired)
            if norm>.25:desired*=.25/norm
            dv=desired-vcmd;limit=self.safety['max_command_accel_m_s2']*.1;norm=np.linalg.norm(dv)
            if norm>limit:dv*=limit/norm
            vcmd+=dv
            if self.state['speed_m_s']>self.safety['max_speed_measured_m_s'] or not self.g.segment(self.p,self.p+vcmd*.7,self.safety['clearance_stop_m'])[0]:raise ValueError('RETURN_ALIGNMENT_UNSAFE')
            self.c.moveByVelocityAsync(float(vcmd[0]),float(vcmd[1]),float(-vcmd[2]),self.safety['command_duration_sim_s'],drivetrain=self.api.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=self.api.YawMode(False,0),vehicle_name='DroneA')
            self.event('velocity_command_requested',world_xyz_m_s=vcmd.tolist(),source_state_timestamp_ns=self.state['state_timestamp_ns'],duration_sim_s=self.safety['command_duration_sim_s'],source='original_return_height_alignment')
            good=abs(self.p[2]-plane[2])<=.02 and self.state['speed_m_s']<=self.safety['arrival_speed_m_s']
            if good:
                if stable is None:stable=int(self.raw_state.timestamp)
                if (int(self.raw_state.timestamp)-stable)/1e9>=self.safety['arrival_dwell_s']:
                    self.hold('RETURN_HEIGHT_ALIGNED','STOP_HOVER');self.sample();return
            else:stable=None
            time.sleep(max(0,.1-(time.monotonic()-tick)))
        raise ValueError('RETURN_ALIGNMENT_TIMEOUT')
    def request_plan(self,execute,actor):
        from fixed_map_perception_task import live_estimate,build_preview
        if not self.service:raise ValueError('PERCEPTION_SERVICE_NOT_CONNECTED')
        if execute and self.service.cfg.get('ground_development_only'):raise ValueError('GROUND_DEVELOPMENT_EXECUTION_DISABLED')
        if self.pending_plan is not None:raise ValueError('BOUNDED_PLANNER_SLOT_OCCUPIED')
        if self.stage not in ['READY','ARRIVED','ABORTED','PAUSED']:raise ValueError('PLAN_REQUIRES_GROUND_OR_STABLE_HOVER')
        if execute and (self.stage=='READY' or self.baseline_mode):raise ValueError('EXECUTION_REQUIRES_DECLARED_OBSERVATION_HOVER')
        if execute and (self.state['speed_m_s']>self.safety['arrival_speed_m_s'] or not self.state['DroneB_landed'] or self.state['DroneB_api_control']):raise ValueError('EXECUTION_SELF_STATE_FAILED')
        submitted=time.monotonic();row=copy.deepcopy(self.service.latest);estimate=live_estimate(row,self.service.cfg,time.monotonic())
        if execute and self.perception_mode:
            from fixed_map_perception_task import identity_context
            if identity_context(estimate)!=identity_context(self.active_plan['estimate']):raise ValueError('RECOVERY_TARGET_CONTEXT_CHANGED')
            if self.service.cfg.get('recovery_plane_gate_v18') and self.stage=='ABORTED':
                if not getattr(self,'recovery_plane_ready',False) or abs(self.p[2]-self.safety['height_world_m'])>.02:raise ValueError('RECOVERY_PLANE_NOT_STABLE')
            else:self.restore_recovery_plane()
            # Alignment takes real time. Only NEW currently valid evidence may
            # supply the resumed target; never refresh the original timestamp.
            row=copy.deepcopy(self.service.latest);estimate=live_estimate(row,self.service.cfg,time.monotonic())
            if identity_context(estimate)!=identity_context(self.active_plan['estimate']):raise ValueError('RECOVERY_TARGET_CONTEXT_CHANGED')
        validation_finished=time.monotonic()
        if mission.sha(__file__)!=self.source_epoch:raise ValueError('EXECUTION_ADAPTER_CHANGED')
        start=self.p.copy();previous_preview=self.preview_result;self.preview_result=dict(status='PLANNING',source_frame_id=estimate['frame_id'],execute_requested=execute)
        if execute and not self.perception_mode and previous_preview and previous_preview.get('status')=='FRESH_REVALIDATED_PERCEPTION_PLAN':
            cached=dict(task=copy.deepcopy(previous_preview['task']),path=copy.deepcopy(previous_preview['path']),planned_from_estimate=copy.deepcopy(previous_preview['estimate']))
            future=self.planner.submit(lambda:cached)
        else:future=self.planner.submit(build_preview,row,start.tolist(),self.static_geometry,self.safety,self.service.cfg,validation_finished)
        self.pending_plan=dict(future=future,execute=execute,actor=actor,submitted_monotonic_s=submitted,validation_finished_monotonic_s=validation_finished)
        self.event('PERCEPTION_PLAN_SUBMITTED',actor=actor,execute_requested=execute,estimate=estimate,start_xyz_m=start.tolist(),submission_validation_wall_s=validation_finished-submitted)
    def restore_recovery_plane(self):
        if self.stage not in ['ABORTED','PAUSED','ARRIVED']:raise ValueError('RECOVERY_REQUIRES_STABLE_STOP')
        if abs(self.p[2]-self.safety['height_world_m'])<=.02:return
        stage=self.stage;self.event('PERCEPTION_RECOVERY_HEIGHT_REQUIRED',actual_xyz_m=self.p.tolist(),fixed_height_m=self.safety['height_world_m'])
        # Reuse the measured return-height controller, including its column
        # clearance, speed, acceleration and three-second stable hold guards.
        self.align_return_height();self.sample()
        if abs(self.p[2]-self.safety['height_world_m'])>.02 or self.state['speed_m_s']>self.safety['arrival_speed_m_s']:raise ValueError('RECOVERY_HEIGHT_STABILITY_FAILED')
        self.stage=stage;self.last_host=time.monotonic();self.last_stamp=None;self.prev=None
        self.stationary_hold_previous=None
        self.stationary_recovery_hold()
        self.event('PERCEPTION_RECOVERY_HEIGHT_RESTORED',actual_xyz_m=self.p.tolist(),height_guard_m=.02,new_fresh_plan_required=True)
    def poll_plan(self):
        if self.pending_plan is None or not self.pending_plan['future'].done():return
        pending=self.pending_plan
        from fixed_map_perception_task import revalidate_preview
        try:
            preview=pending['future'].result()
            if 'finished_monotonic_s' not in pending:
                pending['finished_monotonic_s']=time.monotonic()
                pending['handoff_deadline_monotonic_s']=time.monotonic()+self.service.cfg.get('plan_fresh_handoff_timeout_s',0.)
                self.event('PERCEPTION_PLANNER_FINISHED',elapsed_host_s=time.monotonic()-pending['submitted_monotonic_s'],source_frame_id=preview['planned_from_estimate']['frame_id'])
            row=self.service.latest;started=time.monotonic()
            receipt=revalidate_preview(preview,row,self.p,self.static_geometry,self.safety,self.service.cfg,started)
            if time.monotonic()-receipt['estimate']['received_monotonic_s']>self.service.cfg['current_max_age_s']:raise ValueError('EVIDENCE_EXPIRED_DURING_PLAN_REVALIDATION')
            max_handoff_age=self.service.cfg.get('plan_max_evidence_age_s',self.service.cfg['current_max_age_s'])
            if not 0.<max_handoff_age<=self.service.cfg['current_max_age_s']:raise ValueError('PLAN_EVIDENCE_AGE_BUDGET_INVALID')
            if time.monotonic()-receipt['estimate']['received_monotonic_s']>max_handoff_age:raise ValueError('PLAN_EVIDENCE_TOO_OLD_FOR_HANDOFF')
            if self.service.latest is not row and not self.service.latest.get('valid'):raise ValueError('NEWER_EVIDENCE_INVALIDATED_DURING_CHECK')
            self.pending_plan=None
            receipt['timing']=dict(submission_monotonic_s=pending['submitted_monotonic_s'],planner_finished_monotonic_s=pending['finished_monotonic_s'],revalidation_started_monotonic_s=started,revalidation_finished_monotonic_s=time.monotonic(),current_evidence_age_s=time.monotonic()-receipt['estimate']['received_monotonic_s'],bounded_wait_budget_s=self.service.cfg.get('plan_fresh_handoff_timeout_s',0.))
            self.plan_version+=1;receipt['plan_version']=self.plan_version
            receipt['schema']='perception_surface_execution_v1';self.preview_result=dict(status='FRESH_REVALIDATED_PERCEPTION_PLAN',**receipt)
            destination=self.run_dir or mission.OUT;mission.save(destination/('perception_plan_'+str(self.plan_version)+'.json'),self.preview_result)
            self.event('PERCEPTION_PLAN_FRESH_REVALIDATED',plan=receipt,execute_requested=pending['execute'])
            if pending['execute']:
                if self.stage not in ['ARRIVED','ABORTED','PAUSED'] or self.state['speed_m_s']>self.safety['arrival_speed_m_s']:raise ValueError('SELF_STATE_CHANGED_DURING_PLANNING')
                if abs(self.p[2]-self.safety['height_world_m'])>.03:raise ValueError('SELF_HEIGHT_ALIGNMENT_REQUIRED')
                self.task=dict(self.safety,schema='perception_surface_execution_v1',vehicle='DroneA',idle_vehicle='DroneB',goal_xyz_m=receipt['task']['goal_xyz_m'],target_source='online_onboard_visible_surface_NOT_center',source_binding={k:receipt['estimate'][k] for k in ['session_id','target_generation','description_sha256','frame_id','image_sha256']})
                self.task_hash=digest_json(self.task);self.g=self.static_geometry;self.active_plan=receipt;self.perception_mode=True;self.baseline_mode=False;self.path=receipt['path'];self.index=1
                self.stage='APPROACH';self.reason='';self.vcmd=np.zeros(3);self.hold_since=None;self.prev=None;self.start_stamp=int(self.raw_state.timestamp)
                self.event('PERCEPTION_START_ACCEPTED',task=self.task,task_sha256=self.task_hash,plan_version=self.plan_version,command_lease_requires_fresh_evidence=True)
        except Exception as ex:
            transient=str(ex) in ['CURRENT_FRAME_EXPIRED','NO_VALID_ONLINE_DESIGNATED_ESTIMATE','TARGET_NOT_TRACKING','SEMANTIC_EVIDENCE_EXPIRED','EVIDENCE_EXPIRED_DURING_PLAN_REVALIDATION','PLAN_EVIDENCE_TOO_OLD_FOR_HANDOFF','NEWER_EVIDENCE_INVALIDATED_DURING_CHECK']
            if str(ex)=='SELF_NOT_ON_FIXED_HEIGHT_PLANE' and self.service.cfg.get('recovery_height_handoff_wait_v17') and pending['execute'] and self.perception_mode and self.stage=='ABORTED':transient=True
            if transient and 'handoff_deadline_monotonic_s' in pending and time.monotonic()<pending['handoff_deadline_monotonic_s']:
                self.preview_result=dict(status='WAITING_FRESH_REVALIDATION',reason=str(ex),deadline_monotonic_s=pending['handoff_deadline_monotonic_s'])
                if pending.get('last_wait_reason')!=str(ex):
                    self.event('PERCEPTION_PLAN_WAITING_FRESH_FRAME',reason=str(ex),no_motion_issued=True);pending['last_wait_reason']=str(ex)
                return
            self.pending_plan=None
            self.preview_result=dict(status='REJECTED',reason=str(ex));self.reason='PERCEPTION_PLAN_REJECTED_'+str(ex);self.resume_required=False
            self.event('PERCEPTION_PLAN_REJECTED',reason=str(ex),execute_requested=pending['execute'])
    def run(self):
        super().run()
        # The baseline failure tail intentionally does not accept return
        # requests. This development owner must finish airborne recovery.
        if self.stage not in ('FAILED','FAILED_HOVER'):return
        try:
            self.sample()
            if self.state['landed']:
                self.c.armDisarm(False,vehicle_name='DroneA');self.c.enableApiControl(False,vehicle_name='DroneA');return
            if self.service:self.service.stop(wait=False)
            self.hold('SAFETY_RECOVERY_AFTER_RECORDED_FAILURE','STOP_HOVER')
            self.event('SAFETY_RECOVERY_STARTED',original_failure=self.reason)
            self.sample();self.process_request(dict(action='return',actor='mandatory_safety_recovery'))
            deadline=time.monotonic()+200
            while self.stage!='READY' and time.monotonic()<deadline:
                tick=time.monotonic();self.sample();self.last_host=tick
                super().control();time.sleep(max(0,self.task['sample_period_host_s']-(time.monotonic()-tick)))
            if self.stage=='READY':return
            raise ValueError('SAFETY_RETURN_TIMEOUT')
        except Exception as ex:
            self.event('SAFETY_RETURN_BLOCKED',reason=str(ex));self.stage='FAILED_HOVER'
            try:self.hold('MANUAL_RECOVERY_REQUIRED_'+str(ex),'FAILED_HOVER')
            except Exception as rpc:self.event('SAFETY_HOVER_RPC_FAILED',reason=str(rpc))
            # Keep the owner and measured state alive. Do not silently finish
            # an airborne process after its single bounded return attempt.
            print('AIRBORNE_RECOVERY_REQUIRES_ATTENTION',flush=True)
            while True:
                try:
                    self.sample()
                    if self.state['landed']:
                        self.c.armDisarm(False,vehicle_name='DroneA');self.c.enableApiControl(False,vehicle_name='DroneA');self.stage='READY';return
                except Exception as rpc:self.event('RECOVERY_STATE_UNAVAILABLE',reason=type(rpc).__name__)
                time.sleep(.1)
    def finish(self,status):
        if not self.baseline_mode and self.run_dir:
            mission.save(self.run_dir/('result_'+('perception_' if self.perception_mode else 'observation_')+status.lower()+'_'+str(time.time_ns())+'.json'),dict(utc=mission.utc(),status=status,state=self.state,task_sha256=self.task_hash,active_plan=self.active_plan,reason=self.reason))
        super().finish(status)
