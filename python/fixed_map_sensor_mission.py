"""Bounded orchestration over the existing control owner. No scene target input."""
import copy,time
from fixed_map_sensor_language import validate_record,ACTION_STAGES
from fixed_map_perception_task import live_estimate
from fixed_map_perception_geometry import digest_json

OBSERVATION=dict(observation_goal_xyz_m=[30.,-9.,3.1],heading_yaw_deg=0.,selection_basis='declared static side corridor before target sampling; not target truth',stage='observation_preparation_NOT_perception_guidance')

class SensorMission:
    def __init__(self,executor):self.e=executor;self.active=False;self.phase='IDLE';self.binding=None;self.deadline=None;self.requested=False;self.outcome=None;self.last_refusal=None
    def submit(self,record,cfg):
        e=self.e;p=validate_record(record,e.safety,cfg)
        if e.stage not in ACTION_STAGES[p['action']]:raise ValueError('LANGUAGE_ACTION_STAGE_REJECTED')
        if p['action'] in ['stop','return']:
            self.active=False;self.phase='OPERATOR_'+p['action'].upper();e.command(p['action'],'language_sensor_v8');return
        if e.service is None or e.service.cfg['perception_backend']!='yolo_only_v8' or e.service.cfg['target_description']!=p['target_description']:raise ValueError('LANGUAGE_PERCEPTION_TARGET_NOT_BOUND')
        if self.active:raise ValueError('SENSOR_MISSION_ALREADY_ACTIVE')
        if p['action']=='preview':e.plan_perception(False,'language_sensor_preview_v8');return
        if cfg.get('ground_development_only') or not cfg.get('flight_authorized'):raise ValueError('GROUND_PROFILE_FLIGHT_DISABLED')
        self.binding=dict(schema='natural_language_sensor_mission_v8',language_record_sha256=record['record_sha256'],target_generation=e.service.cfg['target_generation'],description_sha256=__import__('hashlib').sha256(p['target_description'].encode()).hexdigest(),session_id=e.service.session_id if hasattr(e.service,'session_id') else e.service.output.name,runtime_scene_version=e.service.cfg.get('runtime_scene_version'))
        e.event('LANGUAGE_SENSOR_MISSION_SUBMITTED',binding=self.binding,proposal=p,observation=OBSERVATION)
        self.active=True;self.phase='OBSERVATION';self.deadline=time.monotonic()+220;self.requested=False
        e.prepare_observation(copy.deepcopy(OBSERVATION),'language_sensor_observation_v8')
    def tick(self):
        if not self.active:return
        e=self.e;now=time.monotonic()
        if e.stage in ['FAILED','FAILED_HOVER']:self.phase='CONTROL_FAILURE';self.active=False;return
        if self.phase=='RETURN':
            if e.stage=='READY':self.active=False;self.phase='LANDED_RELEASED';e.event('LANGUAGE_SENSOR_MISSION_FINISHED',binding=self.binding,outcome=self.outcome)
            return
        if now>self.deadline:
            self.outcome='FAILED_'+self.phase+'_DEADLINE';self.phase='RETURN';e.event('LANGUAGE_SENSOR_MISSION_DEADLINE',binding=self.binding,outcome=self.outcome,last_refusal=self.last_refusal);e.command('stop','sensor_mission_timeout');e.command('return','sensor_mission_timeout');return
        if self.phase=='OBSERVATION' and e.stage=='ARRIVED':self.phase='CONFIRM';self.deadline=now+30
        if self.phase=='CONFIRM':
            try:
                estimate=live_estimate(e.service.latest,e.service.cfg,now)
                if any(estimate.get(k)!=self.binding[k] for k in ['session_id','target_generation','description_sha256','runtime_scene_version']):raise ValueError('LANGUAGE_TARGET_CONTEXT_CHANGED')
            except (ValueError,KeyError,TypeError) as ex:
                refusal=dict(frame_id=e.service.latest.get('frame_id'),reason=str(ex))
                if refusal!=self.last_refusal:e.event('LANGUAGE_SENSOR_CONFIRM_REFUSED',binding=self.binding,refusal=refusal);self.last_refusal=refusal
                return
            self.phase='APPROACH';self.deadline=now+120;self.requested=True;e.plan_perception(True,'language_sensor_approach_v8')
        elif self.phase=='APPROACH':
            if e.preview_result and e.preview_result.get('status')=='REJECTED':
                self.outcome='FAILED_PLAN_'+e.preview_result['reason'];self.phase='RETURN';e.event('LANGUAGE_SENSOR_PLAN_FAILED',binding=self.binding,reason=e.preview_result['reason']);e.command('return','sensor_mission_plan_failure')
            elif e.perception_mode and e.stage=='ARRIVED':
                self.outcome='APPROACH_HOVER_PASSED';self.phase='RETURN';e.event('LANGUAGE_SENSOR_APPROACH_HOVER_PASSED',binding=self.binding,active_plan=e.active_plan);e.command('return','sensor_mission_complete')
    def status(self):return dict(active=self.active,phase=self.phase,binding=self.binding,deadline_monotonic_s=self.deadline,outcome=self.outcome,last_refusal=self.last_refusal)
