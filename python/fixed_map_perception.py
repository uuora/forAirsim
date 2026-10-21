"""Read-only perception service: bounded capture/inference/network workers.

Only RGB/depth, their response poses, self-vehicle pose and a declared target
description enter inference. No target actor, instance mask or target truth API.
"""
import argparse, copy, hashlib, json, math, queue, threading, time, sys
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import cv2
from camera_geometry import scene_intrinsics
from t1_pose_geometry import quaternion_rotation_xyzw
from t1_semantic_perception import BoundedSemanticPolicy, PolicyConfig, YoloVlxBackend
from t1_temporal_identity import TemporalIdentityTracker, TrackerConfig
from fixed_map_perception_geometry import surface_from_box, validate_pair, sanitize_obstacles, make_proposal, preview_task, digest_json, perception_safety_parameters

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'reports/fixed_map_flight_20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def utc():return datetime.now(timezone.utc).isoformat(timespec='milliseconds')
def xyz(v):return [float(v.x_val),float(v.y_val),float(v.z_val)]
def quat(q):return [float(q.x_val),float(q.y_val),float(q.z_val),float(q.w_val)]
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    # Windows readers/scanners can momentarily deny rename. Same bounded
    # 100ms policy as the control recorder; persistent failures remain errors.
    for attempt in range(21):
        try:tmp.replace(p);return
        except PermissionError:
            if attempt==20:raise
            time.sleep(.005)

def restore_inference_threads(config, torch_module):
    """Predictor initialization may reset CPU threads; restore the declared cap."""
    requested=config.get('torch_threads',2)
    if type(requested) is not int or not 1<=requested<=16:raise ValueError('INVALID_CPU_THREAD_BUDGET')
    before=torch_module.get_num_threads();torch_module.set_num_threads(requested)
    actual=torch_module.get_num_threads()
    if actual!=requested:raise ValueError('CPU_THREAD_BUDGET_NOT_APPLIED')
    return dict(requested_threads=requested,before_restoration_threads=before,actual_threads=actual,enforced_after_predictor_warmup=True)

def seal_current_frame_report(report,config,now):
    """Bind final validity and its recorded timestamp to the same check."""
    report['processed_monotonic_s']=now
    errors=validate_pair(report['capture'],config,now)
    estimate=report['position_estimate']
    if estimate.get('valid') and now>estimate.get('semantic_review_deadline_monotonic_s',-1):errors.append('SEMANTIC_EVIDENCE_EXPIRED')
    if not report.get('valid') or errors:
        estimate['valid']=False;report['valid']=False
    if errors:
        estimate['reason']=errors[0]
        report['rejection_reasons']=list(dict.fromkeys(report['rejection_reasons']+errors))
        report['planning_preview']=dict(status='REJECTED',reason=errors[0],flight_authorized=False)
    report['validity_seal']=dict(version='same_final_timestamp_v19',checked_monotonic_s=now,source_received_monotonic_s=report['capture']['received_monotonic_s'],source_age_s=now-report['capture']['received_monotonic_s'],valid=report['valid'])
    return report


def expired_queue_report(meta,config,now):
    """Reject already-expired queued input before expensive inference."""
    age=now-meta['received_monotonic_s']
    if 0<=age<=config['current_max_age_s']:return None
    estimate={k:meta[k] for k in ['mode','session_id','target_generation','description_sha256','frame_id','image_sha256','received_monotonic_s']}
    estimate.update(valid=False,reason='CURRENT_FRAME_EXPIRED',target_state='LOST',runtime_scene_version=meta.get('runtime_scene_version'))
    row=dict(frame_id=meta['frame_id'],mode='online',valid=False,capture=meta,position_estimate=estimate,processed_monotonic_s=now,inference_wall_s=0.,queue_rejected_before_inference=True,rejection_reasons=['CURRENT_FRAME_EXPIRED'],planning_preview=dict(status='REJECTED',reason='CURRENT_FRAME_EXPIRED',flight_authorized=False),flight_authorized=False)
    return seal_current_frame_report(row,config,now)

class PerceptionService:
    def __init__(self, output, config=None, preview_interface=None, publication_callback=None):
        self.output=Path(output).resolve()
        if self.output==HIST or self.output.is_relative_to(HIST):raise ValueError('HISTORICAL_OUTPUT_IS_READ_ONLY')
        self.output.mkdir(parents=True,exist_ok=False)
        self.cfg=copy.deepcopy(config or read(ROOT/'configs/fixed_map_perception_v1.json'))
        if self.cfg.get('math_thread_budget_v20'):cv2.setNumThreads(1)
        from fixed_map_target_profiles import DESCRIPTIONS
        if self.cfg.get('perception_backend')=='yolo_only_v8' and self.cfg.get('target_description') not in DESCRIPTIONS:
            raise ValueError('UNSUPPORTED_YOLO_TARGET_DESCRIPTION_OR_ENCODING')
        self.publication_callback=None
        self.session_id=self.output.name;self.latest={};self.done=threading.Event();self.model_ready=threading.Event();self.warm_ready=threading.Event();self.queue=queue.Queue(maxsize=self.cfg['queue_capacity']);self.lock=threading.Lock();self.preview_interface=preview_interface
        self.capture_count=self.process_count=self.drop_count=0;self.previous_stamps=None;self.warmed=False
        self.task=perception_safety_parameters(read(HIST/'fixed_version/task.json'))
        sanitizer=sanitize_obstacles
        if self.cfg.get('static_geometry_version')=='v2':
            from fixed_map_perception_execution import static_geometry_v2
            sanitizer=static_geometry_v2
        parts,removed=sanitizer(read(HIST/'fixed_version/editor_audit/planning_geometry.json')['primitives'])
        from fixed_map_mission import Geometry
        self.geometry=Geometry(parts,self.task['envelope_radius_config_m'])
        save(self.output/'static_obstacles.json',dict(primitives=parts,removed_semantic_ids=removed,scope='obstacles_only_no_target_coordinates_or_baseline_goal'))
        self.settings=read(HIST/'fixed_version/snapshot/configs/enclosed_uav_lab_manual_settings.json')
        self.self_reference=read(ROOT/self.cfg['self_calibration'])
        self.world_origin_xyz=self.self_reference['global_origin_xyz_m']
        self.step1=read(ROOT/self.cfg['step1_receipt'])
        if self.step1['status']!='PASS_STEP1_ENTER_STEP2':raise ValueError('STEP1_NOT_PASSED_GROUND_DEVELOPMENT_ONLY')
        self.manifest=dict(started_utc=utc(),config=self.cfg,step1_reference=str(ROOT/self.cfg['step1_receipt']),step1_sha256=sha(ROOT/self.cfg['step1_receipt']),frozen_manifest_sha256=sha(HIST/'fixed_version/manifest.json'),settings_sha256=sha(HIST/'fixed_version/snapshot/configs/enclosed_uav_lab_manual_settings.json'),self_origin_source=str(ROOT/self.cfg['self_calibration']),self_origin_scope='ground-anchored world frame and SELF vehicle conversion only; no target transforms supplied',source_sha256={n:sha(ROOT/'python'/n) for n in ['fixed_map_perception.py','fixed_map_perception_geometry.py','specialist_balloon_detector.py','camera_geometry.py','t1_pose_geometry.py','t1_semantic_perception.py','t1_temporal_identity.py','vlx_api_client.py','fixed_map_mission.py','fixed_map_dashboard.py']},flight_authorized=False,model_training=False,ground_only=True)
        self.manifest.update(ground_only=self.cfg.get('collection_context','ground')=='ground',collection_context=self.cfg.get('collection_context','ground'),observer_issues_motion_commands=False)
        for n in ['fixed_map_perception_execution.py','fixed_map_perception_task.py','fixed_map_perception_process.py','fixed_map_delayed_identity.py','t1_delayed_semantic_bridge.py','t1_source_attribute_guard.py']:
            self.manifest['source_sha256'][n]=sha(ROOT/'python'/n)
        save(self.output/'version_manifest.json',self.manifest)
        self.publication_callback=publication_callback
    @property
    def latest(self):return self._latest
    @latest.setter
    def latest(self,value):
        self._latest=value
        if self.publication_callback is not None:self.publication_callback(self)
    def event(self,kind,**values):
        row=dict(utc=utc(),host_monotonic_s=time.monotonic(),kind=kind,session_id=self.session_id,**values)
        with self.lock:
            with (self.output/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    def start(self):
        self.threads=[threading.Thread(target=self._capture,daemon=True,name='fixed-perception-capture'),threading.Thread(target=self._infer,daemon=True,name='fixed-perception-inference')]
        for t in self.threads:t.start()
    def stop(self,wait=True):
        self.done.set()
        if wait:
            for t in self.threads:t.join(timeout=6)
        self.event('SERVICE_STOP_REQUESTED',threads_alive=[t.name for t in self.threads if t.is_alive()])
    def capture_once(self,c,sequence):
        if self.cfg.get('scene_epoch_required'):
            from fixed_map_scene_binding import engine_identity_alive
            if not engine_identity_alive(self.cfg.get('runtime_engine_identity')):raise ValueError('SCENE_ENGINE_LIFETIME_CHANGED')
        from fixed_map_storage import check_output_storage
        storage=check_output_storage(self.output,'active')
        if storage['status']!='PASS':
            self.done.set()
            self._latest=dict(valid=False,rejection_reasons=['CAPTURE_STORAGE_RESERVE_'+storage['status']],storage=storage)
            try:
                self.event('CAPTURE_STORAGE_RESERVE_STOP',storage=storage,no_frame_folder_created=True)
                save(self.output/'storage_capture_guard.json',dict(status='STOPPED',storage=storage,no_frame_folder_created=True))
            except OSError as ex:print('CAPTURE_STORAGE_GUARD_RECEIPT_WRITE_FAILED',type(ex).__name__,str(ex),flush=True)
            return None
        import airsim
        frame_id=f'frame-{sequence:06d}';folder=self.output/'frames'/frame_id;folder.mkdir(parents=True)
        body0=c.getMultirotorState(vehicle_name='DroneA');selfpose=c.simGetObjectPose('DroneA')
        translation_xyz=np.array(self.world_origin_xyz)+np.array(xyz(selfpose.position))*[1,1,-1]-np.array(xyz(body0.kinematics_estimated.position))*[1,1,-1]
        if self.cfg.get('camera_reference_version')=='frozen_self_origin_v2':
            # Constant spawn translation avoids mixing moving self state and
            # object readbacks at different times. No scene target query.
            translation_xyz=np.array(self.self_reference['DroneA_local_origin_xyz_m'])
        translation_ned=translation_xyz*[1,1,-1]
        info=c.simGetCameraInfo('front_center',vehicle_name='DroneA')
        started=time.monotonic();sent_utc=utc();sent_ns=time.time_ns()
        responses=c.simGetImages([airsim.ImageRequest('front_center',airsim.ImageType.Scene,False,True),airsim.ImageRequest('front_center',airsim.ImageType.DepthPerspective,True,False)],vehicle_name='DroneA')
        received=time.monotonic();received_utc=utc();received_ns=time.time_ns()
        body_after=c.getMultirotorState(vehicle_name='DroneA')
        if len(responses)!=2:raise ValueError('EXPECTED_TWO_IMAGE_RESPONSES')
        rgb,dep=responses;encoded=bytes(rgb.image_data_uint8)
        if encoded and self.cfg.get('image_pipeline_version')=='capture_decode_reuse_v17':
            from fixed_map_frame_image import decode_bound_rgb
            try:image=decode_bound_rgb(encoded,hashlib.sha256(encoded).hexdigest())
            except ValueError:image=None
        else:image=cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR) if encoded else None
        ranges=np.asarray(dep.image_data_float,np.float32);errors=[]
        if rgb.message or dep.message:errors.append('CAMERA_RESPONSE_ERROR')
        if image is None:errors.append('EMPTY_OR_UNDECODABLE_RGB')
        if dep.width<=0 or dep.height<=0 or ranges.size!=dep.width*dep.height:errors.append('EMPTY_OR_INVALID_DEPTH_DIMENSIONS');ranges=np.empty((0,0),np.float32)
        else:ranges=ranges.reshape(dep.height,dep.width)
        (folder/'rgb.png').write_bytes(encoded);np.save(folder/'depth_ray_range_m.npy',ranges,allow_pickle=False)
        cams=self.settings['Vehicles']['DroneA']['Cameras']['front_center'];ds=next(s for s in cams['CaptureSettings'] if s['ImageType']==2)
        rs=next(s for s in cams['CaptureSettings'] if s['ImageType']==0)
        if [rgb.width,rgb.height]!=[rs['Width'],rs['Height']] or [dep.width,dep.height]!=[ds['Width'],ds['Height']]:errors.append('FROZEN_CAPTURE_SIZE_MISMATCH')
        rk=scene_intrinsics(info,max(rgb.width,1),max(rgb.height,1));fx=dep.width/(2*math.tan(math.radians(ds['FOV_Degrees']/2)))
        if abs(rk['horizontal_fov_deg']-rs['FOV_Degrees'])>.05:errors.append('RGB_PROJECTION_FROZEN_HFOV_MISMATCH')
        def pose(r):return dict(position_local_ned_m=xyz(r.camera_position),orientation_xyzw=quat(r.camera_orientation),world_from_local_translation_ned_m=translation_ned.tolist())
        stamps=[int(rgb.time_stamp),int(dep.time_stamp)];advanced=self.previous_stamps is None or all(a>b for a,b in zip(stamps,self.previous_stamps));self.previous_stamps=stamps
        observed_mount=quaternion_rotation_xyzw(quat(body0.kinematics_estimated.orientation)).T@(np.array(xyz(rgb.camera_position))-xyz(body0.kinematics_estimated.position))
        expected_mount=np.array([cams['X'],cams['Y'],cams['Z']]);mount_error=float(np.linalg.norm(observed_mount-expected_mount))
        mount_bound=.03
        if self.cfg.get('camera_reference_version')=='frozen_self_origin_v2':
            # A readback pose and preceding body state are not simultaneous.
            # Bound this diagnostic by measured motion; localization uses the
            # actual response pose, never this inferred mount.
            dt=abs(int(body_after.timestamp)-int(body0.timestamp))/1e9
            angular=np.linalg.norm(xyz(body0.kinematics_estimated.angular_velocity))
            mount_bound+=self.task['max_speed_measured_m_s']*dt+angular*dt*np.linalg.norm(expected_mount)
            observed_translation=np.array(self.world_origin_xyz)+np.array(xyz(selfpose.position))*[1,1,-1]-np.array(xyz(body0.kinematics_estimated.position))*[1,1,-1]
            if np.linalg.norm(observed_translation-translation_xyz)>.03+self.task['max_speed_measured_m_s']*dt:errors.append('SELF_WORLD_REFERENCE_MISMATCH')
        if mount_error>mount_bound:errors.append('SELF_CAMERA_MOUNT_MISMATCH')
        meta=dict(mode='online',session_id=self.session_id,target_generation=self.cfg['target_generation'],description_sha256=hashlib.sha256(self.cfg['target_description'].encode()).hexdigest(),frame_id=frame_id,frame_sequence=sequence,image_sha256=hashlib.sha256(encoded).hexdigest(),depth_sha256=sha(folder/'depth_ray_range_m.npy'),rgb_path=str(folder/'rgb.png'),depth_path=str(folder/'depth_ray_range_m.npy'),rgb_size_wh=[rgb.width,rgb.height],depth_size_wh=[dep.width,dep.height],rgb_timestamp_ns=stamps[0],depth_timestamp_ns=stamps[1],rgb_depth_delta_ms=abs(stamps[0]-stamps[1])/1e6,timestamp_scope='renderer readback marks; NOT exposure timestamps or hardware synchronization',synchronization_verified=False,requested_utc=sent_utc,received_utc=received_utc,requested_host_time_ns=sent_ns,received_host_time_ns=received_ns,requested_monotonic_s=started,received_monotonic_s=received,capture_wall_s=received-started,rgb_K=rk['K_nominal'],rgb_intrinsics=rk,depth_K=[[fx,0,dep.width/2],[0,fx,dep.height/2],[0,0,1]],depth_intrinsics_source='frozen_DepthPerspective_HFOV_and_capture_dimensions_square_pixels',rgb_pose=pose(rgb),depth_pose=pose(dep),pose_source='ImageResponse.camera_position/camera_orientation using self-vehicle local-to-world translation',observed_mount_body_ned_m=observed_mount.tolist(),expected_mount_body_ned_m=expected_mount.tolist(),mount_error_m=mount_error,timestamps_advanced=advanced,capture_errors=errors,start_world_xyz_m=(np.array(self.world_origin_xyz)+np.array(xyz(selfpose.position))*[1,1,-1]).tolist(),flight_commands_issued=False)
        meta.update(mount_diagnostic_bound_m=float(mount_bound),camera_reference_version=self.cfg.get('camera_reference_version','v1'),response_pose_is_localization_source=True,runtime_scene_version=self.cfg.get('runtime_scene_version'),runtime_engine_identity=self.cfg.get('runtime_engine_identity'))
        # Both renderer and MultirotorState use ClockFactory::nowNanos in this
        # frozen AirSim build. ClockSpeed=1: reject old readback marks without
        # claiming they measure exposure synchronization.
        meta.update(state_before_timestamp_ns=int(body0.timestamp),state_after_timestamp_ns=int(body_after.timestamp),actual_aircraft_landed=body_after.landed_state==airsim.LandedState.Landed,observer_issues_motion_commands=False,render_age_sim_s=(int(body_after.timestamp)-min(stamps))/1e9,render_age_clock_source='shared AirSim ClockFactory; frozen ClockSpeed=1')
        save(folder/'capture.json',meta)
        return meta,ranges
    def _capture(self):
        if self.cfg.get('capture_after_model_load'):
            deadline=time.monotonic()+35
            while not self.model_ready.is_set() and not self.done.is_set() and time.monotonic()<deadline:self.done.wait(.05)
            if not self.model_ready.is_set():return
        import airsim
        c=airsim.MultirotorClient(port=41454,timeout_value=3)
        try:
            while not self.done.is_set():
                if not self.model_ready.is_set() or (self.capture_count>0 and not self.warm_ready.is_set()):
                    self.done.wait(.1);continue
                started=time.monotonic();self.capture_count+=1
                try:
                    item=self.capture_once(c,self.capture_count)
                    if item is None:break
                    item[0]['capture_archive_ready_monotonic_s']=time.monotonic()
                    if self.queue.full():
                        old=self.queue.get_nowait();self.drop_count+=1;self.event('FRAME_REJECTED',frame_id=old[0]['frame_id'],reason='BOUNDED_QUEUE_REPLACED')
                    self.queue.put_nowait(item)
                    with self.lock:
                        with (self.output/'frame_index.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(frame_id=item[0]['frame_id'],capture=str(self.output/'frames'/item[0]['frame_id']/'capture.json'),rgb=item[0]['rgb_path'],depth=item[0]['depth_path'],image_sha256=item[0]['image_sha256'],depth_sha256=item[0]['depth_sha256']))+'\n')
                except Exception as ex:self.event('CAPTURE_REJECTED',frame_sequence=self.capture_count,reason=type(ex).__name__,detail=str(ex));self.latest=dict(valid=False,rejection_reasons=['CAPTURE_FAILED_'+type(ex).__name__])
                self.done.wait(max(0,self.cfg['capture_period_s']-(time.monotonic()-started)))
        finally:self.event('CAPTURE_WORKER_CLOSED')
    def _backend(self,request):
        self.active_vlx_binding=request.binding()
        self.event('VLX_REQUEST',binding=request.binding(),description=request.description,source_rgb_path=str(self.output/'frames'/request.frame.frame_id/'rgb.png'),request_host_time_ns=time.time_ns(),request_mode='image_text',proposals_transmitted=False)
        try:
            reply=self.backend(request)
            self.event('VLX_REPLY',binding=reply.binding,completed_monotonic_s=reply.completed_monotonic_s,result=reply.result,reply_binding_provenance='CLIENT_SIDE_BINDING_NOT_SERVER_ECHO')
        except Exception as ex:
            self.event('VLX_REPLY_REJECTED',binding=request.binding(),reason='BACKEND_'+type(ex).__name__)
            raise RuntimeError('VLX_BACKEND_FAILURE_'+type(ex).__name__) from None
        if self.cfg.get('source_attribute_guard'):
            from t1_source_attribute_guard import verify_source_attribute
            audit=verify_source_attribute(request,reply)
            self.event('VLX_SOURCE_ATTRIBUTE_AUDIT',binding=request.binding(),audit=audit)
            if audit['status']!='ACCEPTED':
                reason='SOURCE_RGB_ATTRIBUTE_NOT_VERIFIED_'+audit['reason']
                self.event('VLX_REPLY_REJECTED',binding=request.binding(),reason=reason)
                raise RuntimeError(reason)
        return reply
    def _infer(self):
        policy=None
        try:
            initialization_start=time.monotonic();self.event('MODEL_INITIALIZATION_STARTED')
            from train_t1_balloon_yolo import configure_offline_runtime
            if 'ultralytics' not in sys.modules:configure_offline_runtime()
            import torch
            torch.set_num_threads(self.cfg.get('torch_threads',2))
            if self.cfg.get('cuda_runtime_guard'):
                torch.backends.cudnn.benchmark=False
                torch.backends.cudnn.enabled=self.cfg.get('cuda_cudnn_enabled',False)
                torch.backends.cudnn.deterministic=self.cfg.get('cuda_cudnn_deterministic',False)
                torch.backends.cudnn.allow_tf32=False
                torch.backends.cuda.matmul.allow_tf32=False
                torch.cuda.set_per_process_memory_fraction(.5,0)
            from specialist_balloon_detector import SpecialistBalloonDetector
            dc=read(ROOT/self.cfg['detector_config'])
            for field in ['weights','training_receipt']:
                if sha(dc[field])!=dc[field+'_sha256']:raise ValueError('FROZEN_SPECIALIST_HASH_MISMATCH')
            detector=SpecialistBalloonDetector(dc['weights'],confidence=dc['confidence'],imgsz=dc['imgsz'],device=dc['device'],nms_iou=dc['nms_iou'])
            if self.cfg.get('detector_pipeline_version')=='adaptive_full640_960_v18':
                from fixed_map_conditional_scale import ConditionalScaleBalloonDetector
                detector=ConditionalScaleBalloonDetector(detector,description=self.cfg['target_description'])
                self.manifest['source_sha256']['fixed_map_conditional_scale.py']=sha(ROOT/'python/fixed_map_conditional_scale.py')
            if self.cfg.get('detector_pipeline_version')=='full_image_dual_scale_v7':
                from fixed_map_multiscale_detector import MultiScaleBalloonDetector
                detector=MultiScaleBalloonDetector(detector)
            yolo_only=self.cfg.get('perception_backend')=='yolo_only_v8'
            if yolo_only:
                from fixed_map_yolo_identity import YoloImageBackend,YoloAssociation,AssociationConfig,DisabledReview
                self.backend=YoloImageBackend(detector)
                tc=AssociationConfig(confirmation_frames=self.cfg['association_confirmation_frames'],maximum_gap_s=self.cfg['current_max_age_s'])
                pc=tc;policy=DisabledReview();tracker=YoloAssociation(self.session_id,self.cfg['target_generation'],self.cfg['target_description'],tc)
                self.manifest.update(detector_config=dc,detector_config_sha256=sha(ROOT/self.cfg['detector_config']),policy_config=asdict(pc),tracker_config=asdict(tc),vlx_available=False,vlx_enabled=False,perception_backend='yolo_only_v8',torch_version=torch.__version__,warmup_scope='real first image discarded as evidence; no training')
                if self.cfg.get('math_thread_budget_v20'):
                    import os
                    self.manifest['math_thread_receipt']=dict(environment={key:os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']},opencv_threads=cv2.getNumThreads(),scope='Per-process budgets; control/sensor formulas and source timestamps unchanged')
                self.manifest['source_sha256']['fixed_map_yolo_identity.py']=sha(ROOT/'python/fixed_map_yolo_identity.py')
            else:
                from dotenv import load_dotenv
                from vlx_api_client import VLXSeekAPIClient
                load_dotenv(ROOT/'.env');vlx_available=True
                try:
                    client=VLXSeekAPIClient(timeout=self.cfg['vlx_http_timeout_s'],max_retries=1)
                    original_upload=client.upload_image;original_convert=client._convert_response
                    def bound_upload(image,format='PNG'):
                        import base64
                        url=original_upload(image,format='PNG');data=base64.b64decode(url.split(',',1)[1]);binding=self.active_vlx_binding
                        dest=self.output/'vlx_transport'/(binding['request_id']+'.png');dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
                        self.event('VLX_TRANSPORT_IMAGE',binding=binding,transmitted_image_sha256=hashlib.sha256(data).hexdigest(),path=str(dest),format='lossless PNG',source_image_sha256=binding['image_sha256'])
                        return url
                    def bound_convert(response,boxes):
                        self.event('VLX_RAW_REPLY',binding=self.active_vlx_binding,raw_response=response)
                        return original_convert(response,boxes)
                    client.upload_image=bound_upload;client._convert_response=bound_convert
                except ValueError:
                    vlx_available=False
                    class MissingClient:
                        request_mode='image_text';proposals_transmitted=False
                        def ground(self,*a,**k):raise RuntimeError('VLX_CONFIGURATION_UNAVAILABLE')
                    client=MissingClient()
                self.backend=YoloVlxBackend(detector=detector,vlx_client=client)
                pc=PolicyConfig(response_timeout_s=self.cfg['vlx_response_timeout_s']);tc=TrackerConfig()
                if self.cfg.get('semantic_policy_version')=='delayed_chain_attributes_v3':
                    from t1_delayed_semantic_bridge import DelayedBridgeConfig,DelayedSemanticPolicy,DelayedChainIdentityTracker
                    pc=DelayedBridgeConfig(response_timeout_s=self.cfg['vlx_response_timeout_s'],max_source_age_s=self.cfg['semantic_source_max_age_s'])
                    policy=DelayedSemanticPolicy(self._backend,session_id=self.session_id,target_generation=self.cfg['target_generation'],description=self.cfg['target_description'],config=pc)
                    from fixed_map_delayed_identity import ReviewHandoffTracker
                    tracker=ReviewHandoffTracker(self.session_id,self.cfg['target_generation'],self.cfg['target_description'])
                    tc=tracker.config
                else:
                    policy=BoundedSemanticPolicy(self._backend,session_id=self.session_id,target_generation=self.cfg['target_generation'],description=self.cfg['target_description'],config=pc)
                    tracker=TemporalIdentityTracker(self.session_id,self.cfg['target_generation'],self.cfg['target_description'],tc)
                self.manifest.update(detector_config=dc,detector_config_sha256=sha(ROOT/self.cfg['detector_config']),policy_config=asdict(pc),tracker_config=asdict(tc),review_period_s=self.cfg['vlx_review_period_s'],vlx_available=vlx_available,vlx_model=getattr(client,'model',None),torch_version=torch.__version__,warmup_scope='real first image discarded as evidence; no training')
            save(self.output/'version_manifest.json',self.manifest);last_request=None;last_frame=None;expiry_recorded=False
            self.event('MODEL_READY',initialization_wall_s=time.monotonic()-initialization_start);self.model_ready.set()
            while not self.done.is_set():
                try:meta,depth=self.queue.get(timeout=.2)
                except queue.Empty:
                    if last_frame is not None and not expiry_recorded and time.monotonic()-last_frame.readback_receipt_monotonic_s>self.cfg['current_max_age_s']:
                        lost=tracker.update(last_frame,now=time.monotonic())
                        expired=copy.deepcopy(self.latest);expired.update(valid=False,temporal_identity=lost,rejection_reasons=['NO_NEW_FRAME_CURRENT_EVIDENCE_EXPIRED'])
                        expired.setdefault('position_estimate',{}).update(valid=False,target_state=lost['state'],reason='CURRENT_FRAME_EXPIRED')
                        expired['planning_preview']=dict(status='REJECTED',reason='CURRENT_FRAME_EXPIRED',flight_authorized=False)
                        self.latest=expired;expiry_recorded=True;self.event('EVIDENCE_EXPIRED',frame_id=last_frame.frame_id,temporal_identity=lost)
                    continue
                folder=self.output/'frames'/meta['frame_id']
                if 'EMPTY_OR_UNDECODABLE_RGB' in meta['capture_errors']:
                    invalid=dict(frame_id=meta['frame_id'],mode='online',valid=False,capture=meta,rejection_reasons=meta['capture_errors'],position_estimate=dict(valid=False,reason='EMPTY_OR_UNDECODABLE_RGB'),planning_preview=dict(status='REJECTED',reason='INVALID_CAPTURE',flight_authorized=False))
                    save(folder/'perception.json',invalid);self.latest=invalid;self.event('FRAME_REJECTED',frame_id=meta['frame_id'],reason='EMPTY_OR_UNDECODABLE_RGB');continue
                if not self.warmed:
                    receipt=self.backend.warmup(meta['rgb_path'])
                    if self.cfg.get('runtime_thread_policy')=='restore_after_predictor_warmup_v13':
                        threads=restore_inference_threads(self.cfg,torch);self.manifest['cpu_thread_receipt']=threads
                        save(self.output/'version_manifest.json',self.manifest);self.event('CPU_THREAD_BUDGET_RESTORED',receipt=threads)
                    self.event('YOLO_WARMUP',receipt=receipt);self.warmed=True;self.warm_ready.set()
                    self.event('FRAME_REJECTED',frame_id=meta['frame_id'],reason='WARMUP_FRAME_NOT_EVIDENCE');continue
                if self.cfg.get('runtime_thread_policy')=='restore_after_predictor_warmup_v13' and torch.get_num_threads()!=self.cfg['torch_threads']:raise ValueError('CPU_THREAD_BUDGET_CHANGED_AFTER_WARMUP')
                inference_start=time.monotonic()
                if self.cfg.get('reject_expired_queue_v20'):
                    rejected=expired_queue_report(meta,self.cfg,inference_start)
                    if rejected is not None:
                        self.process_count+=1;self.latest=rejected;save(folder/'perception.json',rejected)
                        self.event('FRAME_REJECTED',frame_id=meta['frame_id'],reason='CURRENT_FRAME_EXPIRED_BEFORE_INFERENCE',original_receipt_retained=True)
                        continue
                frame=self.backend.detect_frame(meta['rgb_path'],session_id=self.session_id,target_generation=self.cfg['target_generation'],frame_id=meta['frame_id'],frame_sequence=meta['frame_sequence'],readback_receipt_monotonic_s=meta['received_monotonic_s'],expected_image_sha256=meta['image_sha256'])
                last_frame=frame;expiry_recorded=False
                now=time.monotonic();errors=validate_pair(meta,self.cfg,now)
                attribute_started=time.monotonic()
                semantic=policy.poll(frame,now=now);tracking=tracker.update(frame,semantic_decision=semantic if semantic['status']=='SEMANTIC_LOCAL_ASSOCIATION_VALID' else None,now=time.monotonic())
                attributes=None
                if self.cfg.get('current_attribute_guard'):
                    from fixed_map_perception_execution import current_attribute_audit
                    attributes=tracking['attribute_audit'] if yolo_only else current_attribute_audit(frame,self.cfg['target_description'],tracking.get('selected_candidate_id'))
                request=dict(status='NOT_REQUESTED',reason='REVIEW_NOT_DUE')
                if not yolo_only and not errors and (last_request is None or now-last_request>=self.cfg['vlx_review_period_s']):
                    request=policy.request(frame,now=time.monotonic())
                    if request['status']=='SUBMITTED':last_request=now
                candidate_positions=[]
                depth_started=time.monotonic()
                if not errors:
                    for cand in frame.candidates:
                        try:position=surface_from_box(cand.bbox_xyxy_pixels,depth,meta,self.cfg,time.monotonic())
                        except ValueError as ex:position=dict(valid=False,reason=str(ex))
                        candidate_positions.append(dict(candidate_id=cand.candidate_id,position=position,designated_instance=False))
                if attributes is not None and attributes['status']!='ACCEPTED':errors.append(attributes['reason'])
                selected=next((x for x in candidate_positions if x['candidate_id']==tracking.get('selected_candidate_id')),None)
                estimate={k:meta[k] for k in ['mode','session_id','target_generation','description_sha256','frame_id','image_sha256','received_monotonic_s']}
                estimate['runtime_scene_version']=meta.get('runtime_scene_version')
                if yolo_only:estimate['perception_backend']='yolo_only_v8'
                estimate.update(valid=False,target_state=tracking['state'],reason=tracking['reason'],semantic_review_deadline_monotonic_s=tracking.get('semantic_review_deadline_monotonic_s'),meaning='visible_surface_point_not_center',flight_authorized=False)
                if not errors and tracking['state']=='TRACKING' and selected and selected['position']['valid']:
                    estimate.update(valid=True,surface_world_xyz_m=selected['position']['surface_world_xyz_m'],track_id=tracking['track_id'],semantic_anchor=tracking['semantic_anchor'],surface=selected['position'])
                else:
                    errors.append('NO_DESIGNATED_INSTANCE_CONFIRMATION' if tracking['state']!='TRACKING' else 'SELECTED_DEPTH_INVALID')
                final_pair_errors=validate_pair(meta,self.cfg,time.monotonic())
                if final_pair_errors:
                    errors.extend(final_pair_errors);estimate.update(valid=False,reason=final_pair_errors[0])
                preview=dict(status='REJECTED',reason='NO_VALID_DESIGNATED_TARGET',flight_authorized=False)
                if estimate['valid']:
                    try:
                        proposal=make_proposal(estimate,meta['start_world_xyz_m'],self.task,self.cfg,time.monotonic())
                        if self.cfg.get('preview_mode')=='external_guarded_async':preview=dict(status='VALID_ESTIMATE_AWAITING_EXPLICIT_PREVIEW',task=proposal,flight_authorized=False)
                        elif self.preview_interface:preview=self.preview_interface(proposal,estimate,meta['start_world_xyz_m'],self.geometry,self.cfg,time.monotonic())
                        else:preview=preview_task(proposal,estimate,meta['start_world_xyz_m'],self.geometry,self.task,self.cfg,time.monotonic())
                    except ValueError as ex:preview.update(reason=str(ex));errors.append('PREVIEW_'+str(ex))
                if not frame.candidates:errors.append('NO_YOLO_BALLOON_CANDIDATE')
                if not yolo_only and semantic['status'] not in ['SEMANTIC_LOCAL_ASSOCIATION_VALID','PENDING']:errors.append('VLX_'+semantic['reason'])
                report=dict(frame_id=frame.frame_id,mode='online',semantic_frame=frame.to_dict(),capture=meta,inference_wall_s=now-inference_start,candidate_positions=candidate_positions,semantic_poll=semantic,semantic_request=request,temporal_identity=tracking,position_estimate=estimate,planning_preview=preview,rejection_reasons=list(dict.fromkeys(errors)),valid=estimate['valid'],processed_monotonic_s=time.monotonic(),flight_authorized=False)
                if attributes is not None:report['current_attribute_audit']=attributes
                report['processing_timings']=dict(inference_started_monotonic_s=inference_start,archive_to_inference_s=inference_start-meta.get('capture_archive_ready_monotonic_s',meta['received_monotonic_s']),receipt_to_capture_archive_s=meta.get('capture_archive_ready_monotonic_s',meta['received_monotonic_s'])-meta['received_monotonic_s'],attribute_stage_s=depth_started-attribute_started,depth_and_decision_stage_s=report['processed_monotonic_s']-depth_started,image_pipeline_version=self.cfg.get('image_pipeline_version','uncached_v9'))
                # Raw RGB/depth and capture metadata already exist. Publish the
                # complete sensor decision before optional overlay/archive IO.
                report['overlay_path']=str(folder/'overlay.jpg')
                report['publication_order']='decision_before_optional_overlay_and_archive_v9'
                if self.cfg.get('final_validity_seal_v19'):seal_current_frame_report(report,self.cfg,time.monotonic())
                self.process_count+=1;self.latest=report
                from fixed_map_frame_image import decode_bound_rgb
                image=decode_bound_rgb(frame.encoded_image,frame.image_sha256).copy()
                for i,cand in enumerate(frame.candidates):
                    a,b,d,e=map(int,cand.bbox_xyxy_pixels);cv2.rectangle(image,(a,b),(d,e),(0,255,255),2);cv2.putText(image,f'candidate-{i} (local only) {cand.detector_score:.2f}',(a,max(20,b-8)),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
                cv2.putText(image,f'{tracking["state"]} / {tracking.get("track_id") or "no designated track"}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,255,255),2)
                cv2.imwrite(report['overlay_path'],image,[cv2.IMWRITE_JPEG_QUALITY,85])
                save(folder/'perception.json',report)
                self.event('FRAME_PROCESSED',frame_id=meta['frame_id'],perception_path=str(folder/'perception.json'),valid=estimate['valid'],rejection_reasons=report['rejection_reasons'])
        except Exception as ex:
            self.event('INFERENCE_BLOCKED',reason=type(ex).__name__,detail=str(ex) if isinstance(ex,(ValueError,OSError)) or type(ex).__name__ in ['AcceleratorError','OutOfMemoryError'] else 'inspect local module compatibility');self.latest=dict(valid=False,rejection_reasons=['INFERENCE_BLOCKED_'+type(ex).__name__])
        finally:
            if policy:policy.close()
            save(self.output/'service_summary.json',dict(finished_utc=utc(),capture_count=self.capture_count,processed_count=self.process_count,dropped_count=self.drop_count,flight_commands_issued=False,last_frame=self.latest.get('frame_id'),last_reasons=self.latest.get('rejection_reasons',[])))
            self.event('INFERENCE_WORKER_CLOSED')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--samples',type=int,default=8);parser.add_argument('--max-seconds',type=float,default=35);args=parser.parse_args()
    import fixed_map_mission as mission
    from fixed_map_runtime import configure_session
    configure_session(Path(args.output).parent/'runtime')
    mission.ensure_scene()
    service=PerceptionService(args.output);service.start();deadline=time.monotonic()+args.max_seconds
    while time.monotonic()<deadline and service.process_count<args.samples:time.sleep(.2)
    service.stop();print(json.dumps(dict(output=args.output,processed=service.process_count,captured=service.capture_count,last_reasons=service.latest.get('rejection_reasons',[])),ensure_ascii=False))
if __name__=='__main__':main()
