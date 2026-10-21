"""Finite, source-bound Stage3 sensor task guards. No simulator truth input."""
import copy,hashlib,time
import numpy as np
from fixed_map_perception_geometry import validate_pair,make_proposal,preview_task,digest_json

def live_estimate(row,cfg,now):
    if row.get('mode')!='online' or not row.get('valid'):raise ValueError('NO_VALID_ONLINE_DESIGNATED_ESTIMATE')
    estimate=row.get('position_estimate',{});meta=row.get('capture',{})
    if cfg.get('scene_epoch_required'):
        from fixed_map_scene_binding import engine_identity_alive
        if meta.get('runtime_engine_identity')!=cfg.get('runtime_engine_identity') or not engine_identity_alive(cfg.get('runtime_engine_identity')):raise ValueError('SCENE_ENGINE_LIFETIME_CHANGED')
    if cfg.get('runtime_scene_version') and (meta.get('runtime_scene_version')!=cfg['runtime_scene_version'] or estimate.get('runtime_scene_version')!=cfg['runtime_scene_version']):raise ValueError('SCENE_VERSION_BINDING_MISMATCH')
    errors=validate_pair(meta,cfg,now)
    if errors:raise ValueError(errors[0])
    if not estimate.get('valid') or estimate.get('target_state')!='TRACKING':raise ValueError('TARGET_NOT_TRACKING')
    for key in ['session_id','target_generation','description_sha256','frame_id','image_sha256','received_monotonic_s']:
        if estimate.get(key)!=meta.get(key):raise ValueError('ESTIMATE_CAPTURE_BINDING_MISMATCH')
    expected=hashlib.sha256(cfg['target_description'].encode()).hexdigest()
    if estimate['target_generation']!=cfg['target_generation'] or estimate['description_sha256']!=expected:raise ValueError('TARGET_VERSION_OR_DESCRIPTION_CHANGED')
    if not now<=estimate['semantic_review_deadline_monotonic_s']:raise ValueError('SEMANTIC_EVIDENCE_EXPIRED')
    attr=row.get('current_attribute_audit',{})
    if attr.get('status')!='ACCEPTED' or attr.get('frame_id')!=estimate['frame_id'] or attr.get('image_sha256')!=estimate['image_sha256']:raise ValueError('FRESH_ATTRIBUTE_CONFIRMATION_MISSING')
    if estimate.get('meaning')!='visible_surface_point_not_center' or estimate.get('surface',{}).get('center_estimated') is not False:raise ValueError('POSITION_MEANING_INVALID')
    sf=row.get('semantic_frame',{});tracking=row.get('temporal_identity',{})
    if sf.get('frame_id')!=estimate['frame_id'] or sf.get('image_sha256')!=estimate['image_sha256']:raise ValueError('DETECTION_FRAME_BINDING_MISMATCH')
    selected=tracking.get('selected_candidate_id')
    if attr.get('selected_candidate_id')!=selected or selected is None:raise ValueError('ATTRIBUTE_TRACK_ASSOCIATION_MISMATCH')
    position=next((c['position'] for c in row.get('candidate_positions',[]) if c['candidate_id']==selected),None)
    if position is None or not position.get('valid') or digest_json(position)!=digest_json(estimate['surface']) or position.get('surface_world_xyz_m')!=estimate['surface_world_xyz_m']:raise ValueError('SELECTED_DEPTH_SOURCE_MISMATCH')
    if not np.isfinite(estimate.get('surface_world_xyz_m',[])).all() or len(estimate.get('surface_world_xyz_m',[]))!=3:raise ValueError('INVALID_SURFACE_POSITION')
    if cfg.get('semantic_policy_version')=='delayed_chain_attributes_v3':
        from t1_delayed_semantic_bridge import _frame_from_dict,verify_delayed_selection
        verify_delayed_selection(_frame_from_dict(row['semantic_frame']),row['temporal_identity'],cfg['target_description'],now)
    if cfg.get('perception_backend')=='yolo_only_v8':
        anchor=estimate.get('semantic_anchor',{})
        if estimate.get('perception_backend')!='yolo_only_v8' or anchor.get('kind')!='yolo_current_rgb_attributes_v8':raise ValueError('YOLO_SOURCE_KIND_MISMATCH')
        for key in ['session_id','target_generation','description_sha256','frame_id','image_sha256']:
            if anchor.get(key)!=estimate.get(key):raise ValueError('YOLO_CURRENT_FRAME_ANCHOR_MISMATCH')
        if anchor.get('selected_candidate_id')!=selected or anchor.get('track_id')!=tracking.get('track_id') or anchor.get('consecutive_frames',0)<cfg['association_confirmation_frames']:raise ValueError('YOLO_CONTINUITY_NOT_CONFIRMED')
    return copy.deepcopy(estimate)

def identity_context(estimate):
    # local track IDs are evidence association only, never map instance IDs.
    return tuple(estimate[k] for k in ['session_id','target_generation','description_sha256'])

def candidate_approach_guard(path,row,cfg):
    """Additional conservative limit for unmapped balloons/tethers.

    Surface error .5m and unobserved extent .75m are declared engineering
    bounds, not measured uncertainty. Sparse RGB boxes do not prove coverage.
    """
    surfaces=[np.array(c['position']['surface_world_xyz_m']) for c in row.get('candidate_positions',[]) if c['position'].get('valid')]
    if not surfaces:raise ValueError('NO_VALID_SENSOR_OBSTACLE_SURFACE')
    limit=cfg['sensor_surface_exclusion_m'];minimum=float('inf')
    for a,b in zip(path,path[1:]):
        a,b=np.array(a),np.array(b);ab=b-a;den=float(ab@ab)
        for p in surfaces:
            t=np.clip(float((p-a)@ab)/den,0,1) if den else 0
            minimum=min(minimum,float(np.linalg.norm(p-(a+t*ab))))
    if minimum<limit:raise ValueError('OBSERVED_SURFACE_EXCLUSION_FAILED')
    return dict(minimum_observed_surface_distance_m=minimum,required_m=limit,scope='observed candidate surfaces and declared extent/error only; NOT full unseen-object safety proof')

def build_preview(row,start,geometry,safety,cfg,now):
    estimate=live_estimate(row,cfg,now)
    proposal=make_proposal(estimate,start,safety,cfg,now)
    result=preview_task(proposal,estimate,start,geometry,safety,cfg,now)
    result.update(sensor_guard=candidate_approach_guard(result['path'],row,cfg),planned_from_estimate=estimate,planned_start_xyz_m=list(start),planning_finished_monotonic_s=time.monotonic(),execution_schema='perception_surface_execution_v1')
    return result

def revalidate_preview(preview,row,start,geometry,safety,cfg,now):
    """Rebind to NEW fresh evidence after planning, without extending old age."""
    old=preview['planned_from_estimate'];new=live_estimate(row,cfg,now)
    if identity_context(old)!=identity_context(new):raise ValueError('PLANNED_TARGET_CONTEXT_CHANGED')
    if np.linalg.norm(np.array(old['surface_world_xyz_m'])-new['surface_world_xyz_m'])>cfg['surface_stability_max_m']:raise ValueError('SURFACE_CHANGED_REPLAN_REQUIRED')
    proposal=make_proposal(new,start,safety,cfg,now)
    if np.linalg.norm(np.array(proposal['goal_xyz_m'])-preview['task']['goal_xyz_m'])>safety['arrival_error_m']:raise ValueError('GOAL_CHANGED_REPLAN_REQUIRED')
    start=np.array(start,float);lo=np.array(cfg['workspace_min_xyz_m']);hi=np.array(cfg['workspace_max_xyz_m'])
    if np.any(start<lo) or np.any(start>hi):raise ValueError('SELF_OUTSIDE_WORKSPACE')
    if abs(start[2]-safety['height_world_m'])>.03:raise ValueError('SELF_NOT_ON_FIXED_HEIGHT_PLANE')
    path=copy.deepcopy(preview['path']);path[0]=start.tolist();path[-1]=proposal['goal_xyz_m']
    if np.any(np.array(path)<lo) or np.any(np.array(path)>hi):raise ValueError('PATH_OUTSIDE_FINITE_WORKSPACE')
    segments=[geometry.segment(a,b,safety['path_clearance_m']) for a,b in zip(path,path[1:])]
    if not all(s[0] for s in segments):raise ValueError('REBOUND_PATH_CLEARANCE_FAILED')
    guard=candidate_approach_guard(path,row,cfg)
    return dict(schema='perception_surface_execution_v1',vehicle='DroneA',task=proposal,task_sha256=digest_json(proposal),path=path,segments=[s[1] for s in segments],sensor_guard=guard,estimate=new,previous_frame_id=old['frame_id'],revalidated_monotonic_s=now,planning_source='sensor_estimate_only',surface_meaning='visible_surface_NOT_balloon_center')

def execution_evidence(plan,row,cfg,now):
    estimate=live_estimate(row,cfg,now)
    if identity_context(estimate)!=identity_context(plan['estimate']):raise ValueError('ACTIVE_TARGET_CONTEXT_CHANGED')
    if np.linalg.norm(np.array(estimate['surface_world_xyz_m'])-plan['estimate']['surface_world_xyz_m'])>cfg['surface_stability_max_m']:raise ValueError('ACTIVE_SURFACE_CHANGED')
    candidate_approach_guard(plan['path'],row,cfg)
    return dict(frame_id=estimate['frame_id'],image_sha256=estimate['image_sha256'],estimate_sha256=digest_json(estimate),session_id=estimate['session_id'],target_generation=estimate['target_generation'],description_sha256=estimate['description_sha256'],semantic_anchor=estimate['semantic_anchor'],evidence_age_s=now-estimate['received_monotonic_s'],source_kind=('fresh_online_onboard_RGB_depth_YOLO_attributes_v8' if cfg.get('perception_backend')=='yolo_only_v8' else 'fresh_online_onboard_RGB_depth_VLX_current_attributes'))
