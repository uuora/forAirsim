"""RGB/depth adapter and preview validation. No simulator or target truth API."""
import copy, hashlib, json, time
from pathlib import Path
import numpy as np
from t1_pose_geometry import CameraPose, SceneCameraModel, unproject_range, quaternion_rotation_xyzw

def digest_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def validate_pair(meta, cfg, now):
    reasons=[]
    if meta.get('capture_errors'):reasons+=meta['capture_errors']
    for key in ['rgb_timestamp_ns','depth_timestamp_ns']:
        if type(meta.get(key)) is not int or meta[key]<=0:reasons.append('INVALID_'+key.upper())
    if not reasons and abs(meta['rgb_timestamp_ns']-meta['depth_timestamp_ns'])/1e9>cfg['pair_skew_max_s']:reasons.append('RGB_DEPTH_READBACK_SKEW_EXCEEDED')
    age=now-meta['received_monotonic_s']
    if age<0 or age>cfg['current_max_age_s']:reasons.append('CURRENT_FRAME_EXPIRED')
    if not meta.get('timestamps_advanced',True):reasons.append('NONADVANCING_RENDER_TIMESTAMPS')
    if 'render_age_sim_s' in meta and (not np.isfinite(meta['render_age_sim_s']) or not 0<=meta['render_age_sim_s']<=cfg['current_max_age_s']):reasons.append('RENDER_TIMESTAMP_STALE_OR_FUTURE')
    try:
        a,b=[CameraPose(**meta[k]) for k in ['rgb_pose','depth_pose']]
        if np.linalg.norm(a.position_world_ned_m-b.position_world_ned_m)>.02:reasons.append('RGB_DEPTH_POSE_TRANSLATION_SKEW')
        qa=np.array(a.orientation_xyzw);qb=np.array(b.orientation_xyzw)
        angle=np.degrees(2*np.arccos(np.clip(abs(qa@qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1)))
        if angle>1:reasons.append('RGB_DEPTH_POSE_ROTATION_SKEW')
    except (ValueError,KeyError,TypeError):reasons.append('INVALID_CAMERA_POSE')
    return reasons

def surface_from_box(box, depth, meta, cfg, now):
    """Select an actual robust depth pixel, reprojected into the RGB inner box.

    Uses separate centered pinhole K for different capture sizes. Depth K is
    derived from the frozen DepthPerspective HFOV; the response pose and actual
    Scene projection bind RGB K. This is renderer metadata, not lens calibration.
    """
    reasons=validate_pair(meta,cfg,now)
    if reasons:raise ValueError(reasons[0])
    rgb=SceneCameraModel.from_k(meta['rgb_K'],*meta['rgb_size_wh'])
    dc=SceneCameraModel.from_k(meta['depth_K'],*meta['depth_size_wh'])
    rp,dp=[CameraPose(**meta[k]) for k in ['rgb_pose','depth_pose']]
    arr=np.asarray(depth,dtype=float)
    if arr.shape!=(dc.height,dc.width):raise ValueError('DEPTH_DIMENSION_MISMATCH')
    x0,y0,x1,y1=map(float,box)
    if not (0<x0<x1<rgb.width and 0<y0<y1<rgb.height):raise ValueError('TRUNCATED_OR_INVALID_BOX')
    inner=[x0+.25*(x1-x0),y0+.25*(y1-y0),x0+.75*(x1-x0),y0+.75*(y1-y0)]
    # Broad search window from angular projection, with margin for the accepted
    # small pose delta. Exact pose/range reprojection below decides membership.
    corners=np.array([[inner[0],inner[1]],[inner[2],inner[3]]])
    mapped=(corners-[rgb.cx,rgb.cy])/[rgb.fx,rgb.fy]*[dc.fx,dc.fy]+[dc.cx,dc.cy]
    left,top=np.maximum(np.floor(mapped[0]-4).astype(int),0)
    right,bottom=np.minimum(np.ceil(mapped[1]+4).astype(int),[dc.width,dc.height])
    if left>=right or top>=bottom:raise ValueError('RGB_BOX_OUTSIDE_DEPTH_FOV')
    yy,xx=np.mgrid[top:bottom,left:right];ranges=arr[top:bottom,left:right]
    finite=np.isfinite(ranges)&(ranges>cfg['surface_min_depth_m'])&(ranges<cfg['surface_max_depth_m'])
    # Invalid-pixel fraction is measured within the expected (unexpanded) ROI.
    nominal=(xx>=mapped[0,0])&(xx<mapped[1,0])&(yy>=mapped[0,1])&(yy<mapped[1,1])
    denominator=int(nominal.sum());fraction=float((finite&nominal).sum()/denominator) if denominator else 0
    if fraction<cfg['surface_min_valid_fraction']:raise ValueError('INSUFFICIENT_VALID_INNER_BOX_DEPTH')
    px=xx[finite];py=yy[finite];rv=ranges[finite]
    rays=np.column_stack([np.ones(len(px)),(px-dc.cx)/dc.fx,(py-dc.cy)/dc.fy]);points=rays*(rv/np.linalg.norm(rays,axis=1))[:,None]
    world=points@dp.rotation_camera_to_ned.T+dp.position_world_ned_m
    rgbpoints=(world-rp.position_world_ned_m)@rp.rotation_camera_to_ned
    front=rgbpoints[:,0]>rgb.near_depth_m
    uv=np.column_stack([rgb.cx+rgb.fx*rgbpoints[:,1]/np.maximum(rgbpoints[:,0],1e-9),rgb.cy+rgb.fy*rgbpoints[:,2]/np.maximum(rgbpoints[:,0],1e-9)])
    inside=front&(uv[:,0]>=inner[0])&(uv[:,0]<inner[2])&(uv[:,1]>=inner[1])&(uv[:,1]<inner[3])
    if inside.sum()<4:raise ValueError('DEPTH_RGB_REPROJECTION_INSUFFICIENT_SUPPORT')
    ids=np.nonzero(inside)[0];med=float(np.median(rv[ids]));center=np.array([(x0+x1)/2,(y0+y1)/2])
    policy=cfg.get('surface_point_policy','median_rank_actual_pixel_v10')
    selection=dict(policy=policy,center_is_not_estimated=True)
    if policy=='central_robust_actual_pixel_v11':
        # Median rank can alternate between two ends of a quantized equal-range
        # ring. Select the actual observed pixel nearest the RGB box center,
        # only inside a robust depth band; this is still a visible surface.
        mad=float(np.median(abs(rv[ids]-med)))
        floor=cfg['surface_depth_band_floor_m'];factor=cfg['surface_depth_band_mad_factor']
        if not np.isfinite([floor,factor]).all() or not 0<floor<=.1 or not 1<=factor<=4:raise ValueError('SURFACE_DEPTH_BAND_CONFIG_INVALID')
        band=max(floor,factor*mad);central=ids[abs(rv[ids]-med)<=band]
        if len(central)<4:raise ValueError('ROBUST_CENTRAL_DEPTH_SUPPORT_INSUFFICIENT')
        best=central[np.lexsort((py[central],px[central],abs(rv[central]-med),np.sum((uv[central]-center)**2,axis=1)))[0]]
        selection.update(median_absolute_depth_deviation_m=mad,depth_band_m=band,depth_band_support_pixels=len(central),engineering_band_not_calibrated_error_bound=True)
    elif policy=='median_rank_actual_pixel_v10':
        best=ids[np.lexsort((py[ids],px[ids],np.sum((uv[ids]-center)**2,axis=1),abs(rv[ids]-med)))[0]]
    else:raise ValueError('SURFACE_POINT_POLICY_NOT_SUPPORTED')
    chosen_world=unproject_range([px[best],py[best]],rv[best],dp,dc)
    return dict(valid=True,surface_world_xyz_m=(chosen_world*[1,1,-1]).tolist(),surface_world_ned_m=chosen_world.tolist(),meaning='representative_actual_visible_depth_surface_point_NOT_balloon_center',center_estimated=False,selection=selection,depth_kind='DepthPerspective_ray_range_m',representative_depth_pixel_xy=[int(px[best]),int(py[best])],rgb_reprojected_pixel_xy=uv[best].tolist(),ray_range_m=float(rv[best]),median_ray_range_m=med,valid_fraction=fraction,reprojected_support_pixels=int(inside.sum()),registration='independent_RGB_Depth_K_and_response_pose_forward_reprojection',synchronization_verified=False,limitations=['readback timestamps do not prove exposure synchronization','depth K derived from frozen HFOV, not an independent depth projection query','bbox membership does not prove instance-specific pixels','simulator ideal depth, not physical sensor accuracy'])

def perception_safety_parameters(task):
    """Closed projection: fixed goal/target/Actor coordinates never enter preview."""
    fields=['height_world_m','max_speed_measured_m_s','max_command_m_s','max_command_accel_m_s2','envelope_radius_config_m','clearance_stop_m','path_clearance_m','arrival_error_m','arrival_speed_m_s','arrival_dwell_s','hover_drift_max_m','sample_period_host_s','command_duration_sim_s','state_gap_stop_host_s','timeout_sim_s']
    return {k:copy.deepcopy(task[k]) for k in fields}

def sanitize_obstacles(primitives):
    """Drop ALL semantic balloons/tethers/anchors and target-labelled geometry.
    Static geometry supplies obstacles only, never a target or fallback point.
    """
    result=[];removed=[]
    for p in primitives:
        name=(p.get('id','')+' '+p.get('display_label','')).lower()
        if p.get('group')=='balloons' or any(s in name for s in ['balloon','target','tether','anchor']):
            removed.append(p.get('id'));continue
        row=copy.deepcopy(p);row['id']=f'static-{len(result):03d}';row['display_label']=row['id'];result.append(row)
    return result,removed

def preview_task(proposal, estimate, start, geometry, base_task, cfg, now):
    """Finite, source-bound preview only. No motion queue or baseline mutation."""
    required={'schema','vehicle','session_id','target_generation','description_sha256','frame_id','image_sha256','estimate_sha256','source_kind','created_monotonic_s','expires_monotonic_s','surface_world_xyz_m','goal_xyz_m'}
    if set(proposal)!=required or proposal['schema']!='perception_surface_approach_preview_v1':raise ValueError('INVALID_PERCEPTION_TASK_SCHEMA')
    expected_source='online_onboard_rgb_depth_yolo_attributes_v8' if cfg.get('perception_backend')=='yolo_only_v8' else 'online_onboard_rgb_depth_vlx_temporal'
    if proposal['source_kind']!=expected_source or estimate.get('mode')!='online':raise ValueError('REPLAY_OR_UNBOUND_SOURCE_NOT_ONLINE_TASK')
    if not estimate.get('valid') or estimate.get('target_state')!='TRACKING':raise ValueError('TARGET_NOT_VERIFIED_AND_TRACKING')
    if proposal['vehicle']!='DroneA':raise ValueError('DRONE_B_NOT_AUTHORIZED')
    for key in ['session_id','target_generation','description_sha256','frame_id','image_sha256','surface_world_xyz_m']:
        if proposal[key]!=estimate[key]:raise ValueError('TASK_SOURCE_BINDING_MISMATCH')
    if proposal['estimate_sha256']!=digest_json(estimate):raise ValueError('ESTIMATE_HASH_MISMATCH')
    expiry=min(estimate['received_monotonic_s']+cfg['current_max_age_s'],estimate['semantic_review_deadline_monotonic_s'])
    if not np.isfinite([now,proposal['created_monotonic_s'],proposal['expires_monotonic_s']]).all() or not estimate['received_monotonic_s']<=proposal['created_monotonic_s']<=now<=proposal['expires_monotonic_s']<=expiry:raise ValueError('PERCEPTION_TASK_EXPIRED_OR_FUTURE')
    return _preview_geometry(proposal,estimate,start,geometry,base_task,cfg)

def _preview_geometry(proposal,estimate,start,geometry,base_task,cfg):
    """Shared static safety checks, called only after source-specific validation."""
    start=np.array(start,float);goal=np.array(proposal['goal_xyz_m'],float);surface=np.array(proposal['surface_world_xyz_m'],float)
    lo=np.array(cfg['workspace_min_xyz_m']);hi=np.array(cfg['workspace_max_xyz_m'])
    if start.shape!=(3,) or goal.shape!=(3,) or surface.shape!=(3,) or not np.isfinite(np.r_[start,goal,surface]).all():raise ValueError('NONFINITE_TASK_POINTS')
    flight_start=start.copy();flight_start[2]=base_task['height_world_m']
    if np.any(flight_start<lo) or np.any(flight_start>hi) or np.any(goal<lo) or np.any(goal>hi):raise ValueError('OUTSIDE_FINITE_WORKSPACE')
    if goal[2]!=base_task['height_world_m'] or np.linalg.norm(goal[:2]-surface[:2])<cfg['standoff_m']-1e-7:raise ValueError('HEIGHT_OR_SURFACE_STANDOFF_FAILED')
    if np.linalg.norm(start[:2]-surface[:2])<1e-6:raise ValueError('APPROACH_BEARING_UNDEFINED')
    expected=approach_goal(surface,start,base_task,cfg)[:2]
    if not np.allclose(goal[:2],expected,atol=1e-7,rtol=0):raise ValueError('GOAL_DOES_NOT_MATCH_SURFACE_APPROACH_MODEL')
    if not geometry.segment(start,flight_start,base_task['clearance_stop_m'],True)[0]:raise ValueError('START_TAKEOFF_COLUMN_BLOCKED')
    path=geometry.plan(flight_start,goal,base_task['path_clearance_m']);segments=[geometry.segment(a,b,base_task['path_clearance_m']) for a,b in zip(path,path[1:])]
    if np.any(np.array(path)<lo) or np.any(np.array(path)>hi):raise ValueError('PATH_OUTSIDE_FINITE_WORKSPACE')
    if not all(r[0] for r in segments):raise ValueError('PATH_CLEARANCE_FAILED')
    return dict(status='STATIC_OBSTACLE_PREVIEW_VALID',task=proposal,task_sha256=digest_json(proposal),path=path,segments=[r[1] for r in segments],preview_only=True,flight_authorized=False,scope='static obstacle geometry; semantic objects excluded to prevent truth leakage; unmodeled objects and perception uncertainty require Stage3 validation')

def approach_goal(surface,start,base_task,cfg):
    s=np.array(surface);start=np.array(start);delta=start[:2]-s[:2]
    if np.linalg.norm(delta)<1e-6:raise ValueError('APPROACH_BEARING_UNDEFINED')
    direction=delta/np.linalg.norm(delta);distance=cfg['standoff_m']
    if cfg.get('approach_x_max_m') is not None:
        if direction[0]>=0:raise ValueError('APPROACH_MUST_STAY_BEFORE_RANDOM_REGION')
        distance=max(distance,(s[0]-cfg['approach_x_max_m'])/(-direction[0]))
    return np.r_[s[:2]+direction*distance,base_task['height_world_m']]

def make_proposal(estimate, start, base_task, cfg, now):
    goal=approach_goal(estimate['surface_world_xyz_m'],start,base_task,cfg)
    proposal={k:estimate[k] for k in ['session_id','target_generation','description_sha256','frame_id','image_sha256','surface_world_xyz_m']}
    proposal.update(schema='perception_surface_approach_preview_v1',vehicle='DroneA',estimate_sha256=digest_json(estimate),source_kind=('online_onboard_rgb_depth_yolo_attributes_v8' if cfg.get('perception_backend')=='yolo_only_v8' else 'online_onboard_rgb_depth_vlx_temporal'),created_monotonic_s=now,expires_monotonic_s=min(estimate['received_monotonic_s']+cfg['current_max_age_s'],estimate['semantic_review_deadline_monotonic_s']),goal_xyz_m=goal.tolist())
    return proposal

def make_replay_preview(estimate,start,geometry,base_task,cfg,original_axis_now):
    """Explicit offline branch. Cannot be submitted to live_estimate/Executor."""
    if estimate.get('mode')!='replay' or not estimate.get('valid') or estimate.get('target_state')!='TRACKING':raise ValueError('REPLAY_REQUIRES_CONFIRMED_REPROCESSED_EVIDENCE')
    expiry=min(estimate['received_monotonic_s']+cfg['current_max_age_s'],estimate['semantic_review_deadline_monotonic_s'])
    if not estimate['received_monotonic_s']<=original_axis_now<=expiry:raise ValueError('REPLAY_ORIGINAL_AXIS_EVIDENCE_EXPIRED')
    proposal=make_proposal(estimate,start,base_task,cfg,original_axis_now)
    proposal.update(schema='perception_surface_replay_preview_v1',source_kind=('recorded_RGB_depth_YOLO_attributes_OFFLINE_v8' if cfg.get('perception_backend')=='yolo_only_v8' else 'recorded_RGB_depth_VLX_reprocessed_OFFLINE'))
    result=_preview_geometry(proposal,estimate,start,geometry,base_task,cfg)
    result.update(mode='REPLAY_REPROCESSING_NOT_ONLINE_NOT_EXECUTABLE',status='REPLAY_STATIC_PREVIEW_VALID',execution_allowed=False,original_axis_check_time_s=original_axis_now)
    return result
