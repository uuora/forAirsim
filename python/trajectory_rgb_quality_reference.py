"""Separate RGB observation centering from the visible depth surface point.

This uses the existing fixed-forward, representative axial-depth plane model.
The resulting reference is not a reconstructed physical balloon center.
"""
import math
import numpy as np
from perception_trajectory_planner import evaluate_quintic


def rgb_quality_reference(surface_world, camera_world, centre_px, image_size_wh, horizontal_fov_deg=90.):
    surface=np.asarray(surface_world,dtype=float);camera=np.asarray(camera_world,dtype=float)
    pixel=np.asarray(centre_px,dtype=float);size=np.asarray(image_size_wh,dtype=float)
    if surface.shape!=(3,) or camera.shape!=(3,) or pixel.shape!=(2,) or size.shape!=(2,):
        raise ValueError('Invalid RGB quality reference dimensions')
    if not all(np.isfinite(v).all() for v in (surface,camera,pixel,size)):
        raise ValueError('Nonfinite RGB quality reference')
    if np.any(size<=0) or np.any(size!=np.floor(size)) or np.any(pixel<0) or np.any(pixel>=size):
        raise ValueError('Invalid image size or RGB centroid outside image')
    fov=float(horizontal_fov_deg)
    if not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError('Invalid RGB quality calibration')
    relative=surface-camera;depth=float(relative[0])
    if depth<=0:
        raise ValueError('Surface lies behind declared forward camera')
    focal=size[0]/(2*math.tan(math.radians(fov)/2))
    principal=(size-1)/2
    reference_camera=np.array([depth,*(pixel-principal)*depth/focal])
    point=camera+reference_camera
    return dict(reference_world_ned_m=point.tolist(),offset_from_surface_ned_m=(point-surface).tolist(),
        centre_px=pixel.tolist(),surface_anchor_px=(principal+focal*relative[1:]/depth).tolist(),
        axial_plane_depth_m=depth,
        semantics='RGB_detected_centroid_bearing_at_representative_surface_axial_depth_not_physical_balloon_center',
        scope='fixed_forward_camera_same_depth_plane; distance and footprint keep the original surface anchor')


def centering_cost(plan, candidate, offset):
    times=np.linspace(0.,candidate['duration_s'],candidate['sample_count'])
    positions=evaluate_quintic(candidate['coefficients'],times)
    state=plan['input_state'];params=plan['params']
    target=np.array(state['target_estimate'])+(state['observation_age_s']+times[:,None])*np.array(state['target_velocity_estimate'])
    camera=positions+np.array(params['camera_offset_ned_m'])
    relative=target+np.array(offset)-camera
    depth=np.maximum(relative[:,0],params['camera_near_depth_m'])
    tangent=np.tan(np.radians([params['horizontal_fov_deg'],params['vertical_fov_deg']])/2)
    normalized=relative[:,1:]/(depth[:,None]*tangent)
    values=np.sum(normalized**2,axis=1)
    integral=float(np.sum((values[:-1]+values[1:])*np.diff(times)/2))
    return integral/candidate['duration_s']


def replay_centering_ranking(plan, offset):
    """Change one cost interpretation; reuse original sampled feasibility."""
    rows=[]
    for candidate in plan['candidates']:
        if not candidate['feasible']:
            continue
        original=centering_cost(plan,candidate,plan['input_state'].get('image_center_reference_offset_ned_m',[0,0,0]))
        if abs(original-candidate['cost_components']['image_center'])>1e-9:
            raise ValueError('Historical centering integral not reproduced')
        updated=centering_cost(plan,candidate,offset)
        weight=plan['cost_weights']['image_center']
        cost=candidate['total_cost']-candidate['weighted_cost_components']['image_center']+weight*updated
        rows.append(dict(index=candidate['index'],endpoint=candidate['endpoint'],original_cost=candidate['total_cost'],
                         proposed_cost=cost,original_center_cost=original,proposed_center_cost=updated))
    old=min(rows,key=lambda r:(r['original_cost'],r['index'])) if rows else None
    new=min(rows,key=lambda r:(r['proposed_cost'],r['index'])) if rows else None
    if old is not None and old['index']!=plan['selected_index']:
        raise ValueError('Historical selected index not reproduced')
    return dict(original_selected_index=plan['selected_index'],proposed_selected_index=new['index'] if new else None,
                original_endpoint=old['endpoint'] if old else None,proposed_endpoint=new['endpoint'] if new else None,
                feasible_candidate_count=len(rows),feasible_set_unchanged=True,candidates=rows,
                scope='offline cost interpretation; no newly feasible trajectory and no new flight outcome')
