"""Measured-box projection and conservative command envelopes (no simulator I/O).

The boundary envelope uses a declared delay/braking model. It does not prove
physical braking capability or constrain unsampled motion peaks.
"""
import math
import numpy as np


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def bbox_support_offsets(bbox_xywh, anchor_px, image_size_wh, camera_depth_m,
                         horizontal_fov_deg=90.):
    """Four observed-box corner offsets at the representative point's axial depth.

    anchor_px must project the same point to which these offsets are added;
    a detection centroid cannot substitute for a different surface pixel.
    Fixed +X camera; Y points right, Z down. No radius/target truth is read.
    This is an observed rectangular footprint approximation, not a 3D hull.
    """
    x, y, w, h = vector(bbox_xywh, 4, "bbox")
    cx, cy = vector(anchor_px, 2, "anchor pixel")
    width, height = vector(image_size_wh, 2, "image size")
    depth, fov = float(camera_depth_m), float(horizontal_fov_deg)
    if min(width, height, w, h, depth) <= 0 or not math.isfinite(depth) or not 0 < fov < 180:
        raise ValueError("Invalid image, bbox, axial depth or FOV")
    if x < 0 or y < 0 or x+w > width or y+h > height or not x <= cx <= x+w or not y <= cy <= y+h:
        raise ValueError("Observed box/anchor lies outside the image")
    focal = width / (2*math.tan(math.radians(fov)/2))
    return [[0., float((u-cx)*depth/focal), float((v-cy)*depth/focal)]
            for u, v in ((x,y),(x+w,y),(x,y+h),(x+w,y+h))]


def projected_bbox_support_offsets(bbox_xywh, image_size_wh, surface_relative_camera_ned_m,
                                   horizontal_fov_deg=90.):
    """Anchor the measured box to the supplied point in the fixed-forward model.

    The provider can select a median-depth pixel away from the detection centre.
    Project the actual estimated point in the planner's declared camera axes so
    point + offsets reconstructs the measured box, rather than a translated box.
    This still assumes one axial depth and fixed-forward attitude; no 3D hull.
    """
    width, height = vector(image_size_wh, 2, "image size")
    point = vector(surface_relative_camera_ned_m, 3, "surface camera-relative point")
    fov = float(horizontal_fov_deg)
    if min(width, height, point[0]) <= 0 or not 0 < fov < 180:
        raise ValueError("Invalid image, forward surface depth or FOV")
    focal = width / (2*math.tan(math.radians(fov)/2))
    anchor = [(width-1)/2 + focal*point[1]/point[0],
              (height-1)/2 + focal*point[2]/point[0]]
    return {"offsets_ned_m": bbox_support_offsets(bbox_xywh, anchor, image_size_wh, point[0], fov),
            "anchor_pixel_xy": [float(value) for value in anchor],
            "anchor_scope": "fixed_forward_projection_of_estimated_surface_point_not_detection_centroid"}


def command_window(age_s, max_age_s, requested_s, minimum_s, reserve_s):
    values = [float(x) for x in (age_s, max_age_s, requested_s, minimum_s, reserve_s)]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("Nonfinite freshness window")
    age, maximum, requested, minimum, reserve = values
    if age < 0 or min(maximum, requested, minimum) <= 0 or reserve < 0 or minimum > requested:
        raise ValueError("Invalid freshness window")
    available = maximum-age-reserve
    duration = min(requested, available)
    accepted = duration >= minimum
    return {"age_proxy_s": age, "maximum_age_s": maximum, "reserve_s": reserve,
            "available_s": available, "requested_s": requested,
            "duration_s": duration if accepted else 0., "accepted": accepted,
            "reason": "FRESH_WINDOW_AVAILABLE" if accepted else "REOBSERVE_BEFORE_EXPIRY"}


def boundary_command(position, desired_xy, previous_xy, low, high, speed_cap,
                     ramp_acceleration, ramp_dt, horizon_s, reserve_m):
    p = vector(position, 3, "position")
    desired, previous = vector(desired_xy, 2, "desired"), vector(previous_xy, 2, "previous")
    lo, hi = vector(low, 3, "lower bound"), vector(high, 3, "upper bound")
    cap, a, dt, horizon, reserve = map(float, (speed_cap, ramp_acceleration, ramp_dt, horizon_s, reserve_m))
    if not all(math.isfinite(x) for x in (cap,a,dt,horizon,reserve)) or min(cap,a,dt,horizon) <= 0 or reserve < 0 or np.any(lo >= hi):
        raise ValueError("Invalid boundary envelope parameters")
    if np.linalg.norm(previous) > cap+1e-9:
        raise ValueError("Previous command exceeds speed budget")
    if np.any(p < lo) or np.any(p > hi):
        raise ValueError("Actual position outside protected workspace")
    distance_lo = np.maximum(p[:2]-lo[:2]-reserve, 0.)
    distance_hi = np.maximum(hi[:2]-p[:2]-reserve, 0.)
    # Solve u*h + u^2/(2*a) <= available distance independently per axis.
    speed_lo = np.sqrt((a*horizon)**2+2*a*distance_lo)-a*horizon
    speed_hi = np.sqrt((a*horizon)**2+2*a*distance_hi)-a*horizon
    norm_limited = desired*min(1.,cap/max(float(np.linalg.norm(desired)),1e-12))
    bounded = np.clip(norm_limited,-speed_lo,speed_hi)
    delta = bounded-previous
    applied = previous+delta*min(1.,a*dt/max(float(np.linalg.norm(delta)),1e-12))
    valid = bool(np.all(applied >= -speed_lo-1e-9) and np.all(applied <= speed_hi+1e-9))
    return {"accepted": valid, "command_velocity_xy_m_s": applied.tolist() if valid else None,
        "desired_velocity_xy_m_s": desired.tolist(), "previous_velocity_xy_m_s": previous.tolist(),
        "lower_velocity_xy_m_s": (-speed_lo).tolist(), "upper_velocity_xy_m_s": speed_hi.tolist(),
        "bounded_requested_velocity_xy_m_s": bounded.tolist(), "command_change_norm_m_s": float(np.linalg.norm(applied-previous)),
        "boundary_limited": bool(np.linalg.norm(bounded-norm_limited)>1e-9),
        "horizon_s": horizon, "position_reserve_m": reserve,
        "braking_acceleration_model_m_s2": a,
        "scope": "command envelope with declared delay/braking approximation; no physical safety certificate",
        "reason": "BOUNDARY_ENVELOPE_ACCEPTED" if valid else "NO_BOUNDARY_AND_RAMP_FEASIBLE_COMMAND"}
