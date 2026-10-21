"""Finite XY quintic candidates scored with the measured Scene camera pose.

Current RGB-box bearing and representative depth surface are estimates, not
physical target centers. Camera orientation is held at its measured value for
preview; sampled constraints and projected box planes give no continuous
collision, future occlusion or persistent-instance guarantee.
"""
from dataclasses import dataclass, asdict
import numpy as np
from perception_trajectory_planner import quintic_coefficients, evaluate_quintic
from t1_pose_geometry import CameraPose, project_world, project_footprint_polygon
from t1_pose_projection_batch import score_pose_path


@dataclass(frozen=True)
class PosePlannerConfig:
    durations_s: tuple = (1.6, 2.0)
    sample_dt_s: float = .1
    endpoint_step_m: float = .024
    max_speed_m_s: float = .03
    max_acceleration_m_s2: float = .15
    desired_surface_range_m: float = 3.0
    image_center_weight: float = 12.0
    range_weight: float = .2
    motion_weight: float = .02
    fixed_world_z_m: float = -3.0
    position_min_ned_m: tuple = (-.3, -3.3, -3.15)
    position_max_ned_m: tuple = (.3, -2.7, -2.85)


def finite(value, n, name):
    result = np.asarray(value, float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(name)
    return result


def plan_pose_trajectory(*, surface, camera, camera_pose, position_ned_m,
                         velocity_ned_m_s, acceleration_ned_m_s2, config=None):
    cfg = config or PosePlannerConfig()
    p = finite(position_ned_m, 3, "position")
    v = finite(velocity_ned_m_s, 3, "velocity").copy()
    a = finite(acceleration_ned_m_s2, 3, "acceleration").copy()
    low = finite(cfg.position_min_ned_m, 3, "low")
    high = finite(cfg.position_max_ned_m, 3, "high")
    if np.any(low >= high) or not low[2] < cfg.fixed_world_z_m < high[2]:
        raise ValueError("workspace")
    if cfg.sample_dt_s <= 0 or cfg.endpoint_step_m <= 0 or cfg.max_speed_m_s <= 0:
        raise ValueError("positive planner limits required")
    out = {"status": "NO_FEASIBLE_CANDIDATE", "configuration": asdict(cfg),
           "candidates": [], "flight_authorized": False,
           "model": "measured_camera_orientation_frozen_XY_translation_preview",
           "constraint_scope": "sampled_times_only",
           "geometry_scope": "current_RGB_box_depth_plane_not_target_center_or_future_visibility"}
    if surface.get("status") != "SURFACE_GEOMETRY_ACCEPTED":
        out["reason"] = "CURRENT_SURFACE_REQUIRED"
        return out
    if abs(p[2]-cfg.fixed_world_z_m) > .12:
        out["reason"] = "FIXED_HEIGHT_NOT_SETTLED"
        return out
    if np.linalg.norm(v[:2]) > cfg.max_speed_m_s or np.linalg.norm(a[:2]) > cfg.max_acceleration_m_s2:
        out["reason"] = "ACTUAL_INITIAL_XY_STATE_EXCEEDS_PLANNER_LIMIT"
        return out
    # Vertical motion is controlled by the independent fixed-height controller.
    # Its actual error and motion are guarded in the runner, not optimized here.
    p[2] = cfg.fixed_world_z_m
    v[2] = a[2] = 0.
    reference = finite(surface["rgb_reference_world_ned_m"], 3, "surface reference")
    corners = np.asarray(surface["bbox_plane_corners_world_ned_m"], float)
    step = cfg.endpoint_step_m
    offsets = [(0.,0.), (step,0.), (-step,0.), (0.,step), (0.,-step),
               (step*.66,step*.66), (step*.66,-step*.66),
               (-step*.66,step*.66), (-step*.66,-step*.66)]
    best = None
    for duration in cfg.durations_s:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration")
        ts = np.linspace(0, duration, int(np.ceil(duration/cfg.sample_dt_s))+1)
        for delta in offsets:
            endpoint = p + [*delta, 0.]
            coeff = quintic_coefficients(p, v, a, endpoint, duration)
            positions = evaluate_quintic(coeff, ts)
            velocities = evaluate_quintic(coeff, ts, 1)
            accelerations = evaluate_quintic(coeff, ts, 2)
            row = {"endpoint_ned_m": endpoint.tolist(), "duration_s": duration,
                   "max_sampled_speed_m_s": float(np.max(np.linalg.norm(velocities,axis=1))),
                   "max_sampled_acceleration_m_s2": float(np.max(np.linalg.norm(accelerations,axis=1)))}
            reasons = []
            if (np.any(positions < low) or np.any(positions > high)):
                reasons.append("SAMPLED_WORKSPACE")
            if row["max_sampled_speed_m_s"] > cfg.max_speed_m_s+1e-9:
                reasons.append("SAMPLED_SPEED")
            if row["max_sampled_acceleration_m_s2"] > cfg.max_acceleration_m_s2+1e-9:
                reasons.append("SAMPLED_ACCELERATION")
            center, ranges, fractions = [], [], []
            try:
                geometry=score_pose_path(reference,corners,positions,p,camera_pose,camera)
                center=geometry['centering_squared_errors']
                ranges=[(value-cfg.desired_surface_range_m)**2 for value in geometry['ray_ranges_m']]
                fractions=geometry['field_area_fractions']
                if min(fractions) < .95:
                    reasons.append("PROJECTED_BOX_FRACTION")
            except ValueError:
                reasons.append("PROJECTION_REJECTED")
            row["rejection_reasons"] = reasons
            row["feasible"] = not reasons
            if not reasons:
                terms = {"image_center": float(np.mean(center)), "range_error": float(np.mean(ranges)),
                         "acceleration_energy": float(np.mean(np.sum(accelerations**2,axis=1)))}
                row.update(cost_terms=terms, minimum_projected_box_fraction=min(fractions),
                           cost=cfg.image_center_weight*terms["image_center"] + cfg.range_weight*terms["range_error"]
                                + cfg.motion_weight*terms["acceleration_energy"])
                if best is None or row["cost"] < best[0]:
                    best = (row["cost"], dict(row, coefficients=coeff.tolist()))
            out["candidates"].append(row)
    if best:
        out.update(status="POSE_TRAJECTORY_SELECTED", trajectory=best[1])
    return out


def preview_command(plan, elapsed_s=.12):
    if plan.get("status") != "POSE_TRAJECTORY_SELECTED":
        return [0., 0.]
    t = min(max(0., float(elapsed_s)), plan["trajectory"]["duration_s"])
    return evaluate_quintic(plan["trajectory"]["coefficients"], t, 1)[:2].tolist()
