"""Finite candidate quintic planner for the first AirSim research prototype.

This module consumes the caller's *estimated* target state and never imports
AirSim or reads simulator ground truth. The camera has fixed yaw 0, faces +x,
and uses y/x and z/x projection in the caller's coordinate frame (NED is OK).
A fixed NED camera offset translates its origin; no attitude or extrinsic
rotation is applied in this fixed-yaw prototype.
Feasibility is checked only at sampled times: it is NOT a continuous collision
avoidance guarantee, a global optimizer, an occlusion model, or calibrated
target identity reasoning. Obstacles must be supplied explicitly by the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil, radians, tan
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PlanningParams:
    durations: tuple[float, ...] = (3.0, 4.0, 5.0)
    sample_dt_s: float = 0.05
    position_min: tuple[float, float, float] = (-10.0, -10.0, -10.0)
    position_max: tuple[float, float, float] = (10.0, 10.0, 1.0)
    max_speed_mps: float = 0.8
    max_acceleration_mps2: float = 0.8
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 60.0
    camera_offset_ned_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_near_depth_m: float = 0.05
    image_border_margin_fraction: float = 0.0
    min_target_distance_m: float = 0.5
    desired_observation_distance_m: float = 1.5
    drone_radius_m: float = 0.2
    obstacle_margin_m: float = 0.1
    preferred_clearance_m: float = 0.4
    max_observation_age_s: float = 0.5
    time_reference_s: float = 5.0
    jerk_reference_mps3: float = 1.0
    acceleration_reference_mps2: float = 0.8
    terminal_observation_weight: float = 1.0
    time_efficiency_weight: float = 0.5
    image_center_weight: float = 4.0
    jerk_energy_weight: float = 0.05
    acceleration_energy_weight: float = 0.05
    safety_margin_weight: float = 0.2


@dataclass(frozen=True)
class SphereObstacle:
    center: tuple[float, float, float]
    radius: float
    label: str = "sphere"


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


def quintic_coefficients(
    position: Sequence[float],
    velocity: Sequence[float],
    acceleration: Sequence[float],
    endpoint: Sequence[float],
    duration_s: float,
    terminal_velocity: Sequence[float] = (0.0, 0.0, 0.0),
    terminal_acceleration: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return coefficients [c0,...,c5] of p(t), shape (6,3)."""
    p = _vector(position, "position")
    v = _vector(velocity, "velocity")
    a = _vector(acceleration, "acceleration")
    goal = _vector(endpoint, "endpoint")
    goal_v = _vector(terminal_velocity, "terminal_velocity")
    goal_a = _vector(terminal_acceleration, "terminal_acceleration")
    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    coefficients = np.zeros((6, 3), dtype=float)
    coefficients[:3] = (p, v, a / 2.0)
    t = duration
    system = np.array([[t**3, t**4, t**5],
                       [3*t**2, 4*t**3, 5*t**4],
                       [6*t, 12*t**2, 20*t**3]])
    residual = np.vstack((goal - p - v*t - a*t*t/2.0,
                          goal_v - v - a*t,
                          goal_a - a))
    coefficients[3:] = np.linalg.solve(system, residual)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("quintic coefficient computation was not finite")
    return coefficients


def evaluate_quintic(
    coefficients: Sequence[Sequence[float]],
    time_s: float | Sequence[float] | np.ndarray,
    derivative: int = 0,
) -> np.ndarray:
    """Evaluate p/v/a/jerk; a scalar time returns (3,), a vector (N,3)."""
    c = np.asarray(coefficients, dtype=float)
    times = np.asarray(time_s, dtype=float)
    if c.shape != (6, 3) or not np.all(np.isfinite(c)):
        raise ValueError("coefficients must have finite shape (6,3)")
    if times.ndim > 1 or not np.all(np.isfinite(times)):
        raise ValueError("time_s must be a finite scalar or one-dimensional array")
    if not isinstance(derivative, int) or not 0 <= derivative <= 5:
        raise ValueError("derivative must be an integer from 0 to 5")
    differentiated = c.copy()
    for _ in range(derivative):
        differentiated = differentiated[1:] * np.arange(1, len(differentiated))[:, None]
    result = np.zeros(times.shape + (3,), dtype=float)
    for coefficient in differentiated[::-1]:
        result = result * times[..., None] + coefficient
    return result


def evaluate_plan(plan: Mapping[str, Any], time_s: float) -> dict[str, np.ndarray]:
    """Evaluate the selected segment; rejects failed plans and extrapolation."""
    if plan.get("status") != "OK" or plan.get("trajectory") is None:
        raise ValueError("plan has no feasible selected trajectory")
    trajectory = plan["trajectory"]
    t = float(time_s)
    if not np.isfinite(t) or t < 0.0 or t > trajectory["duration_s"]:
        raise ValueError("time_s must be within the selected segment")
    c = trajectory["coefficients"]
    return {"position": evaluate_quintic(c, t),
            "velocity": evaluate_quintic(c, t, 1),
            "acceleration": evaluate_quintic(c, t, 2)}


def _validate_params(params: PlanningParams) -> None:
    if not isinstance(params, PlanningParams):
        raise ValueError("params must be PlanningParams")
    low = _vector(params.position_min, "position_min")
    high = _vector(params.position_max, "position_max")
    _vector(params.camera_offset_ned_m, "camera_offset_ned_m")
    if np.any(low >= high):
        raise ValueError("position_min must be below position_max on every axis")
    if not params.durations:
        raise ValueError("durations cannot be empty")
    scalar_names = [name for name in asdict(params) if name not in
                    {"durations", "position_min", "position_max", "camera_offset_ned_m"}]
    for name in scalar_names:
        value = float(getattr(params, name))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if name.endswith("_weight") or name in {
            "min_target_distance_m", "drone_radius_m", "obstacle_margin_m",
            "max_observation_age_s",
            "image_border_margin_fraction",
        }:
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
        elif value <= 0.0:
            raise ValueError(f"{name} must be positive")
    if not 0.0 < params.horizontal_fov_deg < 180.0 or not 0.0 < params.vertical_fov_deg < 180.0:
        raise ValueError("camera FOV must be between 0 and 180 degrees")
    if params.image_border_margin_fraction >= 1.0:
        raise ValueError("image border margin must be less than one")
    if params.desired_observation_distance_m < params.min_target_distance_m:
        raise ValueError("desired observation distance cannot be below its minimum")
    for duration in params.durations:
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("every candidate duration must be finite and positive")
        if duration / params.sample_dt_s > 10000:
            raise ValueError("each candidate is limited to 10000 sampling intervals")


def _normalize_obstacles(obstacles: Iterable[SphereObstacle | Mapping[str, Any]]) -> list[SphereObstacle]:
    normalized = []
    for index, item in enumerate(obstacles):
        if isinstance(item, SphereObstacle):
            obstacle = item
        elif isinstance(item, Mapping):
            obstacle = SphereObstacle(tuple(item["center"]), float(item["radius"]),
                                      str(item.get("label", f"sphere_{index}")))
        else:
            raise ValueError("each obstacle must be SphereObstacle or a center/radius mapping")
        center = _vector(obstacle.center, f"obstacle {index} center")
        radius = float(obstacle.radius)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("obstacle radius must be finite and nonnegative")
        normalized.append(SphereObstacle(tuple(center), radius, str(obstacle.label)))
    return normalized


def _failure(plan: dict[str, Any], status: str, reason: str,
             velocity: np.ndarray | None = None, acceleration: np.ndarray | None = None,
             initial_constraint_violation: bool = False) -> dict[str, Any]:
    if initial_constraint_violation:
        action = "CONTROLLER_CONSTRAINT_RECOVERY_REQUIRED"
    elif velocity is None or acceleration is None:
        action = "REQUEST_VALID_STATE_AND_CONTROLLER_PROTECTION"
    elif np.linalg.norm(velocity) > 1e-4 or np.linalg.norm(acceleration) > 1e-4:
        action = "CONTROLLER_BRAKE_AND_REPLAN"
    else:
        action = "REQUEST_FRESH_ESTIMATE_AND_CONTROLLER_HOLD_ASSESSMENT"
    plan.update(status=status, reason=reason, selected_index=None, total_cost=None, trajectory=None,
                protective_action={"recommendation": action,
                                   "validated_safe_trajectory": False,
                                   "note": "No braking or hover trajectory is certified by this planner."})
    return plan


def _integral(values: np.ndarray, times: np.ndarray) -> float:
    # Works with NumPy versions both before and after np.trapezoid was added.
    return float(np.sum((values[1:] + values[:-1]) * np.diff(times) / 2.0))


def plan_trajectory(
    position: Sequence[float],
    velocity: Sequence[float],
    acceleration: Sequence[float],
    target_position: Sequence[float],
    target_velocity: Sequence[float],
    candidate_endpoints: Iterable[Sequence[float]],
    params: PlanningParams | None = None,
    obstacles: Iterable[SphereObstacle | Mapping[str, Any]] = (),
    observation_age_s: float = 0.0,
    mode: str = "perception",
    target_support_offsets: Iterable[Sequence[float]] = ((0.0, 0.0, 0.0),),
    image_center_reference_offset_ned_m: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Enumerate endpoint × duration candidates and choose a sampled-feasible one.

    target_position is the estimated position at its observation timestamp.
    Prediction advances it by target_velocity * (observation_age_s + t).
    Camera projection uses drone_position + camera_offset_ned_m without
    attitude rotation. Target distance constraints use the drone body center.
    Modes share every candidate, constraint and state; efficiency sets ONLY
    the image-center cost weight to zero. Equal costs select the earlier index.
    All returned records are JSON serializable. INVALID/STALE/NO_FEASIBLE plans
    contain no selected trajectory, and protective recommendations are not
    dynamically verified fallbacks.
    """
    plan: dict[str, Any] = {
        "status": "INVALID", "mode": mode, "selected_index": None,
        "total_cost": None, "trajectory": None, "candidates": [],
        "camera_model": "fixed_yaw_0_forward_positive_x",
        "research_scope": "geometry_only_prototype",
        "feasibility_scope": "sampled_times_only",
        "optimality_scope": "finite_candidate_minimum_only",
        "limitations": ["No continuous collision avoidance guarantee",
                        "Fixed yaw camera translation only; no attitude rotation",
                        "No occlusion prediction", "No target identity confidence calibration",
                        "No simulator ground truth is read", "No verified braking fallback"],
    }
    p = v = a = None
    try:
        params = params if params is not None else PlanningParams()
        _validate_params(params)
        p = _vector(position, "position")
        v = _vector(velocity, "velocity")
        a = _vector(acceleration, "acceleration")
        target_p = _vector(target_position, "target_position")
        target_v = _vector(target_velocity, "target_velocity")
        quality_offset = _vector(image_center_reference_offset_ned_m, "image centering reference offset")
        if abs(float(quality_offset[0])) > 1e-9:
            raise ValueError("Image centering reference must retain representative axial depth")
        supports = np.array([_vector(item, "target footprint offset") for item in target_support_offsets])
        if len(supports) == 0 or len(supports) > 16:
            raise ValueError("Target footprint requires 1 to 16 support offsets")
        endpoints = [_vector(item, f"candidate endpoint {i}") for i, item in enumerate(candidate_endpoints)]
        sphere_obstacles = _normalize_obstacles(obstacles)
        age = float(observation_age_s)
        if not np.isfinite(age) or age < 0.0:
            raise ValueError("observation_age_s must be finite and nonnegative")
        if mode not in {"efficiency", "perception"}:
            raise ValueError("mode must be efficiency or perception")
        if not endpoints:
            raise ValueError("candidate_endpoints cannot be empty")
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return _failure(plan, "INVALID", str(error), v, a)

    plan["params"] = asdict(params)
    plan["input_state"] = {"position": p.tolist(), "velocity": v.tolist(),
                           "acceleration": a.tolist(), "target_estimate": target_p.tolist(),
                           "target_velocity_estimate": target_v.tolist(), "observation_age_s": age,
                           "target_support_offsets": supports.tolist(),
                           "image_center_reference_offset_ned_m": quality_offset.tolist(),
                           "image_center_reference_scope": "caller RGB bearing on representative axial-depth plane; distance and supports retain surface semantics",
                           "support_scope": "caller-estimated footprint; no true object hull or occlusion guarantee",
                           "obstacles": [asdict(item) for item in sphere_obstacles]}
    weights = {"terminal_observation": params.terminal_observation_weight,
               "time_efficiency": params.time_efficiency_weight,
               "image_center": params.image_center_weight if mode == "perception" else 0.0,
               "jerk_energy": params.jerk_energy_weight,
               "acceleration_energy": params.acceleration_energy_weight,
               "safety_margin": params.safety_margin_weight}
    plan["cost_weights"] = weights
    low, high = np.array(params.position_min), np.array(params.position_max)
    if np.any(p < low) or np.any(p > high):
        return _failure(plan, "INVALID", "initial position violates position bounds", v, a, True)
    if np.linalg.norm(v) > params.max_speed_mps + 1e-9:
        return _failure(plan, "INVALID", "initial velocity norm exceeds speed limit", v, a, True)
    if np.linalg.norm(a) > params.max_acceleration_mps2 + 1e-9:
        return _failure(plan, "INVALID", "initial acceleration norm exceeds acceleration limit", v, a, True)
    if age > params.max_observation_age_s:
        return _failure(plan, "STALE", "target observation exceeds maximum age", v, a)

    h_tan = tan(radians(params.horizontal_fov_deg) / 2.0)
    v_tan = tan(radians(params.vertical_fov_deg) / 2.0)
    camera_offset = np.array(params.camera_offset_ned_m)
    tolerance = 1e-8
    for endpoint_index, endpoint in enumerate(endpoints):
        for duration in params.durations:
            index = len(plan["candidates"])
            duration = float(duration)
            times = np.linspace(0.0, duration, int(ceil(duration / params.sample_dt_s)) + 1)
            record: dict[str, Any] = {"index": index, "endpoint_index": endpoint_index,
                                      "endpoint": endpoint.tolist(), "duration_s": duration,
                                      "sample_count": len(times), "sample_dt_max_s": float(np.max(np.diff(times))),
                                      "feasible": False, "reject_reasons": [], "coefficients": None,
                                      "cost_components": None, "weighted_cost_components": None,
                                      "total_cost": None, "sampled_margins": None}
            plan["candidates"].append(record)
            try:
                coefficients = quintic_coefficients(p, v, a, endpoint, duration)
                record["coefficients"] = coefficients.tolist()
                positions = evaluate_quintic(coefficients, times)
                velocities = evaluate_quintic(coefficients, times, 1)
                accelerations = evaluate_quintic(coefficients, times, 2)
                jerks = evaluate_quintic(coefficients, times, 3)
                target_positions = target_p + (age + times[:, None]) * target_v
                relative = target_positions - positions
                camera_relative = target_positions - (positions + camera_offset)
                if not all(np.all(np.isfinite(item)) for item in
                           (positions, velocities, accelerations, jerks, relative, camera_relative)):
                    raise ValueError("nonfinite sampled trajectory or target prediction")
                boundary = np.min(np.minimum(positions - low, high - positions), axis=1)
                speed = np.linalg.norm(velocities, axis=1)
                acceleration_norm = np.linalg.norm(accelerations, axis=1)
                distance = np.linalg.norm(relative, axis=1)
                depth_margin = camera_relative[:, 0] - params.camera_near_depth_m
                projection_depth = np.maximum(camera_relative[:, 0], params.camera_near_depth_m)
                image_center = (camera_relative + quality_offset)[:, 1:] / (projection_depth[:, None] * np.array([h_tan, v_tan]))
                footprint = camera_relative[:, None, :] + supports[None, :, :]
                footprint_depth_margin = np.min(footprint[:, :, 0], axis=1) - params.camera_near_depth_m
                footprint_depth = np.maximum(footprint[:, :, 0], params.camera_near_depth_m)
                footprint_image = footprint[:, :, 1:] / (footprint_depth[:, :, None] * np.array([h_tan, v_tan]))
                fov_margin = 1.0 - params.image_border_margin_fraction - np.max(np.abs(footprint_image), axis=(1, 2))
                target_clearance = distance - params.min_target_distance_m
                obstacle_margins = [np.linalg.norm(positions - np.array(item.center), axis=1)
                                    - item.radius - params.drone_radius_m - params.obstacle_margin_m
                                    for item in sphere_obstacles]
                checked = [("POSITION_BOUNDS", boundary),
                           ("SPEED_LIMIT", params.max_speed_mps - speed),
                           ("ACCELERATION_LIMIT", params.max_acceleration_mps2 - acceleration_norm),
                           ("TARGET_CAMERA_DEPTH", np.minimum(depth_margin, footprint_depth_margin)),
                           ("TARGET_FOV", fov_margin),
                           ("TARGET_MIN_DISTANCE", target_clearance)]
                checked.extend((f"SPHERE_OBSTACLE:{i}:{item.label}", margin)
                               for i, (item, margin) in enumerate(zip(sphere_obstacles, obstacle_margins)))
                if not all(np.all(np.isfinite(margins)) for _, margins in checked):
                    raise ValueError("nonfinite sampled constraint margins")
                record["reject_reasons"] = [reason for reason, margins in checked if float(np.min(margins)) < -tolerance]
                record["sampled_margins"] = {reason: float(np.min(margins)) for reason, margins in checked}
                soft_clearances = [boundary, target_clearance] + obstacle_margins
                clearance_penalty = np.mean(np.array([
                    np.square(np.clip((params.preferred_clearance_m - margins)
                                      / params.preferred_clearance_m, 0.0, 1.0))
                    for margins in soft_clearances]), axis=0)
                components = {
                    "terminal_observation": float(((distance[-1] - params.desired_observation_distance_m)
                                                   / params.desired_observation_distance_m)**2),
                    "time_efficiency": duration / params.time_reference_s,
                    "image_center": _integral(np.sum(image_center**2, axis=1), times) / duration,
                    "jerk_energy": _integral(np.sum(jerks**2, axis=1), times)
                                   / (params.jerk_reference_mps3**2 * params.time_reference_s),
                    "acceleration_energy": _integral(np.sum(accelerations**2, axis=1), times)
                                           / (params.acceleration_reference_mps2**2 * params.time_reference_s),
                    "safety_margin": _integral(clearance_penalty, times) / duration,
                }
                weighted = {name: float(value * weights[name]) for name, value in components.items()}
                total = float(sum(weighted.values()))
                if not np.isfinite(total) or not all(np.isfinite(value) for value in components.values()):
                    raise ValueError("nonfinite candidate cost")
                record.update(cost_components=components, weighted_cost_components=weighted,
                              total_cost=total, feasible=not record["reject_reasons"])
            except (ValueError, np.linalg.LinAlgError, OverflowError, FloatingPointError) as error:
                record.update(feasible=False, reject_reasons=[f"NUMERICAL_FAILURE:{error}"])

    feasible = [item for item in plan["candidates"] if item["feasible"]]
    if not feasible:
        return _failure(plan, "NO_FEASIBLE", "no enumerated candidate passed sampled constraints", v, a)
    selected = min(feasible, key=lambda item: (item["total_cost"], item["index"]))
    plan.update(status="OK", reason="selected minimum cost among sampled-feasible finite candidates",
                selected_index=selected["index"], total_cost=selected["total_cost"],
                trajectory={"duration_s": selected["duration_s"], "coefficients": selected["coefficients"],
                            "endpoint": selected["endpoint"]}, protective_action=None)
    return plan
