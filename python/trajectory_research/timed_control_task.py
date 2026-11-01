"""Time-indexed, fixed-controller objective for the declared curve task.

The reference uses the existing endpoint-preserving parameterization and its
robust *reference-geometry* checks. Execution is a separate first-order lagged
point-mass research model. Its nominal static-sphere and workspace checks use
a Lipschitz sampling-gap margin. They do not establish tracking safety for
uncertain obstacles, AirSim dynamics, a non-point aircraft, or real flight.

Every candidate is evaluated through the entire [0,T] reference interval. The
five feedback features come from that same rollout, without extra evaluations.
Weights, scales and gates are fixed here, not chosen from later validation.
"""
from __future__ import annotations

from dataclasses import asdict
import math
from numbers import Real
import time

import numpy as np

from .curve_task import WORKSPACE_LOWER, WORKSPACE_UPPER
from .parameterizations import curve_from_parameters
from . import perturbation_task as reference_task
from .quality_bound import quality_terms
from .timed_tracking import TrackingCondition, run_timed_tracking


MAX_POSITION_ERROR_M = .5
MAX_TERMINAL_POSITION_ERROR_M = .25
MAX_TERMINAL_VELOCITY_ERROR_MPS = .5
TRACKING_RMSE_SCALE_M = .25
DISTANCE_CUSHION_M = 1e-10
BOUND_CUSHION = 1e-10
FEATURE_NAMES = (
    "rmse_over_0p25_m_capped4",
    "peak_error_over_0p5_m_capped4",
    "terminal_velocity_error_over_0p5_mps_capped4",
    "command_slew_clipped_time_fraction",
    "peak_error_normalized_time",
)
SCOPE = (
    "Independent fixed time-indexed velocity tracker and first-order lagged point mass; "
    "not production RollingController, AirSim, or real flight. Reference geometry uses "
    "the existing deterministic uncertainty envelope. Executed point-path checks use "
    "nominal static spheres and a model-specific Lipschitz sampling-gap margin only; "
    "no robust tracking-to-uncertain-obstacle, rotor, attitude, thrust, wind, or sensing claim. "
    "Position-error gates are sampled metrics. Command clipping is not motor saturation."
)


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
            or not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def control_features(metrics):
    """Normalize the five declared diagnostics without learning/calibration.

    The first three features are capped at four, and the two fractions at one.
    Raw untruncated metrics remain in every complete objective result.
    """
    rmse = _nonnegative(metrics["position_rmse_m"], "position RMSE")
    peak = _nonnegative(metrics["max_position_error_m"], "peak position error")
    terminal_velocity = _nonnegative(metrics["terminal_velocity_error_mps"], "terminal velocity error")
    slew = _nonnegative(metrics["command_slew_clipped_time_fraction"], "command slew clipped fraction")
    peak_u = _nonnegative(metrics["peak_position_error_u"], "normalized peak time")
    if slew > 1. + 1e-12 or peak_u > 1. + 1e-12:
        raise ValueError("diagnostic fractions must lie in [0,1]")
    return [min(4., rmse / TRACKING_RMSE_SCALE_M), min(4., peak / MAX_POSITION_ERROR_M),
            min(4., terminal_velocity / MAX_TERMINAL_VELOCITY_ERROR_MPS),
            min(1., slew), min(1., peak_u)]


def _clipping_intervals(rollout, flag_name):
    """Compress observed held-command clipping flags into [start,end) spans."""
    times = rollout["times_s"]
    flags = rollout[flag_name]
    if len(flags) != len(times)-1:
        raise ValueError("clipping flags do not match the rollout intervals")
    intervals = []
    start = None
    for index, clipped in enumerate(flags):
        if not isinstance(clipped, (bool, np.bool_)):
            raise ValueError("clipping flags must be boolean")
        if clipped and start is None:
            start = float(times[index])
        if start is not None and (not clipped or index == len(flags)-1):
            end = float(times[index+1] if clipped else times[index])
            intervals.append([start, end])
            start = None
    return intervals


def actual_nominal_path_checks(rollout, *, layout="central", condition=None):
    """Model-specific continuous point-path enclosures from sampled states.

    On every zero-order-held command interval, velocity is a convex combination
    of its initial velocity and the held command. If both are within the speed
    ball, the entire interval has ||v|| <= v_cap. Every time is at most h_max/2
    away from a sampled timestamp; 1-Lipschitz signed distances can therefore
    decrease by at most v_cap*h_max/2 between samples. Workspace face distances
    obey the same bound. The positive cushion is ordinary floating allowance,
    not directed interval arithmetic.

    This checker is for traces produced by run_timed_tracking. Checking sampled
    velocities alone would not justify the continuous-speed assumption; held
    command norms and the initial velocity are checked explicitly as well.
    """
    if condition is None:
        condition = TrackingCondition()
    if not isinstance(condition, TrackingCondition):
        raise ValueError("condition must be a TrackingCondition")
    scene = reference_task._layout(layout)
    positions = np.asarray(rollout["actual_position_m"], dtype=float)
    velocities = np.asarray(rollout["actual_velocity_mps"], dtype=float)
    commands = np.asarray(rollout["velocity_command_mps"], dtype=float)
    times = np.asarray(rollout["times_s"], dtype=float)
    if times.ndim != 1 or len(times) < 2 or positions.shape != (len(times), 3) \
            or velocities.shape != positions.shape or commands.shape != (len(times)-1, 3) \
            or not all(np.isfinite(x).all() for x in (positions, velocities, commands, times)) \
            or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("a complete finite ordered 3D timed-tracker trace is required")
    if "condition" in rollout and rollout["condition"] != asdict(condition):
        raise ValueError("trace condition differs from the requested condition")
    cap = condition.max_command_speed_mps
    max_command = float(np.max(np.linalg.norm(commands, axis=1)))
    initial_speed = float(np.linalg.norm(velocities[0]))
    max_sampled_speed = float(np.max(np.linalg.norm(velocities, axis=1)))
    speed_assumption = max(max_command, initial_speed, max_sampled_speed) <= cap + 1e-10
    # The observed command/initial norms also cover tiny projection roundoff.
    effective_cap = max(cap, max_command, initial_speed)
    max_step = float(np.max(np.diff(times)))
    gap = effective_cap * max_step / 2.
    lower_slack = np.min(positions - np.asarray(WORKSPACE_LOWER), axis=0) - gap - DISTANCE_CUSHION_M
    upper_slack = np.min(np.asarray(WORKSPACE_UPPER) - positions, axis=0) - gap - DISTANCE_CUSHION_M
    checks = dict(model_velocity_bound=speed_assumption,
                  actual_nominal_workspace=bool(speed_assumption and np.all(lower_slack >= 0) and np.all(upper_slack >= 0)))
    spheres = []
    for index, obstacle in enumerate(scene.obstacles):
        sampled_net = float(np.min(np.linalg.norm(positions - np.asarray(obstacle.centre), axis=1)) - obstacle.radius)
        lower = sampled_net - gap - DISTANCE_CUSHION_M
        passed = bool(speed_assumption and lower >= reference_task.CLEARANCE)
        checks[f"actual_nominal_obstacle_{index}"] = passed
        spheres.append(dict(obstacle_index=index, centre_m=list(obstacle.centre), radius_m=obstacle.radius,
                            min_sampled_surface_clearance_m=sampled_net,
                            surface_clearance_lower_bound_m=lower if speed_assumption else None,
                            required_surface_clearance_m=reference_task.CLEARANCE, passed=passed))
    return dict(status="BOUNDED" if speed_assumption else "UNRESOLVED_SPEED_ASSUMPTION",
                checks=checks, sampling_gap_margin_m=gap,
                numerical_distance_cushion_m=DISTANCE_CUSHION_M,
                effective_speed_bound_mps=effective_cap, actual_max_timestep_s=max_step,
                initial_speed_mps=initial_speed, maximum_held_command_speed_mps=max_command,
                maximum_sampled_actual_speed_mps=max_sampled_speed,
                workspace_lower_face_clearance_bounds_m=lower_slack.tolist() if speed_assumption else None,
                workspace_upper_face_clearance_bounds_m=upper_slack.tolist() if speed_assumption else None,
                nominal_spheres=spheres,
                scope="Continuous-path sufficient enclosure only for the declared ZOH lagged point-mass model and nominal static geometry; unresolved speed assumption never passes.")


def timed_quality_lower_bound(parameters, *, family="single_bump"):
    """Cheap scalar-score bound for DEFAULT endpoints; no tracking prediction.

    Tracking RMSE contributes a nonnegative term and every infeasibility penalty
    is nonnegative. This uses the same default-endpoint jerk closed form as
    quality_bound.py and deliberately accepts no custom endpoint state.
    """
    terms = quality_terms(parameters, family=family)
    duration, energy = terms["duration_s"], terms["jerk_energy_m2_s5"]
    return float(.7 * duration / (duration + 5.) + .1 * energy / (energy + 30.) - BOUND_CUSHION)


def evaluate_timed_parameters(parameters, *, family="single_bump", representation="hermite",
                              layout="central", condition=None, include_trace=False):
    """Evaluate reference geometry and one fixed, synchronized tracking rollout.

    Each invocation has exactly one reference evaluation and one full rollout.
    The condition defaults to tau=.30 s; changing tau or other settings is an
    explicit new model condition. This function never selects validation lags.
    """
    started = time.perf_counter()
    if condition is None:
        condition = TrackingCondition()
    if not isinstance(condition, TrackingCondition):
        raise ValueError("condition must be a TrackingCondition")
    if not isinstance(include_trace, bool):
        raise ValueError("include_trace must be boolean")
    scene = reference_task._layout(layout)
    curve = curve_from_parameters(parameters, family=family, representation=representation)
    tick = time.perf_counter()
    reference = reference_task.evaluate_parameters(parameters, family=family,
        representation=representation, layout=scene, robust=True)
    reference_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    rollout = run_timed_tracking(curve, condition)
    rollout_seconds = time.perf_counter()-tick
    metrics = rollout["metrics"]
    features = control_features(metrics)
    diagnostics = dict(speed_clip_intervals_s=_clipping_intervals(rollout, "speed_clipped"),
                       command_slew_clip_intervals_s=_clipping_intervals(rollout, "command_slew_clipped"),
                       interval_convention="[start,end) during held command; no motor saturation interpretation")
    terminal_position = _nonnegative(metrics["terminal_position_error_m"], "terminal position error")
    tick = time.perf_counter()
    path = actual_nominal_path_checks(rollout, layout=scene, condition=condition)
    path_seconds = time.perf_counter()-tick
    checks = {f"reference_{name}": bool(passed) for name, passed in reference["score_details"]["checks"].items()}
    checks.update(path["checks"])
    checks.update(tracking_peak_error=bool(metrics["max_position_error_m"] <= MAX_POSITION_ERROR_M),
                  terminal_position_error=bool(terminal_position <= MAX_TERMINAL_POSITION_ERROR_M),
                  terminal_velocity_error=bool(metrics["terminal_velocity_error_mps"] <= MAX_TERMINAL_VELOCITY_ERROR_MPS))
    failed = sum(not passed for passed in checks.values())
    feasible = failed == 0
    duration = curve.duration_s
    energy = reference["score_details"]["jerk_energy_m2_s5"]
    rmse = metrics["position_rmse_m"]
    quality = .7 * duration / (duration + 5.) + .1 * energy / (energy + 30.) \
        + .2 * rmse / (rmse + TRACKING_RMSE_SCALE_M)
    score = quality if feasible else 1. + failed / len(checks) + quality
    if not math.isfinite(score):
        raise ValueError("timed objective produced a nonfinite score")
    result = dict(score=float(score), feasible=feasible, parameters=reference["parameters"],
        family=family, representation=representation, layout=scene.name,
        condition=asdict(condition), model=rollout["model"], scope=SCOPE,
        score_details=dict(quality_term=float(quality), duration_s=duration,
            tracking_condition=asdict(condition),
            jerk_energy_m2_s5=energy, tracking_rmse_m=rmse,
            tracking_metrics=metrics, control_features=features,
            tracking_diagnostics=diagnostics,
            control_feature_names=list(FEATURE_NAMES), checks=checks,
            failed_checks=failed, total_checks=len(checks),
            reference_geometry_feasible=reference["feasible"],
            actual_nominal_path=path,
            reference_evaluation_s=reference_seconds, rollout_evaluation_s=rollout_seconds,
            actual_path_check_s=path_seconds,
            quality_weights=dict(time=.7, jerk=.1, tracking=.2),
            normalization_scales=dict(time_s=5., jerk_m2_s5=30., tracking_rmse_m=TRACKING_RMSE_SCALE_M),
            tracking_error_gates=dict(peak_position_m=MAX_POSITION_ERROR_M,
                terminal_position_m=MAX_TERMINAL_POSITION_ERROR_M, terminal_velocity_mps=MAX_TERMINAL_VELOCITY_ERROR_MPS)),
        evaluation_wall_s=time.perf_counter()-started)
    if include_trace:
        result["tracking_trace"] = rollout
    return result


__all__ = ["FEATURE_NAMES", "SCOPE", "control_features", "actual_nominal_path_checks",
           "timed_quality_lower_bound", "evaluate_timed_parameters"]
