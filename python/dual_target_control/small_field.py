"""Reference and telemetry checks for the small-field AirSim task."""
import math

import numpy as np
from numpy.polynomial import Polynomial
from scipy.spatial.transform import Rotation, Slerp


POSITION_ROUNDOFF_M = 1e-9
POINT_PROTOCOL_REVISION = 17
BODY_PROTOCOL_REVISION = 18


def scaled_vehicle_bounds(visual):
    """Scale the UE render bounds about the vehicle origin, in body FRD metres."""
    scale = visual["drone_scale"]
    bounds = np.asarray(visual["drone_unscaled_bounds_frd_m"], dtype=float) * scale
    if (not math.isfinite(scale) or scale <= 0. or bounds.shape != (2, 3)
            or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1])):
        raise ValueError("小场地可见机身外框无效")
    return bounds.tolist()


def target_distance(positions, target, rules, orientation_ned_xyzw=None):
    """Distance from a target point to the rotated visible vehicle box."""
    delta_zup = np.asarray(target) - np.asarray(positions)
    bounds = rules.get("vehicle_bounds_frd_m")
    if bounds is None:  # Frozen point-based records.
        return np.linalg.norm(delta_zup, axis=-1)
    if orientation_ned_xyzw is None:
        raise ValueError("机身外框判定需要同一时刻的姿态四元数")
    world_ned = delta_zup * [1., 1., -1.]
    body_frd = Rotation.from_quat(orientation_ned_xyzw).apply(world_ned, inverse=True)
    lower, upper = np.asarray(bounds)
    return np.linalg.norm(body_frd - np.clip(body_frd, lower, upper), axis=-1)


def inside_field(position, rules):
    lower, upper = np.asarray(rules["bounds_zup_m"])
    position = np.asarray(position)
    return bool(np.all((position >= lower - POSITION_ROUNDOFF_M)
                       & (position <= upper + POSITION_ROUNDOFF_M)))


def expected_dynamics(receipt):
    return (receipt.get("verified") is True and receipt.get("model") == "GenericQuadX"
            and math.isclose(receipt["mass_kg"], 1.0, rel_tol=2e-5)
            and math.isclose(receipt["nominal_twr"], 2.5, rel_tol=2e-5)
            and math.isclose(receipt["max_thrust_per_rotor_n"], 6.13, abs_tol=.02))


def polynomial_extrema(poly):
    roots = poly.deriv().roots()
    interior = roots.real[(np.abs(roots.imag) < 1e-8) & (roots.real > 0) & (roots.real < 1)]
    values = poly(np.r_[0., interior, 1.])
    return float(values.min()), float(values.max())


def reference_checks(curve):
    rules = curve.spec["small_field"]
    coordinate_extrema = []
    peaks = np.zeros(2)
    for segment in curve.segments:
        axes = [Polynomial(axis) for axis in segment.coefficients.T]
        coordinate_extrema.append([polynomial_extrema(axis) for axis in axes])
        for order in (1, 2):
            norm_squared = sum((axis.deriv(order) / segment.duration_s**order)**2 for axis in axes)
            peaks[order - 1] = max(peaks[order - 1], math.sqrt(max(0., polynomial_extrema(norm_squared)[1])))
    extrema = np.asarray(coordinate_extrema)
    min_pos, max_pos = extrema[:, :, 0].min(axis=0), extrema[:, :, 1].max(axis=0)
    in_bounds = inside_field([min_pos, max_pos], rules)
    at_rest = bool(np.linalg.norm(curve.sample(curve.duration_s, derivative=1)) <= 1e-8)
    passed = in_bounds and at_rest and peaks[0] <= rules["reference_speed_mps"] and peaks[1] <= rules["reference_acceleration_mps2"]
    return dict(passed=bool(passed), peak_speed_mps=float(peaks[0]), peak_acceleration_mps2=float(peaks[1]),
                min_position_zup_m=min_pos.tolist(), max_position_zup_m=max_pos.tolist(),
                inside_field=in_bounds, terminal_reference_at_rest=at_rest)


def total_tilt_deg(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw
    norm_squared = x*x + y*y + z*z + w*w
    if not math.isfinite(norm_squared) or norm_squared <= 0.:
        raise ValueError("AirSim 姿态四元数无效")
    down_dot = 1. - 2. * (x*x + y*y) / norm_squared
    return math.degrees(math.acos(min(1., max(-1., down_dot))))


def ordered_waypoints(times, positions, speeds, spec, orientation_ned_xyzw=None):
    rules = spec["small_field"]
    cursor = 0
    results = []
    for index, point in enumerate(spec["waypoints"][:-1]):
        distances = target_distance(positions, point, rules, orientation_ned_xyzw)
        eligible = distances[cursor:] <= rules["waypoint_radius_m"]
        if spec["policy"] == "pass":
            eligible &= speeds[cursor:] >= rules["minimum_pass_speed_mps"]
        candidates = np.flatnonzero(eligible)
        hit = cursor + int(candidates[0]) if candidates.size else None
        results.append(dict(index=index, point=point, minimum_distance_m=float(distances.min()),
                            hit_time_s=float(times[hit]) if hit is not None else None,
                            hit_distance_m=float(distances[hit]) if hit is not None else None,
                            hit_speed_mps=float(speeds[hit]) if hit is not None else None, passed=hit is not None))
        cursor = hit + 1 if hit is not None else len(times)
    return results


def terminal_hold(times, positions, velocities, goal, duration, rules, orientation_ned_xyzw=None):
    index = int(np.searchsorted(times, duration))
    limit_p, limit_v = rules["terminal_position_m"], rules["terminal_speed_mps"]
    use_vehicle_box = "vehicle_bounds_frd_m" in rules
    distances = target_distance(positions, goal, rules, orientation_ned_xyzw)
    last_distance = float(distances[-1] if use_vehicle_box else np.linalg.norm(positions[-1] - goal))
    last_speed = float(np.linalg.norm(velocities[-1]))
    result = dict(reference_end_position_error_m=None, reference_end_speed_mps=None,
                  end_position_error_m=last_distance, end_speed_mps=last_speed, continuous_s=0.,
                  minimum_distance_m=float(distances.min()),
                  reference_end_passed=False, end_passed=last_distance <= limit_p and last_speed <= limit_v,
                  first_failed_time_s=None, passed=False)
    if not 0 < index < len(times) or times[index] - times[index-1] > rules["max_sample_gap_s"] + 1e-9:
        return result
    weight = (duration - times[index-1]) / (times[index] - times[index-1])
    position = positions[index-1] + weight * (positions[index] - positions[index-1])
    velocity = velocities[index-1] + weight * (velocities[index] - velocities[index-1])
    orientation = None
    if use_vehicle_box:
        adjacent = Rotation.from_quat(orientation_ned_xyzw[index-1:index+1])
        orientation = Slerp([0., 1.], adjacent)(weight).as_quat()
    distance = float(target_distance(position, goal, rules, orientation))
    speed = float(np.linalg.norm(velocity))
    at_end = distance <= limit_p and speed <= limit_v
    result.update(reference_end_position_error_m=distance, reference_end_speed_mps=speed, reference_end_passed=at_end)
    if not at_end:
        result["first_failed_time_s"] = duration
        return result
    previous = duration
    for sample in range(index, len(times)):
        current = float(times[sample])
        sample_distance = distances[sample] if use_vehicle_box else np.linalg.norm(positions[sample] - goal)
        passed = (current - previous <= rules["max_sample_gap_s"] + 1e-9
                  and sample_distance <= limit_p
                  and np.linalg.norm(velocities[sample]) <= limit_v)
        if not passed:
            result["first_failed_time_s"] = current
            break
        previous = current
    result["continuous_s"] = previous - duration
    result["passed"] = (result["first_failed_time_s"] is None and result["end_passed"]
                        and result["continuous_s"] >= rules["terminal_hold_s"] - 1e-9)
    return result


def flight_metrics(states, curve, fault=None):
    rules = curve.spec["small_field"]
    stamps = np.asarray([row["sim_timestamp_ns"] for row in states], dtype=np.int64)
    times = (stamps - stamps[0]) * 1e-9
    positions = np.asarray([row["position_zup_m"] for row in states])
    velocities = np.asarray([row["velocity_zup_mps"] for row in states])
    speeds = np.linalg.norm(velocities, axis=1)
    orientations = np.asarray([row["orientation_ned_xyzw"] for row in states])
    tilts = [total_tilt_deg(orientation) for orientation in orientations]
    gap = float(np.diff(times).max(initial=0.))
    peak_speed = float(speeds.max())
    inside = inside_field(positions, rules)
    if fault:
        gap = max(gap, abs(fault["sample_gap_s"]))
        peak_speed = max(peak_speed, float(np.linalg.norm(fault["velocity_zup_mps"])))
        inside = inside and inside_field(fault["position_zup_m"], rules)
        if fault["orientation_ned_xyzw"] is not None:
            tilts.append(total_tilt_deg(fault["orientation_ned_xyzw"]))
    waypoint_rows = ordered_waypoints(times, positions, speeds, curve.spec, orientations)
    hold = terminal_hold(times, positions, velocities, curve.spec["waypoints"][-1], curve.duration_s, rules, orientations)
    references = curve.sample(np.minimum(times, curve.duration_s))
    errors = np.linalg.norm(positions - references, axis=1)
    checks = dict(waypoints=all(row["passed"] for row in waypoint_rows), terminal_hold=hold["passed"],
                  tilt=max(tilts) <= rules["max_tilt_deg"] and (not fault or fault["orientation_ned_xyzw"] is not None),
                  spatial_bounds=inside, sample_clock=bool(np.all(np.diff(stamps) > 0)
                  and gap <= rules["max_sample_gap_s"] + 1e-9 and (not fault or fault["sample_gap_s"] >= 0)))
    metrics = dict(control_error_gates_passed=all(checks.values()), acceptance_checks=checks,
                waypoint_results=waypoint_rows, terminal_hold=hold, max_sample_gap_s=gap,
                peak_tilt_deg=max(tilts), measured_peak_speed_mps=peak_speed,
                max_position_error_m=float(errors.max()), rmse_position_m=float(np.sqrt(np.mean(errors**2))),
                tracking_time_s=float(times[-1]))
    if "vehicle_bounds_frd_m" in rules:
        metrics["acceptance_vehicle_bounds_frd_m"] = rules["vehicle_bounds_frd_m"]
    return metrics
