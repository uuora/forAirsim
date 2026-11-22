"""Planning-only whole-route proxy operations with reusable sampled arrays.

The sample grid, lag recurrence, gates and quality function match the shared
route evaluator. No simulator I/O, reduced-fidelity check or search is done
here. Flight/log metrics continue to use ``route_curve.route_metrics``.
"""
from __future__ import annotations

import math

import numpy as np

from dual_target_control.route_curve import waypoint_results
from .curve_task import _split_half
from .curves import PolynomialSegment


def rollout_reference(times, reference_positions, reference_velocities,
                      reference_accelerations, command_limits, tracking=None):
    """Keep the scalar recurrence while avoiding per-sample record dictionaries.

    The returned arrays contain the state at every supplied reference sample;
    command columns contain speed/slew clipping for the intervening steps.
    """
    q, qv, qa = reference_positions, reference_velocities, reference_accelerations
    positions = np.empty_like(q)
    velocities = np.empty_like(qv)
    clipping = np.empty((len(times) - 1, 2), dtype=bool)
    p = q[0].tolist()
    v = qv[0].tolist()
    previous = v.copy()
    tracking = tracking or {}
    tau = np.asarray(tracking.get('proxy_lag_s', [.9] * 3))
    delay = np.asarray(tracking.get('proxy_delay_s', [0.] * 3))
    feedforward = tracking.get('acceleration_feedforward_s', [.9] * 3)
    delayed_index = np.searchsorted(times, times[:-1, None] - delay, side='right') - 1
    command_history = [v.copy()]  # Pre-reference command for each delayed axis.
    cap = command_limits['max_speed_mps']
    slew = command_limits['max_slew_mps2']
    dts = np.diff(times).tolist()
    decays = np.exp(-np.diff(times)[:, None] / tau).tolist()
    tau = tau.tolist()

    def length(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

    for i, (point, velocity, acceleration) in enumerate(zip(q.tolist(), qv.tolist(), qa.tolist())):
        positions[i] = p
        velocities[i] = v
        if i == len(times) - 1:
            break
        dt = dts[i]
        raw = [velocity[k] + 2 * (point[k] - p[k]) + .8 * (velocity[k] - v[k]) + feedforward[k] * acceleration[k]
               for k in range(3)]
        raw_norm = length(raw)
        speed_scale = min(1., cap / max(raw_norm, 1e-300))
        speed = [value * speed_scale for value in raw]
        change = [speed[k] - previous[k] for k in range(3)]
        change_norm = length(change)
        slew_scale = min(1., slew * dt / max(change_norm, 1e-300))
        limited = [previous[k] + change[k] * slew_scale for k in range(3)]
        clipping[i] = (bool(raw_norm > cap), bool(change_norm > slew * dt))
        command_history.append(limited)
        applied = [command_history[delayed_index[i, k] + 1][k] for k in range(3)]
        e = decays[i]
        p = [p[k] + applied[k] * dt + (v[k] - applied[k]) * tau[k] * (1 - e[k]) for k in range(3)]
        v = [applied[k] + (v[k] - applied[k]) * e[k] for k in range(3)]
        previous = limited
    return positions, velocities, clipping


def sampled_route_metrics(curve, times, positions, velocities, clipping,
                          reference_positions, reference_velocities):
    """Reuse reference samples on the complete, exact rollout grid.

    This deliberately retains terminal interpolation and trapezoid arithmetic
    from ``route_metrics``. It accepts only the planning grid ending exactly
    at T; arbitrary flight samples must use the shared log metric function.
    """
    t, p, v = times, positions, velocities
    T = curve.duration_s
    if len(t) < 2 or t[0] != 0 or np.any(np.diff(t) <= 0) or t[-1] != T:
        raise ValueError('规划采样须完整且精确覆盖参考时间')
    expected = (len(t), 3)
    if any(np.shape(value) != expected for value in
           (p, v, reference_positions, reference_velocities)) or np.shape(clipping) != (len(t) - 1, 2):
        raise ValueError('规划状态与参考采样维数不一致')
    end = int(np.searchsorted(t, T))
    left = max(0, end - 1)
    if t[end] - t[left] > .1:
        raise ValueError('终端插值间隔过大')
    f = 0 if t[end] == t[left] else (T - t[left]) / (t[end] - t[left])
    pe = p[left] * (1 - f) + p[end] * f
    ve = v[left] * (1 - f) + v[end] * f
    keep = t < T
    mt = np.r_[t[keep], T]
    mp = np.vstack([p[keep], pe])
    # The caller sampled the same curve at exactly these times. Reusing q
    # avoids a fourth vector sample and two separate scalar endpoint samples.
    mq = np.vstack([reference_positions[keep], reference_positions[end]])
    err = np.linalg.norm(mq - mp, axis=1)
    rmse = float(np.sqrt(np.sum(.5 * (err[:-1] ** 2 + err[1:] ** 2) * np.diff(mt)) / T))
    peak = float(err.max())
    terminal = float(np.linalg.norm(reference_positions[end] - pe))
    terminal_v = float(np.linalg.norm(reference_velocities[end] - ve))
    waypoints = waypoint_results(curve, t, p, v)
    checks = dict(sampled_peak_position=peak <= curve.spec.get("peak_position_limit_m", .5), terminal_position=terminal <= .25,
                  terminal_velocity=terminal_v <= .5, waypoints=all(r['passed'] for r in waypoints))
    weights = np.maximum(0, np.minimum(T, t[1:]) - t[:-1])
    n = len(curve.spec['waypoints'])
    energy = curve.jerk_energy
    return dict(reference_duration_s=T, position_rmse_m=rmse, max_position_error_m=peak,
                terminal_position_error_m=terminal, terminal_velocity_error_mps=terminal_v,
                control_error_checks=checks, control_error_gates_passed=all(checks.values()),
                task_success=all(checks.values()),
                speed_clipped_time_fraction=float(np.dot(weights, clipping[:, 0]) / T),
                command_slew_clipped_time_fraction=float(np.dot(weights, clipping[:, 1]) / T),
                max_observed_state_gap_s=float(np.diff(t).max()), jerk_energy_m2_s5=energy,
                waypoint_results=waypoints,
                quality_term=.7 * T / (T + 5 * n) + .1 * energy / (energy + 30 * n) + .2 * rmse / (rmse + .25))


def route_derivative_bounds(segments, *, depth=2):
    """Factor identical derivative prefixes, retaining the existing hull bound.

    Power-to-Bernstein conversion is done once per segment. Physical derivative
    controls, subdivisions, norm calculation and ordering use the same
    arithmetic as three calls to ``subdivided_derivative_norm_bound``.
    """
    if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)) or not 0 <= depth <= 10:
        raise ValueError('depth must be an integer in [0,10]')
    maxima = [0., 0., 0.]
    if not segments:
        raise ValueError('segments must be nonempty')
    for segment in segments:
        if not isinstance(segment, PolynomialSegment):
            raise ValueError('segment must be a PolynomialSegment')
        # An explicit constant hold has exactly zero physical derivatives.
        if segment.degree == 0:
            continue
        control = segment.bezier_control_points()
        for order in range(3):
            try:
                with np.errstate(over='raise', invalid='raise', divide='raise'):
                    n = len(control) - 1
                    control = (n * np.diff(control, axis=0) / segment.duration_s
                               if n else np.zeros((1, control.shape[1])))
            except FloatingPointError as error:
                raise ValueError('derivative controls exceeded finite floating-point range') from error
            if not np.isfinite(control).all():
                raise ValueError('derivative controls exceeded finite floating-point range')
            leaves = [control]
            for _ in range(int(depth)):
                leaves = [part for leaf in leaves for part in _split_half(leaf)]
            bound = max(math.hypot(*row) for leaf in leaves for row in leaf)
            if not math.isfinite(bound):
                raise ValueError('derivative norm bound exceeded numerical range')
            maxima[order] = max(maxima[order], float(bound))
    return maxima
