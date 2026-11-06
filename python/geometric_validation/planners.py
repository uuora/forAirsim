"""Three reference constructions with a common, local time-allocation search.

The outer search constrains reference kinematics, not AirSim tracking errors.
Only measured flights can establish tracking feasibility. No global time
optimality is claimed. Bezier and Hermite here are intentionally equivalent.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import time

import numpy as np
from numpy.polynomial import Polynomial

from .reference import Reference, FrozenReference


METHODS = ("bezier", "hermite", "bspline")
LABELS = {"bezier": "Bezier", "hermite": "Hermite", "bspline": "B-spline"}


def construct(method, points, durations, pass_speed=.8):
    """Return ascending coefficients in local normalized segment time [0,1]."""
    from scipy.interpolate import BPoly, make_interp_spline

    points = np.asarray(points, dtype=float)
    durations = np.asarray(durations, dtype=float)
    times = np.r_[0., np.cumsum(durations)]
    zeros = np.zeros_like(points)
    velocities = zeros.copy()
    for i in range(1, len(points)-1):
        direction = points[i+1]-points[i-1]
        velocities[i] = pass_speed * direction / np.linalg.norm(direction)
    if method == "hermite":
        ref = Reference(dict(start=points[0], points=points[1:], segment_durations=durations,
                             velocities=velocities))
        return np.asarray([[np.pad(p.coef,(0,8-len(p.coef))) for p in segment[0]]
                           for segment in ref.polynomials]).transpose(0, 2, 1)
    if method == "bezier":
        # SciPy stores these pieces in the Bernstein (Bezier) basis.
        curve = BPoly.from_derivatives(times, [list(x) for x in zip(points, velocities, zeros, zeros)])
    elif method == "bspline":
        curve = make_interp_spline(times, points, k=7,
            t=np.r_[[times[0]]*8, times[1:-1], [times[-1]]*8],
            bc_type=([(d, np.zeros(3)) for d in (1, 2, 3)],
                     [(d, np.zeros(3)) for d in (1, 2, 3)]))
    else:
        raise ValueError("Unknown planner: "+str(method))
    return np.asarray([[curve(float(t), nu=k)*T**k/math.factorial(k) for k in range(8)]
                       for t, T in zip(times[:-1], durations)])


def norm_peak(coefficients, durations, derivative):
    """Polynomial extrema include endpoints and every real stationary root."""
    peak = 0.
    for coef, duration in zip(coefficients, durations):
        polynomials = [Polynomial(coef[:, axis]).deriv(derivative)/duration**derivative for axis in range(3)]
        square = sum((p*p for p in polynomials), Polynomial([0.]))
        roots = square.deriv().roots()
        u = [0., 1.] + [float(r.real) for r in roots if abs(r.imag)<1e-7 and 0<r.real<1]
        peak = max(peak, math.sqrt(max(0., float(max(square(x) for x in u)))))
    return peak


def snap_integral(coefficients, durations):
    total = 0.
    for coef, duration in zip(coefficients, durations):
        for axis in range(3):
            snap = Polynomial(coef[:, axis]).deriv(4)/duration**4
            integral = (snap*snap).integ()
            total += duration * (integral(1)-integral(0))
    return float(total)


def plan(method, task, search):
    begin = time.perf_counter()
    from scipy.optimize import minimize

    points = np.asarray(task["points_ned"], dtype=float)
    initial = np.asarray(task["initial_segment_durations"], dtype=float)
    cache = {}
    def evaluate(values):
        key = tuple(map(float, values))
        if key not in cache:
            coef = construct(method, points, values, task["tangent_speed_mps"])
            # Keep segment constraints separate: a global max can hide every
            # finite-difference gradient when several segments share the peak.
            peaks = [norm_peak(coef[i:i+1], values[i:i+1], d)
                     for i in range(len(values)) for d in (1, 2, 3)]
            speeds = [np.linalg.norm(coef[i, 1]/values[i]) for i in range(1, len(values))]
            margin = np.r_[np.tile([search["max_speed_mps"], search["max_acceleration_mps2"],
                                   search["max_jerk_mps3"]],len(values))-peaks,
                           np.asarray(speeds)-search["min_knot_speed_mps"]]
            cache[key] = coef, margin
        return cache[key]
    result = minimize(lambda values: float(sum(values)), initial, method="SLSQP",
                      bounds=[tuple(search["segment_bounds_s"])]*len(initial),
                      constraints={"type": "ineq", "fun": lambda x: evaluate(x)[1]},
                      options={"maxiter": search["max_iterations"], "ftol": 1e-7, "eps": 1e-5})
    durations = result.x
    coefficients, margin = evaluate(durations)
    if not result.success or np.min(margin)<-1e-6:
        raise RuntimeError(f"{method} time search failed: {result.message}; margins={margin}")
    plan_data = dict(format="normalized_power_v1", method=method, points=points.tolist(),
        segment_durations=durations.tolist(), coefficients=coefficients.tolist(), yaw=task["yaw"],
        construction=("C3 septic endpoint interpolation; Bezier/Hermite equivalent" if method in ("bezier", "hermite")
                      else "clamped degree-7 interpolating B-spline, C6 interior"),
        planning=dict(location="offline_before_flight",
                      coefficient_evaluations=len(cache), optimizer="SciPy SLSQP",
                      success=bool(result.success), iterations=int(result.nit), message=str(result.message),
                      objective="sum of segment durations", search=search,
                      optimality="local numerical candidate under reference constraints; not global or AirSim time optimum",
                      initial_duration_s=float(sum(initial)), snap_integral_m2ps7=snap_integral(coefficients, durations),
                      peak_speed_mps=norm_peak(coefficients, durations, 1),
                      peak_acceleration_mps2=norm_peak(coefficients, durations, 2),
                      peak_jerk_mps3=norm_peak(coefficients, durations, 3)),
        libraries={name: importlib.metadata.version(name) for name in ("numpy", "scipy")})
    FrozenReference(plan_data)
    plan_data['planning'].update(wall_s=time.perf_counter()-begin,
        timing_scope="plan call including on-demand imports, solve, validation and metadata; excludes interpreter startup and campaign imports; cold/warm differences are not controlled")
    return plan_data


def common_fingerprint(config):
    """Compare tasks/controllers/gates independently of each planner's output."""
    import json
    common = {k: config[k] for k in ("protocol", "architecture", "visualization", "airsim", "start", "yaw",
                                    "controller", "execution", "limits", "comparison")}
    return hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
