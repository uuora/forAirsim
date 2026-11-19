"""Run the authors' discrete TOPPQuad solver and freeze a smooth tracking reference."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq

from .audit import frozen_plan_issues
from .reference import FrozenReference, RETIME_FORMAT
from .runtime import digest, load_config, write_json

UPSTREAM_URL = "https://github.com/maokat12/lbTOPPQuad.git"
UPSTREAM_COMMIT = "3db181cb0c76baaa1669f859f16a83060c2ee7a0"
NODE_COUNT = 61


def load_solver(checkout, *, checked_sha256=None):
    path = Path(checkout).resolve() / "TOPPQuad/TQ_trajectory.py"
    relative = "TOPPQuad/TQ_trajectory.py"
    source_sha256 = digest(path)
    if checked_sha256 is None:
        expected = subprocess.check_output(
            ["git", "-C", str(path.parents[1]), "show", f"{UPSTREAM_COMMIT}:{relative}"]
        )
        if path.read_bytes().replace(b"\r\n", b"\n") != expected:
            raise ValueError("TOPPQuad source differs from the pinned Git revision")
    elif source_sha256 != checked_sha256:
        raise ValueError("TOPPQuad source changed after startup verification")
    module_spec = importlib.util.spec_from_file_location("toppquad_upstream", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.TQ_trajectory, source_sha256


def solve_path(solver_type, reference, speed_cap, node_count=NODE_COUNT):
    # A uniform grid in the baseline's original time is a regular path
    # parameter away from the stationary endpoints. Use ENU in TOPPQuad.
    progress = np.linspace(0., 1., node_count)
    samples = [reference.sample(s * reference.duration) for s in progress]
    step = reference.duration / (node_count - 1)
    solver = solver_type(np.asarray(reference.points)[:, [1, 0, 2]] * [1, 1, -1],
                         n=node_count, vmax=speed_cap)
    path = {}
    for target, key in zip(
        ("x", "x_dot", "x_ddot", "x_dddot", "x_ddddot"),
        ("p", "v", "a", "jerk", "snap"),
    ):
        ned = np.array([row[key] for row in samples])
        path[target] = ned[:, [1, 0, 2]] * [1, 1, -1]
    solver.set_min_snap_traj(dict(path, d_s=np.full(node_count - 1, step),
                                  h=np.ones(node_count),
                                  T=np.full((node_count, 4), solver.Trest),
                                  vel=path["x_dot"], acc=path["x_ddot"],
                                  ms=None, t=progress * reference.duration))
    result, success, solve_wall = solver.get_optimized_traj()
    if not success:
        raise RuntimeError("Upstream TOPPQuad solve failed")
    times = np.asarray(result["t"])
    if times.shape != progress.shape or not np.isfinite(times).all() or np.min(np.diff(times)) <= 0:
        raise ValueError("TOPPQuad returned invalid times")
    return progress, times, solve_wall, result


def fit_progress(progress, times):
    """Fit t -> s with fixed endpoints and analytically check monotonicity."""
    x = times / times[-1]
    identity = Polynomial([0., 1.])
    boundary = Polynomial([0., 1., -1.])
    basis = np.polynomial.polynomial.polyvander(x, 3) * (x * (1 - x))[:, None]
    fit = identity + boundary * Polynomial(np.linalg.lstsq(basis, progress - x, rcond=None)[0])
    probes = [0., 1., *(r.real for r in fit.deriv(2).roots() if abs(r.imag) < 1e-9 and 0 < r.real < 1)]
    if min(fit.deriv()(np.array(probes))) <= 0:
        raise ValueError("Smooth TOPPQuad fit reverses path progress")
    return fit


def compose(plan, warp, duration):
    original = np.asarray(plan["segment_durations"], dtype=float)
    boundaries = np.r_[0., np.cumsum(original)] / original.sum()
    knots = [0., *(brentq(lambda t: warp(t) - s, 0., 1., xtol=1e-14)
                   for s in boundaries[1:-1]), 1.]
    coefficients = np.zeros((len(original), 36, 3))
    for i, raw in enumerate(plan["coefficients"]):
        local_time = Polynomial([knots[i], knots[i+1] - knots[i]])
        local_path = (warp(local_time) - boundaries[i]) / (boundaries[i+1] - boundaries[i])
        for axis, values in enumerate(np.asarray(raw).T):
            poly = Polynomial(values)(local_path)
            coefficients[i, :len(poly.coef), axis] = poly.coef
    return dict(plan, format=RETIME_FORMAT, coefficients=coefficients.tolist(),
                segment_durations=(np.diff(knots) * duration).tolist())


def retime(cfg, case, solver_type, source_hash, *, node_count=NODE_COUNT):
    begin = time.perf_counter()
    spec = cfg["cases"][case]
    baseline = spec["frozen_reference"]
    ref = FrozenReference(baseline)
    policy = cfg["comparison"]["time_search"]
    progress, times, solve_wall, solution = solve_path(solver_type, ref, policy["max_speed_mps"], node_count)
    warp = fit_progress(progress, times)
    # Apply one common time dilation to the smoothed profile. The original
    # controller's speed, acceleration and jerk gates apply to the export.
    candidate = compose(baseline, warp, times[-1])
    candidate_ref = FrozenReference(candidate)
    ratios = [1.]
    for order, (key, cap_key) in enumerate((
        ("v", "max_speed_mps"), ("a", "max_acceleration_mps2"), ("jerk", "max_jerk_mps3")
    ), 1):
        peak = max(np.linalg.norm(candidate_ref.sample(t)[key])
                   for t in np.linspace(0., candidate_ref.duration, 1001))
        ratios.append((peak / policy[cap_key])**(1 / order))
    dilation = max(ratios) * 1.002
    candidate["segment_durations"] = (np.asarray(candidate["segment_durations"]) * dilation).tolist()
    candidate["planning"] = dict(
        wall_s=time.perf_counter() - begin, upstream_solve_wall_s=solve_wall,
        upstream_commit=UPSTREAM_COMMIT, upstream_url=UPSTREAM_URL, upstream_source_sha256=source_hash,
        model="upstream Crazyflie; NED position converted to ENU; AirSim tracks with fixed baseline yaw",
        baseline_duration_s=ref.duration, raw_duration_s=float(times[-1]), time_dilation=dilation,
        progress_polynomial=warp.coef.tolist(), normalized_progress_nodes=progress.tolist(),
        raw_times_s=times.tolist(), raw_squared_baseline_parameter_rate=np.asarray(solution["h"]).ravel().tolist(),
        raw_rotor_thrust_n=np.asarray(solution["T"]).tolist(),
        progress_fit_max_error=float(np.max(np.abs(warp(times / times[-1]) - progress))),
    )
    spec.update(label=spec["label"] + " · TOPPQuad", frozen_reference=candidate,
                segment_durations=candidate["segment_durations"])
    issues = frozen_plan_issues(cfg, case)
    if issues:
        raise ValueError(f"{case}: " + ", ".join(issues))
    return candidate


def prepare_toppquad(source, output, checkout):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    load_config(source)
    cfg = deepcopy(json.loads(source.read_text(encoding="utf-8-sig")))
    if cfg["protocol"] != "geometric_planner_pass_v1":
        raise ValueError("Expected a frozen Bézier/Hermite/B-spline baseline config")
    cfg["protocol"] = "geometric_toppquad_pass_v1"
    cfg["comparison"].update(
        baseline_config_sha256=digest(source),
        scope="TOPPQuad-derived timing on fixed baseline geometry; smooth export with shared limits",
    )
    solver_type, source_hash = load_solver(checkout)
    for case in cfg["cases"]:
        print(f"Retiming {case}", flush=True)
        retime(cfg, case, solver_type, source_hash)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        validation = Path(temp) / "config.json"
        write_json(validation, cfg)
        load_config(validation)
    write_json(output, cfg)
    return output
