"""Paired random-waypoint comparison using RotorPy Crazyflie dynamics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from copy import deepcopy
import csv
import gzip
import html
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from scipy.spatial.transform import Rotation

from geometric_validation.audit import frozen_plan_issues, measured_passages
from geometric_validation.campaign import PLANNER_CONFIG, ROOT
from geometric_validation.planners import METHODS, plan
from geometric_validation.reference import FrozenReference, RETIME_FORMAT
from geometric_validation.runtime import digest, write_json
from geometric_validation.toppquad import UPSTREAM_COMMIT, load_solver, retime


ROTORPY_COMMIT = "f69a2a4e15788c834dfe58fe95dd55a017fd7bd3"
NODE_COUNT = 100
RATE = 50
SETTLE_ALLOWANCE_S = 3
AXES = np.array([1., 1., -1.])
FLOAT_COLUMNS = (
    "time_s position_e_m position_n_m position_u_m reference_e_m reference_n_m reference_u_m "
    "velocity_e_mps velocity_n_mps velocity_u_mps ref_ve_mps ref_vn_mps ref_vu_mps "
    "ref_ae_mps2 ref_an_mps2 ref_au_mps2 qx qy qz qw "
    "reference_qx reference_qy reference_qz reference_qw command_qx command_qy command_qz command_qw "
    "wx_radps wy_radps wz_radps cmd_f1_n cmd_f2_n cmd_f3_n cmd_f4_n "
    "actual_f1_n actual_f2_n actual_f3_n actual_f4_n position_error_m attitude_error_deg actual_tilt_deg"
).split()
WORKER = {}


def random_tasks(count, seed, template):
    rng = np.random.default_rng(seed)
    tasks = []
    for number in range(1, count + 1):
        task = deepcopy(template)
        point = np.array([0., 0., -5.])
        points = [point.tolist()]
        heading = rng.uniform(-np.pi, np.pi)
        band = ((5, 25), (25, 55), (55, 85))[(number - 1) % 3]
        for segment in range(3):
            if segment:
                heading += np.deg2rad(rng.uniform(*band)) * rng.choice([-1, 1])
            distance = rng.uniform(2., 4.)
            point = point + distance * np.array([np.cos(heading), np.sin(heading), rng.uniform(-.18, .18)])
            points.append(point.tolist())
        task["points_ned"] = points
        tasks.append(dict(id=f"route_{number:03}", heading_change_deg=list(band), task=task))
    return tasks


def enu(values):
    return np.asarray(values)[..., [1, 0, 2]] * AXES


def sample_reference(reference, times):
    """Evaluate the same frozen polynomials in batches, including terminal hold."""
    result = np.zeros((len(times), 5, 3))
    for index, derivatives in enumerate(reference.polynomials):
        start, stop = reference.times[index:index + 2]
        rows = (times >= start) & (times < stop if index + 1 < len(reference.polynomials) else times <= stop)
        duration = reference.segment_durations[index]
        u = (times[rows] - start) / duration
        for order, axes in enumerate(derivatives):
            result[rows, order] = np.column_stack([poly(u) for poly in axes])
    result[times > reference.duration, 0] = reference.end
    return enu(result)


def init_worker(rotorpy, upstream, template):
    sys.path.insert(0, str(rotorpy))
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.vehicles.multirotor import Multirotor

    solver, source_hash = load_solver(upstream)
    WORKER.update(template=template, vehicle=Multirotor, control=SE3Control,
                  params=quad_params, solver=solver, source_hash=source_hash)


def save_reference(folder, frozen):
    write_json(folder / "reference.json", frozen)
    reference = FrozenReference(frozen)
    times = np.linspace(0., reference.duration, NODE_COUNT)
    data = sample_reference(reference, times)
    np.savetxt(folder / "reference_100.csv", np.column_stack((times, data.reshape(len(times), -1))),
               header="time_s," + ",".join(f"{key}_{axis}" for key in ("p_m", "v_mps", "a_mps2", "jerk_mps3", "snap_mps4")
                                            for axis in "enu"), delimiter=",", comments="", fmt="%.10g")
    return reference


def simulate(reference, cfg, case, folder, *, sampler=sample_reference, heading=None):
    params = WORKER["params"]
    vehicle = WORKER["vehicle"](params)
    controller = WORKER["control"](params)
    times = np.arange(math.ceil((reference.duration + SETTLE_ALLOWANCE_S) * RATE) + 1) / RATE
    if heading is None:
        yaw = np.full(len(times), np.pi / 2 - cfg["yaw"])
        yaw_rate = np.zeros_like(yaw)
    else:
        yaw, yaw_rate = heading(times)
    q0 = Rotation.from_euler("z", yaw[0]).as_quat()
    hover = math.sqrt(params["mass"] * controller.g / (params["k_eta"] * 4))
    state = dict(x=enu(reference.points[0]), v=np.zeros(3), q=q0, w=np.zeros(3),
                 wind=np.zeros(3), rotor_speeds=np.full(4, hover))
    limit = cfg["limits"]
    flats = sampler(reference, times)
    thrust = flats[:, 2] + [0., 0., controller.g]
    z_axes = thrust / np.linalg.norm(thrust, axis=1, keepdims=True)
    y_axes = np.cross(z_axes, np.column_stack((np.cos(yaw), np.sin(yaw), np.zeros_like(yaw))))
    y_axes /= np.linalg.norm(y_axes, axis=1, keepdims=True)
    nominal = Rotation.from_matrix(np.stack((np.cross(y_axes, z_axes), y_axes, z_axes), axis=-1))
    nominal_q = nominal.as_quat()
    samples = []
    stable_since = None
    status = "terminal_timeout"
    for index, (t, flat) in enumerate(zip(times, flats)):
        x, v, a, j, s = flat
        flat_output = dict(x=x, x_dot=v, x_ddot=a, x_dddot=j, x_ddddot=s, yaw=yaw[index], yaw_dot=yaw_rate[index])
        command = controller.update(t, state, flat_output)
        actual = Rotation.from_quat(state["q"])
        error = np.linalg.norm(state["x"] - x)
        att = (nominal[index].inv() * actual).magnitude()
        tilt = np.arccos(np.clip(actual.as_matrix()[2, 2], -1, 1))
        speed = np.linalg.norm(state["v"])
        samples.append(np.hstack((t, state["x"], x, state["v"], v, a, state["q"], nominal_q[index],
                                   command["cmd_q"], state["w"], command["cmd_motor_thrusts"],
                                   params["k_eta"] * state["rotor_speeds"]**2, error, np.degrees([att, tilt]))))
        if not np.isfinite(samples[-1]).all() or error > 20 or state["x"][2] < 0:
            status = "nonfinite_or_flyaway_or_ground"
            break
        if t >= reference.duration:
            good = (error <= limit["terminal_position_m"] and att <= limit["terminal_attitude_rad"]
                    and speed <= limit["terminal_speed_mps"])
            if not good:
                stable_since = None
            elif stable_since is None:
                stable_since = t
            elif t - stable_since >= cfg["execution"]["terminal_hold_s"] - 1e-9:
                status = "completed"
                break
        if index + 1 < len(times):
            state = vehicle.step(state, command, 1 / RATE)
    data = np.asarray(samples)
    with gzip.open(folder / "simulation.csv.gz", "wt", encoding="utf-8", newline="") as stream:
        np.savetxt(stream, data, header=",".join(FLOAT_COLUMNS), delimiter=",", comments="", fmt="%.10g")
    rows = [dict(phase="TRACKING", t=float(row[0]), position=enu(row[1:4]), velocity=enu(row[7:10]))
            for row in data if np.isfinite(row).all()]
    passages_ok, passages = measured_passages(rows, cfg, case)
    finite = data[np.isfinite(data).all(axis=1)]
    if not len(finite):
        return dict(sim_status=status, samples=len(data), gate_passed=False, waypoint_passed=False, passages=passages)
    peak_error = float(np.max(finite[:, -3]))
    peak_attitude = float(np.max(finite[:, -2]))
    peak_tilt = float(np.max(finite[:, -1]))
    thrust_max = params["k_eta"] * params["rotor_speed_max"]**2
    thrust_start = FLOAT_COLUMNS.index("cmd_f1_n")
    raw_thrust = finite[:, thrust_start:thrust_start + 4]
    violation = np.maximum(-raw_thrust, 0) + np.maximum(raw_thrust - thrust_max, 0)
    return dict(sim_status=status, samples=len(data), complete_s=float(data[-1, 0]) if status == "completed" else None,
                last_sample_s=float(data[-1, 0]), position_peak_m=peak_error,
                position_rmse_m=float(np.sqrt(np.mean(finite[:, -3]**2))), attitude_peak_deg=peak_attitude,
                tilt_peak_deg=peak_tilt, thrust_violation_avg_n=float(np.mean(violation)),
                thrust_violation_peak_n=float(np.max(violation)), passages=passages,
                waypoint_passed=passages_ok,
                gate_passed=bool(status == "completed" and passages_ok and peak_error <= limit["position_error_m"]
                                 and peak_attitude <= math.degrees(limit["attitude_error_rad"])
                                 and peak_tilt <= math.degrees(limit["safety_tilt_rad"])))


def evaluate_pair(output, entry, method):
    case = method + "_pass"
    cfg = deepcopy(WORKER["template"])
    cfg["start"] = entry["task"]["points_ned"][0]
    cfg["comparison"]["task"] = entry["task"]
    rows = []
    baseline_ok = False
    for arm in ("baseline", "toppquad"):
        relative = Path(entry["id"]) / method / arm
        folder = output / relative
        folder.mkdir(parents=True)
        result = dict(route=entry["id"], method=method, arm=arm, directory=relative.as_posix(), accepted=False)
        try:
            if arm == "baseline":
                frozen = plan(method, entry["task"], cfg["comparison"]["time_search"])
                cfg["cases"] = {case: dict(method=method, label=method, segment_durations=frozen["segment_durations"],
                                            frozen_reference=frozen)}
                baseline_ok = True
            elif baseline_ok:
                with (folder / "solver.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
                    try:
                        frozen = retime(cfg, case, WORKER["solver"], WORKER["source_hash"], node_count=NODE_COUNT)
                    except ValueError:
                        # A formed export can still be tracked diagnostically when its reference audit fails.
                        frozen = cfg["cases"][case]["frozen_reference"]
                        if frozen["format"] != RETIME_FORMAT:
                            raise
            else:
                raise RuntimeError("Baseline reference unavailable")
            issues = frozen_plan_issues(cfg, case)
            reference = save_reference(folder, frozen)
            result.update(plan_issues=issues, reference_s=reference.duration, planning_s=frozen["planning"]["wall_s"],
                          raw_toppquad_s=frozen["planning"].get("raw_duration_s"))
            result.update(simulate(reference, cfg, case, folder))
            result["accepted"] = not issues and result["gate_passed"]
        except Exception as exc:
            result.update(error=f"{type(exc).__name__}: {exc}", sim_status="error")
        write_json(folder / "result.json", result)
        rows.append(result)
    return rows


def write_tables(output, rows):
    rows.sort(key=lambda r: (r["route"], METHODS.index(r["method"]), r["arm"]))
    scalar = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, scalar, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    pairs = []
    for base, retimed in zip(rows[::2], rows[1::2]):
        assert (base["route"], base["method"], base["arm"], retimed["arm"]) == (
            retimed["route"], retimed["method"], "baseline", "toppquad")
        pair = dict(route=base["route"], method=base["method"])
        for row in (base, retimed):
            for key in ("accepted", "reference_s", "complete_s", "position_rmse_m", "position_peak_m", "attitude_peak_deg"):
                pair[f'{row["arm"]}_{key}'] = row.get(key)
        for key in ("reference_s", "complete_s", "position_rmse_m", "position_peak_m", "attitude_peak_deg"):
            x, y = base.get(key), retimed.get(key)
            pair[f"delta_{key}"] = y - x if x is not None and y is not None else None
        pairs.append(pair)
    with (output / "paired.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    summary = {}
    for method in METHODS:
        items = [pair for pair in pairs if pair["method"] == method]
        both = [pair for pair in items if pair["baseline_accepted"] and pair["toppquad_accepted"]]
        summary[method] = dict(pairs=len(items), baseline_accepted=sum(x["baseline_accepted"] for x in items),
                               toppquad_accepted=sum(x["toppquad_accepted"] for x in items), both_accepted=len(both))
        for metric in ("reference_s", "complete_s", "position_rmse_m", "position_peak_m", "attitude_peak_deg"):
            values = [x[f"delta_{metric}"] for x in both if x[f"delta_{metric}"] is not None]
            summary[method][f"median_delta_{metric}_both_accepted"] = float(np.median(values)) if values else None
    write_json(output / "summary.json", summary)

    def cells(row, keys):
        return "".join(f"<td>{html.escape(f'{row[k]:.4f}' if isinstance(row.get(k), float) else str(row.get(k, '—')))}</td>"
                       for k in keys)

    keys = ("route", "method", "arm", "accepted", "sim_status", "reference_s", "complete_s",
            "position_rmse_m", "attitude_peak_deg")
    table = []
    for row in rows:
        directory = row["directory"]
        links = " · ".join(f'<a href="{directory}/{name}">{label}</a>'
                           for name, label in (("result.json", "判据"), ("reference_100.csv", "参考100点"),
                                               ("reference.json", "多项式"), ("simulation.csv.gz", "仿真样本"))
                           if (output / directory / name).is_file())
        table.append(f"<tr>{cells(row, keys)}<td>{links}</td></tr>")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RotorPy 三基线配对实验</title>'
        '<style>body{font:15px system-ui;margin:32px;color:#182538}table{border-collapse:collapse}'
        'td,th{border-bottom:1px solid #d9e1eb;padding:7px;text-align:left}tr:nth-child(even){background:#f4f7fb}</style>'
        '<h1>RotorPy 三基线配对实验</h1><p>Crazyflie 2.0 · 同一航点组、同一控制器、同一验收规则。</p>'
        '<p><a href="runs.csv">逐次指标</a> · <a href="paired.csv">配对差值</a> · '
        '<a href="summary.json">分组汇总</a> · <a href="design.json">任务与协议</a></p>'
        '<p>配对差值为 TOPPQuad − 原始基线。汇总的差值中位数仅基于同一对均验收通过的任务。</p>'
        '<table><thead><tr>' + ''.join(f"<th>{key}</th>" for key in keys) + '<th>数据</th></tr></thead><tbody>'
        + "".join(table) + '</tbody></table>', encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotorpy", required=True, type=Path, help="Root of the pinned RotorPy source checkout")
    parser.add_argument("--upstream", default=ROOT / ".runtime/toppquad_upstream", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--count", default=100, type=int)
    parser.add_argument("--seed", default=1201, type=int)
    parser.add_argument("--jobs", default=6, type=int)
    args = parser.parse_args()
    if args.count < 1 or args.jobs < 1:
        parser.error("--count and --jobs must be positive")
    rotorpy, upstream, output = (path.resolve() for path in (args.rotorpy, args.upstream, args.output))
    commit = subprocess.check_output(["git", "-C", str(rotorpy), "rev-parse", "HEAD"], text=True).strip()
    if commit != ROTORPY_COMMIT:
        parser.error(f"RotorPy must be checked out at {ROTORPY_COMMIT}")
    rotor_files = ("vehicles/multirotor.py", "vehicles/crazyflie_params.py", "controllers/quadrotor_control.py")
    hashes = {}
    for name in rotor_files:
        relative = "rotorpy/" + name
        pinned = subprocess.check_output(["git", "-C", str(rotorpy), "show", f"HEAD:{relative}"])
        if (rotorpy / relative).read_bytes().replace(b"\r\n", b"\n") != pinned:
            parser.error(f"Modified RotorPy source: {relative}")
        hashes[name] = digest(rotorpy / relative)
    template = json.loads(PLANNER_CONFIG.read_text(encoding="utf-8-sig"))
    # Both arms use the current stricter TOPPQuad passage gate in this paired experiment.
    template["comparison"]["passage"]["position_tolerance_m"] = .3
    tasks = random_tasks(args.count, args.seed, template["comparison"]["task"])
    output.mkdir(parents=True)
    begin = perf_counter()
    write_json(output / "design.json", dict(seed=args.seed, tasks=tasks, methods=METHODS, arms=["baseline", "toppquad"],
               rotorpy_commit=ROTORPY_COMMIT, rotorpy_source_sha256=hashes, toppquad_commit=UPSTREAM_COMMIT,
               toppquad_source_sha256=digest(upstream / "TOPPQuad/TQ_trajectory.py"),
               campaign_source_sha256=digest(Path(__file__)), baseline_config_sha256=digest(PLANNER_CONFIG),
               libraries={name: version(name) for name in ("numpy", "scipy", "casadi", "torch", "roma")},
               flight_model="RotorPy Multirotor.step + SE3Control, Crazyflie 2.0, no wind, ideal state, no camera",
               frame="references JSON NED / CSV and simulation ENU, quaternion xyzw",
               yaw_ned_rad=template["yaw"], hz=RATE, toppquad_nodes=NODE_COUNT, reference_export_points=NODE_COUNT,
               limits=template["limits"], time_search=template["comparison"]["time_search"],
               passage=template["comparison"]["passage"], terminal_hold_s=template["execution"]["terminal_hold_s"],
               terminal_extra_s=SETTLE_ALLOWANCE_S, workers=args.jobs))
    rows = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker,
                             initargs=(rotorpy, upstream, template)) as pool:
        futures = [pool.submit(evaluate_pair, output, entry, method) for entry in tasks for method in METHODS]
        for future in as_completed(futures):
            rows.extend(future.result())
            if len(rows) % 30 == 0 or len(rows) == len(futures) * 2:
                print(f"{len(rows)}/{len(futures) * 2} arms recorded", flush=True)
    print(json.dumps(write_tables(output, rows), indent=2), flush=True)
    print(f"Wall time {perf_counter() - begin:.1f} s", flush=True)


if __name__ == "__main__":
    main()
