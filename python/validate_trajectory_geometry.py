"""Offline interface checks using explicit geometry from AirSim configuration.

No simulator connection, sensor inference, or flight commands. These plans are
counterfactual segments from an assumed stationary state, never measured flights.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from math import atan, degrees, radians, tan
from pathlib import Path
import time

import numpy as np

from perception_trajectory_planner import (
    PlanningParams, SphereObstacle, evaluate_quintic, plan_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_lattice(position):
    # Fixed, target-independent finite candidates, identical in both modes.
    return [
        (np.asarray(position) + [dx, dy, dz]).tolist()
        for dx in (0.0, 0.10, 0.20, 0.22)
        for dy in (-0.20, -0.12, 0.0, 0.12, 0.20)
        for dz in (-0.04, 0.0, 0.04)
    ]


def record_inputs(inputs):
    """Record supplied invalid inputs without nonstandard JSON NaN tokens."""
    def pack(value):
        if isinstance(value, float) and not np.isfinite(value):
            return {"nonfinite_value": repr(value)}
        if isinstance(value, dict):
            return {key: pack(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(item) for item in value]
        return value
    return pack({**inputs, "params": asdict(inputs["params"]),
                 "obstacles": [asdict(item) for item in inputs["obstacles"]]})


def input_from_config(path: Path, settings_path: Path):
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    settings = json.loads(settings_path.read_text(encoding="utf-8-sig"))
    camera = settings["Vehicles"][config["vehicle"]]["Cameras"]["front_center"]
    if any(float(camera.get(axis, 0.0)) != 0.0 for axis in ("Pitch", "Roll", "Yaw")):
        raise ValueError("This interface check supports fixed camera yaw/pitch/roll zero only")
    capture = next(item for item in camera["CaptureSettings"] if item["ImageType"] == 0)
    hfov = float(capture["FOV_Degrees"])
    # Ideal rectilinear pinhole model; actual projection still needs RPC validation.
    vfov = degrees(2.0 * atan(capture["Height"] / capture["Width"] * tan(radians(hfov) / 2.0)))
    params = PlanningParams(
        durations=(3.0,), sample_dt_s=0.05,
        position_min=tuple(config["workspace_min_ned_m"]),
        position_max=tuple(config["workspace_max_ned_m"]),
        max_speed_mps=float(config["safety"]["approach_speed_cap_m_s"]),
        max_acceleration_mps2=0.30,
        horizontal_fov_deg=hfov, vertical_fov_deg=vfov,
        camera_offset_ned_m=tuple(float(camera.get(axis, 0.0)) for axis in ("X", "Y", "Z")),
        min_target_distance_m=2.0, desired_observation_distance_m=4.1,
        max_observation_age_s=float(config["visual"]["frame_max_age_s"]),
    )
    position = config["observation_world_ned_m"]
    return {
        "position": position, "velocity": [0.0, 0.0, 0.0],
        "acceleration": [0.0, 0.0, 0.0],
        "target_position": config["target_world_ned_m"],
        "target_velocity": [0.0, 0.0, 0.0],
        "candidate_endpoints": candidate_lattice(position),
        "params": params, "obstacles": (), "observation_age_s": 0.0,
    }


def dense_check(plan, multiplier=10):
    """Independent denser recomputation of each selected segment's constraints.

    This reduces a numerical sampling gap; it is not a continuous-time proof.
    """
    if plan["status"] != "OK":
        return {"performed": False, "reason": "no_selected_segment"}
    params = plan["params"]
    state = plan["input_state"]
    trajectory = plan["trajectory"]
    duration = trajectory["duration_s"]
    times = np.linspace(0.0, duration, int(np.ceil(duration / params["sample_dt_s"] * multiplier)) + 1)
    p = evaluate_quintic(trajectory["coefficients"], times)
    v = evaluate_quintic(trajectory["coefficients"], times, 1)
    a = evaluate_quintic(trajectory["coefficients"], times, 2)
    target = np.asarray(state["target_estimate"]) + (
        state["observation_age_s"] + times[:, None]
    ) * np.asarray(state["target_velocity_estimate"])
    relative = target - p - np.asarray(params["camera_offset_ned_m"])
    fov_tans = np.tan(np.radians([params["horizontal_fov_deg"], params["vertical_fov_deg"]]) / 2.0)
    projection = relative[:, 1:] / (np.maximum(relative[:, 0], params["camera_near_depth_m"])[:, None] * fov_tans)
    margins = {
        "POSITION_BOUNDS": float(np.min(np.minimum(p - params["position_min"], np.asarray(params["position_max"]) - p))),
        "SPEED_LIMIT": float(params["max_speed_mps"] - np.max(np.linalg.norm(v, axis=1))),
        "ACCELERATION_LIMIT": float(params["max_acceleration_mps2"] - np.max(np.linalg.norm(a, axis=1))),
        "TARGET_CAMERA_DEPTH": float(np.min(relative[:, 0] - params["camera_near_depth_m"])),
        "TARGET_FOV": float(1.0 - np.max(np.abs(projection))),
        "TARGET_MIN_DISTANCE": float(np.min(np.linalg.norm(target - p, axis=1)) - params["min_target_distance_m"]),
    }
    for index, obstacle in enumerate(state["obstacles"]):
        margins[f"SPHERE_OBSTACLE:{index}:{obstacle['label']}"] = float(
            np.min(np.linalg.norm(p - obstacle["center"], axis=1)) - obstacle["radius"]
            - params["drone_radius_m"] - params["obstacle_margin_m"]
        )
    return {
        "performed": True, "sample_count": len(times),
        "sample_dt_max_s": float(np.max(np.diff(times))),
        "passed": all(value >= -1e-8 for value in margins.values()),
        "margins": margins, "feasibility_scope": "denser_sampled_times_only",
    }


def matched_mode_check(first, second, expected_candidate_count):
    """Check the whole candidate populations, including count and ordering."""
    comparable = (
        first["params"] == second["params"]
        and first["input_state"] == second["input_state"]
        and len(first["candidates"]) == len(second["candidates"]) == expected_candidate_count
    )
    for a, b in zip(first["candidates"], second["candidates"]):
        comparable = comparable and all(a[key] == b[key] for key in
            ("index", "endpoint_index", "endpoint", "duration_s", "coefficients", "feasible",
             "reject_reasons", "sampled_margins", "cost_components"))
    changed_weights = {key for key in first["cost_weights"] if first["cost_weights"][key] != second["cost_weights"][key]}
    return {"same_inputs_candidates_constraints": comparable,
            "expected_candidate_count": expected_candidate_count,
            "observed_candidate_counts": [len(first["candidates"]), len(second["candidates"])],
            "changed_weights": sorted(changed_weights),
            "passed": comparable and changed_weights == {"image_center"}}


def validate(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    settings_path = ROOT / "configs/settings.json"
    paths = [ROOT / f"configs/phase1_lateral_pilot/{name}.json" for name in
             ("center", "target_y_minus_06", "target_y_plus_06")]
    source_paths = paths + [settings_path, Path(__file__), ROOT / "python/perception_trajectory_planner.py"]
    report = {
        "schema_version": 1, "validation_type": "GEOMETRY_ORACLE_INTERFACE_VALIDATION",
        "created_local": datetime.now().astimezone().isoformat(),
        "online_control_enabled": False, "simulator_connected": False,
        "source_hashes": {str(path): sha256(path) for path in source_paths},
        "input_provenance": "explicit_config_geometry_with_assumed_zero_velocity_and_acceleration",
        "limitations": [
            "Not sensor target estimation or measured flight data",
            "Fixed zero yaw/pitch/roll and ideal pinhole projection; actual camera pose not measured",
            "Only one three-second segment per case; not receding-horizon mission performance",
            "Sampled constraints, including denser checks, do not prove continuous avoidance",
            "No calibrated target identity, occlusion estimate, unknown obstacle handling, or verified brake",
            "Developer-selected 0.30 m/s^2 acceleration and body-center distances need flight validation",
            "Wall clock solve times are offline runtime observations, not end-to-end perception/control latency",
        ],
        "cases": [], "matched_mode_checks": [],
    }
    rows = []
    for path in paths:
        inputs = input_from_config(path, settings_path)
        for mode in ("efficiency", "perception"):
            started = time.perf_counter()
            plan = plan_trajectory(**inputs, mode=mode)
            elapsed = time.perf_counter() - started
            dense = dense_check(plan)
            passed = plan["status"] == "OK" and dense.get("passed") is True
            case = {"case_id": path.stem, "mode": mode, "expected_status": "OK",
                    "check_passed": passed, "offline_plan_seconds": elapsed,
                    "supplied_input": record_inputs(inputs), "dense_check": dense, "plan": plan}
            report["cases"].append(case)
            chosen = next((item for item in plan["candidates"] if item["index"] == plan["selected_index"]), None)
            rows.append({"case_id": path.stem, "mode": mode, "status": plan["status"],
                         "candidate_count": len(plan["candidates"]),
                         "feasible_count": sum(item["feasible"] for item in plan["candidates"]),
                         "selected_index": plan["selected_index"],
                         "endpoint": json.dumps(plan["trajectory"]["endpoint"]) if chosen else "",
                         "total_cost": plan["total_cost"], "offline_plan_seconds": elapsed,
                         "dense_pass": dense.get("passed"),
                         "image_center_component": chosen["cost_components"]["image_center"] if chosen else None})
        pair = report["cases"][-2:]
        first, second = (item["plan"] for item in pair)
        report["matched_mode_checks"].append({"case_id": path.stem, **matched_mode_check(
            first, second, len(inputs["candidate_endpoints"]) * len(inputs["params"].durations))})

    base = input_from_config(paths[0], settings_path)
    failures = [
        ("stale_observation", {"observation_age_s": 0.51}, "STALE"),
        ("initial_overspeed", {"velocity": [0.16, 0.0, 0.0]}, "INVALID"),
        ("outside_workspace", {"position": [-6.0, -0.5, -3.0]}, "INVALID"),
        ("endpoints_outside_workspace", {"candidate_endpoints": [[-2.0, -0.5, -3.0]]}, "NO_FEASIBLE"),
        ("target_behind_camera", {"target_position": [-7.0, -0.5, -3.0]}, "NO_FEASIBLE"),
        ("blocking_sphere_at_start", {"obstacles": [SphereObstacle(tuple(base["position"]), 0.10, "blocking_fixture")]}, "NO_FEASIBLE"),
        ("nonfinite_target", {"target_position": [float("nan"), 0.0, -3.0]}, "INVALID"),
    ]
    for case_id, changes, expected in failures:
        supplied = {**base, **changes}
        plan = plan_trajectory(**supplied, mode="perception")
        passed = plan["status"] == expected and plan["trajectory"] is None and plan["selected_index"] is None
        report["cases"].append({"case_id": case_id, "mode": "perception", "expected_status": expected,
                                "check_passed": passed, "supplied_input": record_inputs(supplied), "plan": plan})
    report["summary"] = {
        "case_count": len(report["cases"]),
        "case_checks_passed": sum(item["check_passed"] for item in report["cases"]),
        "matched_mode_checks_passed": sum(item["passed"] for item in report["matched_mode_checks"]),
        "all_checks_passed": all(item["check_passed"] for item in report["cases"])
                             and all(item["passed"] for item in report["matched_mode_checks"]),
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output_dir / "plans.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    render_plans(report, output_dir)
    lines = ["# 单机轨迹几何接口验证", "", "类型：`GEOMETRY_ORACLE_INTERFACE_VALIDATION`。没有连接AirSim，也没有执行飞行。", "",
             f"检查：{report['summary']['case_checks_passed']}/{report['summary']['case_count']}；同输入对照：{report['summary']['matched_mode_checks_passed']}/3。", "",
             "三种目标位置分别检查效率与感知代价模式；候选、约束、相机配置、状态完全一致，只改变图像居中代价权重。", "",
             "| 工况 | 模式 | 可行候选 | 端点 (NED米) | 离线求解秒 |", "|---|---|---:|---|---:|"]
    for row in rows:
        lines.append(f"| {row['case_id']} | {row['mode']} | {row['feasible_count']}/{row['candidate_count']} | {row['endpoint']} | {row['offline_plan_seconds']:.4f} |")
    lines += ["", "失败工况：陈旧观测、初始超速、越界、全部端点越界、目标在相机后方、起点球障碍与非有限输入。失败均不输出可执行轨迹。", "",
              "所有成功选择使用0.005秒网格复查约束；这仍是采样检查，不证明连续避碰。", "",
              "这些是已知几何下、假定飞机静止时的单段规划结果。不能计为感知准确性、任务完成率、轨迹改善或论文正式实验结果。后续需验证真实相机姿态与时间戳，再接入受保护的飞行闭环。", "",
              "AI使用说明：代码与报告由AI辅助编写；运行记录来自本机离线检查。"]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report["summary"]


def render_plans(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    condition_names = [item["case_id"] for item in report["matched_mode_checks"]]
    for ax, name in zip(axes, condition_names):
        cases = [item for item in report["cases"] if item["case_id"] == name]
        for case in cases:
            plan = case["plan"]
            if plan["status"] != "OK":
                continue
            for candidate in plan["candidates"]:
                points = evaluate_quintic(candidate["coefficients"], np.linspace(0.0, candidate["duration_s"], 45))
                if case["mode"] == "efficiency":
                    ax.plot(points[:, 0], points[:, 1], color="#cbd5e1" if candidate["feasible"] else "#f7d9d4", lw=0.5, alpha=0.6)
            trajectory = plan["trajectory"]
            points = evaluate_quintic(trajectory["coefficients"], np.linspace(0.0, trajectory["duration_s"], 100))
            color = {"efficiency": "#2166ac", "perception": "#d6604d"}[case["mode"]]
            ax.plot(points[:, 0], points[:, 1], color=color,
                    lw=2.0, label=case["mode"])
            ax.scatter(points[-1, 0], points[-1, 1], s=32, color=color)
        ax.set_title(name.replace("target_", ""))
        ax.set_xlabel("World NED x (m)")
        ax.set_ylabel("World NED y (m)")
        ax.set_xlim(-5.025, -4.75)
        ax.set_ylim(-0.73, -0.26)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Offline finite-candidate geometry check — selected 3 s segments, not flights", fontsize=11)
    fig.savefig(output_dir / "candidate_segments.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs" / f"trajectory_geometry_{datetime.now():%Y%m%d_%H%M%S_%f}")
    args = parser.parse_args()
    summary = validate(args.output_dir.resolve())
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), **summary}, ensure_ascii=False))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
