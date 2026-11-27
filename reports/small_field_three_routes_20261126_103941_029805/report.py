"""Assemble the frozen three-route comparison from the AirSim suite receipts."""
import csv
import json
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial import Polynomial

ROOT = Path(__file__).resolve().parents[2]
BATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))

from dual_target_control.small_field import polynomial_extrema
from fast_reference import decode

METHODS = {
    "hermite_ordinary": ("Hermite", "普通计时"),
    "bezier_ordinary": ("Bézier", "普通计时"),
    "bspline_ordinary": ("七次 B 样条", "普通计时"),
    "hermite_toppquad-full": ("Hermite", "完整 TOPPQuad"),
    "hermite_toppquad": ("Hermite", "自动调整 TOPPQuad"),
}
HERMITE_BASELINE = "hermite_ordinary"
ROUTES = {"s": "S 型", "u-turn": "U 型回转", "arch": "三维拱形"}


def read(path):
    return json.loads(path.read_bytes())


def write_csv(name, rows):
    with (BATCH / name).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def side_clearance(run, manifest, design):
    curve = decode(manifest["plan"])
    first_arrival = curve.waypoint_times[0]["arrival_time_s"]
    lower, upper = np.asarray(design["small_field"]["bounds_zup_m"])[:, :2]
    estimates = []
    for segment, end in zip(curve.segments, curve.ends):
        if end <= first_arrival + 1e-9:
            continue
        for axis, axis_bounds in enumerate(zip(lower, upper)):
            lo, hi = polynomial_extrema(Polynomial(segment.coefficients[:, axis]))
            estimates.extend((lo - axis_bounds[0], axis_bounds[1] - hi))
    states = read(run / "trace.json")["states"]
    states_after_first = [state["position_zup_m"][:2] for state in states
                          if state["reference_elapsed_s"] >= first_arrival]
    measured = None
    if states_after_first:
        p = np.asarray(states_after_first)
        measured = float(min(np.min(p - lower), np.min(upper - p)))
    return min(estimates, default=None), measured


def detail(route, trial, design):
    run = Path(trial["run"])
    summary = read(run / "summary.json")
    manifest = read(run / "live_manifest.json")
    method = trial["method"]
    assert method == summary["method"] == manifest["method"]
    assert manifest["plan"]["route"]["waypoints"] == design["routes"][route]["waypoints"]
    assert summary["config"]["small_field"]["start"] == design["routes"][route]["start"]
    assert summary["dynamics"]["verified"] and summary["airsim_search_calls"] == 0
    events = {event["stage"]: event for event in summary["preparation_events"]}
    planner_output = Path(events["PLANNING_STARTED"]["output"])
    validation_path = planner_output / "validation.json"
    planned = read(validation_path) if validation_path.is_file() else {}
    reference = planned.get("validation", {}).get("reference_checks") or trial.get("reference", {})
    nominal = planned.get("validation", {}).get("nominal_command_bounds", [None, None])
    if method == "hermite_toppquad-full" and planned:
        assert planned["retiming_fraction"] == 1.0
        assert [candidate["fraction"] for candidate in planned["retiming_candidates"]] == [1.0]
    if method == "hermite_toppquad" and planned:
        assert planned["retiming_profile"]["mode"] == "segmented"
    started = bool(summary["samples"])
    assert not started or "REFERENCE_STARTED" in events
    metrics = trial.get("metrics", {})
    condition = manifest["condition_fingerprint"]
    hit = sum(row["passed"] for row in metrics.get("waypoint_results", []))
    count = len(design["routes"][route]["waypoints"]) - 1
    failure = trial["error"] or ""
    if "PLANNING_REJECTED" in events and reference:
        failure = "参考预检拒绝"
    if not trial["passed"] and not failure:
        failed_checks = [key for key, value in trial.get("checks", {}).items() if not value]
        failure = ", ".join(failed_checks) or trial["outcome"]
    if started:
        assert trial["condition_fingerprint"] == condition
    reference_wall_gap, measured_wall_gap = side_clearance(run, manifest, design) if started else (None, None)
    planning_s = (summary.get("planning_pipeline_timings") or {}).get("parent_request_to_collect_s")
    if planning_s is None and "PLANNING_REJECTED" in events:
        planning_s = events["PLANNING_REJECTED"]["host_monotonic_s"] - events["PLANNING_STARTED"]["host_monotonic_s"]
    row = dict(
        route=route, method=method, outcome=trial["outcome"], passed=trial["passed"],
        reference_started=started, safe_landing=trial["safe_landing"], sample_count=summary["samples"],
        reference_time_s=manifest.get("effective_duration_s"),
        candidate_time_s=(planned["retiming_candidates"][0]["duration_s"]
                          if method == "hermite_toppquad-full" and planned else None),
        tracking_time_s=metrics.get("tracking_time_s"),
        planning_wall_s=planning_s,
        solver_wall_s=planned.get("timings", {}).get("raw_toppquad_wall_s"),
        reference_peak_speed_mps=reference.get("peak_speed_mps"),
        reference_peak_accel_mps2=reference.get("peak_acceleration_mps2"),
        nominal_command_slew_bound_mps2=nominal[1],
        measured_peak_speed_mps=metrics.get("measured_peak_speed_mps"),
        peak_tilt_deg=metrics.get("peak_tilt_deg"),
        rmse_position_m=metrics.get("rmse_position_m"),
        max_position_error_m=metrics.get("max_position_error_m"),
        max_sample_gap_s=metrics.get("max_sample_gap_s"),
        reference_side_clearance_after_first_waypoint_m=reference_wall_gap,
        measured_side_clearance_after_first_waypoint_m=measured_wall_gap,
        terminal_hold_s=metrics.get("terminal_hold", {}).get("continuous_s"),
        terminal_position_error_m=metrics.get("terminal_hold", {}).get("end_position_error_m"),
        terminal_speed_mps=metrics.get("terminal_hold", {}).get("end_speed_mps"),
        ordered_hits=hit if started else None, intermediate_waypoints=count,
        retiming_fraction=planned.get("retiming_fraction"),
        retiming_profile=json.dumps(planned["retiming_profile"]["fractions"])
                         if "retiming_profile" in planned else None,
        condition_fingerprint=condition, message=failure, raw_message=trial["error"],
        run=str(run), planner_report=str(validation_path) if planned else "",
    )
    waypoints = [dict(route=route, method=method, **entry) for entry in metrics.get("waypoint_results", [])]
    return row, waypoints, planned


def display(value, places=2):
    return "—" if value is None else f"{value:.{places}f}"


def render(rows):
    baseline_times = {row["route"]: row["tracking_time_s"] for row in rows
                      if row["method"] == HERMITE_BASELINE and row["passed"]}
    lines = [
        "# 8 × 8 × 2.5 m 小场地五方法对照",
        "",
        "每条航线的五个组合共用同一套有序航点，每个组合调度一次。"
        "无人机从地面原点起飞，收敛到 (0, 0, 0.50) m 后准备跟踪。"
        "三种几何构造基线分别是 Hermite、Bézier 和七次 B 样条，均采用各自的普通计时。"
        "TOPPQuad 接在普通 Hermite 基线生成的路径上，完整组使用 TOPPQuad 的 100% 原始计时，"
        "自动调整组在同一原始解上搜索分段计时。"
        "中间点连续通过，验收距离 ≤0.30 m 且同一时刻的速度 ≥0.50 m/s。"
        "终点沿用协议 17 的 1 s 连续停稳。参考速度和加速度上限分别为 1.60 m/s 和 1.50 m/s²。"
        "跟踪耗时由 AirSim 时间戳计算，从参考开始记至跟踪结束，含末端停稳时间。规划墙钟单列。",
        "",
        "Hermite 组的耗时差按（组内跟踪用时 ÷ 同航线普通 Hermite 跟踪用时 − 1）× 100% 计算。"
        "固定 12.5% 计时未纳入批次。预检拒绝的组合没有实测跟踪时间。",
        "",
        f"{len(rows)} 个组合记录，{sum(r['reference_started'] for r in rows)} 个进入跟踪，"
        f"{sum(r['passed'] for r in rows)} 个通过遥测复算与任务判据，"
        f"{sum(r['safe_landing'] for r in rows)} 个确认落地、上锁并归还控制。",
        "",
        "| 航线 | 基础轨迹 | 计时方式 | 结果 | 跟踪 s | Hermite 组耗时差 | 规划 s | 中间点 | 峰值位置误差 m | 峰值倾角 ° |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    presentation_order = ("hermite_ordinary", "hermite_toppquad-full", "hermite_toppquad",
                          "bezier_ordinary", "bspline_ordinary")
    ordered_rows = sorted(rows, key=lambda row: (list(ROUTES).index(row["route"]),
                                                presentation_order.index(row["method"])))
    for row in ordered_rows:
        result = "通过" if row["passed"] else row["message"]
        hits = f"{row['ordered_hits']}/{row['intermediate_waypoints']}" if row["reference_started"] else "—"
        geometry, timing = METHODS[row["method"]]
        relative = "—"
        if geometry == "Hermite" and row["passed"] and row["route"] in baseline_times:
            difference = 100 * (row["tracking_time_s"] / baseline_times[row["route"]] - 1)
            relative = "0%" if row["method"] == HERMITE_BASELINE else f"{difference:+.1f}%"
        lines.append(
            f"| {ROUTES[row['route']]} | {geometry} | {timing} | {result.replace('|', '/')} | "
            f"{display(row['tracking_time_s'])} | {relative} | {display(row['planning_wall_s'])} | {hits} | "
            f"{display(row['max_position_error_m'], 3)} | {display(row['peak_tilt_deg'], 1)} |")
    lines += [
        "",
        "## 完整 TOPPQuad 原始参考预检",
        "",
        "| 航线 | 原始参考时长 s | 速度峰值 m/s | 加速度峰值 m/s² | 名义指令变化率上界 m/s² |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["method"] == "hermite_toppquad-full":
            lines.append(
                f"| {ROUTES[row['route']]} | {display(row['candidate_time_s'])} | "
                f"{display(row['reference_peak_speed_mps'], 3)} | {display(row['reference_peak_accel_mps2'], 3)} | "
                f"{display(row['nominal_command_slew_bound_mps2'], 3)} |")
    lines += [
        "",
        "完整 TOPPQuad 的预检同时使用 5.16 m/s² 的名义指令变化率余量线。"
        "候选参考时间来自规划，实际跟踪时间来自 AirSim。"
        "Hermite 与 Bézier 是同一几何和时钟的两种表示。每个组合仅有一次记录。",
        "",
        "首批 S 型、U 型自动组的准备点横坐标出现约半纳米的负向舍入，导致启动保护拒绝。"
        "修订后实时与离线边界检查统一采用已有的 1e-9 m 数值容差，并重新运行全部组合。"
        "首批结果保留在 [首批调度回执](../small_field_three_routes_20261126_103104_777220/campaign.json)。",
        "",
        "[冻结坐标和条件](design.json) · [实测三维图](comparison.png) · [逐架指标](flights.csv) "
        "· [逐航点指标](waypoints.csv) · [批次审计](comparison_audit.json)",
    ]
    (BATCH / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    design = read(BATCH / "design.json")
    receipt = read(BATCH / "campaign.json")
    assert set(receipt) == set(design["routes"])
    rows, waypoints, checks = [], [], {}
    for route in design["routes"]:
        audit_path = Path(receipt[route]["report"]) / "audit.json"
        flights = read(audit_path)["flights"]
        assert [flight["method"] for flight in flights] == list(METHODS)
        assert all(flight["safe_landing"] for flight in flights)
        plans = {}
        for trial in flights:
            row, via, plans[trial["method"]] = detail(route, trial, design)
            rows.append(row)
            waypoints.extend(via)
        baseline = plans[HERMITE_BASELINE]
        full_plan = plans["hermite_toppquad-full"]
        auto_plan = plans["hermite_toppquad"]
        raw, auto = full_plan.get("solver"), auto_plan.get("solver")
        raw_available = bool(raw and auto)
        same_raw = raw_available and len(raw) == len(auto) and all(
            first["h"] == second["h"] and first["anchors"] == second["anchors"]
            for first, second in zip(raw, auto))
        same_base = full_plan.get("segments") == auto_plan.get("segments") if raw_available else None
        same_hermite_input = (same_base and baseline["segments"] == full_plan["segments"]
                              and baseline["timed_segments"] == full_plan["base_timed_segments"]
                              == auto_plan["base_timed_segments"])
        fingerprints = sorted({row["condition_fingerprint"] for row in rows
                               if row["route"] == route and row["reference_started"]})
        assert len(fingerprints) == 1
        checks[route] = dict(suite_audit=str(audit_path), same_raw_timing=same_raw, same_base_segments=same_base,
                             same_hermite_baseline_input=same_hermite_input,
                             tracking_condition_fingerprints=fingerprints, passed=sum(flight["passed"] for flight in flights))
    write_csv("flights.csv", rows)
    write_csv("waypoints.csv", waypoints)
    (BATCH / "comparison_audit.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    render(rows)
    print(f"{len(rows)} trials, {sum(row['passed'] for row in rows)} passed")


if __name__ == "__main__":
    main()
