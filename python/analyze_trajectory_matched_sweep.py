"""Audit a matched-start lateral development sweep, retaining incomplete trials.

This reads frozen plans and raw reports. It never connects to AirSim or tests
method superiority. One separately launched simulator policy pair is the
descriptive experimental unit; telemetry points are not independent repeats.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_trajectory_pilot import finite_scalar, finite_vector, summarize_matched_pairs, summarize_run


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_path(value, anchor):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (Path(anchor).parent / path).resolve()


def descriptive(values):
    finite = [number for value in values if (number := finite_scalar(value)) is not None]
    return {"assessed_count": len(finite), "mean": float(np.mean(finite)) if finite else None,
            "p50": float(np.median(finite)) if finite else None,
            "p95": float(np.percentile(finite, 95)) if finite else None}


def observation_error_values(report):
    values = []
    for observation in report.get("observations", []):
        error = observation.get("error_xy")
        if observation.get("detected") is not True or not isinstance(error, (list, tuple)) or len(error) != 2:
            continue
        xy = [finite_scalar(component) for component in error]
        if all(component is not None for component in xy):
            values.append(max(abs(component) for component in xy))
    return values


def aggregate(rows, condition_ids):
    groups = []
    for condition in condition_ids:
        for policy in ("efficiency", "perception"):
            planned = [row for row in rows if row["condition_id"] == condition and row["policy"] == policy]
            started = [row for row in planned if row["started"] is True]
            unknown = sum(row["started"] is None for row in planned)
            failure_reasons = Counter(row.get("protective_return_reason") or row.get("initial_state_match", {}).get("protective_reason") or row.get("approach_stop_reason") or row.get("error") or row["ledger_status"]
                                      for row in started if row.get("task_success") is not True)
            groups.append({"condition_id": condition, "policy": policy,
                "planned_flights": len(planned), "known_started_flights": len(started),
                "verified_not_started_flights": sum(row["started"] is False for row in planned),
                "unknown_start_status_flights": unknown, "started_denominator_complete": unknown == 0,
                "task_success_count": sum(row.get("task_success") is True for row in started),
                "task_failure_count": sum(row.get("task_success") is not True for row in started),
                "task_failure_reason_counts": dict(sorted(failure_reasons.items())),
                "task_rate_known_started": sum(row.get("task_success") is True for row in started) / len(started) if started else None,
                "safe_landing_and_release_count": sum(row.get("safe_landing_and_release") is True for row in started),
                "missing_terminal_report_count": sum(row["ledger_status"] == "STARTED_NO_TERMINAL_REPORT" for row in started),
                "verified_initial_acceptance_count": sum(row.get("initial_state_match", {}).get("acceptance_verified") is True for row in started),
                "initial_acceptance_failure_count": sum(row.get("initial_state_match", {}).get("acceptance_verified") is False for row in started),
                "positioning_physical_budget_exceedance_run_count": sum(row.get("initial_state_match", {}).get("physical_sampled_budget_status") == "OBSERVED_EXCEEDANCE" for row in started),
                "positioning_command_norm_violation_run_count": sum((row.get("initial_state_match", {}).get("command_xy_norm_violations") or 0) > 0 for row in started),
                "approach_physical_budget_exceedance_run_count": sum(row.get("observed_numeric_physical_budget_exceedance_present") is True for row in started),
                "approach_unquantified_budget_rejection_run_count": sum(row.get("unquantified_motion_budget_rejection_present") is True for row in started),
                "approach_command_norm_violation_run_count": sum((row.get("command_norm_violations") or 0) > 0 for row in started),
                "approach_inner_protection_margin_exceedance_run_count": sum(row.get('approach_spatial_budget',{}).get('status')=='OBSERVED_EXCEEDANCE' for row in started),
                "approach_duration_s": descriptive(row.get("approach_duration_s") for row in started),
                "sampled_approach_path_m": descriptive(row.get("sampled_approach_path_m") for row in started),
                "per_run_mean_solver_s": descriptive(row.get("solver_elapsed_s", {}).get("mean") for row in started),
                "per_run_p50_solver_s": descriptive(row.get("solver_elapsed_s", {}).get("p50") for row in started),
                "per_run_p95_solver_s": descriptive(row.get("solver_elapsed_s", {}).get("p95") for row in started),
                "per_run_mean_observed_max_axis_error": descriptive(row.get("observed_max_axis_error", {}).get("mean") for row in started),
                "per_run_terminal_observed_max_axis_error": descriptive(row.get("terminal_mean_max_axis_error") for row in started),
                "aggregation_scope": "per-run descriptive quantities across known started runs; positioning excluded from approach; solver/observation queries are not independent repeats"})
    return groups


def audit_sweep(sweep_path):
    sweep_path = Path(sweep_path).resolve()
    sweep_bytes = sweep_path.read_bytes()
    sweep = json.loads(sweep_bytes.decode("utf-8-sig"))
    provenance = [{"kind": "sweep", "path": str(sweep_path), "sha256": hashlib.sha256(sweep_bytes).hexdigest(),
                   "registered_hash_verified": None}]
    notes = []

    def checked_json(path, expected, kind, require_registered=False):
        path = Path(path).resolve()
        contents = path.read_bytes()
        actual = hashlib.sha256(contents).hexdigest()
        if expected and actual != expected:
            raise ValueError(f"{kind} hash mismatch: {path}")
        if require_registered and not expected:
            raise ValueError(f"{kind} registered hash missing: {path}")
        if not expected:
            notes.append(f"UNREGISTERED_HASH:{kind}:{path}")
        provenance.append({"kind": kind, "path": str(path), "sha256": actual,
                           "registered_sha256": expected, "registered_hash_verified": bool(expected)})
        return json.loads(contents.decode("utf-8-sig"))

    conditions_path = resolve_path(sweep["conditions_path"], sweep_path)
    conditions = checked_json(conditions_path, sweep.get("conditions_sha256"), "conditions", True)
    configs = {}
    for row in conditions["conditions"]:
        condition = row["condition_id"]
        if condition in configs:
            raise ValueError("Duplicate condition freeze")
        path = resolve_path(row["configuration_path"], conditions_path)
        checked_json(path, row.get("configuration_sha256"), "condition_configuration", True)
        configs[condition] = row
    plans = conditions["session_plan"]
    if sweep.get("planned_sessions") != len(plans) or sweep.get("planned_flights") != sum(len(row["method_order"]) for row in plans):
        raise ValueError("Sweep planned denominator differs from frozen conditions")
    plan_keys = [(row["condition_id"], row["repetition"]) for row in plans]
    if len(set(plan_keys)) != len(plan_keys):
        raise ValueError("Duplicate planned condition/repetition")
    actual_sessions = {}
    for entry in sweep.get("sessions", []):
        key = (entry["condition_id"], entry["repetition"])
        if key not in plan_keys or key in actual_sessions:
            raise ValueError("Unplanned or duplicate sweep session")
        actual_sessions[key] = entry
    rows, session_rows, plot_points = [], [], []
    source_versions = None
    engine_identities = set()
    for plan_index, plan in enumerate(plans, 1):
        condition, repetition = plan["condition_id"], plan["repetition"]
        entry = actual_sessions.get((condition, repetition))
        session_row = {"planned_session_index": plan_index, "condition_id": condition, "repetition": repetition,
                       "planned_method_order": plan["method_order"], "status": entry.get("status") if entry else "NOT_STARTED",
                       "independent_launch_evidence_verified": False, "evidence_notes": []}
        session_rows.append(session_row)
        planned_rows = [{"planned_session_index": plan_index, "session": plan_index, "condition_id": condition,
                         "repetition": repetition, "planned_method_index": index, "policy": policy,
                         "started": False if entry is None else None, "ledger_status": "NOT_STARTED" if entry is None else "UNKNOWN_STARTED_SESSION",
                         "task_success": False, "safe_landing_and_release": False}
                        for index, policy in enumerate(plan["method_order"], 1)]
        rows.extend(planned_rows)
        if entry is None:
            continue
        if entry.get("planned_session_index", plan_index) != plan_index or entry.get("policy_order") != plan["method_order"]:
            raise ValueError("Sweep method order/session index differs from frozen session plan")
        if entry.get("configuration") and sha256(resolve_path(entry["configuration"], sweep_path)) != configs[condition]["configuration_sha256"]:
            raise ValueError("Session configuration differs from condition freeze")
        if not entry.get("manifest") or not resolve_path(entry["manifest"], sweep_path).exists():
            session_row["evidence_notes"].append("STARTED_SESSION_WITHOUT_MANIFEST_FLIGHT_STARTS_UNKNOWN")
            continue
        manifest_path = resolve_path(entry["manifest"], sweep_path)
        manifest = checked_json(manifest_path, entry.get("manifest_sha256"), "manifest")
        session_row["manifest"] = str(manifest_path)
        if manifest.get("experiment_config_sha256") != configs[condition]["configuration_sha256"]:
            raise ValueError("Manifest configuration differs from condition freeze")
        if manifest.get("target_source") != plan.get("target_source", "sensor_surface_estimate"):
            raise ValueError("Mixed target sources in matched sweep")
        if manifest.get("planned_sessions") != 1 or manifest.get("planned_flights_per_session") != len(plan["method_order"]):
            raise ValueError("Each sweep entry must be one planned paired Unreal session")
        inner_sessions = manifest.get("sessions", [])
        if len(inner_sessions) > 1:
            raise ValueError("More than one engine session inside an independent sweep entry")
        if not inner_sessions:
            session_row["evidence_notes"].append("NO_ENGINE_SESSION_RECORD_STARTS_UNKNOWN")
            continue
        inner = inner_sessions[0]
        identity = (inner.get("engine_process_id"), inner.get("engine_started"))
        session_row.update(engine_process_id=identity[0], engine_started=identity[1],
                           engine_closed_after_safe_completion=inner.get("engine_closed_after_safe_completion") is True)
        identity_valid = isinstance(identity[0], int) and not isinstance(identity[0], bool) and identity[0] > 0 and isinstance(identity[1], str) and bool(identity[1])
        if identity_valid:
            if identity in engine_identities:
                raise ValueError("Repeated Unreal PID/StartTime identity across planned independent sessions")
            engine_identities.add(identity)
        else:
            session_row["evidence_notes"].append("MISSING_ENGINE_PID_OR_START_TIME")
        batch_path = resolve_path(inner["batch_path"], manifest_path) / "batch_report.json" if inner.get("batch_path") else None
        if batch_path is None or not batch_path.exists():
            session_row["evidence_notes"].append("MISSING_BATCH_LEDGER_FLIGHT_STARTS_UNKNOWN")
            continue
        batch = checked_json(batch_path, inner.get("batch_report_sha256"), "batch")
        if batch.get("configuration_sha256") != configs[condition]["configuration_sha256"] or batch.get("target_source") != manifest["target_source"]:
            raise ValueError("Batch configuration/target source differs from frozen manifest")
        if batch.get("planned_policies") != plan["method_order"]:
            raise ValueError("Batch method order differs from frozen session plan")
        run_entries = batch.get("runs", [])
        if [row.get("policy") for row in run_entries] != plan["method_order"][:len(run_entries)] or len(run_entries) > len(planned_rows):
            raise ValueError("Started method order differs from frozen session plan")
        analyzed = None
        if entry.get("analysis") and resolve_path(entry["analysis"], sweep_path).exists():
            analyzed = checked_json(resolve_path(entry["analysis"], sweep_path), entry.get("analysis_sha256"), "analysis")
            if analyzed.get("manifest_sha256") != sha256(manifest_path):
                raise ValueError("Analysis does not bind to current manifest hash")
            if analyzed.get("started_flights") != len(run_entries):
                raise ValueError("Analysis started denominator differs from raw batch ledger")
        else:
            session_row["evidence_notes"].append("NO_REGISTERED_SESSION_ANALYSIS_RAW_REPORTS_RECOMPUTED")
        for index, row in enumerate(planned_rows):
            row.update(started=index < len(run_entries), ledger_status="NOT_STARTED" if index >= len(run_entries) else "STARTED_NO_TERMINAL_REPORT")
            if index >= len(run_entries):
                continue
            run_entry = run_entries[index]
            report_path = resolve_path(run_entry["report_path"], batch_path)
            row["report_path"] = str(report_path)
            if not report_path.exists():
                continue
            report = checked_json(report_path, run_entry.get("report_sha256"), "run_report")
            report_hash = provenance[-1]["sha256"]
            if report.get("policy") != row["policy"] or report.get("configuration_sha256") != configs[condition]["configuration_sha256"] or report.get("target_source") != manifest["target_source"]:
                raise ValueError("Raw report policy/configuration/target source mismatch")
            versions = report.get("source_sha256")
            if not isinstance(versions, dict) or not versions:
                raise ValueError("Raw report source version evidence missing")
            if source_versions is None:
                source_versions = versions
            elif source_versions != versions:
                raise ValueError("Different source versions require separate sweep analyses")
            summary_row = summarize_run(report)
            row.update(summary_row, ledger_status="TERMINAL_REPORT", source_sha256=versions, report_sha256=report_hash)
            row["solver_elapsed_s"] = descriptive(record.get("solver_elapsed_s") for record in report.get("planning_records", []))
            row["observed_max_axis_error"] = descriptive(observation_error_values(report))
            row["observed_max_axis_error_scope"] = "finite maximum absolute image-axis error of detected approach observations; truncated detections are not repaired; no identity probability claim"
            row["truncated_approach_observation_count"] = sum(value.get("truncated") is True for value in report.get("observations", []))
            row["undetected_approach_observation_count"] = sum(value.get("detected") is not True for value in report.get("observations", []))
            row["terminal_mean_max_axis_error"] = finite_scalar(report.get("metrics", {}).get("terminal_mean_max_axis_error"))
            if analyzed is not None:
                matches = [value for value in analyzed.get("runs", []) if Path(value.get("report_path", "")).resolve() == report_path]
                if len(matches) != 1:
                    raise ValueError("Session analysis does not enumerate exactly one matching raw report")
                for key in ("task_success", "safe_landing_and_release", "physical_sampled_budget_status", "command_norm_violations"):
                    if matches[0].get(key) != summary_row.get(key):
                        raise ValueError(f"Session analysis/raw recomputation differs: {key}")
            for observation_index, observation in enumerate(report.get("observations", [])):
                position = finite_vector(observation.get("world_ned_m"))
                if position is not None:
                    plot_points.append({"condition_id": condition, "repetition": repetition, "planned_session_index": plan_index,
                        "policy": row["policy"], "report_path": str(report_path), "report_sha256": report_hash,
                        "observation_index": observation_index, "world_x_m": float(position[0]), "world_y_m": float(position[1]),
                        "world_z_ned_m": float(position[2]), "scope": "approach observation only; positioning excluded"})
        all_safe = len(run_entries) == len(planned_rows) and all(row["safe_landing_and_release"] for row in planned_rows)
        session_row["independent_launch_evidence_verified"] = identity_valid and inner.get("engine_closed_after_safe_completion") is True and all_safe
        if not session_row["independent_launch_evidence_verified"]:
            session_row["evidence_notes"].append("INDEPENDENT_SAFE_ENGINE_CLOSURE_EVIDENCE_INCOMPLETE")
        if analyzed is not None and analyzed.get("same_run_source_versions") is not None and analyzed.get("same_run_source_versions") != source_versions:
            raise ValueError("Session analysis/source version mismatch")
    pair_summary = summarize_matched_pairs([row for row in rows if row["started"] is True])
    # Preserve condition labels alongside each pair; source helper's session ID
    # is the globally unique planned index, not each manifest's local index 1.
    for pair in pair_summary["pairs"]:
        plan = plans[pair["session"] - 1]
        pair.update(condition_id=plan["condition_id"], repetition=plan["repetition"])
        run_pair = {row["policy"]: row for row in rows if row["session"] == pair["session"] and row["started"] is True}
        completed = len(run_pair) == 2 and all(row.get("task_success") is True for row in run_pair.values())
        pair["both_tasks_successful"] = completed
        differences = {}
        for field in ("approach_duration_s", "sampled_approach_path_m"):
            left = finite_scalar(run_pair.get("efficiency", {}).get(field))
            right = finite_scalar(run_pair.get("perception", {}).get(field))
            differences[field] = right - left if completed and left is not None and right is not None else None
        pair["common_successful_task_descriptive_difference_perception_minus_efficiency"] = differences
        pair["performance_difference_scope"] = "only two successful tasks share a completed stopping criterion; differences are descriptive, not superiority/significance estimates"
    summary = {"kind": "MATCHED_START_SENSOR_LATERAL_SWEEP_DESCRIPTIVE_ANALYSIS",
        "formal_thesis_effect_results": False, "generated_local": datetime.now().astimezone().isoformat(),
        "sweep_path": str(sweep_path), "sweep_status": sweep.get("status"),
        "planned_flights": len(rows), "known_started_flights": sum(row["started"] is True for row in rows),
        "verified_not_started_flights": sum(row["started"] is False for row in rows),
        "unknown_start_status_flights": sum(row["started"] is None for row in rows),
        "task_success_count": sum(row["task_success"] is True for row in rows),
        "safe_landing_and_release_count": sum(row["safe_landing_and_release"] is True for row in rows),
        "missing_terminal_report_count": sum(row["ledger_status"] == "STARTED_NO_TERMINAL_REPORT" for row in rows),
        "independent_launch_evidence_verified_session_count": sum(row["independent_launch_evidence_verified"] for row in session_rows),
        "same_source_versions": source_versions, "sessions": session_rows, "runs": rows,
        "condition_policy_descriptions": aggregate(rows, list(configs)), "initial_state_pairs": pair_summary,
        "hash_audit": provenance, "evidence_notes": notes,
        "limits": ["Small development pilot, no superiority/significance/causal inference", "Repetition counts follow the frozen session plan; this development sample is not formal thesis effectiveness data",
            "Engine PID plus StartTime and safe closure record show distinct launches; they do not prove statistically independent errors",
            "Missing manifest/batch after STARTED leaves flight-start status unknown; unknown is never recoded as not flown",
            "Positioning and its warmup are excluded from approach metrics and plots", "Recorded state queries do not bound unsampled physical peaks",
            "Fixed-height reference feedback is not exact quintic following or full three-dimensional optimization; actual reference and ramp modes are recorded per run"],
        "ai_assistance": "Analysis implementation and narrative assisted by AI; flight facts bound to the raw files and hash audit"}
    return summary, plot_points


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()} for row in rows)


def render(summary, points, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conditions = list(dict.fromkeys(row["condition_id"] for row in summary["sessions"]))
    figure, axes = plt.subplots(1, len(conditions), figsize=(5 * len(conditions), 4.4), squeeze=False, constrained_layout=True)
    colors = {"efficiency": "#2166ac", "perception": "#d6604d"}
    for axis, condition in zip(axes[0], conditions):
        selected = [point for point in points if point["condition_id"] == condition]
        for session in sorted({point["planned_session_index"] for point in selected}):
            for policy in colors:
                trace = [point for point in selected if point["planned_session_index"] == session and point["policy"] == policy]
                if trace:
                    axis.plot([point["world_x_m"] for point in trace], [point["world_y_m"] for point in trace],
                              ".-", color=colors[policy], alpha=.75, label=f"{policy}, session {session}", linewidth=1)
        axis.set(title=condition, xlabel="World NED x (m)", ylabel="World NED y (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=7)
        else:
            axis.text(.5, .5, "No recorded approach points", ha="center", va="center", transform=axis.transAxes)
    figure.suptitle("Recorded approach positions\nDescriptive matched-start pilot" if len(conditions) == 1
                   else "Recorded approach positions: descriptive matched-start pilot")
    figure.savefig(output / "approach_xy.png", dpi=160)
    plt.close(figure)
    table_rows = []
    for row in summary["runs"]:
        match = row.get("initial_state_match", {})
        values = match.get("accepted_state_recomputed_values") or {}
        motion = match.get("sampled_motion", {})
        number = lambda value: f"{value:.4f}" if finite_scalar(value) is not None else "n/a"
        table_rows.append([str(row["planned_session_index"]), row["policy"], match.get("status", row["ledger_status"]),
            number(values.get("position_error_norm_m")), number(values.get("speed_norm_m_s")), number(values.get("acceleration_norm_m_s2")),
            str(motion.get("speed_budget_exceedances", "n/a")) + "/" + str(motion.get("acceleration_budget_exceedances", "n/a")),
            str(row.get("observed_approach_speed_budget_exceedances", "n/a")) + "/" + str(row.get("observed_approach_acceleration_budget_exceedances", "n/a")),
            str(row.get('approach_spatial_budget',{}).get('entered_inner_margin_samples','n/a'))])
    figure, axis = plt.subplots(figsize=(16, max(3, .35 * len(table_rows) + 1.6)))
    axis.axis("off")
    table = axis.table(cellText=table_rows, colLabels=["Session", "Policy", "Initial gate", "Start error m", "Start speed m/s", "Start acc m/s2", "Positioning v/a violations", "Approach v/a violations", "Approach margin violations"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)
    axis.set_title("Paired start and sampled motion-budget evidence; n/a is unassessed")
    figure.tight_layout()
    figure.savefig(output / "paired_start_budget_table.png", dpi=160)
    plt.close(figure)


def main(sweep_path, output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    summary, points = audit_sweep(sweep_path)
    output.mkdir(parents=True, exist_ok=False)
    analyzer_path = Path(__file__).resolve()
    summary["analysis_source"] = {"path": str(analyzer_path), "sha256": sha256(analyzer_path),
                                  "dependency_path": str(Path(summarize_run.__code__.co_filename).resolve()),
                                  "dependency_sha256": sha256(summarize_run.__code__.co_filename)}
    timed_audit_path = Path(summarize_run.__code__.co_filename).with_name("trajectory_timed_audit.py")
    summary["analysis_source"].update(timed_audit_path=str(timed_audit_path.resolve()), timed_audit_sha256=sha256(timed_audit_path))
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "all_planned_flights.csv", summary["runs"])
    write_csv(output / "condition_policy_descriptions.csv", summary["condition_policy_descriptions"])
    write_csv(output / "plot_points.csv", points)
    write_csv(output / "session_launch_evidence.csv", summary["sessions"])
    write_csv(output / "paired_states.csv", summary["initial_state_pairs"]["pairs"])
    lines = ["# 实测起点匹配横向工况开发先导", "", f"批次状态：`{summary['sweep_status']}`。", "",
        f"计划 {summary['planned_flights']} 次；确认已开始 {summary['known_started_flights']} 次；确认未开始 {summary['verified_not_started_flights']} 次；起始状态未知 {summary['unknown_start_status_flights']} 次；已开始但缺终态报告 {summary['missing_terminal_report_count']} 次。", "",
        f"任务达标 {summary['task_success_count']} 次，落地并释放控制 {summary['safe_landing_and_release_count']} 次。完整台账见 all_planned_flights.csv。", "",
        "| 工况 | 方法 | 任务达标/确认已开始 | 计划 | 起始未知 | 安全收尾 | 初始验收失败 | 定位/接近速度加速度超限试次 | 接近内侧保护界超限试次 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["condition_policy_descriptions"]:
        lines.append("| " + " | ".join(str(value) for value in (group["condition_id"], group["policy"],
            f"{group['task_success_count']}/{group['known_started_flights']}", group["planned_flights"], group["unknown_start_status_flights"],
            group["safe_landing_and_release_count"], group["initial_acceptance_failure_count"],
            f"{group['positioning_physical_budget_exceedance_run_count']}/{group['approach_physical_budget_exceedance_run_count']}",
            group['approach_inner_protection_margin_exceedance_run_count'])) + " |")
    lines += ["", "## 任务失败原因", ""]
    for group in summary["condition_policy_descriptions"]:
        for reason, count in group["task_failure_reason_counts"].items():
            lines.append(f"- `{group['condition_id']}` / `{group['policy']}`：{count} 次；{reason}")
    lines += ["", "所有定位失败、保护退出和缺终态报告保留在确认已开始分母。STARTED 会话缺 manifest 或 batch 时不能断言未飞，相关方法行保持起始未知；存在未知时，表中任务比例仅描述已确认分母。", "",
        f"不同 PID＋StartTime 并记录安全关闭的完整会话 {summary['independent_launch_evidence_verified_session_count']} 个；共同接受范围内的已核验初始状态配对 {summary['initial_state_pairs']['paired_session_sample_count']} 个。两方法状态不要求绝对相同，位置/速度/加速度差阈值为同一验收 norm 门槛的两倍。进程记录支持不同仿真启动，不能证明误差统计独立。", "",
        "接近时间、采样路径、每回合求解均值/P50/P95和已检测观测的最大轴居中误差在 condition_policy_descriptions.csv 与 summary.json；采用每回合先汇总再描述的口径。定位和 warmup 不计入接近指标。未采样峰值没有保证。", "",
        "![实际接近XY轨迹](approach_xy.png)", "", "![起点与预算配对证据](paired_start_budget_table.png)", "",
        "图点由 plot_points.csv 绑定原始报告及哈希；完整文件审计在 summary.json.hash_audit。所有已有登记哈希不一致均拒绝分析，部分失败台账未登记的哈希会显式列出证据缺口。", "",
        "本组仍为固定高度参考反馈的小规模开发先导；实际参考时间与 ramp 模式见每条运行记录。不能据此推断方法优势、显著性、完整三维轨迹优化效果或完成硕士论文正式实验。", "",
        "AI使用说明：实现和整理由AI辅助，结果来自所登记的仿真原始数据。"]
    if summary["evidence_notes"]:
        lines += ["", "## 哈希登记证据缺口", ""] + [f"- `{value}`" for value in summary["evidence_notes"]]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render(summary, points, output)
    artifact_provenance = {"analysis_source": summary["analysis_source"], "inputs": summary["hash_audit"],
        "outputs": [{"path": str(path), "sha256": sha256(path)} for path in sorted(output.iterdir()) if path.is_file()]}
    (output / "provenance.json").write_text(json.dumps(artifact_provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("planned_flights", "known_started_flights", "unknown_start_status_flights", "task_success_count", "safe_landing_and_release_count")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.sweep, args.output)
