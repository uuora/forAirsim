"""Read-only timing evidence for single-drone identity/depth controller pilots.

The source/plan/report/execution seals are verified before timing is used. This
module does not import an AirSim client or call a model. Host query receipts,
motion gate decisions, command dispatches and full frame completions are distinct
clocks; no result establishes hard real time, continuous safety or efficacy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil

from analyze_t1_identity_flight import (TARGET_PURPOSE, STAGE,
    digest, finite, lease_clear_audit, read_json, source_checks, verify_file)


def distribution(values):
    values = sorted(float(value) for value in values if finite(value))
    def percentile(fraction):
        if not values:
            return None
        index = fraction * (len(values)-1)
        lower = math.floor(index)
        upper = math.ceil(index)
        return values[lower] + (values[upper]-values[lower]) * (index-lower)
    return {"count": len(values), "min_s": min(values, default=None),
        "median_s": percentile(.5), "p95_s": percentile(.95),
        "p99_s": percentile(.99), "max_s": max(values, default=None)}


def receipt_cadence(rows, start=None, end=None):
    """Unique host ticks; duplicate receipts never manufacture short intervals."""
    stamps = [row.get("host_receipt_monotonic_s") for row in rows]
    invalid = sum(not finite(value) for value in stamps)
    finite_stamps = [value for value in stamps if finite(value)]
    raw_regressions = sum(right < left for left, right in zip(finite_stamps, finite_stamps[1:]))
    filtered = [value for value in finite_stamps
        if (not finite(start) or value >= start) and (not finite(end) or value <= end)]
    unique = sorted(set(filtered))
    gaps = [right-left for left, right in zip(unique, unique[1:])]
    return {"saved_record_count": len(rows), "invalid_receipt_count": invalid,
        "in_stage_record_count": len(filtered), "unique_host_receipt_count": len(unique),
        "same_host_tick_extra_records": len(filtered)-len(unique),
        "raw_event_order_regressions": raw_regressions, "gap_distribution": distribution(gaps),
        "gaps_over_0p1_s": sum(value > .1+1e-9 for value in gaps),
        "gaps_over_0p12_s": sum(value > .12+1e-9 for value in gaps),
        "gaps_over_0p4_s": sum(value > .4+1e-9 for value in gaps),
        "stage_head_gap_s": unique[0]-start if unique and finite(start) else None,
        "stage_tail_gap_s": end-unique[-1] if unique and finite(end) else None,
        "first_host_receipt_s": unique[0] if unique else None,
        "last_host_receipt_s": unique[-1] if unique else None,
        "receipts_monotonic_s": unique,
        "gap_rows": [{"left_receipt_s": left, "right_receipt_s": right, "gap_s": right-left}
            for left, right in zip(unique, unique[1:])],
        "scope": "intervals between saved same-host query receipts; not exposure, simulator time or a continuous query guarantee"}


def stage_bounds(report, events):
    start = report.get("low_speed_stage_start_state", {}).get("host_receipt_monotonic_s")
    end = report.get("low_speed_stage_end_state", {}).get("host_receipt_monotonic_s")
    stamps = [row.get("host_receipt_monotonic_s") for row in events
        if row.get("kind") == "SELF_STATE" and row.get("stage") == STAGE]
    known = [value for value in stamps if finite(value)]
    if not finite(start):
        start = min(known, default=None)
    if not finite(end):
        end = max(known, default=None)
    return start, end


def is_nonzero_target(command):
    xy = command.get("velocity_xy_m_s")
    return command.get("purpose") == TARGET_PURPOSE and isinstance(xy, (tuple, list)) and len(xy) == 2 \
        and all(finite(value) for value in xy) and math.hypot(*xy) > 1e-6


def interval_evidence(interval, cadence, commands, stops):
    """Coverage includes boundary gaps; zero in-interval samples is unknown."""
    start, end = interval.get("started_monotonic_s"), interval.get("completed_monotonic_s")
    result = dict(interval)
    if not finite(start) or not finite(end) or end <= start:
        return dict(result, status="INVALID_ACTUAL_BLOCK_INTERVAL")
    stamps = cadence["receipts_monotonic_s"]
    inside = [stamp for stamp in stamps if start <= stamp <= end]
    partition = [start] + inside + [end]
    touching_gaps = [row["gap_s"] for row in cadence["gap_rows"]
        if row["left_receipt_s"] < end and row["right_receipt_s"] > start]
    prior = max((stamp for stamp in stamps if stamp <= start), default=None)
    after = min((stamp for stamp in stamps if stamp >= end), default=None)
    dispatched = [row for row in commands if finite(row.get("dispatch_monotonic_s"))
        and start <= row["dispatch_monotonic_s"] < end]
    overlap = []
    for command in commands:
        if not is_nonzero_target(command) or not finite(command.get("dispatch_monotonic_s")):
            continue
        clear = stops.get(command.get("target_frame_id"), {}).get("actual_zero_clear_s")
        if not finite(clear):
            continue
        left = max(start, command["dispatch_monotonic_s"])
        right = min(end, clear)
        if right > left:
            overlap.append({"target_frame_id": command.get("target_frame_id"),
                "overlap_start_s": left, "overlap_end_s": right, "duration_s": right-left})
    return dict(result, status="ACTUAL_BLOCK_INTERVAL_ANALYZED", actual_duration_s=end-start,
        self_state_receipts_inside=len(inside),
        max_boundary_clipped_receipt_gap_s=max((right-left for left, right in zip(partition, partition[1:])), default=None),
        max_whole_receipt_gap_touching_interval_s=max(touching_gaps, default=None),
        bracketed_by_saved_self_receipts=finite(prior) and finite(after),
        prior_self_receipt_s=prior, first_after_self_receipt_s=after,
        own_state_command_dispatches_inside=sum(row.get("purpose", "").startswith("OWN_STATE") for row in dispatched),
        nonzero_target_dispatches_inside=sum(is_nonzero_target(row) for row in dispatched),
        total_command_dispatches_inside=len(dispatched),
        nonzero_target_active_overlap_rows=overlap,
        scope="actual saved block interval and host receipts; no samples does not imply controller coverage")


def frame_deadlines(report):
    frames = {row.get("slot_index"): row for row in report.get("frames", [])}
    rows = []
    for slot in report.get("slots", []):
        frame = frames.get(slot.get("slot_index"), {})
        deadline = slot.get("deadline_monotonic_s")
        gate = frame.get("motion_gate", {}).get("decision_monotonic_s")
        observer = frame.get("observer_result", {}).get("process_completed_host_monotonic_s")
        completed = frame.get("decision_completed_monotonic_s")
        def delay(value):
            return value-deadline if finite(value) and finite(deadline) else None
        def verdict(value):
            if not finite(value) or not finite(deadline):
                return "UNKNOWN"
            return "TIMELY" if value <= deadline else "LATE"
        rows.append({"slot_index": slot.get("slot_index"), "phase_id": slot.get("phase_id"),
            "slot_status": slot.get("status"), "frame_id": frame.get("frame_id"),
            "nominal_deadline_s": deadline, "gate_decision_s": gate, "observer_completed_s": observer,
            "full_completion_s": completed, "gate_deadline_status": verdict(gate),
            "observer_deadline_status": verdict(observer), "full_completion_deadline_status": verdict(completed),
            "gate_after_nominal_deadline_s": delay(gate), "observer_after_nominal_deadline_s": delay(observer),
            "full_completion_after_nominal_deadline_s": delay(completed),
            "gate_status": frame.get("motion_gate", {}).get("status")})
    return {"planned_slots": len(report.get("slots", [])), "saved_frames": len(report.get("frames", [])),
        "gate_deadline_counts": dict(Counter(row["gate_deadline_status"] for row in rows)),
        "observer_deadline_counts": dict(Counter(row["observer_deadline_status"] for row in rows)),
        "full_completion_deadline_counts": dict(Counter(row["full_completion_deadline_status"] for row in rows)),
        "rows": rows,
        "scope": "all reported planned slots, including misses; full completion may include execution and stopping and is not the motion gate time"}


def timing_metrics(report, events):
    start, end = stage_bounds(report, events)
    monitors = [row for row in events if row.get("kind") == "SELF_STATE" and row.get("stage") == STAGE]
    saved = list(monitors)
    saved += [row["command_self_state"] for row in report.get("frames", []) if "command_self_state" in row]
    saved += [report[key] for key in ("low_speed_stage_start_state", "low_speed_stage_end_state") if key in report]
    monitor_cadence = receipt_cadence(monitors, start, end)
    combined_cadence = receipt_cadence(saved, start, end)
    commands = [row for row in events if row.get("kind") == "COMMAND" and row.get("stage") == STAGE]
    frames = {row.get("frame_id"): row for row in report.get("frames", [])}
    clears = []
    for command in commands:
        if is_nonzero_target(command):
            audit = lease_clear_audit(command, frames.get(command.get("target_frame_id")), events)
            clears.append(dict(audit, target_frame_id=command.get("target_frame_id")))
    stops = {row["target_frame_id"]: row for row in clears}
    delays = [interval_evidence(row, monitor_cadence, commands, stops)
        for row in report.get("injected_delays", [])]
    positive = [row.get("clear_after_declared_lease_s") for row in clears
        if finite(row.get("clear_after_declared_lease_s"))]
    margins = [-row["clear_after_observation_deadline_s"] for row in clears
        if finite(row.get("clear_after_observation_deadline_s"))]
    return {"stage_start_host_receipt_s": start, "stage_end_host_receipt_s": end,
        "declared_self_monitor_period_s": report.get("configuration", {}).get("self_monitor_period_s"),
        "monitor_event_receipt_cadence": monitor_cadence,
        "all_saved_stage_receipt_cadence": combined_cadence,
        "nonzero_target_command_count": sum(is_nonzero_target(row) for row in commands),
        "lease_stop_rows": clears,
        "lease_clear_after_declared_end_distribution": distribution(positive),
        "observation_deadline_margin_distribution": distribution(margins),
        "lease_clear_audit_issue_counts": dict(Counter(issue for row in clears for issue in row["issues"])),
        "missing_or_unverified_explicit_target_clears": sum(row["status"] != "EXPLICIT_ZERO_CLEAR_RECORDED" or bool(row["issues"]) for row in clears),
        "injected_delay_rows": delays, "frame_timing": frame_deadlines(report),
        "clock_scope": "same-host monotonic receipts and dispatches; not sensor exposure or simulator-state clock",
        "controller_coverage_scope": "SELF_STATE events only for monitor cadence; command queries and stage bounds included separately",
        "formal_efficacy_results": False, "hard_real_time_demonstrated": False,
        "continuous_safety_proven": False}


def analyze_run(root):
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    result = {"run": str(root), "status": "INPUTS_NOT_VERIFIED", "timing": None,
        "source_receipts": [], "formal_efficacy_results": False}
    try:
        manifest = read_json(manifest_path)
        report_path = Path(manifest.get("report", root / "flight/report.json"))
        checks = [verify_file(report_path, manifest.get("report_sha256"), "launcher_runner_report")]
        result.update(launcher_status=manifest.get("status"), runner_status=manifest.get("runner_status"))
        if checks[0]["status"] != "PASS":
            result.update(source_receipts=checks, status="INPUT_HASH_VERIFICATION_FAILED")
            return result
        report = read_json(report_path)
        plan_path = Path(report["plan"])
        checks += [verify_file(plan_path, report.get("plan_sha256"), "runner_plan"),
            verify_file(plan_path, manifest.get("plan_sha256"), "launcher_plan")]
        if any(row["status"] != "PASS" for row in checks):
            result.update(source_receipts=checks, status="INPUT_HASH_VERIFICATION_FAILED")
            return result
        plan = read_json(plan_path)
        checks += source_checks(plan, root, report)
        execution = report.get("execution_log", {})
        execution_path = Path(execution.get("path", report_path.parent / "execution.jsonl"))
        checks.append(verify_file(execution_path, execution.get("sha256"), "execution_jsonl"))
        checks.append({"label": "nonempty_frozen_project_source_inventory", "status": "PASS" if plan.get("source_inventory") else "FAIL"})
        result.update(session_id=report.get("session_id"), source_receipts=checks)
        if any(row["status"] != "PASS" for row in checks):
            result["status"] = "INPUT_HASH_VERIFICATION_FAILED"
            return result
        events = []
        for number, line in enumerate(execution_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            event = json.loads(line)
            if not isinstance(event, dict):
                raise ValueError(f"Execution line {number} is not an object")
            events.append(event)
        result.update(status="SEALED_TIMING_ANALYZED", report_path=str(report_path),
            report_sha256=checks[0]["actual_sha256"], frozen_plan_path=str(plan_path),
            frozen_plan_sha256=digest(plan_path), execution_path=str(execution_path),
            execution_sha256=digest(execution_path), timing=timing_metrics(report, events),
            saved_source_inventory=plan.get("source_sha256"),
            source_lock_scope=manifest.get("source_lock_scope"))
    except (OSError, KeyError, ValueError, TypeError) as exc:
        result.update(status="INPUT_PARSE_OR_SCHEMA_FAILURE", error=f"{type(exc).__name__}: {exc}")
    return result


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def plot_timing(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    valid = [row for row in runs if row.get("timing")]
    if not valid:
        return []
    fig, axes = plt.subplots(len(valid), 1, figsize=(11, 3.3*len(valid)), squeeze=False)
    for axis, run in zip(axes[:, 0], valid):
        timing = run["timing"]
        start = timing["stage_start_host_receipt_s"]
        gaps = timing["monitor_event_receipt_cadence"]["gap_rows"]
        if finite(start):
            axis.plot([row["right_receipt_s"]-start for row in gaps],
                [row["gap_s"]*1000 for row in gaps], ".-", ms=3, lw=.8, label="Saved SELF_STATE receipt intervals")
            for index, row in enumerate(timing["injected_delay_rows"]):
                if row.get("status") == "ACTUAL_BLOCK_INTERVAL_ANALYZED":
                    axis.axvspan(row["started_monotonic_s"]-start, row["completed_monotonic_s"]-start,
                        color="#e4a94d", alpha=.3, label="Injected perception block" if index == 0 else None)
        axis.axhline(50, color="#267b6f", ls="--", lw=1, label="Declared 50 ms monitor period")
        axis.axhline(120, color="#b04b3c", ls=":", lw=1, label="120 ms self-state age limit")
        axis.set(title=run.get("session_id", Path(run["run"]).name),
            xlabel="Elapsed from stage start receipt (s)", ylabel="Receipt interval (ms)")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8, loc="upper right")
    fig.suptitle("Recorded controller query cadence; no hard real-time or continuous-safety claim", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    files = []
    for suffix in ("png", "pdf"):
        path = output / ("self_state_receipt_cadence."+suffix)
        fig.savefig(path, dpi=170, bbox_inches="tight")
        files.append(path)
    plt.close(fig)
    return files


def analyze(roots, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runs = [analyze_run(root) for root in roots]
    report = {"kind": "T1_ASYNC_CONTROLLER_TIMING_ANALYSIS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["status"] == "SEALED_TIMING_ANALYZED" for row in runs) else "INPUT_FAILURES_RETAINED",
        "runs": runs, "formal_efficacy_results": False,
        "limitations": ["Host receipt intervals are saved query evidence, not continuous physical safety or hard real time.",
            "Monitor SELF_STATE event cadence and union saved self-state receipts have different coverage.",
            "Motion gate, observer completion and full frame completion are separate events.",
            "Injected perception blocking tests controller isolation; they do not reproduce every cloud/network fault.",
            "Chronological pilot runs are development evidence, not matched efficacy trials."]}
    report_path = output / "report.json"
    save(report_path, report)
    figures = plot_timing(runs, output)
    lines = ["# 任务一异步控制器时序证据", "", "统计来自冻结计划、报告与执行日志，仿真结果和失败均按原始状态保留。", "",
        "| 试次 | 输入验证 | SELF_STATE 最大间隙 / P95（ms） | 超过 120 ms | 有效非零目标指令 | 清零审计问题 |", "|---|---|---:|---:|---:|---:|"]
    for run in runs:
        timing = run.get("timing")
        if timing:
            cadence = timing["monitor_event_receipt_cadence"]
            values = cadence["gap_distribution"]
            rendered = " / ".join(f"{values[key]*1000:.1f}" if finite(values[key]) else "未知" for key in ("max_s", "p95_s"))
            lines.append(f"| {run.get('session_id')} | {run['status']} | {rendered} | {cadence['gaps_over_0p12_s']} | {timing['nonzero_target_command_count']} | {sum(timing['lease_clear_audit_issue_counts'].values())} |")
        else:
            lines.append(f"| {Path(run['run']).name} | {run['status']} | 未使用 | 未知 | 未知 | 未知 |")
    lines += ["", "完整处理完成时间与运动门控决策时间分别列于 report.json 的 frame_timing 中，不能互换。", "",
        "时序图仅展示已保存的主机读回记录；50 ms 为配置周期，不能据此声称硬实时或连续安全。感知阻塞期间的边界间隙、主机读回、指令分发与清零记录逐段保留。", "",
        "本报告由 AI 辅助生成，数值由只读脚本计算；论文中的解释应由作者结合原始记录审定。", ""]
    summary_path = output / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    snapshot = output / "source_snapshots"
    snapshot.mkdir()
    sources = [Path(__file__).resolve(), Path(__file__).resolve().with_name("analyze_t1_identity_flight.py")]
    source_rows = []
    for source in sources:
        target = snapshot / source.name
        shutil.copyfile(source, target)
        source_rows.append({"path": str(source), "snapshot_path": str(target), "sha256": digest(source)})
    artifact_paths = [report_path, summary_path]+figures
    receipt = {"kind": "T1_ASYNC_TIMING_ARTIFACT_RECEIPT", "status": report["status"],
        "created_utc": report["created_utc"], "input_runs": [str(Path(root).resolve()) for root in roots],
        "sources": source_rows, "artifacts": [{"path": str(path), "sha256": digest(path)} for path in artifact_paths],
        "formal_efficacy_results": False}
    save(output / "artifact_receipt.json", receipt)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.runs, args.output)
    print(json.dumps({"status": result["status"], "runs": len(result["runs"]), "output": str(args.output.resolve())}, ensure_ascii=False))
