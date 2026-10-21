"""Combine separately audited original/repeat attempts without hiding an interruption."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from analyze_trajectory_matched_sweep import aggregate, audit_sweep, render, sha256, write_csv


def combine(audits):
    rows, sessions, points, hashes = [], [], [], []
    versions = None
    seen_paths = set()
    engine_identities = set()
    for attempt, (summary, attempt_points) in enumerate(audits, 1):
        current_versions = summary.get("same_source_versions")
        if current_versions:
            if versions is None:
                versions = current_versions
            elif versions != current_versions:
                raise ValueError("Different terminal-report source versions; keep separate")
        offset = len(sessions)
        for original in summary["runs"]:
            row = deepcopy(original)
            row.update(attempt_ledger_index=attempt,
                       original_planned_session_index=row["planned_session_index"],
                       planned_session_index=row["planned_session_index"] + offset,
                       session=row["session"] + offset)
            if row.get("report_path"):
                path = str(Path(row["report_path"]).resolve())
                if path in seen_paths:
                    raise ValueError("Duplicate started report path across attempts")
                seen_paths.add(path)
            rows.append(row)
        for original in summary["sessions"]:
            session = deepcopy(original)
            identity = (session.get("engine_process_id"), session.get("engine_started"))
            if identity[0] is not None and identity[1] is not None:
                if identity in engine_identities:
                    raise ValueError("Repeated engine PID/StartTime across attempts")
                engine_identities.add(identity)
            session.update(attempt_ledger_index=attempt,
                           original_planned_session_index=session["planned_session_index"],
                           planned_session_index=session["planned_session_index"] + offset)
            sessions.append(session)
        for original in attempt_points:
            point = deepcopy(original)
            point.update(attempt_ledger_index=attempt,
                         planned_session_index=point["planned_session_index"] + offset)
            points.append(point)
        hashes.extend(summary["hash_audit"])
    condition_ids = list(dict.fromkeys(row["condition_id"] for row in rows))
    summary = {"kind": "MATCHED_START_PILOT_WITH_SEPARATE_REPEAT_AFTER_INTERRUPTION",
        "formal_thesis_effect_results": False,
        "planned_attempt_flights": len(rows),
        "known_started_flights": sum(row["started"] is True for row in rows),
        "verified_not_started_flights": sum(row["started"] is False for row in rows),
        "unknown_start_status_flights": sum(row["started"] is None for row in rows),
        "terminal_report_count": sum(row["ledger_status"] == "TERMINAL_REPORT" for row in rows),
        "missing_terminal_report_count": sum(row["ledger_status"] == "STARTED_NO_TERMINAL_REPORT" for row in rows),
        "task_success_count": sum(row["task_success"] is True for row in rows),
        "safe_landing_and_release_count": sum(row["safe_landing_and_release"] is True for row in rows),
        "independent_safe_pair_sessions": sum(row["independent_launch_evidence_verified"] is True for row in sessions),
        "same_terminal_report_source_versions": versions,
        "runs": rows, "sessions": sessions,
        "condition_policy_descriptions": aggregate(rows, condition_ids),
        "hash_audit": hashes,
        "scope": "all original starts retained; repeated center pair is a separate attempt; unknown terminal safety remains unknown",
        "ai_assistance": "Reconciliation and descriptive analysis assisted by AI; no missing result inferred"}
    return summary, points


def audit_startup_failure(sweep_path):
    """Confirm no flight dispatch reached the batch stage after RPC timeout."""
    sweep_path = Path(sweep_path).resolve()
    sweep = json.loads(sweep_path.read_text(encoding="utf-8-sig"))
    if sweep.get("status") != "STOPPED_FOR_INSPECTION" or len(sweep.get("sessions", [])) != 1:
        raise ValueError("Not one stopped startup attempt")
    entry = sweep["sessions"][0]
    conditions_path = Path(sweep["conditions_path"])
    if sha256(conditions_path) != sweep["conditions_sha256"]:
        raise ValueError("Startup frozen plan hash mismatch")
    conditions = json.loads(conditions_path.read_text(encoding="utf-8-sig"))
    plans = [p for p in conditions["session_plan"]
             if (p["condition_id"], p["repetition"]) ==
             (entry["condition_id"], entry["repetition"])]
    if len(plans) != 1 or plans[0]["method_order"] != entry["policy_order"]:
        raise ValueError("Startup method order differs from frozen plan")
    manifest_path = Path(entry["manifest"])
    if sha256(manifest_path) != entry["manifest_sha256"]:
        raise ValueError("Startup manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    config = next(c for c in conditions["conditions"] if c["condition_id"] == entry["condition_id"])
    if manifest.get("experiment_config_sha256") != config["configuration_sha256"]:
        raise ValueError("Startup manifest configuration differs from frozen plan")
    if manifest.get("status") != "FAIL" or not manifest.get("error", "").startswith("Unreal startup timed out:"):
        raise ValueError("Cannot classify non-startup failure as unstarted flights")
    inner = manifest["sessions"][0]
    if any(key in inner for key in ("scene_report", "probe_report_sha256", "batch_report_sha256")) or (Path(inner["batch_path"]) / "batch_report.json").exists():
        raise ValueError("Evidence reached beyond RPC startup; flight starts require separate audit")
    rows = [{"planned_session_index": 1, "session": 1, "condition_id": entry["condition_id"],
        "repetition": entry["repetition"], "planned_method_index": index, "policy": policy,
        "started": False, "ledger_status": "PRE_RPC_STARTUP_FAILURE_NOT_STARTED",
        "task_success": False, "safe_landing_and_release": False}
        for index, policy in enumerate(entry["policy_order"], 1)]
    return {"runs": rows, "sessions": [{"planned_session_index": 1,
        "condition_id": entry["condition_id"], "repetition": entry["repetition"],
        "status": "PRE_RPC_STARTUP_FAILURE_NOT_STARTED", "independent_launch_evidence_verified": False,
        "scope": "No scene/probe/batch dispatch recorded; zero flight starts, no landing claim"}],
        "same_source_versions": None,
        "hash_audit": [{"kind": "startup_failure_ledger", "path": str(sweep_path), "sha256": sha256(sweep_path)},
                       {"kind": "startup_frozen_plan", "path": str(conditions_path), "sha256": sha256(conditions_path)},
                       {"kind": "startup_failure_manifest", "path": str(manifest_path), "sha256": sha256(manifest_path)}]}, []


def main(original, repeat, output, startup_failure=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    audits = [audit_sweep(original)]
    if startup_failure:
        audits.append(audit_startup_failure(startup_failure))
    audits.append(audit_sweep(repeat))
    summary, points = combine(audits)
    inputs = [original] + ([startup_failure] if startup_failure else []) + [repeat]
    summary["attempt_ledgers"] = [{"path": str(Path(p).resolve()), "sha256": sha256(p)}
                                 for p in inputs]
    summary["pre_rpc_failed_startup_attempts"] = int(startup_failure is not None)
    summary["analysis_source"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__),
        "dependencies": [{"path": str(Path(__file__).with_name(name)),
                          "sha256": sha256(Path(__file__).with_name(name))}
                         for name in ("analyze_trajectory_matched_sweep.py", "analyze_trajectory_pilot.py")]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "all_planned_attempts.csv", summary["runs"])
    write_csv(output / "condition_policy_descriptions.csv", summary["condition_policy_descriptions"])
    write_csv(output / "plot_points.csv", points)
    write_csv(output / "session_launch_evidence.csv", summary["sessions"])
    render(summary, points, output)
    lines = ["# 单机实测起点匹配先导：中断与独立补做汇总", "",
        f"原计划12次，另登记中心反序补做2次；另有{summary['pre_rpc_failed_startup_attempts']}次RPC启动失败的两方法计划没有进入飞行。共{summary['planned_attempt_flights']}次计划尝试。确认已开始{summary['known_started_flights']}次，未开始{summary['verified_not_started_flights']}次，起始未知{summary['unknown_start_status_flights']}次。", "",
        f"终态报告{summary['terminal_report_count']}次；已开始但缺终态{summary['missing_terminal_report_count']}次。任务达标{summary['task_success_count']}次，确认落地释放{summary['safe_landing_and_release_count']}次。缺终态的收尾状态保持未知。", "",
        "| 条件 | 方法 | 达标/全部确认已开始 | 有终态 | 确认安全收尾 | 缺终态 |",
        "|---|---|---:|---:|---:|---:|"]
    for group in summary["condition_policy_descriptions"]:
        lines.append(f"| {group['condition_id']} | {group['policy']} | {group['task_success_count']}/{group['known_started_flights']} | {group['known_started_flights']-group['missing_terminal_report_count']} | {group['safe_landing_and_release_count']} | {group['missing_terminal_report_count']} |")
    lines += ["", "## 失败与中断", ""]
    for group in summary["condition_policy_descriptions"]:
        for reason, count in group["task_failure_reason_counts"].items():
            lines.append(f"- {group['condition_id']} / {group['policy']}：{count}次，{reason}")
    lines += ["", "![实际接近XY轨迹](approach_xy.png)", "",
        "定位/warmup不计入接近图和指标。中断试次缺终态，图未填补其部分日志为完整轨迹；原始JSONL仍保留。完成的同源码三条件各有2个配对会话，属于小规模开发先导；不得从这张图推断方法优势或显著性。", "",
        "固定高度preview仍非完整三维或精确五次多项式跟踪。所有已开始分母和独立补做来源见all_planned_attempts.csv及summary.json。已保存状态无超限只覆盖采样时刻。", "",
        "AI使用说明：工程与整理由AI辅助，数据来自哈希绑定的仿真日志。"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {"inputs": summary["attempt_ledgers"], "analysis_source": summary["analysis_source"],
        "outputs": [{"path": str(p), "sha256": sha256(p)} for p in sorted(output.iterdir())]}
    (output / "provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("known_started_flights", "terminal_report_count",
        "missing_terminal_report_count", "task_success_count", "safe_landing_and_release_count")}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-failure", type=Path)
    args = parser.parse_args()
    main(args.original, args.repeat, args.output, args.startup_failure)
