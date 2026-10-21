"""Analyze a supervised Blocks pilot by session and paired flight.

Failed and unassessed flights remain in the denominator. Plots and paired
differences include completed pairs only; no population inference is made.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hash(path, expected):
    if expected and sha256(path) != expected.lower():
        raise ValueError(f"Recorded report hash no longer matches: {path}")


def analyze(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = load(manifest_path)
    rows, pairs, sessions = [], [], []
    versions = set()
    for session in manifest["sessions"]:
        batch_path = Path(session["batch_path"]) / "batch_report.json"
        if not batch_path.is_file():
            sessions.append({"session": session["session_index"], "status": session["status"],
                             "started_runs": 0, "complete_pairs": 0})
            continue
        verify_hash(batch_path, session.get("batch_sha256"))
        batch = load(batch_path)
        session_start = len(rows)
        groups = {}
        registered = list(batch["runs"])
        active = batch.get("active_run")
        if active and not any(row["run"] == active["run"] for row in registered):
            raw_path = Path(active["report_path"])
            if raw_path.is_file():
                raw = load(raw_path)
                registered.append(dict(active, status=raw["status"],
                                       task_success=raw.get("metrics", {}).get("task_success")))
            else:
                item = {"session": session["session_index"], "run": active["run"],
                        "pair": active["pair"], "policy": active["policy"],
                        "status": "INTERRUPTED" if manifest["status"] in ("FAIL", "INTERRUPTED") else "RUNNING",
                        "task_success": None, "approach_duration_s": None,
                        "approach_path_m": None, "terminal_mean_max_axis_error": None,
                        "failed_phase": "UNRECORDED", "error": "Started flight has no terminal report",
                        "raw_report": str(raw_path), "raw_report_sha256": None}
                rows.append(item)
        for row in registered:
            raw_path = Path(row["report_path"])
            raw = load(raw_path)
            metrics = raw.get("metrics", {})
            if raw["status"] != row["status"] or metrics.get("task_success") != row.get("task_success"):
                raise ValueError(f"Batch and raw report disagree: {raw_path}")
            versions.add(json.dumps(raw.get("source_sha256", {}), sort_keys=True))
            versions.add("config:" + json.dumps(raw["configuration"], sort_keys=True))
            item = {"session": session["session_index"], "run": row["run"],
                    "pair": row["pair"], "policy": row["policy"], "status": row["status"],
                    "task_success": row.get("task_success"),
                    "approach_duration_s": metrics.get("approach_duration_s"),
                    "approach_path_m": metrics.get("approach_path_m"),
                    "terminal_mean_max_axis_error": metrics.get("terminal_mean_max_axis_error"),
                    "failed_phase": raw.get("failed_phase"), "error": raw.get("error"),
                    "raw_report": str(raw_path), "raw_report_sha256": sha256(raw_path)}
            rows.append(item)
            groups.setdefault(item["pair"], []).append(item)
        count = 0
        for pair_id, group in groups.items():
            indexed = {item["policy"]: item for item in group}
            complete = (len(group) == 2 and set(indexed) == {"coordinate", "visual"}
                        and all(item["status"] == "PASS" for item in group)
                        and all(isinstance(item["task_success"], bool) for item in group))
            if not complete:
                continue
            count += 1
            difference = {"session": session["session_index"], "pair": pair_id,
                          "order": " -> ".join(item["policy"] for item in group)}
            for key in ("approach_duration_s", "approach_path_m"):
                values = [indexed[policy][key] for policy in ("coordinate", "visual")]
                if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
                    raise ValueError(f"Completed pair has invalid {key}: {group}")
                difference[f"coordinate_{key}"] = values[0]
                difference[f"visual_{key}"] = values[1]
                difference[f"visual_minus_coordinate_{key}"] = values[1] - values[0]
            pairs.append(difference)
        sessions.append({"session": session["session_index"], "status": batch["status"],
                         "engine_process_id": session["engine_process_id"],
                         "started_runs": len(rows) - session_start, "complete_pairs": count})
    # There should be one configuration and one source-hash dictionary.
    if len(versions) > 2:
        raise ValueError("Mixed controller versions/configurations: analyze them separately")
    policies = {}
    for policy in ("coordinate", "visual"):
        subset = [row for row in rows if row["policy"] == policy]
        policies[policy] = {"started": len(subset),
                            "safe_pass": sum(row["status"] == "PASS" for row in subset),
                            "task_success": sum(row["task_success"] is True for row in subset),
                            "assessed_task_failure": sum(row["task_success"] is False for row in subset),
                            "unassessed": sum(row["task_success"] is None for row in subset)}
    return {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "pilot_status": manifest["status"], "planned_sessions": manifest["planned_sessions"],
            "session_count": len(sessions), "complete_pair_count": len(pairs),
            "sessions": sessions, "policies": policies, "runs": rows, "pairs": pairs,
            "interpretation": "Fixed-scene repeatability pilot; cluster results by Unreal session. No generalization claim."}


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, path):
    if not summary["pairs"]:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    for axis, key, ylabel in zip(axes, ("approach_duration_s", "approach_path_m"),
                                 ("Approach time (s)", "Approach path (m)")):
        for pair in summary["pairs"]:
            axis.plot([0, 1], [pair[f"coordinate_{key}"], pair[f"visual_{key}"]],
                      marker="o", alpha=0.8,
                      label=f"S{pair['session']} P{pair['pair']}")
        axis.set_xticks([0, 1], ["Coordinate", "Visual"])
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("Fixed-target pilot: completed pairs (grouped by session)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write(summary, folder, make_plot=True):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_write(folder / "runs.csv", summary["runs"])
    csv_write(folder / "pairs.csv", summary["pairs"])
    lines = ["# 第一阶段 Blocks 重复性先导结果", "",
             f"运行状态：{summary['pilot_status']}；登记会话 {summary['session_count']}/"
             f"{summary['planned_sessions']}，完整配对 {summary['complete_pair_count']}。", "",
             "| 策略 | 已开始试次 | 安全收尾 | 任务成功 | 已评价任务失败 | 未完成任务评价 |",
             "|---|---:|---:|---:|---:|---:|"]
    for policy, counts in summary["policies"].items():
        lines.append(f"| {policy} | {counts['started']} | {counts['safe_pass']} | "
                     f"{counts['task_success']} | {counts['assessed_task_failure']} | {counts['unassessed']} |")
    lines.extend(["", "| 会话 | 配对 | 顺序 | 视觉减坐标用时 / s | 视觉减坐标路径 / m |",
                  "|---|---|---|---:|---:|"])
    for pair in summary["pairs"]:
        lines.append(f"| {pair['session']} | {pair['pair']} | {pair['order']} | "
                     f"{pair['visual_minus_coordinate_approach_duration_s']:+.3f} | "
                     f"{pair['visual_minus_coordinate_approach_path_m']:+.3f} |")
    lines.extend(["", "这里报告固定场景的重复性；各会话内试次相关，不估计总体优势。"
                  "失败试次保留在已开始试次的分母中，配对差值只使用两策略都完成的配对。", "",
                  "`summary.json` 保存清单和原始报告哈希；`runs.csv` 保留错误阶段与原因。", "",
                  "AI 使用说明：脚本与结果解释由 AI 辅助生成，数值来自实际运行报告。", ""])
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    if make_plot:
        plot(summary, folder / "paired_metrics.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result = analyze(args.manifest)
    write(result, args.output, not args.no_plot)
    print(json.dumps({"status": result["pilot_status"], "complete_pairs": result["complete_pair_count"],
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
