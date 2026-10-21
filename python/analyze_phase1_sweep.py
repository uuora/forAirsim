"""Describe each lateral condition separately; never pool heterogeneous configurations."""
import argparse
import json
from pathlib import Path

from analyze_phase1_pilot import analyze, load, sha256


def analyze_sweep(path):
    sweep = load(path)
    summaries, configurations, versions = [], [], set()
    for entry in sweep["pilots"]:
        if not entry.get("manifest"):
            continue
        if sha256(entry["manifest"]) != entry["manifest_sha256"]:
            raise ValueError("Child manifest hash mismatch")
        summary = analyze(entry["manifest"])
        summary["condition_id"] = entry["condition_id"]
        summaries.append(summary)
        pilot = load(entry["manifest"])
        if sha256(entry["config"]) != entry["config_sha256"]:
            raise ValueError("Condition file changed")
        config = load(entry["config"])
        if pilot["experiment_config_sha256"] != entry["config_sha256"]:
            raise ValueError("Pilot used a different condition configuration")
        normalized = {k: v for k, v in config.items()
                      if k not in ("condition_id", "protocol_id", "target_world_ned_m")}
        configurations.append(json.dumps(normalized, sort_keys=True))
        for row in summary["runs"]:
            if row.get("raw_report_sha256"):
                versions.add(json.dumps(load(row["raw_report"])["source_sha256"], sort_keys=True))
    if len(set(configurations)) > 1 or len(versions) > 1:
        raise ValueError("Unexpected baseline/controller change across lateral conditions")
    return {"sweep": str(Path(path).resolve()), "sweep_sha256": sha256(path),
            "status": sweep["status"], "conditions": summaries,
            "interpretation": "One Unreal session per condition; two correlated pairs per session. Exploratory only."}


def write(result, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 单机单目标位置变化先导", "", f"运行状态：{result['status']}。", "",
             "| 目标条件 | 策略 | 已开始 | 安全收尾 | 达到观察目标 | 任务未达标 | 未评价 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for summary in result["conditions"]:
        for policy, counts in summary["policies"].items():
            lines.append(f"| {summary['condition_id']} | {policy} | {counts['started']} | "
                f"{counts['safe_pass']} | {counts['task_success']} | "
                f"{counts['assessed_task_failure']} | {counts['unassessed']} |")
    lines.extend(["", "只改变目标横向位置：中心、Y=-0.6 m、Y=+0.6 m。"
                  "起点、固定航点、工作区、速度预算和评分保持一致。", "",
                  "每个条件只有一次独立 Unreal 会话，包含两对相关试次；条件与会话混杂，"
                  "属于开发先导，不报告总体优势或显著性。主指标是全部已开始试次的任务成功；"
                  "安全 PASS 与任务成功分别报告。", "", "## 逐条件原始汇总", ""])
    for summary in result["conditions"]:
        child = Path(summary["manifest"]).parent / "analysis/summary.md"
        lines.append(f"- {summary['condition_id']}：[{child.name}]({child.as_posix()})")
    lines.extend(["", "AI 使用说明：分析与解释由 AI 辅助完成；数值来自原始飞行报告。", ""])
    (folder / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_sweep(args.sweep)
    write(result, args.output)
    print(json.dumps({"status": result["status"], "conditions": len(result["conditions"])}))
