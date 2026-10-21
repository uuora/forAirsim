"""Build a conservative, reproducible inventory of phase-one experiment evidence.

This is an offline audit: it never connects to AirSim or calls the VLX API.
The independent unit is a completed flight pair or capture session, not a frame.
"""

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(path):
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def audit_blocks(logs=LOGS):
    batches = []
    complete_pairs = []
    for path in sorted(logs.glob("single_batch_*/batch_report.json")):
        report = read_json(path)
        runs = report.get("runs", [])
        active = report.get("active_run")
        unfinished_started = int(bool(active) and not any(
            row["run"] == active["run"] for row in runs))
        groups = {}
        for row in runs:
            groups.setdefault(row["pair"], []).append(row)
        pair_rows = []
        for pair_id, group in sorted(groups.items()):
            by_policy = {row["policy"]: row for row in group}
            complete = (len(group) == 2 and set(by_policy) == {"coordinate", "visual"}
                        and all(row["status"] == "PASS" and
                                isinstance(row.get("task_success"), bool)
                                for row in group))
            item = {"pair": pair_id, "complete": complete,
                    "batch": path.parent.name,
                    "session": report.get("session"),
                    "statuses": {row["policy"]: row["status"] for row in group}}
            if complete:
                coordinate, visual = by_policy["coordinate"], by_policy["visual"]
                item["task_success"] = {key: bool(by_policy[key]["task_success"])
                                        for key in ("coordinate", "visual")}
                item["visual_minus_coordinate"] = {
                    key: visual[key] - coordinate[key]
                    for key in ("approach_duration_s", "approach_path_m")
                    if isinstance(visual.get(key), (int, float)) and
                    isinstance(coordinate.get(key), (int, float))
                }
                complete_pairs.append(item)
            pair_rows.append(item)
        batches.append({"source": source(path), "status": report["status"],
                        "session": report.get("session"),
                        "pairs": pair_rows,
                        "run_count": len(runs),
                        "started_run_count": len(runs) + unfinished_started,
                        "unfinished_started_count": unfinished_started,
                        "failed_run_count": sum(row["status"] != "PASS" for row in runs)})
    return {"batches": batches, "completed_pair_count": len(complete_pairs),
            "failed_batch_count": sum(batch["status"] == "FAIL" for batch in batches),
            "nonterminal_batch_count": sum(batch["status"] not in ("PASS", "FAIL") for batch in batches),
            "completed_pairs": complete_pairs,
            "unit": "completed coordinate/visual flight pair; group by recorded Unreal session and controller version"}


def audit_maritime(logs=LOGS):
    path = logs / "maritime_single_20260925_205856_936729" / "report.json"
    report = read_json(path)
    conditions = []
    for condition in report["conditions"]:
        frames = condition["frames"]
        conditions.append({"profile": condition["profile"], "frame_count": len(frames),
                           "associations": dict(Counter(frame["association"] for frame in frames))})
    return {"source": source(path), "status": report["status"],
            "session_count": 1, "frame_count": sum(item["frame_count"] for item in conditions),
            "conditions": conditions,
            "unit": "one hover capture session; frames within it are correlated"}


def audit_layout(logs=LOGS):
    folder = logs / "maritime_layout_collection_20260926"
    manifest_path = folder / "collection_manifest.json"
    queue_path = folder / "review_queue.csv"
    manifest = read_json(manifest_path)
    with queue_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {"manifest": source(manifest_path), "review_queue": source(queue_path),
            "session_count": manifest["session_count"], "frame_count": len(rows),
            "review_status": dict(Counter(row["review_status"] for row in rows)),
            "human_verified_count": sum(row["review_status"] == "VERIFIED" for row in rows),
            "unit": "Unreal process/capture session; no random frame split"}


def audit_vlx(logs=LOGS):
    path = logs / "vlx_api_full_test.json"
    report = read_json(path)
    frames = report["frames"]
    by_profile = {}
    for profile in sorted({frame["profile"] for frame in frames}):
        subset = [frame for frame in frames if frame["profile"] == profile]
        by_profile[profile] = {"frames": len(subset),
                               "target_associations": sum(
                                   frame.get("vlx_selection_evaluation", {}).get("association")
                                   == "TARGET" for frame in subset)}
    return {"source": source(path), "status": report["status"],
            "frame_count": len(frames), "by_profile": by_profile,
            "request_mode": report.get("vlx_request_mode", "NOT_RECORDED"),
            "independent_trial_count": 1,
            "interpretation": "Saved cloud replay; candidate-box transport is not attested in this report"}


def render_markdown(bundle):
    blocks = bundle["blocks"]
    maritime = bundle["maritime"]
    layout = bundle["layout"]
    vlx = bundle["vlx"]
    lines = [
        "# 毕业论文第一阶段：现有证据盘点",
        "",
        f"生成时间（UTC）：{bundle['generated_utc']}。由 `python/build_phase1_evidence.py` 离线重建；未运行仿真或 API。",
        "",
        "## 单机接近控制（Blocks）",
        "",
        f"完整配对试次：**{blocks['completed_pair_count']}**；失败批次：**{blocks['failed_batch_count']}**。"
        f"未写入终态的批次：**{blocks['nonterminal_batch_count']}**（以会话清单确认运行或中断）。"
        "PASS 表示安全收尾，任务成功单独计数。失败批次保留在台账中，不并入效果比较。",
        "",
    ]
    for batch in blocks["batches"]:
        lines.append(f"- `{batch['source']['path']}`：{batch['status']}；"
                     f"{batch['started_run_count']} 次已开始，{batch['failed_run_count']} 次运行失败，"
                     f"{batch['unfinished_started_count']} 次尚无登记终态。")
    if blocks["completed_pairs"]:
        lines.extend(["", "完整配对的视觉组减坐标组差值：", ""])
        for pair in blocks["completed_pairs"]:
            diff = pair["visual_minus_coordinate"]
            lines.append(f"- `{pair['batch']}` 第 {pair['pair']} 对：用时 {diff.get('approach_duration_s', float('nan')):+.3f} 秒；"
                         f"路径 {diff.get('approach_path_m', float('nan')):+.3f} 米；"
                         f"任务成功：坐标={pair['task_success']['coordinate']}、"
                         f"视觉={pair['task_success']['visual']}。")
    lines.extend(["", "上表是全部存档的完整记录；历史开发版本与新先导批次分别保留，不合并估计方法效果。"
                  "新批次的会话汇总见对应 `phase1_pilot_*/analysis/summary.md`。", "",
                  "## 海事定点观察（MaritimeHarborV2）", "",
                  f"`{maritime['source']['path']}`：{maritime['status']}；"
                  f"**{maritime['session_count']} 次会话、{maritime['frame_count']} 帧**。", "",
                  "| 条件 | 帧数 | TARGET | OTHER_INSTANCE | NO_CANDIDATE |",
                  "|---|---:|---:|---:|---:|"])
    for item in maritime["conditions"]:
        counts = item["associations"]
        lines.append(f"| {item['profile']} | {item['frame_count']} | {counts.get('TARGET', 0)} | "
                     f"{counts.get('OTHER_INSTANCE', 0)} | {counts.get('NO_CANDIDATE', 0)} |")
    lines.extend(["", "这 30 帧属于一次悬停会话，不是 30 次独立实验；AirSim 投影框仅供事后评价。", "",
                  "## 左右布局与人工复核", "",
                  f"`{layout['manifest']['path']}`：{layout['session_count']} 次采集会话、"
                  f"{layout['frame_count']} 帧；人工 VERIFIED 为 **{layout['human_verified_count']}**。"
                  f"状态分布：`{json.dumps(layout['review_status'], ensure_ascii=False)}`。", "",
                  "未完成独立人工可见框复核前，该组只作为探索性对照。", "",
                  "## 云端 VLX 回放", "",
                  f"`{vlx['source']['path']}`：{vlx['frame_count']} 帧；"
                  f"请求类型字段 `{vlx['request_mode']}`。当前报告未证明外部候选框已传给云端服务。",
                  "",
                  "| 条件 | 帧数 | 与指定目标关联 |", "|---|---:|---:|"])
    for name, item in vlx["by_profile"].items():
        lines.append(f"| {name} | {item['frames']} | {item['target_associations']} |")
    lines.extend(["", "这一批可描述为保存图像上的云端定位回放；模型版本、提示词、候选框传输与人工真值"
                  "必须分别核验，不能由 30/30 推断泛化性能或 YOLO→VLX 的效果。", "",
                  "## 进入下一轮实验的门槛", "",
                  "1. 冻结独立试次单位、主指标和失败处理规则；Blocks 增加反转顺序的完整配对试次。",
                  "2. 海事布局先完成人工可见框复核，再按采集会话分组评分。",
                  "3. 云端 API 明确记录请求类型和候选框是否实际传输；不混用图像＋文本与候选框条件。",
                  "4. 新数据保留失败、中断、源码/配置哈希和原始响应；不覆盖既有记录。", "",
                  "AI 使用说明：本盘点和解释由 AI 辅助生成；原始试次、标注来源与证据限制如上。", ""])
    return "\n".join(lines)


def build(logs=LOGS):
    return {"generated_utc": datetime.now(timezone.utc).isoformat(),
            "blocks": audit_blocks(logs), "maritime": audit_maritime(logs),
            "layout": audit_layout(logs), "vlx": audit_vlx(logs)}


def main():
    bundle = build()
    output = ROOT / "docs" / "第一阶段证据盘点.md"
    output.write_text(render_markdown(bundle), encoding="utf-8")
    machine = output.with_suffix(".json")
    machine.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    print(machine)


if __name__ == "__main__":
    main()
