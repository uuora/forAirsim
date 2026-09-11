"""Aggregate completed batch-validation reports without rerunning the simulator."""
import csv
import json
import statistics
from collections import defaultdict

from setup_scene import ROOT


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        target_offset = tuple(row.get("target_offset_ned_m") or (0.0, 0.0, 0.0))
        grouped[(row["scenario"], row.get("miss_offset_y_m"), target_offset)].append(row)
    output = []
    for scenario, miss_offset, target_offset in sorted(
            grouped, key=lambda key: (key[0], float("-inf") if key[1] is None else key[1], key[2])):
        items = grouped[(scenario, miss_offset, target_offset)]
        separations = [float(x["min_separation_m"]) for x in items if x.get("min_separation_m") is not None]
        drifts = [float(x["max_waiting_drift_m"]) for x in items if x.get("max_waiting_drift_m") is not None]
        delays = [float(x["miss_to_dispatch_s"]) for x in items if x.get("miss_to_dispatch_s") is not None]
        passed = sum(x["status"] == "PASS" for x in items)
        output.append({
            "scenario": scenario,
            "miss_offset_y_m": miss_offset,
            "target_offset_ned_m": list(target_offset),
            "runs": len(items),
            "passed": passed,
            "failed": len(items) - passed,
            "pass_rate": passed / len(items),
            "minimum_separation_m": min(separations) if separations else None,
            "maximum_waiting_drift_m": max(drifts) if drifts else None,
            "mean_miss_to_dispatch_s": statistics.fmean(delays) if delays else None,
            "stdev_miss_to_dispatch_s": statistics.stdev(delays) if len(delays) >= 2 else None,
        })
    return output


def render_markdown(result):
    lines = ["# AirSim 批量验证汇总", "", f"唯一任务运行数：{result['unique_runs']}", "",
             "| 场景 | 漏击偏差 Y (m) | 目标偏移 NED (m) | 通过/总数 | 通过率 | 最小间距 (m) | 最大待命漂移 (m) | 平均漏击到派发 (s) | 样本标准差 (s) |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in result["by_scenario"]:
        show = lambda value, digits=4: "—" if value is None else f"{value:.{digits}f}"
        offset = "未记录/不适用" if row["miss_offset_y_m"] is None else show(row["miss_offset_y_m"], 1)
        target = ", ".join(show(value, 1) for value in row["target_offset_ned_m"])
        lines.append("| {scenario} | {offset} | [{target}] | {passed}/{runs} | {rate} | {sep} | {drift} | {delay} | {stdev} |".format(
            scenario=row["scenario"], offset=offset, target=target, passed=row["passed"], runs=row["runs"],
            rate=show(row["pass_rate"] * 100, 1) + "%", sep=show(row["minimum_separation_m"]),
            drift=show(row["maximum_waiting_drift_m"]), delay=show(row["mean_miss_to_dispatch_s"]),
            stdev=show(row["stdev_miss_to_dispatch_s"])))
    lines.extend(["", "> 这些结果只覆盖已记录的固定 Blocks 场景，不能直接代表动态目标或新环境的成功率。", ""])
    return "\n".join(lines)


def main():
    reports = []
    rows = []
    seen_runs = set()
    for path in sorted((ROOT / "logs").glob("batch_*/batch_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") not in ("PASS", "FAIL"):
            continue
        reports.append(str(path.relative_to(ROOT)))
        for row in report.get("runs", []):
            if row["run"] not in seen_runs:
                item = dict(row)
                item["miss_offset_y_m"] = report.get("miss_offset_y_m")
                item["target_offset_ned_m"] = report.get("target_offset_ned_m")
                rows.append(item)
                seen_runs.add(row["run"])
    summary = summarize(rows)
    result = {"batch_reports": reports, "unique_runs": len(rows), "by_scenario": summary}
    output = ROOT / "logs" / "batch_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "logs" / "batch_summary.md").write_text(render_markdown(result), encoding="utf-8")
    if summary:
        with (ROOT / "logs" / "batch_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
