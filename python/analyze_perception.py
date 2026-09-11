"""Summarize raw detections and temporal tracking states from one mission report."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

from setup_scene import ROOT


def summarize(records):
    groups = {}
    for frame in records:
        phase = frame.get("phase", "UNKNOWN")
        for vehicle, observation in frame.get("vehicles", {}).items():
            key = (phase, vehicle)
            item = groups.setdefault(key, {"phase": phase, "vehicle": vehicle, "frames": 0,
                                           "raw_detections": 0, "tracking_states": Counter(),
                                           "guidance_advice": Counter(),
                                           "empty_images": 0, "depth_values_m": []})
            item["frames"] += 1
            item["raw_detections"] += int(bool(observation.get("detected")))
            item["tracking_states"][observation.get("tracking_state", "UNAVAILABLE")] += 1
            item["guidance_advice"][observation.get("guidance_advice", "UNAVAILABLE")] += 1
            item["empty_images"] += int(observation.get("error") == "empty_image")
            depth = observation.get("depth_m")
            if isinstance(depth, (int, float)) and depth > 0:
                item["depth_values_m"].append(float(depth))
    rows = []
    for item in groups.values():
        depths = item.pop("depth_values_m")
        rows.append({**item, "tracking_states": dict(item["tracking_states"]),
                     "guidance_advice": dict(item["guidance_advice"]),
                     "depth_samples": len(depths),
                     "depth_first_m": depths[0] if depths else None,
                     "depth_last_m": depths[-1] if depths else None,
                     "depth_delta_m": depths[-1] - depths[0] if depths else None,
                     "depth_min_m": min(depths) if depths else None,
                     "depth_median_m": median(depths) if depths else None,
                     "depth_max_m": max(depths) if depths else None})
    return sorted(rows, key=lambda row: (row["phase"], row["vehicle"]))


def render_markdown(report, rows):
    lines = ["# 感知时序统计", "",
             f"任务状态：**{report.get('status', 'UNKNOWN')}**；场景：`{report.get('scenario', 'UNKNOWN')}`。", "",
             "| 阶段 | 飞机 | 帧数 | 原始检测 | DETECTED | TEMPORARILY_LOST | LOST | 深度样本 | 深度中位数(m) | 首末变化(m) | 空图像 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        states = row["tracking_states"]
        lines.append(f"| {row['phase']} | {row['vehicle']} | {row['frames']} | {row['raw_detections']} | "
                     f"{states.get('DETECTED', 0)} | {states.get('TEMPORARILY_LOST', 0)} | "
                     f"{states.get('LOST', 0)} | {row['depth_samples']} | "
                     f"{row['depth_median_m'] if row['depth_median_m'] is not None else '-'} | "
                     f"{row['depth_delta_m'] if row['depth_delta_m'] is not None else '-'} | "
                     f"{row['empty_images']} |")
    lines += ["", "状态只用于感知诊断；航点仍来自仿真坐标，命中仍由新碰撞事件判定。",
              "阶段内出现检测不自动等于真实阳性，需要结合保存的标注图和目标可见性人工复核。", ""]
    return "\n".join(lines)


def latest_report():
    reports = list((ROOT / "logs").glob("mission_*/report.json"))
    if not reports:
        raise FileNotFoundError("No mission report found")
    return max(reports, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", type=Path, help="Mission report.json; default is latest")
    args = parser.parse_args()
    source = args.report or latest_report()
    report = json.loads(source.read_text(encoding="utf-8"))
    rows = summarize(report.get("vision_records", []))
    result = {"source_report": str(source.resolve()), "policy": report.get("perception_policy"),
              "transitions": report.get("perception_transitions", []), "by_phase_vehicle": rows}
    json_path = source.parent / "perception_summary.json"
    md_path = source.parent / "perception_summary.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report, rows), encoding="utf-8")
    print(f"Perception summary: {json_path}")
    print(f"Readable report: {md_path}")


if __name__ == "__main__":
    main()
