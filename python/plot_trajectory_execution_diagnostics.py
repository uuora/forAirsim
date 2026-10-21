"""Plot recorded self-state samples from trajectory interface pilot manifests.

This CLI reads reports without importing AirSim or contacting a simulator. It
never plots planner predictions or commands as measured motion. Host and AirSim
query clocks have separate figures; records without either clock use an index.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PHASES = {
    "settle": ("#666666", "o"),
    "pre_capture": ("#0072B2", "<"),
    "provider_query": ("#56B4E9", "s"),
    "planning_input": ("#009E73", "^"),
    "command_start": ("#CC79A7", "D"),
    "execute": ("#D55E00", "o"),
    "execute_end": ("#E69F00", ">"),
}
METRICS = (
    ("world_z_ned_m", "Sampled body z\n(world NED m; down +)"),
    ("acceleration_norm_m_s2", "Actual sampled acceleration\nnorm (m/s²)"),
    ("speed_norm_m_s", "Actual sampled velocity\nnorm (m/s)"),
)


def read_json(path, expected_sha256=None):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"SHA256 mismatch: {path}")
    return json.loads(raw.decode("utf-8-sig")), digest, str(path)


def resolve_path(value, base):
    path = Path(value)
    return path if path.is_absolute() else Path(base) / path


def number(value):
    if isinstance(value, bool) or value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite actual sample value")
    return result


def vector(value):
    if value is None:
        return [None] * 3
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a recorded 3D vector, got {value!r}")
    result = [number(v) for v in value]
    if any(v is None for v in result):
        raise ValueError("Incomplete recorded vector")
    return result


def norm(values):
    return math.sqrt(sum(v * v for v in values)) if all(v is not None for v in values) else None


def extract_samples(report):
    samples = []

    def add(phase, record, locator, note="", ignore_host=False):
        p = vector(record.get("world_ned_m"))
        v = vector(record.get("velocity_ned_m_s"))
        a = vector(record.get("acceleration_ned_m_s2"))
        if all(value is None for value in p + v + a):
            return
        timestamp = record.get("query_state_timestamp_ns")
        if timestamp is not None:
            if isinstance(timestamp, bool) or int(timestamp) != timestamp or timestamp <= 0:
                raise ValueError(f"Invalid simulator query timestamp at {locator}")
            timestamp = int(timestamp)
        sample = {
            "sample_id": len(samples), "phase": phase, "source_locator": locator,
            "query_state_timestamp_ns": timestamp,
            "host_monotonic_s": None if ignore_host else number(record.get("host_monotonic_s")),
            "recorded_host_monotonic_s": number(record.get("host_monotonic_s")),
            "world_x_ned_m": p[0], "world_y_ned_m": p[1], "world_z_ned_m": p[2],
            "velocity_x_ned_m_s": v[0], "velocity_y_ned_m_s": v[1], "velocity_z_ned_m_s": v[2],
            "acceleration_x_ned_m_s2": a[0], "acceleration_y_ned_m_s2": a[1], "acceleration_z_ned_m_s2": a[2],
            "speed_norm_m_s": norm(v), "acceleration_norm_m_s2": norm(a), "note": note,
        }
        samples.append(sample)

    for i, settling in enumerate(report.get("settling_records", [])):
        for j, record in enumerate(settling.get("samples", [])):
            add("settle", record, f"/settling_records/{i}/samples/{j}")
    for i, observation in enumerate(report.get("observations", [])):
        for key, phase in (("pre_capture_body_state", "pre_capture"),
                           ("provider_query_motion_state", "provider_query"),
                           ("planning_input_body_state", "planning_input")):
            if key in observation:
                add(phase, observation[key], f"/observations/{i}/{key}")
        if "planning_input_body_state" not in observation:
            # Do not borrow the provider's earlier query timestamp or image time.
            add("planning_input", {
                "world_ned_m": observation.get("world_ned_m"),
                "velocity_ned_m_s": observation.get("measured_velocity_ned_m_s"),
                "acceleration_ned_m_s2": observation.get("measured_acceleration_ned_m_s2"),
            }, f"/observations/{i}", "Legacy observation: independent self-state query time not recorded; missing motion remains blank")
    for i, planning in enumerate(report.get("planning_records", [])):
        if "command_start_actual_body_state" in planning:
            add("command_start", planning["command_start_actual_body_state"],
                f"/planning_records/{i}/command_start_actual_body_state")
        else:
            add("command_start", {
                "world_ned_m": planning.get("command_start_world_ned_m"),
                "velocity_ned_m_s": planning.get("command_start_actual_velocity_ned_m_s"),
            }, f"/planning_records/{i}", "Legacy command start: acceleration and independent query time not recorded")
        for j, record in enumerate(planning.get("execution_samples", [])):
            add("execute", record, f"/planning_records/{i}/execution_samples/{j}")
        if "executed_body_state" in planning:
            add("execute_end", planning["executed_body_state"], f"/planning_records/{i}/executed_body_state",
                "Final query overwrites state fields; inherited last-subsegment host time not used", ignore_host=True)
    if not samples:
        raise ValueError("Report has no supported actual self-state samples")
    host_values = [s["host_monotonic_s"] for s in samples if s["host_monotonic_s"] is not None]
    query_values = [s["query_state_timestamp_ns"] for s in samples if s["query_state_timestamp_ns"] is not None]
    host_origin = min(host_values) if host_values else None
    query_origin = min(query_values) if query_values else None
    for sample in samples:
        host, query = sample["host_monotonic_s"], sample["query_state_timestamp_ns"]
        sample["host_elapsed_s"] = host - host_origin if host is not None else None
        # Subtract integer nanoseconds before conversion to avoid epoch precision loss.
        sample["query_elapsed_s"] = (query - query_origin) / 1e9 if query is not None else None
        sample["time_source"] = "host_and_query" if host is not None and query is not None else "host" if host is not None else "query" if query is not None else "sample_index_only"
        sample["fallback_sample_index"] = sample["sample_id"] if host is None and query is None else None
    return samples, {"host_monotonic_origin_s": host_origin, "query_timestamp_origin_ns": query_origin}


def load_manifests(paths):
    provenance, flights, seen = [], [], set()
    for requested in paths:
        manifest, manifest_hash, manifest_path = read_json(requested)
        provenance.append({"kind": "manifest", "path": manifest_path, "sha256": manifest_hash})
        match = re.search(r"_v(\d+)(?:_|\.)", manifest.get("experiment_config", ""), re.I)
        version = f"V{match.group(1)}" if match else Path(manifest_path).parent.name
        for session in manifest.get("sessions", []):
            batch_path = resolve_path(session["batch_path"], Path(manifest_path).parent) / "batch_report.json"
            batch, batch_hash, batch_path = read_json(batch_path, session.get("batch_report_sha256"))
            provenance.append({"kind": "batch_report", "path": batch_path, "sha256": batch_hash})
            for run in batch.get("runs", []):
                report_path = resolve_path(run["report_path"], Path(batch_path).parent)
                report, digest, report_path = read_json(report_path, run.get("report_sha256"))
                if report_path in seen:
                    raise ValueError(f"Duplicate report supplied: {report_path}")
                seen.add(report_path)
                samples, origins = extract_samples(report)
                provenance.append({"kind": "flight_report", "path": report_path, "sha256": digest})
                flights.append({"label": f"{version} | {report.get('policy', run.get('policy', 'unknown'))}",
                    "report_path": report_path, "report_sha256": digest, "manifest_path": manifest_path,
                    "episode_id": report.get("episode_id", Path(report_path).parent.name),
                    "session_index": session.get("session_index"), "run": run.get("run"),
                    "samples": samples, "time_origins": origins,
                    "configuration": report.get("configuration", {}).get("planning", {}),
                    "protective_return_reason": report.get("protective_return_reason"),
                    "approach_stop_reason": report.get("approach_stop_reason"),
                    "task_success": report.get("task_success"),
                    "status": report.get("status"),
                    "planning_step_count": len(report.get("planning_records", [])),
                    "command_completion_count": report.get("metrics", {}).get("command_completion_count"),
                })
    if not flights:
        raise ValueError("No flight reports found in the supplied manifests")
    return flights, provenance


def write_csv(path, records):
    if not records:
        raise ValueError(f"No CSV records for {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def plot_view(flights, view, xkey, xlabel, output_dir):
    fig, axes = plt.subplots(3, len(flights), figsize=(4.0 * len(flights), 9.6), squeeze=False)
    points = []
    for column, flight in enumerate(flights):
        samples = [s for s in flight["samples"] if s[xkey] is not None]
        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, column]
            count = 0
            for phase, (color, marker) in PHASES.items():
                selected = sorted((s for s in samples if s["phase"] == phase and s[metric] is not None), key=lambda s: s[xkey])
                if not selected:
                    continue
                ax.scatter([s[xkey] for s in selected], [s[metric] for s in selected],
                    color=color, marker=marker, s=27, alpha=0.85)
                count += len(selected)
                for sample in selected:
                    points.append({"figure": f"sampled_{view}.png", "flight_label": flight["label"],
                        "episode_id": flight["episode_id"], "report_path": flight["report_path"],
                        "report_sha256": flight["report_sha256"], "sample_id": sample["sample_id"],
                        "phase": phase, "source_locator": sample["source_locator"], "metric": metric,
                        "x_source": xkey, "x_value": sample[xkey], "actual_value": sample[metric]})
            cfg = flight["configuration"]
            limit = cfg.get("fixed_world_height_ned_m" if row == 0 else "max_acceleration_m_s2" if row == 1 else "execution_speed_cap_m_s")
            if limit is not None:
                ax.axhline(float(limit), color="#444444", linestyle="--", linewidth=0.9)
            if row != 0:
                ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.text(0.02, 0.96, f"n = {count} actual points", transform=ax.transAxes, fontsize=8, va="top")
            if not count:
                ax.text(0.5, 0.5, "No recorded values in this clock view", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, wrap=True)
            if row == 0:
                ax.set_title(flight["label"], fontsize=11)
        reason = flight["protective_return_reason"]
        if reason:
            message = "Protection exit:\n" + textwrap.fill(reason, width=38)
            if reason == "Measured initial acceleration exceeds planner budget" and any(
                    s["phase"] == "planning_input" and s["acceleration_norm_m_s2"] is None for s in flight["samples"]):
                message += "\nRejected acceleration vector not recorded"
            axes[0, column].text(0.02, 0.03, message,
                transform=axes[0, column].transAxes, fontsize=7.5, color="#A33A26", va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    legend = [Line2D([], [], color=color, marker=marker, linestyle="None", label=phase)
              for phase, (color, marker) in PHASES.items()]
    legend.append(Line2D([], [], color="#444444", linestyle="--", label="Setpoint / configured budget"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, fontsize=8)
    fig.suptitle(f"Trajectory interface pilot: actual sampled self-state ({view.replace('_', ' ')})", fontsize=14)
    fig.text(0.5, 0.025, "Separate clocks; independent origins per flight. Scatter only: no interpolation, continuous peak claim, or method-effect conclusion.\n"
        "No planner/reference values. Missing actual values remain absent. World NED z is relative to simulation world origin, not terrain altitude.",
        ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.85, wspace=0.3, hspace=0.34)
    fig.savefig(output_dir / f"sampled_{view}.png", dpi=180)
    fig.savefig(output_dir / f"sampled_{view}.pdf")
    plt.close(fig)
    return points


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path,
                        help="Interface pilot manifest; repeat to include multiple batches")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing directories are refused")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"Output directory already exists; refusing overwrite: {args.output_dir.resolve()}")
    try:
        flights, sources = load_manifests(args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    for flight in flights:
        for sample in flight["samples"]:
            records.append({"flight_label": flight["label"], "episode_id": flight["episode_id"],
                "report_path": flight["report_path"], "report_sha256": flight["report_sha256"], **sample})
    write_csv(output_dir / "actual_samples.csv", records)
    points = []
    for view, xkey, xlabel in (("host_time", "host_elapsed_s", "Host monotonic elapsed (s)"),
                              ("query_time", "query_elapsed_s", "AirSim query elapsed (s)"),
                              ("untimed_index", "fallback_sample_index", "Record index (time unavailable)")):
        points.extend(plot_view(flights, view, xkey, xlabel, output_dir))
    write_csv(output_dir / "plot_points.csv", points)
    receipt = {"kind": "ACTUAL_SAMPLED_INTERFACE_DIAGNOSTICS", "formal_thesis_effect_results": False,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "sources": sources,
        "sample_count": len(records), "plotted_point_count": len(points),
        "clock_policy": "Separate host/query figures; no cross-clock conversion; untimed records indexed in extraction order",
        "norm_policy": "Euclidean norm recomputed only from recorded actual vectors; missing vectors stay missing",
        "scope": "Initial settle, sensing/input, command start, execution and execution end; excludes takeoff, return and landing",
        "flights": [{key: value for key, value in flight.items() if key != "samples"} | {
            "sample_count": len(flight["samples"]), "phase_counts": dict(Counter(s["phase"] for s in flight["samples"])),
            "actual_metric_counts": {metric: sum(s[metric] is not None for s in flight["samples"]) for metric, _ in METRICS},
            "sampled_acceleration_max_m_s2": max((s["acceleration_norm_m_s2"] for s in flight["samples"] if s["acceleration_norm_m_s2"] is not None), default=None),
        } for flight in flights]}
    (output_dir / "provenance.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    summary = ["# 轨迹接口短飞行诊断图", "", "仅用于检查实际自状态采样与接口执行，不能作为感知轨迹优化的效果结论。", "",
        "- 三组图分别使用主机单调时间、AirSim 查询时钟和无时间戳的记录索引；不同钟域不换算、不混接。每次飞行独立取时间零点。",
        "- 保留不同阶段记录；查询时间可能相同，点数包含各阶段记录，不代表独立统计样本数。散点未插值，采样最大值不能证明连续过程峰值。",
        "- 只从记录的实际三维速度、加速度计算范数；不使用规划参考、下发速度或命令变化率替代实际运动。",
        "- world NED z 采用仿真世界原点、向下为正；它不是距地高度。初始起飞、返回、降落不在本图范围。",
        "- V2 保护退出处未保留拒绝加速度向量；只显示保护原因，不补加速度点。V2 部分输入和命令起点缺独立时间戳，以记录索引呈现。",
        "- executed_body_state 的最终查询覆写了状态字段；继承的末次子段主机时间未用于绘图，原字段仍保留在 CSV 中。",
        "- actual_samples.csv 保存各采样及来源 JSON 路径；plot_points.csv 一行对应一个图点，同一采样可出现在两个独立钟域图中。provenance.json 保存原始报告、清单、批报告 SHA256 并核对上游哈希。", "",
        "| 飞行 | 实际记录数 | z / a范数 / v范数点数 | 规划 / 完成命令 | 结束原因 |", "|---|---:|---|---|---|"]
    for flight in receipt["flights"]:
        counts = flight["actual_metric_counts"]
        reason = flight["protective_return_reason"] or flight["approach_stop_reason"] or "未记录"
        summary.append(f"| {flight['label'].replace('|', '/')} | {flight['sample_count']} | {counts['world_z_ned_m']} / {counts['acceleration_norm_m_s2']} / {counts['speed_norm_m_s']} | {flight['planning_step_count']} / {flight['command_completion_count']} | {reason} |")
    summary.extend(["", f"总实际记录 {len(records)} 条，图点 {len(points)} 个。所选飞行的 task_success 均按原始报告保留，PASS 不等同任务成功。", "", "AI 辅助：此诊断脚本与图由 AI 辅助生成；原始飞行报告只读，未改写。", ""])
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "flights": len(flights), "actual_sample_count": len(records),
                      "plot_point_count": len(points)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
