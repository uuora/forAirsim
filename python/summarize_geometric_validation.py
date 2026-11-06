"""Read existing geometric commissioning attempts without connecting to AirSim.

The output directory must be new. This command never calls the flight runtime
or auditor, never repairs records, and never rewrites input files. Verdicts
are labelled as existing report/audit verdicts; figures are raw logged samples,
not new flights, independent sensor evidence, or a trajectory-method ranking.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


INPUT_FILES = ("config.json", "manifest.json", "report.json", "audit.json", "events.json",
               "cleanup.json", "metrics_samples.csv", "samples.jsonl")
FIELDS = (
    "condition_group", "condition_fingerprint", "run_id", "case", "time_scale_factor",
    "reference_duration_s", "outcome", "armed_event_recorded", "tracking_samples_recorded",
    "normal_completion", "control_constraints_passed", "data_audit_passed", "cleanup_landed",
    "cleanup_disarmed", "cleanup_api_released", "accepted", "reported_accepted",
    "actual_completion_time_s", "planning_wall_s", "position_rmse_m", "position_peak_m",
    "position_exceedance_s", "velocity_rmse_mps", "reference_attitude_peak_deg",
    "command_attitude_peak_deg", "reference_attitude_exceedance_s", "actual_tilt_peak_deg",
    "actual_rate_peak_radps", "actual_speed_peak_mps", "throttle_min", "throttle_max",
    "command_saturation_s", "max_state_interval_s", "metric_origin", "error", "audit_issues",
    "consistency_notes", "run_directory", "plot_png",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path, notes):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as ex:
        notes.append(f"{path.name}: {ex!r}")
        return {}


def discover(inputs):
    found = set()
    for value in inputs:
        path = Path(value).resolve(strict=True)
        candidates = [path] if path.is_file() else path.rglob("report.json")
        for report in candidates:
            directory = report.parent
            if report.name != "report.json" or not (directory / "manifest.json").is_file():
                continue
            try:
                manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8-sig"))
            except Exception:
                continue
            if str(manifest.get("protocol", "")).startswith("geometric_"):
                found.add(directory)
    return sorted(found, key=lambda path: (path.name, str(path)))


def nested(obj, *keys):
    for key in keys:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def degrees(value):
    return None if value is None else math.degrees(float(value))


def inspect(directory):
    notes = []
    config = read_json(directory / "config.json", notes)
    manifest = read_json(directory / "manifest.json", notes)
    report = read_json(directory / "report.json", notes)
    audit = read_json(directory / "audit.json", notes) if (directory / "audit.json").exists() else {}
    events = read_json(directory / "events.json", notes)
    if not isinstance(events, list):
        notes.append("events.json is not an event list")
        events = []
    cleanup = report.get("cleanup", {})
    metrics = audit.get("metrics") or report.get("metrics", {})
    origin = "audit.json" if audit.get("metrics") else "report.json"
    if audit and report.get("metrics") and audit.get("metrics") and report["metrics"] != audit["metrics"]:
        notes.append("Report metrics differ from existing audit metrics; audit values displayed")
    for field in ("normal_completion", "control_constraints_passed", "data_audit_passed"):
        if field in audit and field in report and audit[field] != report[field]:
            notes.append(f"Existing audit/report disagree on {field}")
    case = manifest.get("case", report.get("case", "UNKNOWN"))
    fingerprint = manifest.get("condition_fingerprint")
    row = {
        "condition_group": None, "condition_fingerprint": fingerprint,
        "run_id": directory.name, "case": case,
        "time_scale_factor": nested(config, "time_scaling", "factor") or 1.,
        "reference_duration_s": sum(nested(config, "cases", case, "segment_durations") or []) or None,
        "outcome": report.get("outcome", "UNKNOWN"),
        "armed_event_recorded": any(e.get("event") == "ARMED" for e in events if isinstance(e, dict)),
        "tracking_samples_recorded": metrics.get("samples", 0),
        "normal_completion": report.get("normal_completion"),
        "control_constraints_passed": audit.get("control_constraints_passed", report.get("control_constraints_passed")),
        "data_audit_passed": audit.get("data_audit_passed", report.get("data_audit_passed")),
        "cleanup_landed": cleanup.get("landed"), "cleanup_disarmed": cleanup.get("disarmed"),
        "cleanup_api_released": cleanup.get("api_control_released"),
        "reported_accepted": report.get("accepted"),
        "actual_completion_time_s": report.get("actual_completion_time_s"),
        "planning_wall_s": report.get("planning_wall_s"),
        "position_rmse_m": nested(metrics, "position_error_m", "rmse"),
        "position_peak_m": nested(metrics, "position_error_m", "peak"),
        "position_exceedance_s": nested(metrics, "position_error_m", "exceedance_s"),
        "velocity_rmse_mps": nested(metrics, "velocity_error_mps", "rmse"),
        "reference_attitude_peak_deg": degrees(nested(metrics, "attitude_reference_error_rad", "peak")),
        "command_attitude_peak_deg": degrees(nested(metrics, "attitude_command_error_rad", "peak")),
        "reference_attitude_exceedance_s": nested(metrics, "attitude_reference_error_rad", "exceedance_s"),
        "actual_tilt_peak_deg": degrees(nested(metrics, "actual_tilt_rad", "peak")),
        "actual_rate_peak_radps": nested(metrics, "actual_rate_radps", "peak"),
        "actual_speed_peak_mps": nested(metrics, "actual_speed_mps", "peak"),
        "throttle_min": nested(metrics, "throttle", "min"), "throttle_max": nested(metrics, "throttle", "max"),
        "command_saturation_s": metrics.get("command_saturation_s"),
        "max_state_interval_s": metrics.get("max_state_interval_s"), "metric_origin": origin,
        "error": report.get("error"), "audit_issues": " | ".join(map(str, audit.get("issues", []))),
        "consistency_notes": None, "run_directory": str(directory), "plot_png": None,
    }
    row["accepted"] = all(row[k] is True for k in (
        "normal_completion", "control_constraints_passed", "data_audit_passed", "cleanup_landed",
        "cleanup_disarmed", "cleanup_api_released"))
    if row["reported_accepted"] != row["accepted"]:
        notes.append("Existing accepted flag differs from conjunction including all cleanup flags")
    row["consistency_notes"] = " | ".join(notes)
    provenance = {"directory": str(directory), "files": {}, "input_read_issues": notes,
                  "source_snapshot_hashes_declared": manifest.get("sources", {})}
    for name in INPUT_FILES:
        path = directory / name
        provenance["files"][name] = {"exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None}
    return row, config, provenance


def _format(value, precision=4):
    if value is None:
        return "缺测"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    if isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def plot_attempt(row, config, output):
    """Plot the existing raw numerical export; no audit/runtime calls."""
    directory = Path(row["run_directory"])
    csvpath = directory / "metrics_samples.csv"
    if not csvpath.is_file():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with csvpath.open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    if not samples:
        return None
    t = np.asarray([float(s["t"]) for s in samples])
    fig = plt.figure(figsize=(12, 8.5), layout="constrained")
    ax_p = fig.add_subplot(221)
    ax_ref = fig.add_subplot(222)
    ax_cmd = fig.add_subplot(223)
    ax_xyz = fig.add_subplot(224, projection="3d")
    for ax, key, label, limit, factor in (
        (ax_p, "position_error_m", "Position error (m)", config["limits"]["position_error_m"], 1.),
        (ax_ref, "attitude_reference_error_rad", "SO(3) to trajectory attitude (deg)", math.degrees(config["limits"]["attitude_error_rad"]), 180/math.pi),
        (ax_cmd, "attitude_command_error_rad", "SO(3) to commanded attitude (deg)", math.degrees(config["limits"]["attitude_error_rad"]), 180/math.pi),
    ):
        y = np.asarray([float(s[key]) for s in samples]) * factor
        ax.plot(t, y, color="#365a9d", linewidth=1., marker=".", markersize=1.8, label="Raw logged samples")
        threshold_label = "Commissioning error limit" if ax is not ax_cmd else "Nominal-error limit (context only)"
        ax.axhline(limit, color="#b84138", linestyle="--", linewidth=.9, label=threshold_label)
        if not row["accepted"]:
            ax.plot(t[-1], y[-1], "x", color="#bd322b", markersize=8, label="Not-accepted attempt endpoint")
        ax.set(xlabel="Synchronized simulation time (s)", ylabel=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    positions, references = [], []
    parse_issues = []
    if (directory / "samples.jsonl").is_file():
        with (directory / "samples.jsonl").open(encoding="utf-8-sig") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    sample = json.loads(line)
                    if sample.get("phase") == "TRACKING":
                        positions.append(sample["position"])
                        references.append(sample["reference"]["p"])
                except Exception as ex:
                    parse_issues.append(f"samples.jsonl:{number}: {ex!r}")
    if positions:
        actual, desired = np.asarray(positions), np.asarray(references)
        ax_xyz.plot(desired[:, 0], desired[:, 1], -desired[:, 2], color="#39834d", linestyle="--", linewidth=1.2, label="Synchronized reference")
        ax_xyz.plot(actual[:, 0], actual[:, 1], -actual[:, 2], color="#ae3c9d", linewidth=1., label="Measured state")
        for point, label, marker in ((desired[0], "Start", "s"), (desired[-1], "Final recorded reference", "D")):
            ax_xyz.scatter([point[0]], [point[1]], [-point[2]], marker=marker, s=26, label=label)
        if not row["accepted"]:
            ax_xyz.scatter([actual[-1, 0]], [actual[-1, 1]], [-actual[-1, 2]], marker="x", color="#bd322b", s=55, label="Not-accepted attempt endpoint")
        joined = np.concatenate((actual, desired));joined[:, 2] *= -1
        center = (joined.max(axis=0) + joined.min(axis=0)) / 2
        radius = max(float(np.ptp(joined, axis=0).max()) / 2, .25)
        ax_xyz.set_xlim(center[0]-radius, center[0]+radius)
        ax_xyz.set_ylim(center[1]-radius, center[1]+radius)
        ax_xyz.set_zlim(center[2]-radius, center[2]+radius)
        ax_xyz.set_box_aspect([1, 1, 1])
        ax_xyz.set_proj_type("ortho")
        ax_xyz.legend(fontsize=6, loc="upper left")
    ax_xyz.set(xlabel="X north (m)", ylabel="Y east (m)", zlabel="Height (m)")
    status = "COMPLETE / ACCEPTED" if row["accepted"] else f"{row['outcome']} / NOT ACCEPTED"
    fig.suptitle(f"{row['case']} | time factor {row['time_scale_factor']:g} | {row['condition_group']} | {status}\n{row['run_id']}", fontsize=11)
    stem = output / "plots" / row["run_id"]
    stem.parent.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix("."+extension), dpi=150)
    plt.close(fig)
    if parse_issues:
        (stem.parent / (stem.name + "_issues.json")).write_text(json.dumps(parse_issues, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(stem.with_suffix(".png").relative_to(output)).replace("\\", "/")


def summarize(inputs, output, plots=False):
    directories = discover(inputs)
    if not directories:
        raise ValueError("No geometric report+manifest attempts found in the explicit inputs")
    output = Path(output).resolve()
    if any(output == p or p in output.parents for p in directories):
        raise ValueError("Output must not be inside an original run directory")
    output.mkdir(parents=True, exist_ok=False)
    rows, configs, sources = [], {}, []
    for directory in directories:
        row, config, provenance = inspect(directory)
        rows.append(row);configs[str(directory)] = config;sources.append(provenance)
    # Missing fingerprints remain distinct; they never acquire a common unknown group.
    groups = {}
    for row in rows:
        key = row["condition_fingerprint"] or "MISSING:"+row["run_directory"]
        if key not in groups:
            groups[key] = f"G{len(groups)+1:02d}"
        row["condition_group"] = groups[key]
        if plots:
            try:
                row["plot_png"] = plot_attempt(row, configs[row["run_directory"]], output)
            except Exception as ex:
                row["consistency_notes"] += f" | Plot not generated: {ex!r}"
    with (output / "attempts.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)
    blocks = [
        "# 几何混合控制调试记录汇总", "",
        "本页只读取指定目录中已有记录，包含失败及启动失败。没有连接AirSim、重新飞行或改写原始记录；也没有重新执行会写文件的审计器。状态取自已有report/audit，数值优先采用已有独立审计输出。", "",
        "控制结构：外部几何期望姿态/总推力计算，SimpleFlight保留姿态、角速度、混控及执行器环。它不是完整自实现SE(3)控制器。阈值仍是逐次记录冻结的调试限值，尚不能作为三种轨迹方法的最终比较标准。", "",
        "统计仅用于本轮开发覆盖，不作显著性、泛化、高速全任务适用性或算法最优性推断。按完整条件指纹分组，不跨条件合并排名；不同任务与速度步骤也分列。缺少指纹的启动失败各自单列。", "",
    ]
    total = len(rows)
    flights = [r for r in rows if r["armed_event_recorded"]]
    blocks += [f"共发现{total}次尝试；记录ARMED事件的飞行尝试{len(flights)}次，未记录ARMED的尝试{total-len(flights)}次。", "",
               f"全部尝试中：正常完成{sum(r['normal_completion'] is True for r in rows)}/{total}，控制约束通过{sum(r['control_constraints_passed'] is True for r in rows)}/{total}，数据审计通过{sum(r['data_audit_passed'] is True for r in rows)}/{total}，全部完成/约束/审计/清理条件通过{sum(r['accepted'] for r in rows)}/{total}。", ""]
    if flights:
        blocks += [f"仅在记录ARMED的飞行尝试中，全部条件通过{sum(r['accepted'] for r in flights)}/{len(flights)}。未记录ARMED不自动解释为零飞行风险或安全落地，须查对应错误及来源。", ""]
    blocks += ["数值时间从统一起点状态释放至终点状态持续稳定；规划墙钟单独列出。姿态分别报告相对轨迹加速度/偏航生成的名义姿态、相对含反馈修正的已命令姿态的SO(3)角误差；实际倾角/角速度是独立安全量。指令饱和不等于已观测电机饱和。缺测显示“缺测”，不填零。", ""]
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["condition_group"]].append(row)
    for group, attempts in by_group.items():
        blocks += [f"## {group}", "", f"条件指纹：`{attempts[0]['condition_fingerprint'] or '缺测；仅此尝试'}`", "",
                   "| 任务/时间倍率 | 正常完成 | 控制通过 | 审计通过 | 落地/解除武装/释放API | 实际时间s | 规划s | 位置RMSE/峰值m | 名义/命令姿态峰值° | 实际速度峰值m/s | 指令饱和s | 最大状态间隔s |", 
                   "|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|"]
        for r in attempts:
            blocks.append("| "+" | ".join((
                f"{r['case']} ×{r['time_scale_factor']:g}", _format(r['normal_completion']),
                _format(r['control_constraints_passed']), _format(r['data_audit_passed']),
                "/".join(_format(r[k]) for k in ("cleanup_landed", "cleanup_disarmed", "cleanup_api_released")),
                _format(r['actual_completion_time_s']), _format(r['planning_wall_s'], 6),
                _format(r['position_rmse_m'])+"/"+_format(r['position_peak_m']),
                _format(r['reference_attitude_peak_deg'])+"/"+_format(r['command_attitude_peak_deg']),
                _format(r['actual_speed_peak_mps']), _format(r['command_saturation_s']),
                _format(r['max_state_interval_s'], 6)))+" |")
        blocks += [""]
        for r in attempts:
            local_path = Path(r['run_directory']).as_posix()
            blocks += [f"- [{r['run_id']}]({local_path})：outcome={r['outcome']}；全部条件通过={r['accepted']}。"]
            if r["error"]:
                blocks.append(f"  错误：`{r['error']}`。")
            if r["audit_issues"] or r["consistency_notes"]:
                blocks.append("  证据说明："+"；".join(x for x in (r["audit_issues"], r["consistency_notes"]) if x))
            if r["plot_png"]:
                blocks.append(f"  [原始样本图]({r['plot_png']})（未平滑；三维轴等尺度；不是相机图像）。")
        blocks += [""]
    blocks += ["## 来源与范围", "",
               "完整数值和原始路径见attempts.csv；输入文件逐项SHA-256、缺失项及汇总器源码绑定见provenance.json。原始日志、失败、配置、源码快照均留在原处。", "",
               "本汇总由AI辅助生成；所有结论限定于所列日志的已有状态与数值，不把图形、界面图标或观察视口当作感知/姿态验证证据。无障碍物、无视觉输入，capture_overview保持关闭。"]
    freeze_path = Path(__file__).resolve().parents[1] / "configs/geometric_control_commissioning_freeze_v1.json"
    freeze_provenance = None
    if freeze_path.is_file():
        freeze_provenance = {"path": freeze_path.as_posix(), "sha256": sha256(freeze_path)}
        blocks += ["", f"后续复现的[工程调试门槛冻结回执]({freeze_path.as_posix()})另行保留；它不是未来Bézier/Hermite/B样条三方法研究的最终验收标准，也不回填改变本轮记录。"]
    (output / "SUMMARY.md").write_text("\n".join(blocks)+"\n", encoding="utf-8")
    provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(), "inputs": [str(Path(x).resolve()) for x in inputs],
        "read_only_inputs": True, "airsim_calls": 0, "auditor_invoked": False, "summary_script": str(Path(__file__).resolve()),
        "summary_script_sha256": sha256(__file__), "attempt_count": total, "armed_attempt_count": len(flights),
        "grouping": "Exact existing condition_fingerprint; each missing fingerprint is a distinct group; no ranking or pooled inference",
        "attempts": sources, "commissioning_freeze": freeze_provenance, "outputs": {}}
    for file in output.rglob("*"):
        if file.is_file():
            provenance["outputs"][str(file.relative_to(output)).replace("\\", "/")] = sha256(file)
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return {"output": str(output), "attempts": total, "armed_attempts": len(flights), "accepted": sum(r["accepted"] for r in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Run directories or parents of geometric run directories")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--plots", action="store_true", help="Export unsmoothed per-attempt PNG/SVG/PDF figures")
    args = parser.parse_args()
    print(json.dumps(summarize(args.inputs, args.output, plots=args.plots), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
