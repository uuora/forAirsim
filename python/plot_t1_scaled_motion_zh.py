"""Chinese, source-bound figures for predeclared 50/100 mm development runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

from plot_t1_identity_zh import choose_font, layout_audit, digest, save_json

LABELS = {"zero": "零输出门控", "scale50": "50毫米尺度", "scale100": "100毫米尺度"}


def plot(analysis, output):
    analysis, output = Path(analysis).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("NEW_SCALED_FIGURE_FOLDER_REQUIRED")
    raw = analysis.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if data["kind"] != "T1_SCALED_MOTION_DEVELOPMENT_ANALYSIS" or data["formal_efficacy_results"]:
        raise ValueError("SCALED_DEVELOPMENT_ANALYSIS_REQUIRED")
    font, _ = choose_font({ord(c) for c in "任务一毫米尺度中文"}, None)
    font_manager.fontManager.addfont(str(font))
    prop = font_manager.FontProperties(fname=str(font))
    plt.rcParams.update({"font.family": prop.get_name(), "font.size": 10,
                         "axes.unicode_minus": True, "pdf.fonttype": 42})
    figures, mappings = [], []
    plottable=[r for r in data["runs"] if r["endpoint"]["net_xy_displacement_m"] is not None]
    unplotted=[dict(session_id=r["session_id"],reason="没有核实终态端点；保留在总分析和Qt回放")
               for r in data["runs"] if r["endpoint"]["net_xy_displacement_m"] is None]
    for index, run in enumerate(plottable, 1):
        base = run["base_evidence"]
        epoch = base["measured_epoch_monotonic_s"]
        condition = run["scaled_motion_condition"]
        label = LABELS[condition["name"]] + ("独立重复" if "_rep" in run["session_id"] else "")
        passed = run["independent_scale_qualification"].startswith("PASS")
        fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.5))
        fig.suptitle(f"任务一：{label}（受控开发，{'通过' if passed else '未通过'}）", fontsize=15, y=.975)
        frames = [row for row in base["frame_rows"] if row["actual_phase"] != "initial_acquisition"]
        identity_points = [dict(frame_id=row["frame_id"], time_s=row["readback_s"]-epoch,
                                identity=row["identity"]["status"]) for row in frames]
        ax = axes[0, 0]
        for identity, height, text, color in [("CORRECT", 1, "正确目标", "#387c6f"),
                                              ("NO_OUTPUT", 0, "无目标输出", "#8a959e")]:
            points = [p for p in identity_points if p["identity"] == identity]
            if points:
                ax.scatter([p["time_s"] for p in points], [height]*len(points), s=10, c=color, label=text)
        other = [p for p in identity_points if p["identity"] not in ("CORRECT", "NO_OUTPUT")]
        if other:
            ax.scatter([p["time_s"] for p in other], [-1]*len(other), s=20, c="#b04f43", marker="x", label="其他输出")
        ax.set(title="保存主帧的独立身份评分", xlabel="相对主阶段开始（秒）", ylabel="身份输出状态",
               yticks=[-1, 0, 1], yticklabels=["其他", "无输出", "正确"], ylim=(-1.4, 1.6))
        ax.legend(fontsize=9, loc="lower right")
        origin = run["endpoint"]["start_state"]["position_world_ned_m"]
        state_points = []
        for row in run["sampled_state_rows"]:
            dx, dy = [(row["position_world_ned_m"][i]-origin[i])*1000 for i in (0, 1)]
            state_points.append(dict(time_s=row["host_receipt_monotonic_s"]-epoch,
                x_mm=dx, y_mm=dy, net_mm=(dx*dx+dy*dy)**.5, state_timestamp_ns=row["state_timestamp_ns"]))
        ax = axes[0, 1]
        ax.plot([p["x_mm"] for p in state_points], [p["y_mm"] for p in state_points], c="#397ba6", linewidth=1.5)
        ax.scatter([0], [0], s=24, c="#387c6f", label="实测起点")
        if state_points:
            ax.scatter([state_points[-1]["x_mm"]], [state_points[-1]["y_mm"]], marker="x", s=38, c="#ac6847", label="实测终点")
        ax.set(title="自身状态测得的固定高度XY轨迹", xlabel="相对起点X（毫米）", ylabel="相对起点Y（毫米）")
        ax.set_aspect("equal", adjustable="datalim"); ax.legend(fontsize=9)
        ax = axes[1, 0]
        ax.plot([p["time_s"] for p in state_points], [p["net_mm"] for p in state_points], c="#397ba6", linewidth=1.5)
        threshold = 5 if condition["name"] == "zero" else condition["minimum_endpoint_mm"]
        threshold_label = "5毫米零漂移上限" if condition["name"] == "zero" else f"预声明{threshold:g}毫米判据"
        ax.axhline(threshold, ls="--", c="#ac7844", label=threshold_label)
        ax.set(title=f"净XY位移：{run['endpoint']['net_xy_displacement_m']*1000:.3f}毫米",
               xlabel="相对主阶段开始（秒）", ylabel="相对起点距离（毫米）")
        ax.legend(fontsize=9)
        ax = axes[1, 1]
        for key, text, color in [("frame_processing", "帧处理完成", "#387c6f"),
                                  ("historical_full", "完整完成（含停稳）", "#ad6846")]:
            points = run["frame_timing_rows"]
            ax.plot([p["slot_index"] for p in points],
                    [p["recorded_timing"][key]["after_due_s"] if p["recorded_timing"][key]["after_due_s"] is not None else float("nan") for p in points],
                    c=color, lw=1, label=text)
        ax.axhline(.4, ls="--", c="#777777", label="原0.4秒完整完成截止点")
        ax.set(title="保留原完成时钟", xlabel="主测量帧序号", ylabel="从本帧计划起点计时（秒）")
        ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid(alpha=.16)
        fig.text(.035, .02, "160/320秒为不同预声明尺度条件；速度、租约及采样停稳要求保持。结果用于执行尺度验证，正式四方法比较待完成。", fontsize=9, color="#59616b")
        fig.tight_layout(rect=(0, .055, 1, .95)); fig.canvas.draw()
        for ax in axes.flat:
            lo, hi = ax.get_xlim(); ax.set_xticks([v for v in ax.get_xticks() if lo <= v <= hi])
            if ax is not axes[0, 0]:
                lo, hi = ax.get_ylim(); ax.set_yticks([v for v in ax.get_yticks() if lo <= v <= hi])
        fig.canvas.draw()
        key = f"scaled_motion_{index}_zh"
        figures.append((key, fig))
        mappings.append(dict(figure=key, session_id=run["session_id"], condition=condition,
            identity_points=identity_points, own_state_points=state_points, timing_points=run["frame_timing_rows"],
            endpoint=run["endpoint"], qualification=run["independent_scale_qualification"], input_receipts=run["input_receipts"]))
    strings = [t.get_text() for _, fig in figures for t in fig.findobj(Text) if t.get_visible()]
    _, font_receipt = choose_font({ord(c) for s in strings for c in s if not c.isspace()}, font)
    layouts = [layout_audit(fig) for _, fig in figures]
    if not figures or any(row["text_bounds_status"] != "PASS" for row in layouts):
        raise ValueError("SCALED_CHINESE_FIGURE_LAYOUT_FAILED")
    output.mkdir(parents=True); files = []
    for key, fig in figures:
        for suffix in ("png", "pdf"):
            path = output/(key+"."+suffix); fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white"); files.append(path)
        plt.close(fig)
    path = output/"plot_data_mapping.json"
    save_json(path, dict(kind="T1_CHINESE_SCALED_MOTION_MAPPING", figures=mappings)); files.append(path)
    for name in ("plot_t1_scaled_motion_zh.py", "plot_t1_identity_zh.py"):
        path = output/name; shutil.copyfile(Path(__file__).with_name(name), path); files.append(path)
    if analysis.read_bytes() != raw:
        raise ValueError("SCALED_FIGURE_SOURCE_CHANGED")
    save_json(output/"artifact_receipt.json", dict(kind="T1_CHINESE_SCALED_MOTION_FIGURE_RECEIPT",
        created_utc=datetime.now(timezone.utc).isoformat(), input_analysis=dict(path=str(analysis), sha256=digest(analysis)),
        font=font_receipt, layout_checks=layouts, unplotted_incomplete_attempts=unplotted, visual_review=dict(status="PENDING"),
        artifacts=[dict(path=str(p), sha256=digest(p)) for p in files], new_flights=0, formal_efficacy_results=False))
    print("中文尺度图表", len(figures), "字体", font_receipt["glyph_coverage"], "布局通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); plot(args.analysis, args.output)
