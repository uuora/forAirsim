"""Plot the recorded AirSim positions and tracking times in the three-route batch."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
import numpy as np

from report import BATCH, HERMITE_BASELINE, METHODS, ROUTES, read


colors = {
    HERMITE_BASELINE: "#2d659f",
    "hermite_toppquad": "#684fa4",
    "bezier_ordinary": "#ad6426",
    "bspline_ordinary": "#087967",
}
paper, ink, muted, grid = "#f3f6fa", "#17263b", "#4e657f", "#cbdbea"
font = Path("C:/Windows/Fonts")
for path in (font / "msyh.ttc", font / "msyhbd.ttc"):
    fontManager.addfont(path)
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False})


def setup_scene(ax):
    ax.set_facecolor(paper)
    ax.set_proj_type("ortho")
    ax.view_init(elev=31, azim=-124)
    ax.set_box_aspect((8, 8, 2.5), zoom=1.04)
    ax.set(xlim=(0, 8), ylim=(0, 8), zlim=(0, 2.5), xticks=[0, 2, 4, 6, 8],
           yticks=[0, 2, 4, 6, 8], zticks=[0, 1, 2, 2.5])
    for axis, title in zip((ax.xaxis, ax.yaxis, ax.zaxis), ("X (m)", "Y (m)", "Z (m)")):
        axis.set_label_text(title, fontsize=9, color=muted)
        axis.labelpad = -4
        axis.set_tick_params(labelsize=8, colors=muted, pad=-2, length=0)
        axis.line.set_color(grid)
        axis.pane.set_facecolor(paper)
        axis.pane.set_edgecolor(grid)
        axis._axinfo["grid"].update(color=grid, linewidth=.6)


def render():
    design = read(BATCH / "design.json")
    with (BATCH / "flights.csv").open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    measured = [row for row in rows if row["reference_started"] == "True"]
    longest = max(float(row["tracking_time_s"]) for row in measured)
    fig = plt.figure(figsize=(13.4, 7.6), dpi=155, facecolor=paper)
    scenes = fig.add_gridspec(1, 3, left=.045, right=.955, top=.763, bottom=.397, wspace=.105)
    times = fig.add_gridspec(1, 3, left=.08, right=.94, top=.330, bottom=.115, wspace=.22)
    fig.text(.063, .948, "三条航线的 AirSim 实测对照", fontsize=24, fontweight="bold", color=ink, va="top")
    fig.text(.064, .888, "实测轨迹与跟踪耗时（含终点停稳）。竖虚线标出普通 Hermite 的用时。",
             fontsize=11, color=muted, va="top")
    handles = [Line2D([], [], color=color, lw=2.6, label=" · ".join(METHODS[key]))
               for key, color in colors.items()]
    handles.append(Line2D([], [], marker="o", linestyle="None", color=muted, markerfacecolor="white",
                          markersize=5, label="航点"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.056, .853), frameon=False, ncol=5,
               labelcolor=muted, fontsize=9.5)
    for scene_cell, time_cell, (route, name) in zip(scenes, times, ROUTES.items()):
        ax = fig.add_subplot(scene_cell, projection="3d", computed_zorder=False)
        setup_scene(ax)
        ax.set_title(name, x=.04, y=.99, loc="left", color=ink, fontsize=14, fontweight="bold")
        p = np.array(design["routes"][route]["waypoints"])
        ax.scatter(*p[:-1].T, marker="o", s=22, c="white", edgecolors=muted, lw=1., depthshade=False, zorder=5)
        ax.scatter(*p[-1], marker="D", s=29, c=ink, edgecolors="white", lw=.8, depthshade=False, zorder=5)
        selected = {row["method"]: row for row in measured if row["route"] == route}
        clock = fig.add_subplot(time_cell)
        clock.set_facecolor(paper)
        clock.set_xlim(0, longest * 1.23)
        clock.set_yticks([])
        clock.set_ylim(-.55, len(colors) - .45)
        clock.invert_yaxis()
        clock.set_axisbelow(True)
        clock.grid(axis="x", color=grid, linewidth=.7)
        clock.tick_params(axis="x", labelsize=9, colors=muted, length=0, pad=5)
        clock.set_xlabel("跟踪时间 (s)", color=muted, fontsize=9)
        for border in clock.spines.values():
            border.set_visible(False)
        baseline = selected.get(HERMITE_BASELINE)
        if baseline and baseline["passed"] == "True":
            clock.axvline(float(baseline["tracking_time_s"]), color=colors[HERMITE_BASELINE],
                          linestyle="--", linewidth=1.15, alpha=.72)
        for y, (key, color) in enumerate(colors.items()):
            row = selected.get(key)
            if row is None:
                clock.text(0, y, "未取得跟踪样本", va="center", color=muted, fontsize=9)
                continue
            passed = row["passed"] == "True"
            run = Path(row["run"])
            trace = read(run / "trace.json")["states"]
            xyz = np.asarray([state["position_zup_m"] for state in trace])
            ax.plot(*xyz.T, color=color, lw=1.75, alpha=.86, ls="-" if passed else "--", zorder=4)
            elapsed = float(row["tracking_time_s"])
            clock.barh(y, elapsed, height=.51, color=color, alpha=.8 if passed else .32)
            label = f"{elapsed:.2f}" if passed else f"{elapsed:.2f} 未通过"
            clock.text(elapsed + longest * .018, y, label, va="center", color=ink, fontsize=9)
    raw = [row for row in rows if row["method"] == "hermite_toppquad-full" and row["reference_started"] == "False"]
    if raw:
        fig.text(.064, .038, f"Hermite + 完整 TOPPQuad 有 {len(raw)} 个组合在参考预检阶段结束。",
                 fontsize=9, color=muted)
    fig.savefig(BATCH / "comparison.png", facecolor=paper)


if __name__ == "__main__":
    render()
