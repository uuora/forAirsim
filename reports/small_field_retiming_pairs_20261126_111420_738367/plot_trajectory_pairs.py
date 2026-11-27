"""Draw the same-condition AirSim pairs from the offline replay data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
import numpy as np


BATCH = Path(__file__).resolve().parent
PAPER = "#f2f5f5"
INK = "#183238"
MUTED = "#465f65"
GRID = "#cedbdc"
BASELINE = "#315f95"
ADJUSTED = "#aa4b16"


def plot_pair(ax, pair):
    ax.set_facecolor(PAPER)
    ax.set_proj_type("ortho")
    ax.view_init(elev=33, azim=-127)
    ax.set_box_aspect((8, 8, 2.5), zoom=1.12)
    ax.set(xlim=(0, 8), ylim=(0, 8), zlim=(0, 2.5),
           xticks=(0, 2, 4, 6, 8), yticks=(0, 2, 4, 6, 8), zticks=(0, 1, 2, 2.5))
    ax.set_zticklabels(("0", "1", "2", "2.5"))
    for axis, name in zip((ax.xaxis, ax.yaxis, ax.zaxis), "XYZ"):
        axis.set_label_text(name, fontsize=9, color=MUTED)
        axis.labelpad = -7
        axis.set_tick_params(labelsize=7, colors=MUTED, pad=-3)
        axis.line.set_color(GRID)
        axis.pane.set_facecolor(PAPER)
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"].update(color=GRID, linewidth=.58)

    retest = pair.get("retry")
    if retest:
        assert retest["same_geometry"] and retest["same_tracking_conditions"]
    auto = retest["toppquad"] if retest else pair["toppquad"]
    ordinary = pair["ordinary"]
    assert pair["same_geometry"] and pair["same_tracking_conditions"]
    for trial, color, linestyle, width in (
        (ordinary, BASELINE, "-", 2.55), (auto, ADJUSTED, (0, (3.2, 2.0)), 1.85),
    ):
        assert trial["passed"] and trial["samples"]
        xyz = np.asarray(trial["samples"])[:, 1:4]
        ax.plot(*xyz.T, color=color, linestyle=linestyle, lw=width, zorder=4)

    points = np.asarray(pair["points"])
    ax.scatter(*points[1:-1].T, s=24, color="white", edgecolor=MUTED, lw=.85, depthshade=False, zorder=5)
    ax.scatter(*points[0], s=33, marker="s", color="white", edgecolor=INK, lw=.9, depthshade=False, zorder=5)
    ax.scatter(*points[-1], s=31, marker="D", color=INK, edgecolor="white", lw=.65, depthshade=False, zorder=5)
    if retest:
        ax.text2D(.92, .88, "自动组复测", transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)


def main():
    html = (BATCH / "index.html").read_text(encoding="utf-8")
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    pairs = json.loads(payload)["pairs"]
    grid = {(pair["route"], pair["geometry"]): pair for pair in pairs}
    assert len(grid) == 9

    for font in ("msyh.ttc", "msyhbd.ttc"):
        fontManager.addfont(Path("C:/Windows/Fonts") / font)
    plt.rcParams.update({"font.family": "Microsoft YaHei"})
    fig = plt.figure(figsize=(13.2, 10.2), dpi=165, facecolor=PAPER)
    fig.text(.044, .960, "AirSim 三维实测轨迹对比", color=INK, fontsize=26, weight="bold", va="top")
    legend = [
        Line2D([], [], color=BASELINE, lw=2.55, label="基线 · 普通计时"),
        Line2D([], [], color=ADJUSTED, lw=1.85, ls=(0, (3.2, 2.0)), label="自动调整 TOPPQuad"),
        Line2D([], [], color=INK, ls="None", marker="s", markerfacecolor="white", label="跟踪起点"),
        Line2D([], [], color=MUTED, ls="None", marker="o", markerfacecolor="white", label="中间航点"),
        Line2D([], [], color=INK, ls="None", marker="D", label="终点"),
    ]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.040, .910), ncol=5,
               frameon=False, labelcolor=MUTED, fontsize=10.5)
    fig.text(.957, .887, "坐标单位 m", ha="right", color=MUTED, fontsize=10)

    routes = ("s", "u-turn", "arch")
    geometries = ("hermite", "bezier", "bspline")
    layout = fig.add_gridspec(3, 3, left=.095, right=.964, top=.819, bottom=.048, wspace=.07, hspace=.06)
    for col, geometry in enumerate(geometries):
        box = layout[0, col].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, .843, grid["s", geometry]["geometry_name"],
                 color=INK, fontsize=14, weight="bold", ha="center")
    for row, route in enumerate(routes):
        box = layout[row, 0].get_position(fig)
        fig.text(.062, (box.y0 + box.y1) / 2, grid[route, "hermite"]["route_name"],
                 ha="center", va="center", color=INK, fontsize=13, weight="bold")
        for col, geometry in enumerate(geometries):
            ax = fig.add_subplot(layout[row, col], projection="3d", computed_zorder=False)
            plot_pair(ax, grid[route, geometry])
    fig.savefig(BATCH / "trajectory_pairs_3d.png", facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
