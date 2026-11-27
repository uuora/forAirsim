"""Compare paired AirSim telemetry at equal elapsed times and at waypoint hits."""
import json
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import fontManager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.proj3d import proj_transform

from plot_trajectory_pairs import (
    ADJUSTED, BASELINE, BATCH, GRID, INK, MUTED, PAPER, plot_pair, plt,
)


GEOMETRIES = ("hermite", "bezier", "bspline")
ROUTES = ("s", "u-turn", "arch")
TIMES_S = (4, 8, 12)


def accepted_trials(pair):
    assert pair["same_geometry"] and pair["same_tracking_conditions"]
    retry = pair.get("retry")
    if retry:
        assert retry["same_geometry"] and retry["same_tracking_conditions"]
    baseline = pair["ordinary"]
    adjusted = retry["toppquad"] if retry else pair["toppquad"]
    assert baseline["passed"] and adjusted["passed"]
    return baseline, adjusted


def event_times(trial):
    waypoints = trial["waypoints"]
    assert all(waypoint["passed"] for waypoint in waypoints)
    times = np.array([*(waypoint["hit_time_s"] for waypoint in waypoints), trial["tracking_s"]])
    assert np.all(np.diff(times) > 0)
    return times


def positions_at_times(trial):
    states = np.asarray(trial["samples"])[:, :4]
    clock = states[:, 0]
    assert clock[0] <= TIMES_S[0] and TIMES_S[-1] <= clock[-1]
    assert np.all(np.diff(clock) > 0)
    return np.column_stack([np.interp(TIMES_S, clock, axis) for axis in states[:, 1:].T])


def plot_same_time(pair):
    baseline, adjusted = accepted_trials(pair)
    fig = plt.figure(figsize=(11.6, 6.4), dpi=170, facecolor=PAPER)
    fig.text(.054, .940, "同一时刻，无人机飞到了哪里？", fontsize=24, weight="bold", color=INK)
    fig.text(.055, .873, "U 型回转 · 七次 B 样条 · AirSim 实测", fontsize=13, color=MUTED)
    fig.legend(
        handles=[
            Line2D([], [], color=BASELINE, marker="s", lw=1.6, label="普通计时"),
            Line2D([], [], color=ADJUSTED, marker="o", lw=1.6, ls="--", label="自动调整 TOPPQuad"),
            Line2D([], [], color=MUTED, lw=1.3, ls=":", label="连接同一时刻的位置"),
        ],
        loc="upper left", bbox_to_anchor=(.044, .839), ncol=3, frameon=False,
        fontsize=11, labelcolor=INK, columnspacing=1.6,
    )

    ax = fig.add_axes([.039, .156, .53, .567], projection="3d", computed_zorder=False)
    plot_pair(ax, pair)
    ax.view_init(elev=44, azim=-99)
    ax.set_box_aspect((8, 8, 2.5), zoom=1.24)
    for line in ax.lines:
        line.set_alpha(.52)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_tick_params(labelsize=9, pad=-2)
        axis.label.set_fontsize(11)
    ax.xaxis.labelpad = 0
    first, second = positions_at_times(baseline), positions_at_times(adjusted)
    for begin, end in zip(first, second):
        ax.plot(*np.vstack([begin, end]).T, color=MUTED, lw=1.4, ls=":", zorder=7)
    for positions, color, marker in ((first, BASELINE, "s"), (second, ADJUSTED, "o")):
        ax.scatter(*positions.T, s=95, color=color, marker=marker, edgecolor="white",
                   linewidth=1.5, zorder=9, depthshade=False)
        offsets = ((0, -18), (0, -18), (13, -1)) if marker == "s" else ((-12, 9), (10, 9), (-10, 10))
        for position, elapsed, offset in zip(positions, TIMES_S, offsets):
            x, y, _ = proj_transform(*position, ax.get_proj())
            ax.annotate(f"{elapsed} s", (x, y), xytext=offset, textcoords="offset points",
                        ha="center", va="center", color=color, fontsize=11, weight="bold", zorder=10,
                        bbox=dict(facecolor=PAPER, alpha=.86, lw=0, pad=.48))
    fig.text(.067, .744, "三维位置（m）", fontsize=12, weight="bold", color=INK)

    fig.text(.627, .744, "航点命中与结束时刻", fontsize=12, weight="bold", color=INK)
    ax = fig.add_axes([.627, .245, .321, .459], facecolor=PAPER)
    first_times, second_times = event_times(baseline), event_times(adjusted)
    rows = np.arange(len(first_times))[::-1]
    ax.hlines(rows, second_times, first_times, color="#93a6a9", lw=2, zorder=2)
    ax.scatter(first_times, rows, marker="s", s=62, color=BASELINE, zorder=3)
    ax.scatter(second_times, rows, marker="o", s=62, color=ADJUSTED, zorder=3)
    for row, first_t, second_t in zip(rows, first_times, second_times):
        ax.text((first_t + second_t) / 2, row + .19, f"早 {first_t - second_t:.2f} s",
                ha="center", va="bottom", fontsize=11, color=INK)
    ax.set(xlim=(0, 23), ylim=(-.48, 4.59), xticks=np.arange(0, 23, 5), yticks=rows,
           yticklabels=[*(f"W{n}" for n in range(1, len(rows))), "结束"])
    ax.set_xlabel("从开始跟踪计时（s）", labelpad=9, fontsize=11, color=MUTED)
    ax.tick_params(axis="both", labelcolor=MUTED, labelsize=10, length=0, pad=7)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)

    fig.text(.054, .116, "三维时间点由相邻实测样本线性插值。W1–W4 为中间航点，结束时刻含末端停稳。", fontsize=10, color=MUTED)
    fig.savefig(BATCH / "trajectory_same_time_u_bspline.png", facecolor=PAPER)
    plt.close(fig)


def plot_event_leads(grid):
    leads = {}
    for key, pair in grid.items():
        baseline, adjusted = accepted_trials(pair)
        leads[key] = event_times(baseline) - event_times(adjusted)

    fig = plt.figure(figsize=(11.6, 6.3), dpi=170, facecolor=PAPER)
    fig.text(.054, .912, "自动调整 TOPPQuad 在各航点提前多久？", fontsize=23, weight="bold", color=INK)
    fig.text(.055, .844, "九组 AirSim 配对 · 同一事件的基线用时减去自动组用时（秒）",
             fontsize=12, color=MUTED)
    ax = fig.add_axes([.055, .176, .89, .58], facecolor=PAPER)
    ax.set(xlim=(-2.72, 5), ylim=(9, 0))
    ax.set_axis_off()
    for column, label in enumerate(("W1", "W2", "W3", "W4", "跟踪结束")):
        ax.text(column + .5, -.3, label, ha="center", va="center", color=INK,
                fontsize=12, weight="bold", clip_on=False)

    ramp = LinearSegmentedColormap.from_list("warm", ["#faeee2", "#dd9765", "#963d19"])
    normalize = Normalize(vmin=0, vmax=max(np.max(values) for values in leads.values()))
    for route_index, route in enumerate(ROUTES):
        ax.text(-2.64, route_index * 3 + 1.5, grid[route, "hermite"]["route_name"], fontsize=13,
                color=INK, va="center", weight="bold")
        for geometry_index, geometry in enumerate(GEOMETRIES):
            row = route_index * 3 + geometry_index
            pair = grid[route, geometry]
            name = pair["geometry_name"] + ("*" if pair.get("retry") else "")
            ax.text(-1.42, row + .5, name, fontsize=11, color=INK, va="center")
            values = leads[route, geometry]
            cells = {**dict(enumerate(values[:-1])), 4: values[-1]}
            for column in range(5):
                value = cells.get(column)
                color = "#e8efef" if value is None else ramp(normalize(value))
                ax.add_patch(Rectangle((column + .033, row + .07), .934, .86, lw=0, facecolor=color))
                label = "—" if value is None else f"{value:+.2f}"
                ax.text(column + .5, row + .5, label, ha="center", va="center", fontsize=12,
                        color="white" if value is not None and normalize(value) >= .71 else INK,
                        weight="bold" if column == 4 else "normal")
        ax.hlines(route_index * 3 + 3, -2.64, 4.96, linewidth=.8, color=GRID)
    fig.text(.055, .103, "颜色越深，提前越多。三维拱形有三个中间航点。跟踪结束含末端停稳。", fontsize=10, color=MUTED)
    fig.text(.055, .061, "* 三维拱形 Hermite 自动组使用同条件通过复测，首测触发采样间隔保护。", fontsize=10, color=MUTED)
    fig.savefig(BATCH / "trajectory_waypoint_time_leads.png", facecolor=PAPER)
    plt.close(fig)


def main():
    html = (BATCH / "index.html").read_text(encoding="utf-8")
    payload = html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0]
    pairs = json.loads(payload)["pairs"]
    grid = {(pair["route"], pair["geometry"]): pair for pair in pairs}
    assert len(grid) == len(ROUTES) * len(GEOMETRIES)
    for font in ("msyh.ttc", "msyhbd.ttc"):
        fontManager.addfont(Path("C:/Windows/Fonts") / font)
    plt.rcParams.update({"font.family": "Microsoft YaHei"})

    plot_same_time(grid["u-turn", "bspline"])
    plot_event_leads(grid)


if __name__ == "__main__":
    main()

