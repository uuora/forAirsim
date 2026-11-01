"""Math/geometry study of equivalent bases and boundary-preserving deformations.

Plots and JSON contain reference geometry, never measurements from a vehicle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from trajectory_research.curves import (
    bezier_segment, c2_bump, cubic_hermite_from_state, junction_residuals,
    quintic_hermite,
)
from trajectory_research.constraints import (
    derivative_norm_bound, quintic_bump_jerk_energy_increment,
    sphere_clearance_certificate,
)

ROOT = Path(__file__).resolve().parents[1]


def study(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261027)
    u = np.linspace(0., 1., 301)
    error = {"cubic": np.zeros(4), "quintic": np.zeros(4), "bump_bezier": np.zeros(4)}
    endpoint_errors = np.zeros(3)
    for _ in range(100):
        p0, p1 = rng.uniform(-5, 5, (2, 3))
        v0, v1 = rng.uniform(-2, 2, (2, 3))
        a0, a1 = rng.uniform(-.5, .5, (2, 3))
        duration = float(rng.uniform(.7, 6))
        cubic = cubic_hermite_from_state(p0, v0, p1, v1, duration)
        quintic = quintic_hermite(p0, v0, a0, p1, v1, a1, duration)
        for label, segment in (("cubic", cubic), ("quintic", quintic)):
            other = bezier_segment(segment.bezier_control_points(), duration_s=duration)
            for k in range(4):
                error[label][k] = max(error[label][k], float(np.max(np.abs(
                    segment.evaluate(u, k, wrt="time") - other.evaluate(u, k, wrt="time")))))
        d = rng.uniform(-.3, .3, 3)
        changed = c2_bump(quintic, d)
        control = quintic.bezier_control_points(6)
        control[3] += 3.2 * d
        alternative = bezier_segment(control, duration_s=duration)
        for k in range(4):
            error["bump_bezier"][k] = max(error["bump_bezier"][k], float(np.max(np.abs(
                changed.evaluate(u, k, wrt="time") - alternative.evaluate(u, k, wrt="time")))))
        for k in range(3):
            endpoint_errors[k] = max(endpoint_errors[k], float(np.max(np.abs(
                changed.evaluate([0, 1], k, wrt="time") - quintic.evaluate([0, 1], k, wrt="time")))))

    base = quintic_hermite([0, 0, 5], [2, 0, 0], [0, 0, 0],
                           [8, 0, 5], [2, 0, 0], [0, 0, 0], 4.)
    deformation = np.array([0., .8, 0.])
    changed = c2_bump(base, deformation)
    other = quintic_hermite([8, 0, 5], [2, 0, 0], [0, 0, 0],
                            [12, 2, 5], [1, 1, 0], [0, 0, 0], 3.)
    geometry = dict(baseline=sphere_clearance_certificate(base, [4, 0, 5], .4),
                    bumped_coarse=sphere_clearance_certificate(changed, [4, 0, 5], .4, max_depth=0),
                    bumped_refined=sphere_clearance_certificate(changed, [4, 0, 5], .4),
                    junction_baseline=junction_residuals(base, other),
                    junction_bumped=junction_residuals(changed, other),
                    derivative_hull_bounds={str(k): derivative_norm_bound(changed, k) for k in (1, 2, 3)},
                    jerk_energy_increase=quintic_bump_jerk_energy_increment(base, deformation))
    report = dict(scope="100 randomized algebraic cases and one continuous-reference geometry example; no flight",
                  random_seed=20261027, cases=100, samples_per_curve=301,
                  maximum_absolute_time_derivative_error={name: values.tolist() for name, values in error.items()},
                  derivative_order_units=["m", "m/s", "m/s^2", "m/s^3"],
                  maximum_endpoint_preservation_errors=endpoint_errors.tolist(), example=geometry,
                  maximum_example_displacement_m=float(np.linalg.norm(deformation)),
                  limitations="Finite-precision sufficient geometry checks; no vehicle tracking or wind model")
    (output / "geometry_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), layout="constrained")
    a = axes[0, 0]
    points, moved = base.evaluate(u), changed.evaluate(u)
    a.plot(points[:, 0], points[:, 1], color="#687477", lw=2, label="五次基线")
    a.plot(moved[:, 0], moved[:, 1], color="#007d80", lw=2, label="C2边界保持形变")
    a.add_patch(Circle((4, 0), .4, color="#bd4c42", alpha=.65))
    a.scatter([0, 8], [0, 0], color="#183445", s=25, zorder=3)
    a.set(xlabel="x (m)", ylabel="y (m)", title="参考几何：改变中段形状，保持端状态", ylim=(-1.3, 1.8))
    a.set_aspect("equal", adjustable="box")
    a.legend(frameon=False, loc="upper left", fontsize=9)
    a.text(.02, .03, "二维投影；高度固定5 m；未运行车辆", transform=a.transAxes, fontsize=8)
    a = axes[0, 1]
    for k, label, color in ((0, "位置偏移 (m)", "#007d80"),
                             (1, "速度偏移 (m/s)", "#c18123"),
                             (2, "加速度偏移 (m/s²)", "#78599d")):
        delta = changed.evaluate(u, k, wrt="time") - base.evaluate(u, k, wrt="time")
        a.plot(u * 4, delta[:, 1], label=label, color=color)
    a.set(xlabel="t (s)", ylabel="各物理量的y分量", title="两端位置、速度、加速度偏移均为零")
    a.axhline(0, color=".75", lw=.6)
    a.legend(frameon=False, fontsize=8)
    a = axes[1, 0]
    delta_j = changed.evaluate(u, 3, wrt="time") - base.evaluate(u, 3, wrt="time")
    a.plot(u * 4, delta_j[:, 1], color="#b24d48", lw=2)
    a.scatter([0, 4], delta_j[[0, -1], 1], color="#b24d48", s=25)
    a.set(xlabel="t (s)", ylabel="jerk变化 (m/s³)", title="C2保持仍会改变端点jerk")
    a.axhline(0, color=".75", lw=.6)
    a.text(.03, .06, f'本例jerk能量增加 {geometry["jerk_energy_increase"]:.4f} m²/s⁵', transform=a.transAxes, fontsize=9)
    a = axes[1, 1]
    durations = np.linspace(1., 7., 201)
    displacement = .25
    a.plot(durations, (192 / (25 * np.sqrt(5))) * displacement / durations, label="速度增量界 (m/s)", color="#c18123")
    a.plot(durations, 24 * displacement / durations ** 2, label="加速度增量界 (m/s²)", color="#78599d")
    a.plot(durations, 384 * displacement / durations ** 3, label="jerk增量界 (m/s³)", color="#b24d48")
    a.set(xlabel="段时长T (s)", ylabel="固定0.25 m形变的解析上界", title="搜索幅度需要随时间与剩余预算调整", yscale="log")
    a.legend(frameon=False, fontsize=8)
    fig.savefig(output / "curve_structure.png")
    fig.savefig(output / "curve_structure.pdf")
    plt.close(fig)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hermite_disturbance_20261027/geometry")
    study(parser.parse_args().output.resolve())
