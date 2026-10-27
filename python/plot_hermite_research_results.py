"""Publication-style plots from frozen reference-study results, without flight claims."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/hermite_disturbance_20261027"
OUTPUT = ROOT / "docs/research-hermite-20261027/figures"
METHODS = ["random", "gaussian", "abom", "abom_perturb", "abom_triggered"]
LABELS = ["随机", "高斯", "ABOM", "ABOM+扰动", "ABOM+触发"]
COLORS = ["#527a84", "#a47b3d", "#715c91", "#b66153", "#477c66"]


def load(name):
    if name == "ellipsoid_nominal":
        name = "ellipsoid_nominal_final"
    folder = BASE / name
    return json.loads((folder / "validation_summary.json").read_text(encoding="utf-8"))["rows"]


def pairs(left, right, method, field):
    a = {r["seed"]: r[field] for r in left if r["method"] == method}
    b = {r["seed"]: r[field] for r in right if r["method"] == method}
    if a.keys() != b.keys():
        raise ValueError("Unpaired experiment seeds")
    return np.array([[a[s], b[s]] for s in sorted(a)])


def finish(fig, name):
    fig.savefig(OUTPUT / f"{name}.png", dpi=180)
    fig.savefig(OUTPUT / f"{name}.pdf")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    nominal, margin = load("uncertainty_nominal"), load("uncertainty_margin")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), layout="constrained")
    x = np.arange(len(METHODS))
    for shift, rows, label, color in ((-.18, nominal, "名义约束", "#a2b4bb"),
                                      (.18, margin, "几何误差裕度", "#267c80")):
        counts = [sum(r["feasible"] for r in rows if r["method"] == method) for method in METHODS]
        bars = axes[0].bar(x + shift, counts, width=.34, label=label, color=color)
        axes[0].bar_label(bars, padding=2, fontsize=8)
    axes[0].set(xticks=x, xticklabels=LABELS, ylim=(0, 23), ylabel="64条件全通过的参考数 / 20个种子",
                title="误差模型改变参考的压力检验表现")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].legend(frameon=False, fontsize=9)
    for method, label, color in zip(METHODS, LABELS, COLORS):
        p = pairs(nominal, margin, method, "reference_duration_s")
        axes[1].scatter(p[:, 0], p[:, 1], s=21, alpha=.75, color=color, label=label)
    axes[1].plot([3, 5.5], [3, 5.5], "--", color=".65", lw=1)
    axes[1].set(xlabel="名义约束的参考时长 (s)", ylabel="加入裕度后的参考时长 (s)",
                title="同种子比较保留参考时长代价", xlim=(3, 5.3), ylim=(3, 5.3))
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("端状态保持曲线：球形几何不确定性先导", fontsize=14)
    fig.supxlabel("每法20个优化种子；相同128次候选评价；每条冻结参考检验64条件；不含车辆/风模型", fontsize=8)
    finish(fig, "sphere_uncertainty")
    exported = []
    for profile, rows in (("nominal", nominal), ("margin", margin)):
        exported.extend(dict(profile=profile, **r) for r in rows if r["method"] != "default")

    directional_paths = [BASE / p / "validation_summary.json" for p in
                         ("ellipsoid_nominal_final", "ellipsoid_isotropic", "ellipsoid_directional")]
    if all(p.exists() for p in directional_paths):
        baseline, isotropic, directional = [load(p.parent.name) for p in directional_paths]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), layout="constrained")
        for shift, rows, label, color in ((-.25, baseline, "名义", "#b9c1c6"),
                                          (0, isotropic, "统一球裕度", "#7c7191"),
                                          (.25, directional, "方向相关裕度", "#267c80")):
            counts = [sum(r["feasible"] for r in rows if r["method"] == method) for method in METHODS]
            bars = axes[0].bar(x + shift, counts, width=.23, label=label, color=color)
            axes[0].bar_label(bars, padding=2, fontsize=7)
        axes[0].set(xticks=x, xticklabels=LABELS, ylim=(0, 24), ylabel="128条件全通过的参考数 / 20个种子",
                    title="相同椭球误差包络下的约束消融")
        axes[0].tick_params(axis="x", labelsize=8)
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        for i, (method, color) in enumerate(zip(METHODS, COLORS)):
            p = pairs(isotropic, directional, method, "reference_duration_s")
            delta = p[:, 1] - p[:, 0]
            jitter = np.linspace(-.08, .08, len(delta))
            axes[1].scatter(i + jitter, delta, s=16, alpha=.6, color=color)
            axes[1].plot([i - .15, i + .15], [delta.mean()] * 2, color=color, lw=2.5)
        axes[1].axhline(0, color=".6", lw=1)
        axes[1].set(xticks=x, xticklabels=LABELS, ylabel="方向相关 − 统一球裕度的参考时长 (s)",
                    title="负值表示较低的参考时长代价")
        axes[1].tick_params(axis="x", labelsize=8)
        fig.suptitle("各向异性几何误差：降低统一膨胀的保守性", fontsize=14)
        fig.supxlabel("相同端状态/控制变量/评价预算；点为配对优化种子，短线为均值；不含车辆跟踪与机体外形", fontsize=8)
        finish(fig, "directional_uncertainty")
        for profile, rows in (("ellipsoid_nominal", baseline), ("ellipsoid_isotropic", isotropic),
                               ("ellipsoid_directional", directional)):
            exported.extend(dict(profile=profile, **r) for r in rows if r["method"] != "default")
    with (OUTPUT / "reference_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    print(OUTPUT)


if __name__ == "__main__":
    main()
