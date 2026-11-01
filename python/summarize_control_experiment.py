"""Summarize several AirSim curve comparison batches into experiment tables.

Reads only existing comparison.csv files produced by compare_airsim_curve_runs.py.
Never calls AirSim, never re-runs a plan and never merges different condition
fingerprints. Descriptive statistics only: medians, quartiles and paired
within-cell differences; no p-values, no confidence intervals, no causal claim.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

CELLS = {
    "bezier_ordinary": ("Bezier", "ordinary"),
    "hermite_ordinary": ("Hermite", "ordinary"),
    "bezier_abom": ("Bezier", "abom_basic"),
    "hermite_abom": ("Hermite", "abom_basic"),
    "bezier_abom_improved_candidate": ("Bezier", "abom_candidate"),
    "hermite_abom_improved_candidate": ("Hermite", "abom_candidate"),
}
PLANNERS = ["ordinary", "abom_basic", "abom_candidate"]
PLANNER_LABEL = {"ordinary": "普通（无搜索）", "abom_basic": "基础ABOM", "abom_candidate": "改进候选"}
BASES = ["Bezier", "Hermite"]

METRICS = [
    ("terminal_position_error_m", "终端位置误差/m"),
    ("max_position_error_m", "峰值位置误差/m"),
    ("position_rmse_m", "同步位置RMSE/m"),
    ("terminal_velocity_error_mps", "终端速度误差/(m/s)"),
    ("reference_duration_s", "参考时长T/s"),
    ("quality_term", "综合分"),
]
RESOURCE = [
    ("planning_search_plus_validation_wall_s", "规划墙钟/s"),
    ("full_surrogate_calls", "完整模型调用"),
    ("execution_host_wall_s", "执行墙钟/s"),
    ("speed_clipped_time_fraction", "速度指令限幅时间比例"),
]
CONTRASTS = [
    ("B0-H0", "bezier_ordinary", "hermite_ordinary", "同参考换基"),
    ("B1-H1", "bezier_abom", "hermite_abom", "基础ABOM同搜索空间换基"),
    ("B2-H2", "bezier_abom_improved_candidate", "hermite_abom_improved_candidate", "改进候选换基"),
    ("B1-B0", "bezier_abom", "bezier_ordinary", "基础ABOM相对普通"),
    ("B2-B1", "bezier_abom_improved_candidate", "bezier_abom", "改进候选相对基础ABOM"),
    ("H1-H0", "hermite_abom", "hermite_ordinary", "基础ABOM相对普通(Hermite)"),
    ("H2-H1", "hermite_abom_improved_candidate", "hermite_abom", "改进候选相对基础ABOM(Hermite)"),
]


def load_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def number(value):
    if value in (None, "", "None"):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if result == result else None


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def describe(values):
    clean = [value for value in values if value is not None]
    if not clean:
        return dict(n=0, median=None, q1=None, q3=None, mean=None)
    return dict(n=len(clean), median=percentile(clean, 0.5),
                q1=percentile(clean, 0.25), q3=percentile(clean, 0.75),
                mean=sum(clean) / len(clean))


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, int) or float(value).is_integer():
        return str(int(value))
    return f"{value:.{digits}g}" if abs(value) < 1e-3 or abs(value) >= 1e5 else f"{value:.{digits}f}".rstrip("0").rstrip(".")


def collect(batch_dirs):
    batches = []
    for directory in batch_dirs:
        comparison = directory / "comparison.csv"
        if not comparison.is_file():
            raise SystemExit(f"缺少 {comparison}")
        rows = load_rows(comparison)
        fingerprints = {row["condition_fingerprint"] for row in rows if row.get("condition_fingerprint")}
        if len(fingerprints) != 1:
            raise SystemExit(f"{directory.name} 内含多个条件指纹，拒绝合并：{sorted(fingerprints)}")
        batches.append(dict(label=directory.name, path=directory, rows=rows,
                            fingerprint=fingerprints.pop()))
    return batches


def usable(row):
    return truth(row.get("included_in_numeric_summary")) and row.get("status") != "NOT_RUN"


def cell_table(batches):
    lines = ["| 参数生成 | 曲线表达 | 预定槽位 | 完成 | 有效样本n | 门槛通过 | 执行失败 | 数据质量失败 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for planner in PLANNERS:
        for base in BASES:
            group = next(key for key, value in CELLS.items() if value == (base, planner))
            planned = completed = valid = passed = failed = quality = 0
            for batch in batches:
                for row in batch["rows"]:
                    if row["group_id"] != group:
                        continue
                    planned += 1
                    completed += truth(row.get("workflow_completed"))
                    if usable(row):
                        valid += 1
                    passed += truth(row.get("control_error_gates_passed"))
                    failed += row.get("status") == "COMPLETED_CONTROL_FAIL"
                    quality += row.get("status") == "DATA_QUALITY_FAILED"
            lines.append(f"| {PLANNER_LABEL[planner]} | {base} | {planned} | {completed} | {valid} | {passed} | {failed} | {quality} |")
    return lines


def quality_table(batches):
    header = ["| 参数生成 | 曲线表达 | n |"]
    rule = ["|---|---|---:|"]
    for _, label in METRICS:
        header.append(f" {label} 中位数 [Q1, Q3] |")
        rule.append("---:|")
    lines = ["".join(header), "".join(rule)]
    for planner in PLANNERS:
        for base in BASES:
            group = next(key for key, value in CELLS.items() if value == (base, planner))
            values = {name: [] for name, _ in METRICS}
            count = 0
            for batch in batches:
                for row in batch["rows"]:
                    if row["group_id"] != group or not usable(row):
                        continue
                    count += 1
                    for name, _ in METRICS:
                        values[name].append(number(row.get(name)))
            cells = []
            for name, _ in METRICS:
                stats = describe(values[name])
                if stats["n"] == 0:
                    cells.append(" — |")
                elif stats["n"] == 1:
                    cells.append(f" {fmt(stats['median'])} |")
                else:
                    cells.append(f" {fmt(stats['median'])} [{fmt(stats['q1'])}, {fmt(stats['q3'])}] |")
            lines.append(f"| {PLANNER_LABEL[planner]} | {base} | {count} |" + "".join(cells))
    return lines


def paired_table(batches):
    lines = ["| 对比 | 含义 | 指标 | 配对数n | 中位差 | 优/平/劣 |",
             "|---|---|---|---:|---:|---:|"]
    for tag, left, right, meaning in CONTRASTS:
        for name, label in METRICS:
            differences = []
            for batch in batches:
                left_rows = [row for row in batch["rows"] if row["group_id"] == left and usable(row)]
                right_rows = [row for row in batch["rows"] if row["group_id"] == right and usable(row)]
                right_by_seed = {row.get("seed"): row for row in right_rows}
                for row in left_rows:
                    partner = right_by_seed.get(row.get("seed"))
                    if partner is None:
                        continue
                    a, b = number(row.get(name)), number(partner.get(name))
                    if a is None or b is None:
                        continue
                    differences.append(a - b)
            if not differences:
                lines.append(f"| {tag} | {meaning} | {label} | 0 | — | — |")
                continue
            median = percentile(differences, 0.5)
            better = sum(1 for value in differences if value < 0)
            worse = sum(1 for value in differences if value > 0)
            lines.append(f"| {tag} | {meaning} | {label} | {len(differences)} | {fmt(median)} | {better}/{len(differences) - better - worse}/{worse} |")
    return lines


def resource_table(batches):
    lines = ["| 参数生成 | 曲线表达 | n |"] + [" " + label + " 中位数 |" for _, label in RESOURCE]
    lines = ["".join(lines), "|---|---|---:|" + "---:|" * len(RESOURCE)]
    for planner in PLANNERS:
        for base in BASES:
            group = next(key for key, value in CELLS.items() if value == (base, planner))
            values = {name: [] for name, _ in RESOURCE}
            count = 0
            for batch in batches:
                for row in batch["rows"]:
                    if row["group_id"] != group or not usable(row):
                        continue
                    count += 1
                    for name, _ in RESOURCE:
                        values[name].append(number(row.get(name)))
            cells = []
            for name, _ in RESOURCE:
                stats = describe(values[name])
                cells.append(f" {fmt(stats['median'])} |" if stats["n"] else " — |")
            lines.append(f"| {PLANNER_LABEL[planner]} | {base} | {count} |" + "".join(cells))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", type=Path, help="包含 comparison.csv 的对比目录")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    batches = collect(args.batches)
    args.output.mkdir(parents=True, exist_ok=True)

    fingerprints = {batch["fingerprint"] for batch in batches}
    schedule = []
    for batch in batches:
        for row in batch["rows"]:
            if usable(row):
                schedule.append(dict(batch=batch["label"], group_id=row["group_id"],
                                     seed=row.get("seed"), source_plan_sha256=row.get("source_plan_sha256")))

    body = [
        "# 六组控制实验汇总",
        "",
        f"批次：{', '.join(batch['label'] for batch in batches)}。",
        f"条件指纹：{'、'.join(sorted(fingerprints))}（{len(fingerprints)} 个）。",
        "",
        "不同指纹的批次仍可并列汇总，但同一张配对表内只按批次和种子配对；跨批次不做合并显著性判断。",
        "",
        "## 表1 运行完整性与门槛",
        "",
    ]
    body += cell_table(batches)
    body += ["", "流程完成、有效完成与门槛通过分开统计；未运行、执行失败和数据质量失败都保留在表中。", "",
             "## 表2 执行质量（有效样本的条件性描述）",
             ""]
    body += quality_table(batches)
    body += ["", "中位数与四分位仅在 n>1 时给出区间；n=1 只报单值，不构成重复证据。", "",
             "## 表3 配对差（同一批次、同一种子）",
             ""]
    body += paired_table(batches)
    body += ["", "差值 = 左组 − 右组。误差类指标越负越好，参考时长与综合分同样越负越好。优/平/劣按差值符号计数，不做显著性检验。",
              "", "## 表4 资源与执行开销", ""]
    body += resource_table(batches)
    body += ["", "规划墙钟来自冻结的简化模型搜索证据，不是本次 AirSim 在线搜索，也不能用来主张墙钟提效。", ""]

    (args.output / "summary.md").write_text("\n".join(body), encoding="utf-8")
    with (args.output / "cells.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["batch", "group_id", "cell", "seed"] + [name for name, _ in METRICS])
        for batch in batches:
            for row in batch["rows"]:
                if not usable(row):
                    continue
                base, planner = CELLS.get(row["group_id"], (None, None))
                writer.writerow([batch["label"], row["group_id"], f"{base}/{planner}", row.get("seed")]
                                + [row.get(name) for name, _ in METRICS])
    (args.output / "provenance.json").write_text(json.dumps(
        dict(schema_version=1, batches=[dict(label=batch["label"], path=str(batch["path"]),
                                             fingerprint=batch["fingerprint"],
                                             rows=len(batch["rows"])) for batch in batches],
             usable_rows=len(schedule), fingerprints=sorted(fingerprints),
             scope="Descriptive summary of recorded replays only; no new execution, no significance test, "
                   "no causal or wall-clock claim."), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), batches=len(batches), usable_rows=len(schedule),
                          fingerprints=len(fingerprints)), ensure_ascii=False))


if __name__ == "__main__":
    main()
