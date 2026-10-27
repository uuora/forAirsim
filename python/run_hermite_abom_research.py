"""Reproducible offline ABOM/search pilot on the existing production controller.

Train and validation are separate commands. Selection uses training scores only.
The existing validation set includes 24 legacy regressions and 6 conditions on
three held-out geometries. This is a lagged point-mass pilot, never a flight test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from dual_target_control.core import BezierParameters
from dual_target_control.trajectory_objective import (
    evaluate_parameters, make_training_cases, make_validation_cases,
)
from trajectory_research.search import METHODS, run_search

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "reports/hermite_disturbance_20261027/search_pilot"
SOURCE_FILES = (
    "python/dual_target_control/core.py",
    "python/dual_target_control/trajectory_objective.py",
    "python/validate_dual_target_turning.py",
    "python/trajectory_research/search.py",
    "python/run_hermite_abom_research.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_selection(output, declaration):
    selection = read(output / "selection.json")
    manifest_sha = digest(output / "manifest.json")
    if selection["manifest_sha256"] != manifest_sha:
        raise ValueError("Selection/manifest mismatch")
    expected = {(method, seed) for method in declaration["methods"] for seed in declaration["seeds"]}
    actual = [(c["method"], c["seed"]) for c in selection["incumbents"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Selection method/seed coverage is not exact and unique")
    for choice in selection["incumbents"]:
        path = output / choice["training_file"]
        if digest(path) != choice["training_sha256"]:
            raise ValueError("Training result changed after selection")
        result = read(path)
        if result.get("manifest_sha256") != manifest_sha or any(
                choice[key] != result[key] for key in ("method", "seed", "best_parameters")) \
                or choice["training_score"] != result["best_score"]:
            raise ValueError("Selected candidate is not the training incumbent")
    return selection


def manifest(args):
    return dict(schema="hermite-abom-offline-pilot/2", scope="development pilot, not AirSim or real flight",
                seeds=args.seeds, methods=args.methods, budget=args.budget,
                population=args.population, dt_s=args.dt,
                initialization=args.initialization, injection_fraction=args.injection_fraction,
                training_cases=[asdict(c) for c in make_training_cases()],
                validation_cases=[asdict(c) for c in make_validation_cases()],
                source_sha256={p: digest(ROOT / p) for p in SOURCE_FILES},
                python=platform.python_version(), numpy=np.__version__,
                upstream=str(args.upstream.resolve()),
                independent_unit="optimizer seed; rollout conditions within a seed are paired, not independent repeats",
                selection="one incumbent per method per seed, selected only by training objective",
                limitations="No force/wind, sensing, rotor, thrust, or attitude model; feasibility is model-specific")


def train(args, output):
    declaration = manifest(args)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read(manifest_path) != declaration:
        raise ValueError("Existing campaign differs; choose another output instead of overwriting evidence")
    save(manifest_path, declaration)
    for relative, expected in declaration["source_sha256"].items():
        frozen = output / "source_snapshot" / relative
        if frozen.exists():
            if digest(frozen) != expected:
                raise ValueError("Frozen source snapshot changed")
        else:
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes((ROOT / relative).read_bytes())
    manifest_sha = digest(manifest_path)
    if any(method.startswith("abom") for method in args.methods):
        import torch
        torch.set_num_threads(1)
        # Explicit one-off initialization, outside per-search timing.
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    cases = make_training_cases()
    baseline = BezierParameters()
    selected = []
    for seed in args.seeds:
        for method in args.methods:
            path = output / f"train_{method}_s{seed}.json"
            if path.exists():
                result = read(path)
                if result.get("manifest_sha256") != manifest_sha or result["budget"] != args.budget or result["seed"] != seed or result["method"] != method:
                    raise ValueError("Inconsistent existing result")
            else:
                result = run_search(lambda x: evaluate_parameters(x, cases, dt=args.dt),
                                    baseline.lower_bounds(), baseline.upper_bounds(), method=method,
                                    seed=seed, budget=args.budget, population=args.population,
                                    anchor=baseline.to_vector(), upstream=args.upstream,
                                    initialization=args.initialization,
                                    injection_fraction=args.injection_fraction)
                result["manifest_sha256"] = manifest_sha
                save(path, result)
            selected.append(dict(method=method, seed=seed, best_parameters=result["best_parameters"],
                                 training_score=result["best_score"], training_file=path.name,
                                 training_sha256=digest(path)))
            print(json.dumps(dict(stage="train", method=method, seed=seed,
                                  score=result["best_score"], evaluations=len(result["evaluations"]),
                                  elapsed_s=result["elapsed_s"])), flush=True)
    # Freeze the parameter bytes before any validation results are generated.
    frozen = dict(selection_basis="training_only", manifest_sha256=manifest_sha, incumbents=selected)
    if (output / "selection.json").exists() and read(output / "selection.json") != frozen:
        raise ValueError("Frozen selection changed; retain this campaign and use another output")
    save(output / "selection.json", frozen)


def validate(args, output):
    declaration = read(output / "manifest.json")
    for path, sha in declaration["source_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Source changed after training: {path}")
    selection = checked_selection(output, declaration)
    cases = make_validation_cases()
    baseline_path = output / "validation_default.json"
    if not baseline_path.exists():
        baseline = evaluate_parameters(BezierParameters(), cases, dt=declaration["dt_s"])
        baseline["manifest_sha256"] = digest(output / "manifest.json")
        save(baseline_path, baseline)
    if read(baseline_path).get("manifest_sha256") != digest(output / "manifest.json") \
            or read(baseline_path)["parameters"] != BezierParameters().to_vector().tolist():
        raise ValueError("Baseline validation belongs to a different manifest")
    rows = []
    for choice in selection["incumbents"]:
        if digest(output / choice["training_file"]) != choice["training_sha256"]:
            raise ValueError("Training result changed after selection")
        path = output / f'validation_{choice["method"]}_s{choice["seed"]}.json'
        if not path.exists():
            result = evaluate_parameters(choice["best_parameters"], cases, dt=declaration["dt_s"])
            result["selection_sha256"] = digest(output / "selection.json")
            result["manifest_sha256"] = digest(output / "manifest.json")
            save(path, result)
        result = read(path)
        if result.get("selection_sha256") != digest(output / "selection.json") \
                or result.get("manifest_sha256") != digest(output / "manifest.json") \
                or result["parameters"] != choice["best_parameters"]:
            raise ValueError("Cached validation is not bound to this selected candidate")
        row = dict(method=choice["method"], seed=choice["seed"], training_score=choice["training_score"],
                   validation_score=result["score"], feasible=result["feasible"],
                   failed_cases=result["score_details"]["failed_cases"],
                   elapsed_mean_s=result["score_details"]["mean_elapsed_s"],
                   jerk_proxy=result["score_details"]["mean_normalized_peak_command_jerk"],
                   cases=len(result["cases"]), held_out_seed_cases_passed=sum(r["passed"] for r in result["cases"][-6:]),
                   held_out_seed_cases=6, result_file=path.name)
        rows.append(row)
        print(json.dumps(dict(stage="validation", **row)), flush=True)
    save(output / "validation_summary.json", dict(rows=rows, selection_sha256=digest(output / "selection.json")))


def summarize(output):
    declaration = read(output / "manifest.json")
    selection = checked_selection(output, declaration)
    summary = read(output / "validation_summary.json")
    if summary["selection_sha256"] != digest(output / "selection.json"):
        raise ValueError("Validation summary belongs to another selection")
    rows = summary["rows"]
    default = read(output / "validation_default.json")
    if default.get("manifest_sha256") != digest(output / "manifest.json") \
            or default["parameters"] != BezierParameters().to_vector().tolist():
        raise ValueError("Baseline manifest mismatch")
    expected = {(c["method"], c["seed"]): c for c in selection["incumbents"]}
    if len(rows) != len(expected) or {(r["method"], r["seed"]) for r in rows} != set(expected):
        raise ValueError("Validation summary coverage mismatch")
    for row in rows:
        choice = expected[row["method"], row["seed"]]
        result = read(output / row["result_file"])
        if result.get("manifest_sha256") != digest(output / "manifest.json") \
                or result.get("selection_sha256") != digest(output / "selection.json") \
                or result["parameters"] != choice["best_parameters"] \
                or row["validation_score"] != result["score"] or row["feasible"] != result["feasible"] \
                or row["elapsed_mean_s"] != result["score_details"]["mean_elapsed_s"] \
                or row["training_score"] != choice["training_score"] \
                or row["failed_cases"] != result["score_details"]["failed_cases"] \
                or row["jerk_proxy"] != result["score_details"]["mean_normalized_peak_command_jerk"] \
                or row["cases"] != len(result["cases"]) or row["held_out_seed_cases"] != 6 \
                or row["held_out_seed_cases_passed"] != sum(r["passed"] for r in result["cases"][-6:]):
            raise ValueError("Validation summary is not bound to its result")
    methods = {}
    for method in declaration["methods"]:
        subset = [r for r in rows if r["method"] == method]
        training = [read(output / f'train_{method}_s{r["seed"]}.json') for r in subset]
        methods[method] = dict(seeds=len(subset), all_cases_feasible_seeds=sum(r["feasible"] for r in subset),
                               failed_cases=sum(r["failed_cases"] for r in subset),
                               mean_validation_time_s=float(np.mean([r["elapsed_mean_s"] for r in subset])),
                               mean_training_score=float(np.mean([r["training_score"] for r in subset])),
                               mean_search_wall_s=float(np.mean([r["elapsed_s"] for r in training])),
                               total_training_candidate_evaluations=sum(len(r["evaluations"]) for r in training))
    pairs = []
    for comparator in ("random", "gaussian", "abom"):
        if comparator not in methods or "abom_perturb" not in methods:
            continue
        a = {r["seed"]: r for r in rows if r["method"] == "abom_perturb"}
        b = {r["seed"]: r for r in rows if r["method"] == comparator}
        seeds = sorted(a.keys() & b.keys())
        difference = np.array([a[s]["elapsed_mean_s"] - b[s]["elapsed_mean_s"] for s in seeds])
        rng = np.random.default_rng(20261027)
        means = rng.choice(difference, size=(10000, len(seeds)), replace=True).mean(axis=1)
        pairs.append(dict(comparator=comparator, paired_seeds=seeds,
                          mean_delta_time_s=float(difference.mean()),
                          bootstrap_95_percentile_ci_s=np.quantile(means, [.025, .975]).tolist(),
                          negative_favors="abom_perturb", interpretation="exploratory seed-paired interval; failures retained"))
    report = dict(methods=methods, paired_time_comparisons=pairs,
                  baseline=dict(feasible=default["feasible"], score=default["score"], **default["score_details"]),
                  budget_accounting=dict(training_candidates=sum(m["total_training_candidate_evaluations"] for m in methods.values()),
                                         training_rollout_conditions=sum(m["total_training_candidate_evaluations"] for m in methods.values()) * len(declaration["training_cases"]),
                                         validation_candidates=len(rows) + 1,
                                         validation_rollout_conditions=(len(rows) + 1) * len(declaration["validation_cases"])),
                  scope=declaration["scope"], limitations=declaration["limitations"])
    save(output / "summary.json", report)
    text = ["# ABOM 参数搜索离线先导", "", "以下为固定控制器的一阶滞后点质量模型结果，不构成 AirSim 或实机结论。", "",
            "每个方法使用相同初始种群、默认参数锚和候选评价预算；验证在训练选择冻结后单独执行。", "",
            "|方法|种子数|全部30条件通过的种子|失败条件总数|平均运行至完成或截止(s)|平均搜索墙钟(s)|",
            "|---|---:|---:|---:|---:|---:|"]
    for name, row in methods.items():
        text.append(f'|{name}|{row["seeds"]}|{row["all_cases_feasible_seeds"]}|{row["failed_cases"]}|{row["mean_validation_time_s"]:.4f}|{row["mean_search_wall_s"]:.2f}|')
    text.extend(["", "失败保持在完成时间汇总中；时间更短不能抵消漏点、入口姿态或边界失败。", "",
                 "24个条件是旧回归矩阵，其中两种布局参与调参；另外6个条件来自17、23、31三个未参与训练的几何种子。", "",
                 "`summary.json` 包含配对探索性区间与独立预算账；小样本区间不能作为正式统计结论。", "",
                 "AI辅助实现与分析；原始逐候选结果和训练/选择/验证哈希保存在本目录。", ""])
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "summarize"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--upstream", type=Path, default=ROOT / "tmp/abom_deploy_probe/upstream")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--dt", type=float, default=.05)
    parser.add_argument("--initialization", choices=("uniform", "lhs"), default="uniform")
    parser.add_argument("--injection-fraction", type=float, default=.5)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seed and method lists must contain no duplicates")
    if args.stage == "train":
        train(args, args.output.resolve())
    elif args.stage == "validate":
        validate(args, args.output.resolve())
    else:
        summarize(args.output.resolve())


if __name__ == "__main__":
    main()
