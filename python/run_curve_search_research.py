"""Frozen offline search/ablation on boundary-preserving polynomial references.

Separate train, validate and summarize commands prevent validation selection.
This task has no vehicle dynamics, controller, wind or perception simulation.
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

from trajectory_research.curve_task import (
    DEFAULT_PARAMETERS, SCOPE, evaluate_curve_parameters, make_training_curve_cases,
    make_validation_curve_cases, parameter_bounds,
)
from trajectory_research.search import METHODS, run_search

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("python/trajectory_research/curves.py", "python/trajectory_research/constraints.py",
           "python/trajectory_research/curve_task.py", "python/trajectory_research/search.py",
           "python/run_curve_search_research.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_new_or_equal(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}; choose another output")
        return
    temp = path.with_suffix(".pending")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def declaration(args):
    profile = args.profile
    if profile == "legacy":
        training = make_training_curve_cases()
        validation_policy = dict(kind="fixed_eight_geometries")
    elif profile.startswith("ellipsoid_"):
        from trajectory_research.directional_task import make_directional_training_cases
        training = make_directional_training_cases()
        validation_policy = dict(kind="ellipsoid_sphere_stress", seed=20261029, count=128,
                                 generated_after_training_selection=True,
                                 centre_factor_m=[[.30, 0., 0.], [0., .05, 0.], [0., 0., .05]],
                                 radius_increment_m=.05, point_reference_clearance_m=.1)
    else:
        from trajectory_research.uncertainty import make_nominal_training_cases, make_uncertainty_training_cases
        training = make_nominal_training_cases() if profile == "nominal" else make_uncertainty_training_cases()
        validation_policy = dict(kind="bounded_sphere_stress", seed=20261028, count=64,
                                 generated_after_training_selection=True)
    sources = list(SOURCES)
    if profile.startswith("ellipsoid_"):
        sources.extend(("python/trajectory_research/directional.py", "python/trajectory_research/directional_task.py"))
    elif profile != "legacy":
        sources.append("python/trajectory_research/uncertainty.py")
    return dict(schema="boundary-preserving-curve-search/1", seeds=args.seeds, methods=args.methods,
                profile=profile, validation_policy=validation_policy,
                population=args.population, budget=args.budget, initialization="uniform", injection_fraction=.5,
                source_sha256={path: digest(ROOT / path) for path in sources},
                training_cases=[asdict(c) for c in training],
                validation_cases=[asdict(c) for c in make_validation_curve_cases()] if profile == "legacy" else None,
                scope=SCOPE, python=platform.python_version(), numpy=np.__version__,
                upstream=str(args.upstream.resolve()), anchor=list(DEFAULT_PARAMETERS),
                selection="one training incumbent per optimizer seed; no validation selection",
                independent_unit="optimizer seed; obstacle conditions are correlated tests of one frozen reference",
                order="cyclic method rotation across seeds after explicit torch/AdamW initialization")


def assert_manifest_sources(manifest):
    for path, sha in manifest["source_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Source changed since freeze: {path}")


def load_selection(output):
    manifest = read(output / "manifest.json")
    selected = read(output / "selection.json")
    if selected["manifest_sha256"] != digest(output / "manifest.json"):
        raise ValueError("Manifest/selection hash mismatch")
    expected = {(method, seed) for method in manifest["methods"] for seed in manifest["seeds"]}
    actual = [(c["method"], c["seed"]) for c in selected["incumbents"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Selection method/seed coverage must be exact and unique")
    for choice in selected["incumbents"]:
        if choice["training_sha256"] != digest(output / choice["training_file"]):
            raise ValueError("Training data changed since selection")
        result = read(output / choice["training_file"])
        if result.get("manifest_sha256") != digest(output / "manifest.json") or any(
                choice[key] != result[key] for key in ("method", "seed", "best_parameters")) \
                or choice["training_score"] != result["best_score"]:
            raise ValueError("Selected candidate differs from training incumbent")
    return manifest, selected


def train(args, output):
    manifest = declaration(args)
    save_new_or_equal(output / "manifest.json", manifest)
    for relative, expected in manifest["source_sha256"].items():
        frozen = output / "source_snapshot" / relative
        if frozen.exists():
            if digest(frozen) != expected:
                raise ValueError("Frozen source snapshot changed")
        else:
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_bytes((ROOT / relative).read_bytes())
    manifest_sha = digest(output / "manifest.json")
    if any(m.startswith("abom") for m in args.methods):
        import torch
        torch.set_num_threads(1)
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    lower, upper = parameter_bounds()
    if args.profile == "legacy":
        objective = evaluate_curve_parameters
    elif args.profile.startswith("ellipsoid_"):
        from trajectory_research.directional_task import evaluate_directional_parameters
        objective = lambda x: evaluate_directional_parameters(x, profile=args.profile)
    else:
        from trajectory_research.uncertainty import evaluate_uncertainty_parameters
        objective = lambda x: evaluate_uncertainty_parameters(x, profile=args.profile)
    chosen = []
    for index, seed in enumerate(args.seeds):
        shift = index % len(args.methods)
        ordered = args.methods[shift:] + args.methods[:shift]
        for method in ordered:
            path = output / f"train_{method}_s{seed}.json"
            if path.exists():
                result = read(path)
                if result.get("manifest_sha256") != manifest_sha or result["method"] != method or result["seed"] != seed:
                    raise ValueError("Training cache mismatch")
            else:
                result = run_search(objective, lower, upper, method=method,
                                    seed=seed, population=args.population, budget=args.budget,
                                    anchor=DEFAULT_PARAMETERS, upstream=args.upstream)
                result["manifest_sha256"] = manifest_sha
                save_new_or_equal(path, result)
            chosen.append(dict(method=method, seed=seed, best_parameters=result["best_parameters"],
                               training_score=result["best_score"], training_file=path.name,
                               training_sha256=digest(path)))
            print(json.dumps(dict(stage="curve_train", method=method, seed=seed, score=result["best_score"],
                                  evaluations=len(result["evaluations"]), elapsed_s=result["elapsed_s"])), flush=True)
    chosen.sort(key=lambda c: (c["seed"], c["method"]))
    save_new_or_equal(output / "selection.json", dict(manifest_sha256=manifest_sha,
                                                     selection_basis="training_only", incumbents=chosen))


def validate(output):
    manifest, selected = load_selection(output)
    assert_manifest_sources(manifest)
    selection_sha, manifest_sha = digest(output / "selection.json"), digest(output / "manifest.json")
    policy = manifest.get("validation_policy", {"kind": "fixed_eight_geometries"})
    if policy["kind"] == "fixed_eight_geometries":
        cases = make_validation_curve_cases()
    elif policy["kind"] == "ellipsoid_sphere_stress":
        from trajectory_research.directional_task import make_directional_stress_cases
        cases = make_directional_stress_cases(policy["seed"], count=policy["count"])
        save_new_or_equal(output / "stress_cases.json", dict(selection_sha256=selection_sha,
                                                           cases=[asdict(c) for c in cases]))
    else:
        from trajectory_research.uncertainty import make_uncertainty_stress_cases
        cases = make_uncertainty_stress_cases(policy["seed"], count=policy["count"])
        save_new_or_equal(output / "stress_cases.json", dict(selection_sha256=selection_sha,
                                                           cases=[asdict(c) for c in cases]))
    choices = selected["incumbents"] + [dict(method="default", seed=None, best_parameters=list(DEFAULT_PARAMETERS))]
    rows = []
    for choice in choices:
        name = f'validation_{choice["method"]}_s{choice["seed"]}.json'
        path = output / name
        if path.exists():
            result = read(path)
        else:
            result = evaluate_curve_parameters(choice["best_parameters"], cases)
            result.update(manifest_sha256=manifest_sha, selection_sha256=selection_sha)
            save_new_or_equal(path, result)
        if result.get("manifest_sha256") != manifest_sha or result.get("selection_sha256") != selection_sha \
                or result["parameters"] != choice["best_parameters"]:
            raise ValueError("Validation cache does not belong to this frozen candidate")
        row = dict(method=choice["method"], seed=choice["seed"], score=result["score"],
                   feasible=result["feasible"], cases=len(result["cases"]),
                   failed_cases=result["score_details"]["failed_cases"],
                   reference_duration_s=result["score_details"]["duration_s"],
                   jerk_energy_m2_s5=result["score_details"]["jerk_energy_m2_s5"],
                   result_file=name, result_sha256=digest(path))
        rows.append(row)
        print(json.dumps(dict(stage="curve_validation", **row)), flush=True)
    save_new_or_equal(output / "validation_summary.json", dict(manifest_sha256=manifest_sha,
                                                             selection_sha256=selection_sha, rows=rows))


def summarize(output):
    manifest, selected = load_selection(output)
    validation = read(output / "validation_summary.json")
    if validation["manifest_sha256"] != digest(output / "manifest.json") or validation["selection_sha256"] != digest(output / "selection.json"):
        raise ValueError("Validation summary hash mismatch")
    for row in validation["rows"]:
        if digest(output / row["result_file"]) != row["result_sha256"]:
            raise ValueError("Validation result changed after summary")
    methods = {}
    for method in manifest["methods"]:
        rows = [r for r in validation["rows"] if r["method"] == method]
        runs = [read(output / f'train_{method}_s{r["seed"]}.json') for r in rows]
        methods[method] = dict(seeds=len(rows), feasible_all_conditions_seeds=sum(r["feasible"] for r in rows),
                               failed_conditions=sum(r["failed_cases"] for r in rows),
                               mean_training_score=float(np.mean([r["best_score"] for r in runs])),
                               mean_validation_score=float(np.mean([r["score"] for r in rows])),
                               mean_reference_duration_s=float(np.mean([r["reference_duration_s"] for r in rows])),
                               mean_search_s=float(np.mean([r["elapsed_s"] for r in runs])))
    pairs = []
    for method in manifest["methods"]:
        if method in ("abom", "random") or "abom" not in manifest["methods"]:
            continue
        a = {r["seed"]: r for r in validation["rows"] if r["method"] == method}
        b = {r["seed"]: r for r in validation["rows"] if r["method"] == "abom"}
        seeds = sorted(a.keys() & b.keys())
        delta = np.array([a[s]["score"] - b[s]["score"] for s in seeds])
        bootstrap = np.random.default_rng(20261027).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        pairs.append(dict(method=method, comparator="abom", seeds=seeds, mean_delta_validation_score=float(delta.mean()),
                          exploratory_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist(),
                          wins=int(np.sum(delta < 0)), ties=int(np.sum(delta == 0)), losses=int(np.sum(delta > 0)),
                          negative_favors=method, independent_unit="optimizer seed; no multiplicity correction"))
    report = dict(scope=manifest["scope"], profile=manifest.get("profile", "legacy"), methods=methods, paired_comparisons=pairs,
                  baseline=next(r for r in validation["rows"] if r["method"] == "default"),
                  training_candidates=manifest["budget"] * len(selected["incumbents"]),
                  training_sphere_conditions=manifest["budget"] * len(selected["incumbents"]) * len(manifest["training_cases"]),
                  validation_candidates=len(validation["rows"]),
                  validation_sphere_conditions=sum(r["cases"] for r in validation["rows"]),
                  interpretation="Exploratory optimization and stress transfer of fixed references; no controller, flight or wind conclusions")
    save_new_or_equal(output / "summary.json", report)
    lines = ["# 边界保持曲线搜索与消融", "", manifest["scope"], "",
             "|方法|种子|全部压力条件通过种子|失败条件总数|平均训练分数|平均验证分数|平均参考时长(s)|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for method, r in methods.items():
        lines.append(f'|{method}|{r["seeds"]}|{r["feasible_all_conditions_seeds"]}|{r["failed_conditions"]}|{r["mean_training_score"]:.6f}|{r["mean_validation_score"]:.6f}|{r["mean_reference_duration_s"]:.4f}|')
    lines += ["", "验证把训练后冻结的同一参考直接转移到新障碍条件，没有使用障碍条件重新规划；这是压力检验。", "",
              "分数优先反映全部参考约束可行性。失败样本全部保留，参考时长不能解释为实际飞行完成时间。", "",
              "配对区间为探索性分析，尚未校正多重比较；结果不能替代联合闭环与飞行验证。", "",
              "AI辅助实现与分析。逐候选、冻结选择与逐条件证据保存在本目录。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "summarize"))
    parser.add_argument("--output", type=Path, default=ROOT / "reports/hermite_disturbance_20261027/curve_search")
    parser.add_argument("--upstream", type=Path, default=ROOT / "tmp/abom_deploy_probe/upstream")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 20)))
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--budget", type=int, default=128)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--profile", choices=("legacy", "nominal", "margin", "ellipsoid_nominal",
                                              "ellipsoid_isotropic", "ellipsoid_directional"), default="legacy")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seed and method lists must be unique")
    output = args.output.resolve()
    if args.stage == "train":
        train(args, output)
    elif args.stage == "validate":
        validate(output)
    else:
        summarize(output)
