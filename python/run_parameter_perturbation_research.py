"""Frozen black-box optimization of initial-curve perturbations.

Train, validate and summarize are separate commands. Each layout is a distinct
optimization task. Method/seed incumbents are selected using training only;
within-envelope stress scenes are generated afterwards. No simulator I/O.
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

from trajectory_research.blackbox import BLACKBOX_METHODS, run_blackbox_search
from trajectory_research.parameterizations import (FAMILIES, default_parameters,
                                                  parameter_bounds, parameter_names)
from trajectory_research.perturbation_task import (declaration as task_declaration,
    evaluate_parameters, get_layout, stress_obstacles)
from trajectory_research.search import run_search


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("random", "gaussian", "de", "jde", "coordinate", "abom", "abom_perturb")
SOURCES = tuple("python/trajectory_research/" + name + ".py" for name in
                ("curves", "constraints", "curve_task", "directional", "search",
                 "blackbox", "parameterizations", "perturbation_task")) \
    + ("python/run_parameter_perturbation_research.py",)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen artifact differs: {path}; use a new output directory")
        return
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def key(family, layout, method, seed):
    return f"{family}__{layout}__{method}__s{seed}"


def manifest_for(args):
    return dict(schema="initial-curve-parameter-perturbation/1",
                families=args.families, layouts=args.layouts, methods=args.methods, seeds=args.seeds,
                population=args.population, budget=args.budget, initialization="uniform",
                representation="hermite", certificate_degree=7,
                source_sha256={path: digest(ROOT / path) for path in SOURCES},
                task=task_declaration(),
                variables={family: dict(names=parameter_names(family),
                    lower=parameter_bounds(family)[0].tolist(), upper=parameter_bounds(family)[1].tolist(),
                    anchor=default_parameters(family).tolist()) for family in args.families},
                validation=dict(seed=args.stress_seed, count=args.stress_count,
                    generation="after all training incumbents frozen; no validation selection",
                    interpretation="within-envelope implementation stress, not unseen-layout generalization"),
                independent_unit="optimizer seed within each fixed development layout",
                ordering="cyclic methods across seeds; torch/AdamW warmup outside timed search",
                upstream=str(args.upstream.resolve()), python=platform.python_version(), numpy=np.__version__)


def assert_sources(manifest, output=None):
    for relative, expected in manifest["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source changed after freeze: {relative}")
        if output is not None and digest(output / "source_snapshot" / relative) != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def checked_training(output, manifest, family, layout, method, seed):
    path = output / "train" / (key(family, layout, method, seed) + ".json")
    run = read(path)
    expected = dict(family=family, layout=layout, method=method, seed=seed,
                    manifest_sha256=digest(output / "manifest.json"),
                    budget=manifest["budget"], population=manifest["population"])
    if any(run.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Training cache binding mismatch: {path}")
    records = run["evaluations"]
    if len(records) != manifest["budget"] or [r["fe"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Wrong or nonconsecutive FE budget")
    if not all(np.isfinite(r["score"]) and isinstance(r["feasible"], bool) for r in records):
        raise ValueError("Invalid training score/feasibility")
    best = min(records, key=lambda r: r["score"])
    if run["best_score"] != best["score"] or not any(
            r["score"] == run["best_score"] and r["parameters"] == run["best_parameters"] for r in records):
        raise ValueError("Incumbent is not an evaluated training minimum")
    variables = manifest["variables"][family]
    lo, hi = np.array(variables["lower"]), np.array(variables["upper"])
    parameters = np.array([r["parameters"] for r in records])
    if parameters.shape != (manifest["budget"], len(lo)) or not np.isfinite(parameters).all() \
            or np.any(parameters < lo) or np.any(parameters > hi):
        raise ValueError("Candidate outside frozen physical box")
    return path, run


def train(args, output):
    manifest = manifest_for(args)
    freeze(output / "manifest.json", manifest)
    for relative, expected in manifest["source_sha256"].items():
        path = output / "source_snapshot" / relative
        if path.exists():
            if digest(path) != expected:
                raise ValueError("Source snapshot mismatch")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
    assert_sources(manifest, output)
    if any(method.startswith("abom") for method in args.methods):
        import torch
        torch.set_num_threads(1)
        torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    choices = []
    for family in args.families:
        lower, upper = parameter_bounds(family)
        for layout in args.layouts:
            objective = lambda x: evaluate_parameters(x, family=family, layout=layout)
            for index, seed in enumerate(args.seeds):
                shift = index % len(args.methods)
                methods = args.methods[shift:] + args.methods[:shift]
                initial = None
                for method in methods:
                    name = key(family, layout, method, seed)
                    path = output / "train" / (name + ".json")
                    if not path.exists():
                        options = dict(method=method, seed=seed, population=args.population,
                                       budget=args.budget, anchor=default_parameters(family))
                        if method in BLACKBOX_METHODS:
                            result = run_blackbox_search(objective, lower, upper, **options)
                        else:
                            result = run_search(objective, lower, upper, upstream=args.upstream, **options)
                        result.update(family=family, layout=layout, manifest_sha256=digest(output / "manifest.json"))
                        freeze(path, result)
                    _, result = checked_training(output, manifest, family, layout, method, seed)
                    if initial is None:
                        initial = result["initial_population_normalized"]
                    elif initial != result["initial_population_normalized"]:
                        raise ValueError("Methods did not receive identical initial populations")
                    choices.append(dict(family=family, layout=layout, method=method, seed=seed,
                        parameters=result["best_parameters"], training_score=result["best_score"],
                        training_file=path.relative_to(output).as_posix(), training_sha256=digest(path)))
                    print(json.dumps(dict(stage="train", key=name, score=result["best_score"],
                        duration_s=result["best_parameters"][-1], search_s=result["elapsed_s"])), flush=True)
    choices.sort(key=lambda x: (x["family"], x["layout"], x["seed"], x["method"]))
    freeze(output / "selection.json", dict(manifest_sha256=digest(output / "manifest.json"),
                                           selection_basis="training_only", incumbents=choices))


def load_selection(output):
    manifest, selection = read(output / "manifest.json"), read(output / "selection.json")
    if selection["manifest_sha256"] != digest(output / "manifest.json") or selection["selection_basis"] != "training_only":
        raise ValueError("Selection/manifest mismatch")
    expected = {(f, l, m, s) for f in manifest["families"] for l in manifest["layouts"]
                for m in manifest["methods"] for s in manifest["seeds"]}
    actual = [(r["family"], r["layout"], r["method"], r["seed"]) for r in selection["incumbents"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Selection coverage differs from the frozen design")
    for choice in selection["incumbents"]:
        path, run = checked_training(output, manifest, *(choice[k] for k in ("family", "layout", "method", "seed")))
        if digest(path) != choice["training_sha256"] or choice["parameters"] != run["best_parameters"] \
                or choice["training_score"] != run["best_score"] or path != output / choice["training_file"]:
            raise ValueError("Selection/training mismatch")
    return manifest, selection


def validate(output):
    manifest, selection = load_selection(output)
    assert_sources(manifest, output)
    selection_sha = digest(output / "selection.json")
    scenes = {}
    for index, layout in enumerate(manifest["layouts"]):
        scenes[layout] = stress_obstacles(layout, seed=manifest["validation"]["seed"] + index,
                                          count=manifest["validation"]["count"])
    freeze(output / "stress_cases.json", dict(selection_sha256=selection_sha,
        cases={layout: [[asdict(o) for o in scene] for scene in values] for layout, values in scenes.items()}))
    choices = selection["incumbents"] + [dict(family=f, layout=l, method="initial", seed=None,
        parameters=default_parameters(f).tolist()) for f in manifest["families"] for l in manifest["layouts"]]
    rows = []
    for choice in choices:
        family, layout = choice["family"], choice["layout"]
        path = output / "validation" / (key(family, layout, choice["method"], choice["seed"]) + ".json")
        if path.exists():
            result = read(path)
        else:
            training_check = evaluate_parameters(choice["parameters"], family=family, layout=layout)
            cases = [evaluate_parameters(choice["parameters"], family=family, layout=layout,
                     robust=False, actual_obstacles=actual) for actual in scenes[layout]]
            details = training_check["score_details"]
            result = dict(**{k: choice[k] for k in ("family", "layout", "method", "seed", "parameters")},
                manifest_sha256=digest(output / "manifest.json"), selection_sha256=selection_sha,
                stress_sha256=digest(output / "stress_cases.json"), training_check=training_check,
                conditions=len(cases), failures=sum(not x["feasible"] for x in cases),
                all_conditions_pass=all(x["feasible"] for x in cases),
                cases=cases, quality=details["quality_term"], duration_s=details["duration_s"],
                jerk_energy_m2_s5=details["jerk_energy_m2_s5"], deformation_rms_m=details["deformation_rms_m"])
            freeze(path, result)
        if result["manifest_sha256"] != digest(output / "manifest.json") \
                or result["selection_sha256"] != selection_sha \
                or result["stress_sha256"] != digest(output / "stress_cases.json") \
                or any(result[k] != choice[k] for k in ("family", "layout", "method", "seed", "parameters")):
            raise ValueError("Validation cache binding mismatch")
        if len(result["cases"]) != manifest["validation"]["count"] \
                or result["failures"] != sum(not x["feasible"] for x in result["cases"]) \
                or result["all_conditions_pass"] != all(x["feasible"] for x in result["cases"]):
            raise ValueError("Validation count mismatch")
        row = {k: result[k] for k in ("family", "layout", "method", "seed", "quality", "duration_s",
            "conditions", "failures", "all_conditions_pass", "jerk_energy_m2_s5", "deformation_rms_m")}
        row.update(result_file=path.relative_to(output).as_posix(), result_sha256=digest(path))
        rows.append(row)
        print(json.dumps(dict(stage="validate", **{k: row[k] for k in ("family", "layout", "method", "seed", "failures")})), flush=True)
    freeze(output / "validation_summary.json", dict(manifest_sha256=digest(output / "manifest.json"),
        selection_sha256=selection_sha, stress_sha256=digest(output / "stress_cases.json"), rows=rows))


def paired_difference(a, b, field):
    if set(a) != set(b) or not a:
        raise ValueError("Paired seed coverage must match")
    seeds = sorted(a)
    delta = np.array([a[s][field] - b[s][field] for s in seeds])
    bootstrap = np.random.default_rng(20261027).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
    return dict(field=field, seeds=seeds, mean_delta=float(delta.mean()),
        exploratory_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist(),
        wins=int(np.sum(delta < -1e-12)), ties=int(np.sum(abs(delta) <= 1e-12)), losses=int(np.sum(delta > 1e-12)),
        interpretation="negative favors first; paired optimizer seeds within fixed layout; unadjusted for multiplicity")


def summarize(output):
    manifest, selection = load_selection(output)
    summary = read(output / "validation_summary.json")
    for filename in ("manifest", "selection", "stress_cases"):
        if summary[filename.replace("_cases", "") + "_sha256"] != digest(output / (filename + ".json")):
            raise ValueError("Validation summary binding mismatch")
    expected = {(f, l, m, s) for f in manifest["families"] for l in manifest["layouts"]
                for m in manifest["methods"] for s in manifest["seeds"]}
    expected |= {(f, l, "initial", None) for f in manifest["families"] for l in manifest["layouts"]}
    actual = [(r["family"], r["layout"], r["method"], r["seed"]) for r in summary["rows"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Validation summary coverage differs from the frozen design")
    for row in summary["rows"]:
        path = output / row["result_file"]
        if digest(path) != row["result_sha256"]:
            raise ValueError("Validation data changed")
        result = read(path)
        if any(row[k] != result[k] for k in row if k not in ("result_file", "result_sha256")):
            raise ValueError("Validation summary values differ from their result")
        if result["manifest_sha256"] != digest(output / "manifest.json") \
                or result["selection_sha256"] != digest(output / "selection.json") \
                or result["stress_sha256"] != digest(output / "stress_cases.json"):
            raise ValueError("Validation result binding mismatch")
    groups, pairs = [], []
    for family in manifest["families"]:
        for layout in manifest["layouts"]:
            for method in manifest["methods"]:
                rows = [r for r in summary["rows"] if (r["family"], r["layout"], r["method"]) == (family, layout, method)]
                if len(rows) != len(manifest["seeds"]) or {r["seed"] for r in rows} != set(manifest["seeds"]):
                    raise ValueError("Validation method/seed coverage differs")
                runs = [checked_training(output, manifest, family, layout, method, r["seed"])[1] for r in rows]
                groups.append(dict(family=family, layout=layout, method=method, runs=len(rows),
                    full_pass_runs=sum(r["all_conditions_pass"] for r in rows), failed_conditions=sum(r["failures"] for r in rows),
                    mean_quality=float(np.mean([r["quality"] for r in rows])),
                    mean_duration_s=float(np.mean([r["duration_s"] for r in rows])),
                    mean_jerk_energy_m2_s5=float(np.mean([r["jerk_energy_m2_s5"] for r in rows])),
                    mean_search_s=float(np.mean([r["elapsed_s"] for r in runs])),
                    mean_final_training_score=float(np.mean([r["best_score"] for r in runs]))))
            for method in manifest["methods"]:
                if method == "abom" or "abom" not in manifest["methods"]:
                    continue
                a = {r["seed"]: r for r in summary["rows"] if (r["family"], r["layout"], r["method"]) == (family, layout, method)}
                b = {r["seed"]: r for r in summary["rows"] if (r["family"], r["layout"], r["method"]) == (family, layout, "abom")}
                pairs.append(dict(family=family, layout=layout, first=method, second="abom", **paired_difference(a, b, "quality")))
    for layout in manifest["layouts"]:
        for method in manifest["methods"]:
            if not all(f in manifest["families"] for f in ("paired_bump", "single_bump")):
                continue
            a = {r["seed"]: r for r in summary["rows"] if (r["family"], r["layout"], r["method"]) == ("paired_bump", layout, method)}
            b = {r["seed"]: r for r in summary["rows"] if (r["family"], r["layout"], r["method"]) == ("single_bump", layout, method)}
            pairs.append(dict(layout=layout, method=method, first="paired_bump", second="single_bump", **paired_difference(a, b, "quality")))
    report = dict(schema=manifest["schema"], manifest_sha256=digest(output / "manifest.json"),
        validation_summary_sha256=digest(output / "validation_summary.json"), groups=groups, paired_comparisons=pairs,
        training_runs=len(selection["incumbents"]), training_candidates=manifest["budget"] * len(selection["incumbents"]),
        validation_conditions=sum(r["conditions"] for r in summary["rows"]),
        failed_validation_conditions=sum(r["failures"] for r in summary["rows"]),
        initial_rows=[r for r in summary["rows"] if r["method"] == "initial"], scope=manifest["task"]["scope"])
    freeze(output / "summary.json", report)
    lines = ["# 初始曲线参数扰动：冻结实验结果", "", "仅为离线点参考的几何与导数约束；按固定开发布局分层。", "",
        "|曲线自由度|布局|搜索器|运行数|全压力条件通过|平均质量(低好)|平均参考时长(s)|平均搜索耗时(s)|",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for g in groups:
        lines.append(f'|{g["family"]}|{g["layout"]}|{g["method"]}|{g["runs"]}|{g["full_pass_runs"]}|{g["mean_quality"]:.6f}|{g["mean_duration_s"]:.4f}|{g["mean_search_s"]:.4f}|')
    lines += ["", "压力条件均在训练声明的误差包络内；不构成未见布局或真实飞行的泛化证据。",
              "配对区间按优化器种子计算，未校正多重比较；同一参考的条件不作为独立样本。",
              "逐候选、冻结源码、选择和逐条件结果全部保留。AI协助实现、分析和写作。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("training_runs", "training_candidates", "validation_conditions", "failed_validation_conditions")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", choices=FAMILIES, nargs="+", default=list(FAMILIES))
    parser.add_argument("--layouts", nargs="+", choices=("central", "early_offset", "two_obstacles"), default=["central", "early_offset", "two_obstacles"])
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(100, 120)))
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--stress-seed", type=int, default=20261101)
    parser.add_argument("--stress-count", type=int, default=64)
    parser.add_argument("--upstream", type=Path, default=ROOT / "tmp/abom_deploy_probe/upstream")
    args = parser.parse_args()
    for name in ("families", "layouts", "methods", "seeds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            parser.error(f"{name} must be unique")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    output = args.output.resolve()
    if args.stage == "train":
        train(args, output)
    elif args.stage == "validate":
        validate(output)
    else:
        summarize(output)


if __name__ == "__main__":
    main()
