"""Read saved research evidence without importing or rerunning experiment code.

The audit checks arithmetic, recorded provenance and frozen source bytes. A
passing audit is not a flight result, a semantic proof, or a signed attestation.
Historical missing bindings remain explicitly LEGACY_UNBOUND.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "reports/hermite_disturbance_20261027"
DEFAULT_CAMPAIGNS = ("search_pilot", "curve_search", "uncertainty_nominal", "uncertainty_margin")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(
        float(a), float(b), rel_tol=1e-11, abs_tol=1e-12)


def vector_close(a, b):
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all() \
        and bool(np.allclose(left, right, rtol=1e-11, atol=1e-12))


def literal_assignments(nodes):
    """Read literal declarations only; never execute a frozen source module."""
    values = {}
    for node in nodes:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        values.update((name, value) for name in names)
    return values


def frozen_bounds(directory, pilot):
    if pilot:
        path = directory / "source_snapshot/python/dual_target_control/core.py"
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BezierParameters")
        values = literal_assignments(node.body)
        lower, upper = values["_LOWER"], values["_UPPER"]
        default = [values[name] for name in ("outgoing_weight", "initial_handle_ratio",
                                           "approach_handle_ratio", "corridor_fraction", "pass_speed_scale")]
    else:
        path = directory / "source_snapshot/python/trajectory_research/curve_task.py"
        values = literal_assignments(ast.parse(path.read_text(encoding="utf-8-sig")).body)
        lower, upper, default = values["LOWER_BOUNDS"], values["UPPER_BOUNDS"], values["DEFAULT_PARAMETERS"]
    return np.asarray(lower), np.asarray(upper), list(default), str(path)


class Checks:
    def __init__(self):
        self.groups = {}
        self.failures = []
        self.legacy = []

    def check(self, group, passed, context):
        counts = self.groups.setdefault(group, Counter())
        counts["PASS" if bool(passed) else "FAIL"] += 1
        if not passed:
            self.failures.append(dict(check=group, context=context))

    def binding(self, group, data, field, expected, context, *, legacy_allowed=False):
        if field not in data and legacy_allowed:
            self.groups.setdefault(group, Counter())["LEGACY_UNBOUND"] += 1
            self.legacy.append(dict(check=group, field=field, context=context,
                                    status="LEGACY_UNBOUND", note="No retroactive binding was added."))
        else:
            self.check(group, data.get(field) == expected, context)


def validate_saved_score(checks, result, pilot, context, expected_cases=None):
    rows, details = result["cases"], result["score_details"]
    if expected_cases is not None:
        checks.check("validation_case_count", len(rows) == expected_cases, context)
    checks.check("case_flags_match_checks", all(
        isinstance(row["passed"], bool) and row["passed"] == all(row["checks"].values())
        for row in rows), context)
    failed_cases = sum(not row["passed"] for row in rows)
    failed_checks = sum(not item for row in rows for item in row["checks"].values())
    total_checks = sum(len(row["checks"]) for row in rows)
    checks.check("validation_failure_arithmetic", details["failed_cases"] == failed_cases
                 and details["failed_checks"] == failed_checks and details["total_checks"] == total_checks
                 and result["feasible"] == (failed_cases == 0), context)
    quality = details["quality_term"]
    expected = quality if result["feasible"] else 1. + failed_checks / total_checks + quality
    if pilot:
        if not result["feasible"]:
            expected += failed_cases / len(rows)
        time_mean = np.mean([row["elapsed_s"] for row in rows])
        jerk_mean = np.mean([row["maxima"]["command_jerk_mps3"] /
                             (2. * row["control_config"]["max_acceleration"] / min(row["intervals_s"]))
                             for row in rows])
        quality_expected = .99 * time_mean / (10. + time_mean) + .001 * jerk_mean / (1. + jerk_mean)
        checks.check("validation_quality_arithmetic", close(details["mean_elapsed_s"], time_mean)
                     and close(details["mean_normalized_peak_command_jerk"], jerk_mean)
                     and close(quality, quality_expected), context)
    else:
        checks.check("validation_quality_arithmetic", close(quality, details["time_term"]
                     + details["energy_term"] + details["displacement_term"]), context)
    checks.check("validation_score_arithmetic", close(result["score"], expected), context)
    return failed_cases


def audit_campaign(directory):
    checks = Checks()
    manifest = read(directory / "manifest.json")
    selection = read(directory / "selection.json")
    validation = read(directory / "validation_summary.json")
    summary = read(directory / "summary.json")
    manifest_sha, selection_sha = digest(directory / "manifest.json"), digest(directory / "selection.json")
    pilot = manifest["schema"].startswith("hermite-abom")
    legacy = manifest["schema"] == "hermite-abom-offline-pilot/1"
    evidence = {name: digest(directory / name) for name in (
        "manifest.json", "selection.json", "validation_summary.json", "summary.json")}
    checks.check("manifest_unique_axes", len(set(manifest["seeds"])) == len(manifest["seeds"])
                 and len(set(manifest["methods"])) == len(manifest["methods"]), "manifest")
    checks.check("declared_budget_128", manifest["budget"] == 128, "manifest")
    expected_pairs = {(m, s) for m in manifest["methods"] for s in manifest["seeds"]}
    choices = {(row["method"], row["seed"]): row for row in selection["incumbents"]}
    checks.check("selection_coverage", set(choices) == expected_pairs
                 and len(selection["incumbents"]) == len(expected_pairs), "selection")
    checks.check("selection_manifest_binding", selection["manifest_sha256"] == manifest_sha, "selection")
    checks.check("selection_training_only", selection["selection_basis"] == "training_only", "selection")
    source_rows = []
    for relative, expected in manifest["source_sha256"].items():
        path = directory / "source_snapshot" / relative
        observed = digest(path) if path.exists() else None
        checks.check("frozen_source_sha256", observed == expected, relative)
        source_rows.append(dict(source=relative, expected_sha256=expected,
                                snapshot_sha256=observed, status="PASS" if observed == expected else "FAIL"))
    lower, upper, default, bounds_source = frozen_bounds(directory, pilot)
    runs, initial_by_seed = {}, {}
    all_train_files = {p.name for p in directory.glob("train_*.json")}
    checks.check("training_file_coverage", all_train_files == {c["training_file"] for c in choices.values()}, "directory")
    for pair, choice in sorted(choices.items()):
        name = choice["training_file"]
        result = read(directory / name)
        runs[pair] = result
        evidence[name] = digest(directory / name)
        checks.check("selection_training_hash", evidence[name] == choice["training_sha256"], name)
        checks.binding("training_manifest_binding", result, "manifest_sha256", manifest_sha,
                       name, legacy_allowed=legacy)
        checks.check("selected_incumbent_binding", result["method"] == pair[0] and result["seed"] == pair[1]
                     and result["best_parameters"] == choice["best_parameters"]
                     and close(result["best_score"], choice["training_score"]), name)
        rows, history = result["evaluations"], result["history"]
        budget, population = manifest["budget"], manifest["population"]
        checks.check("run_budget_128", result["budget"] == budget == len(rows) == 128
                     and result["population"] == population, name)
        checks.check("fe_sequence", [row["fe"] for row in rows] == list(range(1, budget + 1))
                     and all(row["generation"] == index // population for index, row in enumerate(rows)), name)
        scores = [row["score"] for row in rows]
        checks.check("evaluation_finite_and_typed", all(
            math.isfinite(row["score"]) and isinstance(row["feasible"], bool)
            and math.isfinite(row["evaluation_s"]) and row["evaluation_s"] >= 0 for row in rows), name)
        params = np.asarray([row["parameters"] for row in rows])
        checks.check("physical_parameter_bounds", params.shape == (budget, len(lower))
                     and np.isfinite(params).all() and np.all(params >= lower - 1e-12)
                     and np.all(params <= upper + 1e-12), name)
        best = min(scores)
        checks.check("incumbent_is_evaluated_minimum", close(result["best_score"], best)
                     and any(close(row["score"], best) and vector_close(row["parameters"], result["best_parameters"])
                             for row in rows), name)
        checks.check("history_prefix_minima", [row["fe"] for row in history] == list(range(population, budget + 1, population))
                     and all(close(row["best"], min(scores[:row["fe"]])) for row in history), name)
        checks.check("history_monotonic", all(b["best"] <= a["best"] + 1e-12
                     for a, b in zip(history, history[1:])), name)
        initial = np.asarray(result["initial_population_normalized"])
        checks.check("initial_population_bounds", initial.shape == (population, len(lower))
                     and np.isfinite(initial).all() and np.all(np.abs(initial) <= 1. + 1e-12), name)
        checks.check("initial_population_physical_mapping", vector_close(params[:population],
                     lower + (initial + 1.) * .5 * (upper - lower)), name)
        seed = pair[1]
        if seed in initial_by_seed:
            checks.check("shared_initial_population_per_seed", np.array_equal(initial_by_seed[seed], initial), name)
        else:
            initial_by_seed[seed] = initial
        checks.check("training_failure_flags", all(row["feasible"] == (row["details"]["failed_cases"] == 0)
                     for row in rows), name)
        for row in rows:
            details = row["details"]
            expected = details["quality_term"] if row["feasible"] else 1. + details["failed_checks"] / details["total_checks"] + details["quality_term"]
            if pilot and not row["feasible"]:
                expected += details["failed_cases"] / len(manifest["training_cases"])
            checks.check("training_score_arithmetic", close(row["score"], expected), f"{name}:FE{row['fe']}")
    checks.binding("validation_summary_manifest_binding", validation, "manifest_sha256", manifest_sha,
                   "validation_summary", legacy_allowed=legacy)
    checks.check("validation_summary_selection_binding", validation["selection_sha256"] == selection_sha,
                 "validation_summary")
    stress = read(directory / "stress_cases.json") if (directory / "stress_cases.json").exists() else None
    if stress:
        evidence["stress_cases.json"] = digest(directory / "stress_cases.json")
        checks.check("stress_case_selection_binding", stress["selection_sha256"] == selection_sha, "stress_cases")
    declared_cases = stress["cases"] if stress else manifest["validation_cases"]
    val_rows = {(row["method"], row["seed"]): row for row in validation["rows"]}
    expected_validation_pairs = expected_pairs if pilot else expected_pairs | {("default", None)}
    checks.check("validation_coverage_failure_retention", set(val_rows) == expected_validation_pairs
                 and len(validation["rows"]) == len(expected_validation_pairs), "validation_summary")
    val_results = {}
    for pair, row in val_rows.items():
        name = row["result_file"]
        result = read(directory / name)
        val_results[pair] = result
        evidence[name] = digest(directory / name)
        if "result_sha256" in row:
            checks.check("validation_result_hash", row["result_sha256"] == evidence[name], name)
        else:
            checks.legacy.append(dict(check="validation_result_hash", context=name,
                                      status="LEGACY_UNBOUND", note="v1 pilot records parameters and selection hash, not a result SHA."))
            checks.groups.setdefault("validation_result_hash", Counter())["LEGACY_UNBOUND"] += 1
        checks.binding("validation_manifest_binding", result, "manifest_sha256", manifest_sha,
                       name, legacy_allowed=legacy)
        checks.check("validation_selection_binding", result["selection_sha256"] == selection_sha, name)
        expected_parameters = default if pair[0] == "default" else choices[pair]["best_parameters"]
        checks.check("validation_selected_parameter_binding", vector_close(result["parameters"], expected_parameters), name)
        failed = validate_saved_score(checks, result, pilot, name, len(declared_cases))
        details = result["score_details"]
        if pilot:
            passed = (close(row["training_score"], choices[pair]["training_score"])
                      and close(row["validation_score"], result["score"])
                      and close(row["elapsed_mean_s"], details["mean_elapsed_s"])
                      and close(row["jerk_proxy"], details["mean_normalized_peak_command_jerk"])
                      and row["held_out_seed_cases"] == 6
                      and row["held_out_seed_cases_passed"] == sum(c["passed"] for c in result["cases"][-6:]))
            checks.check("validation_condition_identity", all(
                saved["scenario"] == declared["scenario"]["name"]
                and vector_close(saved["targets"], declared["scenario"]["targets"])
                and close(saved["velocity_lag_s"], declared["velocity_lag_s"])
                and close(saved["yaw_lag_s"], declared["yaw_lag_s"])
                for saved, declared in zip(result["cases"], declared_cases)), name)
        else:
            passed = close(row["score"], result["score"]) \
                and close(row["reference_duration_s"], details["duration_s"]) \
                and close(row["jerk_energy_m2_s5"], details["jerk_energy_m2_s5"])
            checks.check("validation_condition_identity", all(
                saved["name"] == declared["name"]
                and vector_close(saved["centre_m"], declared["centre_m"])
                and close(saved["radius_m"], declared["radius_m"])
                and close(saved["clearance_m"], declared["clearance_m"])
                for saved, declared in zip(result["cases"], declared_cases)), name)
        checks.check("validation_row_arithmetic", passed and row["failed_cases"] == failed
                     and row["feasible"] == result["feasible"] and row["cases"] == len(result["cases"]), name)
    if pilot:
        name = "validation_default.json"
        baseline = read(directory / name)
        evidence[name] = digest(directory / name)
        checks.binding("baseline_manifest_binding", baseline, "manifest_sha256", manifest_sha,
                       name, legacy_allowed=legacy)
        checks.binding("baseline_selection_binding", baseline, "selection_sha256", selection_sha,
                       name, legacy_allowed=legacy)
        checks.check("baseline_parameter_identity", vector_close(baseline["parameters"], default), name)
        validate_saved_score(checks, baseline, pilot, name, len(declared_cases))
        expected_baseline = dict(feasible=baseline["feasible"], score=baseline["score"], **baseline["score_details"])
    else:
        expected_baseline = val_rows[("default", None)]
    checks.check("summary_baseline_arithmetic", summary["baseline"] == expected_baseline, "summary")
    for method in manifest["methods"]:
        pairs = [(method, seed) for seed in manifest["seeds"]]
        selected_rows = [val_rows[pair] for pair in pairs]
        training_runs = [runs[pair] for pair in pairs]
        if pilot:
            expected = dict(seeds=len(pairs), all_cases_feasible_seeds=sum(r["feasible"] for r in selected_rows),
                            failed_cases=sum(r["failed_cases"] for r in selected_rows),
                            mean_validation_time_s=float(np.mean([r["elapsed_mean_s"] for r in selected_rows])),
                            mean_training_score=float(np.mean([r["best_score"] for r in training_runs])),
                            mean_search_wall_s=float(np.mean([r["elapsed_s"] for r in training_runs])),
                            total_training_candidate_evaluations=sum(len(r["evaluations"]) for r in training_runs))
        else:
            expected = dict(seeds=len(pairs), feasible_all_conditions_seeds=sum(r["feasible"] for r in selected_rows),
                            failed_conditions=sum(r["failed_cases"] for r in selected_rows),
                            mean_training_score=float(np.mean([r["best_score"] for r in training_runs])),
                            mean_validation_score=float(np.mean([r["score"] for r in selected_rows])),
                            mean_reference_duration_s=float(np.mean([r["reference_duration_s"] for r in selected_rows])),
                            mean_search_s=float(np.mean([r["elapsed_s"] for r in training_runs])))
        checks.check("method_summary_arithmetic_failure_retention", set(summary["methods"][method]) == set(expected)
                     and all(close(summary["methods"][method][key], value) for key, value in expected.items()), method)
    training_count = sum(len(r["evaluations"]) for r in runs.values())
    validation_count = len(val_rows) + int(pilot)
    validation_conditions = sum(len(r["cases"]) for r in val_results.values()) + (len(baseline["cases"]) if pilot else 0)
    expected_budget = dict(training_candidates=training_count,
                           training_rollout_conditions=training_count * len(manifest["training_cases"]),
                           validation_candidates=validation_count, validation_rollout_conditions=validation_conditions)
    if pilot:
        checks.check("summary_budget_arithmetic", summary["budget_accounting"] == expected_budget, "summary")
    else:
        expected_budget = dict(training_candidates=training_count,
                               training_sphere_conditions=training_count * len(manifest["training_cases"]),
                               validation_candidates=validation_count, validation_sphere_conditions=validation_conditions)
        checks.check("summary_budget_arithmetic", all(summary[key] == value for key, value in expected_budget.items()), "summary")
    comparisons = summary["paired_time_comparisons"] if pilot else summary["paired_comparisons"]
    for comparison in comparisons:
        a_method = "abom_perturb" if pilot else comparison["method"]
        b_method = comparison["comparator"]
        seeds = sorted(manifest["seeds"])
        field = "elapsed_mean_s" if pilot else "score"
        delta = np.asarray([val_rows[(a_method, seed)][field] - val_rows[(b_method, seed)][field] for seed in seeds])
        bootstrap = np.random.default_rng(20261027).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
        mean_key = "mean_delta_time_s" if pilot else "mean_delta_validation_score"
        ci_key = "bootstrap_95_percentile_ci_s" if pilot else "exploratory_bootstrap_95_ci"
        seed_key = "paired_seeds" if pilot else "seeds"
        good = comparison[seed_key] == seeds and close(comparison[mean_key], delta.mean()) \
            and vector_close(comparison[ci_key], np.quantile(bootstrap, [.025, .975]))
        if not pilot:
            good = good and comparison["wins"] == int(np.sum(delta < 0)) \
                and comparison["ties"] == int(np.sum(delta == 0)) and comparison["losses"] == int(np.sum(delta > 0))
        checks.check("paired_comparison_arithmetic", good, f"{a_method} vs {b_method}")
    status = "FAIL" if checks.failures else ("PASS_WITH_LEGACY_UNBOUND" if checks.legacy else "PASS")
    return dict(campaign=directory.name, directory=str(directory.resolve()), status=status,
                schema=manifest["schema"], profile=manifest.get("profile", "legacy"),
                optimizer_runs=len(runs), seeds=len(manifest["seeds"]), methods=len(manifest["methods"]),
                training_candidate_evaluations=training_count,
                training_condition_evaluations=training_count * len(manifest["training_cases"]),
                validation_candidates=validation_count, validation_condition_evaluations=validation_conditions,
                validation_failed_conditions=sum(r["score_details"]["failed_cases"] for r in val_results.values())
                + (baseline["score_details"]["failed_cases"] if pilot else 0),
                bounds=dict(lower=lower.tolist(), upper=upper.tolist(), default=default, source=bounds_source),
                checks={key: dict(value) for key, value in checks.groups.items()},
                failures=checks.failures, legacy_unbound=checks.legacy,
                source_snapshots=source_rows, evidence_sha256=evidence)


def render_markdown(report):
    lines = ["# Hermite 与扰动研究证据核查", "", f"总体状态：**{report['status']}**。", "",
             "本核查只读保存的逐候选、选择、验证、摘要及源码快照，不重跑实验，不改写旧证据。", "",
             "|目录|状态|训练运行|候选评价|验证条件评价|保留的失败条件|",
             "|---|---|---:|---:|---:|---:|"]
    for row in report["campaigns"]:
        lines.append(f"|{row['campaign']}|{row['status']}|{row.get('optimizer_runs', 0)}|"
                     f"{row.get('training_candidate_evaluations', 0)}|{row.get('validation_condition_evaluations', 0)}|"
                     f"{row.get('validation_failed_conditions', 0)}|")
    lines += ["", "覆盖：128 候选预算、逐 FE 顺序、历史前缀最小值、同种子共享初始化、已评价最优解、"
              "冻结边界、选择与训练绑定、验证参数与条件对应、失败保留、摘要及配对区间算术、源码快照 SHA-256。", ""]
    for row in report["campaigns"]:
        legacy = row.get("legacy_unbound", [])
        if legacy:
            counts = Counter(item["check"] for item in legacy)
            lines += [f"{row['campaign']} 的历史缺失绑定：" + "；".join(f"{key} × {count}" for key, count in counts.items()) + "。", ""]
        if row.get("failures"):
            lines += [f"{row['campaign']} 的失败：", ""]
            lines.extend(f"- {item['check']}：{item['context']}" for item in row["failures"][:30])
            lines.append("")
    lines += ["`LEGACY_UNBOUND` 表示原文件没有相应哈希字段，不能补称已有绑定。其他可核查字段仍分别检查。", "",
              "源码快照与 manifest 哈希相符，验证的是原记录对应的字节。后补恢复的历史快照不代表当时已经保存，"
              "哈希一致性也不能证明运行发生时间或训练/验证的真实先后顺序。", "",
              "独立单位为优化器种子，同一参考的多个障碍条件不是独立飞行重复。失败计入全部条件汇总。"
              "本核查不证明算法优越性、物理安全、抗风或实飞效果。", "",
              "AI 辅助编写和执行核查，详细检查计数、每个文件哈希及历史缺项见配套 JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--campaigns", nargs="+", default=list(DEFAULT_CAMPAIGNS))
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    rows = []
    for name in args.campaigns:
        try:
            row = audit_campaign(args.base / name)
        except Exception as exc:
            row = dict(campaign=name, status="FAIL", failures=[dict(check="audit_execution", context=f"{type(exc).__name__}: {exc}")])
        rows.append(row)
        print(json.dumps({key: row[key] for key in ("campaign", "status")} | {
            "failures": len(row.get("failures", [])), "legacy_unbound": len(row.get("legacy_unbound", []))}), flush=True)
    status = "FAIL" if any(row["status"] == "FAIL" for row in rows) else (
        "PASS_WITH_LEGACY_UNBOUND" if any(row["status"] == "PASS_WITH_LEGACY_UNBOUND" for row in rows) else "PASS")
    report = dict(schema="hermite-research-integrity-audit/1", status=status,
                  generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  audit_script=str(Path(__file__).resolve()), audit_script_sha256=digest(__file__),
                  audit_mode="read saved evidence; no experiment execution or historical artifact mutation",
                  tolerance=dict(relative=1e-11, absolute=1e-12), campaigns=rows)
    prefix = args.output_prefix or args.base / "integrity_audit"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return int(status == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
