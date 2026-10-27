"""Describe training-budget sensitivity from completed frozen curve campaigns.

Each prefix chooses the first minimum-score evaluation using training only.
Repeated prefixes share one optimization run and are not independent samples.
No validation result is opened, generated or used for selection by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "initial-curve-parameter-perturbation/1"
METRICS = ("score", "quality", "duration_s", "jerk_energy_m2_s5", "deformation_rms_m")
BOOTSTRAP_SEED = 20261027


def read_evidence(path):
    data = Path(path).read_bytes()
    return json.loads(data.decode("utf-8")), hashlib.sha256(data).hexdigest()


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Evidence path leaves the campaign directory")
    return path


def _key(family, layout, method, seed):
    return f"{family}__{layout}__{method}__s{seed}"


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def load_completed_campaign(directory):
    """Check design/selection/training bindings without reading validation."""
    directory = Path(directory).resolve()
    manifest, manifest_sha = read_evidence(directory / "manifest.json")
    selection, selection_sha = read_evidence(directory / "selection.json")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported campaign schema for parameter-budget analysis")
    if selection.get("manifest_sha256") != manifest_sha or selection.get("selection_basis") != "training_only":
        raise ValueError("Selection is not bound to the frozen training-only design")
    for field in ("families", "layouts", "methods", "seeds"):
        values = manifest[field]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Empty or duplicate design field: {field}")
    population, budget = manifest["population"], manifest["budget"]
    if isinstance(population, bool) or not isinstance(population, int) or population < 4 \
            or population % 2 or isinstance(budget, bool) or not isinstance(budget, int) \
            or budget < population or budget % population:
        raise ValueError("Invalid frozen population or budget")
    expected = {(f, l, m, s) for f in manifest["families"] for l in manifest["layouts"]
                for m in manifest["methods"] for s in manifest["seeds"]}
    actual = [(r["family"], r["layout"], r["method"], r["seed"]) for r in selection["incumbents"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Complete unique selection coverage is required")
    snapshot_sha = {}
    for relative, expected_sha in manifest["source_sha256"].items():
        path = _inside(directory / "source_snapshot", relative)
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"Source snapshot hash mismatch: {relative}")
        snapshot_sha[relative] = actual_sha
    runs, train_sha, initials = [], {}, {}
    for choice in sorted(selection["incumbents"], key=lambda r:
                         (r["family"], r["layout"], r["method"], r["seed"])):
        identity = {k: choice[k] for k in ("family", "layout", "method", "seed")}
        expected_file = "train/" + _key(**identity) + ".json"
        if choice["training_file"] != expected_file:
            raise ValueError("Training file does not match the selected identity")
        path = _inside(directory, expected_file)
        run, run_sha = read_evidence(path)
        if choice["training_sha256"] != run_sha:
            raise ValueError("Selected training file hash mismatch")
        if any(run.get(k) != value for k, value in identity.items()) \
                or run.get("manifest_sha256") != manifest_sha \
                or run.get("budget") != budget or run.get("population") != population:
            raise ValueError("Training metadata differs from the frozen design")
        evaluations = run["evaluations"]
        if len(evaluations) != budget or [row["fe"] for row in evaluations] != list(range(1, budget + 1)):
            raise ValueError("Training FE count/order differs from the frozen budget")
        if any(not _finite_number(row["score"]) or not isinstance(row["feasible"], bool) for row in evaluations):
            raise ValueError("Invalid finite score or boolean feasibility in training ledger")
        bounds = manifest["variables"][identity["family"]]
        lower, upper = np.asarray(bounds["lower"], float), np.asarray(bounds["upper"], float)
        points = np.asarray([row["parameters"] for row in evaluations], float)
        if lower.ndim != 1 or not len(lower) or upper.shape != lower.shape or not np.isfinite(lower).all() \
                or not np.isfinite(upper).all() or np.any(upper <= lower) \
                or points.shape != (budget, len(lower)) or not np.isfinite(points).all() \
                or np.any(points < lower) or np.any(points > upper):
            raise ValueError("Training points differ from the declared finite parameter box")
        minimum = min(row["score"] for row in evaluations)
        if run["best_score"] != minimum or not any(row["score"] == minimum and
                row["parameters"] == run["best_parameters"] for row in evaluations):
            raise ValueError("Selected full-run incumbent is not an evaluated minimum")
        if choice["parameters"] != run["best_parameters"] or choice["training_score"] != run["best_score"]:
            raise ValueError("Selection parameters/score differ from the training incumbent")
        initial = np.asarray(run["initial_population_normalized"], float)
        reconstructed = 2 * (points[:population] - lower) / (upper - lower) - 1
        if initial.shape != (population, len(lower)) or not np.isfinite(initial).all() \
                or not np.allclose(initial, reconstructed, rtol=1e-12, atol=1e-12):
            raise ValueError("Initial population does not match initial evaluations")
        initial_key = (identity["family"], identity["layout"], identity["seed"])
        if initial_key in initials and run["initial_population_normalized"] != initials[initial_key]:
            raise ValueError("Methods do not share the same initial population")
        initials[initial_key] = run["initial_population_normalized"]
        runs.append(run)
        train_sha[expected_file] = run_sha
    provenance = dict(campaign_directory=str(directory), manifest_sha256=manifest_sha,
        selection_sha256=selection_sha, source_snapshot_sha256=snapshot_sha,
        training_sha256=train_sha, validation_access="none")
    return manifest, runs, provenance


def prefix_incumbents(manifest, runs, budgets, *, missing_budget="skip"):
    """Return one stable training-only incumbent per run/requested FE prefix."""
    if missing_budget not in ("skip", "error") or not budgets or len(set(budgets)) != len(budgets):
        raise ValueError("Unique nonempty budgets and a valid missing-budget policy are required")
    population, full_budget = manifest["population"], manifest["budget"]
    for budget in budgets:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < population or budget % population:
            raise ValueError("Prefix budgets must be whole generations including the initial population")
    unavailable = [b for b in sorted(budgets) if b > full_budget]
    if unavailable and missing_budget == "error":
        raise ValueError(f"Requested prefixes exceed the executed budget {full_budget}: {unavailable}")
    available = [b for b in sorted(budgets) if b <= full_budget]
    if not available:
        raise ValueError("No requested prefix is present in the completed campaign")
    rows = []
    for run in runs:
        for budget in available:
            prefix = run["evaluations"][:budget]
            chosen = min(prefix, key=lambda x: x["score"])  # Python min keeps the first tie.
            details = chosen["details"]
            selected = dict(score=chosen["score"], quality=details["quality_term"],
                duration_s=details["duration_s"], jerk_energy_m2_s5=details["jerk_energy_m2_s5"],
                deformation_rms_m=details["deformation_rms_m"])
            if any(not _finite_number(value) for value in selected.values()) \
                    or selected["duration_s"] <= 0 or selected["jerk_energy_m2_s5"] < 0 \
                    or selected["deformation_rms_m"] < 0 \
                    or selected["duration_s"] != chosen["parameters"][-1]:
                raise ValueError("Invalid chosen-prefix metrics or duration/parameter mismatch")
            rows.append(dict(**{k: run[k] for k in ("family", "layout", "method", "seed")},
                budget=budget, selected_fe=chosen["fe"], training_feasible=chosen["feasible"],
                parameters=chosen["parameters"], **selected))
    skipped = [dict(budget=b, reason=f"executed full-run budget is {full_budget}") for b in unavailable]
    return rows, skipped


def descriptive(values):
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty metric sample required")
    q1, median, q3 = np.quantile(x, [.25, .5, .75])
    return dict(mean=float(np.mean(x)), median=float(median), q1=float(q1), q3=float(q3), iqr=float(q3 - q1))


def paired_change(first, second, field):
    if not first or set(first) != set(second):
        raise ValueError("Paired comparisons require identical nonempty optimizer-seed coverage")
    seeds = sorted(first)
    delta = np.array([first[seed][field] - second[seed][field] for seed in seeds])
    bootstrap = np.random.default_rng(BOOTSTRAP_SEED).choice(delta, (10000, len(delta)), replace=True).mean(axis=1)
    return dict(field=field, seeds=seeds, mean_delta=float(np.mean(delta)),
                median_delta=float(np.median(delta)),
                exploratory_bootstrap_95_ci=np.quantile(bootstrap, [.025, .975]).tolist(),
                wins=int(np.sum(delta < -1e-12)), ties=int(np.sum(np.abs(delta) <= 1e-12)),
                losses=int(np.sum(delta > 1e-12)))


def analyze(directory, budgets=(128, 256, 512), *, missing_budget="skip"):
    manifest, runs, provenance = load_completed_campaign(directory)
    rows, skipped = prefix_incumbents(manifest, runs, list(budgets), missing_budget=missing_budget)
    available = sorted({r["budget"] for r in rows})
    groups, versus_abom, adjacent = [], [], []
    for family in manifest["families"]:
        for layout in manifest["layouts"]:
            for method in manifest["methods"]:
                selected = [r for r in rows if (r["family"], r["layout"], r["method"]) == (family, layout, method)]
                by_budget = {budget: {r["seed"]: r for r in selected if r["budget"] == budget}
                             for budget in available}
                for budget in available:
                    sample = list(by_budget[budget].values())
                    if len(sample) != len(manifest["seeds"]) or set(by_budget[budget]) != set(manifest["seeds"]):
                        raise ValueError("Prefix group does not cover the full frozen optimizer seeds")
                    groups.append(dict(family=family, layout=layout, method=method, budget=budget,
                        runs=len(sample), training_feasible_runs=sum(r["training_feasible"] for r in sample),
                        metrics={field: descriptive([r[field] for r in sample]) for field in METRICS}))
                    if method != "abom" and "abom" in manifest["methods"]:
                        reference = {r["seed"]: r for r in rows if
                            (r["family"], r["layout"], r["method"], r["budget"]) == (family, layout, "abom", budget)}
                        for field in METRICS:
                            versus_abom.append(dict(family=family, layout=layout, first_method=method,
                                second_method="abom", budget=budget, **paired_change(by_budget[budget], reference, field)))
                for previous, current in zip(available, available[1:]):
                    for field in METRICS:
                        change = paired_change(by_budget[current], by_budget[previous], field)
                        if field == "score" and change["losses"]:
                            raise ValueError("A later-prefix minimum cannot have a larger training score")
                        adjacent.append(dict(family=family, layout=layout, method=method,
                            first_budget=current, second_budget=previous, **change))
    return dict(schema="parameter-search-budget-analysis/1", provenance=provenance,
        frozen_full_budget=manifest["budget"], requested_budgets=list(budgets), skipped_prefixes=skipped,
        groups=groups, prefix_incumbents=rows, comparisons_versus_abom=versus_abom,
        adjacent_budget_changes=adjacent,
        protocol=dict(selection="first minimum raw training score in each FE prefix; stable ties",
            independent_unit="optimizer seed within each fixed development family/layout",
            prefix_dependence="multiple budgets are repeated prefixes of the same run, not independent runs",
            bootstrap=dict(draws=10000, seed=BOOTSTRAP_SEED, correction="none; exploratory"),
            difference_sign="first minus second; negative score/quality/duration/jerk is smaller",
            feasibility="all prefix incumbents included, including infeasible ones; feasibility counts reported",
            validation_claim="none; no prefix incumbent was revalidated by this analysis"))


def write_outputs(report, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "budget_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (destination / "prefix_incumbents.csv").open("w", newline="", encoding="utf-8") as stream:
        names = ["family", "layout", "method", "seed", "budget", "selected_fe", "training_feasible", *METRICS, "parameters"]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in report["prefix_incumbents"]:
            writer.writerow(dict(row, parameters=json.dumps(row["parameters"])))
    with (destination / "budget_groups.csv").open("w", newline="", encoding="utf-8") as stream:
        names = ["family", "layout", "method", "budget", "runs", "training_feasible_runs"]
        names += [f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "median", "q1", "q3", "iqr")]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for group in report["groups"]:
            flat = {k: group[k] for k in names[:6]}
            flat.update({f"{metric}_{stat}": value for metric, stats in group["metrics"].items()
                         for stat, value in stats.items()})
            writer.writerow(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--missing-budget", choices=("skip", "error"), default="skip")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.campaign, args.budgets, missing_budget=args.missing_budget)
    report["provenance"]["analysis_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_outputs(report, args.destination)
    print(json.dumps(dict(groups=len(report["groups"]), prefixes=len(report["prefix_incumbents"]),
        skipped=report["skipped_prefixes"], validation_access="none"), indent=2))


if __name__ == "__main__":
    main()
