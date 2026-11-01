"""Equal-evaluation pilot of diagnostic feedback for a fixed timed tracker.

The tracker/plant are independent research point-mass models. They are not
production RollingController, AirSim, rotor dynamics, or real-flight evidence.
Validation lags are used only after each search has selected its reference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from trajectory_research.blackbox import run_blackbox_search
from trajectory_research.feedback_search import run_feedback_search
from trajectory_research.parameterizations import default_parameters, parameter_bounds
from trajectory_research.perturbation_task import get_layout
from trajectory_research.search import UPSTREAM_COMMIT, UPSTREAM_NETWORK_BLOB, run_search
from trajectory_research.timed_control_task import evaluate_timed_parameters
from trajectory_research.timed_tracking import TrackingCondition


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("scalar", "diagnostic", "zero", "shuffled", "gaussian", "coordinate")
NEURAL_MODES = ("scalar", "diagnostic", "zero", "shuffled")
TRAIN_CONDITION = TrackingCondition(velocity_lag_s=.30, position_gain=2.)
VALIDATION_LAGS_S = (.15, .45)


def write_json(path, value):
    """Keep the latest complete ledger if the experiment is interrupted."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def validate_args(args):
    if args.budget < 16 or args.budget % 16:
        raise ValueError("budget must be a positive multiple of population=16")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("distinct nonnegative seeds are required")
    if not args.methods or len(set(args.methods)) != len(args.methods) \
            or any(method not in METHODS for method in args.methods):
        raise ValueError("distinct supported methods are required")
    if not np.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("dropout must lie in [0,1)")
    for values, label in ((args.families, "families"), (args.layouts, "layouts")):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"distinct {label} are required")
    for family in args.families:
        parameter_bounds(family)
    for layout in args.layouts:
        get_layout(layout)


def make_plan(args):
    """A frozen seeded base order, then cyclic rotations between blocks."""
    base = np.random.default_rng(20261029).permutation(args.methods).tolist()
    plan = []
    for family in args.families:
        for layout in args.layouts:
            for seed in args.seeds:
                shift = len(plan) % len(base)
                plan.append(dict(family=family, layout=layout, seed=seed,
                                 method_order=base[shift:] + base[:shift]))
    return plan


def summaries(rows, args):
    result, paired = [], []
    for family in args.families:
        for layout in args.layouts:
            condition = [r for r in rows if (r["family"], r["layout"]) == (family, layout)]
            for method in args.methods:
                group = [r for r in condition if r["method"] == method]
                assert len(group) == len(args.seeds)
                feasible = [r for r in group if r["feasible"]]
                result.append(dict(family=family, layout=layout, method=method, n=len(group),
                    mean_selected_score=float(np.mean([r["best_score"] for r in group])),
                    median_selected_score=float(np.median([r["best_score"] for r in group])),
                    feasible_selected=len(feasible),
                    mean_feasible_selected_score=float(np.mean([r["best_score"] for r in feasible])) if feasible else None,
                    median_wall_s=float(np.median([r["wall_s"] for r in group])),
                    median_cpu_s=float(np.median([r["cpu_s"] for r in group])),
                    mean_reference_duration_s=float(np.mean([r["reference_duration_s"] for r in group])),
                    mean_tracking_rmse_m=float(np.mean([r["tracking_rmse_m"] for r in group])),
                    total_failed_training_candidates=sum(r["failed_training_candidates"] for r in group),
                    validation_condition_passes=sum(r["validation_passes"] for r in group),
                    validation_condition_count=len(group) * len(VALIDATION_LAGS_S)))
            if "diagnostic" not in args.methods:
                continue
            diagnostic = {r["seed"]: r for r in condition if r["method"] == "diagnostic"}
            for comparator in (m for m in args.methods if m != "diagnostic"):
                comparison = {r["seed"]: r for r in condition if r["method"] == comparator}
                delta = np.array([diagnostic[s]["best_score"] - comparison[s]["best_score"]
                                  for s in args.seeds])
                interval = None
                if len(delta) >= 5:
                    rng = np.random.default_rng(20261029)
                    bootstrap = np.mean(delta[rng.integers(0, len(delta), (5000, len(delta)))], axis=1)
                    interval = np.quantile(bootstrap, [.025, .975]).tolist()
                paired.append(dict(family=family, layout=layout, method="diagnostic", comparator=comparator,
                    n=len(delta), mean_paired_score_difference=float(np.mean(delta)),
                    better_runs=int(np.sum(delta < 0)), equal_runs=int(np.sum(delta == 0)),
                    worse_runs=int(np.sum(delta > 0)), exploratory_paired_bootstrap95=interval,
                    interpretation="negative favors diagnostic; n<5 reports descriptive pilot only; fixed development geometry and declared model lags"))
    return result, paired


def run(args):
    validate_args(args)
    neural = any(method in NEURAL_MODES for method in args.methods)
    sources = [Path(__file__)] + [ROOT / "python/trajectory_research" / name for name in (
        "feedback_search.py", "timed_control_task.py", "timed_tracking.py", "blackbox.py",
        "search.py", "parameterizations.py", "perturbation_task.py", "curves.py",
        "curve_task.py", "constraints.py", "directional.py", "quality_bound.py")]
    hashes = {source.relative_to(ROOT).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()
              for source in sources}
    upstream = None
    if neural:
        source = Path(args.upstream).resolve() / "abom_net.py"
        data = source.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != UPSTREAM_NETWORK_BLOB:
            raise ValueError("pinned upstream network Git blob differs")
        upstream = dict(commit=UPSTREAM_COMMIT, network_git_blob=blob,
                        network_sha256=hashlib.sha256(data).hexdigest())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    plan = make_plan(args)
    manifest = dict(protocol="independent-timed-control-feedback-pilot-v1", args=vars(args),
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_sha256=hashes, upstream=upstream, execution_plan=plan,
        expected_runs=len(plan) * len(args.methods), population=16,
        training_condition=asdict(TRAIN_CONDITION), validation_lags_s=list(VALIDATION_LAGS_S),
        initialization="uniform; shared physical anchor; identical normalized populations verified",
        screening_enabled=False, rank_encoding="neural modes use population rank divided by population-1",
        capacity_control="diagnostic/zero/shuffled share six-input architecture; scalar is one-input comparator",
        dropout_choice="fixed .1 default inherited from geometry calibration; not asserted optimal for control task",
        primary="equal complete-evaluation budget: selected score, feasibility, wall/CPU time and synchronous tracking diagnostics",
        first_feasible_time="not a primary claim; shared initialization includes an anchor, whose feasibility is recorded",
        timer="warm Python process; optimizer initialization, evaluation, feature handling and network work included; warmup, selected rechecks, validation and serialization excluded",
        order="seeded base method permutation with deterministic block rotations; incomplete cycles are not fully position-balanced",
        scope="independent fixed velocity-command tracker and first-order lagged point mass; not production RollingController, AirSim, rotor/attitude/thrust model, or real flight",
        inference="pilot when using defaults; known development layouts, declared uncalibrated model variations, no first/novelty/generalization claim")
    write_json(output / "manifest.json", manifest)
    if neural:
        warm_start = time.perf_counter()
        import torch
        torch.set_num_threads(1)
        torch.zeros(1).sum()
        run_feedback_search(lambda x: dict(score=float(x @ x), feasible=True,
                            score_details=dict(control_features=[0.] * 5)),
            np.full(4, -1.), np.full(4, 1.), mode="diagnostic", budget=32,
            upstream=args.upstream, dropout=args.dropout)
        manifest["torch_version"] = torch.__version__
        manifest["excluded_import_optimizer_warmup_s"] = time.perf_counter() - warm_start
        manifest["synthetic_warmup_calls"] = 32
        write_json(output / "manifest.json", manifest)

    rows, anchors, warmed = [], [], set()
    for block in plan:
        family, layout, seed = block["family"], block["layout"], block["seed"]
        lower, upper = parameter_bounds(family)
        anchor = default_parameters(family)
        if (family, layout) not in warmed:
            tick = time.perf_counter()
            checked = evaluate_timed_parameters(anchor, family=family, layout=layout,
                                                condition=TRAIN_CONDITION)
            anchors.append(dict(family=family, layout=layout, parameters=anchor.tolist(),
                                excluded_warmup_wall_s=time.perf_counter() - tick, result=checked))
            write_json(output / "anchor_checks.json", anchors)
            warmed.add((family, layout))
        initial_population = None
        equal_capacity_model = None
        for position, method in enumerate(block["method_order"]):
            calls = 0

            def objective(parameters):
                nonlocal calls
                calls += 1
                return evaluate_timed_parameters(parameters, family=family, layout=layout,
                                                 condition=TRAIN_CONDITION)

            options = dict(seed=seed, population=16, budget=args.budget, anchor=anchor,
                           initialization="uniform")
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            if method in NEURAL_MODES:
                search = run_feedback_search(objective, lower, upper, mode=method,
                    upstream=args.upstream, dropout=args.dropout, **options)
            elif method == "gaussian":
                search = run_search(objective, lower, upper, method="gaussian", **options)
            else:
                search = run_blackbox_search(objective, lower, upper, method="coordinate", **options)
            wall_s, cpu_s = time.perf_counter() - wall_start, time.process_time() - cpu_start
            assert calls == len(search["evaluations"]) == args.budget, "unexpected extra training queries"
            assert search["best_score"] == min(record["score"] for record in search["evaluations"])
            if initial_population is None:
                initial_population = search["initial_population_normalized"]
            else:
                assert initial_population == search["initial_population_normalized"], "initialization differs by method"
            if method in ("diagnostic", "zero", "shuffled"):
                identity = (search["parameter_count"], search["initial_model_sha256"])
                if equal_capacity_model is None:
                    equal_capacity_model = identity
                else:
                    assert identity == equal_capacity_model, "equal-capacity initialization differs"

            selected = evaluate_timed_parameters(search["best_parameters"], family=family,
                layout=layout, condition=TRAIN_CONDITION, include_trace=True)
            assert selected["score"] == search["best_score"]
            validation = [evaluate_timed_parameters(search["best_parameters"], family=family,
                layout=layout, condition=replace(TRAIN_CONDITION, velocity_lag_s=lag))
                for lag in VALIDATION_LAGS_S]
            metrics = selected["score_details"]["tracking_metrics"]
            row = dict(family=family, layout=layout, method=method, seed=seed,
                execution_order_index=position, budget=args.budget, observed_training_calls=calls,
                best_score=search["best_score"], feasible=selected["feasible"],
                reference_duration_s=selected["score_details"]["duration_s"],
                tracking_rmse_m=metrics["position_rmse_m"], tracking_metrics=metrics,
                wall_s=wall_s, cpu_s=cpu_s, evaluation_s=search["evaluation_s"],
                neural_generation_s=search.get("neural_generation_s", 0.),
                neural_update_s=search.get("neural_update_s", 0.),
                feature_preparation_s=search.get("feature_preparation_s", 0.),
                parameter_count=search.get("parameter_count", 0),
                failed_training_candidates=sum(not record["feasible"] for record in search["evaluations"]),
                validation_passes=sum(result["feasible"] for result in validation),
                validation_conditions=len(validation), selected_training_recheck_passed=True,
                post_selection_rollouts=1 + len(validation), screening_enabled=False)
            stem = f"{family}_{layout}_{method}_{seed}"
            write_json(output / f"{stem}.json", dict(summary=row, search=search,
                selected_training=selected, validation=validation))
            rows.append(row)
            write_json(output / "rows.json", rows)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    assert len(rows) == manifest["expected_runs"]
    for source in sources:
        assert hashlib.sha256(source.read_bytes()).hexdigest() == hashes[source.relative_to(ROOT).as_posix()], \
            "source changed during experiment; retain results as unbound diagnostics"
    summary, paired = summaries(rows, args)
    write_json(output / "summary.json", summary)
    write_json(output / "paired_comparisons.json", paired)
    write_json(output / "completion.json", dict(status="COMPLETE", runs=len(rows),
        training_calls=sum(row["observed_training_calls"] for row in rows),
        post_selection_rollouts=sum(row["post_selection_rollouts"] for row in rows),
        anchor_warmup_rollouts=len(anchors), feasible_selected=sum(row["feasible"] for row in rows),
        shared_initialization_verified=True, equal_capacity_initial_weights_verified=True,
        selected_training_rechecks_passed=True, source_bytes_unchanged=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[590, 591])
    parser.add_argument("--families", nargs="+", default=["single_bump"])
    parser.add_argument("--layouts", nargs="+", default=["central"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--upstream", default=str(ROOT / "tmp/abom_deploy_probe/upstream"))
    run(parser.parse_args())
