"""Calibrate pinned-ABOM dropout on an offline point-reference geometry task.

Dropout is the only search setting varied. No bound screening or controller
rollout is used. All candidate failures are retained in the raw run records.
The experiment does not reproduce or establish the ABOM paper's performance.
"""
from __future__ import annotations

import argparse
from itertools import permutations
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from trajectory_research.parameterizations import default_parameters, parameter_bounds
from trajectory_research.perturbation_task import evaluate_parameters, get_layout
from trajectory_research.search import UPSTREAM_COMMIT, UPSTREAM_NETWORK_BLOB
from trajectory_research.screened_search import run_screened_search


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DROPOUT = .95


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def dropout_label(value):
    return format(value, ".12g").replace(".", "p")


def validate_args(args):
    if args.budget < 16 or args.budget % 16:
        raise ValueError("budget must be a positive multiple of population=16")
    if not args.seeds or any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("distinct nonnegative seeds are required")
    if not args.dropouts or len(args.dropouts) > 5 \
            or any(not np.isfinite(value) or not 0 <= value < 1 for value in args.dropouts) \
            or len(set(args.dropouts)) != len(args.dropouts) \
            or BASELINE_DROPOUT not in args.dropouts:
        raise ValueError("one to five distinct dropout values in [0,1), including .95, are required")
    if not args.families or len(set(args.families)) != len(args.families):
        raise ValueError("distinct curve families are required")
    for family in args.families:
        parameter_bounds(family)
    if not args.layouts or len(set(args.layouts)) != len(args.layouts):
        raise ValueError("distinct layouts are required")
    for layout in args.layouts:
        get_layout(layout)


def make_summaries(rows, args):
    summaries = []
    paired = []
    for family in args.families:
        for layout in args.layouts:
            condition = [row for row in rows if (row["family"], row["layout"]) == (family, layout)]
            baseline = {row["seed"]: row for row in condition if row["dropout"] == BASELINE_DROPOUT}
            for dropout in args.dropouts:
                group = [row for row in condition if row["dropout"] == dropout]
                assert len(group) == len(args.seeds)
                scores = np.array([row["best_score"] for row in group])
                summaries.append(dict(family=family, layout=layout, dropout=dropout, n=len(group),
                    mean_score=float(np.mean(scores)), median_score=float(np.median(scores)),
                    feasible_runs=sum(row["feasible"] for row in group),
                    mean_duration_s=float(np.mean([row["selected_reference_duration_s"] for row in group])),
                    median_wall_s=float(np.median([row["wall_s"] for row in group])),
                    median_cpu_s=float(np.median([row["cpu_s"] for row in group]))))
                if dropout == BASELINE_DROPOUT:
                    continue
                differences = np.array([row["best_score"] - baseline[row["seed"]]["best_score"] for row in group])
                rng = np.random.default_rng(20261029)
                boot = np.mean(differences[rng.integers(0, len(group), (5000, len(group)))], axis=1)
                paired.append(dict(family=family, layout=layout, dropout=dropout,
                    baseline_dropout=BASELINE_DROPOUT, n=len(group),
                    mean_paired_score_difference=float(np.mean(differences)),
                    median_paired_score_difference=float(np.median(differences)),
                    exploratory_paired_bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                    better_runs=int(np.sum(differences < 0)), equal_runs=int(np.sum(differences == 0)),
                    worse_runs=int(np.sum(differences > 0)),
                    interpretation="negative difference means lower selected score; exploratory across fixed development tasks"))
    return summaries, paired


def run(args):
    validate_args(args)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    sources = [Path(__file__),
        ROOT / "python/trajectory_research/search.py",
        ROOT / "python/trajectory_research/screened_search.py",
        ROOT / "python/trajectory_research/parameterizations.py",
        ROOT / "python/trajectory_research/perturbation_task.py",
        ROOT / "python/trajectory_research/curves.py",
        ROOT / "python/trajectory_research/curve_task.py",
        ROOT / "python/trajectory_research/constraints.py",
        ROOT / "python/trajectory_research/directional.py"]
    hashes = {}
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    upstream_path = Path(args.upstream).resolve() / "abom_net.py"
    upstream_bytes = upstream_path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(upstream_bytes)).encode() + b"\0" + upstream_bytes).hexdigest()
    if blob != UPSTREAM_NETWORK_BLOB:
        raise ValueError("upstream ABOM source does not match the pinned Git blob")
    orders = list(permutations(args.dropouts))
    plan = []
    for condition_index, (family, layout) in enumerate(
            (family, layout) for family in args.families for layout in args.layouts):
        for seed_index, seed in enumerate(args.seeds):
            order = orders[(condition_index * len(args.seeds) + seed_index) % len(orders)]
            plan.append(dict(family=family, layout=layout, seed=seed, dropout_order=list(order)))
    manifest = dict(protocol="pinned-abom-dropout-calibration-v1", args=vars(args), sources=hashes,
        python=platform.python_version(), platform=platform.platform(),
        upstream_commit=UPSTREAM_COMMIT, upstream_network_git_blob=blob,
        upstream_network_sha256=hashlib.sha256(upstream_bytes).hexdigest(),
        population=16, method="abom", initialization="uniform with shared physical anchor",
        screening_enabled=False, bounded_objective_enabled=False,
        model_update_schedule="every generation; source-style reset AdamW and repeated best SmoothL1 target",
        scope="offline polynomial point-reference geometry; no controller, aircraft, wind, sensing, or flight validation",
        timer="warm Python process; includes network construction and search; excludes one-time import/optimizer warmup, anchor checks, selected-result re-evaluation, serialization and summary",
        order="cycle all dropout permutations across condition/seed blocks; plan frozen before timed runs",
        execution_plan=plan, expected_runs=len(plan) * len(args.dropouts),
        baseline_dropout=BASELINE_DROPOUT,
        inference="exploratory calibration on fixed development layouts; no global superiority or novel mechanism claim")
    write_json(output / "manifest.json", manifest)

    import torch
    torch.set_num_threads(1)
    warm_start = time.perf_counter()
    torch.zeros(1).sum()
    run_screened_search(lambda x: dict(score=float(x @ x), feasible=True),
        np.full(4, -1.), np.full(4, 1.), method="abom", population=16,
        budget=32, upstream=args.upstream, dropout=BASELINE_DROPOUT)
    manifest["excluded_one_time_optimizer_warmup_s"] = time.perf_counter() - warm_start
    manifest["torch_version"] = torch.__version__
    write_json(output / "manifest.json", manifest)

    rows = []
    warmed_conditions = set()
    for block in plan:
        family, layout, seed = block["family"], block["layout"], block["seed"]
        lo, hi = parameter_bounds(family)
        objective = lambda x: evaluate_parameters(x, family=family, layout=layout)
        if (family, layout) not in warmed_conditions:
            objective(default_parameters(family))
            warmed_conditions.add((family, layout))
        initial_population = None
        for position, dropout in enumerate(block["dropout_order"]):
            tick, cpu = time.perf_counter(), time.process_time()
            result = run_screened_search(objective, lo, hi, lower_bound=None, bounded_objective=None,
                method="abom", seed=seed, population=16, budget=args.budget,
                anchor=default_parameters(family), upstream=args.upstream, dropout=dropout)
            wall_s, cpu_s = time.perf_counter() - tick, time.process_time() - cpu
            # The independent objective call below is deliberately outside timing.
            checked = objective(result["best_parameters"])
            assert checked["score"] == result["best_score"]
            assert len(result["evaluations"]) == args.budget
            assert result["full_evaluations"] == args.budget and result["pruned_count"] == 0
            assert all(row["status"] == "EVALUATED" for row in result["evaluations"])
            assert result["best_score"] == min(row["score"] for row in result["evaluations"])
            if initial_population is None:
                initial_population = result["initial_population_normalized"]
            else:
                assert initial_population == result["initial_population_normalized"], "initialization changed with dropout"
            row = dict(family=family, layout=layout, seed=seed, dropout=dropout,
                execution_order_index=position, budget=args.budget,
                best_score=result["best_score"], feasible=checked["feasible"],
                selected_reference_duration_s=checked["score_details"]["duration_s"],
                wall_s=wall_s, cpu_s=cpu_s, full_evaluations=result["full_evaluations"],
                failed_candidate_count=sum(not record["feasible"] for record in result["evaluations"]),
                screening_enabled=False, best_recheck_passed=True)
            result["outer_wall_s"], result["outer_cpu_s"] = wall_s, cpu_s
            stem = f"{family}_{layout}_{seed}_dropout_{dropout_label(dropout)}"
            write_json(output / f"{stem}.json", dict(summary=row, search=result,
                selected_result_recheck=checked))
            rows.append(row)
            write_json(output / "rows.json", rows)
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    assert len(rows) == manifest["expected_runs"]
    summaries, paired = make_summaries(rows, args)
    write_json(output / "summary.json", summaries)
    write_json(output / "paired_comparisons.json", paired)
    write_json(output / "completion.json", dict(status="COMPLETE", runs=len(rows),
        full_evaluations=sum(row["full_evaluations"] for row in rows),
        feasible_selected=sum(row["feasible"] for row in rows),
        shared_initialization_verified=True, best_result_rechecks_passed=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(401, 411)))
    parser.add_argument("--dropouts", nargs="+", type=float, default=[.95, .5, .1])
    parser.add_argument("--families", nargs="+", default=["single_bump", "paired_bump"])
    parser.add_argument("--layouts", nargs="+", default=["central", "early_offset", "two_obstacles"])
    parser.add_argument("--upstream", default=str(ROOT / "tmp/abom_deploy_probe/upstream"))
    run(parser.parse_args())
