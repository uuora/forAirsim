"""Frozen exploratory two-stage dimension/restart curve-search study.

Reuses the parameter campaign's artifact, validation and summary protocols.
Its short training loop permits honestly different first-stage dimensions.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest import mock

import numpy as np

import run_parameter_perturbation_research as base
from trajectory_research.parameterizations import default_parameters, parameter_bounds
from trajectory_research.staged_search import (SCHEDULES, STAGED_METHODS,
    RESTART_SEED_TAG, run_staged_search, second_stage_seed)


ROOT = Path(__file__).resolve().parents[1]
SOURCES = base.SOURCES + ("python/trajectory_research/staged_search.py", "python/run_staged_curve_research.py")


def manifest_for(args):
    with mock.patch.object(base, "SOURCES", SOURCES):
        result = base.manifest_for(args)
    result.update(schema="initial-curve-staged-perturbation/1", upstream=None,
        initialization="schedule-dependent; 4-D uniform embedded e=0 or 7-D uniform",
        ordering="cyclic schedules across seeds; no neural network",
        stage_policy=dict(stages=2, budgets=[args.budget // 2] * 2,
            dimensions={method: list(SCHEDULES[method]) for method in args.methods},
            stage1_seed="optimizer seed", stage2_seed=f"SeedSequence([seed,1,{RESTART_SEED_TAG}]).generate_state(1,uint32)",
            restart="fresh uniform population, stage-1 best in first evaluated slot; coordinate step resets to .2",
            cross_schedule_initialization="equal within first-stage dimension; different across 4-D and 7-D",
            interpretation="dimension and restart schedule together; not a single-cause tangent-unlock ablation"))
    return result


def train(args, output):
    manifest = manifest_for(args)
    base.freeze(output / "manifest.json", manifest)
    for relative, expected in manifest["source_sha256"].items():
        path = output / "source_snapshot" / relative
        if path.exists():
            if base.digest(path) != expected:
                raise ValueError("Staged source snapshot mismatch")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / relative).read_bytes())
    base.assert_sources(manifest, output)
    choices = []
    lower, upper = parameter_bounds("paired_bump")
    for layout in args.layouts:
        objective = lambda p: base.evaluate_parameters(p, family="paired_bump", layout=layout)
        for index, seed in enumerate(args.seeds):
            shift = index % len(args.methods)
            initial_by_dimension = {}
            for method in args.methods[shift:] + args.methods[:shift]:
                name = base.key("paired_bump", layout, method, seed)
                path = output / "train" / (name + ".json")
                if not path.exists():
                    result = run_staged_search(objective, lower, upper, method=method, seed=seed,
                        population=args.population, budget=args.budget, anchor=default_parameters("paired_bump"))
                    result.update(family="paired_bump", layout=layout, manifest_sha256=base.digest(output / "manifest.json"))
                    base.freeze(path, result)
                _, result = base.checked_training(output, manifest, "paired_bump", layout, method, seed)
                dimension = SCHEDULES[method][0]
                initial = result["initial_population_normalized"]
                if dimension in initial_by_dimension and initial_by_dimension[dimension] != initial:
                    raise ValueError("Schedules with the same first-stage dimension differ in initialization")
                initial_by_dimension[dimension] = initial
                choices.append(dict(family="paired_bump", layout=layout, method=method, seed=seed,
                    parameters=result["best_parameters"], training_score=result["best_score"],
                    training_file=path.relative_to(output).as_posix(), training_sha256=base.digest(path)))
                print(json.dumps(dict(stage="train", key=name, score=result["best_score"])), flush=True)
    choices.sort(key=lambda r: (r["family"], r["layout"], r["seed"], r["method"]))
    base.freeze(output / "selection.json", dict(manifest_sha256=base.digest(output / "manifest.json"),
        selection_basis="training_only", incumbents=choices))


def audit(output):
    """Verify schedule-specific evidence; does not claim the main-study audit."""
    manifest, selection = base.load_selection(output)
    base.assert_sources(manifest, output)
    for choice in selection["incumbents"]:
        _, run = base.checked_training(output, manifest, "paired_bump", choice["layout"], choice["method"], choice["seed"])
        dimensions = SCHEDULES[run["method"]]
        half, population = run["budget"] // 2, run["population"]
        if len(run["stages"]) != 2 or [r["dimension"] for r in run["stages"]] != list(dimensions) \
                or [r["budget"] for r in run["stages"]] != [half, half] \
                or [r["seed"] for r in run["stages"]] != [run["seed"], second_stage_seed(run["seed"])]:
            raise ValueError("Staged dimensions/budget/seed mismatch")
        if run["stages"][1]["anchor_parameters"] != run["stages"][0]["best_parameters"]:
            raise ValueError("Restart anchor did not inherit the first-stage incumbent")
        for stage_index, dimension in enumerate(dimensions):
            chunk = run["evaluations"][stage_index * half:(stage_index + 1) * half]
            if any(row["stage"] != stage_index + 1 or row["stage_dimension"] != dimension \
                    or row["stage_fe"] != i + 1 for i, row in enumerate(chunk)):
                raise ValueError("Staged ledger membership mismatch")
            if dimension == 4 and any(row["parameters"][3:6] != [0., 0., 0.] for row in chunk):
                raise ValueError("Four-dimensional stage did not remain in e=0")
            if run["stages"][stage_index]["best_score"] != min(row["score"] for row in chunk):
                raise ValueError("Stage incumbent is not its training minimum")
        prefix = []
        for row in run["history"]:
            best = min(x["score"] for x in run["evaluations"][:row["fe"]])
            if row["best"] != best:
                raise ValueError("Staged history differs from prefix minimum")
            prefix.append(row["best"])
    result = dict(schema="staged-curve-evidence-audit/1", status="PASS", runs=len(selection["incumbents"]),
        training_candidates=manifest["budget"] * len(selection["incumbents"]),
        manifest_sha256=base.digest(output / "manifest.json"), selection_sha256=base.digest(output / "selection.json"),
        scope="schedule/budget/seed/embedding/anchor/history/hash consistency; no physical validation claim",
        main_study_audit="not applied; cross-schedule first populations intentionally differ")
    base.freeze(output / "staged_audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "summarize", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", choices=STAGED_METHODS, nargs="+", default=list(STAGED_METHODS))
    parser.add_argument("--layouts", choices=("central", "early_offset", "two_obstacles"), nargs="+", default=["central", "early_offset", "two_obstacles"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 310)))
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--stress-seed", type=int, default=20261121)
    parser.add_argument("--stress-count", type=int, default=32)
    args = parser.parse_args()
    args.families, args.upstream = ["paired_bump"], ROOT / "tmp/unused-staged-upstream"
    for field in ("methods", "layouts", "seeds"):
        if len(getattr(args, field)) != len(set(getattr(args, field))):
            parser.error(f"{field} must be unique")
    output = args.output.resolve()
    if args.stage == "train":
        train(args, output)
    elif args.stage == "validate":
        base.validate(output)
    elif args.stage == "summarize":
        base.summarize(output)
    else:
        print(json.dumps(audit(output), indent=2))


if __name__ == "__main__":
    main()
