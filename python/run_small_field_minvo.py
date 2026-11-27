"""Compare the two nominal command bounds using one TOPPQuad solve per route."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from fast_panel import configuration
from fast_reference import decode, nominal_command_bounds, ordinary_plan, retimed, select_segment_timing, solve_blocks, validate_timing
from segment_retiming import SegmentTimeMap


ROOT = Path(__file__).resolve().parents[1]
BASIS = ("bernstein", "minvo")


def compare_route(route_name, route, geometry, dynamics_model):
    config = configuration("pass", geometry, "toppquad", points=route["waypoints"])
    base = ordinary_plan(config["curve_experiment"]["plan"]["route"], geometry)
    base["dynamics_model"] = dynamics_model
    started = perf_counter()
    blocks = solve_blocks(decode(base), dynamics_model)
    solve_s = perf_counter() - started

    spec = base["route"]
    limits = spec["command_limits"]
    options = spec["retiming"]
    nominal_limits = [limits["max_speed_mps"] * options["nominal_speed_fraction"],
                      limits["max_slew_mps2"] * options["nominal_slew_fraction"]]
    variants = {
        basis: dict(base, route=dict(spec, retiming=dict(options, nominal_bound_basis=basis)))
        for basis in BASIS
    }

    full, selected, candidates = {}, {}, {}
    for basis, variant in variants.items():
        raw = retimed(variant, blocks, 1.0)
        duration, validation = validate_timing(raw, nominal_limits, enforce_reserve=True, check_profile=True)
        full[basis] = dict(duration_s=duration, validation=validation)
        started = perf_counter()
        plan, validation, trials = select_segment_timing(variant, blocks, nominal_limits, options["fractions"])
        selected[basis] = dict(duration_s=decode(plan).duration_s, validation=validation,
                               fractions=plan["retiming_profile"]["fractions"],
                               search_wall_s=perf_counter() - started)
        candidates[basis] = {tuple(row["fractions"]): row for row in trials}
        print(route_name, geometry, basis, selected[basis]["duration_s"], len(trials), flush=True)

    (segments, solution), = blocks
    knots = np.r_[0.0, np.cumsum([segment.duration_s for segment in segments])]
    matched = []
    for fractions in sorted(candidates["bernstein"].keys() | candidates["minvo"].keys()):
        retained = next(table[fractions] for table in candidates.values() if fractions in table)
        checks = retained["validation"]["reference_checks"]
        row = dict(fractions=fractions, duration_s=retained["duration_s"], reference_passed=checks["passed"])
        for basis, variant in variants.items():
            trial = candidates[basis].get(fractions)
            if trial is None:
                profile = SegmentTimeMap(solution, knots, fractions)
                bounds = nominal_command_bounds(decode(retimed(variant, blocks, profile=profile)))
            else:
                bounds = trial["validation"]["nominal_command_bounds"]
            row[basis] = dict(bounds=bounds, passed=all(b <= limit for b, limit in zip(bounds, nominal_limits)),
                              searched=trial is not None)
        matched.append(row)

    return dict(route=route_name, geometry=geometry, spec=spec, solve_wall_s=solve_s, solver=solution,
                full=full, automatic=selected, matched_candidates=matched)


def write_result(path, results):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def main():
    routes = json.loads((ROOT / "configs/small_field_routes.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-from", type=Path, required=True, help="Existing validation.json with a recorded dynamics_model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--routes", nargs="+", choices=tuple(routes), default=list(routes))
    parser.add_argument("--geometry", choices=("hermite", "bezier", "bspline"), default="hermite")
    args = parser.parse_args()
    model_bytes = args.model_from.read_bytes()
    dynamics_model = json.loads(model_bytes)["dynamics_model"]
    results = dict(scope="offline_planning", simulator_flights=0, planned_routes=args.routes,
                   model_source=str(args.model_from.resolve()), model_sha256=hashlib.sha256(model_bytes).hexdigest(),
                   dynamics_model=dynamics_model, subdivision_depth=3, comparisons=[])
    args.output.mkdir(parents=True, exist_ok=False)
    for name in args.routes:
        results["comparisons"].append(compare_route(name, routes[name], args.geometry, dynamics_model))
        write_result(args.output / "comparison.json", results)


if __name__ == "__main__":
    main()
