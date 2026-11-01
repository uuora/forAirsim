"""Budgeted DE, jDE and coordinate-poll baselines for curve parameter search.

The operators perturb normalized parameters; curve endpoint constraints belong
to the decoder. DE/jDE trials use a frozen parent generation and one-to-one
greedy selection; coordinate polling uses a frozen incumbent and global elitist
truncation. The same initialization as ``search.run_search`` permits paired
comparisons. Reflection, the common anchor, small population and finite budget
are declared study choices, not a reproduction of either paper's benchmark.

References:
Storn & Price (1997), doi:10.1023/A:1008202821328.
Brest et al. (2006), doi:10.1109/TEVC.2006.872133, Section III.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import time
from typing import Callable

import numpy as np

from .search import latin_hypercube, reflect_box


BLACKBOX_METHODS = ("de", "jde", "coordinate")


def _bounds(lower, upper):
    try:
        lb, ub = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("finite ordered vector bounds are required") from error
    if lb.ndim != 1 or not len(lb) or ub.shape != lb.shape or not np.isfinite(lb).all() \
            or not np.isfinite(ub).all() or np.any(ub <= lb) \
            or not np.isfinite(ub - lb).all():
        raise ValueError("finite ordered vector bounds are required")
    return lb, ub


def run_blackbox_search(objective: Callable, lower, upper, *, method="de", seed=0,
                        population=16, budget=128, anchor=None,
                        initialization="uniform"):
    """Search with an exact objective budget and a complete proposal ledger.

    ``de`` uses F=.5 and CR=.9. ``jde`` starts at those values and refreshes
    each individual's proposed F with probability .1 to .1+.9*U and proposed
    CR with probability .1 to U. A proposed control value survives only when
    its trial is accepted (score <= target score). The F refresh range is
    [.1, 1.0], matching Fl=.1, Fu=.9 in Brest et al.'s Fl+U*Fu formula.

    ``coordinate`` is a generic incumbent-centred polling heuristic. Each
    generation polls paired coordinate directions at step .2 initially; slots
    left after polling all axes use paired random unit directions. A strict
    incumbent improvement multiplies step by 1.2 (cap .5), otherwise step is
    halved (floor 1e-4). With fewer than 2*dimension slots, axes rotate between
    generations. It is not MADS and carries no convergence guarantee.

    Distinct donor indices exclude the target. Binomial crossover forces one
    coordinate from the mutant. Reflected trials are evaluated once; rejected
    trials still count. Objective exceptions propagate. No validation data or
    objective-gradient information enters this function.
    """
    lb, ub = _bounds(lower, upper)
    for value, name in ((population, "population"), (budget, "budget"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if population < 4 or population % 2 or budget < population or budget % population:
        raise ValueError("even population >=4 and whole-generation budget required")
    if method not in BLACKBOX_METHODS or initialization not in ("uniform", "lhs"):
        raise ValueError("unknown method or initialization")
    if not callable(objective):
        raise ValueError("objective must be callable")
    dim = len(lb)
    init_rng = np.random.default_rng(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
    parents = latin_hypercube(population, dim, init_rng) if initialization == "lhs" \
        else init_rng.uniform(-1, 1, (population, dim))
    if anchor is not None:
        a = np.asarray(anchor, dtype=float)
        if a.shape != lb.shape or not np.isfinite(a).all() or np.any(a < lb) or np.any(a > ub):
            raise ValueError("anchor outside bounds")
        parents[0] = 2 * (a - lb) / (ub - lb) - 1
    initial_population = parents.copy()
    factors = np.full(population, .5)
    crossover_rates = np.full(population, .9)
    records, history = [], []
    evaluation_s = 0.
    started = time.perf_counter()

    def evaluate(points, generation, evidence=None):
        nonlocal evaluation_s
        costs = []
        for i, point in enumerate(points):
            physical = lb + (point + 1) * .5 * (ub - lb)
            tick = time.perf_counter()
            result = objective(physical)
            duration = time.perf_counter() - tick
            evaluation_s += duration
            if not isinstance(result, Mapping) or "score" not in result or "feasible" not in result:
                raise ValueError("objective must return score and boolean feasible")
            try:
                score = float(result["score"])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("objective score must be finite") from error
            if not math.isfinite(score) or not isinstance(result["feasible"], (bool, np.bool_)):
                raise ValueError("invalid objective result")
            record = dict(fe=len(records) + 1, generation=generation,
                          source="initial" if generation == 0 else method,
                          parameters=physical.tolist(), score=score,
                          feasible=bool(result["feasible"]), evaluation_s=duration,
                          details=result.get("score_details", {}))
            if evidence is not None:
                record["poll" if method == "coordinate" else "de"] = evidence[i]
            records.append(record)
            costs.append(score)
        return np.asarray(costs)

    costs = evaluate(parents, 0)

    def append_history(fe, accepted=None):
        row = dict(fe=fe, best=float(np.min(costs)),
                   population_diversity=float(np.mean(np.std(parents, axis=0))))
        if method != "coordinate":
            row.update(mean_factor=float(np.mean(factors)),
                       mean_crossover_rate=float(np.mean(crossover_rates)))
        if accepted is not None:
            row["success_rate"] = float(np.mean(accepted))
        history.append(row)

    append_history(population)
    indices = np.arange(population)
    step = .2
    if method == "coordinate":
        history[0]["scale"] = step
    for generation in range(1, budget // population):
        if method == "coordinate":
            best_index = int(np.argmin(costs))
            centre = parents[best_index].copy()
            old_best = float(costs[best_index])
            children, evidence = [], []
            coordinate_pairs = min(dim, population // 2)
            first_axis = ((generation - 1) * coordinate_pairs) % dim
            for pair in range(population // 2):
                if pair < coordinate_pairs:
                    axis = (first_axis + pair) % dim
                    direction = np.eye(dim)[axis]
                    kind = "coordinate"
                else:
                    axis = None
                    direction = rng.normal(size=dim)
                    norm = float(np.linalg.norm(direction))
                    # A Gaussian zero vector has probability zero; handle it
                    # deterministically rather than retrying without a bound.
                    direction = direction / norm if norm > 0 else np.eye(dim)[0]
                    kind = "random_unit"
                for sign in (1., -1.):
                    delta = step * direction * sign
                    children.append(reflect_box(centre + delta))
                    evidence.append(dict(centre_normalized=centre.tolist(), step=step,
                                         direction=direction.tolist(), sign=sign,
                                         kind=kind, axis=axis, incumbent_score=old_best))
            children = np.array(children)
            child_costs = evaluate(children, generation, evidence)
            all_parents = np.concatenate((parents, children))
            all_costs = np.concatenate((costs, child_costs))
            elite = np.argsort(all_costs, kind="stable")[:population]
            parents, costs = all_parents[elite], all_costs[elite]
            improved = bool(float(np.min(costs)) < old_best)
            step = min(.5, step * 1.2) if improved else max(1e-4, step * .5)
            append_history(len(records), child_costs < old_best)
            history[-1]["scale"] = step
            continue
        # Nothing in this proposal loop changes parents, costs or controls.
        children = np.empty_like(parents)
        proposed_factors, proposed_cr = factors.copy(), crossover_rates.copy()
        evidence = []
        for target in range(population):
            if method == "jde":
                # Four independent draws per target implement Eq. (4)/(5).
                random_values = rng.random(4)
                if random_values[1] < .1:
                    proposed_factors[target] = .1 + .9 * random_values[0]
                if random_values[3] < .1:
                    proposed_cr[target] = random_values[2]
            donors = rng.choice(indices[indices != target], size=3, replace=False)
            a, b, c = parents[donors]
            mutant = a + proposed_factors[target] * (b - c)
            mask = rng.random(dim) < proposed_cr[target]
            forced = int(rng.integers(dim))
            mask[forced] = True
            children[target] = reflect_box(np.where(mask, mutant, parents[target]))
            evidence.append(dict(target_index=target, donor_indices=donors.tolist(),
                                 factor=float(proposed_factors[target]),
                                 crossover_rate=float(proposed_cr[target]),
                                 previous_factor=float(factors[target]),
                                 previous_crossover_rate=float(crossover_rates[target]),
                                 forced_coordinate=forced, crossover_mask=mask.tolist(),
                                 parent_score=float(costs[target])))
        child_costs = evaluate(children, generation, evidence)
        accepted = child_costs <= costs
        for row, accept in zip(evidence, accepted):
            row["accepted"] = bool(accept)
        # Synchronous, one-to-one selection. No global truncation or sorting.
        parents[accepted] = children[accepted]
        costs[accepted] = child_costs[accepted]
        factors[accepted] = proposed_factors[accepted]
        crossover_rates[accepted] = proposed_cr[accepted]
        append_history(len(records), accepted)
    best_index = int(np.argmin(costs))
    best = lb + (parents[best_index] + 1) * .5 * (ub - lb)
    return dict(method=method, seed=seed, population=population, budget=budget,
                initial_population_normalized=initial_population.tolist(),
                best_parameters=best.tolist(), best_score=float(costs[best_index]),
                history=history, evaluations=records, initialization=initialization,
                normalized_domain=[-1., 1.], injection_fraction=0.,
                elapsed_s=time.perf_counter() - started, evaluation_s=evaluation_s,
                neural_generation_s=0., neural_update_s=0., upstream_commit=None,
                final_population_normalized=parents.tolist(), final_scores=costs.tolist(),
                final_factors=None if method == "coordinate" else factors.tolist(),
                final_crossover_rates=None if method == "coordinate" else crossover_rates.tolist(),
                algorithm_identity={"de": "DE/rand/1/bin", "jde": "jDE/rand/1/bin",
                                    "coordinate": "Paired coordinate polling heuristic"}[method],
                operator_config=dict(initial_step=.2, step_increase=1.2, step_decrease=.5,
                                     step_floor=1e-4, step_cap=.5, boundary="reflection",
                                     selection="global elitist truncation") if method == "coordinate"
                else dict(initial_factor=.5, initial_crossover_rate=.9,
                                     factor_refresh_probability=.1 if method == "jde" else 0.,
                                     crossover_refresh_probability=.1 if method == "jde" else 0.,
                                     factor_refresh_range=[.1, 1.], boundary="reflection",
                                     selection="synchronous one-to-one <="))
