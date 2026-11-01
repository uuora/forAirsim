"""Research-only lower-bound screening of the pinned ABOM search adapter.

This is a proposal-budget experiment, not an equal-full-evaluation-budget
experiment. A supplied lower bound is a caller-owned mathematical contract:
``lower_bound(x) <= objective(x)['score']`` for every candidate. It must be
deterministic, finite, and must not mutate the search or objective's state/RNG.
We can detect violations only on candidates that are actually evaluated.

With those assumptions, a child whose bound is at least the WORST current
parent score cannot enter stable parent-first elitist selection. Omitting its
expensive evaluation therefore preserves retained parents, adaptation targets,
and the entire candidate stream. This says nothing about global optimality or
physical flight safety. Floating-point bounds need a caller-supplied conservative
roundoff allowance; the optimizer does not silently relax an invalid bound.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real
import time
from typing import Callable

import numpy as np

from .search import (METHODS, UPSTREAM_COMMIT, PerturbationState,
                     latin_hypercube, load_upstream_network, paired_perturbation)


def _finite_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real scalar")
    return value


def run_screened_search(objective: Callable, lower, upper, *, lower_bound=None,
                        bounded_objective=None,
                        method="random", seed=0, population=16, budget=128,
                        anchor=None, upstream=None, initialization="uniform",
                        injection_fraction=0.5, dropout=0.95):
    """Run a fixed number of proposals with optional exact bound screening.

    ``budget`` counts proposals, including initial points and pruned children.
    Leaving both screening callbacks at ``None`` gives a matched full-evaluation
    baseline. Initial points are always fully evaluated. Every measured score
    is checked against its bound when a bound is supplied; violations abort.

    An optional ``bounded_objective(x, cutoff)`` may stop an expensive evaluation
    early only if it proves a lower bound >= cutoff. It returns either the usual
    full result, or ``status='PRUNED', lower_bound=<finite>, score=None,
    feasible=None``. ``cutoff=None`` (initialization) requires a full result.
    This hook must agree with ``objective`` whenever fully evaluated; correctness
    of its partial bound is also a caller-owned contract. Partial callback work
    is timed but never counted as a completed objective evaluation.

    ``dropout`` controls both pinned-network dropout probabilities. Its default
    reproduces ``run_search``; changing it is a separate, explicitly recorded
    ablation. The original every-generation model update is retained unchanged.
    The objective and bound callbacks must not consume Torch's global RNG
    (dropout uses it) or any mutable shared state. No objective call is made for
    pruned children: their score and feasibility remain unknown/null.
    """
    if not callable(objective) or (lower_bound is not None and not callable(lower_bound)) \
            or (bounded_objective is not None and not callable(bounded_objective)):
        raise ValueError("objective and optional callbacks must be callable")
    lb, ub = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lb.ndim != 1 or len(lb) == 0 or ub.shape != lb.shape or not np.isfinite(lb).all() \
            or not np.isfinite(ub).all() or np.any(ub <= lb):
        raise ValueError("finite ordered vector bounds are required")
    for value, name in ((population, "population"), (budget, "budget"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if population < 4 or population % 2 or budget < population or budget % population:
        raise ValueError("even population >=4 and whole-generation budget required")
    if method not in METHODS or initialization not in ("uniform", "lhs"):
        raise ValueError("unknown method or initialization")
    injection_fraction = _finite_scalar(injection_fraction, "injection_fraction")
    dropout = _finite_scalar(dropout, "dropout")
    if not 0 <= injection_fraction <= 1 or not 0 <= dropout < 1:
        raise ValueError("injection_fraction must be in [0,1], dropout in [0,1)")
    dim = len(lb)
    init_rng = np.random.default_rng(seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
    parents = (latin_hypercube(population, dim, init_rng) if initialization == "lhs"
               else init_rng.uniform(-1, 1, (population, dim)))
    if anchor is not None:
        a = np.asarray(anchor, dtype=float)
        if a.shape != lb.shape or not np.isfinite(a).all() or np.any(a < lb) or np.any(a > ub):
            raise ValueError("anchor outside bounds")
        parents[0] = 2 * (a - lb) / (ub - lb) - 1
    initial_population = parents.copy()
    records, history = [], []
    state = PerturbationState()
    neural_s = update_s = evaluation_s = bound_s = 0.0
    full_evaluation_s = partial_evaluation_s = 0.0
    full_evaluations = partial_evaluations = pruned_count = bound_calls = bounded_calls = 0
    started = time.perf_counter()

    def evaluate(points, generation, sources, threshold=None):
        nonlocal evaluation_s, bound_s, full_evaluations, pruned_count, bound_calls
        nonlocal full_evaluation_s, partial_evaluation_s, partial_evaluations, bounded_calls
        costs = []
        for point, source in zip(points, sources):
            physical = lb + (point + 1) * 0.5 * (ub - lb)
            bound = None
            bound_duration = 0.0
            if lower_bound is not None:
                tick = time.perf_counter()
                bound = _finite_scalar(lower_bound(physical.copy()), "lower bound")
                bound_duration = time.perf_counter() - tick
                bound_s += bound_duration
                bound_calls += 1
            common = dict(proposal_index=len(records) + 1, generation=generation,
                          source=source, parameters=physical.tolist(),
                          lower_bound=bound, selection_threshold=threshold,
                          lower_bound_s=bound_duration)
            if threshold is not None and bound is not None and bound >= threshold:
                pruned_count += 1
                records.append(dict(**common, status="PRUNED", fe=None, score=None,
                                    feasible=None, evaluation_s=0.0, details={},
                                    prune_kind="CHEAP_BOUND"))
                # Internal sentinel only: never publish it as a measured score.
                costs.append(math.inf)
                continue
            tick = time.perf_counter()
            if bounded_objective is None:
                result = objective(physical.copy())
            else:
                result = bounded_objective(physical.copy(), threshold)
                bounded_calls += 1
            duration = time.perf_counter() - tick
            evaluation_s += duration
            if isinstance(result, Mapping) and result.get("status") == "PRUNED":
                partial_bound = _finite_scalar(result.get("lower_bound"), "partial lower bound")
                if threshold is None or partial_bound < threshold:
                    raise ValueError("partial pruning requires a bound at least the active cutoff")
                if result.get("score") is not None or result.get("feasible") is not None:
                    raise ValueError("pruned partial evaluations must leave score and feasibility unknown")
                partial_evaluations += 1
                partial_evaluation_s += duration
                pruned_count += 1
                common["cheap_lower_bound"] = bound
                common["lower_bound"] = partial_bound
                records.append(dict(**common, status="PRUNED", fe=None, score=None,
                                    feasible=None, evaluation_s=duration,
                                    details=result.get("score_details", {}),
                                    prune_kind="BOUNDED_OBJECTIVE"))
                costs.append(math.inf)
                continue
            if not isinstance(result, Mapping) or "score" not in result or "feasible" not in result:
                raise ValueError("objective must return score and boolean feasible")
            if result.get("status") not in (None, "EVALUATED"):
                raise ValueError("unknown objective evaluation status")
            score = _finite_scalar(result["score"], "objective score")
            feasible = result["feasible"]
            if not isinstance(feasible, (bool, np.bool_)):
                raise ValueError("invalid objective feasibility")
            if bound is not None and bound > score:
                raise ValueError("lower bound exceeds an evaluated objective score")
            full_evaluations += 1
            full_evaluation_s += duration
            records.append(dict(**common, status="EVALUATED", fe=full_evaluations,
                                score=score, feasible=bool(feasible), evaluation_s=duration,
                                details=result.get("score_details", {}), prune_kind=None))
            costs.append(score)
        return np.array(costs)

    def add_history(**extra):
        history.append(dict(proposals=len(records), full_evaluations=full_evaluations,
                            pruned_count=pruned_count, best=float(costs[0]),
                            scale=state.scale, **extra))

    costs = evaluate(parents, 0, ["initial"] * population)
    order = np.argsort(costs, kind="stable")
    parents, costs = parents[order], costs[order]
    add_history()
    model = torch = None
    if method.startswith("abom"):
        if upstream is None:
            raise ValueError("an explicit pinned ABOM source directory is required")
        import torch
        network = load_upstream_network(upstream)
        torch.set_num_threads(1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = network(1, dim, 1 << (dim.bit_length() - 1), dropout, dropout,
                            dim, "cpu", generator).double()
            for name, parameter in model.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    torch.nn.init.constant_(parameter, 0)
        model.train()

    def generation_loop():
        nonlocal parents, costs, neural_s, update_s
        for generation in range(1, budget // population):
            output = None
            old_best = float(costs[0])
            # Freeze the worst retained parent, NOT the best. A child can
            # improve the population even when it cannot improve its best.
            threshold = float(costs[-1])
            sources = [method] * population
            if model is not None:
                tick = time.perf_counter()
                x = torch.from_numpy(parents.copy()).unsqueeze(0)
                ranks = torch.arange(population, dtype=torch.float64).view(1, population, 1)
                output = model(x, ranks, generation - 1)
                if not torch.isfinite(output).all():
                    raise ValueError("nonfinite neural candidates")
                children = output.detach().clamp(-1, 1).squeeze(0).numpy().copy()
                neural_s += time.perf_counter() - tick
                if method.startswith("abom_perturb") or method == "abom_triggered":
                    count = int(population * injection_fraction) // 2 * 2
                    if method == "abom_triggered":
                        count = min(count, max(2, population // 4 // 2 * 2)) if state.stalled >= 2 else 0
                    if count:
                        children[-count:] = paired_perturbation(
                            parents, count, rng, state,
                            antithetic=method != "abom_perturb_independent",
                            boundary="clip" if method == "abom_perturb_clip" else "reflect")
                        sources[-count:] = ["paired_perturbation"] * count
            elif method == "gaussian":
                children = paired_perturbation(parents, population, rng, state)
            else:
                children = rng.uniform(-1, 1, (population, dim))
            child_costs = evaluate(children, generation, sources, threshold)
            success_rate = float(np.mean(child_costs < old_best))
            all_points = np.concatenate((parents, children))
            all_costs = np.concatenate((costs, child_costs))
            elite_indices = np.argsort(all_costs, kind="stable")[:population]
            parents, costs = all_points[elite_indices], all_costs[elite_indices]
            if model is not None:
                tick = time.perf_counter()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
                target = torch.from_numpy(parents[0].copy()).view(1, 1, dim).repeat(1, population, 1)
                loss = torch.nn.functional.smooth_l1_loss(output, target, beta=1., reduction="sum")
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite adaptation loss")
                optimizer.zero_grad()
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters()):
                    raise ValueError("nonfinite adaptation gradient")
                optimizer.step()
                update_s += time.perf_counter() - tick
            if method != "abom_perturb_fixed":
                state.update(success_rate)
            add_history(success_rate=success_rate,
                        population_diversity=float(np.mean(np.std(parents, axis=0))))

    if model is None:
        generation_loop()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 271828)
            generation_loop()
    best = lb + (parents[0] + 1) * .5 * (ub - lb)
    return dict(method=method, seed=seed, population=population,
                proposal_budget=budget, proposals=len(records), full_evaluations=full_evaluations,
                pruned_count=pruned_count, lower_bound_calls=bound_calls,
                partial_evaluations=partial_evaluations, bounded_objective_calls=bounded_calls,
                screening_enabled=lower_bound is not None or bounded_objective is not None,
                initial_population_normalized=initial_population.tolist(),
                best_parameters=best.tolist(), best_score=float(costs[0]),
                final_population_normalized=parents.tolist(), final_scores=costs.tolist(),
                history=history, evaluations=records, initialization=initialization,
                normalized_domain=[-1., 1.], injection_fraction=injection_fraction,
                elapsed_s=time.perf_counter() - started, evaluation_s=evaluation_s,
                full_evaluation_s=full_evaluation_s, partial_evaluation_s=partial_evaluation_s,
                lower_bound_s=bound_s, neural_generation_s=neural_s, neural_update_s=update_s,
                dropout=dropout if model is not None else None,
                model_update_schedule="every_generation" if model is not None else None,
                upstream_commit=UPSTREAM_COMMIT if model is not None else None,
                algorithm_identity="Pinned ABOM-net adapter with optional exact bound screening"
                if model is not None else "Generic search baseline with optional exact bound screening",
                bound_contract="Caller asserts deterministic finite bounds <= objective score; "
                               "cheap bounds checked only on fully evaluated candidates; "
                               "partial callback must agree with objective and certify its own bounds")
