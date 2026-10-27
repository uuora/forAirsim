"""Exploratory fixed-budget coordinate schedules with canonical 7-D ledgers.

These schedules change dimension and/or restart the population. Their effects
are not attributable solely to unlocking a tangent coordinate. The evaluator
always receives the same paired-bump physical parameter vector (d,e,T).
"""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np

from .blackbox import run_blackbox_search


STAGED_METHODS = ("staged_4to4", "staged_4to7", "staged_7to7")
SCHEDULES = {"staged_4to4": (4, 4), "staged_4to7": (4, 7), "staged_7to7": (7, 7)}
RESTART_SEED_TAG = 271828
FOUR_INDICES = np.array([0, 1, 2, 6])


def second_stage_seed(seed):
    return int(np.random.SeedSequence([seed, 1, RESTART_SEED_TAG]).generate_state(1, dtype=np.uint32)[0])


def embed_four(parameters):
    values = np.asarray(parameters, float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("A finite 4-D (d,T) vector is required")
    result = np.zeros(7)
    result[FOUR_INDICES] = values
    return result


def run_staged_search(objective, lower, upper, *, method="staged_4to7", seed=0,
                      population=16, budget=256, anchor=None, upstream=None,
                      initialization="uniform"):
    """Spend half the exact budget in each coordinate-polling stage.

    Each half includes a new population of ``population`` evaluations. Stage 2
    places stage 1's incumbent in its first slot, then initializes other slots
    anew. Its RNG seed is SeedSequence([seed,1,271828]). The old incumbent's
    evaluation remains in the global ledger even if a stochastic oracle changes
    its value. No uncounted objective call is made for stage selection.
    """
    lb, ub = np.asarray(lower, float), np.asarray(upper, float)
    if lb.shape != (7,) or ub.shape != (7,) or not np.isfinite(lb).all() \
            or not np.isfinite(ub).all() or np.any(ub <= lb) or np.any(lb[3:6] > 0) or np.any(ub[3:6] < 0):
        raise ValueError("Finite 7-D bounds containing the e=0 subspace are required")
    if method not in STAGED_METHODS or initialization != "uniform":
        raise ValueError("Unknown schedule or unsupported initialization")
    for value, name in ((population, "population"), (budget, "budget"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if population < 4 or population % 2 or budget < 2 * population or budget % (2 * population):
        raise ValueError("Two equal whole-generation stages including both initial populations are required")
    if anchor is None:
        current_anchor = .5 * (lb + ub)
        current_anchor[3:6] = 0.
    else:
        current_anchor = np.asarray(anchor, float)
        if current_anchor.shape != (7,) or not np.isfinite(current_anchor).all() \
                or np.any(current_anchor < lb) or np.any(current_anchor > ub):
            raise ValueError("Anchor must be within canonical 7-D bounds")
    if SCHEDULES[method][0] == 4 and np.any(current_anchor[3:6] != 0.):
        raise ValueError("A 4-D first stage requires an e=0 anchor")
    started = time.perf_counter()
    records, history, stages = [], [], []
    seeds = (seed, second_stage_seed(seed))
    half = budget // 2
    running_best = float("inf")
    initial = None
    final = None
    for stage_index, dimension in enumerate(SCHEDULES[method]):
        indices = FOUR_INDICES if dimension == 4 else np.arange(7)
        stage_lower, stage_upper = lb[indices], ub[indices]
        to_canonical = embed_four if dimension == 4 else lambda p: np.asarray(p, float)
        stage_result = run_blackbox_search(lambda p: objective(to_canonical(p)),
            stage_lower, stage_upper, method="coordinate", seed=seeds[stage_index],
            population=population, budget=half, anchor=current_anchor[indices], initialization="uniform")
        stage_initial = []
        for point in stage_result["initial_population_normalized"]:
            physical = stage_lower + (np.asarray(point) + 1.) * .5 * (stage_upper - stage_lower)
            canonical = to_canonical(physical)
            stage_initial.append((2 * (canonical - lb) / (ub - lb) - 1).tolist())
        if stage_index == 0:
            initial = stage_initial
        offset = len(records)
        for original in stage_result["evaluations"]:
            row = deepcopy(original)
            row.update(fe=offset + original["fe"], stage=stage_index + 1,
                       stage_fe=original["fe"], stage_dimension=dimension,
                       generation=offset // population + original["generation"],
                       source=f"{method}:stage{stage_index + 1}:{original['source']}",
                       parameters=to_canonical(original["parameters"]).tolist())
            # Poll evidence stays in its declared stage coordinates; candidate
            # parameters and both saved initial populations are canonical 7-D.
            records.append(row)
        for original in stage_result["history"]:
            running_best = min(running_best, original["best"])
            row = dict(original, fe=offset + original["fe"], best=running_best,
                       stage=stage_index + 1, stage_dimension=dimension)
            history.append(row)
        current_anchor = to_canonical(stage_result["best_parameters"])
        stages.append(dict(stage=stage_index + 1, dimension=dimension, seed=seeds[stage_index],
            budget=half, first_fe=offset + 1, last_fe=offset + half,
            initial_population_normalized=stage_initial, anchor_parameters=records[offset]["parameters"],
            best_parameters=current_anchor.tolist(), best_score=stage_result["best_score"],
            elapsed_s=stage_result["elapsed_s"], evaluation_s=stage_result["evaluation_s"]))
        final = stage_result
    best = min(records, key=lambda row: row["score"])
    if len(records) != budget:
        raise AssertionError("Staged candidate budget mismatch")
    return dict(method=method, seed=seed, population=population, budget=budget,
        initial_population_normalized=initial, best_parameters=best["parameters"], best_score=best["score"],
        history=history, evaluations=records,
        initialization=f"stage1_{SCHEDULES[method][0]}d_uniform_then_fresh_stage2_with_warm_anchor",
        normalized_domain=[-1., 1.], injection_fraction=0., elapsed_s=time.perf_counter() - started,
        evaluation_s=sum(row["evaluation_s"] for row in stages), neural_generation_s=0., neural_update_s=0.,
        upstream_commit=None, stages=stages,
        algorithm_identity="Exploratory two-stage coordinate polling; dimension plus restart schedule",
        operator_config=dict(schedule=list(SCHEDULES[method]), budgets=[half, half],
            restart_seed_tag=RESTART_SEED_TAG, restart="fresh uniform population with evaluated warm anchor",
            cross_schedule_initialization="4-D embedded e=0 differs from 7-D uniform"))
