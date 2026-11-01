"""Controlled diagnostic-feedback ablations of the pinned ABOM adapter.

Every candidate is fully evaluated exactly once. All four modes collect the
same five diagnostics from that evaluation. Only the inputs to the network
differ: scalar has normalized rank alone; diagnostic, zero and shuffled have
rank plus five columns and identical parameter counts/initial weights.

The source network's fitness-query/key projections are replaced in memory for
the six-column modes; upstream files remain unchanged. All modes use rank in
[0,1], unlike search.py's raw integer ranks. This is a separately identified
research adapter, not a reproduction or performance claim for the paper.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import math
from numbers import Real
import time
from typing import Callable

import numpy as np

from .search import UPSTREAM_COMMIT, UPSTREAM_NETWORK_BLOB, latin_hypercube, load_upstream_network


MODES = ("scalar", "diagnostic", "zero", "shuffled")
DIAGNOSTIC_DIMENSION = 5
DIAGNOSTIC_NAMES = (
    "normalized_position_rmse", "normalized_max_position_error",
    "normalized_terminal_velocity_error", "command_slew_clipped_time_fraction",
    "peak_position_error_u",
)


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar")
    return value


def _control_features(result):
    details = result.get("score_details")
    if not isinstance(details, Mapping) or "control_features" not in details:
        raise ValueError("all modes require score_details.control_features")
    try:
        raw = np.asarray(details["control_features"])
    except (TypeError, ValueError) as error:
        raise ValueError("control_features must contain five finite real numbers") from error
    if raw.shape != (DIAGNOSTIC_DIMENSION,) or raw.dtype.kind not in "iuf":
        raise ValueError("control_features must contain five finite real numbers")
    vector = raw.astype(float, copy=True)
    if not np.isfinite(vector).all():
        raise ValueError("control_features must contain five finite real numbers")
    return vector


def _model_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tuple(parameter.shape)).encode("ascii") + b"\0")
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_feedback_search(objective: Callable, lower, upper, *, mode="diagnostic", seed=0,
                        population=16, budget=128, anchor=None, upstream=None,
                        initialization="uniform", dropout=.1):
    """Optimize with fixed evaluation budget and a controlled feedback mode.

    ``objective`` must return finite score, boolean feasible, and a five-value
    ``score_details.control_features`` vector, including for scalar/zero modes.
    That requirement makes extraction costs and available diagnostic records
    comparable. There are no diagnostic-only calls or hidden validation calls.

    The diagnostics' normalization belongs to the declared objective; this
    adapter checks finiteness/shape and does not normalize them again. Parents
    retain their own measured diagnostics through stable elitist selection.
    Shuffling permutes diagnostic rows only, keeping ranks and coordinates in
    place, using a separate seeded RNG. The per-generation consumption ledger
    records the raw parent diagnostics, permutation and actual input matrix.

    Model updates deliberately preserve the local source-style adapter:
    reset AdamW each generation and use SmoothL1 against the repeated single
    best retained candidate. No Gaussian injection or bound screening is used.
    """
    if not callable(objective):
        raise ValueError("objective must be callable")
    if mode not in MODES or initialization not in ("uniform", "lhs"):
        raise ValueError("unknown feedback mode or initialization")
    lb, ub = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if lb.ndim != 1 or not len(lb) or ub.shape != lb.shape or not np.isfinite(lb).all() \
            or not np.isfinite(ub).all() or np.any(ub <= lb):
        raise ValueError("finite ordered vector bounds are required")
    for value, name in ((population, "population"), (budget, "budget"), (seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if population < 4 or population % 2 or budget < population or budget % population:
        raise ValueError("even population >=4 and whole-generation budget required")
    dropout = _finite_scalar(dropout, "dropout")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must lie in [0,1)")
    if upstream is None:
        raise ValueError("an explicit pinned ABOM source directory is required")
    dim = len(lb)
    init_rng = np.random.default_rng(seed)
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, 8675309]))
    parents = (latin_hypercube(population, dim, init_rng) if initialization == "lhs"
               else init_rng.uniform(-1, 1, (population, dim)))
    if anchor is not None:
        a = np.asarray(anchor, dtype=float)
        if a.shape != lb.shape or not np.isfinite(a).all() or np.any(a < lb) or np.any(a > ub):
            raise ValueError("anchor outside bounds")
        parents[0] = 2 * (a - lb) / (ub - lb) - 1
    initial_population = parents.copy()
    records, history, feature_ledger = [], [], []
    evaluation_s = neural_s = update_s = feature_s = 0.
    started = time.perf_counter()

    def evaluate(points, generation):
        nonlocal evaluation_s, feature_s
        scores, diagnostics, ids = [], [], []
        for point in points:
            physical = lb + (point + 1) * .5 * (ub - lb)
            tick = time.perf_counter()
            result = objective(physical.copy())
            duration = time.perf_counter()-tick
            evaluation_s += duration
            if not isinstance(result, Mapping) or "score" not in result or "feasible" not in result:
                raise ValueError("objective must return score and boolean feasible")
            score = _finite_scalar(result["score"], "objective score")
            feasible = result["feasible"]
            if not isinstance(feasible, (bool, np.bool_)):
                raise ValueError("objective feasibility must be boolean")
            tick = time.perf_counter()
            vector = _control_features(result)
            copied_details = deepcopy(dict(result["score_details"]))
            feature_s += time.perf_counter()-tick
            fe = len(records)+1
            records.append(dict(fe=fe, generation=generation, source="initial" if generation == 0 else mode,
                parameters=physical.tolist(), normalized_parameters=point.tolist(), score=score,
                feasible=bool(feasible), control_features=vector.tolist(), evaluation_s=duration,
                details=copied_details))
            scores.append(score)
            diagnostics.append(vector)
            ids.append(fe)
        return np.asarray(scores), np.asarray(diagnostics), np.asarray(ids)

    costs, diagnostics, parent_ids = evaluate(parents, 0)
    order = np.argsort(costs, kind="stable")
    parents, costs, diagnostics, parent_ids = (
        parents[order], costs[order], diagnostics[order], parent_ids[order])
    history.append(dict(fe=population, best=float(costs[0]), parent_evaluation_ids=parent_ids.tolist()))

    import torch
    torch.set_num_threads(1)
    network = load_upstream_network(upstream)
    input_columns = 1 if mode == "scalar" else 1+DIAGNOSTIC_DIMENSION
    swap_generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = network(1, dim, 1 << (dim.bit_length()-1), dropout, dropout,
                        dim, "cpu", swap_generator).double()
        if input_columns > 1:
            for layer in model.layers:
                projections = layer.crossover_module.sublayer.linears
                for index in (2, 3):
                    output_columns = projections[index].out_features
                    projections[index] = torch.nn.Linear(input_columns, output_columns).double()
        for name, parameter in model.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                torch.nn.init.constant_(parameter, 0)
        initial_model_sha256 = _model_fingerprint(model)
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+271828)
        for generation in range(1, budget//population):
            old_best = float(costs[0])
            tick = time.perf_counter()
            ranks = np.arange(population, dtype=float)/(population-1)
            permutation = np.arange(population)
            if mode == "scalar":
                consumed = ranks[:, None]
            else:
                extra = diagnostics.copy()
                if mode == "zero":
                    extra.fill(0.)
                elif mode == "shuffled":
                    permutation = shuffle_rng.permutation(population)
                    extra = extra[permutation]
                consumed = np.column_stack((ranks, extra))
            feature_ledger.append(dict(generation=generation, parent_evaluation_ids=parent_ids.tolist(),
                parent_normalized_parameters=parents.tolist(), parent_scores=costs.tolist(),
                parent_control_features=diagnostics.tolist(), diagnostic_row_permutation=permutation.tolist(),
                consumed_features=consumed.tolist()))
            features = torch.from_numpy(consumed).unsqueeze(0)
            feature_s += time.perf_counter()-tick
            tick = time.perf_counter()
            x = torch.from_numpy(parents.copy()).unsqueeze(0)
            output = model(x, features, generation-1)
            if not torch.isfinite(output).all():
                raise ValueError("nonfinite neural candidates")
            children = output.detach().clamp(-1, 1).squeeze(0).numpy().copy()
            neural_s += time.perf_counter()-tick
            child_costs, child_diagnostics, child_ids = evaluate(children, generation)
            all_points = np.concatenate((parents, children))
            all_costs = np.concatenate((costs, child_costs))
            all_diagnostics = np.concatenate((diagnostics, child_diagnostics))
            all_ids = np.concatenate((parent_ids, child_ids))
            selected = np.argsort(all_costs, kind="stable")[:population]
            parents, costs, diagnostics, parent_ids = (
                all_points[selected], all_costs[selected], all_diagnostics[selected], all_ids[selected])
            tick = time.perf_counter()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
            target = torch.from_numpy(parents[0].copy()).view(1, 1, dim).repeat(1, population, 1)
            loss = torch.nn.functional.smooth_l1_loss(output, target, beta=1., reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("nonfinite adaptation loss")
            optimizer.zero_grad()
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                   for parameter in model.parameters()):
                raise ValueError("nonfinite adaptation gradient")
            optimizer.step()
            update_s += time.perf_counter()-tick
            history.append(dict(fe=len(records), best=float(costs[0]),
                success_rate=float(np.mean(child_costs < old_best)),
                population_diversity=float(np.mean(np.std(parents, axis=0))),
                parent_evaluation_ids=parent_ids.tolist()))

    best = lb + (parents[0]+1)*.5*(ub-lb)
    return dict(method="feedback_abom", mode=mode, seed=seed, population=population, budget=budget,
        full_evaluations=len(records), initial_population_normalized=initial_population.tolist(),
        best_parameters=best.tolist(), best_score=float(costs[0]),
        best_control_features=diagnostics[0].tolist(), best_evaluation_id=int(parent_ids[0]),
        final_population_normalized=parents.tolist(), final_scores=costs.tolist(),
        final_control_features=diagnostics.tolist(), final_parent_evaluation_ids=parent_ids.tolist(),
        history=history, evaluations=records, feature_ledger=feature_ledger,
        initialization=initialization, normalized_domain=[-1., 1.], dropout=dropout,
        rank_encoding="stable population rank divided by population-1; [0,1] in every mode",
        diagnostic_names=list(DIAGNOSTIC_NAMES), network_input_columns=input_columns,
        parameter_count=parameter_count, initial_model_sha256=initial_model_sha256,
        model_update_schedule="every_generation; reset AdamW; repeated-best SmoothL1",
        elapsed_s=time.perf_counter()-started, evaluation_s=evaluation_s,
        neural_generation_s=neural_s, neural_update_s=update_s, feature_preparation_s=feature_s,
        upstream_commit=UPSTREAM_COMMIT, upstream_network_blob=UPSTREAM_NETWORK_BLOB,
        algorithm_identity="Diagnostic ABOM adapter; in-memory feature projections; not paper reproduction",
        screening_enabled=False, diagnostic_only_evaluations=0)


__all__ = ["MODES", "DIAGNOSTIC_NAMES", "run_feedback_search"]
