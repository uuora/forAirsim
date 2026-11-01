"""Budgeted offline search with a separately identified ABOM network adapter.

The official network is loaded from an explicit local source directory. This
module does not vendor it, use MetaBox, or claim to reproduce the ABOM paper.
All algorithms search the same [-1, 1] coordinates and share initial points.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np


UPSTREAM_NETWORK_BLOB = "5959aa7e4aef13c6b128e7c16e51d8a23205d030"
UPSTREAM_COMMIT = "79a2baec700ce0abc6cf7d4cba1e897989d80516"
METHODS = ("random", "gaussian", "abom", "abom_perturb", "abom_triggered",
           "abom_perturb_fixed", "abom_perturb_independent", "abom_perturb_clip")


def reflect_box(points):
    """Reflect finite coordinates into [-1,1], without boundary clipping atoms."""
    x = np.asarray(points, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("candidate coordinates must be finite")
    return 1.0 - np.abs((x + 1.0) % 4.0 - 2.0)


def latin_hypercube(size, dim, rng):
    return np.stack([(rng.permutation(size) + rng.random(size)) / size * 2 - 1
                     for _ in range(dim)], axis=1)


@dataclass
class PerturbationState:
    """Search exploration scale; no physical force compensation is performed."""
    scale: float = 0.15
    stalled: int = 0

    def update(self, success_rate):
        if not math.isfinite(success_rate) or not 0 <= success_rate <= 1:
            raise ValueError("success_rate must lie in [0,1]")
        self.scale = float(np.clip(self.scale * math.exp(0.5 * (success_rate - 0.2)),
                                   0.025, 0.5))
        self.stalled = self.stalled + 1 if success_rate == 0 else 0
        if self.stalled >= 3:
            self.scale = max(self.scale, 0.2)


def paired_perturbation(parents, count, rng, state, *, antithetic=True, boundary="reflect"):
    """Antithetic local proposals around several elites in normalized variables.

    The incumbent is one possible centre, rather than the sole centre. This is
    a generic parameter-search operator. Boundary-preserving physical curve
    deformations are a separate layer, implemented in curves.py.
    """
    p = np.asarray(parents, dtype=float)
    if p.ndim != 2 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("parents must be a nonempty finite matrix")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count % 2:
        raise ValueError("paired proposal count must be an even nonnegative integer")
    if not math.isfinite(state.scale) or not 0 < state.scale <= 1:
        raise ValueError("invalid exploration scale")
    if not isinstance(antithetic, bool) or boundary not in ("reflect", "clip"):
        raise ValueError("invalid pairing or boundary rule")
    half = count // 2
    if half == 0:
        return np.empty((0, p.shape[1]))
    centres = p[np.arange(half) % min(len(p), 4)]
    # A floor prevents a collapsed population from freezing a coordinate.
    widths = np.maximum(np.std(p, axis=0), 0.1)
    widths /= max(float(np.max(widths)), 0.1)
    delta = rng.normal(size=centres.shape) * state.scale * widths
    if antithetic:
        proposals = np.concatenate((centres + delta, centres - delta))
    else:
        other_delta = rng.normal(size=centres.shape) * state.scale * widths
        proposals = np.concatenate((centres + delta, centres + other_delta))
    return reflect_box(proposals) if boundary == "reflect" else np.clip(proposals, -1., 1.)


def load_upstream_network(directory):
    path = Path(directory).resolve() / "abom_net.py"
    content = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
    if blob != UPSTREAM_NETWORK_BLOB:
        raise ValueError("ABOM network differs from the declared upstream snapshot")
    spec = importlib.util.spec_from_file_location("research_pinned_abom_net", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ABOM_net


def run_search(objective: Callable, lower, upper, *, method="random", seed=0,
               population=16, budget=128, anchor=None, upstream=None,
               initialization="uniform", injection_fraction=0.5):
    """Return every evaluation and an incumbent with an exact objective budget.

    An objective result must contain a finite score and a boolean feasible.
    Candidate evaluation exceptions abort visibly; failures are never silently
    replaced with good scores. Validation is deliberately outside this API.
    """
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
    if not math.isfinite(injection_fraction) or not 0 <= injection_fraction <= 1:
        raise ValueError("injection_fraction must lie in [0,1]")
    dim = len(lb)
    init_rng = np.random.default_rng(seed)
    # Separate streams keep initial population identical across methods.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
    if initialization == "lhs":
        parents = latin_hypercube(population, dim, init_rng)
    else:
        parents = init_rng.uniform(-1, 1, (population, dim))
    if anchor is not None:
        a = np.asarray(anchor, dtype=float)
        if a.shape != lb.shape or not np.isfinite(a).all() or np.any(a < lb) or np.any(a > ub):
            raise ValueError("anchor outside bounds")
        parents[0] = 2 * (a - lb) / (ub - lb) - 1
    initial_population = parents.copy()
    records, history = [], []
    state = PerturbationState()
    neural_s = update_s = evaluation_s = 0.0
    started = time.perf_counter()

    def evaluate(points, generation, sources):
        nonlocal evaluation_s
        costs = []
        for point, source in zip(points, sources):
            physical = lb + (point + 1) * 0.5 * (ub - lb)
            tick = time.perf_counter()
            result = objective(physical)
            duration = time.perf_counter() - tick
            evaluation_s += duration
            score = float(result["score"])
            feasible = result["feasible"]
            if not math.isfinite(score) or not isinstance(feasible, (bool, np.bool_)):
                raise ValueError("invalid objective result")
            records.append(dict(fe=len(records) + 1, generation=generation,
                                source=source, parameters=physical.tolist(),
                                score=score, feasible=bool(feasible), evaluation_s=duration,
                                details=result.get("score_details", {})))
            costs.append(score)
        return np.array(costs)

    costs = evaluate(parents, 0, ["initial"] * population)
    order = np.argsort(costs, kind="stable")
    parents, costs = parents[order], costs[order]
    history.append(dict(fe=population, best=float(costs[0]), scale=state.scale))
    model = torch = None
    if method.startswith("abom"):
        if upstream is None:
            raise ValueError("an explicit pinned ABOM source directory is required")
        import torch
        network = load_upstream_network(upstream)
        torch.set_num_threads(1)
        # fork_rng below makes global dropout randomness local to this run.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = network(1, dim, 1 << (dim.bit_length() - 1), .95, .95,
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
            sources = [method] * population
            if model is not None:
                tick = time.perf_counter()
                x = torch.from_numpy(parents.copy()).unsqueeze(0)
                ranks = torch.arange(population, dtype=torch.float64).view(1, population, 1)
                output = model(x, ranks, generation - 1)
                if not torch.isfinite(output).all():
                    raise ValueError("nonfinite neural candidates")
                # Preserve the official loop's clamp for its raw neural proposals.
                children = output.detach().clamp(-1, 1).squeeze(0).numpy().copy()
                neural_s += time.perf_counter() - tick
                if method.startswith("abom_perturb") or method == "abom_triggered":
                    count = int(population * injection_fraction) // 2 * 2
                    # A small, conditional intervention preserves neural candidates
                    # when the search is improving. It adds no objective queries.
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
            child_costs = evaluate(children, generation, sources)
            success_rate = float(np.mean(child_costs < old_best))
            all_points = np.concatenate((parents, children))
            all_costs = np.concatenate((costs, child_costs))
            elite_indices = np.argsort(all_costs, kind="stable")[:population]
            parents, costs = all_points[elite_indices], all_costs[elite_indices]
            if model is not None:
                tick = time.perf_counter()
                # Deliberately preserve source behavior: reset AdamW each generation,
                # and train SmoothL1 against the single best repeated individual.
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
            history.append(dict(fe=len(records), best=float(costs[0]),
                                scale=state.scale, success_rate=success_rate,
                                population_diversity=float(np.mean(np.std(parents, axis=0)))))

    if model is None:
        generation_loop()
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 271828)
            generation_loop()
    best = lb + (parents[0] + 1) * .5 * (ub - lb)
    return dict(method=method, seed=seed, population=population, budget=budget,
                initial_population_normalized=initial_population.tolist(),
                best_parameters=best.tolist(), best_score=float(costs[0]),
                history=history, evaluations=records, initialization=initialization,
                normalized_domain=[-1., 1.], injection_fraction=injection_fraction,
                elapsed_s=time.perf_counter() - started, evaluation_s=evaluation_s,
                neural_generation_s=neural_s, neural_update_s=update_s,
                upstream_commit=UPSTREAM_COMMIT if model is not None else None,
                algorithm_identity="Pinned ABOM-net adapter, not a paper reproduction" if model is not None
                else "Generic search baseline")
