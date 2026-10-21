"""Closed-loop objective for offline tuning of the single-drone planner.

Candidate curves are scored by running the production controller against a
lagged point mass.  The independent sphere-entry and command-limit checks live
in ``validate_dual_target_turning`` and are shared with the regression runner.
This model does not establish AirSim or real-flight performance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import numpy as np

from . import core
from validate_dual_target_turning import (
    INTERVALS, SCENARIOS, YAW_LAGS, TurningScenario, run_scenario, summarize,
)


MODEL = "deterministic first-order lagged point mass; not AirSim or real flight"
LIMITATIONS = (
    "No rotor, thrust, attitude, collision, wind or sensing model. "
    "A passing prediction is not a physical safety guarantee."
)


@dataclass(frozen=True)
class EvaluationCase:
    """A complete rollout condition; None intervals means use evaluation dt."""

    scenario: TurningScenario
    velocity_lag_s: float = 0.30
    yaw_lag_s: float = 0.25
    intervals_s: tuple[float, ...] | None = None

    def validate(self) -> "EvaluationCase":
        for name, value in (("velocity_lag_s", self.velocity_lag_s),
                            ("yaw_lag_s", self.yaw_lag_s),
                            ("deadline", self.scenario.deadline)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
                    or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.intervals_s is not None:
            if not self.intervals_s:
                raise ValueError("intervals_s must be nonempty")
            for value in self.intervals_s:
                _validate_dt(value)
        return self


def _validate_dt(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
            or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("dt must be finite and positive")
    return float(value)


def _close_range_scenario(seed: int) -> TurningScenario:
    """Generate a reproducible pair using the deployment's 6–12 m spacing."""
    settings = dict(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0,
                    height_min=2.0, height_max=25.0, margin=1.0,
                    arrival_radius=0.5, max_speed=4.0, max_acceleration=4.0,
                    max_yaw_rate_deg=120.0, control_hz=50, display_hz=20,
                    heading_tolerance_deg=20.0, response_time_s=0.35,
                    target_distance_min=6.0, target_distance_max=12.0, seed=seed)
    controller = core.RollingController(core.ControlConfig(**settings).validate())
    controller.reset((0.0, 0.0, 8.0), (0.0, 0.0, 0.0), 0.0)
    return TurningScenario(
        f"close_range_seed_{seed}", (0.0, 0.0, 8.0), (0.0, 0.0, 0.0),
        tuple(tuple(float(v) for v in target) for target in controller.targets),
        settings, deadline=45.0,
    )


def make_training_cases() -> tuple[EvaluationCase, ...]:
    """Small declared tuning set; no validation-only seeds enter this set."""
    return (
        EvaluationCase(_close_range_scenario(13), 0.30, 0.25),
        EvaluationCase(replace(SCENARIOS[1], deadline=45.0), 0.35, 0.35),
        EvaluationCase(replace(SCENARIOS[2], deadline=45.0), 0.35, 0.35),
    )


def make_validation_cases() -> tuple[EvaluationCase, ...]:
    """Original 24 regressions plus six held-out close-range rollout conditions.

    The regression layouts include two tuning layouts, but their lag/interval
    combinations differ.  Only the additional seeds 17, 23, and 31 are described
    as held out.  Call this set after selecting a candidate, not during tuning.
    """
    regression = tuple(EvaluationCase(scenario, 0.30, yaw_lag, intervals)
                       for scenario in SCENARIOS
                       for yaw_lag in YAW_LAGS
                       for intervals in INTERVALS)
    held_out = tuple(EvaluationCase(_close_range_scenario(seed), velocity_lag, yaw_lag, intervals)
                     for seed in (17, 23, 31)
                     for velocity_lag, yaw_lag, intervals in (
                         (0.30, 0.25, (0.02,)),
                         (0.35, 0.35, (0.02, 0.02, 0.04, 0.06)),
                     ))
    return regression + held_out


def _score_rows(rows: list[dict]) -> tuple[float, bool, dict]:
    """Bound the quality term so every feasible rollout set ranks first."""
    feasible = all(row["passed"] for row in rows)
    mean_time = float(np.mean([row["elapsed_s"] for row in rows]))
    # The acceleration limiter bounds command jerk by approximately 2*a/dt.
    # This dimensionless peak proxy is intentionally a small secondary term;
    # measured vehicle jerk and energy are not available in this model.
    jerk_ratios = [row["maxima"]["command_jerk_mps3"] /
                   (2.0 * row["control_config"]["max_acceleration"] / min(row["intervals_s"]))
                   for row in rows]
    smoothness = float(np.mean(jerk_ratios))
    quality = 0.99 * (mean_time / (10.0 + mean_time)) + 0.001 * (smoothness / (1.0 + smoothness))
    failed_checks = sum(not passed for row in rows for passed in row["checks"].values())
    total_checks = sum(len(row["checks"]) for row in rows)
    failed_cases = sum(not row["passed"] for row in rows)
    # Feasible scores are strictly below .991.  Infeasible scores are >=1,
    # independently of deadlines, time units, or how quickly a failure occurs.
    score = quality if feasible else 1.0 + failed_cases / len(rows) + failed_checks / total_checks + quality
    details = dict(mean_elapsed_s=mean_time, mean_normalized_peak_command_jerk=smoothness,
                   time_weight=0.99, smoothness_weight=0.001, time_scale_s=10.0,
                   failed_cases=failed_cases, failed_checks=failed_checks,
                   total_checks=total_checks, quality_term=quality,
                   feasibility_rule="all independent regression checks must pass")
    return float(score), bool(feasible), details


def evaluate_parameters(parameters, cases=None, *, dt: float = 0.05) -> dict:
    """Evaluate a BézierParameters object or ordered five-value vector.

    Parameters remain fixed across the two original target legs in each case.
    Each new candidate starts from the exact same measured state, target queue,
    limits and RNG seed.  Generated later targets still affect the production
    controller's preview, but the rollout stops after the two original targets.
    """
    dt = _validate_dt(dt)
    if isinstance(parameters, core.BezierParameters):
        selected = parameters.validate()
    else:
        selected = core.BezierParameters.from_vector(parameters)
    chosen_cases = tuple(make_training_cases() if cases is None else cases)
    if not chosen_cases:
        raise ValueError("at least one evaluation case is required")
    for case in chosen_cases:
        if not isinstance(case, EvaluationCase):
            raise ValueError("cases must contain EvaluationCase objects")
        case.validate()
    bound_core = SimpleNamespace(
        ControlConfig=core.ControlConfig,
        RollingController=lambda config: core.RollingController(config, parameters=selected),
    )
    rows = [run_scenario(bound_core, case.scenario,
                         velocity_lag=case.velocity_lag_s, yaw_lag=case.yaw_lag_s,
                         intervals=case.intervals_s if case.intervals_s is not None else (dt,))
            for case in chosen_cases]
    score, feasible, details = _score_rows(rows)
    return dict(score=score, feasible=feasible, parameters=selected.to_vector().tolist(),
                model=MODEL, limitations=LIMITATIONS,
                score_details=details, summary=summarize(rows), cases=rows)
