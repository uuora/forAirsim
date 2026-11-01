"""Admissible partial evaluation of the existing five-parameter control task.

The production controller and objective are unchanged. Whole deterministic
point-mass scenarios may be skipped after the observed prefix proves that a
candidate cannot enter an elite set. No result is imputed for skipped scenarios.
This is not an AirSim, rotor-level dynamics, tracking-RMSE, or flight test.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np

from dual_target_control import core
from dual_target_control import trajectory_objective as original


NUMERICAL_CUSHION = 1e-12


def parameter_bounds():
    """Return copies of the existing production planner's search bounds."""
    return core.BezierParameters.lower_bounds(), core.BezierParameters.upper_bounds()


def default_parameters():
    return core.BezierParameters().to_vector()


def _finite_nonnegative(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
            or not np.isrealobj(value) or not math.isfinite(float(value)) \
            or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def partial_score_lower_bound(rows, total_cases):
    """Bound the FINAL aggregate score using only completed cases.

    Let S_t and S_j be the observed sums of elapsed time and normalized peak
    command jerk, and N the fixed total case count. Unobserved contributions
    are nonnegative, so mean_t >= S_t/N and mean_j >= S_j/N. Both transforms
    in the original quality term are monotone on the nonnegative domain.

    If f observed cases failed, the final score is at least
    1 + f/N + Q_lower. The nonnegative failed-check fraction is omitted.
    If f=0, use Q_lower: unseen cases might pass or fail. The numerical cushion
    protects ordinary floating arithmetic; this is not interval arithmetic.
    """
    if isinstance(total_cases, (bool, np.bool_)) or not isinstance(total_cases, (int, np.integer)) \
            or total_cases < 1:
        raise ValueError("total_cases must be a positive integer")
    rows = tuple(rows)
    if len(rows) > total_cases:
        raise ValueError("completed cases exceed total_cases")
    elapsed, jerks, failed = [], [], 0
    for row in rows:
        if not isinstance(row['passed'], (bool, np.bool_)):
            raise ValueError("case passed flag must be boolean")
        elapsed.append(_finite_nonnegative(row['elapsed_s'], 'case elapsed time'))
        jerk = _finite_nonnegative(row['maxima']['command_jerk_mps3'], 'command jerk')
        acceleration = _finite_nonnegative(row['control_config']['max_acceleration'],
                                            'maximum acceleration', positive=True)
        intervals = tuple(row['intervals_s'])
        if not intervals:
            raise ValueError("case intervals must be nonempty")
        interval = min(_finite_nonnegative(x, 'case interval', positive=True) for x in intervals)
        # Match the original scorer's operation ordering.
        ratio = jerk / (2.0 * acceleration / interval)
        if not math.isfinite(ratio):
            raise ValueError("normalized command jerk must be finite")
        jerks.append(ratio)
        failed += not row['passed']
    mean_time = math.fsum(elapsed) / total_cases
    mean_jerk = math.fsum(jerks) / total_cases
    if not math.isfinite(mean_time) or not math.isfinite(mean_jerk):
        raise ValueError("partial aggregate exceeded finite range")
    lower = .99 * (mean_time / (10.0 + mean_time)) \
        + .001 * (mean_jerk / (1.0 + mean_jerk))
    if failed:
        lower += 1.0 + failed / total_cases
    return float(lower - NUMERICAL_CUSHION)


def _case_order(value, size):
    if value is None:
        return tuple(range(size))
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError("case_order must be a permutation of case indices") from error
    if len(result) != size or any(isinstance(x, (bool, np.bool_))
                                or not isinstance(x, (int, np.integer)) for x in result) \
            or sorted(result) != list(range(size)):
        raise ValueError("case_order must be a permutation of case indices")
    return tuple(int(x) for x in result)


def evaluate_control_bounded(parameters, cutoff=None, cases=None, *, dt=.05, case_order=None):
    """Evaluate all scenarios, or reject by a conservative partial-score bound.

    ``cutoff`` is a finite worst-retained-elite score for minimization. Equality
    assumes stable ties keep existing parents. None always runs all cases.
    ``case_order`` only changes execution order; completed full rows are scored
    in the original order for bit-identical objective arithmetic.

    ``case_observations`` describes observed work only. PRUNED results have
    nullable score/feasible fields; they do not assert aggregate feasibility.
    """
    started = time.perf_counter()
    dt = original._validate_dt(dt)
    if cutoff is not None:
        if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, float, np.number)) \
                or not np.isrealobj(cutoff) or not math.isfinite(float(cutoff)):
            raise ValueError("cutoff must be finite or None")
        cutoff = float(cutoff)
    selected = parameters.validate() if isinstance(parameters, core.BezierParameters) \
        else core.BezierParameters.from_vector(parameters)
    chosen = tuple(original.make_training_cases() if cases is None else cases)
    if not chosen:
        raise ValueError("at least one evaluation case is required")
    for case in chosen:
        if not isinstance(case, original.EvaluationCase):
            raise ValueError("cases must contain EvaluationCase objects")
        case.validate()
    order = _case_order(case_order, len(chosen))
    bound_core = SimpleNamespace(ControlConfig=core.ControlConfig,
        RollingController=lambda config: core.RollingController(config, parameters=selected))
    observed, observations = {}, []
    for index in order:
        case = chosen[index]
        tick = time.perf_counter()
        row = original.run_scenario(bound_core, case.scenario,
            velocity_lag=case.velocity_lag_s, yaw_lag=case.yaw_lag_s,
            intervals=case.intervals_s if case.intervals_s is not None else (dt,))
        wall = time.perf_counter()-tick
        observed[index] = row
        observations.append(dict(case_index=index, passed=bool(row['passed']), wall_s=wall))
        rows = [observed[i] for i in sorted(observed)]
        lower = partial_score_lower_bound(rows, len(chosen))
        if len(observed) < len(chosen) and cutoff is not None and lower >= cutoff:
            return dict(status='PRUNED', score=None, feasible=None, lower_bound=lower,
                parameters=selected.to_vector().tolist(), model=original.MODEL,
                limitations=original.LIMITATIONS, cases=rows, case_observations=observations,
                completed_cases=len(observed), total_cases=len(chosen), case_order=list(order),
                evaluation_wall_s=time.perf_counter()-started,
                score_details=dict(completed_cases=len(observed), total_cases=len(chosen),
                    observed_failed_cases=sum(not r['passed'] for r in rows),
                    lower_bound=lower, cutoff=cutoff, numerical_cushion=NUMERICAL_CUSHION,
                    status='PRUNED', bound='monotone quality plus observed failed-case fraction; check penalty omitted'))
    rows = [observed[i] for i in range(len(chosen))]
    score, feasible, details = original._score_rows(rows)
    if not math.isfinite(score) or lower > score:
        raise ValueError("partial bound inconsistent with completed objective")
    return dict(status='EVALUATED', score=score, feasible=feasible,
        parameters=selected.to_vector().tolist(), model=original.MODEL,
        limitations=original.LIMITATIONS, score_details=details,
        summary=original.summarize(rows), cases=rows, case_observations=observations,
        completed_cases=len(observed), total_cases=len(chosen), case_order=list(order),
        evaluation_wall_s=time.perf_counter()-started)
