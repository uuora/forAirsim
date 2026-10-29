"""Reproduce finite-precision Hermite/Bézier decoder consistency checks.

The native Bernstein comparison differentiates control polygons and evaluates
them by de Casteljau, separately from the decoders' common power evaluator.
Random checks on a fixed grid are numerical diagnostics, not formal bounds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from trajectory_research.parameterizations import (
    BoundaryStates, FAMILIES, curve_from_parameters, parameter_bounds,
    parameter_names, physical_from_normalized,
)


STATE_DRAW_ORDER = ("p0", "p1", "v0", "v1", "a0", "a1")
STATE_BOUNDS = {"p0": (-3., 3.), "p1": (-3., 3.), "v0": (-2., 2.),
                "v1": (-2., 2.), "a0": (-.5, .5), "a1": (-.5, .5)}
DERIVATIVE_NAMES = ("position_m", "velocity_m_s", "acceleration_m_s2", "jerk_m_s3")


def _native_control(boundary, parameters, family):
    """Construct the Bernstein polygon directly, without power conversion."""
    degree = 6 if family == "single_bump" else 7
    control = boundary.baseline(float(parameters[-1])).bezier_control_points(target_degree=degree)
    d = parameters[:3]
    if family == "single_bump":
        control[3] += (64. / 20.) * d
    else:
        e = parameters[3:6]
        control[3] += 64. / 35. * d - 32. / 35. * e
        control[4] += 64. / 35. * d + 32. / 35. * e
    return control


def _bernstein_physical_derivative(control, u, order, duration):
    """Differentiate a polygon n*diff(B)/T, then evaluate via de Casteljau."""
    values = control.copy()
    for _ in range(order):
        values = (len(values)-1) * np.diff(values, axis=0) / duration
    values = np.broadcast_to(values, (len(u), *values.shape)).copy()
    parameter = u[:, None, None]
    for degree in range(values.shape[1]-1, 0, -1):
        values[:, :degree] = (1.-parameter) * values[:, :degree] + parameter * values[:, 1:degree+1]
    return values[:, 0]


def _metric(names):
    return {name: dict(max_abs=0., witness=None) for name in names}


def _update(metric, name, difference, witness):
    error = float(np.max(np.abs(difference)))
    if error > metric[name]["max_abs"]:
        metric[name] = dict(max_abs=error, witness=witness)


def check_equivalence(*, seed=71027, cases_per_family=250, grid_points=129):
    """Return a reproducible report with sampling domains and worst witnesses."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for value, minimum, name in ((cases_per_family, 1, "cases_per_family"),
                                  (grid_points, 3, "grid_points")):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    rng = np.random.default_rng(seed)
    u = np.linspace(0., 1., grid_points)
    family_reports = []
    global_decoders = _metric(DERIVATIVE_NAMES)
    global_native = _metric(DERIVATIVE_NAMES)
    global_endpoints = _metric(DERIVATIVE_NAMES[:3])
    for family in FAMILIES:
        decoder_difference = _metric(DERIVATIVE_NAMES)
        native_difference = _metric(DERIVATIVE_NAMES)
        endpoint_residual = {name: _metric(DERIVATIVE_NAMES[:3])
                             for name in ("hermite_power", "bezier_power", "native_bernstein")}
        lower, upper = parameter_bounds(family)
        for index in range(cases_per_family):
            values = {name: rng.uniform(*STATE_BOUNDS[name], 3) for name in STATE_DRAW_ORDER}
            boundary = BoundaryStates(**values)
            normalized = rng.uniform(-1., 1., len(parameter_names(family)))
            parameters = physical_from_normalized(normalized, family=family)
            hermite = curve_from_parameters(parameters, family=family,
                                            representation="hermite", boundary=boundary)
            bezier = curve_from_parameters(parameters, family=family,
                                           representation="bezier", boundary=boundary)
            native_control = _native_control(boundary, parameters, family)
            witness = dict(family=family, case_index=index,
                           parameters=parameters.tolist(), normalized_parameters=normalized.tolist(),
                           boundary=asdict(boundary))
            for order, name in enumerate(DERIVATIVE_NAMES):
                h = hermite.evaluate(u, order, wrt="time")
                b = bezier.evaluate(u, order, wrt="time")
                n = _bernstein_physical_derivative(native_control, u, order, float(parameters[-1]))
                _update(decoder_difference, name, h-b, witness)
                _update(global_decoders, name, h-b, witness)
                _update(native_difference, name, h-n, witness)
                _update(global_native, name, h-n, witness)
                if order < 3:
                    expected = np.array((values[("p0", "v0", "a0")[order]],
                                         values[("p1", "v1", "a1")[order]]))
                    for construction, evaluated in (("hermite_power", h), ("bezier_power", b),
                                                      ("native_bernstein", n)):
                        residual = evaluated[[0, -1]] - expected
                        _update(endpoint_residual[construction], name, residual, witness)
                        _update(global_endpoints, name, residual, dict(construction=construction, **witness))
        family_reports.append(dict(
            family=family, case_count=cases_per_family, parameter_names=parameter_names(family),
            parameter_lower_bounds=lower.tolist(), parameter_upper_bounds=upper.tolist(),
            decoder_power_evaluation_difference=decoder_difference,
            hermite_power_vs_native_bernstein_difference=native_difference,
            endpoint_physical_state_residuals=endpoint_residual))
    source_root = Path(__file__).resolve().parent
    source_files = (Path(__file__).resolve(), source_root / "trajectory_research/parameterizations.py",
                    source_root / "trajectory_research/curves.py")
    maximum = max(value["max_abs"] for metric in (global_decoders, global_native, global_endpoints)
                  for value in metric.values())
    tolerance = 1e-9
    return dict(
        schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, family_order=list(FAMILIES), cases_per_family=cases_per_family,
        total_case_count=cases_per_family*len(FAMILIES), numpy_version=np.__version__,
        python_version=platform.python_version(),
        random_generator="numpy.random.default_rng / PCG64; one shared sequential stream",
        sampling=dict(normalized_parameter_domain=[-1., 1.], distribution="independent uniform coordinates",
                      boundary_state_draw_order=STATE_DRAW_ORDER,
                      boundary_state_component_bounds=STATE_BOUNDS,
                      boundary_state_dimension=3,
                      boundary_units=dict(p0="m", p1="m", v0="m/s", v1="m/s", a0="m/s^2", a1="m/s^2")),
        sample_grid=dict(parameter="u=t/T", start=0., end=1., count=grid_points,
                         construction="numpy.linspace(0,1,count), endpoints included"),
        derivative_methods=dict(
            decoder_comparison="Hermite direct polynomial and Bezier-to-power decoder, both differentiated analytically in the power basis",
            independent_basis_comparison="Hermite power derivatives vs native Bernstein controls differentiated as n*diff(B)/T and evaluated by de Casteljau",
            orders=[0, 1, 2, 3], parameter_derivative_to_time_scaling="T**(-order)"),
        source_sha256={str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in source_files},
        families=family_reports,
        overall_decoder_power_evaluation_difference=global_decoders,
        overall_hermite_power_vs_native_bernstein_difference=global_native,
        overall_endpoint_physical_state_residuals=global_endpoints,
        diagnostic_tolerance=tolerance, maximum_observed_abs_error=maximum,
        status="PASS" if maximum <= tolerance else "FAIL",
        scope="finite-precision randomized diagnostic on the recorded parameter boxes and grid; no formal, continuous-interval, optimizer-performance or physical-tracking guarantee")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=71027)
    parser.add_argument("--cases-per-family", type=int, default=250)
    parser.add_argument("--grid-points", type=int, default=129)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hermite_parameter_perturbation_20261027/decoder_equivalence.json"))
    args = parser.parse_args()
    report = check_equivalence(seed=args.seed, cases_per_family=args.cases_per_family,
                               grid_points=args.grid_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], cases=report["total_case_count"],
                          maximum_abs_error=report["maximum_observed_abs_error"],
                          output=str(args.output.resolve()))))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
