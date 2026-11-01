"""Exact-rational and randomized checks of the paired perturbation energy.

Polynomial Gram entries are computed with fractions.Fraction; the complete
quintic-plus-perturbation energy is also compared numerically in both decoders.
No experiment or frozen optimization source is modified by this script.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

from trajectory_research.curve_task import jerk_energy
from trajectory_research.parameterizations import BoundaryStates, curve_from_parameters, parameter_bounds


F = Fraction
STATE_DRAW_ORDER = ("p0", "p1", "v0", "v1", "a0", "a1")
STATE_BOUNDS = {"p0": (-3., 3.), "p1": (-3., 3.), "v0": (-2., 2.),
                "v1": (-2., 2.), "a0": (-.5, .5), "a1": (-.5, .5)}


def _multiply(a, b):
    result = [F(0)] * (len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] += x*y
    return result


def _differentiate(coefficients, order):
    result = coefficients[:]
    for _ in range(order):
        result = [i*result[i] for i in range(1, len(result))] if len(result) > 1 else [F(0)]
    return result


def _inner(a, b):
    return sum((x*y/F(i+j+1) for i, x in enumerate(a) for j, y in enumerate(b)), F(0))


def _at(coefficients, value):
    return sum((coefficient*value**i for i, coefficient in enumerate(coefficients)), F(0))


def exact_symbolic_check():
    # Derive b from its product definition, rather than copying the decoder's
    # expanded coefficients. c=(u-1/2)*b is a separate multiplication.
    b = _multiply([F(0), F(0), F(0), F(64)],
                  [F(math.comb(3, i)*(-1)**i) for i in range(4)])
    c = _multiply(b, [F(-1, 2), F(1)])
    b3, c3 = _differentiate(b, 3), _differentiate(c, 3)
    gram = [[_inner(x, y) for y in (b3, c3)] for x in (b3, c3)]
    endpoint = {str(order): [[_at(_differentiate(x, order), t) for t in (F(0), F(1))]
                            for x in (b, c)] for order in range(4)}
    cross = []
    for degree in range(6):
        mono = [F(0)]*degree+[F(1)]
        cross.append([_inner(_differentiate(mono, 3), b3), _inner(_differentiate(mono, 3), c3)])
    if gram != [[F(147456, 7), F(0)], [F(0), F(4096)]] \
            or any(value != 0 for row in cross for value in row) \
            or any(value != 0 for order in range(3) for row in endpoint[str(order)] for value in row):
        raise AssertionError("Exact-rational identity failed")
    return b, c, gram, dict(
        polynomial_convention="ascending normalized-u powers",
        b_coefficients=[str(x) for x in b], c_coefficients=[str(x) for x in c],
        b_third_derivative_coefficients=[str(x) for x in b3],
        c_third_derivative_coefficients=[str(x) for x in c3],
        jerk_gram_matrix=[[str(x) for x in row] for row in gram],
        shape_gram_matrix=[[str(_inner(x, y)) for y in (b, c)] for x in (b, c)],
        quintic_basis_jerk_cross_inner_products=[[str(x) for x in row] for row in cross],
        quintic_basis_degrees=list(range(6)),
        endpoint_derivative_values={k: [[str(x) for x in row] for row in rows] for k, rows in endpoint.items()},
        status="PASS", arithmetic="Python fractions.Fraction; no floating-point symbolic integration")


def check_energy(*, seed=20261027, count=100):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 \
            or isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("seed must be nonnegative and count must be positive integers")
    _, _, gram, symbolic = exact_symbolic_check()
    rng = np.random.default_rng(seed)
    lower, upper = parameter_bounds("paired_bump")
    nodes, weights = np.polynomial.legendre.leggauss(8)
    u, weights = (nodes+1.)/2., weights/2.
    rows, passed = [], True
    max_abs, max_relative_scaled, max_quadrature, max_endpoint = 0., 0., 0., 0.
    absolute_tolerance, relative_tolerance = 1e-8, 1e-9
    for case in range(count):
        boundary = BoundaryStates(**{name: rng.uniform(*STATE_BOUNDS[name], 3) for name in STATE_DRAW_ORDER})
        parameters = rng.uniform(lower, upper)
        d, e, duration = parameters[:3], parameters[3:6], float(parameters[-1])
        base = boundary.baseline(duration)
        base_energy = jerk_energy(base)
        expected_increment = (float(gram[0][0])*(d@d)+float(gram[1][1])*(e@e))/duration**5
        cases = []
        for representation in ("hermite", "bezier"):
            curve = curve_from_parameters(parameters, family="paired_bump", representation=representation,
                                           boundary=boundary)
            full_energy = jerk_energy(curve)
            increment = full_energy-base_energy
            difference = increment-expected_increment
            quadrature = duration*float(np.sum(weights*np.sum(curve.evaluate(u, 3, wrt="time")**2, axis=1)))
            endpoint_errors = []
            for endpoint, expected_jerk in ((0., (384.*d-192.*e)/duration**3),
                                             (1., (-384.*d-192.*e)/duration**3)):
                actual = curve.evaluate(endpoint, 3, wrt="time")-base.evaluate(endpoint, 3, wrt="time")
                endpoint_errors.append(float(np.max(np.abs(actual-expected_jerk))))
            valid = bool(np.isclose(increment, expected_increment, rtol=relative_tolerance, atol=absolute_tolerance))
            quadrature_valid = bool(np.isclose(full_energy, quadrature, rtol=relative_tolerance, atol=absolute_tolerance))
            endpoint_valid = max(endpoint_errors) <= 1e-9
            passed = passed and valid and quadrature_valid and endpoint_valid
            max_abs = max(max_abs, abs(difference))
            max_relative_scaled = max(max_relative_scaled, abs(difference)/max(1., abs(expected_increment)))
            max_quadrature = max(max_quadrature, abs(full_energy-quadrature))
            max_endpoint = max(max_endpoint, max(endpoint_errors))
            cases.append(dict(representation=representation, complete_jerk_energy_m2_s5=full_energy,
                observed_same_duration_increment_m2_s5=increment,
                increment_error_m2_s5=difference, complete_energy_gauss_quadrature_m2_s5=quadrature,
                complete_energy_quadrature_error_m2_s5=full_energy-quadrature,
                endpoint_jerk_increment_max_abs_errors_m_s3=endpoint_errors,
                status="PASS" if valid and quadrature_valid and endpoint_valid else "FAIL"))
        rows.append(dict(case_index=case, boundary=asdict(boundary), parameters=parameters.tolist(),
            same_duration_quintic_energy_m2_s5=base_energy,
            analytic_increment_m2_s5=float(expected_increment), decoder_checks=cases))
    root = Path(__file__).resolve().parent
    paths = (Path(__file__).resolve(), root/"trajectory_research/curves.py",
             root/"trajectory_research/curve_task.py", root/"trajectory_research/parameterizations.py")
    return dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=seed, random_case_count=count, decoder_check_count=2*count,
        python_version=platform.python_version(), numpy_version=np.__version__,
        longdouble_precision_bits=int(np.finfo(np.longdouble).nmant+1),
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        symbolic=symbolic,
        sampling=dict(generator="numpy.random.default_rng / PCG64, one sequential stream",
            boundary_state_draw_order=STATE_DRAW_ORDER, boundary_state_component_bounds=STATE_BOUNDS,
            parameter_lower_bounds=lower.tolist(), parameter_upper_bounds=upper.tolist(),
            parameter_order=["dx_m", "dy_m", "dz_m", "e_x_m", "e_y_m", "e_z_m", "duration_s"],
            distribution="Independent uniform physical coordinates; sample a new boundary before each parameter vector"),
        numerical_comparison=dict(absolute_tolerance=absolute_tolerance, relative_tolerance=relative_tolerance,
            full_energy_method="trajectory_research.curve_task.jerk_energy; complete coefficient-product integral",
            independent_energy_check="8-point Gauss-Legendre integration of squared physical jerk; polynomial degree <=8",
            endpoint_jerk_tolerance_m_s3=1e-9,
            max_abs_increment_error_m2_s5=max_abs,
            max_abs_increment_error_scaled_by_max_1_expected=max_relative_scaled,
            max_complete_energy_quadrature_difference_m2_s5=max_quadrature,
            max_endpoint_jerk_increment_abs_error_m_s3=max_endpoint),
        interpretation=dict(identity="E(base+b*d+c*e)-E(base)=(147456/7*norm(d)^2+4096*norm(e)^2)/T^5",
            comparison_baseline="The same physical endpoint p/v/a and the same duration T as the perturbed curve",
            changing_T="Reconstruct the quintic baseline; the identity does not time-scale a fixed baseline's total energy",
            units="d and e: m; e is a normalized-u tangent increment, physical midpoint velocity increment e/T; jerk energy: m^2/s^5",
            equi_energy_scale_ratio_e_over_d=float(math.sqrt(float(gram[0][0]/gram[1][1]))),
            equi_energy_search_policy_status="PROPOSED_ONLY; not tested by this identity check"),
        cases=rows, status="PASS" if passed else "FAIL",
        scope="Exact polynomial algebra plus bounded randomized numerical diagnostics; no geometric feasibility, continuous-interval error bound, flight or optimizer improvement claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/hermite_parameter_perturbation_20261027/energy_identity.json"))
    args = parser.parse_args()
    report = check_energy(seed=args.seed, count=args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], cases=report["random_case_count"],
        maximum_increment_error=report["numerical_comparison"]["max_abs_increment_error_m2_s5"],
        output=str(args.output.resolve()))))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
