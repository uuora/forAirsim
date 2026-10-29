"""A frozen-reference geometric stress task, with no vehicle or controller.

The four search variables are a boundary-preserving displacement in metres
and a segment duration in seconds.  Feasibility concerns a polynomial
reference, convex workspace, sphere clearance and reference derivatives.
It is not UAV task success, physical tracking, wind rejection or a probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .constraints import (
    convex_corridor_certificate, derivative_control_points,
    sphere_clearance_certificate,
)
from .curves import PolynomialSegment, c2_bump, quintic_hermite


PARAMETER_NAMES = ("displacement_x_m", "displacement_y_m", "displacement_z_m", "duration_s")
LOWER_BOUNDS = (-.8, -1.4, -.8, 3.)
UPPER_BOUNDS = (.8, 1.4, .8, 7.)
DEFAULT_PARAMETERS = (0., .8, 0., 5.)
START_POSITION = (0., 0., 5.)
END_POSITION = (8., 0., 5.)
ENDPOINT_VELOCITY = (1., 0., 0.)
ENDPOINT_ACCELERATION = (0., 0., 0.)
WORKSPACE_LOWER = (-.5, -1.8, 3.5)
WORKSPACE_UPPER = (8.5, 1.8, 6.5)
DERIVATIVE_LIMITS = (4., 3., 12.)
SCOPE = "frozen-reference geometric stress; no controller, vehicle, wind, sensing or probability model"


def _finite_numeric(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
            or np.iscomplexobj(value) or not math.isfinite(float(value)) \
            or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _vector(value, size, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {size} finite numbers") from error
    if raw.dtype.kind not in "iuf" or raw.shape != (size,):
        raise ValueError(f"{name} must contain {size} finite numbers")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


@dataclass(frozen=True)
class CurveCase:
    """One declared sphere condition for transferring the same frozen reference."""
    name: str
    centre_m: tuple[float, float, float]
    radius_m: float = .4
    clearance_m: float = .1

    def validate(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("case name must be nonempty")
        _vector(self.centre_m, 3, "sphere centre")
        _finite_numeric(self.radius_m, "radius_m", positive=True)
        _finite_numeric(self.clearance_m, "clearance_m")
        return self


def parameter_bounds():
    """Return independent copies of ordered [dx,dy,dz,T] bounds."""
    return np.array(LOWER_BOUNDS), np.array(UPPER_BOUNDS)


def make_training_curve_cases():
    return tuple(CurveCase(f"training_reference_sphere_x{x:.1f}", (x, 0., 5.))
                 for x in (3.8, 4., 4.2))


def make_validation_curve_cases():
    """Eight unseen frozen-reference stress conditions, not probabilistic trials."""
    conditions = (
        ((3.6, 0., 5.), .4), ((4.4, 0., 5.), .4),
        ((4., .15, 5.), .45), ((4., -.15, 5.), .45),
        ((4., 0., 5.2), .45), ((4., 0., 4.8), .45),
        ((3.9, .25, 5.1), .45), ((4.1, -.25, 4.9), .45),
    )
    return tuple(CurveCase(f"frozen_reference_geometric_stress_{i:02d}", centre, radius)
                 for i, (centre, radius) in enumerate(conditions, 1))


def curve_from_parameters(parameters):
    """Decode [dx,dy,dz,T] while locking both endpoint physical p/v/a."""
    vector = _vector(parameters, 4, "parameters")
    lower, upper = parameter_bounds()
    if np.any(vector < lower) or np.any(vector > upper):
        raise ValueError("curve parameters are outside the declared bounds")
    baseline = quintic_hermite(START_POSITION, ENDPOINT_VELOCITY, ENDPOINT_ACCELERATION,
                               END_POSITION, ENDPOINT_VELOCITY, ENDPOINT_ACCELERATION,
                               float(vector[3]))
    return c2_bump(baseline, vector[:3])


def jerk_energy(segment):
    """Analytically integrate actual physical squared jerk, in m^2/s^5.

    This integrates the complete polynomial coefficient products. It is not
    the bump-only increment and does not omit the baseline's jerk energy.
    """
    if not isinstance(segment, PolynomialSegment):
        raise ValueError("segment must be a PolynomialSegment")
    coefficients = segment.coefficients.astype(np.longdouble)
    for _ in range(3):
        if len(coefficients) == 1:
            return 0.
        coefficients = coefficients[1:] * np.arange(1, len(coefficients))[:, None]
    # Integrate in u first, then multiply by T/T^6.
    integral = np.longdouble(0.)
    for coordinate in coefficients.T:
        square = np.polynomial.polynomial.polymul(coordinate, coordinate)
        integral += np.sum(square / np.arange(1, len(square) + 1, dtype=np.longdouble))
    energy = float(integral / np.longdouble(segment.duration_s) ** 5)
    if not math.isfinite(energy) or energy < -1e-9:
        raise ValueError("squared-jerk integration exceeded numerical range")
    return max(0., energy)


def _split_half(control):
    level = control.copy()
    left, right = [level[0].copy()], [level[-1].copy()]
    while len(level) > 1:
        level = .5 * level[:-1] + .5 * level[1:]
        left.append(level[0].copy())
        right.append(level[-1].copy())
    return np.array(left), np.array(right[::-1])


def subdivided_derivative_norm_bound(segment, order=1, *, depth=3):
    """Max leaf-control norm after de Casteljau; a sufficient reference bound.

    Derivative controls are formed in physical units before subdivision.
    Refinement changes neither the derivative nor its units. This is a
    finite-precision convex-hull enclosure, not an estimate from samples.
    """
    if isinstance(depth, (bool, np.bool_)) or not isinstance(depth, (int, np.integer)) \
            or not 0 <= depth <= 10:
        raise ValueError("depth must be an integer in [0,10]")
    controls = [derivative_control_points(segment, order)]
    for _ in range(int(depth)):
        controls = [part for control in controls for part in _split_half(control)]
    bound = max(math.hypot(*row) for control in controls for row in control)
    if not math.isfinite(bound):
        raise ValueError("derivative norm bound exceeded numerical range")
    return float(bound)


def evaluate_curve_parameters(parameters, cases=None):
    """Evaluate one reference against all declared cases, retaining every failure.

    All five checks in every case must pass. Feasible scores are below one;
    all other scores are at least one, including unresolved certificates.
    """
    vector = _vector(parameters, 4, "parameters")
    segment = curve_from_parameters(vector)
    selected = tuple(make_training_curve_cases() if cases is None else cases)
    if not selected or not all(isinstance(case, CurveCase) for case in selected):
        raise ValueError("cases must be a nonempty sequence of CurveCase objects")
    for case in selected:
        case.validate()
    if len({case.name for case in selected}) != len(selected):
        raise ValueError("case names must be unique")
    normals = np.vstack((np.eye(3), -np.eye(3)))
    offsets = np.concatenate((WORKSPACE_UPPER, -np.array(WORKSPACE_LOWER)))
    corridor = convex_corridor_certificate(segment, normals, offsets, max_depth=6)
    bounds = [subdivided_derivative_norm_bound(segment, order) for order in (1, 2, 3)]
    derivative_checks = dict(zip(("reference_speed", "reference_acceleration", "reference_jerk"),
                                 [bound <= limit for bound, limit in zip(bounds, DERIVATIVE_LIMITS)]))
    rows = []
    for case in selected:
        sphere = sphere_clearance_certificate(segment, case.centre_m, case.radius_m,
                                               clearance=case.clearance_m, max_depth=8)
        checks = dict(workspace=corridor["status"] == "CERTIFIED",
                      sphere_clearance=sphere["status"] == "CERTIFIED", **derivative_checks)
        rows.append(dict(name=case.name, centre_m=list(case.centre_m), radius_m=float(case.radius_m),
                         clearance_m=float(case.clearance_m), passed=all(checks.values()), checks=checks,
                         sphere_certificate=sphere))
    feasible = all(row["passed"] for row in rows)
    failed_checks = sum(not check for row in rows for check in row["checks"].values())
    energy = jerk_energy(segment)
    displacement_norm = float(np.linalg.norm(vector[:3]))
    time_term = .9 * float(vector[3]) / (float(vector[3]) + 5.)
    energy_term = .05 * energy / (energy + 30.)
    displacement_term = .05 * displacement_norm / (displacement_norm + 1.)
    quality = time_term + energy_term + displacement_term
    score = quality if feasible else 1. + failed_checks / (5 * len(rows)) + quality
    details = dict(quality_term=quality, time_term=time_term, energy_term=energy_term,
                   displacement_term=displacement_term, jerk_energy_m2_s5=energy,
                   duration_s=float(vector[3]), displacement_norm_m=displacement_norm,
                   reference_speed_bound_m_s=bounds[0], reference_acceleration_bound_m_s2=bounds[1],
                   reference_jerk_bound_m_s3=bounds[2], derivative_subdivision_depth=3,
                   failed_checks=failed_checks, total_checks=5 * len(rows),
                   failed_cases=sum(not row["passed"] for row in rows),
                   workspace_certificate=corridor,
                   feasibility_rule="all reference certificates and derivative sufficient bounds pass")
    return dict(score=float(score), feasible=bool(feasible), score_details=details,
                parameters=vector.tolist(), cases=rows, scope=SCOPE)
