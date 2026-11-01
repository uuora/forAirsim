"""Directional sufficient clearance checks for an uncertain spherical obstacle.

The obstacle centre is c + L*z, ||z|| <= 1, and its radius is no larger
than r + radius_increment. For a separating direction n with ||n|| <= 1, the centre
uncertainty has support h(n) = ||L.T*n||.  If every control point B of a
Bezier leaf satisfies

    n.(B-c) > r + radius_increment + clearance + ||L.T*n||,

then every point of that leaf has at least the requested clearance from every
sphere in this deterministic error envelope.  This follows from the Bezier
convex hull, Cauchy--Schwarz, and ||x|| >= n.x.  The direction may differ across
leaves; the quantified centre-error set is unchanged.

Only candidate separating directions are tried. Failure to separate is not
proof of collision. Depth exhaustion is UNRESOLVED; VIOLATED requires an
explicit curve endpoint and one realizable centre perturbation as a witness.
These are conservative, finite-precision polynomial-reference checks, with
an explicit numeric margin. They are not interval arithmetic and supply no
vehicle tracking, wind compensation, or probability guarantee.
"""
from __future__ import annotations

import math

import numpy as np

from .curves import PolynomialSegment


def _array(value, name, ndim):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim or not raw.size:
        raise ValueError(f"{name} must be a nonempty numeric array of dimension {ndim}")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
            or np.iscomplexobj(value) or not math.isfinite(float(value)) \
            or (value <= 0 if positive else value < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _norm(value):
    result = math.hypot(*value)
    if not math.isfinite(result):
        raise ValueError("clearance computation exceeded finite numeric range")
    return result


def _contracted_unit(vector, length):
    """Keep numerical unit vectors slightly inside the closed unit ball."""
    return (vector / length) * (1.0 - 8.0 * np.finfo(float).eps)


def _split_half(control):
    level = control.copy()
    left, right = [level[0].copy()], [level[-1].copy()]
    while len(level) > 1:
        level = .5 * level[:-1] + .5 * level[1:]
        left.append(level[0].copy())
        right.append(level[-1].copy())
    return np.array(left), np.array(right[::-1])


def _candidate_directions(control, centre):
    # A small set suffices for a useful conservative check; completeness is not
    # assumed. The mean uses division before summation to avoid avoidable overflow.
    offsets = (np.sum(control / len(control), axis=0) - centre,
               control[0] - centre, control[-1] - centre)
    result = []
    for offset in offsets:
        length = _norm(offset)
        if length > 0:
            result.append(_contracted_unit(offset, length))
    return result


def _point_witness(point, parameter, centre, factor, inflated, tolerance):
    """Try explicit admissible centres; failure to find one proves nothing."""
    z_values = [np.zeros(factor.shape[1])]
    offset = point - centre
    length = _norm(offset)
    if length > 0:
        dual = factor.T @ _contracted_unit(offset, length)
        support = _norm(dual)
        if support > 0:
            # z is in the closed unit ball and L*z is therefore realizable.
            z_values.append(_contracted_unit(dual, support))
    for z in z_values:
        perturbation = factor @ z
        actual_centre = centre + perturbation
        distance = _norm(point - actual_centre)
        if distance < inflated - tolerance:
            return dict(u=float(parameter), position=point.tolist(),
                        uncertainty_coordinates=z.tolist(),
                        uncertainty_coordinate_norm=_norm(z),
                        centre_perturbation=perturbation.tolist(),
                        obstacle_centre=actual_centre.tolist(), distance_to_centre=distance,
                        inflated_radius=inflated)
    return None


def ellipsoid_sphere_certificate(segment, centre, radius, uncertainty_factor, *,
                                 radius_increment=.05, clearance=.1, max_depth=10,
                                 numerical_margin=1e-10):
    """Check a reference against all spheres in a bounded centre-error ellipsoid.

    ``uncertainty_factor`` has shape (spatial_dimension, latent_dimension).
    It may be singular or zero; no inverse is used. Positions and its entries
    use metres. The latent vector z is dimensionless and has Euclidean norm <=1.
    The returned minimum margin is over certified leaf separating planes,
    rather than an exact minimum Euclidean clearance.
    """
    if not isinstance(segment, PolynomialSegment):
        raise ValueError("segment must be a PolynomialSegment")
    centre = _array(centre, "centre", 1)
    factor = _array(uncertainty_factor, "uncertainty_factor", 2)
    dimension = segment.coefficients.shape[1]
    if centre.shape != (dimension,) or factor.shape[0] != dimension:
        raise ValueError("centre and uncertainty_factor must match the segment dimension")
    radius = _scalar(radius, "radius", positive=True)
    increment = _scalar(radius_increment, "radius_increment")
    clearance = _scalar(clearance, "clearance")
    tolerance = _scalar(numerical_margin, "numerical_margin")
    if isinstance(max_depth, (bool, np.bool_)) or not isinstance(max_depth, (int, np.integer)) \
            or not 0 <= max_depth <= 20:
        raise ValueError("max_depth must be an integer in [0,20]")
    inflated = radius + increment + clearance
    if not math.isfinite(inflated + tolerance):
        raise ValueError("inflated radius exceeded finite numeric range")
    controls = segment.bezier_control_points()
    pending = [(controls, 0., 1., 0)]
    unresolved, certified, witness = [], [], None
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            while pending:
                control, start, end, depth = pending.pop()
                best = None
                for direction in _candidate_directions(control, centre):
                    support = _norm(factor.T @ direction)
                    lower = float(np.min((control - centre) @ direction))
                    margin = lower - inflated - support
                    if not math.isfinite(margin):
                        raise ValueError("separating margin exceeded finite numeric range")
                    if margin > tolerance and (best is None or margin > best["separating_margin_m"]):
                        best = dict(interval=[start, end], normal=direction.tolist(),
                                    projection_lower_bound_m=lower, uncertainty_support_m=support,
                                    separating_margin_m=margin)
                if best is not None:
                    certified.append(best)
                    continue
                for index, parameter in ((0, start), (-1, end)):
                    witness = _point_witness(control[index], parameter, centre, factor, inflated, tolerance)
                    if witness is not None:
                        break
                if witness is not None:
                    break
                if depth == max_depth:
                    unresolved.append([start, end])
                else:
                    left, right = _split_half(control)
                    middle = .5 * (start + end)
                    pending.extend(((right, middle, end, depth + 1),
                                    (left, start, middle, depth + 1)))
    except FloatingPointError as error:
        raise ValueError("clearance computation exceeded finite numeric range") from error
    status = "VIOLATED" if witness is not None else ("UNRESOLVED" if unresolved else "CERTIFIED")
    return dict(status=status, certified_leaves=len(certified),
                leaf_certificates=certified, unresolved_intervals=unresolved,
                violation_witness=witness, max_depth=int(max_depth), numerical_margin=tolerance,
                minimum_separating_margin_m=min((r["separating_margin_m"] for r in certified), default=None),
                inflated_radius_without_centre_uncertainty=inflated,
                uncertainty_factor=factor.tolist(),
                scope="continuous polynomial reference; bounded ellipsoid centre error; finite-precision support-plane sufficient condition")
