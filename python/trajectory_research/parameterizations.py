"""Endpoint-preserving search coordinates for polynomial curve perturbations.

The canonical variables are a midpoint displacement ``d`` in metres, a
midpoint *u*-tangent increment ``e`` in metres, and physical duration ``T`` in
seconds.  ``u=t/T``, so the corresponding midpoint velocity increment is e/T.
The paired family is the same degree-seven curve space in both representations:

    q(u) = p_H(u; T) + 64*u**3*(1-u)**3 * (d + (u-.5)*e).

The Bézier implementation only changes the two interior degree-seven controls
P3/P4; P0..P2 and P5..P7, and hence endpoint physical p/v/a, remain fixed.
Both decoders use the same canonical parameter box.  Different boxes in native
control-point coordinates would define different search domains, not evidence
for one curve basis being better.  This module provides reference geometry;
feasibility and vehicle tracking are outside its contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import PolynomialSegment, bezier_segment, c2_bump, quintic_hermite


FAMILIES = ("single_bump", "paired_bump")
REPRESENTATIONS = ("hermite", "bezier")
DISPLACEMENT_LIMITS_M = (.8, 1.4, .8)
TANGENT_LIMITS_M = (2., 2., 2.)
DURATION_BOUNDS_S = (3., 7.)


def _vector(value, size, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {size} finite real numbers") from error
    if raw.dtype.kind not in "iuf" or raw.shape != (size,):
        raise ValueError(f"{name} must contain {size} finite real numbers")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite real numbers")
    return result


def _family(family):
    if not isinstance(family, str) or family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    return family


def _representation(representation):
    if not isinstance(representation, str) or representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}")
    return representation


@dataclass(frozen=True)
class BoundaryStates:
    """Six physical 3-D endpoint states; tuples are copied and finite-checked.

    Positions are in metres, velocities in m/s, accelerations in m/s**2.
    A new baseline is constructed for each T to preserve these physical states.
    """

    p0: tuple[float, float, float] = (0., 0., 5.)
    v0: tuple[float, float, float] = (1., 0., 0.)
    a0: tuple[float, float, float] = (0., 0., 0.)
    p1: tuple[float, float, float] = (8., 0., 5.)
    v1: tuple[float, float, float] = (1., 0., 0.)
    a1: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        for name in ("p0", "v0", "a0", "p1", "v1", "a1"):
            object.__setattr__(self, name, tuple(float(x) for x in _vector(getattr(self, name), 3, name)))

    def baseline(self, duration_s):
        """Return a quintic Hermite baseline with the supplied physical duration."""
        return quintic_hermite(self.p0, self.v0, self.a0,
                               self.p1, self.v1, self.a1, duration_s)


def parameter_names(family="paired_bump"):
    """Canonical variable names; e is explicitly a normalized-u tangent."""
    _family(family)
    names = ("displacement_x_m", "displacement_y_m", "displacement_z_m")
    if family == "paired_bump":
        names += ("midpoint_u_tangent_x_m", "midpoint_u_tangent_y_m", "midpoint_u_tangent_z_m")
    return names + ("duration_s",)


def parameter_bounds(family="paired_bump"):
    """Return independent arrays for the canonical physical parameter box."""
    _family(family)
    amplitudes = np.array(DISPLACEMENT_LIMITS_M + (
        TANGENT_LIMITS_M if family == "paired_bump" else ()))
    return np.r_[-amplitudes, DURATION_BOUNDS_S[0]], np.r_[amplitudes, DURATION_BOUNDS_S[1]]


def default_parameters(family="paired_bump"):
    """A common initialization anchor; feasibility is decided by the caller."""
    _family(family)
    return np.array((0., 1.15, 0.) + ((0., 0., 0.) if family == "paired_bump" else ()) + (5.5,))


def _physical(parameters, family):
    lower, upper = parameter_bounds(family)
    parameters = _vector(parameters, len(lower), "parameters")
    if np.any(parameters < lower) or np.any(parameters > upper):
        raise ValueError("parameters are outside the declared canonical bounds")
    return parameters


def physical_from_normalized(candidate, *, family="paired_bump"):
    """Map a finite canonical [-1,1] vector to physical search coordinates."""
    lower, upper = parameter_bounds(family)
    candidate = _vector(candidate, len(lower), "normalized candidate")
    if np.any(candidate < -1.) or np.any(candidate > 1.):
        raise ValueError("normalized candidate must lie in [-1,1]")
    # This convex combination preserves exact declared endpoints and avoids
    # upper-bound overshoots from lower + fraction*(upper-lower).
    weight = .5 * (candidate + 1.)
    return (1. - weight) * lower + weight * upper


def normalized_from_physical(parameters, *, family="paired_bump"):
    """Map canonical physical coordinates back to their common [-1,1] box."""
    parameters = _physical(parameters, family)
    lower, upper = parameter_bounds(family)
    return 2. * (parameters - lower) / (upper - lower) - 1.


def middle_state_to_bezier_offsets(displacement, u_tangent):
    """Return (delta_P3,delta_P4) for a degree-seven endpoint-C2 perturbation.

    At u=.5 the two Bernstein values are 35/128 and their derivatives are
    -35/64,+35/64.  Inverting these two equations gives the stated map.
    This mapping is linear and does not impose a different control-point box.
    """
    d = _vector(displacement, 3, "midpoint displacement")
    e = _vector(u_tangent, 3, "midpoint u tangent")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return (64. / 35. * d - 32. / 35. * e,
                    64. / 35. * d + 32. / 35. * e)
        except FloatingPointError as error:
            raise ValueError("control-point mapping exceeded finite range") from error


def bezier_offsets_to_middle_state(delta_p3, delta_p4):
    """Inverse map to (midpoint displacement, normalized-u tangent increment)."""
    p3 = _vector(delta_p3, 3, "delta P3")
    p4 = _vector(delta_p4, 3, "delta P4")
    with np.errstate(over="raise", invalid="raise"):
        try:
            # Scale before addition/subtraction to avoid gratuitous overflow.
            return (35. / 128. * p3 + 35. / 128. * p4,
                    -35. / 64. * p3 + 35. / 64. * p4)
        except FloatingPointError as error:
            raise ValueError("midpoint-state mapping exceeded finite range") from error


def curve_from_parameters(parameters, *, family="paired_bump", representation="hermite", boundary=None):
    """Decode one canonical candidate, reconstructing physical endpoint states.

    Single-bump is degree six, paired-bump degree seven.  A zero e in the
    paired family is the same geometric deformation as single-bump, although
    its stored polynomial degree remains seven.  The representation argument
    changes the construction, not the canonical domain or physical meaning.
    """
    _family(family)
    _representation(representation)
    parameters = _physical(parameters, family)
    if boundary is None:
        boundary = BoundaryStates()
    if not isinstance(boundary, BoundaryStates):
        raise ValueError("boundary must be a BoundaryStates instance")
    duration = float(parameters[-1])
    baseline = boundary.baseline(duration)
    d = parameters[:3]
    e = parameters[3:6] if family == "paired_bump" else np.zeros(3)
    if representation == "hermite":
        displaced = c2_bump(baseline, d)
        if family == "single_bump":
            return displaced
        coefficients = np.zeros((8, 3))
        coefficients[:len(displaced.coefficients)] = displaced.coefficients
        coefficients[3:8] += np.array((-32., 160., -288., 224., -64.))[:, None] * e
        return PolynomialSegment(coefficients, duration)
    if family == "single_bump":
        control = baseline.bezier_control_points(target_degree=6)
        control[3] += 3.2 * d
    else:
        control = baseline.bezier_control_points(target_degree=7)
        p3, p4 = middle_state_to_bezier_offsets(d, e)
        control[3] += p3
        control[4] += p4
    return bezier_segment(control, duration_s=duration)


def decode_normalized(candidate, *, family="paired_bump", representation="hermite", boundary=None):
    """Construct reference geometry from a common normalized search vector."""
    parameters = physical_from_normalized(candidate, family=family)
    return curve_from_parameters(parameters, family=family, representation=representation, boundary=boundary)


__all__ = ["FAMILIES", "REPRESENTATIONS", "BoundaryStates", "parameter_names",
           "parameter_bounds", "default_parameters", "physical_from_normalized",
           "normalized_from_physical", "middle_state_to_bezier_offsets",
           "bezier_offsets_to_middle_state", "curve_from_parameters", "decode_normalized"]
