"""Adapt an existing degree <=7 reference to endpoint-preserving perturbations.

This interface accepts actual initial Bézier/Hermite geometry rather than a
fixed research anchor. The caller owns perturbation bounds and feasibility
checks. No controller, simulator, tracking or automatic safety check is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .curves import PolynomialSegment, bezier_segment
from .parameterizations import BoundaryStates


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


def _duration(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
            or np.iscomplexobj(value) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("duration must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class InitialCurveAdapter:
    """Physical endpoint states plus an initial midpoint displacement/u tangent.

    d0 and e0 have units of metres; the midpoint physical velocity contribution
    of e0 is e0/T0. Values are copied and unconstrained by the research boxes.
    Endpoint jerk is generally changed by a nonzero shape perturbation.
    """

    boundary: BoundaryStates
    d0: tuple[float, float, float]
    e0: tuple[float, float, float]
    T0: float

    def __post_init__(self):
        if not isinstance(self.boundary, BoundaryStates):
            raise ValueError("boundary must be a BoundaryStates instance")
        for name in ("d0", "e0"):
            object.__setattr__(self, name, tuple(float(x) for x in _vector(getattr(self, name), 3, name)))
        object.__setattr__(self, "T0", _duration(self.T0))

    @classmethod
    def from_segment(cls, segment):
        """Extract a 3-D polynomial of stored degree zero through seven.

        In exact arithmetic, zero delta reconstructs the input because the
        endpoint-C2 nullspace at degree seven is spanned by b and (u-.5)*b.
        Floating-point reconstruction has ordinary basis-conversion error.
        """
        if not isinstance(segment, PolynomialSegment) or segment.degree > 7 \
                or segment.coefficients.shape[1] != 3:
            raise ValueError("initial reference must be a 3D PolynomialSegment of degree <=7")
        boundary = BoundaryStates(
            p0=segment.evaluate(0.), v0=segment.evaluate(0., 1, wrt="time"),
            a0=segment.evaluate(0., 2, wrt="time"), p1=segment.evaluate(1.),
            v1=segment.evaluate(1., 1, wrt="time"), a1=segment.evaluate(1., 2, wrt="time"))
        baseline = boundary.baseline(segment.duration_s)
        return cls(boundary, segment.evaluate(.5)-baseline.evaluate(.5),
                   segment.evaluate(.5, 1)-baseline.evaluate(.5, 1), segment.duration_s)

    @classmethod
    def from_bezier(cls, control_points, duration_s):
        """Copy a degree <=7 Bézier control polygon with a physical duration."""
        try:
            control = np.asarray(control_points)
        except (TypeError, ValueError) as error:
            raise ValueError("control points must be a finite (1..8,3) matrix") from error
        if control.dtype.kind not in "iuf" or control.ndim != 2 \
                or control.shape[1:] != (3,) or not 1 <= len(control) <= 8 \
                or not np.isfinite(control).all():
            raise ValueError("control points must be a finite (1..8,3) matrix")
        return cls.from_segment(bezier_segment(control.copy(), duration_s=_duration(duration_s)))

    def decode_delta(self, delta):
        """Decode [delta_d(3),delta_e(3),delta_T] without fixed search bounds.

        Rebuild the quintic at T0+delta_T to retain the input's *physical*
        endpoint p/v/a. Zero shape deltas with a different duration do not
        preserve every interior point of the original curve. A caller must
        validate geometry, derivatives and any vehicle/tracking constraints.
        """
        delta = _vector(delta, 7, "delta")
        try:
            with np.errstate(over="raise", invalid="raise"):
                duration = _duration(self.T0+delta[-1])
                d, e = np.asarray(self.d0)+delta[:3], np.asarray(self.e0)+delta[3:6]
                base = self.boundary.baseline(duration)
                coefficients = np.zeros((8, 3))
                coefficients[:len(base.coefficients)] = base.coefficients
                coefficients[3:7] += np.array((64., -192., 192., -64.))[:, None]*d
                coefficients[3:8] += np.array((-32., 160., -288., 224., -64.))[:, None]*e
        except FloatingPointError as error:
            raise ValueError("perturbation exceeded finite floating-point range") from error
        return PolynomialSegment(coefficients, duration)
