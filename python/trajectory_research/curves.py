"""Polynomial trajectory bases with explicit parameter and time derivatives.

Every curve uses ``u = (t - t0) / duration_s`` on the closed interval [0, 1].
Positions and control points have units of metres when the caller uses metres.
Hermite tangents named ``m0``/``m1`` are derivatives with respect to *u*;
state constructors instead accept physical velocity (m/s) and acceleration
(m/s**2).  A change of basis does not change a curve or improve its dynamics.

This module has no simulator I/O and supplies geometry, not a collision or
physical tracking guarantee.  Its control-point hull is an exact polynomial
convex-hull enclosure; feasibility still depends on the caller's constraints.
Stored and elevated degrees are limited to nine: high-degree conversions to
the power basis are numerically ill-conditioned and are outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MAX_DEGREE = 9


def _array(value, name: str, ndim: int | None = None) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must be a finite numeric array of dimension {ndim}")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _vectors(*values) -> tuple[np.ndarray, ...]:
    converted = tuple(_array(value, f"vector {i}", 1) for i, value in enumerate(values))
    if not converted or not converted[0].size or any(
            value.shape != converted[0].shape for value in converted):
        raise ValueError("vectors must be nonempty and have the same dimension")
    return converted


def _positive_duration(value) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) \
            or np.iscomplexobj(value) or not math.isfinite(float(value)) or value <= 0:
        raise ValueError("duration_s must be finite and positive")
    return float(value)


def _derivative(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or not 0 <= value <= 3:
        raise ValueError("derivative must be an integer in [0, 3]")
    return int(value)


def _parameter(value, upper: float = 1.0) -> np.ndarray:
    result = _array(value, "curve parameter")
    if result.ndim > 1 or np.any(result < 0) or np.any(result > upper):
        raise ValueError(f"curve parameter must be scalar or one-dimensional in [0, {upper}]")
    return result


@dataclass(frozen=True)
class PolynomialSegment:
    """A vector polynomial with ascending powers of normalized ``u``.

    ``coefficients[k]`` multiplies u**k and has shape (spatial_dimension,).
    ``evaluate`` defaults to derivatives in u; ``sample`` always uses physical
    seconds and time derivatives.  Both refuse extrapolation.  The coefficient
    array is defensively copied and made read-only on construction. Stored
    degrees must be between zero and MAX_DEGREE (nine), including zero padding.
    """

    coefficients: np.ndarray
    duration_s: float = 1.0

    def __post_init__(self) -> None:
        coefficients = _array(self.coefficients, "coefficients", 2)
        if not coefficients.shape[0] or not coefficients.shape[1]:
            raise ValueError("coefficients must have nonempty degree and spatial axes")
        if coefficients.shape[0] - 1 > MAX_DEGREE:
            raise ValueError(f"polynomial degree must be at most {MAX_DEGREE}")
        coefficients.setflags(write=False)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "duration_s", _positive_duration(self.duration_s))

    @property
    def degree(self) -> int:
        """Stored polynomial degree, including intentionally retained zeros."""
        return self.coefficients.shape[0] - 1

    def evaluate(self, u, derivative: int = 0, *, wrt: str = "u") -> np.ndarray:
        """Evaluate position or derivatives 1..3 at normalized ``u``.

        ``wrt='time'`` divides the u derivative by duration_s**derivative.
        A scalar input returns (dimension,); a vector returns (samples, dimension).
        """
        derivative = _derivative(derivative)
        if wrt not in ("u", "time"):
            raise ValueError("wrt must be 'u' or 'time'")
        parameter = _parameter(u)
        coefficients = self.coefficients.copy()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for _ in range(derivative):
                    if coefficients.shape[0] == 1:
                        coefficients = np.zeros_like(coefficients)
                    else:
                        coefficients = coefficients[1:] * np.arange(1, coefficients.shape[0])[:, None]
                    if wrt == "time":
                        coefficients = coefficients / self.duration_s
                result = np.zeros(parameter.shape + (self.coefficients.shape[1],))
                for coefficient in coefficients[::-1]:
                    result = result * parameter[..., None] + coefficient
        except FloatingPointError as error:
            raise ValueError("curve evaluation exceeded finite floating-point range") from error
        return result

    def sample(self, time_s, derivative: int = 0) -> np.ndarray:
        """Evaluate at elapsed seconds; derivatives are physical time derivatives."""
        time_s = _parameter(time_s, self.duration_s)
        return self.evaluate(time_s / self.duration_s, derivative, wrt="time")

    def bezier_control_points(self, target_degree: int | None = None) -> np.ndarray:
        """Return an equivalent Bernstein representation, elevated at most to degree nine."""
        degree = self.degree
        result = np.zeros_like(self.coefficients)
        for k in range(degree + 1):
            for j in range(k + 1):
                result[k] += self.coefficients[j] * (math.comb(k, j) / math.comb(degree, j))
        if not np.isfinite(result).all():
            raise ValueError("Bézier conversion exceeded finite floating-point range")
        return result if target_degree is None else elevate_bezier(result, target_degree)

    def control_hull_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned enclosure derived from the curve's full control-point hull."""
        control = self.bezier_control_points()
        return control.min(axis=0), control.max(axis=0)


def bezier_segment(control_points, *, duration_s: float = 1.0) -> PolynomialSegment:
    """Convert a finite Bézier polygon of degree zero to nine to the power basis."""
    control = _array(control_points, "control_points", 2)
    if not control.shape[0] or not control.shape[1]:
        raise ValueError("control_points must have nonempty degree and spatial axes")
    degree = control.shape[0] - 1
    if degree > MAX_DEGREE:
        raise ValueError(f"Bézier degree must be at most {MAX_DEGREE}")
    coefficients = np.zeros_like(control)
    for k in range(degree + 1):
        for i in range(k + 1):
            coefficients[k] += (math.comb(degree, i) * math.comb(degree - i, k - i)
                                * (-1) ** (k - i)) * control[i]
    return PolynomialSegment(coefficients, duration_s)


def cubic_hermite_to_bezier(p0, p1, m0, m1) -> np.ndarray:
    """Map endpoint positions and *u* tangents to four cubic Bézier points.

    B0=p0, B1=p0+m0/3, B2=p1-m1/3, B3=p1.  For physical velocities,
    first set m0=duration_s*v0 and m1=duration_s*v1.
    """
    p0, p1, m0, m1 = _vectors(p0, p1, m0, m1)
    return _array(np.array((p0, p0 + m0 / 3.0, p1 - m1 / 3.0, p1)), "cubic control_points", 2)


def cubic_bezier_to_hermite(control_points) -> tuple[np.ndarray, ...]:
    """Return (p0, p1, m0, m1), where m0/m1 are normalized-parameter tangents."""
    control = _array(control_points, "cubic control_points", 2)
    if control.shape[0] != 4 or not control.shape[1]:
        raise ValueError("a cubic Bézier segment requires exactly four nonempty control points")
    return _vectors(control[0], control[3],
                    3.0 * (control[1] - control[0]), 3.0 * (control[3] - control[2]))


def cubic_hermite(p0, p1, m0, m1, *, duration_s: float = 1.0) -> PolynomialSegment:
    """Construct a cubic from endpoint positions and derivatives with respect to u."""
    p0, p1, m0, m1 = _vectors(p0, p1, m0, m1)
    return PolynomialSegment(np.array((p0, m0,
        -3.0 * p0 + 3.0 * p1 - 2.0 * m0 - m1,
        2.0 * p0 - 2.0 * p1 + m0 + m1)), duration_s)


def cubic_hermite_from_state(p0, v0, p1, v1, duration_s: float) -> PolynomialSegment:
    """Construct a cubic from positions, physical velocities, and duration in seconds."""
    p0, v0, p1, v1 = _vectors(p0, v0, p1, v1)
    duration_s = _positive_duration(duration_s)
    return cubic_hermite(p0, p1, duration_s * v0, duration_s * v1, duration_s=duration_s)


def quintic_hermite(p0, v0, a0, p1, v1, a1, duration_s: float) -> PolynomialSegment:
    """Construct a quintic with specified endpoint physical p/v/a and duration.

    This enforces six endpoint conditions.  It does not claim to solve a joint
    path/time minimum-jerk optimization or ensure C3 continuity at a junction.
    """
    p0, v0, a0, p1, v1, a1 = _vectors(p0, v0, a0, p1, v1, a1)
    duration_s = _positive_duration(duration_s)
    c0, c1, c2 = p0, duration_s * v0, (0.5 * duration_s * a0) * duration_s
    delta_p = p1 - c0 - c1 - c2
    delta_v = duration_s * v1 - c1 - 2.0 * c2
    delta_a = (duration_s * a1) * duration_s - 2.0 * c2
    return PolynomialSegment(np.array((c0, c1, c2,
        10.0 * delta_p - 4.0 * delta_v + 0.5 * delta_a,
        -15.0 * delta_p + 7.0 * delta_v - delta_a,
        6.0 * delta_p - 3.0 * delta_v + 0.5 * delta_a)), duration_s)


def elevate_bezier(control_points, target_degree: int) -> np.ndarray:
    """Raise a Bézier degree, at most to nine, without changing its curve.

    Both the source and target degree are bounded by MAX_DEGREE so later
    conversion to the module's normalized power representation stays in scope.
    """
    control = _array(control_points, "control_points", 2)
    degree = control.shape[0] - 1
    if degree < 0 or degree > MAX_DEGREE or not control.shape[1] \
            or isinstance(target_degree, (bool, np.bool_)) \
            or not isinstance(target_degree, (int, np.integer)) \
            or not degree <= target_degree <= MAX_DEGREE:
        raise ValueError(f"source and target degrees must satisfy 0 <= source <= target <= {MAX_DEGREE}")
    for n in range(degree, int(target_degree)):
        elevated = np.empty((n + 2, control.shape[1]))
        elevated[0], elevated[-1] = control[0], control[-1]
        for k in range(1, n + 1):
            alpha = k / (n + 1.0)
            elevated[k] = alpha * control[k - 1] + (1.0 - alpha) * control[k]
        control = elevated
    return control


def bump_weight(u, derivative: int = 0) -> np.ndarray:
    """Return 64*u**3*(1-u)**3 or its first three u derivatives.

    The nonnegative weight peaks at one at u=.5; value, first and second
    derivatives vanish at both endpoints.  Third derivatives do not vanish.
    """
    segment = PolynomialSegment(np.array((0., 0., 0., 64., -192., 192., -64.))[:, None])
    return segment.evaluate(u, derivative)[..., 0]


def c2_bump(segment: PolynomialSegment, displacement) -> PolynomialSegment:
    """Add a boundary-preserving degree-six deformation in metres.

    q(u)=p(u)+64*u**3*(1-u)**3*displacement preserves endpoint physical p/v/a
    for the same duration.  Endpoint jerk changes by +/-384*displacement/T**3.
    The maximum displacement norm is exactly norm(displacement), attained at
    u=.5.  A quintic input becomes a degree-six curve, with its own hull bounds.
    """
    if not isinstance(segment, PolynomialSegment):
        raise ValueError("segment must be a PolynomialSegment")
    displacement, = _vectors(displacement)
    if displacement.shape[0] != segment.coefficients.shape[1]:
        raise ValueError("displacement dimension must match the segment")
    coefficients = np.zeros((max(7, segment.degree + 1), displacement.size))
    coefficients[:segment.degree + 1] = segment.coefficients
    coefficients[3:7] += np.array((64., -192., 192., -64.))[:, None] * displacement
    return PolynomialSegment(coefficients, segment.duration_s)


def junction_residuals(left: PolynomialSegment, right: PolynomialSegment, *, atol: float = 1e-9) -> dict:
    """Check a shared node using endpoint physical p/v/a/jerk, including unequal T.

    Ck is cumulative: C2 requires agreement in position, velocity and acceleration.
    Residuals use Euclidean norms with units m, m/s, m/s**2, m/s**3 respectively.
    A true result is a numeric endpoint check, not a physical tracking guarantee.
    """
    if not isinstance(left, PolynomialSegment) or not isinstance(right, PolynomialSegment) \
            or left.coefficients.shape[1] != right.coefficients.shape[1]:
        raise ValueError("segments must be PolynomialSegment objects with equal spatial dimensions")
    if isinstance(atol, (bool, np.bool_)) or not isinstance(atol, (int, float, np.number)) \
            or np.iscomplexobj(atol) or not math.isfinite(float(atol)) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    names = ("position_m", "velocity_m_s", "acceleration_m_s2", "jerk_m_s3")
    residuals = [float(np.linalg.norm(left.evaluate(1.0, order, wrt="time")
                                    - right.evaluate(0.0, order, wrt="time")))
                 for order in range(4)]
    result = dict(zip(names, residuals))
    result.update({f"c{order}": all(value <= atol for value in residuals[:order + 1])
                   for order in range(4)})
    return result
