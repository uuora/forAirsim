"""Conservative continuous-parameter bounds for polynomial reference curves.

These are finite-precision, sufficient geometry and reference-derivative checks.
They do not model a vehicle, compensate wind, or bound tracking error. A failed
hull check means UNRESOLVED unless an actual curve point proves a violation.
"""
from __future__ import annotations

import math

import numpy as np

from .curves import PolynomialSegment


BUMP_DERIVATIVE_MAXIMA = (1.0, 192.0 / (25.0 * math.sqrt(5.0)), 24.0, 384.0)
BUMP_JERK_ENERGY_COEFFICIENT = 147456.0 / 7.0


def _numeric_array(value, name, ndim):
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
        condition = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {condition}")
    return float(value)


def _segment(segment):
    if not isinstance(segment, PolynomialSegment):
        raise ValueError("segment must be a PolynomialSegment")
    return segment


def _order(order):
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) \
            or not 0 <= order <= 3:
        raise ValueError("order must be an integer in [0,3]")
    return int(order)


def _depth(max_depth):
    if isinstance(max_depth, (bool, np.bool_)) or not isinstance(max_depth, (int, np.integer)) \
            or not 0 <= max_depth <= 20:
        raise ValueError("max_depth must be an integer in [0,20]")
    return int(max_depth)


def derivative_control_points(segment, order=1):
    """Bézier controls of a physical time derivative, with order 0 meaning p.

    Repeated differentiation uses n*(B[i+1]-B[i])/duration_s. For a derivative
    above the stored degree, a single zero control point represents zero.
    """
    segment = _segment(segment)
    order = _order(order)
    control = segment.bezier_control_points()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for _ in range(order):
                n = len(control) - 1
                control = (n * np.diff(control, axis=0) / segment.duration_s
                           if n else np.zeros((1, control.shape[1])))
    except FloatingPointError as error:
        raise ValueError("derivative controls exceeded finite floating-point range") from error
    if not np.isfinite(control).all():
        raise ValueError("derivative controls exceeded finite floating-point range")
    return control


def derivative_norm_bound(segment, order=1):
    """Return max control norm, a conservative bound on sup ||d^order p/dt^order||.

    This convex-hull sufficient bound may overestimate the true peak; exceeding
    a limit here does not prove that the curve exceeds that limit.
    """
    control = derivative_control_points(segment, order)
    result = max(math.hypot(*row) for row in control)
    if not math.isfinite(result):
        raise ValueError("derivative norm bound exceeded finite floating-point range")
    return float(result)


def _split_half(control):
    """de Casteljau subdivision at u=.5; controls keep their original units."""
    level = control.copy()
    left, right = [level[0].copy()], [level[-1].copy()]
    while len(level) > 1:
        # Half before addition avoids overflow for same-sign finite controls.
        level = .5 * level[:-1] + .5 * level[1:]
        left.append(level[0].copy())
        right.append(level[-1].copy())
    return np.array(left), np.array(right[::-1])


def convex_corridor_certificate(segment, normals, offsets, *, margin=0.0,
                                max_depth=0, numerical_margin=1e-10):
    """Certify p(u) in {x: A*x <= b-margin} for the whole u interval.

    ``margin`` and ``numerical_margin`` use units of each row's A*x; normalize
    normals if a metric clearance is intended. Subdivision can reduce hull
    conservatism. CERTIFIED requires every leaf's controls to satisfy every
    halfspace with the numerical margin. VIOLATED requires an endpoint of a
    subinterval outside a halfspace. Depth exhaustion returns UNRESOLVED.
    """
    segment = _segment(segment)
    normals = _numeric_array(normals, "normals", 2)
    offsets = _numeric_array(offsets, "offsets", 1)
    if normals.shape != (len(offsets), segment.coefficients.shape[1]):
        raise ValueError("corridor dimensions must match the segment")
    if any(math.hypot(*row) == 0 for row in normals):
        raise ValueError("corridor normals must be nonzero")
    margin = _scalar(margin, "margin")
    tolerance = _scalar(numerical_margin, "numerical_margin")
    depth = _depth(max_depth)
    threshold = offsets - margin
    pending = [(segment.bezier_control_points(), 0.0, 1.0, 0)]
    unresolved, certified, witness = [], 0, None
    while pending:
        control, start, end, level = pending.pop()
        with np.errstate(over="raise", invalid="raise"):
            try:
                excess = control @ normals.T - threshold
            except FloatingPointError as error:
                raise ValueError("corridor computation exceeded finite floating-point range") from error
        if np.max(excess) <= -tolerance:
            certified += 1
            continue
        for index, parameter in ((0, start), (-1, end)):
            if np.max(excess[index]) > tolerance:
                witness = dict(u=parameter, position=control[index].tolist(),
                               maximum_halfspace_excess=float(np.max(excess[index])))
                break
        if witness is not None:
            break
        if level == depth:
            unresolved.append([start, end])
        else:
            left, right = _split_half(control)
            middle = .5 * (start + end)
            pending.extend(((right, middle, end, level + 1), (left, start, middle, level + 1)))
    status = "VIOLATED" if witness is not None else ("UNRESOLVED" if unresolved else "CERTIFIED")
    return dict(status=status, certified_leaves=certified, unresolved_intervals=unresolved,
                violation_witness=witness, max_depth=depth, numerical_margin=tolerance,
                scope="continuous polynomial reference; finite-precision hull sufficient condition")


def sphere_clearance_certificate(segment, centre, radius, *, clearance=0.0,
                                 max_depth=12, numerical_margin=1e-10):
    """Conservatively certify strict clearance from one closed sphere.

    Each de Casteljau leaf is enclosed in its control-point AABB. A leaf is
    certified when its AABB is farther from the centre than radius+clearance
    plus numerical_margin. Actual subinterval endpoints can prove VIOLATED.
    AABB overlap alone never proves collision; depth exhaustion is UNRESOLVED.
    The test uses no discrete sample grid as a safety argument.
    """
    segment = _segment(segment)
    centre = _numeric_array(centre, "centre", 1)
    if centre.shape != (segment.coefficients.shape[1],):
        raise ValueError("sphere centre dimension must match the segment")
    radius = _scalar(radius, "radius", positive=True)
    clearance = _scalar(clearance, "clearance")
    tolerance = _scalar(numerical_margin, "numerical_margin")
    depth = _depth(max_depth)
    inflated = radius + clearance
    if not math.isfinite(inflated + tolerance):
        raise ValueError("sphere bound exceeded finite floating-point range")
    pending = [(segment.bezier_control_points(), 0.0, 1.0, 0)]
    unresolved, certified, witness = [], 0, None
    minimum_certified_box_distance = None
    while pending:
        control, start, end, level = pending.pop()
        lower, upper = control.min(axis=0), control.max(axis=0)
        with np.errstate(over="raise", invalid="raise"):
            try:
                separation = np.maximum(np.maximum(lower - centre, centre - upper), 0.0)
            except FloatingPointError as error:
                raise ValueError("sphere computation exceeded finite floating-point range") from error
        distance = math.hypot(*separation)
        if not math.isfinite(distance):
            raise ValueError("sphere computation exceeded finite floating-point range")
        if distance > inflated + tolerance:
            certified += 1
            minimum_certified_box_distance = (distance if minimum_certified_box_distance is None
                                               else min(minimum_certified_box_distance, distance))
            continue
        for index, parameter in ((0, start), (-1, end)):
            point_distance = math.hypot(*(control[index] - centre))
            if point_distance < inflated - tolerance:
                witness = dict(u=parameter, position=control[index].tolist(),
                               distance_to_centre=point_distance)
                break
        if witness is not None:
            break
        if level == depth:
            unresolved.append([start, end])
        else:
            left, right = _split_half(control)
            middle = .5 * (start + end)
            pending.extend(((right, middle, end, level + 1), (left, start, middle, level + 1)))
    status = "VIOLATED" if witness is not None else ("UNRESOLVED" if unresolved else "CERTIFIED")
    return dict(status=status, certified_leaves=certified, unresolved_intervals=unresolved,
                violation_witness=witness, max_depth=depth, numerical_margin=tolerance,
                minimum_certified_box_distance=minimum_certified_box_distance,
                inflated_radius=inflated,
                scope="continuous polynomial reference versus one sphere; finite-precision AABB sufficient condition")


def c2_bump_amplitude_budget(duration_s, *, speed_margin, acceleration_margin,
                             jerk_margin, clearance):
    """Bound ||d|| for an additive b(u)*d with b=64*u^3*(1-u)^3.

    Remaining speed, acceleration and jerk budgets have units m/s, m/s^2 and
    m/s^3. Clearance is a conservative isotropic displacement allowance in m.
    The sufficient triangle-inequality radius is
      min(clearance, speed_margin*T/c1, acceleration_margin*T^2/24,
          jerk_margin*T^3/384).
    This bound assumes the supplied margins are valid for the unperturbed
    reference over the complete segment. It does not create those margins.
    Component caps are truncated at clearance to keep outputs finite.
    """
    duration = _scalar(duration_s, "duration_s", positive=True)
    remaining = [_scalar(speed_margin, "speed_margin"),
                 _scalar(acceleration_margin, "acceleration_margin"),
                 _scalar(jerk_margin, "jerk_margin")]
    clearance = _scalar(clearance, "clearance")
    caps = []
    for power, (budget, constant) in enumerate(zip(remaining, BUMP_DERIVATIVE_MAXIMA[1:]), 1):
        if budget == 0 or clearance == 0:
            cap = 0.0
        else:
            log_cap = math.log(budget) + power * math.log(duration) - math.log(constant)
            cap = clearance if log_cap >= math.log(clearance) else math.exp(log_cap)
        caps.append(float(cap))
    return dict(maximum_displacement_norm_m=float(min(clearance, *caps)),
                clearance_cap_m=clearance, speed_cap_m=caps[0],
                acceleration_cap_m=caps[1], jerk_cap_m=caps[2], duration_s=duration,
                scope="sufficient additive-reference budget; no tracking or external-force guarantee")


def quintic_bump_jerk_energy_increment(segment, displacement):
    """Exact analytic increase in integral ||p'''(t)||^2 dt for degree <=5.

    For a quintic (or lower-degree) baseline, adding b(u)*d preserves endpoint
    p/v/a. Three integrations by parts give integral p'''*delta_p'''=0,
    because the baseline's sixth derivative vanishes and all boundary terms
    vanish. Thus the increase is (147456/7)*||d||^2/T^5. This orthogonality
    does not hold for an arbitrary baseline of degree six or above.
    """
    segment = _segment(segment)
    if segment.degree > 5:
        raise ValueError("the jerk-energy identity requires a baseline degree at most five")
    displacement = _numeric_array(displacement, "displacement", 1)
    if displacement.shape != (segment.coefficients.shape[1],):
        raise ValueError("displacement dimension must match the segment")
    norm = math.hypot(*displacement)
    if norm == 0:
        return 0.0
    if not math.isfinite(norm):
        raise ValueError("displacement norm exceeded finite floating-point range")
    exponent = math.log(BUMP_JERK_ENERGY_COEFFICIENT) + 2 * math.log(norm) - 5 * math.log(segment.duration_s)
    try:
        result = math.exp(exponent)
    except OverflowError as error:
        raise ValueError("jerk-energy increment exceeded finite floating-point range") from error
    if not math.isfinite(result):
        raise ValueError("jerk-energy increment exceeded finite floating-point range")
    return float(result)
