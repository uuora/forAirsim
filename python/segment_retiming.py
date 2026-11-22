"""Phase-dependent timing of one continuous route, with no simulator calls."""
import numpy as np
from scipy.interpolate import BPoly, CubicSpline


class SegmentTimeMap:
    """Positive phase-rate squared; the blend is C2 across the original vias."""

    def __init__(self, solution, knots, fractions):
        self.knots = np.asarray(knots, dtype=float)
        self.fractions = np.asarray(fractions, dtype=float)
        anchors = np.asarray(solution["anchors"], dtype=float)
        if (self.knots.ndim != 1 or self.knots.size < 2 or self.fractions.shape != self.knots.shape
                or not np.isfinite(self.fractions).all() or not np.isfinite(self.knots).all()
                or np.any((self.fractions < 0) | (self.fractions > 1)) or np.any(np.diff(self.knots) <= 0)
                or self.knots[0] != anchors[0] or self.knots[-1] != anchors[-1]):
            raise ValueError("Invalid blend coefficients or phase knots")
        self.raw = CubicSpline(anchors, solution["h"], bc_type="natural", extrapolate=False)
        # With zero first and second derivatives at the knots the six Bernstein
        # ordinates on each leg are [left,left,left,right,right,right].
        self.blend = BPoly.from_derivatives(self.knots, [[f, 0., 0.] for f in self.fractions],
                                            extrapolate=False)

    def __call__(self, phase, derivative=0):
        delta = self.raw(phase) - 1.
        if derivative == 0:
            return 1. + self.blend(phase) * delta
        if derivative == 1:
            return self.blend(phase, 1) * delta + self.blend(phase) * self.raw(phase, 1)
        raise ValueError("Timing export uses only h and its first derivative")


def search_profiles(knot_count, uniform_seeds, evaluate):
    """Return the fastest *tested feasible* profile and its evaluation record.

    An incumbent changes only after full-route validation. A reverse sweep
    visits the braking end early on alternating mesh refinements.
    """
    tried = {}
    best = None
    origin = (0.,) * knot_count

    def consider(weights):
        nonlocal best
        if weights not in tried:
            row = evaluate(weights)
            tried[weights] = row
            if row["model_feasible"] and (best is None or row["duration_s"] < tried[best]["duration_s"]):
                best = weights

    for fraction in (*uniform_seeds, 0.):
        consider((float(fraction),) * knot_count)
    for level, step in enumerate((.5, .25, .125, .0625, .03125, .015625)):
        indices = range(knot_count) if level % 2 == 0 else reversed(range(knot_count))
        for index in indices:
            center = best if best is not None else origin
            for sign in (1, -1):
                neighbor = list(center)
                neighbor[index] = min(1., max(0., center[index] + sign * step))
                consider(tuple(neighbor))
    return best, list(tried.values())


def tracking_reserves(times, q, v, a, actual_v, spec, peak_error, error_margin):
    """Check the lag-model prediction against error and remaining braking work."""
    acceleration_limit = spec["command_limits"]["max_slew_mps2"]
    speed = np.linalg.norm(v, axis=1)
    tangent = np.divide(v, speed[:, None], out=np.zeros_like(v), where=speed[:, None] > 1.e-9)
    normal = a - (a * tangent).sum(axis=1)[:, None] * tangent
    tangential = np.sqrt(np.maximum(0., acceleration_limit**2 - (normal * normal).sum(axis=1)))
    # Integrate the available tangential deceleration backwards along the route.
    # The terminal measured-speed gate is 0.5 m/s. The model already includes
    # the command delay and axis-specific velocity lag in actual_v.
    distance = np.linalg.norm(np.diff(q, axis=0), axis=1)
    work = 2. * np.minimum(tangential[:-1], tangential[1:]) * distance
    braking_speed = np.sqrt(.5**2 + np.r_[np.cumsum(work[::-1])[::-1], 0.])
    reserve = braking_speed - np.linalg.norm(actual_v, axis=1)
    critical = int(np.argmin(reserve))
    error_limit = spec.get("peak_position_limit_m", .5) - error_margin
    return dict(predicted_peak_limit_m=error_limit, predicted_peak_error_m=peak_error,
                peak_margin_passed=bool(peak_error <= error_limit),
                min_braking_speed_margin_mps=float(reserve[critical]),
                braking_critical_time_s=float(times[critical]), braking_passed=bool(reserve[critical] >= 0.))


def sampled_motor_forces(times, acceleration, model):
    """Sample an inverse-dynamics lift at constant ENU yaw for the exported path.

    Estimates nominal individual thrust bounds, separately from the raw NLP
    motor record and from the yaw-following SimpleFlight execution.
    """
    t = np.linspace(0., times[-1], max(3, int(np.ceil(times[-1] / .02)) + 1))
    specific_force = np.column_stack([np.interp(t, times, acceleration[:, i]) for i in (1, 0, 2)])
    specific_force[:, 2] += model["nominal_gravity_mps2"]
    magnitude = np.linalg.norm(specific_force, axis=1)
    if np.min(magnitude) < 1.e-6:
        return dict(passed=False, reason="zero_specific_force")
    z = specific_force / magnitude[:, None]
    y = np.cross(z, [1., 0., 0.])
    norm_y = np.linalg.norm(y, axis=1)
    if np.min(norm_y) < 1.e-6:
        return dict(passed=False, reason="singular_fixed_yaw")
    y /= norm_y[:, None]
    rot = np.stack([np.cross(y, z), y, z], axis=2)
    body_rate = rot.transpose(0, 2, 1) @ np.gradient(rot, t, axis=0, edge_order=2)
    skew = (body_rate - body_rate.transpose(0, 2, 1)) / 2.
    omega = skew[:, [2, 0, 1], [1, 2, 0]]
    inertia = np.asarray(model["inertia_diag_kgm2"])
    torque = inertia * np.gradient(omega, t, axis=0, edge_order=2) + np.cross(omega, inertia * omega)
    wrench = np.column_stack([model["mass_kg"] * magnitude, torque])
    forces = np.linalg.solve(model["thrust_allocation_flu"], wrench.T)
    low, high = float(forces.min()), float(forces.max())
    return dict(passed=bool(low >= 0. and high <= model["max_thrust_per_rotor_n"]),
                min_n=low, max_n=high, samples=len(t), yaw_enu_rad=0.)
