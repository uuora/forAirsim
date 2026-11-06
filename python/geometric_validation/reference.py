"""Explicit timed C3 septic-Hermite reference for initial controller validation.

This is a small test generator, not a comparison of Bezier/Hermite/B-spline
representations or an optimizer. Position and its first four derivatives are
analytic polynomial evaluations in SI/NED. Interior derivatives can be supplied
to create curved passes; otherwise each knot is a full stop. Snap is piecewise
defined and need not be continuous at interior knots.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial import Polynomial


def _vectors(value, n, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (n, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be {n} finite 3-vectors")
    return array


def _coefficient_matrix():
    # Ascending polynomial powers; endpoint derivatives with respect to u.
    rows = []
    for u in (0., 1.):
        for derivative in range(4):
            rows.append([0. if power < derivative else
                         math.factorial(power) / math.factorial(power - derivative)
                         * u ** (power - derivative) for power in range(8)])
    return np.asarray(rows)


_ENDPOINT_MATRIX = _coefficient_matrix()


class Reference:
    def __init__(self, case_cfg: dict):
        self.start = np.asarray(case_cfg.get("start", [0, 0, -5]), dtype=float)
        if self.start.shape != (3,) or not np.all(np.isfinite(self.start)):
            raise ValueError("start must be a finite NED vector")
        if "points" in case_cfg:
            points = list(case_cfg["points"])
        else:
            points = [(self.start + np.asarray(offset, dtype=float)).tolist()
                      for offset in case_cfg.get("offsets", [[0, 0, 0]])]
        self.points = _vectors([self.start.tolist()] + points, len(points) + 1, "knots")
        n = len(self.points)
        if n < 2:
            raise ValueError("reference needs at least one segment")
        self.segment_durations = np.asarray(case_cfg.get("segment_durations", [5.] * (n-1)), dtype=float)
        if self.segment_durations.shape != (n-1,) or not np.all(np.isfinite(self.segment_durations)) or np.any(self.segment_durations <= 0):
            raise ValueError("one positive finite duration is required per segment")
        self.times = np.concatenate(([0.], np.cumsum(self.segment_durations)))
        self.duration = float(self.times[-1])
        self.end = self.points[-1].copy()
        self.yaw = float(case_cfg.get("yaw", 0.))
        if not math.isfinite(self.yaw):
            raise ValueError("yaw must be finite")
        derivatives = [self.points]
        for key in ("velocities", "accelerations", "jerks"):
            derivative = _vectors(case_cfg.get(key, np.zeros((n, 3))), n, key)
            if np.any(np.abs(derivative[[0, -1]]) > 1e-12):
                raise ValueError("validation reference requires rest through jerk at start/end")
            derivatives.append(derivative)
        self.polynomials = []
        for segment, duration in enumerate(self.segment_durations):
            values = np.array([derivatives[d][k] * duration**d
                               for k in (segment, segment+1) for d in range(4)])
            coefficients = np.linalg.solve(_ENDPOINT_MATRIX, values)
            xyz = [Polynomial(coefficients[:, axis]) for axis in range(3)]
            self.polynomials.append([[poly.deriv(d) / duration**d for poly in xyz]
                                     for d in range(5)])

    def sample(self, t: float) -> dict:
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("reference time must be finite")
        if t < 0 or t > self.duration:
            p = self.start if t < 0 else self.end
            return {"p": p.tolist(), "v": [0., 0., 0.], "a": [0., 0., 0.],
                    "jerk": [0., 0., 0.], "snap": [0., 0., 0.],
                    "yaw": self.yaw, "yaw_rate": 0., "segment": 0 if t < 0 else len(self.segment_durations)-1}
        segment = min(int(np.searchsorted(self.times, t, side="right")) - 1,
                      len(self.segment_durations)-1)
        u = float((t - self.times[segment]) / self.segment_durations[segment])
        result = {key: [float(poly(u)) for poly in polys]
                  for key, polys in zip(("p", "v", "a", "jerk", "snap"), self.polynomials[segment])}
        result.update(yaw=self.yaw, yaw_rate=0., segment=segment)
        return result

    def sample_path(self, count: int = 300) -> list[list[float]]:
        return [self.sample(t)["p"] for t in np.linspace(0., self.duration, max(2, int(count)))]


def build_reference(config: dict, case: str) -> Reference:
    case_cfg = {"start": config.get("start", [0., 0., -5.]),
                "yaw": config.get("yaw", 0.), **config["cases"][case]}
    return Reference(case_cfg)
