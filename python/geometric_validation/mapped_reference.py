"""Stable path/time composition and continuous bounds for RotorPy experiments."""
from functools import lru_cache
from math import comb

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq, minimize

from .reference import FrozenReference


@lru_cache(None)
def choose(degree):
    return np.array([comb(degree, i) for i in range(degree + 1)], dtype=float)


def bernstein(power):
    degree = len(power) - 1
    return np.array([sum(power[j] * (comb(i, j) / comb(degree, j))
                         for j in range(i + 1)) for i in range(degree + 1)])


def derivative(controls, duration):
    return np.diff(controls, axis=0) * ((len(controls) - 1) / duration)


def basis(degree, u):
    u = np.asarray(u)[..., None]
    powers = np.arange(degree + 1)
    return choose(degree) * u**powers * (1 - u)**(degree - powers)


def evaluate(controls, u):
    return basis(len(controls) - 1, u) @ controls


def multiply(left, right):
    p, q = len(left) - 1, len(right) - 1
    return np.convolve(left * choose(p), right * choose(q)) / choose(p + q)


def compose(path, clock):
    # Polynomial-valued de Casteljau, with the path translated before differentiation.
    inverse = 1 - clock
    values = (path - path[0])[:, None, :]
    while len(values) > 1:
        values = np.array([[multiply(left[:, axis], inverse) + multiply(right[:, axis], clock)
                            for axis in range(3)] for left, right in zip(values, values[1:])]).transpose(0, 2, 1)
    return values[0]


def split(controls):
    left, right = [controls[0]], [controls[-1]]
    while len(controls) > 1:
        controls = (controls[1:] + controls[:-1]) * .5
        left.append(controls[0])
        right.append(controls[-1])
    return np.array(left), np.array(right[::-1])


def bound(controls, inside, depth=0):
    if not inside(controls[[0, -1]]):
        return 'violation'
    if inside(controls):
        return 'pass'
    if depth == 14:
        return 'unresolved'
    statuses = [bound(part, inside, depth + 1) for part in split(controls)]
    return 'violation' if 'violation' in statuses else 'unresolved' if 'unresolved' in statuses else 'pass'


def chain(path, u, rates):
    p, v, a, j, s = [evaluate(control, u) for control in path]
    r1, r2, r3, r4 = (rate[..., None] for rate in rates)
    return np.stack((p, v * r1, a * r1**2 + v * r2,
                     j * r1**3 + 3 * a * r1 * r2 + v * r3,
                     s * r1**4 + 6 * j * r1**2 * r2 + a * (3 * r2**2 + 4 * r1 * r3) + v * r4), axis=-2)


class MappedReference:
    def __init__(self, source, durations, clocks, geometry=None):
        base = FrozenReference(source) if geometry is None else geometry.base
        self.base, self.points, self.end = base, base.points, base.end
        self.segment_durations = np.asarray(durations, dtype=float)
        self.clocks = np.asarray(clocks, dtype=float)
        count = len(base.segment_durations)
        if (self.segment_durations.shape != (count,) or self.clocks.ndim != 2 or self.clocks.shape[0] != count
                or self.clocks.shape[1] < 5 or not np.isfinite(self.clocks).all()
                or not np.isfinite(self.segment_durations).all() or np.any(self.segment_durations <= 0)):
            raise ValueError('Invalid mapped reference dimensions or values')
        self.times = np.r_[0., np.cumsum(self.segment_durations)]
        self.duration = float(self.times[-1])
        if geometry is None:
            self.paths = []
            for powers in source['coefficients']:
                path = [bernstein(np.asarray(powers))]
                for _ in range(4):
                    path.append(derivative(path[-1], 1.))
                self.paths.append(path)
        else:
            self.paths = geometry.paths
        self.rates = []
        for clock, duration in zip(self.clocks, self.segment_durations):
            rates = [clock]
            for _ in range(4):
                rates.append(derivative(rates[-1], duration))
            self.rates.append(rates)

    def segment(self, i, time):
        u, *rates = [evaluate(control, time) for control in self.rates[i]]
        return chain(self.paths[i], u, rates)

    def sample_batch(self, times):
        samples = np.zeros((len(times), 5, 3))
        samples[times < 0, 0] = self.points[0]
        samples[times > self.duration, 0] = self.end
        for i, (start, end) in enumerate(zip(self.times, self.times[1:])):
            mask = (times >= start) & (times < end if i + 2 < len(self.times) else times <= end)
            samples[mask] = self.segment(i, (times[mask] - start) / self.segment_durations[i])
        return samples

    def certify(self, policy, limits):
        issues, checks = [], []
        caps = [policy[key] for key in ('max_speed_mps', 'max_acceleration_mps2', 'max_jerk_mps3')]
        endpoints = np.array([self.segment(i, np.array([0., 1.])) for i in range(len(self.paths))])
        jumps = np.max(np.abs(endpoints[1:, 0, :4] - endpoints[:-1, 1, :4]), axis=(0, 2), initial=0.)
        waypoint_speeds = np.linalg.norm(endpoints[1:, 0, 1], axis=1)
        if np.any(jumps > 1.e-7):
            issues.append('Frozen curve is not C3')
        if (np.max(np.abs(endpoints[:, 0, 0] - self.points[:-1])) > 1.e-7 or
                np.max(np.abs(endpoints[:, 1, 0] - self.points[1:])) > 1.e-7):
            issues.append('Frozen waypoint mismatch')
        if np.max(np.abs(endpoints[[0, -1], [0, 1], 1:4])) > 1.e-7:
            issues.append('Frozen endpoint derivative nonzero')
        if np.any(waypoint_speeds < policy['min_knot_speed_mps'] - 1.e-6):
            issues.append('Frozen waypoint speed below reference gate')
        low, high = np.asarray(limits['bounds_ned_min']), np.asarray(limits['bounds_ned_max'])
        for i, (path, clock, duration) in enumerate(zip(self.paths, self.clocks, self.segment_durations)):
            row = dict(monotonicity=bound(self.rates[i][1], lambda b: np.all(b > 1.e-10)))
            if abs(clock[0]) > 1.e-10 or abs(clock[-1] - 1.) > 1.e-10 or row['monotonicity'] != 'pass':
                issues.append('Time mapping endpoints or monotonicity failed')
            controls = compose(path[0], clock)
            row['safety_box'] = bound(controls + path[0][0], lambda b: np.all((b >= low) & (b <= high)))
            if row['safety_box'] != 'pass':
                issues.append('Frozen curve outside or unresolved safety box')
            for order, cap in enumerate(caps, 1):
                controls = derivative(controls, duration)
                status = bound(controls, lambda b: np.all(np.sum(b*b, axis=-1) <= cap**2 + 1.e-5))
                row[f'derivative_{order}'] = status
                if status != 'pass':
                    issues.append(f'Frozen derivative limit {status}: {order}')
            checks.append(row)
        return dict(issues=list(dict.fromkeys(issues)), max_abs_knot_jumps=jumps.tolist(),
                    waypoint_speeds_mps=waypoint_speeds.tolist(), segments=checks)


def original_mapping(source, power, duration):
    warp = Polynomial(power)
    original = np.asarray(source['segment_durations'])
    partitions = np.r_[0., np.cumsum(original)] / np.sum(original)
    knots = np.array([0., *(brentq(lambda t: warp(t) - s, 0., 1., xtol=1.e-14)
                             for s in partitions[1:-1]), 1.])
    clocks = []
    for index, gap in enumerate(np.diff(partitions)):
        local = (warp(Polynomial([knots[index], knots[index+1] - knots[index]])) - partitions[index]) / gap
        clock = bernstein(np.pad(local.coef, (0, 6 - len(local.coef))))
        clocks.append(clock)
    return MappedReference(source, np.diff(knots) * duration, clocks)


def optimize_mapping(source, old, policy, limits):
    geometry = old if old is not None else original_mapping(source, [0., 1.], sum(source['segment_durations']))
    original = geometry.base.segment_durations
    size = len(original)
    speed = np.array([np.linalg.norm(evaluate(path[1], 0.) / T)
                      for path, T in zip(geometry.paths[1:], original[1:])])
    lower = np.r_[.25, (policy['min_knot_speed_mps'] + 1.e-6) / speed, .25]
    caps = np.array([policy[key] for key in ('max_speed_mps', 'max_acceleration_mps2', 'max_jerk_mps3')])
    grid = np.linspace(0., 1., 97)
    bases = [basis(7 - order, grid) for order in range(4)]

    def clocks_for(x):
        fraction, rate = x[:size], x[size:]
        start, end = rate[:-1] * fraction / 7, rate[1:] * fraction / 7
        zeros, ones = np.zeros(size), np.ones(size)
        return np.stack((zeros, start, 2*start, 3*start, 1-3*end, 1-2*end, 1-end, ones), axis=1)

    def margins(x):
        durations = original * x[:size]
        clocks = clocks_for(x)
        values = [(clocks @ bases[0].T)]
        for order in range(1, 4):
            clocks = np.diff(clocks, axis=1) * ((8 - order) / durations[:, None])
            values.append(clocks @ bases[order].T)
        derivatives = []
        for index, path in enumerate(geometry.paths):
            rates = [value[index] for value in values[1:]]
            samples = chain(path, values[0][index], [*rates, np.zeros_like(rates[0])])
            derivatives.append(1 - np.sum(samples[:, 1:4]**2, axis=-1) / (caps * .996)**2)
        return np.r_[np.asarray(derivatives).ravel(), (values[1] - 1.e-4).ravel()]

    identity_params = np.r_[np.ones(size), np.ones(size + 1)]
    bounds = [(.55, 1.75)] * size + [(rate, max(2., rate + .1)) for rate in lower]
    old_rates = np.r_[[evaluate(rates[1], 0.) * T for rates, T in zip(geometry.rates, original)],
                      evaluate(geometry.rates[-1][1], 1.) * original[-1]]
    initial = np.clip(np.r_[geometry.segment_durations / original, old_rates], *np.asarray(bounds).T)
    objective = np.r_[original, np.zeros(size + 1)]
    result = minimize(lambda x: objective @ x, initial, jac=lambda x: objective,
                      bounds=bounds, constraints=[dict(type='ineq', fun=margins)], method='SLSQP',
                      options=dict(ftol=2.e-8, maxiter=45))
    stored = dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit),
                  minimum_grid_margin=float(np.min(margins(result.x))), knot_rates=result.x[size:].tolist())
    stored['rejected_candidates'] = []
    for fraction in (1., .99, .97, .93, .85, .65, .35):
        candidate = identity_params + fraction * (result.x - identity_params)
        optimized = MappedReference(source, original * candidate[:size], clocks_for(candidate), geometry=geometry)
        audit = optimized.certify(policy, limits)
        if not audit['issues'] and optimized.duration < float(original.sum()) - 1.e-7:
            stored.update(selected='optimized', optimizer_fraction=fraction, knot_rates=candidate[size:].tolist())
            return optimized, audit, stored
        stored['rejected_candidates'].append(dict(optimizer_fraction=fraction, duration_s=optimized.duration,
                                                   issues=audit['issues']))
    stored['selected'] = 'identity_fallback'
    identity = MappedReference(source, original, np.tile(np.linspace(0., 1., 8), (size, 1)), geometry=geometry)
    return identity, identity.certify(policy, limits), stored
