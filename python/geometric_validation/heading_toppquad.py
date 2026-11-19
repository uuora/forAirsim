"""Continuous TOPPQuad refinement on a fixed path with a flown heading."""
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from .mapped_reference import MappedReference, basis, chain, derivative
from .paper_topp import flat_thrust


class RouteHeading:
    """Look ahead along the geometric path and smooth the resulting heading."""
    def __init__(self, geometry):
        from .paper_topp import path_values

        progress = np.linspace(0.,geometry.base.duration,601)
        # The look-ahead is defined in route progress rather than measured velocity.
        span = .27
        look = enu(path_values(geometry.base,np.clip(progress + span,0.,geometry.base.duration)))[:,0]
        before = enu(path_values(geometry.base,np.clip(progress - .5*span,0.,geometry.base.duration)))[:,0]
        horizontal = (look-before)[:,:2]
        good = np.linalg.norm(horizontal,axis=1) >= .12
        if np.count_nonzero(good) < 2:
            raise ValueError('Path lacks a horizontal direction')
        observed = np.unwrap(np.arctan2(horizontal[good,1],horizontal[good,0]))
        bearings = np.interp(progress,progress[good],observed)
        smooth = gaussian_filter1d(bearings,.30/(progress[1]-progress[0]),mode='nearest')
        self.source_times = np.linspace(0.,geometry.base.duration,61)
        self.angles = np.interp(self.source_times,progress,smooth)
        self.spline = CubicSpline(self.source_times,self.angles,bc_type='clamped')

    def sample(self, progress, progress_rate, progress_acceleration):
        progress = np.clip(progress, self.source_times[0], self.source_times[-1])
        first, second = self.spline(progress, 1), self.spline(progress, 2)
        return (self.spline(progress), first*progress_rate,
                second*progress_rate**2 + first*progress_acceleration)


def identity_reference(source, durations):
    return MappedReference(source, durations, np.tile(np.linspace(0., 1., 8), (len(durations), 1)))


def enu(flat):
    return flat[..., [1, 0, 2]] * [1., 1., -1.]


def segment_values(geometry, durations, clocks, heading, mesh):
    """Evaluate flat outputs and heading derivatives on each clock segment."""
    degree = clocks.shape[1] - 1
    mapped = clocks
    values = [mapped @ mesh[0].T]
    for index in range(1, 5):
        mapped = np.diff(mapped, axis=1) * ((degree+1-index)/durations[:, None])
        values.append(mapped @ mesh[index].T)
    flats, attitudes = [], []
    for index, path in enumerate(geometry.paths):
        u, d1, d2, d3, d4 = [value[index] for value in values]
        flats.append(chain(path, u, (d1, d2, d3, d4)))
        source_d = geometry.base.segment_durations[index]
        attitudes.append(heading.sample(geometry.base.times[index] + source_d*u, source_d*d1, source_d*d2))
    return enu(np.concatenate(flats)), np.concatenate(attitudes, axis=1)


def constraint_values(flats, angles, controller, policy, rotor_limit):
    force = flat_thrust(flats, controller, *angles)
    speed = np.linalg.norm(flats[:, 1], axis=1)
    specific = flats[:, 2] + [0., 0., controller.g]
    horizontal = np.linalg.norm(specific[:, :2], axis=1)
    tilt = np.degrees(np.arctan2(horizontal, specific[:, 2]))
    margins = np.concatenate((
        1. - speed / policy['speed_cap_mps'],
        1. - np.linalg.norm(flats[:, 2], axis=1) / policy['acceleration_cap_mps2'],
        (specific[:, 2]*np.tan(np.radians(policy['reference_tilt_deg']))-horizontal) / controller.g,
        1. - np.abs(angles[1]) / policy['heading_rate_radps'],
        (force-policy['motor_margin_n']).ravel() / rotor_limit,
        (rotor_limit-policy['motor_margin_n']-force).ravel() / rotor_limit,
    ))
    return speed, tilt, force, margins


def clocks_for(durations, rates, original, amplitudes):
    start, end = (rates[:-1]*durations/original/7, rates[1:]*durations/original/7)
    septic = np.column_stack((np.zeros(len(durations)), start, 2*start, 3*start,
                              1-3*end, 1-2*end, 1-end, np.ones(len(durations))))
    weight = np.arange(1,8)/8
    elevated = np.column_stack((septic[:,0], septic[:,:-1]*weight + septic[:,1:]*(1-weight), septic[:,-1]))
    # A middle Bernstein control changes within-segment timing while leaving
    # the four boundary conditions on each end intact.
    elevated[:,4] += amplitudes
    return elevated


@lru_cache(maxsize=6)
def mesh_for(count, degree):
    grid = np.linspace(0.,1.,count)
    return [basis(degree-order, grid) for order in range(5)]


def validate_reference(reference, heading, controller, policy, rotor_limit):
    samples_per_segment = 385
    flats, angles = segment_values(reference, reference.segment_durations, reference.clocks, heading,
                                   mesh_for(samples_per_segment,reference.clocks.shape[1]-1))
    speed, tilt, force, margins = constraint_values(flats, angles, controller, policy, rotor_limit)
    clock_margin = min(float(derivative(c, t).min()) for c,t in zip(reference.clocks, reference.segment_durations))
    return flats, angles, speed, force, dict(
        minimum_planning_margin=float(margins.min()), speed_cap_passed=bool(speed.max() <= policy['speed_cap_mps']+1.e-6),
        reference_tilt_peak_deg=float(tilt.max()), reference_heading_rate_peak_degps=float(np.degrees(np.abs(angles[1]).max())),
        reference_waypoint_speed_mps=float(speed[samples_per_segment::samples_per_segment].min(initial=float('inf'))),
        monotonicity_passed=bool(clock_margin > 0.),
    )


def passage_reference(geometry, radius, node_count):
    """Select a reference speed check near each knot, inside the waypoint ball."""
    tests = []
    grid = np.linspace(0.,1.,node_count)
    for i, point in enumerate(geometry.points[1:-1],1):
        near = []
        for segment in (i-1,i):
            position, velocity = [basis(len(control)-1,grid) @ control for control in geometry.paths[segment][:2]]
            eligible = np.linalg.norm(position-point,axis=1) <= radius
            if not eligible.any():
                continue
            physical = np.linalg.norm(velocity,axis=1) / geometry.segment_durations[segment]
            index = int(np.argmax(np.where(eligible,physical,-1.)))
            near.append((segment,grid[index],float(physical[index])))
        if not near:
            raise ValueError('Frozen waypoint is outside its reference passage window')
        tests.append(max(near,key=lambda entry:entry[2]))
    return tests


def optimize_reference(base, heading, controller, policy, rotor_limit, raw):
    n = len(base.segment_durations)
    source = base.base.segment_durations
    scale = source.sum()/base.duration
    passage = passage_reference(base, .20, 97)
    limits = [(.50, 1.35)] * n + [(.25, 2.7)] * (n+1) + [(-.18,.18)]*n
    bounds = np.asarray(limits)
    objective = np.r_[base.segment_durations, np.zeros(2*n+1)]
    grid = mesh_for(81,8)

    def unpack(x):
        t = base.segment_durations*x[:n]
        return t, clocks_for(t, scale*x[n:2*n+1], source, x[2*n+1:])

    def margins(x):
        t, clocks = unpack(x)
        flats, angles = segment_values(base, t, clocks, heading, grid)
        _, _, _, envelope = constraint_values(flats, angles, controller, policy, rotor_limit)
        return np.r_[envelope-1.e-3, np.diff(clocks,axis=1).ravel()-1.e-3]

    def passage_speed(reference):
        from scipy.optimize import brentq
        speeds = []
        for segment, u, _ in passage:
            clock = reference.clocks[segment]
            local_time = brentq(lambda x: basis(len(clock)-1,np.asarray(x))@clock-u,0.,1.)
            value = reference.segment(segment,np.asarray(local_time))
            speeds.append(float(np.linalg.norm(value[1])))
        return min(speeds,default=policy['reference_waypoint_speed_mps'])

    receipt = {}
    if raw is None:
        return None, dict(status='no_upstream_solution')
    node_time = np.asarray(raw['raw_times_s'])
    raw_phase = np.linspace(0., base.base.duration, len(node_time))
    knot_times = np.interp(base.base.times, raw_phase, node_time)
    h = np.asarray(raw['raw_h'])
    rates = np.sqrt(np.maximum(np.interp(base.base.times, raw_phase, h), 1.e-8)) / policy['source_nominal_mps']
    initial = np.clip(np.r_[np.diff(knot_times)/base.segment_durations, rates/scale, np.zeros(n)], bounds[:,0], bounds[:,1])
    receipt['raw_traversal_s'] = float(node_time[-1])
    safe_guess = np.clip(np.r_[np.ones(n),np.ones(n+1),np.zeros(n)], bounds[:,0], bounds[:,1])
    warm = safe_guess + .10*(initial-safe_guess)
    trials = []
    for label, guess in [('upstream_blend', warm), ('baseline', safe_guess)]:
        fit = minimize(lambda x: objective@x, guess, jac=lambda _: objective, method='SLSQP',
                       bounds=limits, constraints=[dict(type='ineq', fun=margins)],
                       options=dict(maxiter=45, ftol=1.e-7))
        candidate, check = None, {}
        t, clocks = unpack(fit.x)
        for factor in (1., 1.004, 1.012, 1.025):
            fitted = MappedReference(None, t*factor, clocks, geometry=base)
            _, _, _, _, check = validate_reference(fitted, heading, controller, policy, rotor_limit)
            check['reference_inside_waypoint_speed_mps'] = passage_speed(fitted)
            if (check['minimum_planning_margin'] >= -1.e-6 and check['monotonicity_passed'] and
                    check['reference_inside_waypoint_speed_mps'] >= policy['reference_waypoint_speed_mps']-1.e-6):
                candidate = fitted
                break
        trials.append(dict(initialization=label, optimizer_success=bool(fit.success), optimizer_message=str(fit.message),
                           iterations=int(fit.nit), candidate_duration_s=fitted.duration, **check))
        if candidate is not None:
            return candidate, dict(receipt, trials=trials, status='reference_checked_on_grid')
    # A clipped, smooth blend with the validated baseline gives a feasible
    # search direction when the nonlinear solver cannot enter the envelope.
    for fraction in (.30,.15,.07):
        blend = safe_guess*(1.-fraction) + initial*fraction
        t, c = unpack(blend)
        fitted = MappedReference(None,t,c,geometry=base)
        _, _, _, _, check = validate_reference(fitted,heading,controller,policy,rotor_limit)
        check['reference_inside_waypoint_speed_mps'] = passage_speed(fitted)
        if (check['minimum_planning_margin'] >= -1.e-6 and check['monotonicity_passed'] and
                check['reference_inside_waypoint_speed_mps'] >= policy['reference_waypoint_speed_mps']-1.e-6):
            return fitted, dict(receipt,trials=trials,upstream_fraction=fraction,status='reference_checked_from_blend')
    return None, dict(receipt, trials=trials, status='reference_check_failed')


def sample_heading(reference, heading, times):
    indices = np.clip(np.searchsorted(reference.times, times, side='right')-1, 0, len(reference.times)-2)
    yaw, rate = np.zeros_like(times), np.zeros_like(times)
    for index, duration in enumerate(reference.segment_durations):
        mask = indices == index
        t = np.clip((times[mask]-reference.times[index])/duration, 0., 1.)
        s, v, a = [basis(len(c)-1, t) @ c for c in reference.rates[index][:3]]
        source_d = reference.base.segment_durations[index]
        yaw[mask], rate[mask], _ = heading.sample(reference.base.times[index]+source_d*s, source_d*v, source_d*a)
    rate[times >= reference.duration] = 0.
    return yaw, rate
