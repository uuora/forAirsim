"""Path-speed optimization and flatness reconstruction for the paper comparison."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import roots_legendre

from .reference import FrozenReference

QUAD_X, QUAD_W = roots_legendre(8)
ARMS = ('flat', 'velocity', 'collective', 'flat_alpha', 'velocity_alpha', 'collective_alpha', 'toppquad')
LABELS = ('原始平坦性轨迹', '速度 TOPP', '速度＋总推力 TOPP', '平坦性 α', '速度 TOPP α', '速度＋总推力 TOPP α', 'TOPPQuad')


def routes(seed, count):
    rng = np.random.default_rng(seed)
    for number in range(1, count+1):
        points = [rng.uniform([-8., -8., -10.], [8., 8., -6.])]
        while len(points) < 4:
            point = rng.uniform([-8., -8., -10.], [8., 8., -6.])
            distances = np.linalg.norm(np.asarray(points)-point, axis=1)
            if 5. <= distances[-1] <= 11. and distances.min() >= 3.:
                points.append(point)
        yield dict(id=f'route_{number:03}', points_ned=np.asarray(points).tolist())


def optimize_speed(s, path, vmax, max_collective_acceleration=None):
    import casadi as ca

    opt = ca.Opti()
    h = opt.variable(len(s))
    speed = np.linalg.norm(path[:, 1], axis=1)
    upper = np.where(speed >= .001, np.minimum(100., vmax**2 / np.maximum(speed, .001)**2), 1.)
    upper[[0, -1]] = np.minimum(upper[[0, -1]], 1.)
    ds = float(s[1] - s[0])
    first = (h[1:] - h[:-1]) / ds
    third = (h[3:] - 3*h[2:-1] + 3*h[1:-2] - h[:-3]) / ds**3
    opt.subject_to(opt.bounded(1.e-5, h, upper))
    cost = ca.sum1(2*ds / (ca.sqrt(h[:-1]) + ca.sqrt(h[1:]))) + .1 * ca.sumsqr(third)
    opt.minimize(cost)
    if max_collective_acceleration is not None:
        for node_offset, profile in ((0, h[:-1]), (1, h[1:])):
            values = path[node_offset:node_offset+len(s)-1]
            a = ca.DM(values[:, 1].T) * ca.repmat(first.T / 2, 3, 1) + ca.DM(values[:, 2].T) * ca.repmat(profile.T, 3, 1)
            a += ca.repmat(ca.DM([0., 0., 9.81]), 1, len(s)-1)
            opt.subject_to(ca.sum1(a*a) <= max_collective_acceleration**2)
    opt.set_initial(h, min(.8, float(upper.min()) * .8))
    opt.solver('ipopt', {'expand': True, 'print_time': False, 'ipopt.print_level': 0, 'ipopt.sb': 'yes'},
               {'max_iter': 500, 'tol': 1.e-7})
    solution = opt.solve()
    return np.asarray(solution.value(h)).ravel(), int(solution.stats()['iter_count'])


def path_values(path, s):
    s = np.asarray(s)
    values = np.empty((len(s), 5, 3))
    for i, (left, right) in enumerate(zip(path.times, path.times[1:])):
        mask = (s >= left) & (s < right if i+2 < len(path.times) else s <= right)
        u = np.clip((s[mask]-left) / path.segment_durations[i], 0., 1.)
        values[mask] = np.array([[poly(u) for poly in axes] for axes in path.polynomials[i]]).transpose(2, 0, 1)
    return values


class PositiveRate:
    def __init__(self, nodes, h):
        self.log = CubicSpline(nodes, np.log(np.maximum(h, 1.e-6)), bc_type='natural')

    def __call__(self, s, order=0):
        value = np.exp(self.log(s))
        if not order:
            return value
        d = self.log(s, 1)
        if order == 1:
            return value*d
        dd = self.log(s, 2)
        if order == 2:
            return value*(dd+d*d)
        return value*(self.log(s, 3)+3*d*dd+d**3)


class SpeedReference:
    def __init__(self, base, source_nodes, h, alpha=1.):
        self.path = FrozenReference(base)
        self.points, self.end = self.path.points, self.path.end
        self.s = np.asarray(source_nodes)
        self.h = np.asarray(h)
        self.alpha = float(alpha)
        self.profile = CubicSpline(self.s, self.h, bc_type='natural')
        probes = np.r_[self.s, self.profile.derivative().roots(extrapolate=False)]
        probes = probes[np.isfinite(probes)]
        self.interpolation = 'cubic'
        if np.min(self.profile(probes)) <= 1.e-8:
            self.profile = PositiveRate(self.s, self.h)
            self.interpolation = 'log_cubic_floor_1e-6'
        self.node_times = np.r_[0., np.cumsum(self.interval(self.s[:-1], self.s[1:]))]
        self.times = self.time_at(self.path.times) * alpha
        self.duration = float(self.times[-1])
        self.segment_durations = np.diff(self.times)

    def interval(self, left, right):
        delta = (right-left)*.5
        samples = left[..., None] + delta[..., None]*(QUAD_X+1)
        return delta * (1/np.sqrt(self.profile(samples)) @ QUAD_W)

    def time_at(self, s):
        i = np.clip(np.searchsorted(self.s, s, side='right')-1, 0, len(self.s)-2)
        return self.node_times[i] + self.interval(self.s[i], s)

    def sample_batch(self, times):
        result = np.zeros((len(times), 5, 3))
        result[times > self.duration, 0] = self.end
        result[times < 0., 0] = self.points[0]
        keep = (times >= 0.) & (times <= self.duration)
        result[keep] = self.at_progress(self.progress_at_time(times[keep]))
        return result

    def progress_at_time(self, times):
        t = np.clip(times / self.alpha, 0., self.node_times[-1])
        i = np.clip(np.searchsorted(self.node_times, t, side='right')-1, 0, len(self.s)-2)
        low, high = self.s[i].copy(), self.s[i+1].copy()
        s = low + (high-low) * (t-self.node_times[i]) / (self.node_times[i+1]-self.node_times[i])
        for _ in range(26):
            err = self.time_at(s) - t
            if np.max(np.abs(err), initial=0.) < 1.e-12:
                break
            low, high = np.where(err < 0, s, low), np.where(err > 0, s, high)
            step = s - err*np.sqrt(self.profile(s))
            s = np.where((step >= low) & (step <= high), step, (low+high)/2)
        return s

    def at_progress(self, s):
        p, v, a, j, snap = path_values(self.path, s).transpose(1, 0, 2)
        h, d, dd, ddd = [self.profile(s, k)[:, None] for k in range(4)]
        q = np.sqrt(h)/self.alpha
        r = .5*d/self.alpha**2
        t = .5*dd*np.sqrt(h)/self.alpha**3
        u = (.5*ddd*h+.25*dd*d)/self.alpha**4
        return np.stack((p, v*q, a*q*q+v*r, j*q**3+3*a*q*r+v*t,
                         snap*q**4+6*j*q*q*r+a*(3*r*r+4*q*t)+v*u), axis=1)


def unit_derivatives(x, dx, ddx):
    norm = np.linalg.norm(x, axis=1)[:, None]
    if np.any(norm < 1.e-8):
        raise ValueError('Singular flatness frame')
    e = x/norm
    dn = np.sum(e*dx, axis=1)[:, None]
    de = (dx-e*dn)/norm
    ddn = np.sum(de*dx+e*ddx, axis=1)[:, None]
    dde = (ddx-2*de*dn-e*ddn)/norm
    return e, de, dde


def flat_thrust(flats, controller, yaw, yaw_rate=0., yaw_acceleration=0.):
    acceleration, jerk, snap = flats[:, 2], flats[:, 3], flats[:, 4]
    thrust = acceleration + [0., 0., controller.g]
    yaw = np.broadcast_to(yaw, (len(flats),))
    yaw_rate = np.broadcast_to(yaw_rate, yaw.shape)[:, None]
    yaw_acceleration = np.broadcast_to(yaw_acceleration, yaw.shape)[:, None]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    x_yaw = np.column_stack((cosine, sine, np.zeros_like(yaw)))
    tangent = np.column_stack((-sine, cosine, np.zeros_like(yaw)))
    dx_yaw = tangent * yaw_rate
    ddx_yaw = tangent * yaw_acceleration - x_yaw * yaw_rate**2
    z, dz, ddz = unit_derivatives(thrust, jerk, snap)
    y, dy, ddy = unit_derivatives(np.cross(z, x_yaw),
                                np.cross(dz, x_yaw) + np.cross(z, dx_yaw),
                                np.cross(ddz, x_yaw) + 2*np.cross(dz, dx_yaw) + np.cross(z, ddx_yaw))
    x = np.cross(y, z)
    dx = np.cross(dy, z) + np.cross(y, dz)
    ddx = np.cross(ddy, z) + 2*np.cross(dy, dz) + np.cross(y, ddz)
    R, dR, ddR = (np.stack(axes, axis=-1) for axes in ((x,y,z), (dx,dy,dz), (ddx,ddy,ddz)))
    Rt = R.transpose(0, 2, 1)
    skew = Rt @ dR
    dskew = dR.transpose(0, 2, 1) @ dR + Rt @ ddR
    i, j = [2, 0, 1], [1, 2, 0]
    w, dw = ((matrix[:, i, j] - matrix[:, j, i])*.5 for matrix in (skew, dskew))
    torque = dw @ controller.inertia.T + np.cross(w, w @ controller.inertia.T)
    collective = controller.mass * np.linalg.norm(thrust, axis=1)
    return np.column_stack((collective, torque)) @ controller.TM_to_f.T


def dense_progress(reference, subdivisions=12):
    knots = np.unique(np.r_[reference.s, reference.path.times])
    left, right = knots[:-1], knots[1:]
    # Both sides of knots are included for the piecewise snap used by the mixer.
    return np.r_[(left[:, None] + (right-left)[:, None]*np.arange(subdivisions)/subdivisions).ravel(),
                 np.nextafter(knots[1:], knots[:-1]), knots[-1]]
