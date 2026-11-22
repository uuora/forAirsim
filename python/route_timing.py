"""TOPPQuad with the selected forAirsim route's speed, slew and passage conditions."""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from numpy.polynomial.legendre import leggauss

from trajectory_research.curves import quintic_hermite


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "3db181cb0c76baaa1669f859f16a83060c2ee7a0"
GRID = 100


@lru_cache(maxsize=1)
def author_solver():
    import subprocess

    checkout = ROOT / ".runtime/toppquad_upstream"
    path = checkout / "TOPPQuad/TQ_trajectory.py"
    blob = subprocess.check_output(["git", "-C", str(checkout), "show", UPSTREAM + ":TOPPQuad/TQ_trajectory.py"])
    if path.read_bytes().replace(b"\r\n", b"\n") != blob:
        raise RuntimeError("TOPPQuad solver differs from the published revision")
    sys.path.insert(0, str(ROOT / ".runtime/toppquad_deps"))
    spec = importlib.util.spec_from_file_location("author_toppquad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TQ_trajectory, hashlib.sha256(blob).hexdigest()


def path_sample(segments, ends, phase, order):
    phase = np.atleast_1d(phase)
    index = np.searchsorted(ends[:-1], phase, side="right")
    starts = np.r_[0., ends[:-1]]
    xyz = np.empty((len(phase), 3))
    for i in np.unique(index):
        mask = index == i
        piece = segments[i]
        t = np.clip(phase[mask]-starts[i], 0., piece.duration_s)
        if order <= 3:
            xyz[mask] = piece.sample(t, derivative=order)
        else:
            c = np.polynomial.polynomial.polyder(piece.coefficients, m=order, axis=0) / piece.duration_s**order
            xyz[mask] = np.polynomial.polynomial.polyval(t / piece.duration_s,c).T
    return xyz


def solve_motion(segments, speed, acceleration, dynamics_model, *, fixed_waypoint_speed=True):
    """Optimize motion only. Constant dwell is inserted by the route decoder."""
    base, digest = author_solver()
    import casadi as ca

    ends = np.cumsum([p.duration_s for p in segments])
    grid = np.linspace(0., ends[-1], GRID)
    ds = float(grid[1])
    via_phases = ends[:-1]
    fixed_via_phases = via_phases if fixed_waypoint_speed else ()
    # The baseline uses north/east/height. The author uses east/north/up.
    enu = [path_sample(segments, ends, grid, d)[:, [1, 0, 2]] for d in range(5)]
    zero = np.zeros((GRID, 3))

    class RouteTOPPQuad(base):
        def set_cost(self, opti, variables):
            h, _, w, _ = variables
            travel = ca.sum1(2*ds/(ca.sqrt(h[:-1])+ca.sqrt(h[1:])))
            # Same published cost, including its angular regularizer. The square
            # has a finite derivative when consecutive rates are both zero.
            opti.minimize(travel + 1.e-4*ca.sumsqr((w[:,1:]-w[:,:-1])/ds))

        def set_cons(self, opti, variables):
            super().set_cons(opti, variables)
            h = variables[0]
            # Interpret the author's independent h samples with a natural cubic
            # interpolant for the continuous velocity controller and exact via times.
            hpp = opti.variable(GRID)
            opti.set_initial(hpp, 0.)
            opti.subject_to(h >= 1.e-6)
            opti.subject_to(hpp[0] == 0.)
            opti.subject_to(hpp[-1] == 0.)
            opti.subject_to(hpp[:-2] + 4*hpp[1:-1] + hpp[2:] ==
                            6*(h[2:] - 2*h[1:-1] + h[:-2])/ds**2)
            slope = (h[1:]-h[:-1])/ds
            hp = ca.vertcat(slope - ds*(2*hpp[:-1]+hpp[1:])/6,
                            slope[-1] + ds*(hpp[-2]+2*hpp[-1])/6)
            # Positive Bézier controls certify h>0 between sampled nodes,
            # so dt/ds is defined throughout every exported interval.
            opti.subject_to(h[:-1]+ds*hp[:-1]/3 >= 1.e-6)
            opti.subject_to(h[1:]-ds*hp[1:]/3 >= 1.e-6)
            for via in fixed_via_phases:
                i = min(int(via / ds), GRID - 2)
                b = (via-grid[i]) / ds
                a = 1.-b
                value = a*h[i]+b*h[i+1]+ds**2*((a**3-a)*hpp[i]+(b**3-b)*hpp[i+1])/6
                opti.subject_to(value == 1.)
            def scaled(values, scalar):
                return ca.DM(values.T) * ca.repmat(scalar.T, 3, 1)
            acc = scaled(enu[2], h) + scaled(enu[1], hp/2)
            jerk_over_rate = scaled(enu[3], h) + scaled(enu[2], 1.5*hp) + scaled(enu[1], hpp/2)
            opti.subject_to(ca.sum1(acc**2) <= acceleration**2)
            opti.subject_to(ca.sum1(jerk_over_rate**2) * h.T <= 30.**2)
            # The inherited velocity controller sends v + 0.9 a at zero
            # tracking error. Its command and slew are both limited to six.
            rate = ca.sqrt(h)
            nominal_command = scaled(enu[1],rate) + .9*acc
            nominal_slew = acc + .9*jerk_over_rate*ca.repmat(rate.T,3,1)
            opti.subject_to(ca.sum1(nominal_command**2) <= speed**2)
            opti.subject_to(ca.sum1(nominal_slew**2) <= acceleration**2)

    solver = RouteTOPPQuad(enu[0][[0,-1]], n=GRID, vmax=speed)
    if dynamics_model["model"] != "GenericQuadX":
        raise ValueError("TOPPQuad requires the AirSim GenericQuadX dynamics")
    solver.g = np.array([0., 0., dynamics_model["nominal_gravity_mps2"]])
    hover_thrust_n = dynamics_model["mass_kg"] * solver.g[2] / 4
    # Divide both sides of the wrench equations by the same force scale.
    # This expresses the inherited thrust bounds and tolerances relative
    # to hover while preserving the GenericQuadX equations in physical units.
    newtons_per_unit = hover_thrust_n / solver.Trest
    solver.m = dynamics_model["mass_kg"] / newtons_per_unit
    solver.J = np.diag(dynamics_model["inertia_diag_kgm2"]) / newtons_per_unit
    solver.F_thrust = np.array(dynamics_model["thrust_allocation_flu"])
    solver.Tmax = dynamics_model["max_thrust_per_rotor_n"] / newtons_per_unit
    solver.umax = dynamics_model["max_rpm"] * (2 * np.pi / 60)
    solver.k_thrust = solver.Tmax / solver.umax**2
    data = dict(zip(("x", "x_dot", "x_ddot", "x_dddot", "x_ddddot"), enu))
    solver.set_min_snap_traj(dict(data, d_s=np.full(GRID-1, ds), h=np.ones(GRID),
                                  T=np.full((GRID,4), solver.Trest), vel=enu[1], acc=enu[2], ms=zero, t=grid))
    result, solved, seconds = solver.get_optimized_traj()
    if not solved:
        raise RuntimeError("TOPPQuad 未能求得运动段解")
    h = np.ravel(result["h"])
    spline = CubicSpline(grid, h, bc_type="natural")
    stationary = spline.derivative().roots(extrapolate=False)
    probes = np.r_[grid, stationary[np.isfinite(stationary)]]
    if not np.isfinite(h).all() or spline(probes).min() <= 0:
        raise ValueError("TOPPQuad 时间映射无效")
    raw_via_speed = np.linalg.norm(path_sample(segments, ends, via_phases, 1), axis=1)
    raw_via_speed *= np.sqrt(spline(via_phases))
    return dict(anchors=grid.tolist(), h=h.tolist(), solver_s=float(seconds),
                fixed_waypoint_speed=fixed_waypoint_speed,
                raw_waypoint_speeds_mps=raw_via_speed.tolist(),
                discrete_duration_s=float(result["t"][-1]), source_sha256=digest,
                upstream=UPSTREAM, nodes=GRID, hover_thrust_per_rotor_n=hover_thrust_n,
                thrust_step_limit_n=.01 * newtons_per_unit,
                motor_thrust_range_n=(np.array([np.min(result["T"]), np.max(result["T"])])
                                      * newtons_per_unit).tolist())


def export_motion(segments, solution):
    """Quintic position interpolation from path samples, physical velocities and accelerations."""
    ends = np.cumsum([p.duration_s for p in segments])
    anchors = np.array(solution["anchors"])
    h = CubicSpline(anchors, solution["h"], bc_type="natural")
    interior = anchors[np.min(np.abs(anchors[:,None]-ends),axis=1) > 1.e-8]
    edges = np.sort(np.r_[interior, ends])
    u, weights = leggauss(8)
    ds = np.diff(edges)
    s = edges[:-1,None] + ds[:,None]*(u+1.)/2
    dt = ds/2 * np.sum(weights/np.sqrt(h(s)), axis=1)
    p, ps, pss = [path_sample(segments, ends, edges, d) for d in range(3)]
    vel = ps * np.sqrt(h(edges))[:,None]
    acc = pss*h(edges)[:,None] + .5*ps*h(edges,1)[:,None]
    pieces = [quintic_hermite(p[i],vel[i],acc[i],p[i+1],vel[i+1],acc[i+1],float(delta))
              for i,delta in enumerate(dt)]
    end_indices = np.searchsorted(edges, ends)
    return pieces, end_indices
