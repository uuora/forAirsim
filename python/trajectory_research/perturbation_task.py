"""Shared curve-perturbation tasks for fair black-box comparisons.

The search operates on a canonical physical (d,e,T) box. Hermite and Bezier
decoders therefore describe the same candidate distribution, not two unrelated
control-point boxes. Each optimizer solves each declared layout separately.
Robust geometry checks are deterministic envelopes; held-out draws check the
implementation within those envelopes and are not new independent flights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .constraints import convex_corridor_certificate, sphere_clearance_certificate
from .curve_task import (DERIVATIVE_LIMITS, WORKSPACE_LOWER, WORKSPACE_UPPER,
                         jerk_energy, subdivided_derivative_norm_bound)
from .directional import ellipsoid_sphere_certificate
from .curves import PolynomialSegment
from .parameterizations import curve_from_parameters, parameter_bounds


SCOPE = "offline polynomial point-reference optimization; no aircraft, controller, wind or perception simulation"
GEOMETRY_FACTOR = ((.30, 0., 0.), (0., .05, 0.), (0., 0., .05))
RADIUS_INCREMENT = .05
CLEARANCE = .1
# Integral of [64*u^3*(1-u)^3]^2 on [0,1]; the second basis has 1/60 of this.
SHAPE_SQUARED_INTEGRAL = 1024. / 3003.


@dataclass(frozen=True)
class Obstacle:
    centre: tuple[float, float, float]
    radius: float = .4


@dataclass(frozen=True)
class Layout:
    name: str
    obstacles: tuple[Obstacle, ...]


def layouts():
    """Three development layouts, frozen before optimization comparisons.

    No claim of generalization to an unseen layout is attached to this suite.
    Different obstacle counts have different per-FE computation costs, reported
    separately. Within each layout all methods receive identical oracles.
    """
    return (
        Layout("central", (Obstacle((4., 0., 5.)),)),
        Layout("early_offset", (Obstacle((2.8, .15, 5.)),)),
        Layout("two_obstacles", (Obstacle((2.8, .15, 5.)), Obstacle((5.2, -.15, 5.)))),
    )


def get_layout(name):
    for layout in layouts():
        if layout.name == name:
            return layout
    raise ValueError(f"unknown layout: {name}")


def _layout(value):
    if isinstance(value, str):
        value = get_layout(value)
    if not isinstance(value, Layout) or not value.obstacles:
        raise ValueError("a nonempty Layout is required")
    for obstacle in value.obstacles:
        centre = np.asarray(obstacle.centre)
        if centre.shape != (3,) or centre.dtype.kind not in "iuf" or not np.isfinite(centre).all() \
                or isinstance(obstacle.radius, (bool, np.bool_)) \
                or not isinstance(obstacle.radius, (int, float, np.number)) \
                or not math.isfinite(obstacle.radius) or obstacle.radius <= 0:
            raise ValueError("finite 3D sphere geometry is required")
    return value


def shape_rms(parameters, *, family="paired_bump"):
    """Exact RMS deformation in normalized u, including the asymmetric mode."""
    vector = np.asarray(parameters, dtype=float)
    dim = 7 if family == "paired_bump" else 4 if family == "single_bump" else 0
    if not dim or vector.shape != (dim,) or not np.isfinite(vector).all():
        raise ValueError("invalid shape parameter vector")
    tangent_square = float(vector[3:6] @ vector[3:6]) if dim == 7 else 0.
    return math.sqrt(SHAPE_SQUARED_INTEGRAL * (float(vector[:3] @ vector[:3]) + tangent_square / 60.))


def evaluate_parameters(parameters, *, family="paired_bump", representation="hermite",
                        layout="central", robust=True, actual_obstacles=None):
    """One FE: complete geometry, derivative, workspace and quality evaluation.

    ``actual_obstacles`` is for frozen validation only and disables the robust
    envelope. It must contain one actual sphere for each declared obstacle.
    All candidate evaluation exceptions remain visible to the caller.
    """
    if not isinstance(robust, bool):
        raise ValueError("robust must be boolean")
    scene = _layout(layout)
    segment = curve_from_parameters(parameters, family=family, representation=representation)
    # Identical degree for identical-curve hull checks. Degree elevation can
    # tighten a convex hull, so checking degree-six and degree-seven controls
    # directly would confound added freedom with certificate conservatism.
    if segment.degree < 7:
        segment = PolynomialSegment(np.pad(segment.coefficients,
                                            ((0, 7 - segment.degree), (0, 0))), segment.duration_s)
    vector = np.asarray(parameters, dtype=float)
    if actual_obstacles is not None:
        if robust:
            raise ValueError("actual validation spheres cannot use the training envelope")
        scene = _layout(Layout(scene.name, tuple(actual_obstacles)))
        if len(scene.obstacles) != len(_layout(layout).obstacles):
            raise ValueError("actual obstacle count differs from the layout")
    normals = np.vstack((np.eye(3), -np.eye(3)))
    offsets = np.concatenate((WORKSPACE_UPPER, -np.asarray(WORKSPACE_LOWER)))
    corridor = convex_corridor_certificate(segment, normals, offsets, max_depth=6)
    bounds = [subdivided_derivative_norm_bound(segment, order, depth=3) for order in (1, 2, 3)]
    checks = dict(workspace=corridor["status"] == "CERTIFIED")
    checks.update(zip(("speed", "acceleration", "jerk"),
                      (bound <= limit for bound, limit in zip(bounds, DERIVATIVE_LIMITS))))
    certificates = []
    for index, obstacle in enumerate(scene.obstacles):
        if robust:
            certificate = ellipsoid_sphere_certificate(
                segment, obstacle.centre, obstacle.radius, GEOMETRY_FACTOR,
                radius_increment=RADIUS_INCREMENT, clearance=CLEARANCE, max_depth=10)
        else:
            certificate = sphere_clearance_certificate(segment, obstacle.centre, obstacle.radius,
                                                        clearance=CLEARANCE, max_depth=10)
        checks[f"obstacle_{index}"] = certificate["status"] == "CERTIFIED"
        certificates.append(certificate)
    energy, deformation = jerk_energy(segment), shape_rms(vector, family=family)
    duration = float(segment.duration_s)
    quality = .9 * duration / (duration + 5.) + .05 * energy / (energy + 30.) \
        + .05 * deformation / (deformation + 1.)
    failures = sum(not value for value in checks.values())
    feasible = failures == 0
    return dict(score=float(quality if feasible else 1. + failures / len(checks) + quality),
                feasible=feasible, parameters=vector.tolist(), layout=scene.name,
                family=family, representation=representation, robust=robust,
                score_details=dict(quality_term=float(quality), duration_s=duration,
                                   jerk_energy_m2_s5=energy, deformation_rms_m=deformation,
                                   derivative_bounds=bounds, failed_checks=failures,
                                   total_checks=len(checks), checks=checks),
                geometry_certificates=certificates, workspace_certificate=corridor, scope=SCOPE)


def stress_obstacles(layout, *, seed, count=64):
    """Generate complete scenes, all obstacles perturbed within the fixed envelope."""
    scene = _layout(layout)
    for value, name in ((seed, "seed"), (count, "count")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
                or value < (0 if name == "seed" else 1):
            raise ValueError(f"invalid {name}")
    if count > 10000:
        raise ValueError("count must not exceed 10000")
    rng = np.random.default_rng(seed)
    factor = np.asarray(GEOMETRY_FACTOR)
    scenes = []
    for _ in range(count):
        actual = []
        for obstacle in scene.obstacles:
            z = rng.normal(size=3)
            z /= max(float(np.linalg.norm(z)), 1e-300)
            z *= rng.random() ** (1. / 3.)
            centre = np.asarray(obstacle.centre) + factor @ z
            lower_radius = max(float(np.nextafter(0., 1.)), obstacle.radius - RADIUS_INCREMENT)
            actual.append(Obstacle(tuple(float(x) for x in centre),
                                   float(rng.uniform(lower_radius, obstacle.radius + RADIUS_INCREMENT))))
        scenes.append(tuple(actual))
    return tuple(scenes)


def declaration():
    return dict(layouts=[asdict(x) for x in layouts()], geometry_factor_m=GEOMETRY_FACTOR,
                radius_increment_m=RADIUS_INCREMENT, clearance_m=CLEARANCE,
                derivative_limits=DERIVATIVE_LIMITS, workspace=[WORKSPACE_LOWER, WORKSPACE_UPPER],
                quality=".9*T/(T+5)+.05*jerk_energy/(jerk_energy+30)+.05*shape_rms/(shape_rms+1)",
                infeasible_score="1+failed_checks/total_checks+quality", scope=SCOPE)
