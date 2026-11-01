"""Anisotropic geometry uncertainty: an ablation of conservative clearances.

All profiles use the same curve variables, quality and initial population.
Only the geometry model differs. The uncertainty is about a sphere's geometry,
not wind, tracking error or the spatial extent of an actual aircraft.
"""
from __future__ import annotations

import numpy as np

from .constraints import sphere_clearance_certificate
from .curve_task import CurveCase, curve_from_parameters, evaluate_curve_parameters
from .directional import ellipsoid_sphere_certificate


CENTRE = (4., 0., 5.)
RADIUS = .4
RADIUS_INCREMENT = .05
CLEARANCE = .1
FACTOR = np.diag([.30, .05, .05])
PROFILES = ("ellipsoid_nominal", "ellipsoid_isotropic", "ellipsoid_directional")


def make_directional_training_cases():
    return (CurveCase("anisotropic_nominal_sphere", CENTRE, RADIUS, CLEARANCE),)


def evaluate_directional_parameters(parameters, *, profile="ellipsoid_directional"):
    if profile not in PROFILES:
        raise ValueError("Unknown anisotropic geometry profile")
    result = evaluate_curve_parameters(parameters, make_directional_training_cases())
    segment = curve_from_parameters(parameters)
    if profile == "ellipsoid_nominal":
        certificate = result["cases"][0]["sphere_certificate"]
    elif profile == "ellipsoid_isotropic":
        certificate = sphere_clearance_certificate(
            segment, CENTRE, RADIUS + RADIUS_INCREMENT + float(np.linalg.norm(FACTOR, 2)),
            clearance=CLEARANCE, max_depth=10)
    else:
        certificate = ellipsoid_sphere_certificate(
            segment, CENTRE, RADIUS, FACTOR, radius_increment=RADIUS_INCREMENT,
            clearance=CLEARANCE, max_depth=10)
    row = result["cases"][0]
    row["checks"]["sphere_clearance"] = certificate["status"] == "CERTIFIED"
    row["sphere_certificate"] = certificate
    row["passed"] = all(row["checks"].values())
    failures = sum(not passed for passed in row["checks"].values())
    result["feasible"] = row["passed"]
    result["score_details"].update(failed_checks=failures, failed_cases=int(not row["passed"]))
    quality = result["score_details"]["quality_term"]
    result["score"] = quality if row["passed"] else 1. + failures / 5. + quality
    result["geometry_profile"] = profile
    result["uncertainty_model"] = dict(centre_error_factor_m=FACTOR.tolist(), unit_ball_constraint="||z||_2<=1",
                                       radius_increment_m=RADIUS_INCREMENT, point_reference_clearance_m=CLEARANCE)
    return result


def make_directional_stress_cases(seed, *, count=128):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or not 1 <= count <= 10000:
        raise ValueError("count must be an integer in [1,10000]")
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-300)
    directions *= rng.random((count, 1)) ** (1. / 3.)
    centres = np.array(CENTRE) + directions @ FACTOR.T
    radii = rng.uniform(.35, .45, count)
    return tuple(CurveCase(f"ellipsoid_frozen_stress_{i:03d}", tuple(centre), float(radius), CLEARANCE)
                 for i, (centre, radius) in enumerate(zip(centres, radii)))
