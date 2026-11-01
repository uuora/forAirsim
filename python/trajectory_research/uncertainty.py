"""Declared geometric uncertainty for one sphere, without a wind model.

Nominal centre c0=(4,0,5) m, nominal radius r0=.4 m and clearance h=.1 m.
The uncertain centre obeys ||c-c0||<=.30 m and radius obeys r<=r0+.05 m.
For a reference point p, the reverse triangle inequality gives

    ||p-c|| >= ||p-c0|| - .30.

Consequently ||p-c0|| > (.4+.05+.30)+.1 implies ||p-c|| > r+.1 for every
sphere in the declared envelope. The equivalent training radius is .75 m;
clearance remains .1 m and must not be added to the radius a second time.
This is a sufficient geometric condition on a polynomial position reference.
It assumes the stated envelope and does not include tracking error, sensing
calibration, model identification, physical external forces or wind rejection.

Random stress cases below sample centres uniformly by volume inside the error
ball and radii uniformly in [.35,.45] m. They are a declared stress distribution,
not evidence that real errors have that distribution or that finite samples
establish a probability guarantee. No optimizer results are consumed here.
"""
from __future__ import annotations

import numpy as np

from .curve_task import CurveCase, evaluate_curve_parameters


NOMINAL_CENTRE_M = (4., 0., 5.)
NOMINAL_RADIUS_M = .4
CLEARANCE_M = .1
CENTRE_ERROR_RADIUS_M = .30
RADIUS_INCREMENT_M = .05
EQUIVALENT_UNCERTAINTY_RADIUS_M = (
    NOMINAL_RADIUS_M + RADIUS_INCREMENT_M + CENTRE_ERROR_RADIUS_M
)
STRESS_RADIUS_RANGE_M = (.35, .45)


def make_nominal_training_cases():
    """One nominal sphere; the same number of evaluations as margin training."""
    return (CurveCase("nominal_single_sphere", NOMINAL_CENTRE_M,
                      NOMINAL_RADIUS_M, CLEARANCE_M),)


def make_uncertainty_training_cases():
    """One inflated sphere giving a sufficient geometric envelope condition."""
    return (CurveCase("uncertainty_margin_single_sphere", NOMINAL_CENTRE_M,
                      EQUIVALENT_UNCERTAINTY_RADIUS_M, CLEARANCE_M),)


def evaluate_uncertainty_parameters(parameters, profile="nominal"):
    """Use the existing reference evaluator under nominal or margin geometry.

    Both profiles preserve the same quality objective, curve, derivative limits,
    workspace and single-sphere case count. Only the feasibility geometry differs.
    """
    if profile not in ("nominal", "margin"):
        raise ValueError("profile must be 'nominal' or 'margin'")
    cases = make_nominal_training_cases() if profile == "nominal" else make_uncertainty_training_cases()
    result = evaluate_curve_parameters(parameters, cases)
    result["uncertainty_profile"] = profile
    result["uncertainty_model"] = dict(
        nominal_centre_m=list(NOMINAL_CENTRE_M), nominal_radius_m=NOMINAL_RADIUS_M,
        centre_error_radius_m=CENTRE_ERROR_RADIUS_M, maximum_radius_increment_m=RADIUS_INCREMENT_M,
        equivalent_margin_radius_m=EQUIVALENT_UNCERTAINTY_RADIUS_M, clearance_m=CLEARANCE_M,
        scope="declared geometric error envelope only; no tracking or physical-force guarantee",
    )
    return result


def _integer(value, name, *, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def make_uncertainty_stress_cases(seed, count=64):
    """Sample reproducible geometric stress cases without consulting outcomes.

    Isotropic normal directions are uniform on the sphere; U**(1/3)
    supplies uniform volume, rather than a uniform radial distribution.
    Separate seeds for optimization and this held-out generator are the caller's
    responsibility. The function does not freeze or choose optimizer candidates.
    """
    seed = _integer(seed, "seed", minimum=0)
    count = _integer(count, "count", minimum=1)
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(count, 3))
    norms = np.linalg.norm(direction, axis=1)
    # The zero normal vector has probability zero under the model. Handle a
    # finite-precision occurrence explicitly without producing a NaN direction.
    while np.any(norms == 0):
        zero = norms == 0
        direction[zero] = rng.normal(size=(int(np.sum(zero)), 3))
        norms[zero] = np.linalg.norm(direction[zero], axis=1)
    direction /= norms[:, None]
    radial = CENTRE_ERROR_RADIUS_M * np.cbrt(rng.random(count))
    centres = np.array(NOMINAL_CENTRE_M) + direction * radial[:, None]
    radii = rng.uniform(*STRESS_RADIUS_RANGE_M, size=count)
    return [CurveCase(f"geometric_uncertainty_stress_s{seed}_{index:04d}",
                      tuple(float(x) for x in centre), float(radius), CLEARANCE_M)
            for index, (centre, radius) in enumerate(zip(centres, radii))]
