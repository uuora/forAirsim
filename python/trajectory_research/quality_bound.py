"""Closed-form quality bound for the declared default curve-perturbation task.

This module applies ONLY to parameterizations.BoundaryStates() defaults:
p0=(0,0,5), p1=(8,0,5), v0=v1=(1,0,0), a0=a1=0. It deliberately has no
custom-boundary argument. A different task needs its own derivation.

For u=t/T, the baseline is p0 + T*u*ex + (8-T)*(10*u^3-15*u^4+6*u^5)*ex.
Its physical squared-jerk integral is 720*(8-T)^2/T^5. Every permitted
deformation and its first two derivatives vanish at both endpoints, so
integration by parts makes its jerk inner product with the quintic baseline
zero. The symmetric and asymmetric deformation modes are mutually orthogonal.
Consequently their nonnegative jerk energies and squared RMS values add.

The original oracle's score is Q when feasible and >= 1+Q otherwise; Q is
therefore a lower bound regardless of geometry certificates. The closed form
is exact in real arithmetic. A fixed engineering cushion is subtracted for
floating arithmetic; this is not an interval-arithmetic proof or a physical
safety certificate. No curve construction, integration or simulation occurs
here. The derivation is specific algebra, not a new curve family.
"""
from __future__ import annotations

import math

from .parameterizations import _physical


NUMERICAL_CUSHION = 1e-10


def quality_terms(parameters, *, family="paired_bump"):
    """Return default-task quality components without constructing the curve.

    Parameters use the existing canonical boxes: [dx,dy,dz,T] for single_bump,
    [dx,dy,dz,ex,ey,ez,T] for paired_bump. Displacements and normalized-u
    tangent increments are in metres; T is in seconds. Input validation is
    shared with the original curve decoder.
    """
    vector = _physical(parameters, family)
    displacement_squared = float(vector[:3] @ vector[:3])
    tangent_squared = float(vector[3:6] @ vector[3:6]) if family == "paired_bump" else 0.
    duration = float(vector[-1])
    denominator = duration ** 5
    baseline_energy = 720. * (8. - duration) ** 2 / denominator
    deformation_energy = ((147456. / 7.) * displacement_squared
                          + 4096. * tangent_squared) / denominator
    energy = baseline_energy + deformation_energy
    deformation_rms = math.sqrt((1024. / 3003.) * displacement_squared
                                + (256. / 45045.) * tangent_squared)
    quality = .9 * duration / (duration + 5.) + .05 * energy / (energy + 30.) \
        + .05 * deformation_rms / (deformation_rms + 1.)
    return dict(duration_s=duration, baseline_jerk_energy_m2_s5=baseline_energy,
                deformation_jerk_energy_m2_s5=deformation_energy,
                jerk_energy_m2_s5=energy, deformation_rms_m=deformation_rms,
                quality_term=quality)


def quality_lower_bound(parameters, *, family="paired_bump"):
    """Bound the default task's full scalar score, with a numerical cushion.

    This does not predict feasibility. It must not be used for custom endpoint
    states or a changed objective merely because their parameter sizes agree.
    """
    return float(quality_terms(parameters, family=family)["quality_term"] - NUMERICAL_CUSHION)


__all__ = ["NUMERICAL_CUSHION", "quality_terms", "quality_lower_bound"]
