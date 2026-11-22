import math
import numpy as np
from .curves import PolynomialSegment


def jerk_energy(segment):
    """Analytically integrate actual physical squared jerk, in m^2/s^5.

    This integrates the complete polynomial coefficient products. It is not
    the bump-only increment and does not omit the baseline's jerk energy.
    """
    if not isinstance(segment, PolynomialSegment):
        raise ValueError("segment must be a PolynomialSegment")
    coefficients = segment.coefficients.astype(np.longdouble)
    for _ in range(3):
        if len(coefficients) == 1:
            return 0.
        coefficients = coefficients[1:] * np.arange(1, len(coefficients))[:, None]
    # Integrate in u first, then multiply by T/T^6.
    integral = np.longdouble(0.)
    for coordinate in coefficients.T:
        square = np.polynomial.polynomial.polymul(coordinate, coordinate)
        integral += np.sum(square / np.arange(1, len(square) + 1, dtype=np.longdouble))
    energy = float(integral / np.longdouble(segment.duration_s) ** 5)
    if not math.isfinite(energy) or energy < -1e-9:
        raise ValueError("squared-jerk integration exceeded numerical range")
    return max(0., energy)


def _split_half(control):
    level = control.copy()
    left, right = [level[0].copy()], [level[-1].copy()]
    while len(level) > 1:
        level = .5 * level[:-1] + .5 * level[1:]
        left.append(level[0].copy())
        right.append(level[-1].copy())
    return np.array(left), np.array(right[::-1])
