"""MINVO norm bounds for the cubic and quartic commands of a quintic reference.

Roots and leading magnitudes are from mit-acl/minvo at
884c9fd8e16da9879e80d7683012a21fa9f18f85, src/solutions.

BSD 3-Clause License

Copyright (c) 2020, Jesus Tordesillas Torres, MIT Aerospace Controls Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from fractions import Fraction
from math import comb, hypot

import numpy as np


_FACTORS = {
    3: (
        ("0.43020387241957075", ("1",), ("0.03088254123024711",)),
        ("0.83490734598607630", ("1",), ("-0.7735485873195366",)),
        ("0.43020387241957075", ("-1",), ("-0.03088254123024711",)),
        ("0.83490734598607630", ("-1",), ("0.7735485873195366",)),
    ),
    4: (
        ("1.1084608375595766", ("-1", "1"), ("-0.36572107484137545",)),
        ("0.5255197096031325", (), ("0.2872071541366619", "-0.8350357080112095")),
        ("1.165882255912888", (), ("0.8617752373038693", "-0.8617752373038693")),
        ("1.1084608375595766", ("-1", "1"), ("0.36572107484137545",)),
        ("0.5255197096031325", (), ("-0.2872071541366619", "0.8350357080112095")),
    ),
}


def _conversion(degree, factors):
    # Factoring on [-1, 1] keeps the rounded MINVO weights nonnegative.
    # Substitute x = 2s - 1 and compute Bernstein ordinates exactly at startup.
    rows = []
    for magnitude, endpoints, double_roots in factors:
        power = np.array([Fraction(magnitude)], dtype=object)
        for endpoint in endpoints:
            power = np.convolve(power, (0, 2) if endpoint == "-1" else (2, -2))
        for root in double_roots:
            factor = (-1 - Fraction(root), 2)
            power = np.convolve(power, np.convolve(factor, factor))
        rows.append([
            sum(power[j] * Fraction(comb(k, j), comb(degree, j)) for j in range(k + 1))
            for k in range(degree + 1)
        ])
    weight_sum_bound = float(max(map(sum, zip(*rows))))
    to_bernstein = np.asarray(rows, dtype=float).T
    return to_bernstein, np.linalg.inv(to_bernstein), weight_sum_bound


_CONVERSIONS = {degree: _conversion(degree, factors) for degree, factors in _FACTORS.items()}


def control_norm_bound(leaves):
    """Bound the norm from the Bernstein controls of equally subdivided leaves."""
    degree = len(leaves[0]) - 1
    if degree not in _CONVERSIONS:
        return max(hypot(*point) for leaf in leaves for point in leaf)
    to_bernstein, to_minvo, weight_sum_bound = _CONVERSIONS[degree]
    bernstein = np.asarray(leaves)
    minvo = to_minvo @ bernstein
    # The converted curve plus its Bernstein residual equals the input curve.
    # The sum bound accounts for rounding of the published MINVO parameters.
    residual = bernstein - to_bernstein @ minvo
    control_norms = np.hypot.reduce(minvo, axis=-1).max(axis=-1)
    residual_norms = np.hypot.reduce(residual, axis=-1).max(axis=-1)
    bound = float(np.max(weight_sum_bound * control_norms + residual_norms))
    return bound * (1.0 + 64.0 * np.finfo(float).eps)
