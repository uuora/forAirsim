"""Caller-specific lower bound for the obstacle-free rest-to-rest objective.

For q=q_quintic+64u^3(1-u)^3 d with endpoint v=a=0, integration by
parts gives E_jerk=(720||p1-p0||^2+(147456/7)||d||^2)/T^5.
The quintic/bump jerk cross term vanishes. The nonnegative tracking-quality
term can be dropped. If the objective's own reference derivative upper-bound
tests fail, their known penalty terms lower-bound its infeasible score.

This bounds the declared program's score, not aircraft safety. The engineering
roundoff cushion is not directed interval arithmetic. Do not reuse with
nonzero endpoint velocities, a different curve, weights or penalty definition.
"""
import numpy as np
from .parameterizations import BoundaryStates,curve_from_parameters
from .curve_task import subdivided_derivative_norm_bound


def target_quality_lower_bound(parameters,boundary,representation='hermite'):
    b=BoundaryStates(**boundary)
    if any(np.any(np.asarray(getattr(b,k))!=0) for k in ('v0','v1','a0','a1')):
        raise ValueError('lower bound requires rest-to-rest zero-acceleration endpoints')
    x=np.asarray(parameters,dtype=float)
    curve=curve_from_parameters(x,family='single_bump',representation=representation,boundary=b)
    ratios=[subdivided_derivative_norm_bound(curve,d,depth=3)/cap for d,cap in zip((1,2,3),(4.,4.,20.))]
    excess=sum(max(0.,r-1.) for r in ratios)
    if excess>0:return float(1.+excess-1e-9)
    T=float(x[-1]);distance2=float(np.sum((np.asarray(b.p1)-b.p0)**2));displacement2=float(x[:3]@x[:3])
    energy=(720*distance2+(147456/7)*displacement2)/T**5
    return float(.7*T/(T+5)+.1*energy/(energy+30)-1e-9)
