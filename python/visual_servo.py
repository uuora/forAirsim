"""Camera-aligned body-NED velocity policy; no RPC calls in this module."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ServoCommand:
    velocity: tuple
    reason: str
    duration_s: float = 0.2


def stop_forward_at_range(velocity, reached):
    """Keep alignment corrections available after reaching the range boundary."""
    return (0.0, velocity[1], velocity[2]) if reached else velocity


def command(detected, error_xy, depth_m, truncated=False, stop_depth=2.0):
    """Requires a level, forward-facing camera and current confirmed detection.

    Positive image x is body right; positive image y is body down.
    A runner must separately enforce freshness, workspace bounds and timeout.
    """
    stop = lambda reason: ServoCommand((0.0, 0.0, 0.0), reason)
    if not detected or truncated or error_xy is None or depth_m is None:
        return stop('UNRELIABLE_OBSERVATION')
    if len(error_xy) != 2 or not all(math.isfinite(v) for v in (*error_xy, depth_m)):
        return stop('INVALID_OBSERVATION')
    if depth_m <= 0 or any(abs(v) > 1 for v in error_xy):
        return stop('INVALID_OBSERVATION')
    if depth_m <= stop_depth:
        return stop('DISTANCE_STOP')
    x, y = error_xy
    lateral = max(-0.15, min(0.15, 0.4*x)) if abs(x) > 0.08 else 0.0
    vertical = max(-0.15, min(0.15, 0.4*y)) if abs(y) > 0.08 else 0.0
    if lateral or vertical:
        return ServoCommand((0.0, lateral, vertical), 'ALIGN')
    return ServoCommand((0.15, 0.0, 0.0), 'ADVANCE')
