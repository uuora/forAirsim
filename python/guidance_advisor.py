"""Non-actuating visual guidance recommendations for audit before closed-loop use."""
from enum import Enum

from perception_state import TrackingState


class GuidanceAdvice(str, Enum):
    HOLD = "HOLD"
    ALIGN_LEFT = "ALIGN_LEFT"
    ALIGN_RIGHT = "ALIGN_RIGHT"
    ALIGN_UP = "ALIGN_UP"
    ALIGN_DOWN = "ALIGN_DOWN"
    ADVANCE = "ADVANCE"
    STOP = "STOP"


def advise(tracking_state, normalised_error_xy, depth_m,
           alignment_deadband=0.10, stop_distance_m=1.50):
    """Describe the next camera-frame action without sending a flight command."""
    state = TrackingState(tracking_state)
    if state != TrackingState.DETECTED or normalised_error_xy is None or depth_m is None:
        return GuidanceAdvice.HOLD
    error_x, error_y = normalised_error_xy
    if abs(error_x) > alignment_deadband:
        return GuidanceAdvice.ALIGN_RIGHT if error_x > 0 else GuidanceAdvice.ALIGN_LEFT
    if abs(error_y) > alignment_deadband:
        return GuidanceAdvice.ALIGN_DOWN if error_y > 0 else GuidanceAdvice.ALIGN_UP
    if depth_m <= stop_distance_m:
        return GuidanceAdvice.STOP
    return GuidanceAdvice.ADVANCE
