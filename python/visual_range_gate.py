"""Freshness gate for the next range-guided flight stage; no RPC calls."""
import math
from visual_servo import ServoCommand, command


def range_command(observation, tracking_state, age_s):
    if not math.isfinite(age_s) or age_s < 0 or age_s > 0.5:
        return ServoCommand((0,0,0),'STALE_OBSERVATION')
    return command(observation.detected and tracking_state == 'DETECTED',
                   observation.normalised_error_xy, observation.depth_m,
                   observation.truncated)
