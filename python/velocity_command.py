"""Velocity-command limiter from the forAirsim flight controller."""
import math
import numpy as np


def limit_command(raw, previous, dt, speed_cap, slew_cap):
    raw, previous = np.asarray(raw, dtype=float), np.asarray(previous, dtype=float)
    if raw.shape != (3,) or previous.shape != (3,) or not np.isfinite(raw).all() \
            or not np.isfinite(previous).all() or not math.isfinite(dt) or dt <= 0:
        raise ValueError("finite 3D commands and positive observed ramp interval required")
    norm = float(np.linalg.norm(raw))
    speed = raw * min(1., speed_cap / max(norm, 1e-300))
    change = speed - previous
    change_norm = float(np.linalg.norm(change))
    limited = previous + change * min(1., slew_cap * dt / max(change_norm, 1e-300))
    return dict(raw=raw.tolist(), speed_limited=speed.tolist(), limited=limited.tolist(),
                speed_clipped=bool(norm > speed_cap), slew_clipped=bool(change_norm > slew_cap * dt),
                ramp_interval_s=float(dt), command_slew_mps2=float(np.linalg.norm(limited-previous)/dt))
