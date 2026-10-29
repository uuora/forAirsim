"""Independent time-indexed reference tracking for offline curve research.

This is a fixed velocity-command tracker and a first-order lagged point mass,
not the production RollingController, AirSim, or an aircraft dynamics model.
Reference and actual states are compared at the SAME elapsed times on [0,T].
No terminal hold is appended: a curve ending with nonzero velocity continues
to have that reference velocity at T. Commands are zero-order held over each
declared interval; the lagged model is integrated in closed form.

Command speed projection and command slew limiting are observable operations
here. They must not be described as motor/thrust saturation. This module does
not decide feasibility, collision avoidance, or physical flight safety.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
import time

import numpy as np

from .curves import PolynomialSegment


MODEL = "fixed time-indexed velocity tracker with first-order lagged 3D point mass"
SCOPE = (
    "Offline research model; not production RollingController, AirSim or real flight. "
    "No rotor, attitude, thrust, wind, sensing, collision or feasibility model. "
    "Tracking errors are synchronized sampled-reference errors; command clipping "
    "is not motor saturation."
)
MAX_STEPS = 100_000


def _scalar(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _vector(value, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three finite real numbers") from error
    if raw.shape != (3,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain three finite real numbers")
    result = raw.astype(float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite real numbers")
    return result


def _norm(value, name):
    result = math.hypot(*(float(x) for x in value))
    if not math.isfinite(result):
        raise ValueError(f"{name} norm exceeded finite range")
    return result


@dataclass(frozen=True)
class TrackingCondition:
    """Explicit fixed tracker/model settings, using seconds and SI commands.

    ``max_command_slew_mps2`` limits the change of velocity COMMAND per second;
    it is not a rotor-level acceleration constraint. ``timestep_s`` is an upper
    bound: ceil(T/dt) equal intervals end exactly at the curve's physical T.
    A caller changing these settings must record a different condition.
    """

    velocity_lag_s: float = .30
    timestep_s: float = .02
    position_gain: float = 2.0
    max_command_speed_mps: float = 4.0
    max_command_slew_mps2: float = 4.0

    def __post_init__(self):
        for name in ("velocity_lag_s", "timestep_s", "position_gain",
                     "max_command_speed_mps", "max_command_slew_mps2"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name,
                               allow_zero=name == "position_gain"))


def run_timed_tracking(segment, condition=None, *, initial_position_offset_m=(0., 0., 0.)):
    """Return a complete, JSON-serializable deterministic tracking rollout.

    p_dot=v, v_dot=(c-v)/tau; c_raw=q_dot+kp*(q-p). Project c_raw into the
    command-speed ball, then limit its change from the preceding command.
    Initial actual velocity and preceding command equal q_dot(0). Actual
    position is q(0) plus the explicitly recorded position offset.

    State/reference arrays contain N+1 timestamps including 0 and T. Command
    arrays contain N entries, one for each [times_s[i],times_s[i+1]) interval.
    Position RMSE uses trapezoidal quadrature of squared synchronous error on
    this actual time grid. It is a sampled metric, not an exact error integral.
    At most MAX_STEPS intervals are accepted to bound accidental allocation.
    """
    started = time.perf_counter()
    if not isinstance(segment, PolynomialSegment) or segment.coefficients.shape[1] != 3:
        raise ValueError("segment must be a three-dimensional PolynomialSegment")
    if condition is None:
        condition = TrackingCondition()
    if not isinstance(condition, TrackingCondition):
        raise ValueError("condition must be a TrackingCondition")
    offset = _vector(initial_position_offset_m, "initial_position_offset_m")
    duration = segment.duration_s
    ratio = duration / condition.timestep_s
    if not math.isfinite(ratio) or ratio > MAX_STEPS:
        raise ValueError(f"rollout requires more than {MAX_STEPS} intervals")
    count = max(1, math.ceil(ratio))
    times = np.linspace(0., duration, count+1)
    intervals = np.diff(times)
    if not np.isfinite(intervals).all() or np.any(intervals <= 0):
        raise ValueError("rollout time grid must have finite positive intervals")
    reference_position = segment.sample(times, derivative=0)
    reference_velocity = segment.sample(times, derivative=1)
    if not np.isfinite(reference_position).all() or not np.isfinite(reference_velocity).all():
        raise ValueError("reference states must be finite")
    previous_command = reference_velocity[0].copy()
    if _norm(previous_command, "initial reference velocity") > condition.max_command_speed_mps:
        raise ValueError("initial reference velocity exceeds the command-speed cap")

    position = np.empty_like(reference_position)
    velocity = np.empty_like(reference_velocity)
    raw_commands = np.empty((count,3))
    speed_commands = np.empty((count,3))
    commands = np.empty((count,3))
    command_slew = np.empty(count)
    speed_projection = np.empty(count)
    slew_projection = np.empty(count)
    speed_clipped = np.zeros(count, dtype=bool)
    slew_clipped = np.zeros(count, dtype=bool)

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            position[0] = reference_position[0] + offset
            velocity[0] = reference_velocity[0]
            for i, step in enumerate(intervals):
                raw = reference_velocity[i] + condition.position_gain * (reference_position[i]-position[i])
                raw_norm = _norm(raw, "raw velocity command")
                speed_limited = raw * min(1., condition.max_command_speed_mps/max(raw_norm, 1e-300))
                speed_clipped[i] = raw_norm > condition.max_command_speed_mps
                delta = speed_limited - previous_command
                delta_norm = _norm(delta, "velocity command change")
                allowed_change = condition.max_command_slew_mps2 * step
                command = previous_command + delta * min(1., allowed_change/max(delta_norm, 1e-300))
                slew_clipped[i] = delta_norm > allowed_change
                raw_commands[i] = raw
                speed_commands[i] = speed_limited
                commands[i] = command
                command_slew[i] = _norm(command-previous_command, "limited command change")/step
                speed_projection[i] = _norm(speed_limited-raw, "speed projection difference")
                slew_projection[i] = _norm(command-speed_limited, "slew projection difference")

                # expm1 avoids loss in 1-exp(-h/tau) for small h/tau.
                alpha = -math.expm1(-float(step)/condition.velocity_lag_s)
                velocity[i+1] = velocity[i] + (command-velocity[i])*alpha
                position[i+1] = position[i] + command*step \
                    + (velocity[i]-command)*condition.velocity_lag_s*alpha
                previous_command = command

            errors = reference_position-position
            squared_error = np.sum(errors*errors, axis=1)
            position_error = np.sqrt(squared_error)
            rmse = math.sqrt(float(np.sum(.5*(squared_error[:-1]+squared_error[1:])*intervals)/duration))
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("tracking rollout exceeded finite floating-point range") from error
    if not all(np.isfinite(x).all() for x in (position, velocity, raw_commands,
            speed_commands, commands, command_slew, speed_projection, slew_projection, position_error)) \
            or not math.isfinite(rmse):
        raise ValueError("tracking rollout produced a nonfinite result")
    peak = int(np.argmax(position_error))
    metrics = dict(position_rmse_m=rmse,
        max_position_error_m=float(position_error[peak]),
        peak_position_error_time_s=float(times[peak]),
        peak_position_error_u=float(times[peak]/duration),
        terminal_position_error_m=float(position_error[-1]),
        terminal_velocity_error_mps=_norm(reference_velocity[-1]-velocity[-1], "terminal velocity error"),
        speed_clipped_time_fraction=float(np.sum(intervals[speed_clipped])/duration),
        command_slew_clipped_time_fraction=float(np.sum(intervals[slew_clipped])/duration),
        max_velocity_command_mps=max(_norm(x, "velocity command") for x in commands),
        max_command_slew_mps2=float(np.max(command_slew)))
    return dict(model=MODEL, scope=SCOPE, condition=asdict(condition), duration_s=duration,
        intervals=count, actual_max_timestep_s=float(np.max(intervals)),
        initial_position_offset_m=offset.tolist(), initial_velocity_command_mps=reference_velocity[0].tolist(),
        metrics=metrics, times_s=times.tolist(),
        reference_position_m=reference_position.tolist(), reference_velocity_mps=reference_velocity.tolist(),
        actual_position_m=position.tolist(), actual_velocity_mps=velocity.tolist(),
        position_error_m=position_error.tolist(),
        command_times_s=times[:-1].tolist(), command_intervals_s=intervals.tolist(),
        raw_velocity_command_mps=raw_commands.tolist(), speed_limited_velocity_command_mps=speed_commands.tolist(),
        velocity_command_mps=commands.tolist(), command_slew_mps2=command_slew.tolist(),
        speed_projection_difference_mps=speed_projection.tolist(),
        command_slew_projection_difference_mps=slew_projection.tolist(),
        speed_clipped=speed_clipped.tolist(), command_slew_clipped=slew_clipped.tolist(),
        rmse_definition="sqrt(trapezoidal integral of ||q(t_i)-p(t_i)||^2 / T)",
        evaluation_wall_s=time.perf_counter()-started)


__all__ = ["MODEL", "SCOPE", "MAX_STEPS", "TrackingCondition", "run_timed_tracking"]
