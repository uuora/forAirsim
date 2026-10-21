"""Measured starting-state matching for the fixed-height AirSim pilot.

This module has no RPCs and never changes a vehicle pose. The caller owns the
fixed world-height to local-height conversion, command dispatch, full sample
logging, workspace guards, and protected return. Matching samples belong to a
pre-approach phase and must not enter the planner performance denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class MatchedStartParams:
    position_tolerance_m: float = .015
    speed_tolerance_m_s: float = .01
    acceleration_tolerance_m_s2: float = .05
    attitude_tolerance_deg: float = 1.
    stable_s: float = .5
    timeout_s: float = 30.
    max_sample_gap_s: float = .25
    proportional_gain: float = .8
    reposition_speed_cap_m_s: float = .05
    reposition_acceleration_cap_m_s2: float = .05
    command_dt_s: float = .1
    physical_speed_cap_m_s: float = .15
    physical_acceleration_cap_m_s2: float = .3
    physical_attitude_limit_deg: float = 7.5


def _positive_number(value, name):
    if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a numeric value, not text or boolean")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def validate_matched_start_config(config):
    """Validate full experiment cfg or an already constructed parameter object.

    The full cfg uses only ``initial_state_match`` and inherited ``planning``
    physical budgets. It cannot raise the inspected .15 m/s or .30 m/s² caps.
    """
    if isinstance(config, MatchedStartParams):
        params = config
    else:
        if not isinstance(config, Mapping) or not isinstance(config.get("initial_state_match"), Mapping):
            raise ValueError("Full experiment config must contain initial_state_match")
        section = config["initial_state_match"]
        if "enabled" in section and not isinstance(section["enabled"], bool):
            raise ValueError("initial_state_match.enabled must be boolean")
        planning = config.get("planning", {})
        if not isinstance(planning, Mapping):
            raise ValueError("planning must be a mapping")
        parameter_names = {field.name for field in fields(MatchedStartParams)}
        overrides = {name: section[name] for name in parameter_names if name in section}
        overrides.update(
            physical_speed_cap_m_s=planning.get("execution_speed_cap_m_s", .15),
            physical_acceleration_cap_m_s2=planning.get("max_acceleration_m_s2", .3),
            physical_attitude_limit_deg=planning.get("attitude_limit_deg", 7.5))
        params = MatchedStartParams(**overrides)
    for field in fields(params):
        _positive_number(getattr(params, field.name), field.name)
    if params.physical_speed_cap_m_s > .15 or params.physical_acceleration_cap_m_s2 > .3:
        raise ValueError("Matching must not raise the .15 m/s and .30 m/s² physical budgets")
    if params.physical_attitude_limit_deg > 7.5:
        raise ValueError("Matching must preserve the inspected attitude model limit")
    if params.reposition_speed_cap_m_s > min(.05, params.physical_speed_cap_m_s):
        raise ValueError("Reposition velocity must fit the .05 m/s cap and physical budget")
    if params.reposition_acceleration_cap_m_s2 > min(.05, params.physical_acceleration_cap_m_s2):
        raise ValueError("Reposition command ramp must fit the .05 m/s² cap and physical budget")
    if params.speed_tolerance_m_s > params.physical_speed_cap_m_s:
        raise ValueError("Accepted starting speed must fit the physical speed budget")
    if params.acceleration_tolerance_m_s2 > params.physical_acceleration_cap_m_s2:
        raise ValueError("Accepted starting acceleration must fit the physical acceleration budget")
    if params.attitude_tolerance_deg > params.physical_attitude_limit_deg:
        raise ValueError("Accepted attitude must fit the inherited model range")
    if params.command_dt_s > .1 or params.max_sample_gap_s < params.command_dt_s:
        raise ValueError("Command period must be <= .1 s and fit the accepted sample gap")
    if params.stable_s < .5 or params.timeout_s <= params.stable_s:
        raise ValueError("Require at least .5 s stability and a longer timeout")
    return params


def _vector(value, length, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"INVALID_{name}") from exc
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"INVALID_{name}")
    return result


def _raw_values(value):
    """Retain nonfinite input semantics without emitting JSON NaN literals."""
    try:
        values = np.asarray(value, dtype=float)
        return [float(x) if math.isfinite(float(x)) else str(float(x))
                for x in values.reshape(-1)]
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def _rpy_deg(quaternion):
    norm = float(np.linalg.norm(quaternion))
    if not .99 <= norm <= 1.01:
        raise ValueError("INVALID_ORIENTATION_QUATERNION_NORM")
    x, y, z, w = quaternion / norm
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1., min(1., 2 * (w*y - z*x))))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.degrees([roll, pitch, yaw])


def initial_state_check(position, velocity, acceleration, quaternion,
                        desired_position, config):
    """Return measured norm criteria; low acceptance failures only wait.

    Malformed/nonfinite self state and violations of inherited physical/model
    budgets require protection. No measured vector is clipped or repaired.
    Quaternion order is x, y, z, w; positions use the caller's common world NED.
    """
    params = validate_matched_start_config(config)
    inputs = (("POSITION", position, 3), ("VELOCITY", velocity, 3),
              ("ACCELERATION", acceleration, 3), ("ORIENTATION", quaternion, 4),
              ("DESIRED_POSITION", desired_position, 3))
    record = {
        "position_world_ned_m": _raw_values(position),
        "velocity_ned_m_s": _raw_values(velocity),
        "acceleration_ned_m_s2": _raw_values(acceleration),
        "orientation_xyzw": _raw_values(quaternion),
        "desired_position_world_ned_m": _raw_values(desired_position),
        "position_error_ned_m": None, "position_error_norm_m": None,
        "speed_norm_m_s": None, "acceleration_norm_m_s2": None,
        "body_rpy_deg": None, "max_abs_attitude_deg": None,
        "valid": False, "accepted": False, "reasons": [],
        "protection_required": False,
    }
    parsed = {}
    for name, value, length in inputs:
        try:
            parsed[name] = _vector(value, length, name)
        except ValueError as exc:
            record["reasons"].append(str(exc))
    if "ORIENTATION" in parsed:
        try:
            rpy = _rpy_deg(parsed["ORIENTATION"])
            record.update(body_rpy_deg=rpy.tolist(), max_abs_attitude_deg=float(np.max(np.abs(rpy))))
        except ValueError as exc:
            record["reasons"].append(str(exc))
    if record["reasons"]:
        record["protection_required"] = True
        return record
    error = parsed["POSITION"] - parsed["DESIRED_POSITION"]
    position_norm = float(np.linalg.norm(error))
    speed = float(np.linalg.norm(parsed["VELOCITY"]))
    acceleration_norm = float(np.linalg.norm(parsed["ACCELERATION"]))
    attitude = record["max_abs_attitude_deg"]
    record.update(valid=True, position_error_ned_m=error.tolist(),
                  position_error_norm_m=position_norm, speed_norm_m_s=speed,
                  acceleration_norm_m_s2=acceleration_norm)
    criteria = ((position_norm, params.position_tolerance_m, "POSITION_TOLERANCE_EXCEEDED"),
                (speed, params.speed_tolerance_m_s, "SPEED_TOLERANCE_EXCEEDED"),
                (acceleration_norm, params.acceleration_tolerance_m_s2, "ACCELERATION_TOLERANCE_EXCEEDED"),
                (attitude, params.attitude_tolerance_deg, "ATTITUDE_TOLERANCE_EXCEEDED"))
    record["reasons"].extend(reason for value, limit, reason in criteria if value > limit + 1e-12)
    protections = ((speed, params.physical_speed_cap_m_s, "ACTUAL_SPEED_BUDGET_EXCEEDED"),
                   (acceleration_norm, params.physical_acceleration_cap_m_s2, "ACTUAL_ACCELERATION_BUDGET_EXCEEDED"),
                   (attitude, params.physical_attitude_limit_deg, "ACTUAL_ATTITUDE_MODEL_LIMIT_EXCEEDED"))
    physical_reasons = [reason for value, limit, reason in protections if value > limit + 1e-9]
    record["reasons"].extend(physical_reasons)
    record["protection_required"] = bool(physical_reasons)
    record["accepted"] = not record["reasons"]
    return record


def reposition_xy_command(position, desired_position, previous_xy, elapsed_s, config):
    """Ramp a world-axis XY P command from the last *dispatched* command.

    Large elapsed intervals cannot increase the ramp allowance past command_dt_s.
    Actual physical acceleration is separately guarded by initial_state_check.
    """
    params = validate_matched_start_config(config)
    p = _vector(position, 3, "POSITION")
    desired = _vector(desired_position, 3, "DESIRED_POSITION")
    previous = _vector(previous_xy, 2, "PREVIOUS_XY_COMMAND")
    dt = _positive_number(elapsed_s, "elapsed_s")
    if np.linalg.norm(previous) > params.reposition_speed_cap_m_s + 1e-12:
        raise ValueError("PREVIOUS_XY_COMMAND_EXCEEDS_REPOSITION_CAP")
    xy_error = desired[:2] - p[:2]
    xy_error_norm = float(np.linalg.norm(xy_error))
    position_error_norm = float(np.linalg.norm(desired - p))
    deadband = position_error_norm <= params.position_tolerance_m + 1e-12
    raw = np.zeros(2) if deadband else params.proportional_gain * xy_error
    raw_norm = float(np.linalg.norm(raw))
    bounded = raw * min(1., params.reposition_speed_cap_m_s / max(raw_norm, 1e-12))
    change = bounded - previous
    change_norm = float(np.linalg.norm(change))
    ramp_dt = min(dt, params.command_dt_s)
    allowance = params.reposition_acceleration_cap_m_s2 * ramp_dt
    applied = previous + change * min(1., allowance / max(change_norm, 1e-12))
    return {"command_velocity_xy_m_s": applied.tolist(),
            "command_velocity_xy_norm_m_s": float(np.linalg.norm(applied)),
            "raw_velocity_xy_m_s": raw.tolist(),
            "xy_position_error_norm_m": xy_error_norm,
            "position_error_norm_m": position_error_norm,
            "position_deadband_active": bool(deadband),
            "previous_dispatched_velocity_xy_m_s": previous.tolist(),
            "command_delta_norm_m_s": float(np.linalg.norm(applied - previous)),
            "command_ramp_elapsed_s": dt, "command_ramp_effective_dt_s": ramp_dt,
            "speed_norm_limited": raw_norm > params.reposition_speed_cap_m_s,
            "command_delta_limited": change_norm > allowance,
            "command_budget_scope": "XY RPC velocity and command ramp only; physical v/a independently observed"}


class MatchedStartGate:
    """Continuous observed stability and timeout; never dispatches commands."""

    def __init__(self, config, start_time_s=None):
        self.params = validate_matched_start_config(config)
        if start_time_s is not None and not math.isfinite(float(start_time_s)):
            raise ValueError("start_time_s must be finite")
        self.start_time_s = None if start_time_s is None else float(start_time_s)
        self.previous_time_s = None
        self.stable_start_time_s = None

    def update(self, check, sample_time_s):
        now = float(sample_time_s)
        if not math.isfinite(now):
            raise ValueError("sample_time_s must be finite")
        if self.start_time_s is None:
            self.start_time_s = now
        gap = None if self.previous_time_s is None else now - self.previous_time_s
        backwards = now < self.start_time_s or (gap is not None and gap <= 0)
        elapsed = now - self.start_time_s
        reset_reason = None
        sample_ready = bool(check.get("valid") and check.get("accepted") and not check.get("protection_required"))
        if backwards:
            self.stable_start_time_s = None
            reset_reason = "NONMONOTONIC_SAMPLE_TIME"
        elif not sample_ready:
            self.stable_start_time_s = None
            reset_reason = "SAMPLE_DID_NOT_MEET_ACCEPTANCE"
        else:
            if gap is not None and gap > self.params.max_sample_gap_s:
                self.stable_start_time_s = None
                reset_reason = "SAMPLE_GAP_EXCEEDED"
            if self.stable_start_time_s is None:
                self.stable_start_time_s = now
        stable = 0. if self.stable_start_time_s is None else now - self.stable_start_time_s
        timed_out = elapsed >= self.params.timeout_s
        protection = bool(check.get("protection_required") or backwards)
        accepted = sample_ready and stable >= self.params.stable_s - 1e-12 and not timed_out and not protection
        status = ("PROTECTION_REQUIRED" if protection else "TIMEOUT" if timed_out
                  else "ACCEPTED" if accepted else "WAITING")
        self.previous_time_s = now
        return {"gate_accepted": bool(accepted), "gate_status": status,
                "continuous_stable_s": float(stable), "elapsed_s": float(elapsed),
                "sample_time_s": now, "sample_gap_s": gap,
                "stable_reset_reason": reset_reason, "timed_out": bool(timed_out),
                "protection_required": protection, "sample_meets_conditions": sample_ready}
