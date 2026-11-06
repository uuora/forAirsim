"""Geometric desired attitude + thrust for a SimpleFlight hybrid controller.

Reference: Lee, Leok & McClamroch, CDC 2010, equations (12), (15),
https://mleok.science/pdf/LeLeMc2010_quadrotor.pdf . Their equation (16)
moment controller is NOT implemented: SimpleFlight retains its attitude,
rate, mixer and actuator loops. The paper's stability results therefore do
not apply to this bounded, discrete-time hybrid implementation.

World: NED metres/seconds; body: FRD. R maps body to world; thrust is -f R e3.
Quaternion order is explicitly w,x,y,z. Output angles are physical FRD
Euler angles in radians (Rz(yaw) Ry(pitch) Rx(roll)); RPC sign conversion
belongs to the transport, not this module. Specific thrust is in m/s^2.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite and have shape {shape}")
    return arr


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def quaternion_to_matrix(q_wxyz: Any) -> np.ndarray:
    q = _array(q_wxyz, (4,), "quaternion_wxyz")
    scale = float(np.max(np.abs(q)))
    if scale < 1e-12:
        raise ValueError("quaternion norm is zero")
    q = q / scale
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    angles = _array([roll, pitch, yaw], (3,), "Euler angles")
    sr, sp, sy = np.sin(angles)
    cr, cp, cy = np.cos(angles)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr],
    ])


def matrix_to_euler(rotation: Any) -> tuple[float, float, float]:
    r = _array(rotation, (3, 3), "rotation")
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if math.hypot(r[0, 0], r[1, 0]) < 1e-10:
        # Deterministic representation at gimbal lock; flight cone avoids it.
        return 0.0, pitch, math.atan2(-r[0, 1], r[1, 1])
    return math.atan2(r[2, 1], r[2, 2]), pitch, math.atan2(r[1, 0], r[0, 0])


def so3_angle(actual: Any, desired: Any) -> float:
    """Principal relative rotation angle [0, pi], robust at pi and zero."""
    actual = _array(actual, (3, 3), "actual rotation")
    desired = _array(desired, (3, 3), "desired rotation")
    relative = desired.T @ actual
    sine = 0.5 * float(np.linalg.norm([
        relative[2, 1] - relative[1, 2],
        relative[0, 2] - relative[2, 0],
        relative[1, 0] - relative[0, 1],
    ]))
    cosine = float(np.clip((np.trace(relative) - 1) * 0.5, -1, 1))
    return math.atan2(sine, cosine)


def desired_rotation(b3: Any, yaw: float) -> tuple[np.ndarray, bool]:
    """Lee's heading-vector construction, including deterministic degeneracy."""
    b3 = _array(b3, (3,), "desired body z")
    norm = float(np.linalg.norm(b3))
    if norm < 1e-12 or not math.isfinite(float(yaw)):
        raise ValueError("desired body z must be nonzero and yaw finite")
    b3 = b3 / norm
    heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b2 = np.cross(b3, heading)
    singular = float(np.linalg.norm(b2)) < 1e-9
    if singular:
        # Choose the coordinate axis least parallel to b3, never divide by zero.
        heading = np.eye(3)[int(np.argmin(np.abs(b3)))]
        b2 = np.cross(b3, heading)
    b2 /= np.linalg.norm(b2)
    return np.column_stack((np.cross(b2, b3), b2, b3)), singular


def command(p: Any, v: Any, q_wxyz: Any, ref: dict, cfg: dict) -> dict:
    """Return a finite JSON-compatible physical attitude/collective command.

    Limits are declared engineering adaptations, not present in the cited
    theorem. Linear throttle scaling uses the local AirSim rotor model's
    control_signal -> speed_squared -> thrust relation; it does not mean
    measured motor thrust. No integral, drag, or motor-lag compensation.
    """
    p = _array(p, (3,), "position")
    v = _array(v, (3,), "velocity")
    r = quaternion_to_matrix(q_wxyz)
    p_ref = _array(ref["p"], (3,), "reference position")
    v_ref = _array(ref["v"], (3,), "reference velocity")
    a_ref = _array(ref["a"], (3,), "reference acceleration")
    kp = _array(cfg.get("kp", [2, 2, 3]), (3,), "kp")
    kv = _array(cfg.get("kv", [2.2, 2.2, 2.5]), (3,), "kv")
    if np.any(kp <= 0) or np.any(kv <= 0):
        raise ValueError("controller gains must be positive")
    gravity = _positive(cfg.get("gravity", 9.80665), "gravity")
    hover = _positive(cfg.get("hover_throttle", .4), "hover_throttle")
    max_acc = _positive(cfg.get("max_acceleration", 5), "max_acceleration")
    max_tilt = _positive(cfg.get("max_tilt_rad", math.radians(25)), "max_tilt_rad")
    minimum = float(cfg.get("min_throttle", .08))
    maximum = float(cfg.get("max_throttle", .8))
    if not (0 <= minimum < hover < maximum <= 1) or not (max_tilt < math.pi / 2):
        raise ValueError("require 0 <= min < hover < max <= 1 and 0 < tilt < pi/2")
    ep, ev = p - p_ref, v - v_ref
    with np.errstate(over="ignore", invalid="ignore"):
        requested_acc = a_ref - kp * ep - kv * ev
    if not np.all(np.isfinite(requested_acc)):
        raise ValueError("feedback acceleration overflow")
    acc = requested_acc.copy()
    sat = {"acceleration": False, "vertical_force": False, "tilt": False,
           "force_magnitude": False, "collective": False,
           "nonpositive_projection": False, "heading_singularity": False}
    norm = float(np.linalg.norm(acc))
    if norm > max_acc:
        acc *= max_acc / norm
        sat["acceleration"] = True
    # Eq12 after dividing the force vector by mass. Floor ensures upright
    # feasibility before the tilt-cone operation, even for excessive down-accel.
    force = np.array([0., 0., gravity]) - acc
    vertical_floor = _positive(cfg.get("min_vertical_specific_force", .1 * gravity),
                               "min_vertical_specific_force")
    if force[2] < vertical_floor:
        force[2] = vertical_floor
        sat["vertical_force"] = True
    lateral_norm = float(np.linalg.norm(force[:2]))
    lateral_max = force[2] * math.tan(max_tilt)
    if lateral_norm > lateral_max:
        force[:2] *= lateral_max / lateral_norm
        sat["tilt"] = True
    force_norm = float(np.linalg.norm(force))
    force_max = maximum / hover * gravity
    if force_norm > force_max:
        force *= force_max / force_norm
        force_norm = force_max
        sat["force_magnitude"] = True
    rd, sat["heading_singularity"] = desired_rotation(force / force_norm, float(ref.get("yaw", 0)))
    roll, pitch, yaw = matrix_to_euler(rd)
    # Eq15 projected on measured body-down, not desired body-down. Negative
    # values cannot be achieved by this quadrotor and are explicitly flagged.
    projection = float(np.dot(force, r[:, 2]))
    sat["nonpositive_projection"] = projection <= 0
    requested_throttle = hover * projection / gravity
    throttle = float(np.clip(requested_throttle, minimum, maximum))
    sat["collective"] = abs(throttle - requested_throttle) > 1e-12
    return {
        "roll": roll, "pitch": pitch, "yaw": yaw, "throttle": throttle,
        "R_des": rd.tolist(), "R_actual": r.tolist(),
        "a_requested": requested_acc.tolist(),
        "a_command": (np.array([0., 0., gravity]) - force).tolist(),
        "force_specific": force.tolist(), "thrust_specific_projection": projection,
        "throttle_unsaturated": requested_throttle, "saturation": sat,
        "attitude_error_rad": so3_angle(r, rd),
        "tilt_rad": math.acos(float(np.clip(r[2, 2], -1, 1))),
        "desired_tilt_rad": math.acos(float(np.clip(rd[2, 2], -1, 1))),
        "position_error": ep.tolist(), "velocity_error": ev.tolist(),
    }
