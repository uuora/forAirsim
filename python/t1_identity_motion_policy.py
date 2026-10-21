"""Pure command gate for the developmental T1 low-speed XY pilot.

The caller supplies the command actually issued and a newly read own state.
This class neither infers that a returned command was executed nor accesses a
simulator, detector, API, or target truth. Accepted geometry is only one input
to this gate. The declared delay/braking envelope is not a continuous physical
safety certificate or a calibrated instance-identity claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import time
from typing import Any, Callable

from t1_semantic_perception import SemanticFrame
from trajectory_execution_guard import boundary_command, command_window


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"INVALID_{name}")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"INVALID_{name}")
    return value


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"INVALID_{name}")
    if any(isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item)
           for item in value):
        raise ValueError(f"INVALID_{name}")
    return tuple(float(item) for item in value)


def _norm(value: tuple[float, ...]) -> float:
    return math.sqrt(sum(item*item for item in value))


@dataclass(frozen=True)
class MotionPolicyConfig:
    current_frame_max_age_s: float = 0.5
    command_lease_s: float = 0.12
    minimum_command_lease_s: float = 0.01
    freshness_reserve_s: float = 0.02
    speed_cap_m_s: float = 0.03
    command_acceleration_cap_m_s2: float = 0.15
    max_previous_command_interval_s: float = 0.4
    self_state_max_age_s: float = 0.12
    stationary_velocity_threshold_m_s: float = 0.005
    measured_xy_speed_ceiling_m_s: float = 0.06
    boundary_horizon_s: float = 1.0
    braking_acceleration_model_m_s2: float = 0.15
    position_reserve_m: float = 0.02
    workspace_low_ned_m: tuple[float, float, float] = (-1.0, -1.0, -3.2)
    workspace_high_ned_m: tuple[float, float, float] = (1.0, 1.0, -2.8)

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if key not in ("workspace_low_ned_m", "workspace_high_ned_m"):
                _number(value, key.upper(), positive=True)
        low = _vector(self.workspace_low_ned_m, 3, "WORKSPACE_LOW")
        high = _vector(self.workspace_high_ned_m, 3, "WORKSPACE_HIGH")
        if any(left >= right for left, right in zip(low, high)):
            raise ValueError("INVALID_WORKSPACE_BOUNDS")
        if (self.current_frame_max_age_s > 0.5 or self.command_lease_s > 0.12 or
                self.minimum_command_lease_s > self.command_lease_s or
                self.freshness_reserve_s < 0.02 or
                self.freshness_reserve_s >= self.current_frame_max_age_s or
                self.speed_cap_m_s > 0.06 or
                self.command_acceleration_cap_m_s2 > 0.15 or
                self.max_previous_command_interval_s > 0.4 or
                self.self_state_max_age_s > 0.12 or
                self.measured_xy_speed_ceiling_m_s > 0.06 or
                self.measured_xy_speed_ceiling_m_s < self.speed_cap_m_s or
                self.stationary_velocity_threshold_m_s > self.speed_cap_m_s or
                self.boundary_horizon_s < 1.0 or
                self.braking_acceleration_model_m_s2 > 0.15 or self.position_reserve_m < 0.02):
            raise ValueError("CONFIG_EXCEEDS_DEVELOPMENTAL_XY_ENVELOPE")


class IdentityMotionPolicy:
    """Validate current temporal selection before returning a bounded XY command.

    ``previous_command_monotonic_s`` is the actual dispatch timestamp, never a
    nominal loop period. ``self_state_receipt_monotonic_s`` is the receipt of
    the freshly read vehicle position and velocity. All times use one injected
    host monotonic clock. RGB times are readback receipts, not exposure times.
    A target frame can authorize only one command; repeating it cannot renew a
    lease. The final dispatcher must also check the returned end deadline.
    """

    def __init__(self, session_id: str, target_generation: int, description: str,
                 config: MotionPolicyConfig | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("INVALID_SESSION_ID")
        if isinstance(target_generation, bool) or not isinstance(target_generation, int) or target_generation < 0:
            raise ValueError("INVALID_TARGET_GENERATION")
        if not isinstance(description, str) or not description.strip():
            raise ValueError("INVALID_DESCRIPTION")
        if config is not None and not isinstance(config, MotionPolicyConfig):
            raise ValueError("INVALID_MOTION_POLICY_CONFIG")
        if not callable(clock):
            raise ValueError("INVALID_CLOCK")
        self.config = config or MotionPolicyConfig()
        self.session_id, self.target_generation = session_id, target_generation
        self.description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
        self.clock = clock
        self._last_now: float | None = None
        self._last_frame_sequence: int | None = None
        self._last_frame_receipt: float | None = None
        self._seen_frame_ids: dict[str, str] = {}
        self._used_target_frames: set[str] = set()
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def _result(self, status: str, reason: str, now: float | None, **values) -> dict:
        result = {"schema_version": 1, "status": status, "reason": reason,
            "decision_monotonic_s": now, "session_id": self.session_id,
            "target_generation": self.target_generation,
            "description_sha256": self.description_sha256,
            "command_velocity_xy_m_s": None, "command_duration_s": 0.0,
            "target_command_authorized": False, "self_state_brake_authorized": False,
            "current_binding": None, "identity_probability_calibrated": False,
            "instance_identity_guaranteed": False, "flight_authorized": False,
            "execution_guard_checked": False, "continuous_safety_established": False,
            "configured_envelope": asdict(self.config),
            "time_scope": "host_monotonic_readback_receipts_not_exposure_times",
            "scope": "developmental_fixed_height_low_speed_XY_command_gate_only"}
        result.update(values)
        return result

    def _binding(self, frame: SemanticFrame) -> dict:
        return {"session_id": frame.session_id, "target_generation": frame.target_generation,
            "description_sha256": self.description_sha256, "frame_id": frame.frame_id,
            "frame_sequence": frame.frame_sequence, "image_sha256": frame.image_sha256,
            "readback_receipt_monotonic_s": frame.readback_receipt_monotonic_s,
            "candidates_ready_monotonic_s": frame.candidates_ready_monotonic_s}

    def _frame_problem(self, frame: Any, now: float) -> tuple[str | None, bool]:
        if not isinstance(frame, SemanticFrame):
            return "MISSING_CURRENT_FRAME", False
        if (frame.session_id, frame.target_generation) != (self.session_id, self.target_generation):
            return "FRAME_CONTEXT_MISMATCH", True
        if frame.candidates_ready_monotonic_s > now:
            return "FRAME_READY_IN_FUTURE", True
        fingerprint = hashlib.sha256(json.dumps(frame.to_dict(), sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        old = self._seen_frame_ids.get(frame.frame_id)
        if old is not None and old != fingerprint:
            return "FRAME_ID_REUSED_WITH_DIFFERENT_BINDING", True
        if self._last_frame_sequence is not None:
            if frame.frame_sequence < self._last_frame_sequence:
                return "FRAME_SEQUENCE_REGRESSION", True
            if frame.frame_sequence == self._last_frame_sequence and old is None:
                return "FRAME_SEQUENCE_REUSED", True
            if (frame.frame_sequence > self._last_frame_sequence and
                    frame.readback_receipt_monotonic_s <= self._last_frame_receipt):
                return "FRAME_RECEIPT_NOT_ACTUALLY_NEWER", True
        self._seen_frame_ids[frame.frame_id] = fingerprint
        self._last_frame_sequence = frame.frame_sequence
        self._last_frame_receipt = frame.readback_receipt_monotonic_s
        if now-frame.readback_receipt_monotonic_s > self.config.current_frame_max_age_s:
            return "CURRENT_FRAME_EXPIRED", False
        if frame.frame_id in self._used_target_frames:
            return "TARGET_FRAME_ALREADY_USED_FOR_COMMAND", True
        return None, False

    def _selection_problem(self, frame: SemanticFrame, selection: Any, surface: Any,
                           now: float) -> tuple[str | None, bool]:
        if not isinstance(selection, dict) or selection.get("status") == "NO_TARGET_OUTPUT":
            return "NO_CURRENT_DESIGNATED_TARGET", False
        if selection.get("status") != "TEMPORAL_LOCAL_ASSOCIATION_VALID" or selection.get("state") != "TRACKING":
            return "SELECTION_NOT_TEMPORAL_TRACKING", True
        expected = {"session_id": self.session_id, "target_generation": self.target_generation,
                    "description_sha256": self.description_sha256,
                    "current_frame_id": frame.frame_id, "current_frame_sequence": frame.frame_sequence,
                    "current_image_sha256": frame.image_sha256}
        if any(selection.get(key) != value or isinstance(selection.get(key), bool)
               for key, value in expected.items()):
            return "SELECTION_CURRENT_CONTEXT_MISMATCH", True
        try:
            binding_json = json.dumps(selection.get("current_binding"), sort_keys=True,
                                      separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            return "SELECTION_RECEIPT_BINDING_MISMATCH", True
        if (not isinstance(selection.get("current_binding"), dict) or
                binding_json != json.dumps(self._binding(frame), sort_keys=True, separators=(",", ":"))):
            return "SELECTION_RECEIPT_BINDING_MISMATCH", True
        if any(selection.get(key) is not False for key in
               ("identity_probability_calibrated", "instance_identity_guaranteed", "appearance_continuity_verified",
                "flight_authorized")):
            return "SELECTION_IDENTITY_CLAIM_EXCEEDS_EVIDENCE", True
        if selection.get("semantic_binding_provenance") != "CLIENT_SIDE_BINDING; NOT SERVER_ECHO":
            return "SELECTION_PROVENANCE_MISMATCH", True
        if not all(isinstance(selection.get(key), str) and selection[key].strip()
                   for key in ("track_id", "semantic_anchor")):
            return "SELECTION_ANCHOR_OR_TRACK_MISSING", True
        try:
            deadline = _number(selection.get("semantic_review_deadline_monotonic_s"), "SEMANTIC_REVIEW_DEADLINE")
        except ValueError:
            return "SELECTION_REVIEW_DEADLINE_INVALID", True
        if deadline <= now:
            return "SEMANTIC_REVIEW_DEADLINE_EXPIRED", False
        matched = [item for item in frame.candidates if item.candidate_id == selection.get("selected_candidate_id")]
        if len(matched) != 1 or selection.get("selected_candidate") != matched[0].to_dict():
            return "SELECTION_CANDIDATE_CONTENT_MISMATCH", True
        candidate = matched[0]
        if not isinstance(surface, dict) or surface.get("status") != "SURFACE_GEOMETRY_ACCEPTED":
            return "NO_ACCEPTED_SELECTED_SURFACE", False
        expected_surface = {"session_id": self.session_id, "target_generation": self.target_generation,
            "frame_id": frame.frame_id, "frame_sequence": frame.frame_sequence,
            "image_sha256": frame.image_sha256, "selected_candidate_id": candidate.candidate_id,
            "selected_bbox_xyxy_pixels": list(candidate.bbox_xyxy_pixels),
            "selection_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID"}
        if any(surface.get(key) != value or isinstance(surface.get(key), bool)
               for key, value in expected_surface.items()):
            return "SURFACE_CURRENT_CONTEXT_MISMATCH", True
        if (surface.get("freshness_checked") is not True or
                surface.get("identity_probability_calibrated") is not False or
                surface.get("instance_identity_guaranteed") is not False):
            return "SURFACE_FRESHNESS_OR_IDENTITY_SCOPE_MISMATCH", True
        expected_v2 = {"temporal_track_id": selection["track_id"],
            "semantic_anchor": selection["semantic_anchor"],
            "description_sha256": self.description_sha256, "current_binding": self._binding(frame)}
        if any(surface.get(key) != value for key, value in expected_v2.items()):
            return "SURFACE_TEMPORAL_CERTIFICATE_BINDING_MISMATCH", True
        try:
            selection_sha = hashlib.sha256(json.dumps(selection, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            return "SELECTION_NOT_FINITE_JSON_CERTIFICATE", True
        if surface.get("selection_certificate_sha256") != selection_sha:
            return "SURFACE_SELECTION_CERTIFICATE_SHA256_MISMATCH", True
        # These are geometry's honest scope declarations, not reasons to reject.
        # The motion envelope is checked separately below.
        if surface.get("depth_kind") != "DepthPerspective":
            return "SURFACE_DEPTH_KIND_MISMATCH", True
        try:
            _vector(surface.get("selected_bbox_xyxy_pixels"), 4, "SELECTED_SURFACE_BBOX")
            _vector(surface.get("surface_position_world_ned_m"), 3, "SURFACE_WORLD_POSITION")
            _vector(surface.get("surface_position_local_ned_m"), 3, "SURFACE_LOCAL_POSITION")
            _vector(surface.get("rgb_reference_world_ned_m"), 3, "RGB_REFERENCE_WORLD_POSITION")
            _number(surface.get("ray_range_m"), "SURFACE_RAY_RANGE", positive=True)
        except ValueError:
            return "SURFACE_GEOMETRY_VALUE_INVALID", True
        return None, False

    def _envelope(self, position, desired, previous, elapsed) -> dict:
        cfg = self.config
        # boundary_command requires positive dt. A zero elapsed interval is
        # evaluated with an infinitesimal dt, then forced to the exact previous
        # command and checked against the reported bounds; no positive change.
        envelope = boundary_command(position, desired, previous,
            cfg.workspace_low_ned_m, cfg.workspace_high_ned_m, cfg.speed_cap_m_s,
            cfg.command_acceleration_cap_m_s2, max(elapsed, 1e-12), cfg.boundary_horizon_s,
            cfg.position_reserve_m)
        if elapsed == 0:
            feasible = all(low-1e-9 <= value <= high+1e-9 for value, low, high in zip(
                previous, envelope["lower_velocity_xy_m_s"], envelope["upper_velocity_xy_m_s"]))
            envelope.update(accepted=feasible,
                command_velocity_xy_m_s=list(previous) if feasible else None,
                command_change_norm_m_s=0.0)
        envelope["actual_elapsed_previous_command_s"] = elapsed
        envelope["command_acceleration_cap_m_s2"] = cfg.command_acceleration_cap_m_s2
        # Existing helper uses ramp acceleration also in its braking model.
        # A smaller declared braking acceleration requires the stricter helper.
        if cfg.braking_acceleration_model_m_s2 < cfg.command_acceleration_cap_m_s2:
            strict = boundary_command(position, desired, previous, cfg.workspace_low_ned_m,
                cfg.workspace_high_ned_m, cfg.speed_cap_m_s, cfg.braking_acceleration_model_m_s2,
                max(elapsed, 1e-12), cfg.boundary_horizon_s, cfg.position_reserve_m)
            envelope = strict
            if elapsed == 0:
                feasible = all(low-1e-9 <= value <= high+1e-9 for value, low, high in zip(
                    previous, strict["lower_velocity_xy_m_s"], strict["upper_velocity_xy_m_s"]))
                envelope.update(accepted=feasible, command_velocity_xy_m_s=list(previous) if feasible else None,
                                command_change_norm_m_s=0.0)
            envelope["actual_elapsed_previous_command_s"] = elapsed
            envelope["command_acceleration_cap_m_s2"] = cfg.command_acceleration_cap_m_s2
        return envelope

    def _withdraw(self, reason, invalid, now, previous, actual, position, elapsed, binding=None, **audit):
        moving = _norm(previous) > 1e-12 or _norm(actual) > self.config.stationary_velocity_threshold_m_s
        if not moving:
            return self._result("REJECT" if invalid else "WAIT_NO_TARGET", reason, now,
                current_binding=binding, elapsed_previous_command_s=elapsed,
                command_velocity_xy_m_s=[0.0, 0.0], **audit)
        braking_dt = min(elapsed, self.config.max_previous_command_interval_s)
        envelope = self._envelope(position, (0.0, 0.0), previous, braking_dt)
        if not envelope["accepted"]:
            return self._result("REJECT", "NO_FEASIBLE_SELF_STATE_BRAKE", now,
                triggering_reason=reason, current_binding=binding,
                elapsed_previous_command_s=elapsed, boundary_envelope=envelope, **audit)
        return self._result("REQUIRE_SELF_STATE_BRAKE", reason, now,
            current_binding=binding, elapsed_previous_command_s=elapsed,
            self_state_brake_authorized=True, execution_guard_checked=True,
            command_velocity_xy_m_s=envelope["command_velocity_xy_m_s"],
            command_duration_s=self.config.command_lease_s,
            latest_dispatch_monotonic_s=audit["self_state_receipt_monotonic_s"]+self.config.self_state_max_age_s,
            dispatch_rule="dispatch_monotonic_s <= latest_dispatch_monotonic_s; remeasure own state if late",
            boundary_envelope=envelope, target_observation_used_for_brake=False,
            brake_interval_capped_for_long_gap=elapsed > braking_dt, **audit)

    def decide(self, *, frame: SemanticFrame | None, selection: dict | None, surface: dict | None,
               desired_xy_m_s, actual_velocity_xy_m_s, self_position_ned_m,
               previous_command_xy_m_s, previous_command_monotonic_s,
               self_state_receipt_monotonic_s, now: float | None = None) -> dict:
        try:
            instant = _number(self.clock() if now is None else now, "DECISION_TIME")
        except (ValueError, TypeError):
            return self._result("REJECT", "INVALID_DECISION_TIME", None)
        if self._closed:
            return self._result("REJECT", "MOTION_POLICY_CLOSED", instant)
        if self._last_now is not None and instant < self._last_now:
            return self._result("REJECT", "DECISION_CLOCK_REGRESSION", instant)
        self._last_now = instant
        try:
            previous_time = _number(previous_command_monotonic_s, "PREVIOUS_COMMAND_TIME")
            self_receipt = _number(self_state_receipt_monotonic_s, "SELF_STATE_RECEIPT_TIME")
            desired = _vector(desired_xy_m_s, 2, "DESIRED_XY")
            previous = _vector(previous_command_xy_m_s, 2, "PREVIOUS_COMMAND_XY")
            actual = _vector(actual_velocity_xy_m_s, 2, "ACTUAL_VELOCITY_XY")
            position = _vector(self_position_ned_m, 3, "SELF_POSITION_NED")
        except ValueError as error:
            return self._result("REJECT", str(error), instant)
        if not previous_time <= self_receipt <= instant:
            return self._result("REJECT", "SELF_OR_COMMAND_CLOCK_ORDER_INVALID", instant)
        elapsed = instant-previous_time
        self_age = instant-self_receipt
        if self_age > self.config.self_state_max_age_s:
            return self._result("REJECT", "SELF_STATE_EXPIRED_REMEASURE_BEFORE_BRAKE", instant,
                elapsed_previous_command_s=elapsed, self_state_age_proxy_s=self_age)
        if _norm(previous) > self.config.speed_cap_m_s+1e-9:
            return self._result("REJECT", "PREVIOUS_COMMAND_OUTSIDE_SPEED_CAP", instant)
        if _norm(actual) > self.config.measured_xy_speed_ceiling_m_s+1e-9:
            return self._result("REJECT", "MEASURED_XY_SPEED_OUTSIDE_VALIDATED_CEILING", instant)
        if any(value < low or value > high for value, low, high in zip(
                position, self.config.workspace_low_ned_m, self.config.workspace_high_ned_m)):
            return self._result("REJECT", "SELF_POSITION_OUTSIDE_PROTECTED_WORKSPACE", instant)
        try:
            problem, invalid = self._frame_problem(frame, instant)
        except (TypeError, ValueError):
            problem, invalid = "INVALID_FRAME_JSON_BINDING", True
        binding = self._binding(frame) if isinstance(frame, SemanticFrame) else None
        audit = {"self_state_age_proxy_s": self_age,
                 "self_state_receipt_monotonic_s": self_receipt,
                 "actual_velocity_xy_m_s": list(actual), "self_position_ned_m": list(position),
                 "previous_command_xy_m_s": list(previous),
                 "previous_command_monotonic_s": previous_time}
        if problem is None:
            problem, invalid = self._selection_problem(frame, selection, surface, instant)
        if problem is not None:
            return self._withdraw(problem, invalid, instant, previous, actual, position, elapsed,
                                  binding=binding, **audit)
        if elapsed > self.config.max_previous_command_interval_s:
            return self._withdraw("PREVIOUS_COMMAND_INTERVAL_EXCEEDED", False, instant,
                previous, actual, position, elapsed, binding=binding, **audit)
        window = command_window(instant-frame.readback_receipt_monotonic_s,
            self.config.current_frame_max_age_s, self.config.command_lease_s,
            self.config.minimum_command_lease_s, self.config.freshness_reserve_s)
        review_available = selection["semantic_review_deadline_monotonic_s"]-instant-self.config.freshness_reserve_s
        duration = min(window["duration_s"], review_available)
        if not window["accepted"] or duration < self.config.minimum_command_lease_s:
            return self._withdraw("REOBSERVE_BEFORE_LEASE_EXPIRY", False, instant,
                previous, actual, position, elapsed, binding=binding,
                freshness_window=window, semantic_review_available_s=review_available, **audit)
        if elapsed == 0 and _norm(previous) == 0 and _norm(desired) > 0:
            return self._result("WAIT_NO_TARGET", "INITIAL_ZERO_ELAPSED_PREVENTS_POSITIVE_CHANGE", instant,
                current_binding=binding, command_velocity_xy_m_s=[0.0, 0.0],
                elapsed_previous_command_s=elapsed, freshness_window=window, **audit)
        envelope = self._envelope(position, desired, previous, elapsed)
        if not envelope["accepted"]:
            return self._withdraw("NO_TARGET_BOUNDARY_AND_RAMP_FEASIBLE_COMMAND", True, instant,
                previous, actual, position, elapsed, binding=binding, **audit)
        self._used_target_frames.add(frame.frame_id)
        deadline = min(frame.readback_receipt_monotonic_s+self.config.current_frame_max_age_s,
                       selection["semantic_review_deadline_monotonic_s"])-self.config.freshness_reserve_s
        return self._result("ALLOW_BOUNDED_COMMAND", "FRESH_BOUND_CURRENT_TARGET_AND_XY_ENVELOPE", instant,
            current_binding=binding, target_command_authorized=True, execution_guard_checked=True,
            selected_candidate_id=selection["selected_candidate_id"], temporal_track_id=selection["track_id"],
            semantic_anchor=selection["semantic_anchor"], command_velocity_xy_m_s=envelope["command_velocity_xy_m_s"],
            command_duration_s=duration, command_end_deadline_monotonic_s=deadline,
            latest_dispatch_monotonic_s=min(deadline-duration,
                self_receipt+self.config.self_state_max_age_s),
            dispatch_rule="dispatch_monotonic_s <= latest_dispatch_monotonic_s and dispatch_monotonic_s + duration_s <= command_end_deadline_monotonic_s",
            elapsed_previous_command_s=elapsed, freshness_window=window,
            semantic_review_available_s=review_available, boundary_envelope=envelope, **audit)
