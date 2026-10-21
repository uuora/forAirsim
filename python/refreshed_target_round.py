"""Replayable round protocol for two drones and one refreshed target actor.

This module has no simulator or perception dependencies. A caller supplies scene
inventory, cancellation/hold attestations, time-stamped perception estimates,
and collision evidence. Actions are requests for that caller to execute; accepting
an attestation is not proof that a simulator or aircraft is safe. A hit means an
accepted simulator contact event, not proof of a balloon bursting.

All ``now`` / ``*_at`` values use one caller-owned monotonic seconds clock.
Collision timestamps use the caller's separate integer collision clock and are
compared only with the dispatch baseline and earlier accepted collisions.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


class RoundPhase(str, Enum):
    WAIT_SCENE = "WAIT_SCENE"
    WAIT_HOLD = "WAIT_HOLD"
    WAIT_OBSERVATIONS = "WAIT_OBSERVATIONS"
    READY = "READY"
    ASSIGNED = "ASSIGNED"
    EXECUTING = "EXECUTING"
    CONTACT = "CONTACT"
    CANCEL_PENDING = "CANCEL_PENDING"
    REFRESH_PENDING = "REFRESH_PENDING"


@dataclass(frozen=True)
class RoundThresholds:
    max_observation_age_s: float = 0.5
    max_eta_age_s: float = 0.5
    max_message_age_s: float = 1.0
    max_assignment_age_s: float = 1.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not _finite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    phase: str
    actions: Tuple[Dict[str, Any], ...] = ()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _name(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _pack(value: Any) -> Any:
    """Keep even rejected nonfinite inputs JSON-safe and reversible."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite": repr(value)}
    if isinstance(value, Mapping):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(item) for item in value]
    return value


def _unpack(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"$nonfinite"}:
        return float(value["$nonfinite"])
    if isinstance(value, dict):
        return {key: _unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack(item) for item in value]
    return value


class RefreshedTargetRound:
    """Deterministic event admission and allocation; never executes actions.

    Call ``submit(event, payload)`` with the fields in ``EVENT_FIELDS``. Every
    event, including a rejection, appends its input, decision and resulting state
    to ``event_log``. ``replay(event_log)`` checks the recorded decisions/states.
    The three allocation policies are ``fixed_A``, ``alternating`` and
    ``nearest_eta``. ETA must be estimated from observations by the caller;
    target/drone coordinates are deliberately absent from this protocol.
    """

    DRONES = ("DroneA", "DroneB")
    EVENT_FIELDS = {
        "prepare_round": {"round_id", "target_generation", "actor_name", "active_target_actors", "prepared_at", "now"},
        "acknowledge_hold": {"round_id", "target_generation", "drone", "request_token", "recorded_at", "old_actions_cancelled", "safe_hold", "ready", "now"},
        "observe": {"round_id", "target_generation", "drone", "actor_name", "message_id", "source", "captured_at", "identity_confirmed", "ready", "eta_seconds", "estimate_at", "now"},
        "assign": {"round_id", "target_generation", "policy", "assignment_id", "plan_id", "created_at", "now"},
        "dispatch": {"round_id", "target_generation", "executor", "assignment_id", "plan_id", "issued_at", "collision_baseline_timestamp", "now"},
        "mark_contact": {"round_id", "target_generation", "executor", "assignment_id", "plan_id", "issued_at", "now"},
        "confirm_hit": {"round_id", "target_generation", "executor", "assignment_id", "plan_id", "actor_name", "has_collided", "collision_timestamp", "reported_at", "now"},
        "acknowledge_cancel": {"round_id", "target_generation", "drone", "request_token", "recorded_at", "old_actions_cancelled", "safe_hold", "ready", "now"},
    }

    def __init__(self, thresholds: Optional[RoundThresholds] = None):
        self.thresholds = thresholds or RoundThresholds()
        self.phase = RoundPhase.WAIT_SCENE
        self.round_id = 0
        self.target_generation = 0
        self.actor_name = None
        self.scene_prepared_at = None
        self.hold_token = None
        self.cancel_token = None
        self.hold_acks: Dict[str, float] = {}
        self.cancel_acks: Dict[str, float] = {}
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.assignment = None
        self.dispatch_at = None
        self.hit_at = None
        self.collision_baseline_timestamp = 0
        self.last_collision_timestamps = dict.fromkeys(self.DRONES, 0)
        self.accepted_hit_rounds = set()
        self.used_message_ids = set()
        self.used_assignment_ids = set()
        self.used_plan_ids = set()
        self.last_now = None
        self._event_log = [{"event": "initialize", "protocol_version": 1,
                            "payload": {"thresholds": asdict(self.thresholds)}, "state": self.snapshot()}]

    @staticmethod
    def actor_for_round(round_id: int) -> str:
        return f"BalloonTarget_r{round_id:04d}"

    @property
    def event_log(self) -> List[Dict[str, Any]]:
        return deepcopy(self._event_log)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "round_id": self.round_id,
            "target_generation": self.target_generation,
            "actor_name": self.actor_name,
            "scene_prepared_at": self.scene_prepared_at,
            "hold_token": self.hold_token,
            "cancel_token": self.cancel_token,
            "hold_acks": deepcopy(self.hold_acks),
            "cancel_acks": deepcopy(self.cancel_acks),
            "observations": deepcopy(self.observations),
            "assignment": deepcopy(self.assignment),
            "dispatch_at": self.dispatch_at,
            "hit_at": self.hit_at,
            "collision_baseline_timestamp": self.collision_baseline_timestamp,
            "last_collision_timestamps": deepcopy(self.last_collision_timestamps),
            "accepted_hit_rounds": sorted(self.accepted_hit_rounds),
            "used_message_ids": sorted(self.used_message_ids),
            "used_assignment_ids": sorted(self.used_assignment_ids),
            "used_plan_ids": sorted(self.used_plan_ids),
            "last_now": self.last_now,
        }

    def _finish(self, event, payload, accepted, reason, actions=()):
        if accepted:
            self.last_now = payload["now"]
        decision = Decision(accepted, reason, self.phase.value, tuple(deepcopy(actions)))
        self._event_log.append({
            "event": event,
            "payload": _pack(deepcopy(payload)),
            "decision": _pack(asdict(decision)),
            "state": _pack(self.snapshot()),
        })
        return decision

    def submit(self, event: str, payload: Mapping[str, Any]) -> Decision:
        """Admit a closed-schema event. Missing/extra fields are rejected."""
        if not isinstance(payload, Mapping):
            return self._finish(event, payload, False, "invalid_payload")
        payload = dict(payload)
        if not isinstance(event, str) or event not in self.EVENT_FIELDS:
            return self._finish(event, payload, False, "unknown_event")
        if set(payload) != self.EVENT_FIELDS[event]:
            return self._finish(event, payload, False, "event_fields_mismatch")
        now = payload["now"]
        if not _finite(now):
            return self._finish(event, payload, False, "invalid_now")
        if self.last_now is not None and now < self.last_now:
            return self._finish(event, payload, False, "clock_regression")
        if not _positive_int(payload["round_id"]) or not _positive_int(payload["target_generation"]):
            return self._finish(event, payload, False, "invalid_version")
        if event != "prepare_round":
            if payload["round_id"] != self.round_id:
                return self._finish(event, payload, False, "round_id_mismatch")
            if payload["target_generation"] != self.target_generation:
                return self._finish(event, payload, False, "target_generation_mismatch")
        reason, actions = getattr(self, "_on_" + event)(payload)
        return self._finish(event, payload, reason == "accepted", reason, actions)

    def _age_error(self, timestamp, now, max_age, earliest=None):
        if not _finite(timestamp):
            return "invalid_timestamp"
        if timestamp > now:
            return "future_timestamp"
        if earliest is not None and timestamp < earliest:
            return "timestamp_before_barrier"
        if now - timestamp > max_age:
            return "stale_message"
        return None

    def _on_prepare_round(self, p):
        if self.phase not in (RoundPhase.WAIT_SCENE, RoundPhase.REFRESH_PENDING):
            return "cancellation_barrier_not_complete", ()
        if p["round_id"] != self.round_id + 1 or p["target_generation"] != self.target_generation + 1:
            return "next_version_required", ()
        expected = self.actor_for_round(p["round_id"])
        if p["actor_name"] != expected:
            return "actor_name_mismatch", ()
        actors = p["active_target_actors"]
        if not isinstance(actors, (list, tuple)) or list(actors) != [expected]:
            return "single_active_target_required", ()
        error = self._age_error(p["prepared_at"], p["now"], self.thresholds.max_message_age_s, self.last_now)
        if error:
            return error, ()
        self.round_id = p["round_id"]
        self.target_generation = p["target_generation"]
        self.actor_name = expected
        self.scene_prepared_at = p["prepared_at"]
        self.hold_token = f"hold_r{self.round_id:04d}_g{self.target_generation:04d}"
        self.cancel_token = None
        self.hold_acks.clear()
        self.cancel_acks.clear()
        self.observations.clear()
        self.assignment = None
        self.dispatch_at = None
        self.hit_at = None
        self.collision_baseline_timestamp = 0
        self.phase = RoundPhase.WAIT_HOLD
        return "accepted", self._hold_actions(self.hold_token)

    def _hold_actions(self, token):
        return tuple({"operation": "CANCEL_AND_HOLD", "drone": drone, "round_id": self.round_id,
                      "target_generation": self.target_generation, "request_token": token}
                     for drone in self.DRONES)

    def _ack_error(self, p, token, earliest, acks):
        if p["drone"] not in self.DRONES:
            return "unknown_drone"
        if p["request_token"] != token:
            return "request_token_mismatch"
        if p["drone"] in acks:
            return "duplicate_acknowledgement"
        if not all(p[key] is True for key in ("old_actions_cancelled", "safe_hold", "ready")):
            return "safe_cancellation_not_confirmed"
        return self._age_error(p["recorded_at"], p["now"], self.thresholds.max_message_age_s, earliest)

    def _on_acknowledge_hold(self, p):
        if self.phase != RoundPhase.WAIT_HOLD:
            return "hold_phase_required", ()
        error = self._ack_error(p, self.hold_token, self.scene_prepared_at, self.hold_acks)
        if error:
            return error, ()
        self.hold_acks[p["drone"]] = p["recorded_at"]
        if len(self.hold_acks) == 2:
            self.phase = RoundPhase.WAIT_OBSERVATIONS
        return "accepted", ()

    def _on_observe(self, p):
        if self.phase not in (RoundPhase.WAIT_OBSERVATIONS, RoundPhase.READY):
            return "observation_phase_required", ()
        if p["drone"] not in self.DRONES:
            return "unknown_drone", ()
        if p["actor_name"] != self.actor_name:
            return "actor_name_mismatch", ()
        if not _name(p["source"]) or not _name(p["message_id"]):
            return "invalid_message_identity", ()
        if p["message_id"] in self.used_message_ids:
            return "duplicate_message", ()
        if not isinstance(p["identity_confirmed"], bool) or not isinstance(p["ready"], bool):
            return "invalid_observation_flags", ()
        error = self._age_error(p["captured_at"], p["now"], self.thresholds.max_observation_age_s,
                                max(self.hold_acks.values()))
        if error:
            return "observation_" + error, ()
        previous = self.observations.get(p["drone"])
        if previous and p["captured_at"] <= previous["captured_at"]:
            return "old_observation", ()
        if p["eta_seconds"] is None:
            if p["estimate_at"] is not None:
                return "eta_estimate_missing", ()
        else:
            if not _finite(p["eta_seconds"]) or p["eta_seconds"] < 0:
                return "invalid_eta", ()
            error = self._age_error(p["estimate_at"], p["now"], self.thresholds.max_eta_age_s, p["captured_at"])
            if error:
                return "eta_" + error, ()
        self.used_message_ids.add(p["message_id"])
        self.observations[p["drone"]] = {key: p[key] for key in
                                        ("message_id", "source", "captured_at", "identity_confirmed", "ready", "eta_seconds", "estimate_at")}
        if len(self.observations) == 2:
            self.phase = RoundPhase.READY
        return "accepted", ()

    def _observations_error(self, now, require_eta=False):
        if len(self.observations) != 2 or len(self.hold_acks) != 2:
            return "two_drone_observations_required"
        for drone in self.DRONES:
            obs = self.observations[drone]
            error = self._age_error(obs["captured_at"], now, self.thresholds.max_observation_age_s)
            if error:
                return "observation_" + error
            if not obs["identity_confirmed"] or not obs["ready"]:
                return "observation_identity_or_readiness_unconfirmed"
            if require_eta:
                if obs["eta_seconds"] is None:
                    return "eta_required"
                error = self._age_error(obs["estimate_at"], now, self.thresholds.max_eta_age_s)
                if error:
                    return "eta_" + error
        return None

    def _on_assign(self, p):
        if self.phase != RoundPhase.READY:
            return "ready_phase_required", ()
        if p["policy"] not in ("fixed_A", "alternating", "nearest_eta"):
            return "unknown_allocation_policy", ()
        if not _name(p["assignment_id"]) or not _name(p["plan_id"]):
            return "invalid_assignment_identity", ()
        if p["assignment_id"] in self.used_assignment_ids or p["plan_id"] in self.used_plan_ids:
            return "assignment_or_plan_id_reused", ()
        error = self._age_error(p["created_at"], p["now"], self.thresholds.max_assignment_age_s,
                                max(obs["captured_at"] for obs in self.observations.values()))
        if error:
            return "assignment_" + error, ()
        error = self._observations_error(p["now"], p["policy"] == "nearest_eta")
        if error:
            return error, ()
        if p["policy"] == "fixed_A":
            executor = "DroneA"
        elif p["policy"] == "alternating":
            executor = self.DRONES[(self.round_id - 1) % 2]
        else:
            executor = min(self.DRONES, key=lambda drone: self.observations[drone]["eta_seconds"])
        observer = self.DRONES[1] if executor == self.DRONES[0] else self.DRONES[0]
        self.assignment = {key: p[key] for key in ("assignment_id", "plan_id", "policy", "created_at")}
        self.assignment.update(executor=executor, observer=observer, round_id=self.round_id,
                               target_generation=self.target_generation, actor_name=self.actor_name)
        self.used_assignment_ids.add(p["assignment_id"])
        self.used_plan_ids.add(p["plan_id"])
        self.phase = RoundPhase.ASSIGNED
        return "accepted", ({"operation": "ASSIGN_ROLES", **deepcopy(self.assignment)},)

    def _plan_error(self, p):
        if self.assignment is None:
            return "assignment_required"
        for key in ("assignment_id", "plan_id", "executor"):
            if p[key] != self.assignment[key]:
                return key + "_mismatch"
        return None

    def _on_dispatch(self, p):
        error = self._plan_error(p)
        if error:
            return error, ()
        if self.phase != RoundPhase.ASSIGNED:
            return "assigned_phase_required", ()
        error = self._age_error(self.assignment["created_at"], p["now"], self.thresholds.max_assignment_age_s)
        if error:
            return "assignment_" + error, ()
        error = self._age_error(p["issued_at"], p["now"], self.thresholds.max_message_age_s, self.assignment["created_at"])
        if error:
            return error, ()
        error = self._observations_error(p["now"], self.assignment["policy"] == "nearest_eta")
        if error:
            return error, ()
        stamp = p["collision_baseline_timestamp"]
        if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp < self.last_collision_timestamps[p["executor"]]:
            return "invalid_collision_baseline", ()
        self.collision_baseline_timestamp = stamp
        self.dispatch_at = p["now"]
        self.phase = RoundPhase.EXECUTING
        return "accepted", (
            {"operation": "DISPATCH_PLAN", **deepcopy(self.assignment)},
            {"operation": "OBSERVER_HOLD", "drone": self.assignment["observer"],
             "round_id": self.round_id, "target_generation": self.target_generation},
        )

    def _on_mark_contact(self, p):
        error = self._plan_error(p)
        if error:
            return error, ()
        if self.phase != RoundPhase.EXECUTING:
            return "executing_phase_required", ()
        error = self._age_error(p["issued_at"], p["now"], self.thresholds.max_message_age_s, self.dispatch_at)
        if error:
            return error, ()
        self.phase = RoundPhase.CONTACT
        return "accepted", ()

    def _on_confirm_hit(self, p):
        error = self._plan_error(p)
        if error:
            return error, ()
        if self.round_id in self.accepted_hit_rounds:
            return "duplicate_hit", ()
        if self.phase not in (RoundPhase.EXECUTING, RoundPhase.CONTACT):
            return "contact_or_executing_phase_required", ()
        if p["actor_name"] != self.actor_name:
            return "actor_name_mismatch", ()
        if p["has_collided"] is not True:
            return "contact_not_confirmed", ()
        stamp = p["collision_timestamp"]
        if not _positive_int(stamp):
            return "invalid_collision_timestamp", ()
        if stamp <= max(self.collision_baseline_timestamp, self.last_collision_timestamps[p["executor"]]):
            return "old_collision_timestamp", ()
        error = self._age_error(p["reported_at"], p["now"], self.thresholds.max_message_age_s, self.dispatch_at)
        if error:
            return error, ()
        self.last_collision_timestamps[p["executor"]] = stamp
        self.accepted_hit_rounds.add(self.round_id)
        self.hit_at = p["now"]
        self.cancel_token = f"cancel_r{self.round_id:04d}_g{self.target_generation:04d}"
        self.cancel_acks.clear()
        self.phase = RoundPhase.CANCEL_PENDING
        return "accepted", self._hold_actions(self.cancel_token)

    def _on_acknowledge_cancel(self, p):
        if self.phase != RoundPhase.CANCEL_PENDING:
            return "cancel_phase_required", ()
        error = self._ack_error(p, self.cancel_token, self.hit_at, self.cancel_acks)
        if error:
            return error, ()
        self.cancel_acks[p["drone"]] = p["recorded_at"]
        if len(self.cancel_acks) != 2:
            return "accepted", ()
        self.phase = RoundPhase.REFRESH_PENDING
        return "accepted", ({
            "operation": "REQUEST_TARGET_REFRESH", "retire_actor_name": self.actor_name,
            "next_round_id": self.round_id + 1, "next_target_generation": self.target_generation + 1,
            "next_actor_name": self.actor_for_round(self.round_id + 1),
        },)

    @classmethod
    def replay(cls, events: Iterable[Mapping[str, Any]]) -> "RefreshedTargetRound":
        events = list(events)
        if not events or events[0].get("event") != "initialize" or events[0].get("protocol_version") != 1:
            raise ValueError("initialize event required")
        machine = cls(RoundThresholds(**events[0]["payload"]["thresholds"]))
        if machine.snapshot() != events[0]["state"]:
            raise ValueError("initial state mismatch")
        for index, recorded in enumerate(events[1:], 1):
            machine.submit(recorded["event"], _unpack(recorded["payload"]))
            if machine._event_log[-1] != dict(recorded):
                raise ValueError(f"event replay mismatch at {index}")
        return machine
