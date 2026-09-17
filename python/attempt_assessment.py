"""Observation-based contact-attempt assessment for the AirSim balloon task.

The evaluator is deliberately flight-independent.  It turns a temporal visual
state, frame freshness, and collision evidence into an auditable assessment;
the caller still decides whether a vehicle is allowed to move or whether a
fallback vehicle should be dispatched.  This keeps simulator truth out of the
policy input while allowing the evaluator to use a collision as authoritative
completion evidence.
"""

from dataclasses import dataclass
from enum import Enum
import math


class AttemptState(str, Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    UNCONFIRMED = "UNCONFIRMED"
    CONTACT_CONFIRMED = "CONTACT_CONFIRMED"
    FAILED = "FAILED"


class AttemptAction(str, Enum):
    WAIT = "WAIT"
    CONTINUE = "CONTINUE"
    REACQUIRE = "REACQUIRE"
    HANDOFF_ELIGIBLE = "HANDOFF_ELIGIBLE"
    COMPLETE = "COMPLETE"
    STOP = "STOP"


@dataclass(frozen=True)
class AttemptAssessment:
    state: AttemptState
    action: AttemptAction
    reason: str
    elapsed_s: float
    tracking_state: str | None
    observation_fresh: bool
    collision_kind: str
    changed: bool


class ContactAttemptAssessment:
    """Classify one bounded contact attempt without issuing flight commands.

    ``LOST`` is kept distinct from ``FAILED``: the former makes a handoff
    eligible, while the latter is emitted only after the configured attempt
    window expires without authoritative target contact.  A temporary visual
    dropout remains a reacquisition opportunity.
    """

    _TRACKING_STATES = {"DETECTED", "TRACKING"}
    _TEMPORARY_STATES = {"TEMPORARILY_LOST", "STALE", "UNRELIABLE_OBSERVATION"}
    _LOST_STATES = {"LOST", "UNCONFIRMED"}
    _COLLISIONS = {"NONE", "TARGET", "OTHER"}

    def __init__(self, timeout_s=18.0):
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Contact-attempt timeout must be finite and positive")
        self.timeout_s = timeout_s
        self.state = AttemptState.IDLE
        self.started = False
        self.transitions: list[dict] = []

    def start(self) -> AttemptAssessment:
        if self.started:
            raise ValueError("Contact attempt has already started")
        self.started = True
        return self._set(AttemptState.UNCONFIRMED, AttemptAction.WAIT,
                         "attempt_started", 0.0, None, False, "NONE")

    def update(self, *, tracking_state=None, observation_fresh=False,
               collision_kind="NONE", elapsed_s=0.0) -> AttemptAssessment:
        if not self.started:
            raise ValueError("Call start() before updating a contact attempt")
        if self.state in (AttemptState.CONTACT_CONFIRMED, AttemptState.FAILED):
            raise ValueError(f"Attempt is terminal: {self.state.value}")
        try:
            elapsed_s = float(elapsed_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("Attempt elapsed time must be numeric") from exc
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Attempt elapsed time must be finite and non-negative")
        collision_kind = str(collision_kind).upper()
        if collision_kind not in self._COLLISIONS:
            raise ValueError(f"Unsupported collision kind: {collision_kind}")
        tracking_state = None if tracking_state is None else str(tracking_state).upper()
        observation_fresh = bool(observation_fresh)

        if collision_kind == "TARGET":
            return self._set(AttemptState.CONTACT_CONFIRMED, AttemptAction.COMPLETE,
                             "authoritative_target_collision", elapsed_s,
                             tracking_state, observation_fresh, collision_kind)
        if collision_kind == "OTHER":
            return self._set(AttemptState.FAILED, AttemptAction.STOP,
                             "non_target_collision", elapsed_s,
                             tracking_state, observation_fresh, collision_kind)
        if elapsed_s >= self.timeout_s:
            return self._set(AttemptState.FAILED, AttemptAction.HANDOFF_ELIGIBLE,
                             "attempt_timeout_without_target_collision", elapsed_s,
                             tracking_state, observation_fresh, collision_kind)

        if observation_fresh and tracking_state in self._TRACKING_STATES:
            return self._set(AttemptState.TRACKING, AttemptAction.CONTINUE,
                             "fresh_target_track", elapsed_s, tracking_state,
                             observation_fresh, collision_kind)
        if tracking_state in self._LOST_STATES:
            return self._set(AttemptState.UNCONFIRMED, AttemptAction.HANDOFF_ELIGIBLE,
                             "track_lost_before_authoritative_contact", elapsed_s,
                             tracking_state, observation_fresh, collision_kind)
        if tracking_state in self._TEMPORARY_STATES or not observation_fresh:
            return self._set(AttemptState.TEMPORARILY_LOST, AttemptAction.REACQUIRE,
                             "temporary_or_stale_observation", elapsed_s,
                             tracking_state, observation_fresh, collision_kind)
        return self._set(AttemptState.UNCONFIRMED, AttemptAction.WAIT,
                         "insufficient_observation_evidence", elapsed_s,
                         tracking_state, observation_fresh, collision_kind)

    def fail(self, reason, *, elapsed_s=0.0, tracking_state=None,
             observation_fresh=False):
        """Close an attempt for a caller-side safety bound.

        A visual controller can hit a limit that is distinct from its normal
        timeout (for example, the bounded occlusion dash).  Recording that
        limit through the same terminal state keeps the handoff decision
        auditable without pretending that a collision occurred.
        """
        if not self.started:
            raise ValueError("Call start() before failing a contact attempt")
        if self.state in (AttemptState.CONTACT_CONFIRMED, AttemptState.FAILED):
            raise ValueError(f"Attempt is terminal: {self.state.value}")
        reason = str(reason).strip()
        if not reason:
            raise ValueError("Failure reason must not be empty")
        try:
            elapsed_s = float(elapsed_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("Attempt elapsed time must be numeric") from exc
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Attempt elapsed time must be finite and non-negative")
        tracking_state = None if tracking_state is None else str(tracking_state).upper()
        return self._set(AttemptState.FAILED, AttemptAction.HANDOFF_ELIGIBLE,
                         reason, elapsed_s, tracking_state,
                         bool(observation_fresh), "NONE")

    def summary(self) -> dict:
        return {
            "state": self.state.value,
            "started": self.started,
            "timeout_s": self.timeout_s,
            "transitions": list(self.transitions),
        }

    def _set(self, state, action, reason, elapsed_s, tracking_state,
             observation_fresh, collision_kind):
        previous = self.state
        self.state = state
        changed = previous != state
        assessment = AttemptAssessment(
            state=state,
            action=action,
            reason=reason,
            elapsed_s=float(elapsed_s),
            tracking_state=tracking_state,
            observation_fresh=observation_fresh,
            collision_kind=collision_kind,
            changed=changed,
        )
        if changed or not self.transitions:
            self.transitions.append({
                "from": previous.value,
                "to": state.value,
                "action": action.value,
                "reason": reason,
                "elapsed_s": float(elapsed_s),
                "tracking_state": tracking_state,
                "observation_fresh": observation_fresh,
                "collision_kind": collision_kind,
            })
        return assessment
