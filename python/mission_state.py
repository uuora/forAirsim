"""Small, deterministic mission state machine for the balloon task."""

from enum import Enum


class MissionState(str, Enum):
    READY = "READY"
    A_APPROACH = "A_APPROACH"
    A_MISSED = "A_MISSED"
    B_APPROACH = "B_APPROACH"
    HIT_CONFIRMED = "HIT_CONFIRMED"
    FAILED = "FAILED"


def transition(state: MissionState, event: str) -> MissionState:
    """Return the next state for a validated mission event."""
    transitions = {
        (MissionState.READY, "start"): MissionState.A_APPROACH,
        (MissionState.A_APPROACH, "hit"): MissionState.HIT_CONFIRMED,
        (MissionState.A_APPROACH, "miss"): MissionState.A_MISSED,
        (MissionState.A_MISSED, "fallback_ready"): MissionState.B_APPROACH,
        (MissionState.B_APPROACH, "hit"): MissionState.HIT_CONFIRMED,
        (MissionState.B_APPROACH, "miss"): MissionState.FAILED,
        (MissionState.B_APPROACH, "timeout"): MissionState.FAILED,
    }
    try:
        return transitions[(state, event)]
    except KeyError as exc:
        raise ValueError(f"Invalid mission transition: {state.value} + {event}") from exc
