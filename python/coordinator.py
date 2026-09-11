"""Flight-independent task arbitration for first attempt and fallback."""
from dataclasses import dataclass

from mission_state import MissionState, transition


@dataclass(frozen=True)
class Dispatch:
    vehicle: str
    reason: str


class MissionCoordinator:
    def __init__(self, first_vehicle="DroneA", fallback_vehicle="DroneB"):
        if first_vehicle == fallback_vehicle:
            raise ValueError("First and fallback vehicles must differ")
        self.first_vehicle = first_vehicle
        self.fallback_vehicle = fallback_vehicle
        self.state = MissionState.READY
        self.active_vehicle = None
        self.corridor_clear = False

    def start(self):
        self.state = transition(self.state, "start")
        self.active_vehicle = self.first_vehicle
        return Dispatch(self.first_vehicle, "first_attempt")

    def confirm_hit(self, vehicle):
        if vehicle != self.active_vehicle:
            raise ValueError("Only the active vehicle may confirm a hit")
        self.state = transition(self.state, "hit")
        return "mission_complete"

    def confirm_miss(self, vehicle):
        if vehicle != self.active_vehicle or self.state not in (MissionState.A_APPROACH, MissionState.B_APPROACH):
            raise ValueError("Only the active vehicle may report a miss")
        self.state = transition(self.state, "miss")
        self.active_vehicle = None
        self.corridor_clear = False
        return "awaiting_corridor_clear"

    def mark_corridor_clear(self):
        if self.state != MissionState.A_MISSED:
            raise ValueError("Corridor can only clear after the first attempt misses")
        self.corridor_clear = True
        self.state = transition(self.state, "fallback_ready")
        self.active_vehicle = self.fallback_vehicle
        return Dispatch(self.fallback_vehicle, "first_attempt_missed")

    def timeout(self):
        if self.state != MissionState.B_APPROACH:
            raise ValueError("Only a dispatched fallback can time out")
        self.state = transition(self.state, "timeout")
        return "mission_failed"

    def abort(self):
        """Record execution failure, including failure after a confirmed hit."""
        self.state = MissionState.FAILED
        self.active_vehicle = None
        self.corridor_clear = False
        return "mission_failed"
