"""Temporal filtering for advisory camera detections."""
from dataclasses import dataclass
from enum import Enum


class TrackingState(str, Enum):
    DETECTED = "DETECTED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    LOST = "LOST"


@dataclass(frozen=True)
class TrackingUpdate:
    state: TrackingState
    previous_state: TrackingState
    changed: bool
    detection_streak: int
    miss_streak: int


class PerceptionTracker:
    """Debounce first acquisition and tolerate short camera dropouts.

    A new track needs ``confirm_frames`` consecutive detections. Once acquired,
    the first missed frame becomes TEMPORARILY_LOST and ``lost_frames``
    consecutive misses make the track LOST. A detection immediately reacquires
    a temporarily lost track.
    """

    def __init__(self, confirm_frames=2, lost_frames=3):
        if confirm_frames < 1 or lost_frames < 1:
            raise ValueError("Frame thresholds must be positive")
        self.confirm_frames = int(confirm_frames)
        self.lost_frames = int(lost_frames)
        self.state = TrackingState.LOST
        self.detection_streak = 0
        self.miss_streak = 0

    def update(self, detected):
        previous = self.state
        if detected:
            self.detection_streak += 1
            self.miss_streak = 0
            if previous == TrackingState.TEMPORARILY_LOST:
                self.state = TrackingState.DETECTED
            elif self.detection_streak >= self.confirm_frames:
                self.state = TrackingState.DETECTED
        else:
            self.detection_streak = 0
            self.miss_streak += 1
            if previous == TrackingState.DETECTED:
                self.state = TrackingState.TEMPORARILY_LOST
            elif previous == TrackingState.TEMPORARILY_LOST and self.miss_streak >= self.lost_frames:
                self.state = TrackingState.LOST

        return TrackingUpdate(
            state=self.state,
            previous_state=previous,
            changed=self.state != previous,
            detection_streak=self.detection_streak,
            miss_streak=self.miss_streak,
        )
