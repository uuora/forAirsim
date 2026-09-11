"""Validated, file-backed mission geometry and safety limits."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MissionConfig:
    vehicles: tuple
    target_name: str
    target_world_ned_m: np.ndarray
    pads_world_ned_m: dict
    staging_world_ned_m: dict
    approach_world_ned_m: np.ndarray
    deliberate_miss_offset_y_m: float
    perception_confirm_frames: int
    perception_lost_frames: int
    perception_alignment_deadband: float
    perception_stop_distance_m: float
    minimum_drone_separation_m: float
    maximum_waiting_drift_m: float
    contact_timeout_s: float

    @classmethod
    def load(cls, path):
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        vehicles = tuple(raw["vehicles"])
        if vehicles != ("DroneA", "DroneB"):
            raise ValueError("The base mission requires exactly DroneA and DroneB")
        pads = {name: np.array(raw["pads_world_ned_m"][name], dtype=float) for name in vehicles}
        staging = {name: np.array(raw["staging_world_ned_m"][name], dtype=float) for name in vehicles}
        cfg = cls(vehicles, raw["target_name"], np.array(raw["target_world_ned_m"], dtype=float),
                   pads, staging, np.array(raw["approach_world_ned_m"], dtype=float),
                   float(raw["test_parameters"]["deliberate_miss_offset_y_m"]),
                   int(raw["perception"]["confirm_frames"]),
                   int(raw["perception"]["lost_frames"]),
                   float(raw["perception"]["alignment_deadband"]),
                   float(raw["perception"]["stop_distance_m"]),
                   float(raw["safety"]["minimum_drone_separation_m"]),
                   float(raw["safety"]["maximum_waiting_drift_m"]),
                   float(raw["safety"]["contact_timeout_s"]))
        cfg.validate()
        return cfg

    def validate(self):
        if self.target_world_ned_m.shape != (3,) or self.approach_world_ned_m.shape != (3,):
            raise ValueError("World positions must be three-element vectors")
        if np.linalg.norm(self.staging_world_ned_m["DroneA"] - self.staging_world_ned_m["DroneB"]) < self.minimum_drone_separation_m:
            raise ValueError("Staging positions violate minimum drone separation")
        if self.minimum_drone_separation_m <= 0 or self.maximum_waiting_drift_m <= 0 or self.contact_timeout_s <= 0:
            raise ValueError("Safety limits must be positive")
        if abs(self.deliberate_miss_offset_y_m) < 1.5:
            raise ValueError("Deliberate miss offset must remain outside the fixed balloon body")
        if self.perception_confirm_frames < 1 or self.perception_lost_frames < 1:
            raise ValueError("Perception frame thresholds must be positive")
        if not 0 < self.perception_alignment_deadband < 1 or self.perception_stop_distance_m <= 0:
            raise ValueError("Perception guidance thresholds are outside safe configuration bounds")

    def to_dict(self):
        return {
            "vehicles": list(self.vehicles),
            "target_name": self.target_name,
            "target_world_ned_m": self.target_world_ned_m.tolist(),
            "pads_world_ned_m": {name: value.tolist() for name, value in self.pads_world_ned_m.items()},
            "staging_world_ned_m": {name: value.tolist() for name, value in self.staging_world_ned_m.items()},
            "approach_world_ned_m": self.approach_world_ned_m.tolist(),
            "test_parameters": {"deliberate_miss_offset_y_m": self.deliberate_miss_offset_y_m},
            "perception": {
                "confirm_frames": self.perception_confirm_frames,
                "lost_frames": self.perception_lost_frames,
                "alignment_deadband": self.perception_alignment_deadband,
                "stop_distance_m": self.perception_stop_distance_m,
                "role": "advisory_only",
            },
            "safety": {
                "minimum_drone_separation_m": self.minimum_drone_separation_m,
                "maximum_waiting_drift_m": self.maximum_waiting_drift_m,
                "contact_timeout_s": self.contact_timeout_s,
            },
        }
