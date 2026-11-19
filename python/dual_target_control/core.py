"""Coordinate bounds for the native route overlay."""
from dataclasses import dataclass, fields
import math
from typing import Mapping

import numpy as np


@dataclass
class ControlConfig:
    x_min: float = -20.0
    x_max: float = 20.0
    y_min: float = -20.0
    y_max: float = 20.0
    height_min: float = 2.0
    height_max: float = 10.0
    margin: float = 1.0
    arrival_radius: float = 0.5
    max_speed: float = 4.0
    max_acceleration: float = 4.0
    max_yaw_rate_deg: float = 120.0
    control_hz: float = 50.0
    display_hz: float = 20.0
    seed: int = 13
    heading_tolerance_deg: float = 20.0
    response_time_s: float = 0.35
    # Metres between randomly generated consecutive targets (and measured
    # reset position -> first target). Both zero retain whole-field sampling.
    target_distance_min: float = 0.0
    target_distance_max: float = 0.0

    def validate(self) -> "ControlConfig":
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"{field.name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{field.name} must be finite")
        if self.margin < 0:
            raise ValueError("margin must be non-negative")
        for name in ("arrival_radius", "max_speed", "max_acceleration",
                     "max_yaw_rate_deg", "control_hz", "display_hz", "response_time_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.heading_tolerance_deg <= 180:
            raise ValueError("heading_tolerance_deg must be in (0, 180]")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        spans = np.array([self.x_max - self.x_min, self.y_max - self.y_min,
                          self.height_max - self.height_min]) - 2 * self.margin
        if np.any(spans <= 2 * self.arrival_radius):
            raise ValueError("each field span must exceed 2 * (margin + arrival_radius)")
        if self.target_distance_min != 0 or self.target_distance_max != 0:
            if not 0 < self.target_distance_min <= self.target_distance_max:
                raise ValueError("target distances must satisfy 0 < min <= max, or both be zero")
            if self.target_distance_min <= 2 * self.arrival_radius:
                raise ValueError("target_distance_min must exceed twice arrival_radius")
            # At the box centre the farthest feasible point is half a diagonal
            # away. This bound guarantees a successor from every in-field anchor,
            # rather than accepting a setting that can strand a rolling queue.
            if self.target_distance_min > 0.5 * float(np.linalg.norm(spans)):
                raise ValueError("target_distance_min is unreachable from the buffered field centre")
        return self

    @classmethod
    def from_dict(cls, values: Mapping) -> "ControlConfig":
        """Load known control settings, allowing a file to contain GUI/API settings."""
        if not isinstance(values, Mapping):
            raise ValueError("configuration must be a mapping")
        known = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in values.items() if name in known}).validate()

    @property
    def lower(self) -> np.ndarray:
        return np.array([self.x_min, self.y_min, self.height_min], dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.array([self.x_max, self.y_max, self.height_max], dtype=float)
