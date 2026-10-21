"""Small, deterministic two-target planner and controller.

Coordinates are metres in an x/y/height-up world. Measured yaw is in radians;
the absolute yaw command is in degrees. The core deliberately has no simulator,
GUI, network, or perception dependency. Its bounds are command-level protections,
not a guarantee that an already unsafe physical state can stop before a wall.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from time import perf_counter
from typing import ClassVar, Mapping

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite numbers")
    return result.copy()


def _unit(value: np.ndarray, fallback=None) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length > 1e-9:
        return value / length
    return np.zeros(3) if fallback is None else np.asarray(fallback).copy()


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _limited(vector: np.ndarray, limit: float) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector * (limit / length) if length > limit else vector.copy()


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

    @property
    def inner_lower(self) -> np.ndarray:
        return self.lower + self.margin

    @property
    def inner_upper(self) -> np.ndarray:
        return self.upper - self.margin


@dataclass(frozen=True)
class BezierParameters:
    """Bounded search variables; defaults reproduce the original planner.

    These parameters shape the curve and its passage speed. They do not change
    the arrival gate, field bounds, or command speed/acceleration/yaw limits.
    """

    outgoing_weight: float = 0.4
    initial_handle_ratio: float = 0.3
    approach_handle_ratio: float = 0.2
    corridor_fraction: float = 0.45
    pass_speed_scale: float = 1.0

    _LOWER: ClassVar[tuple[float, ...]] = (0.0, 0.15, 0.1, 0.2, 0.6)
    _UPPER: ClassVar[tuple[float, ...]] = (0.8, 0.5, 0.35, 0.65, 1.3)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "BezierParameters":
        for field, lower, upper in zip(fields(self), self._LOWER, self._UPPER):
            value = getattr(self, field.name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{field.name} must be numeric")
            if not math.isfinite(float(value)) or not lower <= value <= upper:
                raise ValueError(f"{field.name} must be finite and in [{lower}, {upper}]")
        return self

    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, field.name) for field in fields(self)], dtype=float)

    def to_dict(self) -> dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}

    @classmethod
    def from_vector(cls, values) -> "BezierParameters":
        try:
            vector = np.asarray(values, dtype=object)
        except (TypeError, ValueError) as error:
            raise ValueError("Bezier parameters must be a vector of five numbers") from error
        if vector.shape != (5,):
            raise ValueError("Bezier parameters must be a vector of five numbers")
        return cls(*vector.tolist())

    @classmethod
    def from_dict(cls, values: Mapping) -> "BezierParameters":
        if not isinstance(values, Mapping):
            raise ValueError("Bezier parameters must be a mapping")
        known = {field.name for field in fields(cls)}
        if any(name not in known for name in values):
            raise ValueError("unknown Bezier parameter")
        return cls(**values)

    @classmethod
    def lower_bounds(cls) -> np.ndarray:
        return np.array(cls._LOWER, dtype=float)

    @classmethod
    def upper_bounds(cls) -> np.ndarray:
        return np.array(cls._UPPER, dtype=float)


class RollingController:
    """Bounded cubic path through a rolling queue of two directly read targets.

    The next target changes the tangent and speed before the first target. Each
    hit reuses the measured velocity as the new curve's starting tangent. Curve
    construction is closed form; no iterative optimizer sits in the control loop.
    Arrival gates position and horizontal body yaw. Camera pitch is an adapter
    concern, and must be checked separately for a full optical-axis constraint.
    """

    _SAMPLES = 81

    def __init__(self, config: ControlConfig, parameters: BezierParameters | None = None):
        self.config = config.validate()
        if parameters is not None and not isinstance(parameters, BezierParameters):
            raise ValueError("parameters must be BezierParameters")
        self._parameters = BezierParameters() if parameters is None else parameters.validate()
        self._rng = np.random.default_rng(int(config.seed))
        self._targets: list[np.ndarray] = []
        self._path = np.empty((0, 3))
        self._arc = np.empty(0)
        self._curvature = np.empty(0)
        self._speed_profile = np.empty(0)
        self._progress = 0.0
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._last_position = None
        self._last_yaw = 0.0
        self._heading = 0.0
        self._last_command = np.zeros(3)
        self._yaw_command = 0.0
        self._planning_ms = 0.0
        self._planning_revision = 0
        self._hits = 0
        self._last_result = None
        self._target_speed = config.max_speed
        self._entry_direction = np.zeros(3)

    @property
    def targets(self) -> list[np.ndarray]:
        return [target.copy() for target in self._targets]

    @property
    def planned_path(self) -> np.ndarray:
        return self._path.copy()

    @property
    def parameters(self) -> BezierParameters:
        return self._parameters

    @property
    def planning_revision(self) -> int:
        return self._planning_revision

    @property
    def current_position(self) -> np.ndarray:
        return self._position.copy()

    @property
    def current_velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def current_yaw(self) -> float:
        return self._last_yaw

    def set_parameters(self, parameters: BezierParameters) -> dict | None:
        """Rebuild a curve without resetting commands, arrivals, or queue state.

        Unlike pause/resume ``replan``, this keeps the previous measured sample
        for the next swept arrival check and preserves command slew continuity.
        Before reset it only selects the parameters for the initial plan.
        """
        if not isinstance(parameters, BezierParameters):
            raise ValueError("parameters must be BezierParameters")
        self._parameters = parameters.validate()
        if self._last_position is None:
            return None
        boundary_limited = self._last_result["boundary_limited"]
        self._plan(self._position, self._velocity)
        self._last_result = self._result(None, boundary_limited, self._last_command, self._last_yaw)
        return self.snapshot()

    def reset(self, position, velocity=None, yaw: float = 0.0) -> dict:
        self._position = _vector(position, "position")
        self._velocity = _vector(np.zeros(3) if velocity is None else velocity, "velocity")
        if not math.isfinite(yaw):
            raise ValueError("yaw must be finite radians")
        self._rng = np.random.default_rng(int(self.config.seed))
        self._hits = 0
        self._last_position = self._position.copy()
        self._last_yaw = self._yaw_command = _wrap(float(yaw))
        self._last_command = _limited(self._velocity, self.config.max_speed)
        first = self._sample_target(self._position)
        self._targets = [first, self._sample_target(first, [self._position])]
        self._heading = self._bearing(first - self._position, yaw)
        self._plan(self._position, self._velocity)
        self._last_result = self._result(None, False, self._last_command, yaw)
        return self.snapshot()

    def set_targets(self, targets) -> None:
        """Replace the queue for a reproducible test or externally supplied task.

        Explicit tasks may exceed the random-generation distance interval. The
        buffered field and non-overlapping arrival spheres still apply. After a
        hit, newly generated successors again use the configured interval.
        """
        if self._last_position is None:
            raise RuntimeError("reset the controller before setting targets")
        if len(targets) != 2:
            raise ValueError("exactly two targets are required")
        result = [_vector(target, "target") for target in targets]
        for target in result:
            if np.any(target < self.config.inner_lower) or np.any(target > self.config.inner_upper):
                raise ValueError("target lies outside the buffered field")
        if np.linalg.norm(result[0] - result[1]) <= 2 * self.config.arrival_radius:
            raise ValueError("target arrival spheres must not overlap")
        self._targets = result
        self._heading = self._bearing(result[0] - self._position, self._last_yaw)
        self._plan(self._position, self._velocity)
        self._last_result = self._result(None, False, self._last_command, self._last_yaw)

    def replan(self, position, velocity=None, yaw: float | None = None) -> dict:
        """Resume/recover from measured state while preserving targets and hit count.

        A paused adapter should brake its plant before calling this method. The
        command warm start is then the measured velocity, not a stale pre-pause
        velocity. No segment from before the pause can count as a swept arrival.
        """
        if self._last_position is None:
            raise RuntimeError("call reset before replan")
        self._position = _vector(position, "position")
        self._velocity = _vector(self._velocity if velocity is None else velocity, "velocity")
        yaw = self._last_yaw if yaw is None else yaw
        if not math.isfinite(yaw):
            raise ValueError("yaw must be finite radians")
        self._last_position = self._position.copy()
        self._last_yaw = self._yaw_command = _wrap(float(yaw))
        self._last_command = _limited(self._velocity, self.config.max_speed)
        self._heading = self._bearing(self._targets[0] - self._position, yaw)
        self._plan(self._position, self._velocity)
        self._last_result = self._result(None, False, self._last_command, yaw)
        return self.snapshot()

    def _sample_target(self, anchor: np.ndarray,
                       excluded: list[np.ndarray] | tuple[np.ndarray, ...] = ()) -> np.ndarray:
        """Generate a successor relative to its preceding target, not the plant.

        Extra exclusions preserve the legacy whole-field generator's clearance
        around the aircraft. In distance-limited mode the preceding target is
        the sole distance anchor; its minimum distance separates arrival spheres.
        """
        if self.config.target_distance_max > 0:
            return self._sample_local_target(anchor)
        low, high = self.config.inner_lower, self.config.inner_upper
        excluded = [anchor, *excluded]
        separation = max(2.2 * self.config.arrival_radius,
                         min(4.0, 0.2 * float(np.linalg.norm(high - low))))
        for _ in range(256):
            candidate = self._rng.uniform(low, high)
            if all(np.linalg.norm(candidate - point) >= separation for point in excluded):
                return candidate
        # Bounded deterministic fallback when a very small user field is crowded.
        corners = np.array([[x, y, z] for x in (low[0], high[0])
                            for y in (low[1], high[1]) for z in (low[2], high[2])])
        distances = [min(np.linalg.norm(candidate - point) for point in excluded)
                     for candidate in corners]
        candidate = corners[int(np.argmax(distances))]
        if max(distances) <= 2 * self.config.arrival_radius:
            raise ValueError("field is too small to generate separated targets")
        return candidate.copy()

    def _sample_local_target(self, anchor: np.ndarray) -> np.ndarray:
        """Sample a bounded spherical shell intersected with the buffered box.

        Each direction is intersected with the box before choosing a radius,
        which keeps corner/edge anchors local without clipping a distant point.
        The bounded fallback uses a segment inside the box whose endpoints
        bracket the requested radius, so it never reverts to a long field leg.
        """
        low, high = self.config.inner_lower, self.config.inner_upper
        minimum, maximum = self.config.target_distance_min, self.config.target_distance_max
        nearest = np.clip(anchor, low, high)
        farthest = np.where(np.abs(high - anchor) >= np.abs(low - anchor), high, low)
        nearest_distance = float(np.linalg.norm(nearest - anchor))
        farthest_distance = float(np.linalg.norm(farthest - anchor))
        if nearest_distance > maximum or farthest_distance < minimum:
            raise ValueError("target distance interval is unreachable from the measured reset position")

        for _ in range(128):
            direction = _unit(self._rng.normal(size=3))
            if not np.any(direction):
                continue
            lower_radius, upper_radius = minimum, maximum
            for axis in range(3):
                if abs(direction[axis]) < 1e-12:
                    if not low[axis] <= anchor[axis] <= high[axis]:
                        upper_radius = -1.0
                        break
                    continue
                intersections = ((low[axis] - anchor[axis]) / direction[axis],
                                 (high[axis] - anchor[axis]) / direction[axis])
                lower_radius = max(lower_radius, min(intersections))
                upper_radius = min(upper_radius, max(intersections))
            if lower_radius <= upper_radius:
                radius = float(self._rng.uniform(lower_radius, upper_radius))
                # Roundoff at a slab face may be a few ulps outside the box.
                candidate = np.clip(anchor + radius * direction, low, high)
                distance = float(np.linalg.norm(candidate - anchor))
                if minimum - 1e-10 <= distance <= maximum + 1e-10:
                    return candidate

        # Projection onto a convex box makes distance along nearest->farthest
        # monotonic. Solve that segment's sphere intersection at a feasible
        # midpoint radius, including when the measured anchor is outside it.
        radius = 0.5 * (max(minimum, nearest_distance) + min(maximum, farthest_distance))
        direction = _unit(farthest - nearest)
        offset = nearest - anchor
        along = float(np.dot(offset, direction))
        radial_delta = max(0.0, radius ** 2 - float(np.dot(offset, offset)))
        denominator = along + math.sqrt(along ** 2 + radial_delta)
        travel = radial_delta / denominator if denominator > 0 else 0.0
        return np.clip(nearest + travel * direction, low, high)

    def _handle_length(self, point: np.ndarray, direction: np.ndarray,
                       requested: float) -> float:
        """Scalar clipping preserves a handle's direction and convex-hull bound."""
        available = requested
        for axis in range(3):
            if direction[axis] > 1e-9:
                available = min(available, (self.config.inner_upper[axis] - point[axis]) / direction[axis])
            elif direction[axis] < -1e-9:
                available = min(available, (self.config.inner_lower[axis] - point[axis]) / direction[axis])
        return max(0.0, float(available))

    @staticmethod
    def _bezier(points: np.ndarray, samples: int = _SAMPLES) -> np.ndarray:
        t = np.linspace(0.0, 1.0, samples)[:, None]
        return ((1 - t) ** 3 * points[0] + 3 * (1 - t) ** 2 * t * points[1]
                + 3 * (1 - t) * t ** 2 * points[2] + t ** 3 * points[3])

    def _plan(self, position: np.ndarray, velocity: np.ndarray) -> None:
        began = perf_counter()
        start = np.clip(position, self.config.inner_lower, self.config.inner_upper)
        first, second = self._targets
        incoming, outgoing = _unit(first - start), _unit(second - first)
        start_direction = _unit(velocity, incoming) if np.linalg.norm(velocity) > 0.3 else incoming
        # Bias the passage towards the incoming leg. An unweighted bisector is
        # ill-conditioned at a near reversal and can demand a last-metre climb
        # or sideways turn. The next point still changes both curves and speed.
        tangent = _unit(incoming + self._parameters.outgoing_weight * outgoing, incoming)
        self._entry_direction = tangent
        first_length = float(np.linalg.norm(first - start))
        second_length = float(np.linalg.norm(second - first))
        common = min(0.3 * first_length, 0.3 * second_length, 5.0)
        common = min(self._handle_length(first, tangent, common),
                     self._handle_length(first, -tangent, common))
        initial = self._handle_length(start, start_direction,
            min(self._parameters.initial_handle_ratio * first_length, 5.0))
        final = self._handle_length(second, -outgoing, min(0.3 * second_length, 5.0))
        # Finish the turn before entering a straight final corridor. Preserve
        # the fixed 81-sample first leg and its exact target at index 80.
        # Reserve an inbound corridor independently of the outgoing handle.
        # Near a wall the forward handle can be millimetres long, even though
        # there is ample room behind the target to settle lateral velocity.
        corridor = self._handle_length(first, -tangent,
            min(self._parameters.corridor_fraction * first_length,
                max(3.0, 4.0 * self.config.max_speed * self.config.response_time_s)))
        approach = first - corridor * tangent
        approach_handle = self._handle_length(approach, -tangent,
            min(self._parameters.approach_handle_ratio * first_length, 3.0))
        curved = self._bezier(np.array([start, start + initial * start_direction,
            approach - approach_handle * tangent, approach]), self._SAMPLES - 20)
        straight = np.linspace(approach, first, 21)[1:]
        curve_a = np.concatenate((curved, straight), axis=0)
        curve_b = self._bezier(np.array([first, first + common * tangent,
                                         second - final * outgoing, second]))
        self._path = np.concatenate((curve_a, curve_b[1:]), axis=0)
        increments = np.linalg.norm(np.diff(self._path, axis=0), axis=1)
        self._arc = np.concatenate(([0.0], np.cumsum(increments)))
        derivative = np.gradient(self._path, axis=0)
        second_derivative = np.gradient(derivative, axis=0)
        self._curvature = np.linalg.norm(np.cross(derivative, second_derivative), axis=1) / np.maximum(
            np.linalg.norm(derivative, axis=1) ** 3, 1e-9)
        self._progress = 0.0
        cosine = float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0))
        self._target_speed = min(self.config.max_speed, self.config.max_speed
            * max(0.2, math.sqrt((1.0 + cosine) / 2.0)) * self._parameters.pass_speed_scale)
        self._speed_profile = np.minimum(self.config.max_speed, np.sqrt(
            0.7 * self.config.max_acceleration / np.maximum(self._curvature, 1e-9)))
        self._speed_profile[self._SAMPLES - 1] = min(self._speed_profile[self._SAMPLES - 1], self._target_speed)
        # Propagate braking distance backward rather than imposing the slowest
        # distant curve on the whole leg. This is a bounded O(samples) pass.
        for index in range(len(self._path) - 2, -1, -1):
            reachable = math.sqrt(self._speed_profile[index + 1] ** 2
                                  + 1.4 * self.config.max_acceleration * increments[index])
            self._speed_profile[index] = min(self._speed_profile[index], reachable)
        self._planning_ms = (perf_counter() - began) * 1000.0
        self._planning_revision += 1

    @staticmethod
    def _bearing(delta: np.ndarray, fallback: float) -> float:
        if np.linalg.norm(delta[:2]) < 1e-7:
            return _wrap(fallback)
        return math.atan2(float(delta[1]), float(delta[0]))

    def _hit(self, position: np.ndarray, yaw: float) -> dict | None:
        previous = self._last_position
        step = position - previous
        length_sq = float(np.dot(step, step))
        first = self._targets[0]
        alpha = float(np.clip(np.dot(first - previous, step) / length_sq, 0.0, 1.0)) if length_sq > 1e-12 else 0.0
        nearest = previous + alpha * step
        distance = float(np.linalg.norm(first - nearest))
        if distance > self.config.arrival_radius:
            return None
        # Solve the actual sphere intersection. The nearest projection above is
        # clamped to the segment; treating that endpoint as the centre of the
        # chord would evaluate an earlier point outside the arrival sphere.
        offset = previous - first
        outside = float(np.dot(offset, offset) - self.config.arrival_radius ** 2)
        entry = 0.0
        if outside > 0.0 and length_sq > 1e-12:
            linear = 2.0 * float(np.dot(offset, step))
            discriminant = max(0.0, linear ** 2 - 4.0 * length_sq * outside)
            entry = float(np.clip((-linear - math.sqrt(discriminant))
                                  / (2.0 * length_sq), 0.0, 1.0))
        entry_point = previous + entry * step
        reference = self._bearing(first - entry_point, self._heading)
        measured = _wrap(self._last_yaw + entry * _wrap(yaw - self._last_yaw))
        yaw_error = abs(math.degrees(_wrap(reference - measured)))
        if yaw_error > self.config.heading_tolerance_deg:
            return None
        return {"index": self._hits + 1, "target": first.copy(), "position": entry_point,
                "distance": distance, "yaw_error_deg": yaw_error,
                "speed": float(np.linalg.norm(self._velocity)),
                "swept": bool(np.linalg.norm(position - first) > self.config.arrival_radius)}

    def _path_guidance(self, position: np.ndarray, yaw_error: float) -> np.ndarray:
        # Only project onto the current leg. The second leg still shapes its tangent,
        # turn speed and curvature preview, but cannot skip the red target.
        starts = self._path[:self._SAMPLES - 1]
        deltas = self._path[1:self._SAMPLES] - starts
        lengths_sq = np.sum(deltas * deltas, axis=1)
        fractions = np.clip(np.sum((position - starts) * deltas, axis=1)
                            / np.maximum(lengths_sq, 1e-12), 0.0, 1.0)
        projections = starts + fractions[:, None] * deltas
        distances_sq = np.sum((position - projections) ** 2, axis=1)
        index = int(np.argmin(distances_sq))
        along = self._arc[index] + fractions[index] * math.sqrt(lengths_sq[index])
        self._progress = max(self._progress, along)
        speed = float(np.interp(self._progress, self._arc, self._speed_profile))
        lookahead = max(0.5, min(2.0, self.config.response_time_s
                                * float(np.linalg.norm(self._velocity)) + 0.4))
        first_arc = self._arc[self._SAMPLES - 1]
        aim_arc = min(first_arc, self._progress + lookahead)
        preview = min(self._SAMPLES - 2, max(0, int(np.searchsorted(self._arc, aim_arc) - 1)))
        tangent = _unit(deltas[preview], self._entry_direction)
        cross_velocity = self._velocity - tangent * np.dot(self._velocity, tangent)
        remaining = max(0.0, first_arc - self._progress)
        delta = self._targets[0] - position
        distance = float(np.linalg.norm(delta))
        tracking_error = float(np.sqrt(distances_sq[index]))
        # Reserve response distance before the arrival region, using measured
        # speed and tracking error. This is a command envelope, not a claim that
        # the physical aircraft can instantaneously satisfy it.
        brake = 0.55 * self.config.max_acceleration
        lag = self.config.response_time_s
        measured_speed = float(np.linalg.norm(self._velocity))
        # Position alone reacts too late when residual lateral inertia carries
        # the aircraft across the terminal centreline. Reserve its lagged drift.
        forecast_error = tracking_error + lag * float(np.linalg.norm(cross_velocity))
        usable = max(0.0, distance - self.config.arrival_radius
                     - forecast_error - measured_speed * lag)
        terminal = self._target_speed / (1.0 + 2.0 * forecast_error)
        speed = min(speed, math.sqrt((brake * lag + terminal) ** 2
                                    + 2.0 * brake * usable) - brake * lag)
        # Allow sufficient time to point towards the target before sphere entry.
        if abs(yaw_error) > math.radians(0.7 * self.config.heading_tolerance_deg):
            turn_time = lag + abs(yaw_error) / math.radians(self.config.max_yaw_rate_deg)
            speed = min(speed, max(0.0, distance - self.config.arrival_radius) / turn_time)
        # Measured yaw controls this slowdown; an unrealistically fast yaw command
        # cannot grant a hit or permit acceleration while the body faces away.
        alignment = max(0.0, math.cos(yaw_error))
        speed *= 0.15 + 0.85 * alignment
        # A passed, unqualified target must be recovered without counting a hit.
        # The normal entry corridor and speed envelope are intended to prevent
        # this branch during a feasible approach.
        if remaining < 0.05 and np.dot(position - self._targets[0], self._entry_direction) > 0:
            return _unit(delta) * min(speed, 1.5 * distance)
        correction = projections[index] - position
        # Tangent feedforward plus cross-track position/velocity feedback. A
        # single point-chasing direction cuts corners when the plant is lagged.
        # Correct the cross-track error within the same response/yaw envelope;
        # otherwise a large correction could restore full speed near the goal.
        return _limited(tangent * speed + 1.8 * correction - 0.35 * cross_velocity, speed)

    def _boundary_guidance(self, desired: np.ndarray, position: np.ndarray,
                           velocity: np.ndarray, dt: float) -> tuple[np.ndarray, bool]:
        result = desired.copy()
        limited = False
        # Split acceleration across axes, and reserve 0.2 s for command/plant lag.
        # The user margin remains an additional buffer outside this inner box.
        brake = self.config.max_acceleration / math.sqrt(3.0)
        reaction = max(0.2, dt)
        for axis in range(3):
            lower_distance = max(0.0, position[axis] - self.config.inner_lower[axis])
            upper_distance = max(0.0, self.config.inner_upper[axis] - position[axis])
            negative = math.sqrt((brake * reaction) ** 2 + 2 * brake * lower_distance) - brake * reaction
            positive = math.sqrt((brake * reaction) ** 2 + 2 * brake * upper_distance) - brake * reaction
            before = result[axis]
            result[axis] = np.clip(result[axis], -negative, positive)
            if velocity[axis] > positive + 1e-6:
                result[axis] = min(result[axis], max(0.0, velocity[axis] - self.config.max_acceleration * dt))
                limited = True
            elif velocity[axis] < -negative - 1e-6:
                result[axis] = max(result[axis], min(0.0, velocity[axis] + self.config.max_acceleration * dt))
                limited = True
            if position[axis] < self.config.inner_lower[axis]:
                result[axis] = max(result[axis], min(self.config.max_speed, 2 * (self.config.inner_lower[axis] - position[axis])))
                limited = True
            elif position[axis] > self.config.inner_upper[axis]:
                result[axis] = min(result[axis], -min(self.config.max_speed, 2 * (position[axis] - self.config.inner_upper[axis])))
                limited = True
            limited |= abs(result[axis] - before) > 1e-6
        return result, bool(limited)

    def update(self, position, velocity, yaw: float, dt: float) -> dict:
        if self._last_position is None:
            raise RuntimeError("call reset before update")
        position = _vector(position, "position")
        velocity = _vector(velocity, "velocity")
        if not math.isfinite(yaw) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("yaw must be finite radians and dt must be positive")
        yaw = _wrap(float(yaw))
        self._position, self._velocity = position, velocity
        hit = self._hit(position, yaw)
        if hit is not None:
            self._hits += 1
            next_target = self._targets[1]
            self._targets = [next_target, self._sample_target(next_target, [position])]
            self._plan(position, velocity)
            self._heading = self._bearing(self._targets[0] - position, yaw)
        distance = float(np.linalg.norm(self._targets[0] - position))
        passed = np.dot(position - self._targets[0], self._entry_direction) > 0.0
        # Keep the last direction only at near-coincidence. Freezing throughout
        # the arrival sphere prevents an unqualified passed target being faced
        # during recovery and can leave the aircraft stalled without a hit.
        if distance > 0.15 * self.config.arrival_radius or passed:
            self._heading = self._bearing(self._targets[0] - position, self._heading)
        error = _wrap(self._heading - yaw)
        desired = self._path_guidance(position, error)
        desired, boundary_limited = self._boundary_guidance(desired, position, velocity, dt)
        desired = _limited(desired, self.config.max_speed)
        command = self._last_command + _limited(desired - self._last_command,
                                                self.config.max_acceleration * dt)
        command = _limited(command, self.config.max_speed)
        yaw_step = math.radians(self.config.max_yaw_rate_deg) * dt
        # Slew the commanded trajectory. Measured yaw still gates speed and hits;
        # accumulating a command lead avoids under-driving a lagged yaw P loop.
        command_error = _wrap(self._heading - self._yaw_command)
        self._yaw_command = _wrap(self._yaw_command + float(np.clip(command_error, -yaw_step, yaw_step)))
        self._last_command = command
        self._last_result = self._result(hit, boundary_limited, command, yaw)
        self._last_position, self._last_yaw = position.copy(), yaw
        return self.snapshot()

    def _result(self, hit, boundary_limited, command, yaw) -> dict:
        return {"velocity_command": command.copy(),
                "yaw_command_deg": math.degrees(self._yaw_command),
                "targets": self.targets, "planned_path": self.planned_path,
                "hits": self._hits, "hit_event": hit,
                "distance": float(np.linalg.norm(self._targets[0] - self._position)),
                "yaw_error_deg": abs(math.degrees(_wrap(self._heading - yaw))),
                "planning_ms": self._planning_ms, "planning_revision": self._planning_revision,
                "bezier_parameters": self._parameters.to_dict(), "boundary_limited": boundary_limited}

    def snapshot(self) -> dict:
        if self._last_result is None:
            raise RuntimeError("call reset before snapshot")
        result = dict(self._last_result)
        result["targets"] = self.targets
        result["planned_path"] = self.planned_path
        result["velocity_command"] = self._last_command.copy()
        result["bezier_parameters"] = self._parameters.to_dict()
        if result["hit_event"] is not None:
            result["hit_event"] = {key: value.copy() if isinstance(value, np.ndarray) else value
                                   for key, value in result["hit_event"].items()}
        return result
