"""Timestamped RGB/ray-depth sensing, with no target truth or flight commands.

ImageResponse.camera_position and the returned surface estimate are explicitly
VEHICLE_LOCAL_NED. A caller-audited translation is required to obtain world NED.
DepthPerspective is ray range: range * unit pixel ray is rotated by the returned
camera quaternion. This estimates a representative *visible surface point*, not
the actual balloon center or a calibrated target identity probability.

Image timestamps mark render READBACK completion, and state timestamps mark the
STATE QUERY, not exposure or the exact kinematics measurement time. Both use the
simulator clock domain. Absolute host epoch is never used. The ClockSpeed=1
development protection proxy adds the same-sim-clock readback-to-query delta to
monotonic host elapsed after state receipt, retaining both components. It is NOT
exact capture-to-execute latency or a synchronization proof. RPC durations are
recorded separately, never substituted for a readback age.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import acos, degrees, radians, tan
from numbers import Integral
from time import monotonic
from typing import Any, Callable, Sequence

import airsim
import cv2
import numpy as np

from balloon_detector import detect_balloon


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def _timestamp(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer simulator timestamp in nanoseconds")
    return int(value)


def _quaternion(response: Any) -> tuple[np.ndarray, np.ndarray]:
    value = response.camera_orientation
    raw = _vector((value.x_val, value.y_val, value.z_val, value.w_val), 4,
                  "camera_orientation_xyzw")
    norm = float(np.linalg.norm(raw))
    if not np.isfinite(norm) or abs(norm - 1.0) > 0.02:
        raise ValueError("camera quaternion is not near unit norm")
    return raw, raw / norm


def _camera_position(response: Any) -> np.ndarray:
    value = response.camera_position
    return _vector((value.x_val, value.y_val, value.z_val), 3,
                   "camera_position_local_ned_m")


def _rotate_camera_vector(vector: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = quaternion_xyzw[:3]
    cross = np.cross(q, vector)
    return vector + 2.0 * quaternion_xyzw[3] * cross + 2.0 * np.cross(q, cross)


@dataclass
class TimestampedTargetObservation:
    """Arrays/raw state are separate from the JSON metadata representation."""
    _data: dict[str, Any]
    image: np.ndarray | None = None
    depth: np.ndarray | None = None
    state_sample: Any = None
    _monotonic_clock: Callable[[], float] = monotonic

    @property
    def valid(self) -> bool:
        return self.status == "OK" and not self.rejection_reasons

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def rejection_reasons(self) -> list[str]:
        return list(self._data["rejection_reasons"])

    @property
    def vehicle(self) -> str:
        return self._data["vehicle"]

    @property
    def source(self) -> str:
        return "timestamped_rgb_depth_surface"

    @property
    def detected(self) -> bool:
        return bool(self._data["detected"])

    @property
    def truncated(self) -> bool:
        return bool(self._data["truncated"])

    @property
    def centre_px(self) -> tuple[float, float] | None:
        value = self._data["centre_px"]
        return tuple(value) if value is not None else None

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float] | None:
        value = self._data["bbox_xywh"]
        return tuple(value) if value is not None else None

    @property
    def normalised_error_xy(self) -> tuple[float, float] | None:
        value = self._data["normalised_error_xy"]
        return tuple(value) if value is not None else None

    @property
    def candidate_count(self) -> int:
        return self._data["candidate_count"]

    @property
    def confidence(self) -> float | None:
        """Legacy name for a shape score; explicitly NOT an identity probability."""
        return self._data["shape_score"]

    @property
    def depth_m(self) -> float | None:
        return self._data["depth_m"]

    @property
    def camera_position_local_ned_m(self) -> tuple[float, float, float] | None:
        value = self._data["camera_position_local_ned_m"]
        return tuple(value) if value is not None else None

    @property
    def camera_orientation_xyzw(self) -> tuple[float, float, float, float] | None:
        value = self._data["camera_orientation_xyzw"]
        return tuple(value) if value is not None else None

    @property
    def state_timestamp_ns(self) -> int | None:
        return self._data["state_timestamp_ns"]

    def metadata(self) -> dict[str, Any]:
        """Return a detached JSON-serializable record; never includes pixel arrays."""
        # Rejected host-clock diagnostics must not leak NaN/Inf into strict JSON.
        def sanitize(value):
            if isinstance(value, float) and not np.isfinite(value):
                return None
            if isinstance(value, dict):
                return {key: sanitize(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [sanitize(item) for item in value]
            return value
        return sanitize(deepcopy(self._data))

    def _reject(self, reason: str, status: str = "INVALID") -> None:
        self._data["status"] = status
        if reason not in self._data["rejection_reasons"]:
            self._data["rejection_reasons"].append(reason)

    def capture_age_at_receipt_s(self, state_timestamp: int | None = None,
                               host_receipt_monotonic_s: float | None = None) -> float:
        """Bind/check same-sim-clock readback-to-state-query age (legacy API name).

        Default uses the state read immediately after the image RPC. Rebinding
        to a different state requires its explicitly supplied host receipt time.
        These timestamps are readback/query marks, not exposure/measurement marks.
        The simulator delta is anchored at STATE receipt, not image RPC receipt.
        """
        rgb = _timestamp(self._data["rgb_timestamp_ns"], "rgb timestamp")
        depth = _timestamp(self._data["depth_timestamp_ns"], "depth timestamp")
        state = _timestamp(self.state_timestamp_ns if state_timestamp is None else state_timestamp,
                           "state timestamp")
        if state < max(rgb, depth):
            raise ValueError("SIM_CAPTURE_IN_FUTURE_OR_CLOCK_RESET")
        previous_state = self._data.get("age_anchor_state_timestamp_ns")
        if previous_state is not None and state < previous_state:
            raise ValueError("SIM_STATE_CLOCK_REGRESSION")
        if host_receipt_monotonic_s is None:
            if state != self.state_timestamp_ns:
                raise ValueError("a different state timestamp requires its host receipt anchor")
            anchor = self._data["timing"]["state_received_monotonic_s"]
        else:
            anchor = float(host_receipt_monotonic_s)
        if anchor is None or not np.isfinite(anchor):
            raise ValueError("invalid host receipt monotonic anchor")
        previous_anchor = self._data.get("age_anchor_host_monotonic_s")
        if previous_anchor is not None and anchor < previous_anchor:
            raise ValueError("HOST_RECEIPT_CLOCK_REGRESSION")
        age = (state - min(rgb, depth)) / 1e9
        self._data.update(image_readback_age_sim_s=age,
                          age_anchor_state_timestamp_ns=state,
                          age_anchor_host_monotonic_s=anchor)
        return age

    def age_at(self, host_now: float | None = None) -> float:
        """ClockSpeed=1 development proxy; not exact capture-to-execution age."""
        anchor = self._data.get("age_anchor_host_monotonic_s")
        sim_age = self._data.get("image_readback_age_sim_s")
        if anchor is None or sim_age is None or not self._data["clock_valid"]:
            raise ValueError("observation has no valid simulator/host age anchor")
        now = float(self._monotonic_clock() if host_now is None else host_now)
        if not np.isfinite(now) or now < anchor:
            raise ValueError("HOST_MONOTONIC_CLOCK_REGRESSION_OR_INVALID")
        host_delay = now - anchor
        age = float(sim_age + host_delay)
        self._data["age_components_last_evaluation"] = {
            "image_readback_age_sim_s": float(sim_age), "host_processing_delay_s": host_delay,
            "development_guard_age_s": age, "host_evaluated_monotonic_s": now,
        }
        return age

    def estimate_local_ned(self) -> tuple[float, float, float]:
        """Return the visible surface representative point in vehicle local NED."""
        if not self.valid:
            raise ValueError("observation is rejected; no accepted surface estimate")
        if self._data["coordinate_frame"] != "VEHICLE_LOCAL_NED":
            raise ValueError("camera coordinate frame is not confirmed VEHICLE_LOCAL_NED")
        return tuple(self._data["surface_position_local_ned_m"])

    def estimate_world(self, world_from_local_translation_ned_m: Sequence[float] | None = None
                       ) -> tuple[float, float, float]:
        """Apply a caller-audited pure translation; no implicit global frame claim."""
        if world_from_local_translation_ned_m is None:
            raise ValueError("world estimate requires an explicit audited local-to-world translation")
        translation = _vector(world_from_local_translation_ned_m, 3,
                              "world_from_local_translation_ned_m")
        point = np.asarray(self.estimate_local_ned()) + translation
        if not np.all(np.isfinite(point)):
            raise ValueError("world surface estimate is nonfinite")
        return tuple(float(item) for item in point)


class TimestampedTargetProvider:
    def __init__(self, client: Any, camera_name: str = "front_center",
                 horizontal_fov_deg: float = 90.0, coordinate_frame: str = "VEHICLE_LOCAL_NED",
                 max_rgb_depth_skew_s: float = 0.02, min_valid_depth_fraction: float = 0.5,
                 max_observation_age_s: float = 0.5,
                 monotonic_clock: Callable[[], float] | None = None,
                 max_camera_pose_translation_m: float = 0.02,
                 max_camera_pose_rotation_deg: float = 1.0):
        self.client = client
        self.camera_name = str(camera_name)
        self.horizontal_fov_deg = float(horizontal_fov_deg)
        self.coordinate_frame = coordinate_frame
        self.max_rgb_depth_skew_s = float(max_rgb_depth_skew_s)
        self.min_valid_depth_fraction = float(min_valid_depth_fraction)
        self.max_observation_age_s = float(max_observation_age_s)
        self.max_camera_pose_translation_m = float(max_camera_pose_translation_m)
        self.max_camera_pose_rotation_deg = float(max_camera_pose_rotation_deg)
        self.monotonic_clock = monotonic_clock or monotonic
        self._last_timestamps: dict[str, tuple[int, int, int]] = {}
        if not np.isfinite(self.horizontal_fov_deg) or not 0 < self.horizontal_fov_deg < 180:
            raise ValueError("horizontal FOV must be finite and between 0 and 180 degrees")
        for value in (self.max_rgb_depth_skew_s, self.max_observation_age_s,
                      self.max_camera_pose_translation_m, self.max_camera_pose_rotation_deg):
            if not np.isfinite(value) or value < 0:
                raise ValueError("timing and camera pose tolerances must be finite and nonnegative")
        if not np.isfinite(self.min_valid_depth_fraction) or not 0 < self.min_valid_depth_fraction <= 1:
            raise ValueError("min_valid_depth_fraction must be in (0,1]")

    def reset_clock_tracking(self, vehicle: str | None = None) -> None:
        """Explicitly acknowledge a simulator reset before accepting its new epoch."""
        if vehicle is None:
            self._last_timestamps.clear()
        else:
            self._last_timestamps.pop(vehicle, None)

    def _empty(self, vehicle: str) -> TimestampedTargetObservation:
        data = {
            "status": "INVALID", "rejection_reasons": [], "vehicle": vehicle,
            "camera_name": self.camera_name, "coordinate_frame": self.coordinate_frame,
            "source": "timestamped_rgb_depth_surface", "detected": False,
            "truncated": False, "centre_px": None, "bbox_xywh": None,
            "normalised_error_xy": None, "candidate_count": 0, "shape_score": None,
            "confidence_semantics": "uncalibrated_color_ellipse_shape_score_not_identity_probability",
            "depth_m": None, "depth_semantics": "DepthPerspective_range_along_pixel_ray",
            "rgb_timestamp_ns": None, "depth_timestamp_ns": None, "state_timestamp_ns": None,
            "rgb_render_readback_timestamp_ns": None, "depth_render_readback_timestamp_ns": None,
            "query_state_timestamp_ns": None,
            "timestamp_semantics": "image_render_readback_completion_and_state_query_not_exposure_or_measurement",
            "state_sample_semantics": "simulation_self_state; SimpleFlight_kinematics_estimated_uses_simulation_kinematics",
            "rgb_depth_skew_s": None, "clock_valid": False,
            "clock_domains": {"images_and_state": "AirSim_ClockFactory_simulator_nanoseconds",
                              "request_receive_inference": "host_monotonic_seconds",
                              "host_epoch_used": False,
                              "age_combination": "same_sim_delta_plus_host_delay_after_state_receipt",
                              "scope": "ClockSpeed_1_development_protection_proxy_not_capture_to_execute_latency",
                              "assumed_clock_speed": 1.0},
            "timing": {"image_request_monotonic_s": None, "image_received_monotonic_s": None,
                       "state_request_monotonic_s": None, "state_received_monotonic_s": None,
                       "inference_started_monotonic_s": None, "inference_finished_monotonic_s": None,
                       "image_rpc_duration_s": None, "state_rpc_duration_s": None,
                       "inference_duration_s": None},
            "camera_position_local_ned_m": None, "camera_orientation_xyzw": None,
            "rgb_camera_pose": None, "depth_camera_pose": None,
            "rgb_depth_pose_translation_m": None, "rgb_depth_pose_rotation_deg": None,
            "depth_statistics": None, "projection": None,
            "surface_position_local_ned_m": None,
            "point_semantics": "visible_surface_representative_point_not_balloon_center",
            "image_readback_age_sim_s": None,
            "age_anchor_state_timestamp_ns": None, "age_anchor_host_monotonic_s": None,
            "age_components_last_evaluation": None,
        }
        return TimestampedTargetObservation(data, _monotonic_clock=self.monotonic_clock)

    def capture(self, vehicle: str) -> TimestampedTargetObservation:
        """Get one paired image request, then state clock, then RGB inference."""
        observation = self._empty(vehicle)
        data, timing = observation._data, observation._data["timing"]
        timing["image_request_monotonic_s"] = float(self.monotonic_clock())
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, True),
                airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False),
            ], vehicle_name=vehicle)
            timing["image_received_monotonic_s"] = float(self.monotonic_clock())
            timing["state_request_monotonic_s"] = float(self.monotonic_clock())
            observation.state_sample = self.client.getMultirotorState(vehicle_name=vehicle)
            timing["state_received_monotonic_s"] = float(self.monotonic_clock())
        except Exception as error:
            observation._reject(f"SENSOR_RPC_FAILURE:{type(error).__name__}:{error}")
            return observation
        timing["image_rpc_duration_s"] = timing["image_received_monotonic_s"] - timing["image_request_monotonic_s"]
        timing["state_rpc_duration_s"] = timing["state_received_monotonic_s"] - timing["state_request_monotonic_s"]
        host_times = list(timing[key] for key in ("image_request_monotonic_s", "image_received_monotonic_s",
                                                "state_request_monotonic_s", "state_received_monotonic_s"))
        if not all(np.isfinite(value) for value in host_times) or any(a > b for a, b in zip(host_times, host_times[1:])):
            observation._reject("HOST_MONOTONIC_CLOCK_REGRESSION_OR_INVALID")
            return observation
        if not isinstance(responses, (list, tuple)) or len(responses) != 2:
            observation._reject("PAIRED_IMAGE_RESPONSE_COUNT_INVALID")
            return observation
        rgb_response, depth_response = responses
        try:
            if rgb_response.image_type != airsim.ImageType.Scene or depth_response.image_type != airsim.ImageType.DepthPerspective:
                raise ValueError("PAIRED_IMAGE_TYPES_INVALID")
            rgb_timestamp = _timestamp(rgb_response.time_stamp, "rgb timestamp")
            depth_timestamp = _timestamp(depth_response.time_stamp, "depth timestamp")
            state_timestamp = _timestamp(observation.state_sample.timestamp, "state timestamp")
            data.update(rgb_timestamp_ns=rgb_timestamp, depth_timestamp_ns=depth_timestamp,
                        state_timestamp_ns=state_timestamp,
                        rgb_render_readback_timestamp_ns=rgb_timestamp,
                        depth_render_readback_timestamp_ns=depth_timestamp,
                        query_state_timestamp_ns=state_timestamp,
                        rgb_depth_skew_s=abs(rgb_timestamp - depth_timestamp) / 1e9)
            previous = self._last_timestamps.get(vehicle)
            current = (rgb_timestamp, depth_timestamp, state_timestamp)
            if previous is not None and any(now < before for now, before in zip(current, previous)):
                raise ValueError("SIM_CLOCK_RESET_OR_REGRESSION_REQUIRES_EXPLICIT_ACKNOWLEDGEMENT")
            self._last_timestamps[vehicle] = current
            observation.capture_age_at_receipt_s()
            data["clock_valid"] = True
            if data["rgb_depth_skew_s"] > self.max_rgb_depth_skew_s:
                raise ValueError("RGB_DEPTH_TIMESTAMP_SKEW_EXCEEDED")
            rgb_position, depth_position = _camera_position(rgb_response), _camera_position(depth_response)
            rgb_raw_q, rgb_q = _quaternion(rgb_response)
            depth_raw_q, depth_q = _quaternion(depth_response)
            pose_delta = float(np.linalg.norm(rgb_position - depth_position))
            angle_delta = degrees(2 * acos(float(np.clip(abs(np.dot(rgb_q, depth_q)), 0.0, 1.0))))
            data.update(camera_position_local_ned_m=depth_position.tolist(),
                        camera_orientation_xyzw=depth_raw_q.tolist(),
                        rgb_camera_pose={"position_local_ned_m": rgb_position.tolist(), "orientation_xyzw": rgb_raw_q.tolist()},
                        depth_camera_pose={"position_local_ned_m": depth_position.tolist(), "orientation_xyzw": depth_raw_q.tolist()},
                        rgb_depth_pose_translation_m=pose_delta, rgb_depth_pose_rotation_deg=angle_delta)
            if pose_delta > self.max_camera_pose_translation_m or angle_delta > self.max_camera_pose_rotation_deg:
                raise ValueError("RGB_DEPTH_CAMERA_POSE_SKEW_EXCEEDED")
            if self.coordinate_frame != "VEHICLE_LOCAL_NED":
                raise ValueError("CAMERA_COORDINATE_FRAME_UNCONFIRMED")
            if not rgb_response.compress or rgb_response.pixels_as_float:
                raise ValueError("RGB_RESPONSE_FORMAT_INVALID_EXPECT_COMPRESSED_PNG")
            encoded = rgb_response.image_data_uint8
            buffer = np.frombuffer(encoded, np.uint8) if isinstance(encoded, (bytes, bytearray)) else np.asarray(encoded, dtype=np.uint8)
            if not buffer.size:
                raise ValueError("RGB_IMAGE_EMPTY")
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise ValueError("RGB_IMAGE_DECODE_FAILED")
            rgb_width, rgb_height = int(rgb_response.width), int(rgb_response.height)
            if rgb_width <= 0 or rgb_height <= 0 or image.shape[:2] != (rgb_height, rgb_width):
                raise ValueError("RGB_RESPONSE_DIMENSIONS_INVALID")
            observation.image = image
            width, height = int(depth_response.width), int(depth_response.height)
            if not depth_response.pixels_as_float or depth_response.compress:
                raise ValueError("DEPTH_RESPONSE_FORMAT_INVALID_EXPECT_FLOAT_RAY_RANGE")
            values = np.asarray(depth_response.image_data_float, dtype=float)
            if width <= 0 or height <= 0 or values.ndim != 1 or values.size != width * height:
                raise ValueError("DEPTH_RESPONSE_DIMENSIONS_INVALID")
            if (width, height) != (rgb_width, rgb_height):
                raise ValueError("RGB_DEPTH_RESOLUTION_MISMATCH_REQUIRES_AUDITED_NORMALIZED_INTRINSICS")
            observation.depth = values.reshape(height, width)
        except (ValueError, TypeError, AttributeError, cv2.error, OverflowError) as error:
            observation._reject(str(error))
            return observation

        timing["inference_started_monotonic_s"] = float(self.monotonic_clock())
        try:
            detection = detect_balloon(observation.image)
            selected = detection["selected"]
            data.update(detected=bool(detection["detected"]), candidate_count=int(detection["candidate_count"]))
            if not data["detected"] or selected is None:
                observation._reject("NO_BALLOON_COLOR_ELLIPSE_DETECTION", "NO_DETECTION")
            else:
                box = _vector(selected["bbox_xywh"], 4, "bbox_xywh")
                centre = _vector(selected["centre_px"], 2, "centre_px")
                error_xy = _vector(selected["normalised_error_xy"], 2, "normalised_error_xy")
                shape_score = float(selected["shape_score"])
                if not np.isfinite(shape_score):
                    raise ValueError("shape score is nonfinite")
                data.update(centre_px=centre.tolist(), bbox_xywh=box.tolist(),
                            normalised_error_xy=error_xy.tolist(), shape_score=shape_score,
                            truncated=bool(selected["truncated"]))
                x, y, w, h = box
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > rgb_width or y + h > rgb_height:
                    raise ValueError("DETECTION_BOX_INVALID")
                if data["truncated"]:
                    raise ValueError("TRUNCATED_DETECTION_REJECTED")
                scale_x, scale_y = width / rgb_width, height / rgb_height
                x0, x1 = max(0, int(np.floor((x + .25*w)*scale_x))), min(width, int(np.ceil((x + .75*w)*scale_x)))
                y0, y1 = max(0, int(np.floor((y + .25*h)*scale_y))), min(height, int(np.ceil((y + .75*h)*scale_y)))
                region = observation.depth[y0:y1, x0:x1]
                valid = np.isfinite(region) & (region > .05) & (region < 1000.0)
                count = int(np.count_nonzero(valid))
                fraction = count / region.size if region.size else 0.0
                data["depth_statistics"] = {"inner_bbox_depth_xyxy": [x0, y0, x1, y1],
                                            "pixel_count": int(region.size), "valid_count": count,
                                            "valid_fraction": fraction,
                                            "accepted_range_open_interval_m": [.05, 1000.0],
                                            "median_range_m": None, "representative_pixel_xy": None}
                if count == 0 or fraction < self.min_valid_depth_fraction:
                    observation._reject("INSUFFICIENT_VALID_INNER_BOX_DEPTH", "NO_VALID_DEPTH")
                else:
                    ys, xs = np.nonzero(valid)
                    ranges = region[valid]
                    median = float(np.median(ranges))
                    # Select an ACTUAL pixel's range, not median range on a different center ray.
                    cx_depth = (centre[0] + .5) * scale_x - .5
                    cy_depth = (centre[1] + .5) * scale_y - .5
                    distance_to_centre = (xs + x0 - cx_depth)**2 + (ys + y0 - cy_depth)**2
                    choice = int(np.lexsort((xs, ys, distance_to_centre, np.abs(ranges - median)))[0])
                    u, v = int(xs[choice] + x0), int(ys[choice] + y0)
                    ray_range = float(ranges[choice])
                    fx = width / (2 * tan(radians(self.horizontal_fov_deg) / 2))
                    # Same-FOV rectified pinhole images and square pixels are an explicit assumption.
                    principal_x, principal_y = (width - 1) / 2, (height - 1) / 2
                    direction = np.array([1.0, (u - principal_x) / fx, (v - principal_y) / fx])
                    unit_direction = direction / np.linalg.norm(direction)
                    camera_point = unit_direction * ray_range
                    local_point = depth_position + _rotate_camera_vector(camera_point, depth_q)
                    if not np.all(np.isfinite(local_point)):
                        raise ValueError("SURFACE_POSITION_NONFINITE")
                    data.update(depth_m=ray_range, surface_position_local_ned_m=local_point.tolist())
                    data["depth_statistics"].update(median_range_m=median, representative_pixel_xy=[u, v])
                    data["projection"] = {"depth_size_wh": [width, height], "rgb_size_wh": [rgb_width, rgb_height],
                                          "horizontal_fov_deg": self.horizontal_fov_deg,
                                          "focal_lengths_px": [fx, fx], "principal_pixel_xy": [principal_x, principal_y],
                                          "camera_axes": "forward_x_right_y_down_z",
                                          "unit_ray_camera": unit_direction.tolist(),
                                          "point_camera_m": camera_point.tolist(),
                                          "assumption": "same_resolution_rectified_RGB_depth_same_audited_horizontal_FOV_square_pixels",
                                          "pose_quaternion_normalized_for_rotation": depth_q.tolist()}
                    data["status"] = "OK"
        except (ValueError, TypeError, AttributeError, KeyError, cv2.error, OverflowError) as error:
            observation._reject(f"INFERENCE_OR_PROJECTION_INVALID:{error}")
        finally:
            timing["inference_finished_monotonic_s"] = float(self.monotonic_clock())
            timing["inference_duration_s"] = timing["inference_finished_monotonic_s"] - timing["inference_started_monotonic_s"]
        if not np.isfinite(timing["inference_duration_s"]) or timing["inference_duration_s"] < 0:
            observation._reject("HOST_INFERENCE_CLOCK_REGRESSION_OR_INVALID")
            return observation
        try:
            age = observation.age_at(timing["inference_finished_monotonic_s"])
            data["operational_age_at_inference_completion_s"] = age
            if age > self.max_observation_age_s:
                observation._reject("OBSERVATION_STALE_AT_INFERENCE_COMPLETION", "STALE")
        except ValueError as error:
            observation._reject(str(error))
        return observation
