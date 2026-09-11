"""Target sensing ports: ground truth for baseline, camera candidates for later control."""
from dataclasses import dataclass

import airsim
import cv2
import numpy as np

from balloon_detector import detect_balloon


@dataclass(frozen=True)
class TargetObservation:
    vehicle: str
    detected: bool
    source: str
    centre_px: tuple | None = None
    bbox_xywh: tuple | None = None
    candidate_count: int = 0
    confidence: float | None = None
    normalised_error_xy: tuple | None = None
    truncated: bool = False
    depth_m: float | None = None


class GroundTruthTargetProvider:
    def __init__(self, client, target_name):
        self.client = client
        self.target_name = target_name

    def observe(self, vehicle):
        pose = self.client.simGetObjectPose(self.target_name)
        values = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
        return TargetObservation(vehicle, all(np.isfinite(values)), "ground_truth")


class CameraTargetProvider:
    def __init__(self, client, camera_name="front_center"):
        self.client = client
        self.camera_name = camera_name

    def observe(self, vehicle):
        observation, _, _ = self.observe_with_frame(vehicle)
        return observation

    def observe_with_frame(self, vehicle):
        payload = self.client.simGetImage(self.camera_name, airsim.ImageType.Scene, vehicle_name=vehicle)
        if not payload:
            return TargetObservation(vehicle, False, "camera", candidate_count=0), None, None
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return TargetObservation(vehicle, False, "camera", candidate_count=0), None, None
        result = detect_balloon(image)
        selected = result["selected"]
        observation = TargetObservation(
            vehicle, result["detected"], "camera",
            tuple(selected["centre_px"]) if selected else None,
            tuple(selected["bbox_xywh"]) if selected else None,
            result["candidate_count"], selected["shape_score"] if selected else None,
            tuple(selected["normalised_error_xy"]) if selected else None,
            bool(selected["truncated"]) if selected else False)
        return observation, image, result

    def observe_with_depth(self, vehicle):
        """Return an RGB observation plus an advisory median depth estimate."""
        observation, image, result = self.observe_with_frame(vehicle)
        if not observation.detected or observation.bbox_xywh is None:
            return observation, image, result
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective,
                                pixels_as_float=True, compress=False)
        ], vehicle_name=vehicle)
        if not responses:
            return observation, image, result
        response = responses[0]
        width, height = int(response.width), int(response.height)
        values = np.asarray(response.image_data_float, dtype=np.float32)
        if width <= 0 or height <= 0 or values.size != width * height:
            return observation, image, result
        depth = values.reshape(height, width)
        x, y, w, h = observation.bbox_xywh
        rgb_height, rgb_width = image.shape[:2]
        if rgb_width <= 0 or rgb_height <= 0:
            return observation, image, result
        scale_x, scale_y = width / rgb_width, height / rgb_height
        x, w = x * scale_x, w * scale_x
        y, h = y * scale_y, h * scale_y
        # Use the inner half of the box to reduce background pixels at its edge.
        x0 = max(0, int(x + 0.25 * w))
        x1 = min(width, max(x0 + 1, int(x + 0.75 * w)))
        y0 = max(0, int(y + 0.25 * h))
        y1 = min(height, max(y0 + 1, int(y + 0.75 * h)))
        sample = depth[y0:y1, x0:x1]
        valid = sample[np.isfinite(sample) & (sample > 0.05) & (sample < 1000.0)]
        if valid.size:
            observation = TargetObservation(**{**observation.__dict__, "depth_m": float(np.median(valid))})
        return observation, image, result
