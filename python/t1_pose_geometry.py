"""Pose-aware Scene pinhole geometry for T1; no simulator/control dependency.

Camera vectors use forward X, right Y, down Z. Camera orientation is xyzw,
camera-to-NED. AirSim camera positions are vehicle-local NED, so every world
operation requires an explicit, audited world-minus-local translation.

Scene K uses (width / 2, height / 2), as in camera_geometry.scene_intrinsics.
Projection/roundtrip consistency is a software check, not empirical image
calibration. Footprints below predict field-of-view inclusion only: they do
not measure occlusion, identity confidence, or a physical target's shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np

from camera_geometry import scene_intrinsics


def _finite(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def quaternion_rotation_xyzw(value: Sequence[float]) -> np.ndarray:
    """Camera-to-NED rotation; normalize only already-near-unit input."""
    raw = _finite(value, 4, "camera_orientation_xyzw")
    norm = float(np.linalg.norm(raw))
    if abs(norm - 1.0) > .02:
        raise ValueError("camera quaternion must be within .02 of unit norm")
    x, y, z, w = raw / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


@dataclass(frozen=True)
class SceneCameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_depth_m: float = .05

    def __post_init__(self):
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n <= 0
               for n in (self.width, self.height)):
            raise ValueError("image dimensions must be positive integers")
        values = np.asarray([self.fx, self.fy, self.cx, self.cy, self.near_depth_m], float)
        if not np.isfinite(values).all() or min(self.fx, self.fy, self.near_depth_m) <= 0:
            raise ValueError("intrinsics and positive near depth must be finite")
        if not np.isclose(self.cx, self.width/2, rtol=0, atol=1e-7) or not np.isclose(
                self.cy, self.height/2, rtol=0, atol=1e-7):
            raise ValueError("Scene principal pixel must be width/2,height/2; legacy halfpixel K is incompatible")
        if not np.isclose(self.fx, self.fy, rtol=1e-4, atol=0):
            raise ValueError("this Scene contract requires audited square pixels")

    @classmethod
    def from_camera_info(cls, camera_info, width: int, height: int, near_depth_m=.05):
        """Read the Scene projection, rather than CinemAirSim viewer FOV."""
        return cls.from_k(scene_intrinsics(camera_info, width, height)["K_nominal"],
                          width, height, near_depth_m)

    @classmethod
    def from_k(cls, matrix, width: int, height: int, near_depth_m=.05):
        k = np.asarray(matrix, dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise ValueError("K must be a finite 3x3 matrix")
        expected = np.array([[k[0, 0], 0., k[0, 2]], [0., k[1, 1], k[1, 2]], [0., 0., 1.]])
        if not np.allclose(k, expected, rtol=0, atol=1e-9):
            raise ValueError("skewed or non-pinhole K is unsupported")
        return cls(width, height, *map(float, (k[0, 0], k[1, 1], k[0, 2], k[1, 2])),
                   near_depth_m=float(near_depth_m))

    def metadata(self) -> dict:
        return {"K_nominal": [[self.fx, 0., self.cx], [0., self.fy, self.cy], [0., 0., 1.]],
                "image_size_wh": [int(self.width), int(self.height)],
                "principal_pixel_convention": "Scene_width_over_2_height_over_2",
                "camera_axes": "forward_x_right_y_down_z", "near_depth_m": self.near_depth_m,
                "semantics": "Scene_renderer_pinhole_metadata_not_empirical_calibration"}


@dataclass(frozen=True)
class CameraPose:
    position_local_ned_m: Sequence[float]
    orientation_xyzw: Sequence[float]
    world_from_local_translation_ned_m: Sequence[float]

    def __post_init__(self):
        object.__setattr__(self, "position_local_ned_m", tuple(_finite(
            self.position_local_ned_m, 3, "position_local_ned_m")))
        quaternion_rotation_xyzw(self.orientation_xyzw)
        object.__setattr__(self, "orientation_xyzw", tuple(_finite(
            self.orientation_xyzw, 4, "orientation_xyzw")))
        object.__setattr__(self, "world_from_local_translation_ned_m", tuple(_finite(
            self.world_from_local_translation_ned_m, 3, "world_from_local_translation_ned_m")))

    @property
    def position_world_ned_m(self) -> np.ndarray:
        return np.asarray(self.position_local_ned_m) + self.world_from_local_translation_ned_m

    @property
    def rotation_camera_to_ned(self) -> np.ndarray:
        return quaternion_rotation_xyzw(self.orientation_xyzw)

    def camera_to_world(self, point_camera_m: Sequence[float]) -> np.ndarray:
        result = self.position_world_ned_m + self.rotation_camera_to_ned @ _finite(
            point_camera_m, 3, "point_camera_m")
        if not np.isfinite(result).all():
            raise ValueError("camera-to-world transform overflowed or became nonfinite")
        return result

    def world_to_camera(self, point_world_ned_m: Sequence[float]) -> np.ndarray:
        result = self.rotation_camera_to_ned.T @ (_finite(point_world_ned_m, 3,
            "point_world_ned_m") - self.position_world_ned_m)
        if not np.isfinite(result).all():
            raise ValueError("world-to-camera transform overflowed or became nonfinite")
        return result


def project_world(point_world_ned_m: Sequence[float], pose: CameraPose,
                  camera: SceneCameraModel) -> dict:
    relative = pose.world_to_camera(point_world_ned_m)
    axial = float(relative[0])
    if axial <= camera.near_depth_m:
        raise ValueError("point is behind or on the declared camera near-depth plane")
    uv = np.array([camera.cx + camera.fx*relative[1]/axial,
                   camera.cy + camera.fy*relative[2]/axial])
    if not np.isfinite(uv).all():
        raise ValueError("projected pixel is nonfinite")
    normalized = (uv - [camera.cx, camera.cy]) / [camera.width/2, camera.height/2]
    return {"pixel_xy": uv.tolist(), "point_camera_m": relative.tolist(),
            "axial_depth_m": axial, "ray_range_m": float(np.linalg.norm(relative)),
            "inside_image": bool(0 <= uv[0] < camera.width and 0 <= uv[1] < camera.height),
            "image_edge_margin_px": float(min(uv[0], uv[1], camera.width-uv[0], camera.height-uv[1])),
            "normalized_center_error_xy": normalized.tolist(),
            "centering_squared_error": float(normalized @ normalized),
            "semantics": "pose_aware_pinhole_field_geometry_not_observed_visibility_or_identity"}


def unproject_range(pixel_xy: Sequence[float], ray_range_m: float, pose: CameraPose,
                    camera: SceneCameraModel) -> np.ndarray:
    """DepthPerspective ray range, not axial depth; pixels may extrapolate."""
    pixel = _finite(pixel_xy, 2, "pixel_xy")
    depth = float(ray_range_m)
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("ray range must be finite and positive")
    ray = np.array([1., (pixel[0]-camera.cx)/camera.fx, (pixel[1]-camera.cy)/camera.fy])
    point = ray * (depth / np.linalg.norm(ray))
    if point[0] <= camera.near_depth_m:
        raise ValueError("unprojected point is on or behind near-depth plane")
    return pose.camera_to_world(point)


def unproject_axial(pixel_xy: Sequence[float], axial_depth_m: float, pose: CameraPose,
                    camera: SceneCameraModel) -> np.ndarray:
    pixel = _finite(pixel_xy, 2, "pixel_xy")
    depth = float(axial_depth_m)
    if not np.isfinite(depth) or depth <= camera.near_depth_m:
        raise ValueError("axial depth must be finite and beyond near-depth plane")
    return pose.camera_to_world([depth, (pixel[0]-camera.cx)*depth/camera.fx,
                                (pixel[1]-camera.cy)*depth/camera.fy])


def rgb_reference_on_surface_depth(centroid_pixel_xy, surface_world_ned_m,
                                   pose: CameraPose, camera: SceneCameraModel) -> dict:
    """RGB bearing at the observed surface axial depth, not a target center."""
    surface = project_world(surface_world_ned_m, pose, camera)
    reference = unproject_axial(centroid_pixel_xy, surface["axial_depth_m"], pose, camera)
    return {"reference_world_ned_m": reference.tolist(),
            "offset_from_surface_ned_m": (reference-_finite(surface_world_ned_m, 3, "surface")).tolist(),
            "surface_anchor_pixel_xy": surface["pixel_xy"],
            "axial_plane_depth_m": surface["axial_depth_m"],
            "semantics": "RGB_centroid_bearing_on_representative_surface_depth_plane_not_physical_target_center"}


def bbox_plane_corners_world(bbox_xywh, axial_depth_m, pose: CameraPose,
                             camera: SceneCameraModel) -> np.ndarray:
    """Freeze a four-corner observed-box plane for candidate-pose scoring."""
    x, y, width, height = _finite(bbox_xywh, 4, "bbox_xywh")
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x+width > camera.width or y+height > camera.height:
        raise ValueError("bbox must be positive and contained in the source image")
    return np.array([unproject_axial(pixel, axial_depth_m, pose, camera)
                     for pixel in ((x, y), (x+width, y), (x+width, y+height), (x, y+height))])


def _area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.
    return abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                     np.dot(points[:, 1], np.roll(points[:, 0], -1)))) / 2


def _clip(points, axis, boundary, keep_greater):
    output = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        a = start[axis] >= boundary if keep_greater else start[axis] <= boundary
        b = end[axis] >= boundary if keep_greater else end[axis] <= boundary
        if a != b:
            weight = (boundary-start[axis])/(end[axis]-start[axis])
            intersection = start + weight*(end-start)
            if a:
                output.append(intersection)
            else:
                output.extend([intersection, end])
        elif b:
            output.append(end)
    return np.asarray(output, dtype=float).reshape(-1, 2)


def project_footprint_polygon(points_world_ned_m, pose: CameraPose,
                              camera: SceneCameraModel) -> dict:
    """Convex projected polygon's in-frame area fraction; near crossings reject.

    A box plane is a geometric proxy of the observed box, not an occlusion model
    or a reconstructed balloon volume. Rejecting a near-plane crossing is a
    conservative candidate rejection, not an assertion that no part is visible.
    """
    world = np.asarray(points_world_ned_m, dtype=float)
    if world.ndim != 2 or world.shape[1] != 3 or len(world) < 3 or not np.isfinite(world).all():
        raise ValueError("footprint must contain at least three finite ordered 3D vertices")
    rows = [project_world(point, pose, camera) for point in world]
    pixels = np.array([row["pixel_xy"] for row in rows])
    edges = np.roll(pixels, -1, axis=0)-pixels
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0]*next_edges[:, 1]-edges[:, 1]*next_edges[:, 0]
    if _area(pixels) <= 1e-12 or not (np.all(crosses > 1e-12) or np.all(crosses < -1e-12)):
        raise ValueError("footprint must project to a nondegenerate strictly convex ordered polygon")
    # Consecutive turns alone also accept a self-intersecting regular star.
    # Every other vertex must occupy the same inside halfplane of every edge.
    orientation = 1 if crosses[0] > 0 else -1
    for start, edge in zip(pixels, edges):
        relative = pixels-start
        signed = orientation*(edge[0]*relative[:, 1]-edge[1]*relative[:, 0])
        if np.any(signed < -1e-10):
            raise ValueError("footprint order is self-intersecting or not convex")
    clipped = pixels
    for axis, boundary, keep in ((0, 0., True), (0, camera.width, False),
                                 (1, 0., True), (1, camera.height, False)):
        clipped = _clip(clipped, axis, boundary, keep) if len(clipped) else clipped
    total, inside = _area(pixels), _area(clipped)
    return {"projected_vertices_pixel_xy": pixels.tolist(),
            "clipped_vertices_pixel_xy": clipped.tolist(),
            "projected_area_px2": total, "inside_area_px2": inside,
            "field_area_fraction": float(np.clip(inside/total, 0., 1.)),
            "all_vertices_in_image": all(row["inside_image"] for row in rows),
            "minimum_vertex_edge_margin_px": min(row["image_edge_margin_px"] for row in rows),
            "semantics": "projected_polygon_field_fraction_not_occlusion_or_identity_probability",
            "near_plane_policy": "reject_any_vertex_on_or_behind_near_plane"}
