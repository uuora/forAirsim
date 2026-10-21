"""Batch the existing frozen-orientation trajectory projection geometry.

Only arithmetic and Python-loop overhead change. Reference points, ordered box
planes, camera-to-NED rotations, near-plane rejection and partial clipping have
the same meanings as t1_pose_geometry. An in-frame polygon still must be
nondegenerate, strictly convex and correctly ordered before its area fraction
can be set to one. There is no simulator, detector, identity or command API.
"""
from __future__ import annotations

import numpy as np

from t1_pose_geometry import CameraPose, SceneCameraModel, project_footprint_polygon


def _finite(value, shape, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain finite coordinates") from None
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


def score_pose_path(reference_world, corners_world, positions_ned, origin_body_ned,
                    camera_pose, camera):
    """Score exactly the original translated-camera geometry at every sample.

    Camera local position is ``camera_pose.local + positions - origin_body``.
    Its measured orientation and explicit local-to-world translation remain
    fixed. Every reference and polygon vertex must lie beyond the declared
    near plane at every sample; crossing rejects the whole candidate rather
    than clamping depth. Partial/outside polygons call the original clipping
    implementation. Returned lists retain the input sample order.
    """
    if not isinstance(camera_pose, CameraPose) or not isinstance(camera, SceneCameraModel):
        raise ValueError("camera pose and Scene camera model are required")
    reference = _finite(reference_world, (3,), "reference_world")
    origin = _finite(origin_body_ned, (3,), "origin_body_ned")
    try:
        positions = np.asarray(positions_ned, dtype=float)
        corners = np.asarray(corners_world, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("path and footprint must contain finite coordinates") from None
    if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0 or not np.isfinite(positions).all():
        raise ValueError("positions_ned must contain at least one finite 3D sample")
    if corners.ndim != 2 or corners.shape[1:] != (3,) or len(corners) < 3 or not np.isfinite(corners).all():
        raise ValueError("footprint must contain at least three finite ordered 3D vertices")
    # Preserve the original operation order (local+delta, then translation),
    # rather than changing floating-point summation with a world+delta shortcut.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        camera_local_paths = np.asarray(camera_pose.position_local_ned_m)+(positions-origin)
        camera_world_paths = camera_local_paths+camera_pose.world_from_local_translation_ned_m
        rotation = camera_pose.rotation_camera_to_ned
        relative = (reference-camera_world_paths) @ rotation
        relative_corners = (corners[None, :, :]-camera_world_paths[:, None, :]) @ rotation
    if not all(np.isfinite(value).all() for value in (camera_local_paths, camera_world_paths,
                                                     relative, relative_corners)):
        raise ValueError("world-to-camera transform overflowed or became nonfinite")
    if np.any(relative[:, 0] <= camera.near_depth_m) or np.any(relative_corners[:, :, 0] <= camera.near_depth_m):
        raise ValueError("point is behind or on the declared camera near-depth plane")
    focal = np.array([camera.fx, camera.fy])
    principal = np.array([camera.cx, camera.cy])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels = principal+focal*relative[:, 1:]/relative[:, :1]
        polygon_pixels = principal+focal*relative_corners[:, :, 1:]/relative_corners[:, :, :1]
    if not np.isfinite(pixels).all() or not np.isfinite(polygon_pixels).all():
        raise ValueError("projected pixel is nonfinite")
    normalized = (pixels-principal)/np.array([camera.width/2, camera.height/2])
    center_errors = np.sum(normalized*normalized, axis=1)
    ranges = np.linalg.norm(relative, axis=1)

    # Match the original strict-convexity and edge-halfplane checks even when
    # all vertices lie in-frame. Simply testing the vertices would incorrectly
    # accept degenerate boxes, bowties and regular self-intersecting stars.
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.roll(polygon_pixels, -1, axis=1)-polygon_pixels
        next_edges = np.roll(edges, -1, axis=1)
        crosses = edges[:, :, 0]*next_edges[:, :, 1]-edges[:, :, 1]*next_edges[:, :, 0]
        areas = np.abs(np.sum(polygon_pixels[:, :, 0]*np.roll(polygon_pixels[:, :, 1], -1, axis=1), axis=1)
                       -np.sum(polygon_pixels[:, :, 1]*np.roll(polygon_pixels[:, :, 0], -1, axis=1), axis=1))/2
        orientation = np.where(crosses[:, 0] > 0, 1.0, -1.0)
        vertex_offsets = polygon_pixels[:, None, :, :]-polygon_pixels[:, :, None, :]
        signed = orientation[:, None, None]*(edges[:, :, None, 0]*vertex_offsets[:, :, :, 1]
                                            -edges[:, :, None, 1]*vertex_offsets[:, :, :, 0])
    if not all(np.isfinite(value).all() for value in (center_errors, ranges, crosses, areas, signed)):
        raise ValueError("projection scores or polygon checks became nonfinite")
    turns_valid = np.all(crosses > 1e-12, axis=1) | np.all(crosses < -1e-12, axis=1)
    if np.any(areas <= 1e-12) or not np.all(turns_valid):
        raise ValueError("footprint must project to a nondegenerate strictly convex ordered polygon")
    if np.any(signed < -1e-10):
        raise ValueError("footprint order is self-intersecting or not convex")

    inside = ((polygon_pixels[:, :, 0] >= 0) & (polygon_pixels[:, :, 0] < camera.width)
              & (polygon_pixels[:, :, 1] >= 0) & (polygon_pixels[:, :, 1] < camera.height))
    all_inside = np.all(inside, axis=1)
    fractions = np.ones(len(positions), dtype=float)
    for index in np.flatnonzero(~all_inside):
        pose = CameraPose(camera_local_paths[index], camera_pose.orientation_xyzw,
                          camera_pose.world_from_local_translation_ned_m)
        fractions[index] = project_footprint_polygon(corners, pose, camera)["field_area_fraction"]
    return {"centering_squared_errors": center_errors.tolist(),
            "ray_ranges_m": ranges.tolist(), "field_area_fractions": fractions.tolist()}
