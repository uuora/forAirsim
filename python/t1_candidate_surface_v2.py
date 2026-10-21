"""Temporal selection to paired DepthPerspective geometry, with no flight API.

V2 accepts the explicit temporal certificate without translating it into a
highest-score detector or fresh semantic result. Geometry below is copied from
V1 candidate_surface_estimate and preserves actual-pixel selection and pairing
limits. No target actor truth or segmentation is an input.
"""
from __future__ import annotations

import hashlib
import json
from numbers import Integral, Real
from typing import Any

import numpy as np

from t1_candidate_surface import SurfaceConfig, SurfaceRejection, _number, _stamp
from t1_pose_geometry import (CameraPose, SceneCameraModel, bbox_plane_corners_world,
    project_world, rgb_reference_on_surface_depth, unproject_range)
from t1_semantic_perception import SemanticFrame

GEOMETRY_SOURCE_VERSION = "t1_candidate_surface_v1_actual_depth_pixel"
GEOMETRY_SOURCE_SHA256 = "d9fb6bf0127e356f7dbb4c43908778a1f8f403a96ff94534e3123fea040e5d91"


def _temporal_selected(frame, certificate, description, now, current_max_age,
                       semantic_max_age, unreviewed_max_age):
    if not isinstance(frame, SemanticFrame) or not isinstance(certificate, dict):
        raise SurfaceRejection("INVALID_FRAME_OR_SELECTION_CERTIFICATE")
    if certificate.get("status") != "TEMPORAL_LOCAL_ASSOCIATION_VALID":
        raise SurfaceRejection("TEMPORAL_SELECTION_CERTIFICATE_NOT_ACCEPTED")
    if certificate.get("state") != "TRACKING":
        raise SurfaceRejection("TEMPORAL_SELECTION_STATE_NOT_TRACKING")
    if not isinstance(description, str) or not description.strip():
        raise SurfaceRejection("INVALID_CURRENT_TASK_DESCRIPTION")
    description_sha = hashlib.sha256(description.encode("utf-8")).hexdigest()
    expected = {"session_id": frame.session_id, "target_generation": frame.target_generation,
        "description_sha256": description_sha, "current_frame_id": frame.frame_id,
        "current_frame_sequence": frame.frame_sequence, "current_image_sha256": frame.image_sha256}
    for key in ("target_generation", "current_frame_sequence"):
        if isinstance(certificate.get(key), bool) or not isinstance(certificate.get(key), Integral):
            raise SurfaceRejection("INVALID_SELECTION_CONTEXT_INTEGER")
    if any(certificate.get(key) != value for key, value in expected.items()):
        raise SurfaceRejection("SELECTION_CURRENT_FRAME_BINDING_MISMATCH")
    expected_binding = {"session_id": frame.session_id, "target_generation": frame.target_generation,
        "description_sha256": description_sha, "frame_id": frame.frame_id,
        "frame_sequence": frame.frame_sequence, "image_sha256": frame.image_sha256,
        "readback_receipt_monotonic_s": frame.readback_receipt_monotonic_s,
        "candidates_ready_monotonic_s": frame.candidates_ready_monotonic_s}
    binding = certificate.get("current_binding")
    if not isinstance(binding, dict) or set(binding) != set(expected_binding):
        raise SurfaceRejection("INCOMPLETE_TEMPORAL_CURRENT_BINDING")
    for key in ("target_generation", "frame_sequence"):
        if isinstance(binding[key], bool) or not isinstance(binding[key], Integral):
            raise SurfaceRejection("INVALID_TEMPORAL_BINDING_INTEGER")
    for key in ("readback_receipt_monotonic_s", "candidates_ready_monotonic_s"):
        _number(binding[key], key.upper())
    if binding != expected_binding:
        raise SurfaceRejection("TEMPORAL_CURRENT_BINDING_MISMATCH")
    flags = ("identity_probability_calibrated", "instance_identity_guaranteed",
             "appearance_continuity_verified", "flight_authorized")
    if any(certificate.get(key) is not False for key in flags):
        raise SurfaceRejection("SELECTION_CERTIFICATE_EXCEEDS_EVIDENCE_SCOPE")
    if certificate.get("semantic_binding_provenance") != "CLIENT_SIDE_BINDING; NOT SERVER_ECHO":
        raise SurfaceRejection("TEMPORAL_CLIENT_BINDING_PROVENANCE_MISMATCH")
    for key in ("semantic_anchor", "track_id"):
        if not isinstance(certificate.get(key), str) or not certificate[key].strip():
            raise SurfaceRejection("MISSING_TEMPORAL_ANCHOR_OR_TRACK")
    now = _number(now, "CURRENT_HOST_MONOTONIC_TIME")
    current_max_age = _number(current_max_age, "CURRENT_FRAME_MAX_AGE", positive=True)
    semantic_max_age = _number(semantic_max_age, "SEMANTIC_SOURCE_MAX_AGE", positive=True)
    unreviewed_max_age = _number(unreviewed_max_age, "MAX_UNREVIEWED_TRACK", positive=True)
    if semantic_max_age < current_max_age or unreviewed_max_age < semantic_max_age:
        raise SurfaceRejection("INVALID_TEMPORAL_FRESHNESS_CONFIG")
    trusted = {"current_frame_max_age_s": current_max_age,
        "semantic_source_max_age_s": semantic_max_age,
        "max_unreviewed_track_s": unreviewed_max_age}
    recorded = certificate.get("configured_timeouts")
    if not isinstance(recorded, dict):
        raise SurfaceRejection("MISSING_TEMPORAL_TIMEOUT_CONFIG")
    for key, value in trusted.items():
        recorded_value = _number(recorded.get(key), key.upper(), positive=True)
        if recorded_value != value:
            raise SurfaceRejection("TEMPORAL_TIMEOUT_CONFIG_MISMATCH")
    _number(recorded.get("max_link_gap_s"), "MAX_LINK_GAP", positive=True)
    if frame.candidates_ready_monotonic_s > now:
        raise SurfaceRejection("CURRENT_FRAME_READY_IN_FUTURE")
    current_age = now - frame.readback_receipt_monotonic_s
    if current_age > current_max_age:
        raise SurfaceRejection("CURRENT_FRAME_EXPIRED")
    expires = _number(certificate.get("semantic_expires_monotonic_s"), "SEMANTIC_EXPIRES")
    deadline = _number(certificate.get("semantic_review_deadline_monotonic_s"), "SEMANTIC_REVIEW_DEADLINE")
    source_receipt = expires - semantic_max_age
    if (source_receipt < 0 or source_receipt >= frame.readback_receipt_monotonic_s
            or not np.isclose(deadline, source_receipt + unreviewed_max_age, atol=1e-8, rtol=0)):
        raise SurfaceRejection("TEMPORAL_REVIEW_DEADLINE_INCONSISTENT")
    if now > deadline:
        raise SurfaceRejection("TEMPORAL_SEMANTIC_REVIEW_DEADLINE_EXPIRED")
    _number(certificate.get("semantic_review_remaining_s"), "SEMANTIC_REVIEW_REMAINING")
    for key in ("confirmation_count", "confirmation_required"):
        if isinstance(certificate.get(key), bool) or not isinstance(certificate.get(key), Integral):
            raise SurfaceRejection("INVALID_TEMPORAL_CONFIRMATION_COUNT")
    if certificate["confirmation_required"] < 2 or certificate["confirmation_count"] < certificate["confirmation_required"]:
        raise SurfaceRejection("TEMPORAL_CONFIRMATION_INCOMPLETE")
    candidates = [candidate for candidate in frame.candidates
                  if candidate.candidate_id == certificate.get("selected_candidate_id")]
    if len(candidates) != 1:
        raise SurfaceRejection("NO_UNIQUE_SELECTED_CURRENT_CANDIDATE")
    candidate = candidates[0]
    selected_copy = certificate.get("selected_candidate")
    if not isinstance(selected_copy, dict) or selected_copy != candidate.to_dict():
        raise SurfaceRejection("SELECTED_CANDIDATE_CONTENT_MISMATCH")
    _number(selected_copy.get("detector_score"), "SELECTED_DETECTOR_SCORE")
    if candidate.truncated is not False or selected_copy.get("truncated") is not False:
        raise SurfaceRejection("TRUNCATED_SELECTED_BOX")
    try:
        json.dumps(certificate, sort_keys=True, allow_nan=False)
    except (ValueError, TypeError):
        raise SurfaceRejection("NONFINITE_OR_NONJSON_TEMPORAL_CERTIFICATE") from None
    box = np.asarray(candidate.bbox_xyxy_pixels, float)
    x0, y0, x1, y1 = box
    if not np.isfinite(box).all() or not 0 < x0 < x1 < frame.width or not 0 < y0 < y1 < frame.height:
        raise SurfaceRejection("INVALID_OR_EDGE_TOUCHING_SELECTED_BOX")
    return candidate, {"checked_host_monotonic_s": now,
        "current_readback_age_proxy_s": current_age, "current_frame_max_age_s": current_max_age,
        "semantic_source_receipt_inferred_from_certificate_s": source_receipt,
        "semantic_review_deadline_monotonic_s": deadline,
        "semantic_review_remaining_at_check_s": deadline - now,
        "fresh_semantic_reply_required_for_each_frame": False,
        "time_scope": "same-host monotonic readback proxy; not exposure",
        "certificate_authentication_verified": False}


def temporal_candidate_surface_estimate(frame: SemanticFrame, *, description: str, now: float, depth,
        camera: SceneCameraModel, rgb_pose: CameraPose, depth_pose: CameraPose,
        selection_certificate: dict, pairing_contract: dict,
        config: SurfaceConfig | None = None, current_frame_max_age_s: float = .5,
        semantic_source_max_age_s: float = 2., max_unreviewed_track_s: float = 8.) -> dict[str, Any]:
    """Estimate an ACTUAL pixel's visible surface and a bbox-center reference.

    ``pairing_contract`` requires frame_id, image_sha256, depth_kind,
    rgb_readback_timestamp_ns, depth_readback_timestamp_ns, and
    intrinsics_pairing_scope='caller_declared_equal_RGB_depth_Scene_intrinsics'.
    The declaration remains a caller assumption: one K cannot independently
    prove registration or exact exposure/pose synchronization.
    """
    cfg = config or SurfaceConfig()
    if not isinstance(cfg, SurfaceConfig):
        raise SurfaceRejection("INVALID_SURFACE_CONFIG")
    candidate, freshness = _temporal_selected(frame, selection_certificate, description, now,
        current_frame_max_age_s, semantic_source_max_age_s, max_unreviewed_track_s)
    if not isinstance(camera, SceneCameraModel) or not isinstance(rgb_pose, CameraPose) or not isinstance(depth_pose, CameraPose):
        raise SurfaceRejection("INVALID_CAMERA_GEOMETRY")
    if (camera.width, camera.height) != (frame.width, frame.height):
        raise SurfaceRejection("RGB_SCENE_INTRINSICS_SIZE_MISMATCH")
    if not isinstance(pairing_contract, dict):
        raise SurfaceRejection("MISSING_EXPLICIT_RGB_DEPTH_PAIRING_CONTRACT")
    if (pairing_contract.get("frame_id") != frame.frame_id or
            pairing_contract.get("image_sha256") != frame.image_sha256):
        raise SurfaceRejection("GEOMETRY_CURRENT_FRAME_BINDING_MISMATCH")
    if pairing_contract.get("depth_kind") != "DepthPerspective":
        raise SurfaceRejection("DEPTH_KIND_MUST_BE_RAY_RANGE")
    if pairing_contract.get("intrinsics_pairing_scope") != "caller_declared_equal_RGB_depth_Scene_intrinsics":
        raise SurfaceRejection("RGB_DEPTH_INTRINSICS_PAIRING_NOT_DECLARED")
    rgb_stamp = _stamp(pairing_contract.get("rgb_readback_timestamp_ns"), "RGB_READBACK_TIMESTAMP")
    depth_stamp = _stamp(pairing_contract.get("depth_readback_timestamp_ns"), "DEPTH_READBACK_TIMESTAMP")
    skew = abs(rgb_stamp-depth_stamp)/1e9
    if skew > cfg.max_rgb_depth_readback_skew_s:
        raise SurfaceRejection("RGB_DEPTH_READBACK_SKEW_EXCEEDED")
    if not np.array_equal(rgb_pose.world_from_local_translation_ned_m,
                          depth_pose.world_from_local_translation_ned_m):
        raise SurfaceRejection("RGB_DEPTH_LOCAL_WORLD_TRANSLATION_MISMATCH")
    translation_skew = float(np.linalg.norm(rgb_pose.position_world_ned_m-depth_pose.position_world_ned_m))
    rgb_q = np.asarray(rgb_pose.orientation_xyzw)/np.linalg.norm(rgb_pose.orientation_xyzw)
    depth_q = np.asarray(depth_pose.orientation_xyzw)/np.linalg.norm(depth_pose.orientation_xyzw)
    rotation_skew = float(np.degrees(2*np.arccos(np.clip(abs(np.dot(rgb_q, depth_q)), 0., 1.))))
    if translation_skew > cfg.max_camera_pose_translation_m or rotation_skew > cfg.max_camera_pose_rotation_deg:
        raise SurfaceRejection("RGB_DEPTH_CAMERA_POSE_SKEW_EXCEEDED")
    try:
        values = np.asarray(depth, dtype=float)
    except (TypeError, ValueError):
        raise SurfaceRejection("INVALID_DEPTH_ARRAY") from None
    if values.shape != (frame.height, frame.width):
        raise SurfaceRejection("MISSING_OR_MISMATCHED_DEPTH_DIMENSIONS")
    x0, y0, x1, y1 = candidate.bbox_xyxy_pixels
    width, height = x1-x0, y1-y0
    rx0, rx1 = int(np.floor(x0+.25*width)), int(np.ceil(x0+.75*width))
    ry0, ry1 = int(np.floor(y0+.25*height)), int(np.ceil(y0+.75*height))
    region = values[ry0:ry1, rx0:rx1]
    valid = np.isfinite(region) & (region > cfg.min_ray_range_m) & (region < cfg.max_ray_range_m)
    count = int(np.count_nonzero(valid))
    fraction = count/region.size if region.size else 0.
    if not count or fraction < cfg.min_valid_depth_fraction:
        raise SurfaceRejection("INSUFFICIENT_VALID_INNER_BOX_DEPTH")
    ys, xs = np.nonzero(valid)
    ranges = region[valid]
    median = float(np.median(ranges))
    center = np.array([(x0+x1)/2, (y0+y1)/2])
    center_distance_squared = (xs+rx0-center[0])**2+(ys+ry0-center[1])**2
    chosen = int(np.lexsort((xs, ys, center_distance_squared, np.abs(ranges-median)))[0])
    pixel = [int(xs[chosen]+rx0), int(ys[chosen]+ry0)]
    ray_range = float(ranges[chosen])
    world = unproject_range(pixel, ray_range, depth_pose, camera)
    depth_reprojection = project_world(world, depth_pose, camera)
    reference = rgb_reference_on_surface_depth(center, world, rgb_pose, camera)
    reference_reprojection = project_world(reference["reference_world_ned_m"], rgb_pose, camera)
    footprint = bbox_plane_corners_world([x0, y0, width, height], reference["axial_plane_depth_m"], rgb_pose, camera)
    return {"status": "SURFACE_GEOMETRY_ACCEPTED", "session_id": frame.session_id,
        "target_generation": frame.target_generation, "frame_id": frame.frame_id,
        "frame_sequence": frame.frame_sequence, "image_sha256": frame.image_sha256,
        "selected_candidate_id": candidate.candidate_id, "selected_bbox_xyxy_pixels": list(candidate.bbox_xyxy_pixels),
        "selection_status": selection_certificate["status"],
        "semantic_local_association_certificate_accepted": selection_certificate["status"] == "SEMANTIC_LOCAL_ASSOCIATION_VALID",
        "detector_score": candidate.detector_score, "identity_probability_calibrated": False,
        "instance_identity_guaranteed": False, "online_control_enabled": False,
        "flight_authorized": False, "freshness_checked": True,
        "freshness_metadata_checks": freshness, "temporal_local_association_certificate_accepted": True,
        "temporal_track_id": selection_certificate["track_id"],
        "semantic_anchor_request_id": selection_certificate["semantic_anchor"], "execution_guard_checked": False,
        "semantic_anchor": selection_certificate["semantic_anchor"],
        "description_sha256": selection_certificate["description_sha256"],
        "current_binding": dict(selection_certificate["current_binding"]),
        "selection_certificate_sha256": hashlib.sha256(json.dumps(selection_certificate,
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
        "geometry_source_version": GEOMETRY_SOURCE_VERSION,
        "geometry_source_sha256": GEOMETRY_SOURCE_SHA256,
        "depth_kind": "DepthPerspective", "ray_range_m": ray_range,
        "surface_position_world_ned_m": world.tolist(),
        "surface_position_local_ned_m": (world-depth_pose.world_from_local_translation_ned_m).tolist(),
        "surface_semantics": "representative_actual_visible_depth_pixel_not_physical_target_center",
        "depth_statistics": {"inner_bbox_xyxy_pixels": [rx0, ry0, rx1, ry1],
            "pixel_count": int(region.size), "valid_count": count, "valid_fraction": fraction,
            "required_min_valid_fraction": cfg.min_valid_depth_fraction,
            "accepted_range_open_interval_m": [cfg.min_ray_range_m, cfg.max_ray_range_m],
            "median_ray_range_m": median, "representative_pixel_xy": pixel,
            "selection_rule": "actual_pixel_nearest_median_range_then_bbox_center_then_y_then_x"},
        "rgb_bbox_center_pixel_xy": center.tolist(),
        "rgb_reference_world_ned_m": reference["reference_world_ned_m"],
        "rgb_reference_offset_from_surface_ned_m": reference["offset_from_surface_ned_m"],
        "rgb_reference_axial_depth_m": reference["axial_plane_depth_m"],
        "rgb_reference_semantics": "YOLO_bbox_center_bearing_at_selected_surface_axial_depth_not_physical_target_center",
        "bbox_plane_corners_world_ned_m": footprint.tolist(),
        "footprint_semantics": "observed_box_plane_geometric_proxy_not_occlusion_or_physical_target_volume",
        "depth_pixel_reprojection_max_axis_error_px": float(np.max(np.abs(np.asarray(depth_reprojection["pixel_xy"])-pixel))),
        "rgb_reference_reprojection_max_axis_error_px": float(np.max(np.abs(np.asarray(reference_reprojection["pixel_xy"])-center))),
        "pairing_metadata_checks": {"rgb_readback_timestamp_ns": rgb_stamp,
            "depth_readback_timestamp_ns": depth_stamp, "readback_skew_s": skew,
            "camera_pose_translation_skew_m": translation_skew, "camera_pose_rotation_skew_deg": rotation_skew,
            "intrinsics_pairing_scope": pairing_contract["intrinsics_pairing_scope"],
            "synchronization_verified": False,
            "time_scope": "render_readback_marks_not_exposure_or_exact_measurement_time"},
        "limitations": ["box membership and depth validity do not prove selected pixels belong to the specified instance",
            "caller declares equal RGB/depth Scene intrinsics; not independently established by this function",
            "readback/pose skew checks do not prove exposure synchronization",
            "readback freshness is a host proxy, not exposure freshness or a new online result during replay",
            "client temporal certificate consistency is not cryptographic authentication or independent identity verification",
            "surface/reference output requires a separate current motion execution safety gate"]}

