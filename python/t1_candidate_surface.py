"""Bind a selected current YOLO box to a visible DepthPerspective surface.

Pure perception geometry: no AirSim client, model inference, target truth, clock
freshness authorization, or flight command. A semantic certificate is client
side policy evidence, not a cryptographic server attestation or calibrated
instance probability. RGB/depth timestamps are readback marks, not exposures.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from t1_pose_geometry import (CameraPose, SceneCameraModel, bbox_plane_corners_world,
    project_world, rgb_reference_on_surface_depth, unproject_range)
from t1_semantic_perception import SemanticFrame


class SurfaceRejection(ValueError):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise SurfaceRejection(f"INVALID_{name}")
    result = float(value)
    if result < 0 or (positive and result == 0):
        raise SurfaceRejection(f"INVALID_{name}")
    return result


def _stamp(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise SurfaceRejection(f"INVALID_{name}")
    return int(value)


@dataclass(frozen=True)
class SurfaceConfig:
    min_valid_depth_fraction: float = .5
    min_ray_range_m: float = .05
    max_ray_range_m: float = 1000.
    max_rgb_depth_readback_skew_s: float = .02
    max_camera_pose_translation_m: float = .02
    max_camera_pose_rotation_deg: float = 1.

    def __post_init__(self):
        for field in ("min_valid_depth_fraction", "min_ray_range_m", "max_ray_range_m"):
            _number(getattr(self, field), field.upper(), positive=True)
        for field in ("max_rgb_depth_readback_skew_s", "max_camera_pose_translation_m",
                      "max_camera_pose_rotation_deg"):
            _number(getattr(self, field), field.upper())
        if self.min_valid_depth_fraction > 1 or self.min_ray_range_m >= self.max_ray_range_m:
            raise SurfaceRejection("INVALID_DEPTH_CONFIG")


def detector_only_selection_certificate(frame: SemanticFrame) -> dict[str, Any]:
    """Highest-score detector baseline, with no semantic or identity claim."""
    if not isinstance(frame, SemanticFrame):
        raise SurfaceRejection("INVALID_SEMANTIC_FRAME")
    if not frame.candidates:
        raise SurfaceRejection("NO_YOLO_CANDIDATE")
    candidates = sorted(frame.candidates, key=lambda item: item.detector_score, reverse=True)
    if len(candidates) > 1 and abs(candidates[0].detector_score-candidates[1].detector_score) <= 1e-12:
        raise SurfaceRejection("AMBIGUOUS_HIGHEST_DETECTOR_SCORE")
    selected = candidates[0]
    return {"status": "DETECTOR_ONLY_UNCALIBRATED", "reason": "UNIQUE_HIGHEST_YOLO_DETECTOR_SCORE",
        "session_id": frame.session_id, "target_generation": frame.target_generation,
        "current_frame_id": frame.frame_id, "current_frame_sequence": frame.frame_sequence,
        "current_image_sha256": frame.image_sha256, "selected_candidate_id": selected.candidate_id,
        "selected_candidate": selected.to_dict(), "online_control_enabled": False,
        "identity_probability_calibrated": False, "instance_identity_guaranteed": False}


def _selected(frame, certificate):
    if not isinstance(frame, SemanticFrame) or not isinstance(certificate, dict):
        raise SurfaceRejection("INVALID_FRAME_OR_SELECTION_CERTIFICATE")
    status = certificate.get("status")
    if status not in ("DETECTOR_ONLY_UNCALIBRATED", "SEMANTIC_LOCAL_ASSOCIATION_VALID"):
        raise SurfaceRejection("SELECTION_CERTIFICATE_NOT_ACCEPTED")
    expected = {"session_id": frame.session_id, "target_generation": frame.target_generation,
        "current_frame_id": frame.frame_id, "current_frame_sequence": frame.frame_sequence,
        "current_image_sha256": frame.image_sha256}
    if any(isinstance(certificate.get(key), bool) or not isinstance(certificate.get(key), Integral)
           for key in ("target_generation", "current_frame_sequence")):
        raise SurfaceRejection("INVALID_SELECTION_CONTEXT_INTEGER")
    if any(certificate.get(key) != value for key, value in expected.items()):
        raise SurfaceRejection("SELECTION_CURRENT_FRAME_BINDING_MISMATCH")
    if any(certificate.get(key) is not False for key in ("online_control_enabled",
            "identity_probability_calibrated", "instance_identity_guaranteed")):
        raise SurfaceRejection("SELECTION_CERTIFICATE_EXCEEDS_EVIDENCE_SCOPE")
    if status == "SEMANTIC_LOCAL_ASSOCIATION_VALID" and (certificate.get("request_mode") != "image_text"
            or certificate.get("proposals_transmitted") is not False
            or certificate.get("server_proposal_ranking") is not False):
        raise SurfaceRejection("SEMANTIC_TRANSPORT_CONTRACT_MISMATCH")
    candidates = [candidate for candidate in frame.candidates
                  if candidate.candidate_id == certificate.get("selected_candidate_id")]
    if len(candidates) != 1:
        raise SurfaceRejection("NO_UNIQUE_SELECTED_CURRENT_CANDIDATE")
    candidate = candidates[0]
    selected_copy = certificate.get("selected_candidate")
    if not isinstance(selected_copy, dict) or selected_copy != candidate.to_dict():
        raise SurfaceRejection("SELECTED_CANDIDATE_CONTENT_MISMATCH")
    if candidate.truncated is not False:
        raise SurfaceRejection("TRUNCATED_SELECTED_BOX")
    box = np.asarray(candidate.bbox_xyxy_pixels, float)
    x0, y0, x1, y1 = box
    if (not np.isfinite(box).all() or not 0 < x0 < x1 < frame.width or
            not 0 < y0 < y1 < frame.height):
        # A touching box is rejected even if a caller forgot to mark truncation.
        raise SurfaceRejection("INVALID_OR_EDGE_TOUCHING_SELECTED_BOX")
    return candidate


def candidate_surface_estimate(frame: SemanticFrame, *, depth,
        camera: SceneCameraModel, rgb_pose: CameraPose, depth_pose: CameraPose,
        selection_certificate: dict, pairing_contract: dict,
        config: SurfaceConfig | None = None) -> dict[str, Any]:
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
    candidate = _selected(frame, selection_certificate)
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
        "flight_authorized": False, "freshness_checked": False, "execution_guard_checked": False,
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
            "surface/reference output requires separate observation freshness and execution safety gates"]}


def replay_saved_yolo_surfaces(probe_path, detection_path, output):
    """Read saved YOLO outputs and matching RGB/depth; no truth is projected."""
    import csv
    import hashlib
    import json
    from pathlib import Path
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from t1_semantic_perception import Candidate

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    probe_path, detection_path, output = map(Path, (probe_path, detection_path, output))
    probe = json.loads(probe_path.read_text(encoding="utf-8-sig"))
    detection = json.loads(detection_path.read_text(encoding="utf-8-sig"))
    if sha(probe_path) != detection["source_report_sha256"]:
        raise ValueError("YOLO replay source probe hash mismatch")
    saved = {row["frame_id"]: row for row in detection["frames"]}
    if len(saved) != len(detection["frames"]) or set(saved) != {row["frame_id"] for row in probe["frames"]}:
        raise ValueError("YOLO replay frame set differs from the all-frame probe denominator")
    rows, images, sources = [], [], [{"path": str(probe_path.resolve()), "sha256": sha(probe_path)},
        {"path": str(detection_path.resolve()), "sha256": sha(detection_path)}]
    for index, raw in enumerate(probe["frames"], 1):
        yolo = saved[raw["frame_id"]]["methods"]["yolo"]
        encoded = Path(raw["media"]["rgb"]["path"]).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != raw["media"]["rgb"]["sha256"] or digest != saved[raw["frame_id"]]["rgb_sha256"]:
            raise ValueError("current RGB hash differs from probe or YOLO output binding")
        if sha(raw["media"]["depth"]["path"]) != raw["media"]["depth"]["sha256"]:
            raise ValueError("saved DepthPerspective array hash mismatch")
        for kind in ("rgb", "depth"):
            sources.append({"path": raw["media"][kind]["path"], "sha256": raw["media"][kind]["sha256"]})
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != tuple(raw["image_size_wh"][::-1]):
            raise ValueError("saved RGB dimensions differ from bound frame")
        images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        record = {"frame_id": raw["frame_id"], "case_id": raw["case_id"], "image_sha256": digest,
                  "status": "NO_SURFACE", "reason": None, "selected_bbox_xyxy_pixels": None,
                  "image_size_wh": raw["image_size_wh"]}
        try:
            if yolo["status"] != "RETURNED":
                raise SurfaceRejection("YOLO_SOURCE_DID_NOT_RETURN")
            candidates = tuple(Candidate(f"{raw['frame_id']}:yolo:{ordinal:03d}",
                tuple(candidate["bbox_xyxy_pixels"]), candidate["confidence"],
                candidate["source_class_name"], candidate["truncated"])
                for ordinal, candidate in enumerate(yolo["result"]["candidates"])
                if candidate["class_name"] == "balloon")
            # No timing evidence is inferred from an old image and later replay.
            # Zero values only instantiate the pure frame-binding schema.
            frame = SemanticFrame(raw["session_id"], raw["target_generation"], raw["frame_id"],
                index, digest, *raw["image_size_wh"], 0., 0., candidates, encoded)
            certificate = detector_only_selection_certificate(frame)
            if certificate["selected_candidate"]["bbox_xyxy_pixels"] != yolo["selected_box"]:
                raise SurfaceRejection("SAVED_HIGHEST_SCORE_SELECTION_NOT_REPRODUCED")
            camera = SceneCameraModel.from_k(raw["scene_intrinsics"]["K_nominal"], *raw["image_size_wh"])
            if raw["scene_intrinsics"]["intrinsics_source"] != "AirSim_Scene_projection_matrix":
                raise SurfaceRejection("SOURCE_SCENE_INTRINSICS_NOT_BOUND")
            poses = [CameraPose(pose["position_local_ned_m"], pose["orientation_xyzw"],
                               raw["world_from_local_translation_ned_m"])
                     for pose in raw["response_camera_poses"][:2]]
            if len(poses) != 2:
                raise SurfaceRejection("MISSING_RGB_DEPTH_RESPONSE_POSES")
            contract = {"frame_id": frame.frame_id, "image_sha256": frame.image_sha256,
                "depth_kind": "DepthPerspective",
                "rgb_readback_timestamp_ns": raw["response_camera_poses"][0]["render_readback_timestamp_ns"],
                "depth_readback_timestamp_ns": raw["response_camera_poses"][1]["render_readback_timestamp_ns"],
                "intrinsics_pairing_scope": "caller_declared_equal_RGB_depth_Scene_intrinsics"}
            surface = candidate_surface_estimate(frame, depth=np.load(raw["media"]["depth"]["path"], allow_pickle=False),
                camera=camera, rgb_pose=poses[0], depth_pose=poses[1], selection_certificate=certificate,
                pairing_contract=contract)
            record.update(surface)
            record["selection_certificate"] = certificate
        except (SurfaceRejection, ValueError, KeyError, TypeError) as error:
            record["reason"] = str(error)
        rows.append(record)
    accepted = [row for row in rows if row["status"] == "SURFACE_GEOMETRY_ACCEPTED"]
    summary = {"schema_version": 1, "kind": "T1_SAVED_YOLO_SELECTED_SURFACE_REPLAY",
        "status": "COMPLETED_WITH_NO_OUTPUTS_RETAINED", "frame_count_all": len(rows),
        "surface_geometry_accepted": len(accepted), "no_surface_output": len(rows)-len(accepted),
        "no_surface_frame_ids": [row["frame_id"] for row in rows if row["status"] != "SURFACE_GEOMETRY_ACCEPTED"],
        "max_depth_pixel_reprojection_error_px": max((row["depth_pixel_reprojection_max_axis_error_px"] for row in accepted), default=None),
        "max_rgb_reference_reprojection_error_px": max((row["rgb_reference_reprojection_max_axis_error_px"] for row in accepted), default=None),
        "min_accepted_valid_inner_depth_fraction": min((row["depth_statistics"]["valid_fraction"] for row in accepted), default=None),
        "independent_UE_sessions_in_source": probe["independent_UE_sessions"],
        "new_simulator_flights": 0, "new_inference_or_API_calls": 0, "formal_efficacy_result": False,
        "selection_condition": "unique_highest_YOLO_detector_score_uncalibrated",
        "target_truth_used_as_geometry_input": False, "identity_accuracy_evaluated": False,
        "freshness_or_flight_authorization": False,
        "offline_frame_clock_field_scope": "zero_schema_placeholders_not_timing_evidence_or_freshness_check",
        "sources": sources, "source_YOLO_weights_sha256": detection["weights_sha256"],
        "limits": ["saved images and saved detector outputs, not an online control loop",
            "reprojection is internal geometry consistency, not surface-location ground-truth accuracy",
            "14 accepted boxes do not independently establish specified target identity or calibrated probability",
            "equal RGB/depth Scene intrinsics are caller-declared; readback/pose skew metadata are not exposure sync",
            "one UE session; cases and frames are nested observations"],
        "ai_assistance": "AI assisted pure surface adapter, tests and readonly saved replay"}
    snapshot = output/"source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("t1_candidate_surface.py", "test_t1_candidate_surface.py", "t1_pose_geometry.py", "t1_semantic_perception.py"):
        source = Path(__file__).parent/name
        (snapshot/name).write_bytes(source.read_bytes())
        summary.setdefault("analysis_source_sha256", {})[name] = sha(source)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output/"all_frames.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    with (output/"all_frames.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        columns = ["frame_id", "case_id", "status", "reason", "selected_candidate_id", "ray_range_m",
                   "depth_pixel_reprojection_max_axis_error_px", "rgb_reference_reprojection_max_axis_error_px"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots((len(rows)+2)//3, 3, figsize=(15, ((len(rows)+2)//3)*3.8), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(rows):
            ax.set_visible(False)
            continue
        row = rows[index]
        ax.imshow(images[index])
        if row["status"] == "SURFACE_GEOMETRY_ACCEPTED":
            x0, y0, x1, y1 = row["selected_bbox_xyxy_pixels"]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#27dbe8", lw=1.)
            ax.plot(*row["depth_statistics"]["representative_pixel_xy"], "+", color="#ffdf4b", ms=10, mew=1.5)
            ax.plot(*row["rgb_bbox_center_pixel_xy"], "x", color="#27dbe8", ms=7)
            title = f"{row['frame_id']} | actual depth {row['ray_range_m']:.3f} m"
        else:
            title = f"{row['frame_id']} | {row['reason']}"
        ax.set_title(title, fontsize=9)
        width, height = row["image_size_wh"]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Saved YOLO boxes to actual DepthPerspective surface pixels\nGeometry replay only; no target-truth input, new inference, freshness or flight authorization", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .98))
    figure = output/"selected_surface_contact_sheet.png"
    fig.savefig(figure, dpi=130)
    plt.close(fig)
    (output/"figure_receipt.json").write_text(json.dumps({"path": str(figure.resolve()), "sha256": sha(figure),
        "all_frames_sha256": sha(output/"all_frames.jsonl"), "source_probe_sha256": sha(probe_path),
        "scope": "YOLO_box_and_actual_depth_sample_overlay_no_target_truth_input"}, indent=2), encoding="utf-8")
    (output/"summary.md").write_text("\n".join(["# YOLO 候选框到可见表面点回放", "",
        f"全部 {len(rows)} 帧保留：{len(accepted)} 帧产生几何表面输出，{len(rows)-len(accepted)} 帧没有输出。", "",
        f"深度像素重投影最大误差：{summary['max_depth_pixel_reprojection_error_px']:.3g} 像素；RGB 框中心参考最大回环误差：{summary['max_rgb_reference_reprojection_error_px']:.3g} 像素。", "",
        "使用保存的 YOLO 最高分候选、实际 Scene 内参、返回相机姿态和 DepthPerspective 数组。目标真值与实例掩膜未进入该几何接口。", "",
        "代表深度必须取自实际像素；优先接近框内部有效深度中位数，再接近框中心。框中心在这个表面像素对应的 RGB 相机轴向深度平面上建立参考，不是气球物理中心。", "",
        "黄色加号为实际取样深度像素，青色框和叉号为 YOLO 框及框中心。", "",
        f"![表面取样图]({figure.resolve().as_posix()})", "",
        "这是一次已有数据的几何回放，没有新增 API 调用、推理或飞行，也没有验证帧新鲜度、目标身份概率或方法效果。零时钟值只是离线结构占位；不产生时延证据。", "",
        "AI 辅助说明：接口实现、测试与数据整理由 AI 辅助，原始数据保持不变。", ""]), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("frame_count_all", "surface_geometry_accepted", "no_surface_output",
        "no_surface_frame_ids", "max_depth_pixel_reprojection_error_px", "max_rgb_reference_reprojection_error_px")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Read-only saved YOLO box/depth geometry replay")
    parser.add_argument("--ground-probe", required=True)
    parser.add_argument("--yolo-replay", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    replay_saved_yolo_surfaces(args.ground_probe, args.yolo_replay, args.output)
