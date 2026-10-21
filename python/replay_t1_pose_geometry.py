"""Read-only pose/Scene-convention migration replay of saved T1 observations.

This does not run a simulator, overwrite raw observations, replan a flown path,
or validate physical image calibration. A Scene-convention K is constructed
from saved nominal focal lengths; actual renderer K was not saved in V6.2.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from t1_pose_geometry import (CameraPose, SceneCameraModel, bbox_plane_corners_world,
    project_footprint_polygon, project_world, rgb_reference_on_surface_depth, unproject_range)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path, expected_sha=None):
    actual = _sha(path)
    if expected_sha is not None and actual != expected_sha:
        raise ValueError(f"source hash mismatch: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig")), actual


def replay_observation(row, translation):
    """Trace one observed depth pixel under legacy K and new Scene convention."""
    metadata = row["timestamped_observation"]
    if metadata["status"] != "OK" or metadata["rejection_reasons"]:
        raise ValueError("historical observation was not accepted by its original provider")
    if metadata["coordinate_frame"] != "VEHICLE_LOCAL_NED":
        raise ValueError("camera frame is not declared vehicle-local NED")
    projection = metadata["projection"]
    width, height = projection["depth_size_wh"]
    if projection["rgb_size_wh"] != [width, height]:
        raise ValueError("different RGB/depth intrinsics are not audited")
    fx, fy = projection["focal_lengths_px"]
    principal = np.asarray(projection["principal_pixel_xy"], dtype=float)
    expected_principal = (np.asarray([width, height], float)-1)/2
    if not np.allclose(principal, expected_principal, rtol=0, atol=1e-7):
        raise ValueError("not the declared V6.2 halfpixel migration input")
    camera = SceneCameraModel(width, height, fx, fy, width/2, height/2)
    pose = CameraPose(metadata["camera_position_local_ned_m"],
                      metadata["camera_orientation_xyzw"], translation)
    pixel = np.asarray(metadata["depth_statistics"]["representative_pixel_xy"], float)
    surface_world = np.asarray(metadata["surface_position_local_ned_m"], float)+translation
    depth = float(metadata["depth_m"])
    legacy_relative = pose.world_to_camera(surface_world)
    legacy_uv = principal + np.array([fx, fy])*legacy_relative[1:]/legacy_relative[0]
    legacy_ray = np.array([1., (pixel[0]-principal[0])/fx, (pixel[1]-principal[1])/fy])
    legacy_point = pose.camera_to_world(legacy_ray/np.linalg.norm(legacy_ray)*depth)
    legacy_pixel_error = float(np.max(np.abs(legacy_uv-pixel)))
    legacy_point_error = float(np.linalg.norm(legacy_point-surface_world))
    if legacy_pixel_error > 1e-8 or legacy_point_error > 1e-8:
        raise ValueError("saved legacy surface estimate is not reproduced by its pose/range/pixel")
    new_world = unproject_range(pixel, depth, pose, camera)
    new_projection = project_world(new_world, pose, camera)
    old_under_scene = project_world(surface_world, pose, camera)
    delta = np.asarray(old_under_scene["pixel_xy"])-pixel
    if not np.allclose(delta, [.5, .5], rtol=0, atol=1e-8):
        raise ValueError("legacy-to-Scene migration is not the expected +.5 pixel per axis")
    reference = rgb_reference_on_surface_depth(metadata["centre_px"], new_world, pose, camera)
    corners = bbox_plane_corners_world(metadata["bbox_xywh"], new_projection["axial_depth_m"], pose, camera)
    footprint = project_footprint_polygon(corners, pose, camera)
    if abs(footprint["field_area_fraction"]-1.) > 1e-9:
        raise ValueError("source-image contained box did not reproject to a full field fraction")
    return {
        "legacy_pixel_reprojection_max_axis_error_px": legacy_pixel_error,
        "legacy_surface_reconstruction_error_m": legacy_point_error,
        "new_scene_roundtrip_max_axis_error_px": float(np.max(np.abs(np.array(new_projection["pixel_xy"])-pixel))),
        "legacy_surface_under_scene_pixel_delta_xy": delta.tolist(),
        "scene_reconstructed_surface_shift_m": float(np.linalg.norm(new_world-surface_world)),
        "scene_reconstructed_surface_world_ned_m": new_world.tolist(),
        "scene_rgb_reference_world_ned_m": reference["reference_world_ned_m"],
        "source_box_field_area_fraction": footprint["field_area_fraction"],
        "camera_rotation_angle_from_identity_deg": float(np.degrees(2*np.arccos(np.clip(
            abs(pose.orientation_xyzw[3])/np.linalg.norm(pose.orientation_xyzw), 0, 1)))),
        "source_focal_scope": "saved_nominal_focal_lengths_not_independently_saved_Scene_projection",
    }


def collect_sweep_reports(sweep_path):
    sweep, sha = _read(sweep_path)
    sources = [{"path": str(Path(sweep_path).resolve()), "sha256": sha, "kind": "sweep"}]
    reports = []
    for session in sweep["sessions"]:
        manifest, manifest_sha = _read(session["manifest"], session.get("manifest_sha256"))
        sources.append({"path": session["manifest"], "sha256": manifest_sha, "kind": "manifest"})
        for manifest_session in manifest["sessions"]:
            batch_path = Path(manifest_session["batch_path"])/"batch_report.json"
            batch, batch_sha = _read(batch_path, manifest_session.get("batch_report_sha256"))
            sources.append({"path": str(batch_path), "sha256": batch_sha, "kind": "batch"})
            for run in batch["runs"]:
                report, report_sha = _read(run["report_path"], run.get("report_sha256"))
                sources.append({"path": run["report_path"], "sha256": report_sha, "kind": "flight_report"})
                reports.append((run["report_path"], report))
    return reports, sources


def replay_reports(reports, sources):
    records = []
    for path, report in reports:
        translation = report["origin_calibration"]["world_from_vehicle_local_translation_ned_m"]
        for index, row in enumerate(report["observations"], 1):
            record = {"report_path": str(path), "frame_id": row.get("frame_id", f"observation:{index}"),
                      "policy": report["policy"], "status": "REJECTED", "rejection_reason": None}
            try:
                record.update(replay_observation(row, translation))
                record["status"] = "REPLAYED"
            except (KeyError, ValueError, TypeError, OverflowError) as error:
                record["rejection_reason"] = str(error)
            records.append(record)
    accepted = [row for row in records if row["status"] == "REPLAYED"]
    metrics = ("legacy_pixel_reprojection_max_axis_error_px", "legacy_surface_reconstruction_error_m",
               "new_scene_roundtrip_max_axis_error_px", "scene_reconstructed_surface_shift_m",
               "camera_rotation_angle_from_identity_deg")
    summary = {"schema_version": 1, "kind": "T1_SAVED_POSE_GEOMETRY_MIGRATION_REPLAY",
        "status": "PASS_INTERNAL_GEOMETRY" if accepted and len(accepted) == len(records) else "PARTIAL_OR_REJECTED",
        "flight_report_count": len(reports), "observation_count_all": len(records),
        "replayed_observation_count": len(accepted), "rejected_observation_count": len(records)-len(accepted),
        "new_simulator_flights": 0, "formal_method_efficacy_result": False,
        "maxima": {key: max((row[key] for row in accepted), default=None) for key in metrics},
        "legacy_to_Scene_principal_shift_px": [.5, .5],
        "source_scope": "historical_actual_flight_metadata_replayed_offline_no_rewritten_raw_data",
        "limitations": ["nominal focal lengths inherited from saved legacy metadata; actual Scene projection not saved",
            "roundtrip and +.5 pixel discrepancy check algebraic software consistency only",
            "no independent physical pixel alignment, occlusion, target identity or accuracy validation",
            "historical fixed-height near-forward flights do not validate rotated-camera flight control"],
        "sources": sources, "ai_assistance": "AI assisted geometry implementation and evidence replay"}
    return summary, records


def replay_probe_frame(row, depth, target_mask):
    """Fresh camera-only probe geometry; instance truth is evaluation-only."""
    width, height = row["image_size_wh"]
    if depth.shape != (height, width) or target_mask.shape != depth.shape:
        raise ValueError("saved depth/target mask shape differs from frame dimensions")
    if row["scene_intrinsics"]["intrinsics_source"] != "AirSim_Scene_projection_matrix":
        raise ValueError("fresh probe must bind saved actual Scene projection intrinsics")
    camera = SceneCameraModel.from_k(row["scene_intrinsics"]["K_nominal"], width, height)
    poses = [CameraPose(p["position_local_ned_m"], p["orientation_xyzw"],
                       row["world_from_local_translation_ned_m"]) for p in row["response_camera_poses"]]
    if len(poses) != 3:
        raise ValueError("probe needs ordered RGB, DepthPerspective and Segmentation poses")
    pose_translation_skew = max(float(np.linalg.norm(p.position_world_ned_m-poses[0].position_world_ned_m))
                                for p in poses)
    pose_rotation_skew = max(float(np.linalg.norm(p.rotation_camera_to_ned-poses[0].rotation_camera_to_ned))
                             for p in poses)
    if pose_translation_skew > 1e-5 or pose_rotation_skew > 1e-5:
        raise ValueError("camera-only probe response poses are not static and aligned")
    rgb_pose, depth_pose = poses[:2]
    target = row["evaluation_only"]["target"]
    actor = project_world(row["evaluation_only"]["target_world_ned_m"], rgb_pose, camera)
    ys, xs = np.nonzero(target_mask)
    if len(xs) != target["visible_pixel_count"]:
        raise ValueError("saved segmentation pixel count differs from raw target-ID mask")
    if len(xs):
        centroid = np.array([xs.mean(), ys.mean()])
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
        if bbox != target["bbox_xyxy_pixels"] or not np.allclose(centroid, target["centroid_xy_pixels"],
                                                                rtol=0, atol=1e-9):
            raise ValueError("saved mask bbox/centroid differs from raw target-ID mask")
    else:
        centroid = None
        bbox = None
        if target["centroid_xy_pixels"] is not None or target["bbox_xyxy_pixels"] is not None:
            raise ValueError("empty mask must have absent visible bbox and centroid")
    accepted_mask = target_mask & np.isfinite(depth) & (depth > .05) & (depth < 1000.)
    dy, dx = np.nonzero(accepted_mask)
    samples = []
    indices = np.unique(np.linspace(0, len(dx)-1, min(9, len(dx))).astype(int)) if len(dx) else []
    for choice in indices:
        pixel = [int(dx[choice]), int(dy[choice])]
        ray_range = float(depth[pixel[1], pixel[0]])
        world = unproject_range(pixel, ray_range, depth_pose, camera)
        projected = project_world(world, depth_pose, camera)
        samples.append({"pixel_xy": pixel, "ray_range_m": ray_range,
                        "surface_world_ned_m": world.tolist(),
                        "reprojected_pixel_xy": projected["pixel_xy"],
                        "pixel_roundtrip_max_axis_error_px": float(np.max(np.abs(np.asarray(projected["pixel_xy"])-pixel))),
                        "ray_range_roundtrip_error_m": abs(projected["ray_range_m"]-ray_range),
                        "surface_semantics": "rendered_visible_depth_surface_not_actor_pivot_or_physical_center"})
    delta = np.asarray(actor["pixel_xy"])-centroid if centroid is not None else None
    return {"frame_id": row["frame_id"], "case_id": row["case_id"], "status": "VERIFIED_GEOMETRY",
        "image_size_wh": [width, height], "actor_pivot_projected_pixel_xy": actor["pixel_xy"],
        "actor_pivot_inside_image": actor["inside_image"], "visible_target_pixel_count": len(xs),
        "target_mask_centroid_xy_pixels": centroid.tolist() if centroid is not None else None,
        "target_mask_bbox_xyxy_pixels": bbox,
        "actor_pivot_minus_visible_centroid_xy_px": delta.tolist() if delta is not None else None,
        "actor_pivot_centroid_distance_px": float(np.linalg.norm(delta)) if delta is not None else None,
        "actor_pivot_inside_visible_bbox": bool(bbox[0] <= actor["pixel_xy"][0] < bbox[2] and
            bbox[1] <= actor["pixel_xy"][1] < bbox[3]) if bbox else None,
        "valid_target_depth_pixel_count": len(dx), "sampled_depth_ray_count": len(samples),
        "max_depth_pixel_roundtrip_error_px": max((s["pixel_roundtrip_max_axis_error_px"] for s in samples), default=None),
        "max_depth_range_roundtrip_error_m": max((s["ray_range_roundtrip_error_m"] for s in samples), default=None),
        "camera_position_metadata_consistency_error_m": float(np.linalg.norm(
            rgb_pose.position_world_ned_m-np.asarray(row["camera_world_ned_m"]))),
        "response_camera_translation_skew_m": pose_translation_skew,
        "response_camera_rotation_matrix_skew_frobenius": pose_rotation_skew,
        "rgb_depth_seg_readback_skew_s": row["rgb_depth_seg_readback_skew_s"],
        "depth_samples": samples,
        "truth_access": "evaluation_only_controlled_instance_mask_and_actor_pose",
        "projection_centroid_comparison_scope": "actor_pivot_vs_visible_mask_centroid_diagnostic_not_empirical_lens_calibration"}


def analyze_ground_probe(probe_path, output):
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probe, sha = _read(probe_path)
    if probe["flight_commands_issued"] or probe["api_control_acquired"] or not probe["safe_ground_closure"]:
        raise ValueError("probe did not retain the declared grounded camera-only scope")
    target_id = probe["evaluation_segmentation_ids"]["T1ProbeTarget"]
    color_bgr = np.asarray(probe["segmentation_palette_rgb"][str(target_id)][::-1], np.uint8)
    records, sources, images = [], [{"path": str(probe_path.resolve()), "sha256": sha, "kind": "ground_probe"}], []
    for row in probe["frames"]:
        for vehicle, state in row["ground_state"].items():
            if not state["landed"] or state["api_control_enabled"] or not np.isfinite(state["speed_m_s"]) or state["speed_m_s"] > .01:
                raise ValueError(f"{row['frame_id']}: {vehicle} was not grounded without API control")
        for kind, entry in row["media"].items():
            if _sha(entry["path"]) != entry["sha256"]:
                raise ValueError(f"saved {kind} hash changed in {row['frame_id']}")
            sources.append({"path": entry["path"], "sha256": entry["sha256"], "kind": kind})
        depth = np.load(row["media"]["depth"]["path"], allow_pickle=False)
        segmentation = cv2.imread(row["media"]["segmentation"]["path"], cv2.IMREAD_COLOR)
        rgb = cv2.imread(row["media"]["rgb"]["path"], cv2.IMREAD_COLOR)
        if segmentation is None or rgb is None:
            raise ValueError("raw RGB or segmentation could not be decoded")
        target_mask = np.all(segmentation == color_bgr, axis=2)
        record = replay_probe_frame(row, depth, target_mask)
        records.append(record)
        images.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    samples = [sample for row in records for sample in row["depth_samples"]]
    invisible = [row for row in records if row["visible_target_pixel_count"] == 0]
    visible = [row for row in records if row["visible_target_pixel_count"] > 0]
    summary = {"schema_version": 1, "kind": "T1_ACTUAL_CAMERA_POSE_METADATA_AND_DEPTH_REPLAY",
        "status": "VERIFIED_CAMERA_ONLY_DEVELOPMENT", "independent_UE_sessions": probe["independent_UE_sessions"],
        "frame_count_all": len(records), "verified_frame_count": len(records),
        "case_count": len(set(row["case_id"] for row in records)), "visible_target_frames": len(visible),
        "zero_visible_target_frames": len(invisible), "zero_visible_target_frame_ids": [row["frame_id"] for row in invisible],
        "field_in_frame_but_zero_visible_target_frame_ids": [row["frame_id"] for row in invisible if row["actor_pivot_inside_image"]],
        "sampled_target_depth_rays": len(samples),
        "max_depth_pixel_roundtrip_error_px": max((s["pixel_roundtrip_max_axis_error_px"] for s in samples), default=None),
        "max_depth_ray_range_roundtrip_error_m": max((s["ray_range_roundtrip_error_m"] for s in samples), default=None),
        "max_camera_position_metadata_consistency_error_m": max((row["camera_position_metadata_consistency_error_m"] for row in records), default=None),
        "max_actor_pivot_visible_centroid_distance_px": max((row["actor_pivot_centroid_distance_px"] for row in visible), default=None),
        "max_rgb_depth_seg_readback_skew_s": max((row["rgb_depth_seg_readback_skew_s"] for row in records), default=None),
        "intrinsics_scope": "actual_saved_Scene_projection_K_nominal_with_width_over_2_height_over_2",
        "truth_scope": "actor_pose_and_instance_segmentation_used_only_for_this_offline_evaluation",
        "new_simulator_flights": 0, "flight_commands_issued_by_source_probe": False, "formal_method_efficacy_result": False,
        "limitations": ["one independently launched UE session with nested cases and frames",
            "virtual camera relocations and rotations do not validate vehicle 3D/attitude flight control",
            "roundtrip checks implementation algebra and finite geometry, not empirical range or lens calibration",
            "actor pivot differs from visible mask centroid due to raster convention, perspective and partial occlusion",
            "visible mask is controlled rendered identity evidence, not human annotation or real-world generalization",
            "image timestamps are readback completion, not exposure; static response poses do not imply exact RGB-depth-seg synchronization"],
        "sources": sources, "source_probe_report_sha256": sha,
        "ai_assistance": "AI assisted pose geometry implementation and offline evaluation"}
    for name in ("t1_pose_geometry.py", "replay_t1_pose_geometry.py", "test_t1_pose_geometry.py"):
        path = Path(__file__).parent/name
        summary.setdefault("analysis_source_sha256", {})[name] = _sha(path)
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output/"frame_geometry.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    columns = ["frame_id", "case_id", "status", "actor_pivot_inside_image", "visible_target_pixel_count",
        "actor_pivot_centroid_distance_px", "actor_pivot_inside_visible_bbox", "valid_target_depth_pixel_count",
        "sampled_depth_ray_count", "max_depth_pixel_roundtrip_error_px", "max_depth_range_roundtrip_error_m",
        "camera_position_metadata_consistency_error_m", "rgb_depth_seg_readback_skew_s"]
    with (output/"all_frames.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    rows = (len(records)+2)//3
    fig, axes = plt.subplots(rows, 3, figsize=(15, rows*3.8), squeeze=False)
    for index, ax in enumerate(axes.flat):
        if index >= len(records):
            ax.set_visible(False)
            continue
        row = records[index]
        ax.imshow(images[index])
        u, v = row["actor_pivot_projected_pixel_xy"]
        ax.plot(u, v, "+", color="#27dbe8", ms=10, mew=1.8, label="Projected actor pivot")
        if row["target_mask_centroid_xy_pixels"] is not None:
            u, v = row["target_mask_centroid_xy_pixels"]
            ax.plot(u, v, "x", color="#ffdf4b", ms=7, mew=1.5, label="Visible mask centroid")
            x0, y0, x1, y1 = row["target_mask_bbox_xyxy_pixels"]
            ax.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], color="#ffdf4b", lw=.8)
        ax.set_title(f"{row['frame_id']} | visible pixels: {row['visible_target_pixel_count']}", fontsize=9)
        width, height = row["image_size_wh"]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        if index == 0:
            ax.legend(loc="upper left", fontsize=6)
    fig.suptitle("Actual camera-only probe: pose projection vs rendered instance mask\nGeometry diagnostic; one UE session; no flight efficacy or lens calibration", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .98))
    figure = output/"pose_projection_contact_sheet.png"
    fig.savefig(figure, dpi=130)
    plt.close(fig)
    (output/"figure_receipt.json").write_text(json.dumps({"path": str(figure.resolve()), "sha256": _sha(figure),
        "frame_geometry_sha256": _sha(output/"frame_geometry.jsonl"), "source_probe_sha256": sha,
        "scope": "offline_actual_camera_metadata_mask_overlay_not_method_efficacy"}, indent=2), encoding="utf-8")
    content = ["# T1 实际相机姿态与深度接口检查", "", "状态：VERIFIED_CAMERA_ONLY_DEVELOPMENT。", "",
        f"- 独立 UE 会话：{summary['independent_UE_sessions']}；场景：{summary['case_count']}；帧：{len(records)}。",
        f"- 目标掩膜可见帧：{len(visible)}；完全不可见帧：{len(invisible)}。",
        f"- 采样目标深度射线：{len(samples)}；投影回环最大误差：{summary['max_depth_pixel_roundtrip_error_px']:.3g} 像素。",
        f"- 射线距离回环最大误差：{summary['max_depth_ray_range_roundtrip_error_m']:.3g} m。",
        f"- 相机位置元数据一致性误差：{summary['max_camera_position_metadata_consistency_error_m']:.3g} m。",
        f"- 演员枢轴投影与可见掩膜质心最大距离：{summary['max_actor_pivot_visible_centroid_distance_px']:.6g} 像素；该值不是标定误差或目标定位 RMSE。",
        f"- 几何上位于视场内但目标掩膜为零的帧：{', '.join(summary['field_in_frame_but_zero_visible_target_frame_ids'])}。说明视场几何必须与实际遮挡评价分别报告。",
        "", "## 适用范围", "", "使用逐帧保存的 Scene 投影矩阵内参、返回相机四元数以及显式局部到世界平移。目标演员姿态和实例掩膜只进入本离线评价。",
        "", "这是一次地面相机接口测试，所有帧两架机体均落地且未启用 API 控制。相机高度与姿态变化不等于机体三维飞行。回环误差证明实现的一致性；演员枢轴、可见掩膜质心和实际气球中心也不是同一个量。",
        "", "图中青色加号为演员枢轴投影，黄色叉号和框为目标实例的可见掩膜质心与边界。", "",
        f"![姿态投影检查]({figure.resolve().as_posix()})", "", "AI 辅助说明：实现与统计整理由 AI 辅助，原始模拟器图像和元数据保持不变。", ""]
    (output/"summary.md").write_text("\n".join(content), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "frame_count_all", "case_count", "sampled_target_depth_rays",
        "zero_visible_target_frame_ids", "max_depth_pixel_roundtrip_error_px", "max_actor_pivot_visible_centroid_distance_px")}, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sweep", type=Path)
    group.add_argument("--report", type=Path, action="append")
    group.add_argument("--ground-probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ground_probe:
        analyze_ground_probe(args.ground_probe, args.output)
        return
    if args.sweep:
        reports, sources = collect_sweep_reports(args.sweep)
    else:
        reports, sources = [], []
        for path in args.report:
            report, sha = _read(path)
            reports.append((path, report))
            sources.append({"path": str(path.resolve()), "sha256": sha, "kind": "flight_report"})
    summary, records = replay_reports(reports, sources)
    for name in ("t1_pose_geometry.py", "replay_t1_pose_geometry.py", "test_t1_pose_geometry.py"):
        path = Path(__file__).parent/name
        summary.setdefault("replay_source_sha256", {})[name] = _sha(path)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output/"summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (args.output/"observations.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    columns = ["report_path", "frame_id", "policy", "status", "rejection_reason", *summary["maxima"],
               "source_box_field_area_fraction"]
    with (args.output/"all_observations.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"status": summary["status"], "all": len(records),
                      "replayed": summary["replayed_observation_count"], "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
