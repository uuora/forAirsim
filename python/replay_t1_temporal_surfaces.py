"""Readonly replay of saved temporal selections against paired RGB/depth.

The geometry projection extracts a closed allowlist from each raw frame. Actor
truth, phase labels, segmentation masks and independent identity evaluation do
not enter surface estimation. All captured frames and missed planned slots are
retained. Historic receipt freshness is checked at the saved observer completion
time, never at a fabricated zero clock or as authorization for current flight.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from t1_candidate_surface import SurfaceRejection
from t1_candidate_surface_v2 import temporal_candidate_surface_estimate
from t1_pose_geometry import CameraPose, SceneCameraModel
from t1_semantic_perception import Candidate, SemanticFrame

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "logs/t1_continuous_identity_analysis_v1p3_final_20261004/report.json"
DEFAULT_OUTPUT = ROOT / "logs/t1_temporal_surface_replay_20261004"


def digest(path):
    path = Path(path)
    if path.name == ".env":
        raise ValueError("Environment credentials are outside replay inputs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def geometry_inputs(raw):
    """Project metadata before spatial computation; no truth/phase access."""
    keys = ("session_id", "target_generation", "frame_id", "frame_sequence", "description",
        "image_size_wh", "media", "scene_intrinsics", "world_from_local_translation_ned_m",
        "response_camera_poses", "capture_received_host_s", "observer_record")
    return {key: raw[key] for key in keys}


def restore_frame(saved, rgb_bytes):
    candidates = tuple(Candidate(item["candidate_id"], tuple(item["bbox_xyxy_pixels"]),
        item["detector_score"], item["source_class_name"], item["truncated"])
        for item in saved["candidates"])
    return SemanticFrame(saved["session_id"], saved["target_generation"], saved["frame_id"],
        saved["frame_sequence"], saved["image_sha256"], *saved["image_size_wh"],
        saved["readback_receipt_monotonic_s"], saved["candidates_ready_monotonic_s"],
        candidates, rgb_bytes, json.dumps(saved["detector_report"], allow_nan=False))


def replay(analysis_path=DEFAULT_ANALYSIS, output=DEFAULT_OUTPUT):
    analysis_path, output = Path(analysis_path).resolve(), Path(output).resolve()
    analysis = json.loads(analysis_path.read_text(encoding="utf-8-sig"))
    source_hashes = {str(Path(item["path"]).resolve()): item["sha256"] for item in analysis["sources"]}
    sources = [{"path": str(analysis_path), "sha256": digest(analysis_path)}]
    rows, missed, episodes, examples = [], [], [], []
    model_bindings = []
    media_checks = []
    identities = set()
    for episode in analysis["episodes"]:
        source = Path(episode["report_path"]).resolve()
        if digest(source) != source_hashes.get(str(source)):
            raise ValueError("Saved raw report differs from independently analyzed source hash")
        sources.append({"path": str(source), "sha256": digest(source)})
        report = json.loads(source.read_text(encoding="utf-8-sig"))
        model_binding = report["observer_close_receipt"]["model_binding"]
        if digest(model_binding["weights_path"]) != model_binding["weights_sha256"]:
            raise ValueError("Current specialist weights differ from saved model binding")
        model_bindings.append(model_binding)
        if report["new_flights"] != 0 or report["api_control_acquired"] is not False:
            raise ValueError("Replay source is not the declared ground-only pilot")
        for slot in report["scheduled_slots"]:
            if slot["capture_status"] != "CAPTURED":
                missed.append({"episode_id": episode["episode_id"], "frame_sequence": slot["frame_sequence"],
                    "status": "NOT_CAPTURED", "reason": slot["not_captured_reason"]})
        episode_rows = []
        for raw in report["frames"]:
            data = geometry_inputs(raw)
            identity = (data["session_id"], data["frame_sequence"], data["frame_id"])
            if identity in identities:
                raise ValueError("Duplicate replay current-frame context")
            identities.add(identity)
            # All media hashes checked, but segmentation bytes are never decoded
            # or interpreted. Spatial inputs remain RGB, depth and camera data.
            for kind in ("rgb", "depth", "segmentation"):
                item = data["media"][kind]
                if digest(item["path"]) != item["sha256"]:
                    raise ValueError("Saved media hash changed")
                media_checks.append({"kind": kind, **item})
            encoded = Path(data["media"]["rgb"]["path"]).read_bytes()
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if image is None or list(image.shape[:2][::-1]) != data["image_size_wh"]:
                raise ValueError("Saved RGB dimensions differ from acquisition binding")
            depth = np.load(data["media"]["depth"]["path"], allow_pickle=False)
            if depth.shape != tuple(data["image_size_wh"][::-1]):
                raise ValueError("Saved paired depth dimensions differ from acquisition binding")
            observer = data["observer_record"]
            decision = observer.get("decision", {})
            certificate = decision.get("temporal_identity", {})
            row = {"episode_id": episode["episode_id"], "session_id": data["session_id"],
                "frame_id": data["frame_id"], "frame_sequence": data["frame_sequence"],
                "image_sha256": data["media"]["rgb"]["sha256"], "status": "NO_SURFACE",
                "reason": "NO_TEMPORAL_SELECTION", "source_observer_status": observer["status"],
                "source_selection_status": certificate.get("status"),
                "source_tracker_state": certificate.get("state"),
                "source_tracker_reason": certificate.get("reason"),
                "source_selected_candidate_id": certificate.get("selected_candidate_id"),
                "saved_observer_completed_host_s": observer["process_completed_host_monotonic_s"],
                "geometry_input_scope": "bound_RGB_candidate_depth_and_self_camera_only",
                "offline_replay": True, "flight_authorized": False}
            if observer["status"] != "RETURNED":
                row["reason"] = "SOURCE_OBSERVER_DID_NOT_RETURN"
            elif certificate.get("status") == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
                frame = restore_frame(decision["semantic_frame"], encoded)
                expected = (data["session_id"], data["target_generation"], data["frame_id"],
                    data["frame_sequence"], data["media"]["rgb"]["sha256"], data["capture_received_host_s"])
                actual = (frame.session_id, frame.target_generation, frame.frame_id, frame.frame_sequence,
                    frame.image_sha256, frame.readback_receipt_monotonic_s)
                if expected != actual:
                    raise ValueError("Saved semantic current frame differs from acquisition binding")
                if data["scene_intrinsics"]["intrinsics_source"] != "AirSim_Scene_projection_matrix":
                    raise ValueError("Scene nominal intrinsics source is not declared")
                poses = [CameraPose(item["position_local_ned_m"], item["orientation_xyzw"],
                    data["world_from_local_translation_ned_m"]) for item in data["response_camera_poses"][:2]]
                pairing = {"frame_id": frame.frame_id, "image_sha256": frame.image_sha256,
                    "depth_kind": "DepthPerspective",
                    "rgb_readback_timestamp_ns": data["response_camera_poses"][0]["render_readback_timestamp_ns"],
                    "depth_readback_timestamp_ns": data["response_camera_poses"][1]["render_readback_timestamp_ns"],
                    "intrinsics_pairing_scope": "caller_declared_equal_RGB_depth_Scene_intrinsics"}
                try:
                    result = temporal_candidate_surface_estimate(frame, description=data["description"],
                        now=observer["process_completed_host_monotonic_s"], depth=depth,
                        camera=SceneCameraModel.from_k(data["scene_intrinsics"]["K_nominal"], *data["image_size_wh"]),
                        rgb_pose=poses[0], depth_pose=poses[1], selection_certificate=certificate,
                        pairing_contract=pairing)
                    row.update(result)
                    row["reason"] = "TEMPORAL_CERTIFICATE_AND_PAIRED_SURFACE_ACCEPTED"
                    if not any(example[0] == episode["episode_id"] for example in examples):
                        examples.append((episode["episode_id"], cv2.cvtColor(image, cv2.COLOR_BGR2RGB), row))
                except SurfaceRejection as error:
                    row["reason"] = error.reason
            rows.append(row)
            episode_rows.append(row)
        episodes.append({"episode_id": episode["episode_id"], "captured_frame_count": len(episode_rows),
            "source_temporal_selection_count": sum(row["source_selection_status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID" for row in episode_rows),
            "surface_geometry_accepted": sum(row["status"] == "SURFACE_GEOMETRY_ACCEPTED" for row in episode_rows),
            "reasons": dict(Counter(row["reason"] for row in episode_rows))})
    if len(rows) != analysis["captured_slots_all"] or len(rows) + len(missed) != analysis["planned_slots_all"]:
        raise ValueError("All-frame/planned-slot denominator differs from source campaign")
    if len({item["weights_sha256"] for item in model_bindings}) != 1:
        raise ValueError("Source sessions do not share the frozen specialist weights")
    accepted = [row for row in rows if row["status"] == "SURFACE_GEOMETRY_ACCEPTED"]
    selected = [row for row in rows if row["source_selection_status"] == "TEMPORAL_LOCAL_ASSOCIATION_VALID"]
    summary = {"schema_version": 2, "kind": "T1_TEMPORAL_SELECTED_SURFACE_SAVED_REPLAY",
        "status": "COMPLETE_ALL_CAPTURED_FRAMES_RETAINED", "created_utc": datetime.now(timezone.utc).isoformat(),
        "planned_slots_all": len(rows)+len(missed), "captured_frames_all": len(rows),
        "not_captured_slots_all": len(missed), "source_temporal_selected_frames": len(selected),
        "source_no_temporal_selection_frames": len(rows)-len(selected),
        "surface_geometry_accepted": len(accepted), "no_surface_output_frames": len(rows)-len(accepted),
        "selected_but_surface_rejected_frames": len(selected)-len(accepted),
        "all_no_output_reasons": dict(Counter(row["reason"] for row in rows if row["status"] != "SURFACE_GEOMETRY_ACCEPTED")),
        "media_hashes_verified_all": len(media_checks), "independent_source_UE_sessions": len(episodes),
        "max_depth_pixel_reprojection_error_px": max((row["depth_pixel_reprojection_max_axis_error_px"] for row in accepted), default=None),
        "max_rgb_reference_reprojection_error_px": max((row["rgb_reference_reprojection_max_axis_error_px"] for row in accepted), default=None),
        "minimum_valid_inner_depth_fraction": min((row["depth_statistics"]["valid_fraction"] for row in accepted), default=None),
        "max_historical_readback_age_proxy_s": max((row["freshness_metadata_checks"]["current_readback_age_proxy_s"] for row in accepted), default=None),
        "max_rgb_depth_readback_skew_s": max((row["pairing_metadata_checks"]["readback_skew_s"] for row in accepted), default=None),
        "source_selection_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID",
        "source_specialist_weights_sha256": model_bindings[0]["weights_sha256"],
        "source_model_bindings": model_bindings,
        "geometry_truth_input_used": False, "segmentation_spatial_input_used": False,
        "physical_center_accuracy_evaluated": False, "formal_efficacy_result": False,
        "new_simulator_sessions": 0, "new_flights": 0, "new_inference_or_API_calls": 0,
        "historic_freshness_check_scope": "saved same-host observer completion vs saved readback/candidate readiness",
        "current_online_freshness_or_flight_authorization": False, "episodes": episodes, "sources": sources,
        "limitations": ["saved-frame geometry replay, not an online control experiment",
            "a representative visible surface and bbox-bearing reference are not physical balloon center",
            "reprojection errors measure numerical consistency, not spatial accuracy against ground truth",
            "client certificates do not authenticate a server reply or guarantee specified instance identity",
            "readback and paired pose checks do not establish exposure synchronization",
            "all frames are nested within four ground development UE sessions"],
        "ai_assistance": "AI assisted temporal surface adapter, contract tests and readonly replay"}
    output.mkdir(parents=True, exist_ok=True)
    with (output/"all_frames.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    columns = ["episode_id", "frame_id", "frame_sequence", "status", "reason",
        "source_tracker_state", "selected_candidate_id", "ray_range_m", "selection_status",
        "depth_pixel_reprojection_max_axis_error_px", "rgb_reference_reprojection_max_axis_error_px"]
    with (output/"all_frames.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    save(output/"not_captured_planned_slots.json", missed)
    save(output/"verified_media.json", media_checks)
    snapshot = output/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    summary["analysis_source_sha256"] = {}
    for name in ("t1_candidate_surface_v2.py", "test_t1_candidate_surface_v2.py",
        "replay_t1_temporal_surfaces.py", "t1_candidate_surface.py", "t1_pose_geometry.py", "t1_semantic_perception.py"):
        source = Path(__file__).parent/name
        (snapshot/name).write_bytes(source.read_bytes())
        summary["analysis_source_sha256"][name] = digest(source)
    save(output/"summary.json", summary)
    (output/"summary.md").write_text("\n".join([
        "# 连续关联候选到可见深度表面回放", "",
        f"四个独立地面 UE 会话共有 {len(rows)+len(missed)} 个预定时隙；实采 {len(rows)} 帧，漏采 {len(missed)} 个时隙另列。",
        f"保存结果中 {len(selected)} 帧带有效 TEMPORAL 当前选择；{len(accepted)} 帧形成可见深度表面与框中心视线参考，{len(rows)-len(accepted)} 帧保留无输出。", "",
        "空间输入只含绑定的当前 YOLO 候选、RGB、DepthPerspective、Scene 内参及 RGB/深度返回相机姿态。",
        "369 个媒体文件均校验哈希；分割文件仅校验字节摘要，不解码，不输入几何。指定实例真值、场景阶段和演员位置不进入表面入口。", "",
        f"接受输出在保存的 observer 完成时刻检查当前帧宿主读回代理时效，最大为 {summary['max_historical_readback_age_proxy_s']:.6f} s；RGB/深度读回最大差为 {summary['max_rgb_depth_readback_skew_s']:.9f} s。",
        f"深度实际像素与 RGB 参考的最大数值回环误差均为 {summary['max_depth_pixel_reprojection_error_px']:.3g} px。数值回环只说明变换自洽，未评价气球物理中心位置准确度。", "",
        "图中青色点表示有效表面射线距离；灰色叉表示没有表面输出，漏采位置留空。示例图固定选择每个会话首次接受的帧。", "",
        "本轮是保存数据的只读几何回放，没有新增 UE、推理、API 或飞行。TEMPORAL 选择保留为局部连续关联证书，不转换成即时语义回复或最高分检测选择。",
        "读回时刻不是曝光时刻，表面不是目标物理中心，证书一致性不是实例身份保证；所有输出仍需单独的实时运动门控。", "",
        "AI 辅助说明：AI 辅助接口实现、契约测试、数据回放和制图；已有原始数据保持不变。", ""
    ]), encoding="utf-8")
    figures(rows, episodes, examples, output)
    save(output/"artifact_receipt.json", {"artifacts": [{"path": str(path), "sha256": digest(path)}
        for path in (output/"summary.json", output/"summary.md", output/"all_frames.jsonl", output/"all_frames.csv",
            output/"not_captured_planned_slots.json", output/"verified_media.json",
            output/"surface_timeline.png", output/"surface_timeline.pdf", output/"first_selected_surfaces.png")],
        "source_analysis_sha256": digest(analysis_path), "geometry_truth_input_used": False})
    print(json.dumps({key: summary[key] for key in ("captured_frames_all", "source_temporal_selected_frames",
        "surface_geometry_accepted", "no_surface_output_frames", "selected_but_surface_rejected_frames",
        "media_hashes_verified_all", "max_depth_pixel_reprojection_error_px",
        "max_rgb_reference_reprojection_error_px")}, ensure_ascii=False))
    return summary


def figures(rows, episodes, examples, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(episodes), 1, figsize=(10, 7), squeeze=False)
    for ax, episode in zip(axes.flat, episodes):
        records = [row for row in rows if row["episode_id"] == episode["episode_id"]]
        for row in records:
            if row["status"] == "SURFACE_GEOMETRY_ACCEPTED":
                ax.scatter(row["frame_sequence"], row["ray_range_m"], color="#167c80", s=20)
            else:
                ax.scatter(row["frame_sequence"], 0, marker="x", color="#777777", s=20)
        ax.set_title(episode["episode_id"], fontsize=9)
        ax.set_xlim(.5, 31.5)
        ax.set_ylabel("Range (m)")
        ax.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Planned frame sequence (missing slot remains a gap)")
    fig.suptitle("Saved temporal selections to actual visible depth surfaces\nTeal: accepted ray range; grey x: no surface; no new inference or flight")
    fig.tight_layout(rect=(0, 0, 1, .93))
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"surface_timeline.{suffix}", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for ax, (episode_id, image, row) in zip(axes.flat, examples):
        ax.imshow(image)
        x0, y0, x1, y1 = row["selected_bbox_xyxy_pixels"]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#27dbe8", lw=1.3)
        ax.plot(*row["depth_statistics"]["representative_pixel_xy"], "+", color="#ffe055", ms=10, mew=1.5)
        ax.plot(*row["rgb_bbox_center_pixel_xy"], "x", color="#27dbe8", ms=7)
        ax.set_title(f"{episode_id}\n{row['frame_id']}: {row['ray_range_m']:.3f} m", fontsize=9)
        ax.axis("off")
    fig.suptitle("First accepted selection in each source session (deterministic examples)\nYellow +: actual depth pixel; cyan x: bbox-bearing reference; not physical target center")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(output/"first_selected_surfaces.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default=str(DEFAULT_ANALYSIS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    replay(args.analysis, args.output)
