"""Camera-only T1 development evidence; vehicles stay grounded and released.

The virtual camera is relocated, not the vehicle. RGB-only policies never
receive segmentation or actor poses; those are saved separately for evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import airsim
import cv2
import numpy as np

from camera_geometry import scene_intrinsics
from setup_scene import ensure_mesh, rotate, vector

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")


def quat(q):
    return [q.x_val, q.y_val, q.z_val, q.w_val]


def ground_states(client):
    rows = {}
    for name in ("DroneA", "DroneB"):
        state = client.getMultirotorState(vehicle_name=name)
        speed = float(np.linalg.norm(vector(state.kinematics_estimated.linear_velocity)))
        control = bool(client.isApiControlEnabled(vehicle_name=name))
        landed = state.landed_state == airsim.LandedState.Landed
        if control or not landed or speed > .02:
            raise ValueError("Camera probe requires grounded, stationary, released vehicles")
        rows[name] = {"landed": landed, "api_control_enabled": control,
                      "speed_m_s": speed, "state_timestamp_ns": int(state.timestamp)}
    return rows


def visible_mask(segmentation, rgb_color):
    mask = np.all(segmentation == np.asarray(rgb_color[::-1], np.uint8), axis=2)
    ys, xs = np.nonzero(mask)
    return {"visible_pixel_count": int(mask.sum()),
            "bbox_xyxy_pixels": [int(xs.min()), int(ys.min()), int(xs.max())+1,
                                 int(ys.max())+1] if len(xs) else None,
            "centroid_xy_pixels": [float(xs.mean()), float(ys.mean())] if len(xs) else None}


def capture(client, camera="front_center"):
    started = time.monotonic()
    responses = client.simGetImages([
        airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True),
        airsim.ImageRequest(camera, airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, True)],
        vehicle_name="DroneA")
    received = time.monotonic()
    if len(responses) != 3:
        raise ValueError("Three paired responses required")
    rgb, depth, seg = responses
    image = cv2.imdecode(np.frombuffer(rgb.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
    segmentation = cv2.imdecode(np.frombuffer(seg.image_data_uint8, np.uint8), cv2.IMREAD_COLOR)
    ranges = np.asarray(depth.image_data_float, float).reshape(depth.height, depth.width)
    if image is None or segmentation is None or image.shape != segmentation.shape or image.shape[:2] != ranges.shape:
        raise ValueError("Paired RGB/depth/segmentation sizes invalid")
    poses = [{"position_local_ned_m": vector(r.camera_position),
              "orientation_xyzw": quat(r.camera_orientation),
              "render_readback_timestamp_ns": int(r.time_stamp)} for r in responses]
    skew = (max(p["render_readback_timestamp_ns"] for p in poses) -
            min(p["render_readback_timestamp_ns"] for p in poses))/1e9
    if skew > .03 or any(p["render_readback_timestamp_ns"] <= 0 for p in poses):
        raise ValueError("Paired readback timestamps invalid")
    for p in poses[1:]:
        if (np.linalg.norm(np.array(p["position_local_ned_m"])-np.array(poses[0]["position_local_ned_m"])) > .01
                or abs(np.dot(p["orientation_xyzw"], poses[0]["orientation_xyzw"])) < .99999):
            raise ValueError("Paired camera poses differ")
    return image, ranges, segmentation, {"capture_started_host_s": started,
        "capture_received_host_s": received, "capture_duration_s": received-started,
        "rgb_depth_seg_readback_skew_s": skew, "response_camera_poses": poses}


def run(output, protocol, observer=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(protocol.read_text(encoding="utf-8-sig"))
    report = {"status": "RUNNING", "kind": "T1_CAMERA_ONLY_DEVELOPMENT_PROBE",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "flight_commands_issued": False, "api_control_acquired": False,
        "virtual_camera_relocated": True, "independent_UE_sessions": 1,
        "formal_efficacy_results": False, "configuration": cfg,
        "protocol": str(protocol.resolve()), "protocol_sha256": digest(protocol),
        "source_sha256": {name: digest(ROOT/"python"/name) for name in (
            "t1_ground_perception_probe.py", "setup_scene.py", "camera_geometry.py")},
        "frames": [], "evaluation_truth_access": "scenario setup and scoring only",
        "timestamp_scope": "Render-readback completion; camera pose sampled at OnEndDraw, not exposure time",
        "ai_assistance": "AI assisted probe implementation and evidence summary"}
    save(output/"report.json", report)
    client = airsim.MultirotorClient(timeout_value=15)
    original_camera = None
    try:
        if not client.ping() or client.simIsPause():
            raise ValueError("New, unpaused owned simulation required")
        report["ground_before"] = ground_states(client)
        vehicle = client.simGetVehiclePose(vehicle_name="DroneA")
        world = np.asarray(vector(client.simGetObjectPose("DroneA").position))
        local = np.asarray(vector(vehicle.position))
        translation = world-local
        report["world_from_local_translation_ned_m"] = translation.tolist()
        camera = client.simGetCameraInfo("front_center", vehicle_name="DroneA").pose
        inverse = vehicle.orientation.inverse()
        original_camera = airsim.Pose(rotate(camera.position-vehicle.position, inverse),
                                     inverse*camera.orientation)
        if not client.simSetSegmentationObjectID(".*", 0, True):
            raise ValueError("Background segmentation reset failed")
        assets = {x.casefold(): x for x in client.simListAssets()}
        texture = output/"red_texture.png"
        cv2.imwrite(str(texture), np.full((16,16,3), [55,35,235], np.uint8))
        truth_id_audit = {}
        for name, asset, scale, segid in (
            ("T1ProbeTarget", "sphere", cfg["target_scale_xyz"], 42),
            ("T1ProbeDistractor", "sphere", cfg["target_scale_xyz"], 43),
            ("T1ProbeOccluder", "1m_cube_chamfer", [.2,1.6,2.0], 44)):
            ensure_mesh(client, name, assets[asset], [0,0,-3] if segid==42 else cfg["inactive_object_world_ned_m"], scale)
            if segid in (42,43) and not client.simSetObjectMaterialFromTexture(name, str(texture)):
                raise ValueError("Red material assignment failed")
            if not client.simSetSegmentationObjectID(name, segid):
                raise ValueError("Instance segmentation assignment failed")
            actual = client.simGetSegmentationObjectID(name.lower())
            if actual != segid:
                raise ValueError("Instance segmentation getter mismatch")
            truth_id_audit[name] = actual
        report["evaluation_segmentation_ids"] = truth_id_audit
        palette_path = ROOT/"external/AirSim/docs/seg_rgbs.txt"
        palette = {int(line.split("\t")[0]): json.loads(line.split("\t")[1])
                   for line in palette_path.read_text(encoding="utf-8").splitlines()}
        report["segmentation_palette_sha256"] = digest(palette_path)
        report["segmentation_palette_rgb"] = {str(i): palette[i] for i in (0,42,43,44)}
        for case in cfg["cases"]:
            target = case["target_world_ned_m"]
            ensure_mesh(client, "T1ProbeTarget", assets["sphere"], target, cfg["target_scale_xyz"])
            ensure_mesh(client, "T1ProbeDistractor", assets["sphere"],
                        case.get("distractor_world_ned_m", cfg["inactive_object_world_ned_m"]), cfg["target_scale_xyz"])
            ensure_mesh(client, "T1ProbeOccluder", assets["1m_cube_chamfer"],
                        case.get("occluder_world_ned_m", cfg["inactive_object_world_ned_m"]), [.2,1.6,2.0])
            # Cameras are vehicle-relative; translations use audited startup origin.
            pitch, roll, yaw = np.radians(case["camera_pitch_roll_yaw_deg"])
            orientation = airsim.to_quaternion(float(pitch), float(roll), float(yaw))
            eye = airsim.Vector3r(*case["camera_world_ned_m"])
            relative = airsim.Pose(rotate(eye-airsim.Vector3r(*world), inverse), inverse*orientation)
            client.simSetCameraPose("front_center", relative, vehicle_name="DroneA")
            for _ in range(3):
                capture(client)  # Discard renderer transition frames.
            for repeat in range(cfg["frames_per_case"]):
                image, ranges, seg, meta = capture(client)
                frame_id = f"{case['case_id']}_{repeat+1:02d}"
                paths = {"rgb": output/f"{frame_id}_rgb.png",
                         "depth": output/f"{frame_id}_depth.npy",
                         "segmentation": output/f"{frame_id}_seg.png"}
                if not cv2.imwrite(str(paths["rgb"]), image) or not cv2.imwrite(str(paths["segmentation"]), seg):
                    raise ValueError("Media persistence failed")
                np.save(paths["depth"], ranges, allow_pickle=False)
                info = client.simGetCameraInfo("front_center", vehicle_name="DroneA")
                intrinsic = scene_intrinsics(info, image.shape[1], image.shape[0])
                pose = meta["response_camera_poses"][0]
                observed_eye = np.array(pose["position_local_ned_m"])+translation
                if np.linalg.norm(observed_eye-np.array(case["camera_world_ned_m"])) > .02:
                    raise ValueError("Actual virtual camera origin disagrees with requested world pose")
                meta.update(frame_id=frame_id, case_id=case["case_id"],
                    session_id=cfg["protocol_id"]+"_"+output.name, target_generation=1,
                    target_description=case["description"], image_size_wh=[image.shape[1],image.shape[0]],
                    media={key: {"path": str(path), "sha256": digest(path)} for key,path in paths.items()},
                    scene_intrinsics=intrinsic, camera_world_ned_m=observed_eye.tolist(),
                    world_from_local_translation_ned_m=translation.tolist(),
                    evaluation_only={"target_world_ned_m": vector(client.simGetObjectPose("T1ProbeTarget").position),
                        "target": visible_mask(seg, palette[42]),
                        "distractor": visible_mask(seg, palette[43]),
                        "occluder": visible_mask(seg, palette[44]),
                        "truth_kind": "Controlled AirSim instance-ID pixel masks, not human annotations or real-world accuracy"},
                    ground_state=ground_states(client))
                report["frames"].append(meta)
                save(output/"report.json", report)
                truth = meta["evaluation_only"]
                if "distractor_world_ned_m" not in case and truth["distractor"]["visible_pixel_count"] != 0:
                    raise ValueError("Inactive distractor unexpectedly visible: scene acceptance failed")
                if "occluder_world_ned_m" not in case and truth["occluder"]["visible_pixel_count"] != 0:
                    raise ValueError("Inactive occluder unexpectedly visible: scene acceptance failed")
                if "occluder_world_ned_m" in case and truth["target"]["visible_pixel_count"] != 0:
                    raise ValueError("Intended full target occlusion not achieved")
                if "occluder_world_ned_m" not in case and truth["target"]["visible_pixel_count"] < 500:
                    raise ValueError("Intended visible target has insufficient rendered support")
                if "distractor_world_ned_m" in case and truth["distractor"]["visible_pixel_count"] < 500:
                    raise ValueError("Intended visible distractor missing")
                meta["scene_acceptance"] = "PASS_INTENDED_VISIBLE_INSTANCES"
                save(output/"report.json", report)
            if observer is not None:
                # The callback receives description and RGB path, no actor pose or instance ID.
                report.setdefault("semantic_case_records", []).append(observer.observe(
                    client, case["case_id"], case["description"], paths["rgb"],
                    report["frames"][-1]["session_id"], output))
                save(output/"report.json", report)
            print(f"Captured camera case {case['case_id']}", flush=True)
        report["status"] = "PASS"
        report["limits"] = ["One UE session; frames and cases are nested, not independent trials",
            "Virtual camera orientation/height tests, not vehicle attitude or 3D trajectory flights",
            "Controlled masks validate rendered visible identity only; no amodal shape truth",
            "No formal algorithm effectiveness or calibrated identity probability"]
    except BaseException as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if observer is not None:
            observer.close()
        try:
            if original_camera is not None:
                client.simSetCameraPose("front_center", original_camera, vehicle_name="DroneA")
            report["ground_after"] = ground_states(client)
            client.simPause(True)
            report["paused_after"] = bool(client.simIsPause())
            report["safe_ground_closure"] = report["paused_after"] and all(
                row["landed"] and not row["api_control_enabled"] for row in report["ground_after"].values())
        except Exception as exc:
            report.update(status="FAIL", closure_error_type=type(exc).__name__, safe_ground_closure=False)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save(output/"report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--live-semantic", action="store_true")
    args = parser.parse_args()
    observer = None
    if args.live_semantic:
        from t1_live_ground_semantic import GroundSemanticObserver
        observer = GroundSemanticObserver()
    run(args.output, args.protocol, observer)
