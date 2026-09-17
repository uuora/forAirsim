"""Stationary camera plumbing audit. Simulator depth/poses are reference data."""
import hashlib
import time

import airsim
import cv2
import msgpack
import numpy as np

from camera_geometry import scene_intrinsics


def xyz(value):
    return np.array([value.x_val, value.y_val, value.z_val], dtype=float)


def wxyz(value):
    return [float(value.w_val), float(value.x_val), float(value.y_val), float(value.z_val)]


def rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Invalid orientation quaternion")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def audit_cameras(client, settings, folder, pairs=3, stationary_tolerance_m=.01):
    if msgpack.Unpacker.__module__ == "msgpack.fallback":
        raise RuntimeError("Float depth capture needs the compiled MessagePack decoder; see docs/传感器验收与限制.md")
    out = folder / "cameras"
    out.mkdir()
    records, errors, pair_deltas = [], [], []
    vehicles = list(settings["Vehicles"])
    for pair in range(pairs):
        stamps = []
        for vehicle in vehicles:
            camera = "front_center"
            config = settings["Vehicles"][vehicle]["Cameras"][camera]
            body0 = client.simGetVehiclePose(vehicle_name=vehicle)
            world0 = client.simGetObjectPose(vehicle)
            host_start = time.monotonic()
            rgb, depth = client.simGetImages([
                airsim.ImageRequest(camera, airsim.ImageType.Scene, False, True),
                airsim.ImageRequest(camera, airsim.ImageType.DepthPlanar, True, False),
            ], vehicle_name=vehicle)
            host_end = time.monotonic()
            world1 = client.simGetObjectPose(vehicle)
            body1 = client.simGetVehiclePose(vehicle_name=vehicle)
            info = client.simGetCameraInfo(camera, vehicle_name=vehicle)
            frame = cv2.imdecode(np.frombuffer(bytes(rgb.image_data_uint8), np.uint8), cv2.IMREAD_COLOR)
            if rgb.message or depth.message or frame is None:
                raise RuntimeError(f"{vehicle}: RGB/depth capture error: {rgb.message} {depth.message}")
            expected = next(x for x in config["CaptureSettings"] if x["ImageType"] == 0)
            depth_config = next(x for x in config["CaptureSettings"] if x["ImageType"] == 1)
            if (frame.shape[:2] != (expected["Height"], expected["Width"])
                    or (depth.height, depth.width) != (depth_config["Height"], depth_config["Width"])
                    or len(depth.image_data_float) != depth.width * depth.height):
                raise RuntimeError(f"{vehicle}: unexpected camera dimensions")
            depth_m = np.array(depth.image_data_float, dtype=np.float32).reshape(depth.height, depth.width)
            positive = np.isfinite(depth_m) & (depth_m > 0)
            if not positive.any():
                errors.append(f"{vehicle}: no positive finite depth pixels")
            if min(rgb.time_stamp, depth.time_stamp) <= 0:
                errors.append(f"{vehicle}: invalid image timestamp")
            rgb_path = out / f"{vehicle}_{pair:02d}_rgb.png"
            depth_path = out / f"{vehicle}_{pair:02d}_depth_planar_m.npy"
            rgb_path.write_bytes(bytes(rgb.image_data_uint8))
            np.save(depth_path, depth_m)
            offset0 = xyz(world0.position) - xyz(body0.position)
            offset1 = xyz(world1.position) - xyz(body1.position)
            finite_geometry = np.isfinite(np.concatenate([
                offset0, offset1, xyz(rgb.camera_position), wxyz(rgb.camera_orientation)])).all()
            if not finite_geometry:
                raise RuntimeError(f"{vehicle}: invalid pose geometry")
            movement = float(np.linalg.norm(xyz(body1.position) - xyz(body0.position)))
            offset_drift = float(np.linalg.norm(offset1 - offset0))
            # World/local NED have equal axes in this AirSim source; origins differ.
            # Offset is derived from simulator reference, not measured by GPS/vision.
            global_camera = xyz(rgb.camera_position) + (offset0 + offset1) / 2
            mount = np.array([config[axis] for axis in ("X", "Y", "Z")])
            observed_mount = rotation(wxyz(body1.orientation)).T @ (xyz(rgb.camera_position) - xyz(body1.position))
            mount_error = float(np.linalg.norm(observed_mount - mount))
            if movement > stationary_tolerance_m or offset_drift > .01 or mount_error > .02:
                errors.append(f"{vehicle}: stationary geometry check failed; inspect motion or mounting")
            intrinsics = scene_intrinsics(info, rgb.width, rgb.height)
            if abs(intrinsics["horizontal_fov_deg"] - expected["FOV_Degrees"]) > .01:
                errors.append(f"{vehicle}: Scene projection disagrees with configured FOV")
            records.append({
                "pair_id": pair, "vehicle": vehicle,
                "rgb": str(rgb_path.relative_to(folder)), "depth_planar_m": str(depth_path.relative_to(folder)),
                "rgb_sha256": hashlib.sha256(rgb_path.read_bytes()).hexdigest(),
                "depth_sha256": hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                "rgb_timestamp_ns": int(rgb.time_stamp), "depth_timestamp_ns": int(depth.time_stamp),
                "rgb_depth_timestamp_delta_ms": abs(rgb.time_stamp-depth.time_stamp)/1e6,
                "host_rpc_latency_ms": (host_end-host_start)*1000,
                "positive_finite_depth_fraction": float(positive.mean()),
                "center_depth_planar_m": float(depth_m[depth.height//2, depth.width//2]),
                "vehicle_local_to_global_translation_m": offset0.tolist(),
                "offset_repeat_difference_m": offset_drift, "body_motion_during_rpc_m": movement,
                "camera_local_position_m": xyz(rgb.camera_position).tolist(),
                "camera_global_position_reference_m": global_camera.tolist(),
                "camera_orientation_wxyz": wxyz(rgb.camera_orientation),
                "mount_observed_body_m": observed_mount.tolist(), "mount_position_error_m": mount_error,
                **intrinsics,
                "depth_source": "ideal_simulator_render_reference_not_stereo_estimation",
            })
            stamps.append(rgb.time_stamp)
            if pair == 0:
                # Fixed display clipping is only for the preview, not the saved metres.
                visible_depth = np.where(positive, np.clip(depth_m, 0, 50)/50*255, 0).astype(np.uint8)
                colored = cv2.applyColorMap(visible_depth, cv2.COLORMAP_TURBO)
                preview = np.hstack([frame, colored])
                cv2.putText(preview, vehicle + " RGB | simulator depth 0-50 m (clipped display)",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
                cv2.imwrite(str(out / f"{vehicle}_rgb_depth_preview.jpg"), preview)
        pair_deltas.append((max(stamps)-min(stamps))/1e6)
    baselines = []
    if len(vehicles) == 2:
        for pair in range(pairs):
            rows = [r for r in records if r["pair_id"] == pair]
            positions = [np.array(r["camera_global_position_reference_m"]) for r in rows]
            baselines.append(float(np.linalg.norm(positions[1]-positions[0])))
    for vehicle in vehicles:
        stamps = [r["rgb_timestamp_ns"] for r in records if r["vehicle"] == vehicle]
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            errors.append(f"{vehicle}: camera timestamp did not advance monotonically")
    return {
        "status": "FAIL" if errors else "PASS", "errors": errors, "records": records,
        "stationary_tolerance_m": stationary_tolerance_m,
        "cross_vehicle_rgb_delta_ms": pair_deltas, "camera_baseline_reference_m": baselines,
        "stereo_ready": False,
        "limitations": ["Cross-vehicle calls are sequential, not synchronized stereo.",
                        "Same-call RGB/depth timestamps are measured, not a hardware sync claim.",
                        "Global translations use simulator reference, not a deployable localization estimate.",
                        "Geometry acceptance requires stationary vehicles; no moving-platform calibration.",
                        "No lens calibration, rectification, stereo matching or depth accuracy evaluation.",
                        "DepthPlanar is optical-axis depth in metres, not Euclidean range.",
                        "CameraInfo.fov can describe the viewer rather than Scene; use Scene projection for K.",
                        "Camera pose axes: forward/right/down; OpenCV optical axes: right/down/forward."],
    }
