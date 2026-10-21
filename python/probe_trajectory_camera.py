"""Ground-only camera/clock/coordinate probe; never takes API control or flies."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import airsim
import cv2
import numpy as np

from setup_scene import ROOT, vector
from timestamped_target_provider import TimestampedTargetProvider


def main(output_dir):
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "kind": "GROUND_CAMERA_COORDINATE_CLOCK_PROBE",
              "flight_commands_executed": False, "api_control_taken": False,
              "ai_assistance": "probe code and report authored with AI assistance",
              "observations": [], "source_sha256": {
                  name: hashlib.sha256((ROOT / "python" / name).read_bytes()).hexdigest()
                  for name in ("probe_trajectory_camera.py", "timestamped_target_provider.py")}}
    client = airsim.MultirotorClient(timeout_value=10)
    try:
        client.simPause(True)
        ground = {}
        for vehicle in ("DroneA", "DroneB"):
            state = client.getMultirotorState(vehicle_name=vehicle)
            world = np.asarray(vector(client.simGetObjectPose(vehicle).position))
            local = np.asarray(vector(state.kinematics_estimated.position))
            speed = float(np.linalg.norm(vector(state.kinematics_estimated.linear_velocity)))
            control = client.isApiControlEnabled(vehicle_name=vehicle)
            if control or state.landed_state != airsim.LandedState.Landed or speed >= 0.08:
                raise RuntimeError(f"Ground-only probe requires landed, released, stationary {vehicle}")
            if not np.isfinite(world).all() or not np.isfinite(local).all():
                raise RuntimeError("Invalid calibration pose")
            ground[vehicle] = {"world_ned_m": world.tolist(), "local_ned_m": local.tolist(),
                               "world_from_local_translation_ned_m": (world - local).tolist(),
                               "speed_m_s": speed, "api_control": control, "landed": True}
        report["ground_calibration"] = ground
        # Resume simulation clock only. Both vehicles remain landed and released.
        client.simPause(False)
        provider = TimestampedTargetProvider(client, coordinate_frame="VEHICLE_LOCAL_NED")
        for index in range(3):
            observation = provider.capture("DroneA")
            row = observation.metadata()
            if observation.image is None or observation.depth is None:
                raise RuntimeError("RGB/depth interface returned missing images")
            if observation.image.shape[:2] != observation.depth.shape:
                raise RuntimeError("Probe requires matching RGB/depth resolution")
            readback_age = observation.capture_age_at_receipt_s()
            protection_age = observation.age_at()
            if not np.isfinite(readback_age) or readback_age < 0 or not np.isfinite(protection_age):
                raise RuntimeError("Invalid image/state time comparison")
            row.update(readback_to_query_age_sim_s=readback_age,
                       age_proxy_at_save_s=protection_age,
                       world_from_local_translation_ned_m=ground["DroneA"]["world_from_local_translation_ned_m"])
            image_path = folder / f"rgb_{index + 1:02d}.png"
            depth_path = folder / f"depth_{index + 1:02d}.npy"
            if not cv2.imwrite(str(image_path), observation.image):
                raise RuntimeError("Could not preserve probe image")
            np.save(depth_path, observation.depth, allow_pickle=False)
            row.update(raw_rgb=str(image_path), raw_rgb_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                       raw_depth=str(depth_path), raw_depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest())
            report["observations"].append(row)
            time.sleep(0.12)
        # These checks prove only local interface availability in this session.
        report["status"] = "PASS"
        report["limits"] = ["Ground camera need not see the balloon", "No target estimation accuracy measured",
                            "Image timestamp is render readback completion, not exact exposure",
                            "Simulator and host clock domains retained separately; summed age is a protection proxy",
                            "No movement or trajectory tracking validation"]
    except BaseException as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            client.simPause(True)
            report["paused"] = True
            report["vehicles_released_and_grounded_at_end"] = all(
                not client.isApiControlEnabled(vehicle_name=vehicle)
                and client.getMultirotorState(vehicle_name=vehicle).landed_state == airsim.LandedState.Landed
                for vehicle in ("DroneA", "DroneB"))
            if not report["vehicles_released_and_grounded_at_end"]:
                report["status"] = "FAIL"
        except Exception as exc:
            report.update(status="FAIL", paused=False, pause_error=str(exc))
        report["finished_local"] = datetime.now().astimezone().isoformat()
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        print(folder)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    raise SystemExit(main(parser.parse_args().output_dir))
