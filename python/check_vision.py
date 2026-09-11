"""Capture unmodified front cameras; optional paused-scene presence/absence test."""
import argparse
import json
from datetime import datetime

import airsim
import cv2
import numpy as np

from setup_scene import ROOT, vector
from balloon_detector import detect_balloon, annotate


def aim_cameras_at_target(client):
    """Aim each front camera toward the configured world target for validation."""
    target = np.array([0.0, 0.0, -3.0])
    for name in ("DroneA", "DroneB"):
        vehicle = client.simGetObjectPose(name)
        camera = client.simGetCameraInfo("front_center", vehicle_name=name).pose
        # Keep the configured relative position, change only yaw in vehicle frame.
        relative = camera.position - vehicle.position
        delta = target - np.array(vector(vehicle.position))
        yaw = float(np.arctan2(delta[1], delta[0]))
        client.simSetCameraPose("front_center", airsim.Pose(relative, airsim.to_quaternion(0, 0, yaw)),
                                vehicle_name=name)


def capture(client, folder, label):
    results = {}
    for name in ("DroneA", "DroneB"):
        # Discard one frame after scene changes.
        client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name=name)
        data = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name=name)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        result = detect_balloon(frame)
        (folder / f"{label}_{name}_raw.png").write_bytes(data)
        if not cv2.imwrite(str(folder / f"{label}_{name}_detected.png"), annotate(frame, result)):
            raise RuntimeError("Could not save annotated frame")
        results[name] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--show", action="store_true", help="Open the saved four-camera comparison")
    args = parser.parse_args()
    folder = ROOT / "logs" / datetime.now().strftime("vision_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)
    client = airsim.MultirotorClient(timeout_value=10)
    report = {"mode": "paused_presence_absence_validation" if args.validate else "camera_snapshot", "status": "RUNNING"}
    try:
        report["present"] = capture(client, folder, "present")
        if args.validate:
            if not client.simIsPause():
                raise RuntimeError("Validation requires a paused simulator")
            pose = client.simGetObjectPose("BalloonTarget")
            if not np.isfinite(pose.position.to_numpy_array()).all():
                raise RuntimeError("BalloonTarget does not exist")
            try:
                if not client.simSetObjectPose("BalloonTarget", airsim.Pose(airsim.Vector3r(0,0,-100)), True):
                    raise RuntimeError("Could not move target for absence test")
                report["absent"] = capture(client, folder, "absent")
            finally:
                if not client.simSetObjectPose("BalloonTarget", pose, True):
                    raise RuntimeError("Could not restore balloon pose")
            visible = sum(1 for r in report["present"].values() if r["detected"])
            report["present_camera_count"] = visible
            if visible < 1:
                raise RuntimeError("No camera detected the visible balloon")
            if any(r["detected"] for r in report["absent"].values()):
                raise RuntimeError("False positive in absent-target frame")
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=str(exc))
        raise
    finally:
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Vision {report['status']}: {folder}")
    rows = []
    for label in ("present", "absent") if args.validate else ("present",):
        tiles = []
        for name in ("DroneA", "DroneB"):
            tile = cv2.imread(str(folder / f"{label}_{name}_detected.png"))
            tile = cv2.copyMakeBorder(tile, 35, 0, 0, 0, cv2.BORDER_CONSTANT)
            cv2.putText(tile, f"{name} / {label.upper()}", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
            tiles.append(tile)
        rows.append(np.hstack(tiles))
    preview = np.vstack(rows)
    cv2.imwrite(str(folder / "preview.png"), preview)
    if args.show:
        cv2.namedWindow("AirSim Vision - press any key to close", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AirSim Vision - press any key to close", 1000, 800)
        cv2.imshow("AirSim Vision - press any key to close", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
