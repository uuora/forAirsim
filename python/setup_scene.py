"""Verify both simulated vehicles and create a reusable balloon target.

Run after Blocks starts. Positions are WORLD NED metres, not vehicle-local NED.
Objects persist for this simulation session; rerun after restarting the level.
"""
import argparse
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def vector(v):
    return [v.x_val, v.y_val, v.z_val]


def rotate(v, q):
    p = q * airsim.Quaternionr(v.x_val, v.y_val, v.z_val, 0) * q.inverse()
    return airsim.Vector3r(p.x_val, p.y_val, p.z_val)


def capture_overview(client, output):
    """Temporarily relocate a camera, then restore its original relative pose."""
    vehicle = client.simGetVehiclePose(vehicle_name="DroneA")
    camera = client.simGetCameraInfo("front_center", vehicle_name="DroneA").pose
    inverse = vehicle.orientation.inverse()
    original = airsim.Pose(rotate(camera.position - vehicle.position, inverse),
                          inverse * camera.orientation)
    world = client.simGetObjectPose("DroneA").position
    eye = airsim.Vector3r(-12, -14, -8)
    direction = airsim.Vector3r(-2, 0, -1) - eye
    orientation = airsim.to_quaternion(
        math.atan2(-direction.z_val, math.hypot(direction.x_val, direction.y_val)),
        0, math.atan2(direction.y_val, direction.x_val))
    overview = airsim.Pose(rotate(eye - world, inverse), inverse * orientation)
    try:
        client.simSetCameraPose("front_center", overview, vehicle_name="DroneA")
        # Discard the first frame to allow the render capture to update.
        client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name="DroneA")
        data = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name="DroneA")
        if not data:
            raise RuntimeError("Empty overview image")
        output.write_bytes(data)
    finally:
        client.simSetCameraPose("front_center", original, vehicle_name="DroneA")


def ensure_mesh(client, name, asset, position, scale):
    pose = airsim.Pose(airsim.Vector3r(*position), airsim.Quaternionr())
    if name in client.simListSceneObjects():
        if not client.simSetObjectPose(name, pose, teleport=True):
            raise RuntimeError(f"Cannot position {name}")
        if not client.simSetObjectScale(name, airsim.Vector3r(*scale)):
            raise RuntimeError(f"Cannot scale {name}")
    else:
        actual = client.simSpawnObject(name, asset, pose, airsim.Vector3r(*scale), False)
        if actual != name:
            raise RuntimeError(f"Unexpected object name: {actual}")
    observed = vector(client.simGetObjectPose(name).position)
    if not np.allclose(observed, position, atol=0.02):
        raise RuntimeError(f"{name} pose mismatch: {observed}")
    return observed


def validate_target_offset(values):
    offset = np.array(values if values is not None else [0.0, 0.0, 0.0], dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError("Target offset must be three finite values")
    if abs(offset[0]) > 1.0 or abs(offset[1]) > 1.0 or abs(offset[2]) > 0.5:
        raise ValueError("Target offset exceeds the inspected Blocks test region")
    return offset


def main(target_offset=None):
    target_offset = validate_target_offset(target_offset)
    run_dir = ROOT / "logs" / datetime.now().strftime("scene_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(run_dir / "scene.log", encoding="utf-8")])
    report = {"status": "RUNNING", "timestamp": datetime.now().astimezone().isoformat()}
    try:
        cfg = json.loads((ROOT / "configs/balloon.json").read_text(encoding="utf-8"))
        client = airsim.MultirotorClient(timeout_value=20)
        vehicles = client.listVehicles()
        logging.info("Vehicles: %s", vehicles)
        if not {"DroneA", "DroneB"}.issubset(vehicles):
            raise RuntimeError(f"Expected DroneA and DroneB, got {vehicles}")
        report["vehicles"] = {}
        for name in ("DroneA", "DroneB"):
            state = client.getMultirotorState(vehicle_name=name)
            world = vector(client.simGetObjectPose(name).position)
            if not all(math.isfinite(v) for v in world):
                raise RuntimeError(f"Invalid world pose: {name}")
            report["vehicles"][name] = {
                "world_ned_m": world,
                "vehicle_local_ned_m": vector(state.kinematics_estimated.position),
                "landed_state": state.landed_state,
            }
            logging.info("%s world NED: %s", name, world)
        assets = set(client.simListAssets())
        resolved = {}
        for required in ("Sphere", "Cylinder"):
            matches = [asset for asset in assets if asset.casefold() == required.casefold()]
            if len(matches) != 1:
                raise RuntimeError(f"Required asset {required} missing or ambiguous: {matches}")
            resolved[required] = matches[0]
        name, scale = cfg["name"], cfg["scale"]
        position = (np.array(cfg["position_ned_m"], dtype=float) + target_offset).tolist()
        ensure_mesh(client, name, resolved["Sphere"], position, scale)
        # Engine basic sphere/cylinder are approximately 1 metre at unit scale.
        string_z = position[2] + scale[2] / 2 + cfg["string_length_m"] / 2
        ensure_mesh(client, name + "_String", resolved["Cylinder"],
                    [position[0], position[1], string_z],
                    [0.012, 0.012, cfg["string_length_m"]])
        texture = run_dir / "balloon_red.png"
        rgb = cfg["color_rgb"]
        if not cv2.imwrite(str(texture), np.full((16, 16, 3), rgb[::-1], dtype=np.uint8)):
            raise RuntimeError("Could not write balloon texture")
        if not client.simSetObjectMaterialFromTexture(name, str(texture)):
            raise RuntimeError("Could not apply red balloon material")
        if not client.simSetSegmentationObjectID(name, cfg["segmentation_id"]):
            raise RuntimeError("Could not set balloon segmentation ID")
        # AirSim getter lowercases mesh names but does not normalize its input.
        observed_id = client.simGetSegmentationObjectID(name.lower())
        if observed_id != cfg["segmentation_id"]:
            raise RuntimeError(f"Unexpected segmentation ID: {observed_id}")
        report["balloon"] = dict(cfg, requested_position_ned_m=position,
                                 target_offset_ned_m=target_offset.tolist(),
                                 observed_position_ned_m=vector(client.simGetObjectPose(name).position),
                                 observed_scale=vector(client.simGetObjectScale(name)),
                                 physics_enabled=False, flight_contact_test="NOT_RUN")
        if not np.allclose(report["balloon"]["observed_scale"], scale, atol=0.001):
            raise RuntimeError("Balloon scale mismatch")
        # Checks blocking geometry in the target region, not a drone contact test.
        voxel_path = run_dir / "balloon_collision.binvox"
        if not client.simCreateVoxelGrid(airsim.Vector3r(*position), 2, 2, 2, 0.1, str(voxel_path)):
            raise RuntimeError("Could not sample target collision region")
        encoded = voxel_path.read_bytes().split(b"data\n", 1)[1]
        occupied = sum(count for value, count in zip(encoded[::2], encoded[1::2]) if value)
        if occupied == 0:
            raise RuntimeError("No blocking geometry detected around the target")
        report["balloon"]["region_blocking_voxels"] = occupied
        logging.info("Target region blocking voxels: %s", occupied)
        # Capture existing camera feeds without changing vehicle or camera poses.
        for vehicle in ("DroneA", "DroneB"):
            data = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name=vehicle)
            if not data:
                raise RuntimeError(f"Empty camera image: {vehicle}")
            (run_dir / f"{vehicle}_front.png").write_bytes(data)
        capture_overview(client, run_dir / "overview.png")
        report["status"] = "PASS"
        logging.info("SCENE_SETUP_PASS: %s at %s; output %s", name, position, run_dir)
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        logging.exception("Scene setup failed")
        raise
    finally:
        (run_dir / "scene_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"))
    main(parser.parse_args().target_offset)
