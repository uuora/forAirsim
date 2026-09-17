"""Independent static maritime fixture and RGB capture; never commands flight.

All labels are simulator metadata, not detector output. Cameras are temporarily
staged in the scene and restored; these are not real airborne stereo observations.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import airsim
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "MarineLab_"
CLASSES = {"vessel", "lifebuoy", "floating_supply", "person_proxy", "floating_debris"}


def xyz(vector):
    return [vector.x_val, vector.y_val, vector.z_val]


def same_orientation(first, second):
    a = np.array([first.x_val, first.y_val, first.z_val, first.w_val])
    b = np.array([second.x_val, second.y_val, second.z_val, second.w_val])
    return np.allclose(a, b, atol=0.002) or np.allclose(a, -b, atol=0.002)


def rotate(vector, quaternion):
    result = quaternion * airsim.Quaternionr(*xyz(vector), 0) * quaternion.inverse()
    return airsim.Vector3r(result.x_val, result.y_val, result.z_val)


def validate_config(cfg):
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported maritime scene schema")
    ids, segments = set(), set()
    for item in cfg["objects"]:
        key = item["instance_id"]
        if not key or not all(c.isalnum() or c == "_" for c in key) or key in ids:
            raise ValueError("Invalid or duplicate instance ID")
        if item["class_name"] not in CLASSES:
            raise ValueError("Unsupported class")
        segment = item["segmentation_id"]
        if type(segment) is not int or not 180 <= segment <= 219 or segment in segments:
            raise ValueError("Invalid or duplicate segmentation ID")
        ids.add(key)
        segments.add(segment)
    names = [v["name"] for v in cfg["views"]]
    if len(names) != len(set(names)) or any(not n.isidentifier() for n in names):
        raise ValueError("Invalid or duplicate view name")
    vectors = [cfg["origin_world_ned_m"]]
    vectors += [i["position_relative_m"] for i in cfg["objects"]]
    for view in cfg["views"]:
        vectors += [view["eye_relative_m"], view["look_at_relative_m"]]
        if view["eye_relative_m"] == view["look_at_relative_m"]:
            raise ValueError("Camera eye equals look-at position")
    if any(len(v) != 3 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in v) for v in vectors):
        raise ValueError("Scene vectors must contain three finite numbers")


def parts_for(class_name):
    """Names, primitive kinds, offsets, scales and colors; geometry is illustrative."""
    parts = []
    def add(name, asset, pos, scale, color):
        parts.append(dict(part=name, asset=asset, offset=pos, scale=scale, rgb=color))
    orange, white, dark = [242, 101, 35], [230, 236, 234], [28, 43, 57]
    if class_name == "vessel":
        add("hull", "Sphere", [0, 0, -0.35], [5.5, 2.1, 1.0], dark)
        add("deck", "box", [-0.2, 0, -0.75], [3.8, 1.7, 0.18], white)
        add("cabin", "box", [-0.7, 0, -1.22], [1.8, 1.3, 0.85], white)
        add("roof", "box", [-0.7, 0, -1.71], [2.0, 1.5, 0.15], orange)
        add("window_left", "box", [-0.7, -0.661, -1.3], [1.3, 0.03, 0.35], [41, 137, 175])
        add("window_right", "box", [-0.7, 0.661, -1.3], [1.3, 0.03, 0.35], [41, 137, 175])
        add("mast", "Cylinder", [-0.7, 0, -2.05], [0.08, 0.08, 0.7], white)
    elif class_name == "lifebuoy":
        for i in range(16):
            theta = i * 2 * math.pi / 16
            add(f"ring_{i:02}", "Sphere", [0.6*math.cos(theta), 0.6*math.sin(theta), -0.13], [0.34, 0.34, 0.22], white if i % 4 == 0 else orange)
    elif class_name == "floating_supply":
        add("crate", "box", [0, 0, -0.42], [1.2, 1.0, 0.8], [192, 133, 63])
        add("strap_x", "box", [0, 0, -0.83], [1.22, 0.15, 0.03], dark)
        add("strap_y", "box", [0, 0, -0.84], [0.15, 1.02, 0.03], dark)
    elif class_name == "person_proxy":
        add("vest", "box", [0, 0, -0.25], [0.65, 0.48, 0.28], orange)
        add("head", "Sphere", [0.53, 0, -0.29], [0.32, 0.32, 0.32], [216, 174, 134])
        for side in (-1, 1):
            add(f"arm_{side+1}", "box", [0.06, side*0.36, -0.16], [0.65, 0.14, 0.14], [216, 174, 134])
            add(f"leg_{side+1}", "box", [-0.64, side*0.14, -0.12], [0.72, 0.16, 0.14], [41, 61, 87])
    elif class_name == "floating_debris":
        add("panel", "box", [0, 0, -0.12], [1.8, 0.6, 0.2], [102, 82, 57])
        add("crosspiece", "box", [0.3, 0.1, -0.23], [0.25, 1.4, 0.14], [128, 102, 63])
    return parts


def ensure_part(client, name, asset, position, scale, texture, segment, existing):
    pose = airsim.Pose(airsim.Vector3r(*position), airsim.Quaternionr())
    if name in existing:
        if not client.simSetObjectPose(name, pose, teleport=True) or not client.simSetObjectScale(name, airsim.Vector3r(*scale)):
            raise RuntimeError(f"Cannot update {name}")
    else:
        actual = client.simSpawnObject(name, asset, pose, airsim.Vector3r(*scale), False)
        if actual != name:
            raise RuntimeError(f"Unexpected actor name: {actual}")
        existing.add(name)
    if not client.simSetObjectMaterialFromTexture(name, str(texture)):
        raise RuntimeError(f"Cannot apply texture: {name}")
    if not client.simSetSegmentationObjectID(name, segment, False):
        raise RuntimeError(f"Cannot set annotation ID: {name}")
    if client.simGetSegmentationObjectID(name.lower()) != segment:
        raise RuntimeError(f"Annotation ID mismatch: {name}")
    if not np.allclose(xyz(client.simGetObjectPose(name).position), position, atol=0.02):
        raise RuntimeError(f"Actor position mismatch: {name}")
    if not np.allclose(xyz(client.simGetObjectScale(name)), scale, atol=0.002):
        raise RuntimeError(f"Actor scale mismatch: {name}")


def capture_view(client, view, origin, output):
    vehicle = view["vehicle"]
    pose = client.simGetVehiclePose(vehicle_name=vehicle)
    info = client.simGetCameraInfo("front_center", vehicle_name=vehicle)
    inverse = pose.orientation.inverse()
    original = airsim.Pose(rotate(info.pose.position-pose.position, inverse), inverse*info.pose.orientation)
    body_world = client.simGetObjectPose(vehicle).position
    eye = airsim.Vector3r(*(np.array(origin)+view["eye_relative_m"]))
    focus = airsim.Vector3r(*(np.array(origin)+view["look_at_relative_m"]))
    direction = focus-eye
    orientation = airsim.to_quaternion(math.atan2(-direction.z_val, math.hypot(direction.x_val, direction.y_val)), 0, math.atan2(direction.y_val, direction.x_val))
    staged = airsim.Pose(rotate(eye-body_world, inverse), inverse*orientation)
    try:
        client.simSetCameraPose("front_center", staged, vehicle_name=vehicle)
        client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name=vehicle)
        image = client.simGetImage("front_center", airsim.ImageType.Scene, vehicle_name=vehicle)
        decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR) if image else None
        if decoded is None or decoded.size == 0:
            raise RuntimeError(f"Empty/invalid image: {vehicle}")
        (output/f"{view['name']}.png").write_bytes(image)
    finally:
        client.simSetCameraPose("front_center", original, vehicle_name=vehicle)
        restored = client.simGetCameraInfo("front_center", vehicle_name=vehicle)
        if not np.allclose(xyz(restored.pose.position), xyz(info.pose.position), atol=0.03) or not same_orientation(restored.pose.orientation, info.pose.orientation):
            raise RuntimeError(f"Camera restoration failed: {vehicle}")
    return dict(name=view["name"], vehicle=vehicle, file=f"{view['name']}.png", width=int(decoded.shape[1]), height=int(decoded.shape[0]), camera_world_ned_m=xyz(eye), camera_restored=True, source="staged_virtual_camera", simultaneous_stereo=False)


def run(config_path):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)
    output = ROOT/"logs"/datetime.now().strftime("maritime_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    report = dict(status="RUNNING", timestamp=datetime.now().astimezone().isoformat(), scene_config=cfg, detection_status="NOT_RUN", localization_status="NOT_RUN", flight_commands_issued=False, objects=[], views=[])
    try:
        client = airsim.MultirotorClient(timeout_value=15)
        vehicles = client.listVehicles()
        report["vehicles"] = vehicles
        for view in cfg["views"]:
            name = view["vehicle"]
            if name not in vehicles:
                raise RuntimeError(f"Missing vehicle: {name}")
            if client.getMultirotorState(vehicle_name=name).landed_state != airsim.LandedState.Landed or client.isApiControlEnabled(vehicle_name=name):
                raise RuntimeError("Scene setup requires landed vehicles with no active API control")
        assets = set(client.simListAssets())
        box = next((a for a in ("1M_Cube_Chamfer", "EditorCube", "Cube") if a in assets), None)
        if not box or not {"Sphere", "Cylinder"}.issubset(assets):
            raise RuntimeError("Required built-in primitive assets unavailable")
        report["box_asset"] = box
        existing = set(client.simListSceneObjects())
        textures = output/"textures"
        textures.mkdir()
        def texture(rgb):
            path = textures/("color_"+"_".join(map(str, rgb))+".png")
            if not path.exists() and not cv2.imwrite(str(path), np.full((16,16,3), rgb[::-1], dtype=np.uint8)):
                raise RuntimeError("Texture write failed")
            return path
        origin = np.array(cfg["origin_world_ned_m"])
        # Large flat visual backdrop only; placed away from the existing experiment.
        ensure_part(client, PREFIX+"WaterBackdrop", box, origin+[0,0,0.15], [250,250,0.2], texture([12,43,64]), 180, existing)
        for item in cfg["objects"]:
            actor_names = []
            for part in parts_for(item["class_name"]):
                name = PREFIX+item["instance_id"]+"_"+part["part"]
                position = origin+item["position_relative_m"]+np.array(part["offset"])
                ensure_part(client, name, box if part["asset"]=="box" else part["asset"], position, part["scale"], texture(part["rgb"]), item["segmentation_id"], existing)
                actor_names.append(name)
            report["objects"].append(dict(**item, actor_names=actor_names, label_source="scene_definition_not_prediction"))
        for view in cfg["views"]:
            report["views"].append(capture_view(client, view, origin, output))
        panels = []
        for view in report["views"]:
            panel = cv2.imread(str(output/view["file"]))
            cv2.rectangle(panel, (0,0), (panel.shape[1],46), (25,29,35), -1)
            cv2.putText(panel, view["name"]+" | staged camera | NOT detector output", (12,29), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245,245,245), 1, cv2.LINE_AA)
            panels.append(panel)
        if panels and len({p.shape for p in panels})==1:
            if not cv2.imwrite(str(output/"overview.png"), np.hstack(panels)):
                raise RuntimeError("Overview write failed")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc))
        raise
    finally:
        (output/"scene_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs"/"maritime_scene.json")
    args = parser.parse_args()
    run(args.config)
