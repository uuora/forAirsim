"""Compare parked and airborne sensor/image quality at a bounded hover point."""
import hashlib
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np

from check_onboard_sensors import assess_sequence
from onboard_sensors import OnboardSensorMonitor
from sensor_camera_audit import audit_cameras

ROOT = Path(__file__).resolve().parents[1]
VEHICLES = ("DroneA", "DroneB")
CAMERA = "front_center"
OBSERVATION_LOCAL_NED_M = np.array([0.0, 0.0, -6.0])


def vec(value):
    return np.array([value.x_val, value.y_val, value.z_val], dtype=float)


def pose_snapshot(client, vehicle):
    state = client.getMultirotorState(vehicle_name=vehicle)
    collision = client.simGetCollisionInfo(vehicle_name=vehicle)
    return {
        "local_ned_m": vec(state.kinematics_estimated.position).tolist(),
        "world_ned_reference_m": vec(client.simGetObjectPose(vehicle).position).tolist(),
        "linear_speed_m_s": float(np.linalg.norm(vec(state.kinematics_estimated.linear_velocity))),
        "landed_state": int(state.landed_state),
        "collision_timestamp_ns": int(collision.time_stamp),
        "collision_has_collided": bool(collision.has_collided),
        "collision_object": str(collision.object_name),
    }


def wait_positions(client, targets, collision_timestamps, trace_path, timeout=35, hold=2.0,
                   position_tolerance=.35, speed_tolerance=.30):
    started, stable = time.monotonic(), None
    samples = []
    with trace_path.open("w", encoding="utf-8") as stream:
        while time.monotonic() - started < timeout:
            row, ready = {}, True
            for vehicle, target in targets.items():
                state = client.getMultirotorState(vehicle_name=vehicle)
                position = vec(state.kinematics_estimated.position)
                speed = float(np.linalg.norm(vec(state.kinematics_estimated.linear_velocity)))
                collision = client.simGetCollisionInfo(vehicle_name=vehicle)
                if collision.time_stamp != collision_timestamps[vehicle]:
                    raise RuntimeError(f"{vehicle} collision changed during observation flight: {collision.object_name}")
                error = float(np.linalg.norm(position - target))
                row[vehicle] = {"local_ned_m": position.tolist(), "position_error_m": error,
                                "linear_speed_m_s": speed}
                ready &= error < position_tolerance and speed < speed_tolerance
            sample={"elapsed_s":time.monotonic()-started,"vehicles":row}
            samples.append(sample); stream.write(json.dumps(sample)+"\n"); stream.flush()
            stable = (stable or time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable >= hold:
                return samples
            time.sleep(.2)
    last=samples[-1]["vehicles"] if samples else {}
    raise TimeoutError(f"Observation positions did not become stable; last={last}")


def land_and_verify(client, vehicles, ground_local_z, timeout=45):
    """Approach the pad, land, and require a stable landed-state before disarm."""
    approaches=[client.moveToZAsync(-1.0,1.0,timeout_sec=20,vehicle_name=v) for v in vehicles]
    for future in approaches: future.join()
    lands=[client.landAsync(timeout_sec=timeout,vehicle_name=v) for v in vehicles]
    for future in lands: future.join()
    deadline=time.monotonic()+20; stable=None; evidence={}
    while time.monotonic()<deadline:
        ready=True
        for vehicle in vehicles:
            state=client.getMultirotorState(vehicle_name=vehicle)
            speed=float(np.linalg.norm(vec(state.kinematics_estimated.linear_velocity)))
            position=vec(state.kinematics_estimated.position)
            pose_grounded=abs(position[2]-ground_local_z[vehicle])<.15 and speed<.10
            flag_grounded=state.landed_state==airsim.LandedState.Landed and speed<.10
            evidence[vehicle]={"landed_state":int(state.landed_state),"speed_m_s":speed,
                               "local_ned_m":position.tolist(),
                               "ground_z_reference_m":ground_local_z[vehicle],
                               "grounded_by_flag_or_reference_pose":flag_grounded or pose_grounded}
            ready &= flag_grounded or pose_grounded
        stable=(stable or time.monotonic()) if ready else None
        if stable is not None and time.monotonic()-stable>=1.0:
            return evidence
        time.sleep(.25)
    raise TimeoutError(f"Landing did not become stable: {evidence}")


def sample_sensors(client, folder, duration=3.5, interval=.25, max_stale=1.5):
    monitor = OnboardSensorMonitor(client, max_stale_s=max_stale)
    samples, started = [], time.monotonic()
    with (folder / "sensor_samples.jsonl").open("w", encoding="utf-8") as stream:
        while True:
            sample = monitor.read_all(VEHICLES)
            samples.append(sample)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            if time.monotonic() - started >= duration:
                break
            time.sleep(interval)
    failed, summary = assess_sequence(samples, VEHICLES, max_stale)
    return {"polls": len(samples), "failed_sensors": failed, "summary": summary}


def setup_detection(client):
    registry = json.loads((ROOT / "configs/maritime_targets.json").read_text(encoding="utf-8"))
    scene = client.simListSceneObjects("MH2_.*")
    resolved = {}
    for target in registry["targets"]:
        matches = [name for name in scene if re.fullmatch(target["actor_pattern"], name)]
        if len(matches) != target["expected_components"]:
            raise RuntimeError(f"Target registry mismatch: {target['id']}: {matches}")
        resolved[target["id"]] = matches
    for vehicle in VEHICLES:
        client.simClearDetectionMeshNames(CAMERA, airsim.ImageType.Scene, vehicle_name=vehicle)
        client.simSetDetectionFilterRadius(CAMERA, airsim.ImageType.Scene, 20000, vehicle_name=vehicle)
        for names in resolved.values():
            for name in names:
                client.simAddDetectionFilterMeshName(CAMERA, airsim.ImageType.Scene,
                                                     name, vehicle_name=vehicle)
    return registry, resolved


def target_visibility(client, registry, resolved, camera_records, stage_folder):
    output = {}
    for vehicle in VEHICLES:
        detections = client.simGetDetections(CAMERA, airsim.ImageType.Scene, vehicle_name=vehicle)
        boxes = {item.name: [float(item.box2D.min.x_val), float(item.box2D.min.y_val),
                             float(item.box2D.max.x_val), float(item.box2D.max.y_val)]
                 for item in detections}
        record = next(row for row in camera_records if row["vehicle"] == vehicle and row["pair_id"] == 0)
        image_path = stage_folder / record["rgb"]
        frame = cv2.imread(str(image_path))
        targets, returned = [], 0
        for target in registry["targets"]:
            components = resolved[target["id"]]
            parts = [boxes[name] for name in components if name in boxes]
            item = {"target_id": target["id"], "returned_components": len(parts),
                    "expected_components": target["expected_components"],
                    "status": "returned" if parts else "not_returned_visibility_unknown"}
            if parts:
                x1=max(0,min(x[0] for x in parts)); y1=max(0,min(x[1] for x in parts))
                x2=min(frame.shape[1],max(x[2] for x in parts)); y2=min(frame.shape[0],max(x[3] for x in parts))
                if x2 > x1 and y2 > y1:
                    item["bbox_xyxy_pixels"] = [x1,y1,x2,y2]
                    item["bbox_area_fraction"] = (x2-x1)*(y2-y1)/(frame.shape[0]*frame.shape[1])
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),2)
                    cv2.putText(frame,target["id"],(int(x1),max(20,int(y1)-5)),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                returned += 1
            targets.append(item)
        review = stage_folder / f"{vehicle}_target_review.jpg"
        cv2.imwrite(str(review), frame)
        output[vehicle] = {"targets_returned": returned, "targets": targets,
                           "review_image": str(review.relative_to(stage_folder))}
    return output


def image_quality(stage_folder, records):
    result = {}
    for record in records:
        if record["pair_id"] != 0:
            continue
        frame = cv2.imread(str(stage_folder / record["rgb"]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
        probability = histogram[histogram > 0] / histogram.sum()
        metrics = {
            "mean_luminance": float(gray.mean()), "luminance_std": float(gray.std()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "dark_fraction_le_5": float((gray <= 5).mean()),
            "bright_fraction_ge_250": float((gray >= 250).mean()),
            "entropy_bits": float(-(probability*np.log2(probability)).sum()),
            "positive_finite_depth_fraction": record["positive_finite_depth_fraction"],
        }
        metrics["basic_quality_pass"] = (metrics["luminance_std"] >= 5 and
            metrics["laplacian_variance"] >= 5 and metrics["dark_fraction_le_5"] < .98 and
            metrics["bright_fraction_ge_250"] < .98 and metrics["positive_finite_depth_fraction"] > .2)
        result[record["vehicle"]] = metrics
    return result


def collect_stage(client, root, name, settings, registry, resolved, airborne=False):
    folder = root / name
    folder.mkdir()
    sensor = sample_sensors(client, folder)
    camera = audit_cameras(client, settings, folder, pairs=3,
                           stationary_tolerance_m=.10 if airborne else .01)
    visibility = target_visibility(client, registry, resolved, camera["records"], folder)
    quality = image_quality(folder, camera["records"])
    report = {"sensor": sensor, "camera": camera, "visibility": visibility,
              "image_quality": quality,
              "poses": {vehicle: pose_snapshot(client, vehicle) for vehicle in VEHICLES}}
    (folder / "stage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare(ground, air):
    result = {"vehicles": {}}
    failures = []
    for vehicle in VEHICLES:
        ground_down = ground["sensor"]["summary"][vehicle]["distance_down"]
        air_down = air["sensor"]["summary"][vehicle]["distance_down"]
        quality = air["image_quality"][vehicle]
        returned = air["visibility"][vehicle]["targets_returned"]
        vessel = next(x for x in air["visibility"][vehicle]["targets"] if x["target_id"]=="vessel_001")
        item = {
            "ground_down_in_range_ratio": ground_down["in_range_samples"]/ground_down["samples"],
            "air_down_in_range_ratio": air_down["in_range_samples"]/air_down["samples"],
            "ground_targets_returned": ground["visibility"][vehicle]["targets_returned"],
            "air_targets_returned": returned,
            "air_vessel_component_completeness": vessel["returned_components"]/vessel["expected_components"],
            "air_image_quality": quality,
        }
        result["vehicles"][vehicle] = item
        if air_down["in_range_samples"] / air_down["samples"] < .8:
            failures.append(f"{vehicle}: airborne down-range usable ratio below 0.8")
        if not quality["basic_quality_pass"]:
            failures.append(f"{vehicle}: airborne image failed basic blank/exposure/sharpness checks")
        if returned < 2 or vessel["returned_components"] == 0:
            failures.append(f"{vehicle}: insufficient registered targets visible")
    result["failures"] = failures
    result["status"] = "PASS" if not failures else "FAIL"
    result["interpretation"] = ("Basic observation-point acceptance only; target boxes are simulator review aids, "
                                "not detector predictions or label-quality certification.")
    return result


def make_comparison(root):
    panels = []
    for stage in ("ground","air"):
        row=[]
        for vehicle in VEHICLES:
            image=cv2.imread(str(root/stage/f"{vehicle}_target_review.jpg"))
            cv2.putText(image,f"{stage.upper()} {vehicle}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
            row.append(image)
        panels.append(np.hstack(row))
    cv2.imwrite(str(root/"ground_vs_air_targets.jpg"),np.vstack(panels))


def main():
    settings_path = ROOT / "configs/maritime_harbor_settings.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    root = ROOT / "logs" / datetime.now().strftime("air_observation_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    (root/"settings.json").write_bytes(settings_bytes)
    report = {"status":"RUNNING", "settings_sha256":hashlib.sha256(settings_bytes).hexdigest(),
              "observation_local_ned_m":OBSERVATION_LOCAL_NED_M.tolist(),
              "flight_scope":"vertical takeoff, stationary observation, landing",
              "target_boxes_scope":"simulator visibility review only"}
    client=airsim.MultirotorClient(port=settings["ApiServerPort"],timeout_value=10)
    api_enabled=[]; airborne=False
    registry=resolved=None
    try:
        if not client.ping() or not set(VEHICLES).issubset(client.listVehicles()):
            raise RuntimeError("Maritime AirSim instance with both vehicles is required")
        if json.loads(client.getSettingsString()).get("Vehicles") != settings["Vehicles"]:
            raise RuntimeError("Live settings differ from maritime_harbor_settings.json")
        before={vehicle:pose_snapshot(client,vehicle) for vehicle in VEHICLES}
        if any(x["landed_state"] != airsim.LandedState.Landed for x in before.values()):
            raise RuntimeError("Both vehicles must start landed")
        report["before"]=before
        collisions={vehicle:before[vehicle]["collision_timestamp_ns"] for vehicle in VEHICLES}
        registry,resolved=setup_detection(client)
        report["ground"]=collect_stage(client,root,"ground",settings,registry,resolved)
        for vehicle in VEHICLES:
            client.enableApiControl(True,vehicle_name=vehicle); api_enabled.append(vehicle)
            if not client.armDisarm(True,vehicle_name=vehicle):
                raise RuntimeError(f"Cannot arm {vehicle}")
        takeoffs=[client.takeoffAsync(timeout_sec=20,vehicle_name=v) for v in VEHICLES]
        for future in takeoffs: future.join()
        airborne=True
        moves=[client.moveToPositionAsync(*OBSERVATION_LOCAL_NED_M,1.5,timeout_sec=25,
               yaw_mode=airsim.YawMode(False,0),vehicle_name=v) for v in VEHICLES]
        for future in moves: future.join()
        hovers=[client.hoverAsync(vehicle_name=v) for v in VEHICLES]
        for future in hovers: future.join()
        report["position_samples"]=wait_positions(client,{v:OBSERVATION_LOCAL_NED_M for v in VEHICLES},
                                                   collisions,root/"position_samples.jsonl")
        report["air"]=collect_stage(client,root,"air",settings,registry,resolved,airborne=True)
        report["comparison"]=compare(report["ground"],report["air"])
        make_comparison(root)
        report["status"]=report["comparison"]["status"]
    except BaseException as exc:
        report.update(status="FAIL",error=f"{type(exc).__name__}: {exc}")
    finally:
        recovery_errors=[]
        if airborne:
            try:
                ground_z={v:report["before"][v]["local_ned_m"][2] for v in VEHICLES}
                report["landing_evidence"]=land_and_verify(client,VEHICLES,ground_z)
            except Exception as exc: recovery_errors.append(f"landing: {exc}")
        if not recovery_errors:
            for vehicle in reversed(api_enabled):
                try:
                    if not client.armDisarm(False,vehicle_name=vehicle):
                        raise RuntimeError("disarm returned false")
                    client.enableApiControl(False,vehicle_name=vehicle)
                except Exception as exc: recovery_errors.append(f"{vehicle} release: {exc}")
        elif api_enabled:
            try: client.simPause(True); report["paused_for_manual_recovery"]=True
            except Exception as exc: recovery_errors.append(f"pause after recovery failure: {exc}")
        try:
            report["after"]={vehicle:pose_snapshot(client,vehicle) for vehicle in VEHICLES}
        except Exception as exc: recovery_errors.append(f"final snapshot: {exc}")
        if registry is not None:
            for vehicle in VEHICLES:
                try: client.simClearDetectionMeshNames(CAMERA,airsim.ImageType.Scene,vehicle_name=vehicle)
                except Exception as exc: recovery_errors.append(f"{vehicle} detection cleanup: {exc}")
        report["recovery_errors"]=recovery_errors
        if recovery_errors:
            report["status"]="FAIL"
        (root/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        (ROOT/"logs"/"air_observation_latest.json").write_text(json.dumps({"output":str(root)},indent=2),encoding="utf-8")
        print(json.dumps({"status":report["status"],"output":str(root),"error":report.get("error"),
                          "recovery_errors":recovery_errors},indent=2))
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
