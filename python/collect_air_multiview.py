"""Collect a bounded five-view airborne sample set for manual label review."""
import argparse
import csv
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np

from sensor_camera_audit import audit_cameras
from validate_air_observation import (
    CAMERA,
    VEHICLES,
    image_quality,
    land_and_verify,
    pose_snapshot,
    sample_sensors,
    setup_detection,
    target_visibility,
    wait_positions,
)

ROOT = Path(__file__).resolve().parents[1]

# Local NED coordinates. Both vehicles receive the same offset and retain their
# configured four-metre separation. The negative X offset moves the cameras away
# from the near balloon; the lower altitude keeps that balloon above the bottom
# image edge while preserving a clear view over the observation platform.
FRAMING_PROFILE = "rear_low_v2"
MIN_TARGET_EDGE_MARGIN_PX = 10.0
WAYPOINTS = (
    ("rear_center", np.array([-2.0, 0.0, -4.0])),
    ("rear_left", np.array([-2.0, -1.5, -4.0])),
    ("rear_right", np.array([-2.0, 1.5, -4.0])),
    ("far_center", np.array([-3.5, 0.0, -4.5])),
    ("near_center", np.array([-1.5, 0.0, -3.5])),
)


def move_and_settle(client, target, collisions, trace_path, speed=1.25):
    moves = [
        client.moveToPositionAsync(
            *target,
            speed,
            timeout_sec=30,
            yaw_mode=airsim.YawMode(False, 0),
            vehicle_name=vehicle,
        )
        for vehicle in VEHICLES
    ]
    for future in moves:
        future.join()
    hovers = [client.hoverAsync(vehicle_name=vehicle) for vehicle in VEHICLES]
    for future in hovers:
        future.join()
    return wait_positions(
        client,
        {vehicle: target for vehicle in VEHICLES},
        collisions,
        trace_path,
        timeout=40,
        hold=2.0,
        position_tolerance=0.70,
        speed_tolerance=0.05,
    )


def collect_waypoint(client, root, waypoint_id, target, settings, registry, resolved):
    folder = root / waypoint_id
    folder.mkdir(exist_ok=True)
    sensor = sample_sensors(client, folder, duration=1.25, interval=0.25)
    camera = audit_cameras(client, settings, folder, pairs=1, stationary_tolerance_m=0.10)
    visibility = target_visibility(client, registry, resolved, camera["records"], folder)
    quality = image_quality(folder, camera["records"])
    report = {
        "waypoint_id": waypoint_id,
        "commanded_local_ned_m": target.tolist(),
        "poses": {vehicle: pose_snapshot(client, vehicle) for vehicle in VEHICLES},
        "sensor": sensor,
        "camera": camera,
        "visibility": visibility,
        "image_quality": quality,
    }
    failures = []
    summary = {}
    capture = settings["Vehicles"][VEHICLES[0]]["Cameras"][CAMERA]["CaptureSettings"][0]
    image_width = float(capture["Width"])
    image_height = float(capture["Height"])
    for vehicle in VEHICLES:
        down = sensor["summary"][vehicle]["distance_down"]
        ratio = down["in_range_samples"] / down["samples"]
        vessel = next(
            item for item in visibility[vehicle]["targets"] if item["target_id"] == "vessel_001"
        )
        summary[vehicle] = {
            "down_range_usable_ratio": ratio,
            "targets_returned": visibility[vehicle]["targets_returned"],
            "vessel_components_returned": vessel["returned_components"],
            "vessel_components_expected": vessel["expected_components"],
            "mean_luminance": quality[vehicle]["mean_luminance"],
            "luminance_std": quality[vehicle]["luminance_std"],
            "laplacian_variance": quality[vehicle]["laplacian_variance"],
            "positive_finite_depth_fraction": quality[vehicle]["positive_finite_depth_fraction"],
            "target_edge_margins_px": {},
        }
        if ratio < 0.8:
            failures.append(f"{vehicle}: downward range usable ratio below 0.8")
        if not quality[vehicle]["basic_quality_pass"]:
            failures.append(f"{vehicle}: basic image-quality check failed")
        if visibility[vehicle]["targets_returned"] < 2 or vessel["returned_components"] == 0:
            failures.append(f"{vehicle}: insufficient registered targets returned")
        if visibility[vehicle]["targets_returned"] != len(registry["targets"]):
            failures.append(
                f"{vehicle}: expected all {len(registry['targets'])} registered targets in frame"
            )
        for item in visibility[vehicle]["targets"]:
            box = item.get("bbox_xyxy_pixels")
            if not box:
                continue
            x1, y1, x2, y2 = box
            edge_margin = min(x1, y1, image_width - x2, image_height - y2)
            summary[vehicle]["target_edge_margins_px"][item["target_id"]] = edge_margin
            if edge_margin < MIN_TARGET_EDGE_MARGIN_PX:
                failures.append(
                    f"{vehicle}: {item['target_id']} edge margin {edge_margin:.1f}px "
                    f"below {MIN_TARGET_EDGE_MARGIN_PX:.1f}px"
                )
    if camera["status"] != "PASS":
        failures.extend(camera["errors"])
    report["summary"] = summary
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    (folder / "waypoint_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def write_manual_review_csv(root, waypoint_reports):
    path = root / "manual_box_review.csv"
    fields = [
        "waypoint_id",
        "vehicle",
        "rgb_image",
        "review_image",
        "target_id",
        "returned_components",
        "expected_components",
        "suggested_x1",
        "suggested_y1",
        "suggested_x2",
        "suggested_y2",
        "bbox_area_fraction",
        "review_decision",
        "review_notes",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in waypoint_reports:
            waypoint_id = report["waypoint_id"]
            for vehicle in VEHICLES:
                visibility = report["visibility"][vehicle]
                record = next(
                    row for row in report["camera"]["records"] if row["vehicle"] == vehicle
                )
                for target in visibility["targets"]:
                    box = target.get("bbox_xyxy_pixels", ["", "", "", ""])
                    writer.writerow(
                        {
                            "waypoint_id": waypoint_id,
                            "vehicle": vehicle,
                            "rgb_image": f"{waypoint_id}/{record['rgb']}",
                            "review_image": f"{waypoint_id}/{visibility['review_image']}",
                            "target_id": target["target_id"],
                            "returned_components": target["returned_components"],
                            "expected_components": target["expected_components"],
                            "suggested_x1": box[0],
                            "suggested_y1": box[1],
                            "suggested_x2": box[2],
                            "suggested_y2": box[3],
                            "bbox_area_fraction": target.get("bbox_area_fraction", ""),
                            "review_decision": "",
                            "review_notes": "",
                        }
                    )
    return path


def make_contact_sheet(root, waypoint_reports):
    rows = []
    for report in waypoint_reports:
        panels = []
        for vehicle in VEHICLES:
            image_path = root / report["waypoint_id"] / report["visibility"][vehicle]["review_image"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError(f"Cannot read review image: {image_path}")
            image = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
            label = f"{report['waypoint_id']} | {vehicle}"
            cv2.rectangle(image, (0, 0), (390, 38), (0, 0, 0), -1)
            cv2.putText(image, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            panels.append(image)
        rows.append(np.hstack(panels))
    output = root / "multiview_contact_sheet.jpg"
    cv2.imwrite(str(output), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=("baseline", "low_sun", "raised_waves", "occlusion"),
        help="apply a named scene profile and restore baseline after landing",
    )
    args = parser.parse_args()
    settings_path = ROOT / "configs" / "maritime_harbor_settings.json"
    settings_bytes = settings_path.read_bytes()
    settings = json.loads(settings_bytes)
    prefix = "air_multiview" if args.scenario is None else f"air_multiview_{args.scenario}"
    root = ROOT / "logs" / datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True)
    (root / "settings.json").write_bytes(settings_bytes)
    report = {
        "status": "RUNNING",
        "scenario": args.scenario or "scene_as_found",
        "framing_profile": FRAMING_PROFILE,
        "minimum_target_edge_margin_px": MIN_TARGET_EDGE_MARGIN_PX,
        "settings_sha256": hashlib.sha256(settings_bytes).hexdigest(),
        "waypoints": [
            {"waypoint_id": waypoint_id, "local_ned_m": target.tolist()}
            for waypoint_id, target in WAYPOINTS
        ],
        "scope": "small airborne multi-view sample set for manual review",
        "target_boxes_scope": "simulator review aids only; not approved training labels",
        "manual_review_required": True,
    }
    client = airsim.MultirotorClient(port=settings["ApiServerPort"], timeout_value=10)
    api_enabled = []
    airborne = False
    registry = resolved = None
    collisions = {}
    scenario_attempted = False
    try:
        if not client.ping() or not set(VEHICLES).issubset(client.listVehicles()):
            raise RuntimeError("Maritime AirSim instance with both vehicles is required")
        if json.loads(client.getSettingsString()).get("Vehicles") != settings["Vehicles"]:
            raise RuntimeError("Live settings differ from maritime_harbor_settings.json")
        before = {vehicle: pose_snapshot(client, vehicle) for vehicle in VEHICLES}
        if any(item["landed_state"] != airsim.LandedState.Landed for item in before.values()):
            raise RuntimeError("Both vehicles must start landed")
        report["before"] = before
        if args.scenario:
            import maritime_experiments

            scenario_attempted = True
            report["experiment"] = maritime_experiments.apply(client, args.scenario)
            for filename in ("maritime_experiments.json", "maritime_targets.json"):
                (root / filename).write_bytes((ROOT / "configs" / filename).read_bytes())
            (root / "experiment.json").write_text(
                json.dumps(report["experiment"], indent=2), encoding="utf-8"
            )
        collisions = {
            vehicle: before[vehicle]["collision_timestamp_ns"] for vehicle in VEHICLES
        }
        registry, resolved = setup_detection(client)
        for vehicle in VEHICLES:
            client.enableApiControl(True, vehicle_name=vehicle)
            api_enabled.append(vehicle)
            if not client.armDisarm(True, vehicle_name=vehicle):
                raise RuntimeError(f"Cannot arm {vehicle}")
        takeoffs = [
            client.takeoffAsync(timeout_sec=20, vehicle_name=vehicle) for vehicle in VEHICLES
        ]
        for future in takeoffs:
            future.join()
        airborne = True
        waypoint_reports = []
        for waypoint_id, target in WAYPOINTS:
            trace = root / waypoint_id / "position_samples.jsonl"
            trace.parent.mkdir()
            position_samples = move_and_settle(client, target, collisions, trace)
            waypoint_report = collect_waypoint(
                client, root, waypoint_id, target, settings, registry, resolved
            )
            waypoint_report["position_samples"] = position_samples
            (root / waypoint_id / "waypoint_report.json").write_text(
                json.dumps(waypoint_report, indent=2), encoding="utf-8"
            )
            waypoint_reports.append(waypoint_report)
        report["waypoint_reports"] = waypoint_reports
        report["manual_review_csv"] = str(write_manual_review_csv(root, waypoint_reports).name)
        report["contact_sheet"] = str(make_contact_sheet(root, waypoint_reports).name)
        report["failures"] = [
            f"{item['waypoint_id']}: {failure}"
            for item in waypoint_reports
            for failure in item["failures"]
        ]
        report["status"] = "PASS" if not report["failures"] else "FAIL"
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        recovery_errors = []
        if airborne:
            try:
                return_target = np.array([0.0, 0.0, -3.0])
                report["return_position_samples"] = move_and_settle(
                    client, return_target, collisions, root / "return_position_samples.jsonl"
                )
                ground_z = {
                    vehicle: report["before"][vehicle]["local_ned_m"][2]
                    for vehicle in VEHICLES
                }
                report["landing_evidence"] = land_and_verify(client, VEHICLES, ground_z)
            except Exception as exc:
                recovery_errors.append(f"return/landing: {exc}")
        if not recovery_errors:
            for vehicle in reversed(api_enabled):
                try:
                    if not client.armDisarm(False, vehicle_name=vehicle):
                        raise RuntimeError("disarm returned false")
                    client.enableApiControl(False, vehicle_name=vehicle)
                except Exception as exc:
                    recovery_errors.append(f"{vehicle} release: {exc}")
        elif api_enabled:
            try:
                client.simPause(True)
                report["paused_for_manual_recovery"] = True
            except Exception as exc:
                recovery_errors.append(f"pause after recovery failure: {exc}")
        try:
            report["after"] = {
                vehicle: pose_snapshot(client, vehicle) for vehicle in VEHICLES
            }
        except Exception as exc:
            recovery_errors.append(f"final snapshot: {exc}")
        if registry is not None:
            for vehicle in VEHICLES:
                try:
                    client.simClearDetectionMeshNames(
                        CAMERA, airsim.ImageType.Scene, vehicle_name=vehicle
                    )
                except Exception as exc:
                    recovery_errors.append(f"{vehicle} detection cleanup: {exc}")
        if scenario_attempted:
            try:
                import maritime_experiments

                report["baseline_restore"] = maritime_experiments.apply(client, "baseline")
                report["scene_restored_to_baseline"] = True
            except Exception as exc:
                report["scene_restored_to_baseline"] = False
                recovery_errors.append(f"scene baseline restore: {exc}")
        report["recovery_errors"] = recovery_errors
        if recovery_errors:
            report["status"] = "FAIL"
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (ROOT / "logs" / "air_multiview_latest.json").write_text(
            json.dumps({"output": str(root)}, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(root),
                    "error": report.get("error"),
                    "failures": report.get("failures", []),
                    "recovery_errors": recovery_errors,
                },
                indent=2,
            )
        )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
