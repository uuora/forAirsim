"""Freeze one new T1 low-speed flight pilot without simulator or API calls.

Only the explicit project dependency list, twelve historical source locks,
camera settings, and trained detector artifacts are copied. No secret file is
read, inventoried, or copied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHTS_SHA256 = "92b0dc0f4e17ff14e9c1e110b9cad9ef249274136ee9a28d99b7885dfb396844"
EXPECTED_TRAINING_RECEIPT_SHA256 = "f98db6fb0f916bd00ea74ee36596fb222d662f3a913d36610d9e96aa7c4b22a5"
SOURCE_FILES = (
    "python/t1_identity_flight.py", "python/t1_flight_observer.py",
    "python/t1_pose_trajectory_planner.py", "python/t1_identity_motion_policy.py",
    "python/t1_pose_projection_batch.py",
    "python/t1_candidate_surface_v2.py", "python/t1_candidate_surface.py",
    "python/t1_continuous_semantic_observer.py", "python/t1_temporal_identity.py",
    "python/t1_semantic_perception.py", "python/t1_pose_geometry.py",
    "python/specialist_balloon_detector.py", "python/train_t1_balloon_yolo.py",
    "python/vlx_api_client.py", "python/t1_ground_perception_probe.py",
    "python/t1_balloon_data_probe.py", "python/t1_identity_sequence_probe_v1p3.py",
    "python/t1_identity_schedule.py", "python/camera_geometry.py",
    "python/setup_scene.py", "python/trajectory_execution_guard.py",
    "python/braking_startup_readiness.py", "python/prepare_t1_identity_flight.py",
    "scripts/run-t1-identity-flight-v1.ps1", "scripts/assert-trajectory-source-freeze.ps1",
)
LEGACY_NAMES = (
    "single_drone_experiment.py", "visual_servo.py", "target_provider.py",
    "balloon_detector.py", "trajectory_single_experiment.py",
    "perception_trajectory_planner.py", "timestamped_target_provider.py",
    "matched_trajectory_start.py", "trajectory_execution_guard.py",
    "trajectory_timed_reference.py", "trajectory_timed_experiment.py",
    "trajectory_rgb_quality_reference.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")


def absolute_artifact(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def copy_verified(source, target, expected):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if digest(target) != expected or digest(source) != expected:
        raise ValueError("Declared artifact changed during snapshot")


def phase_grid(phases, period_s):
    """Declared slots retain the shorter interval at each phase boundary."""
    slots, start = [], 0.0
    for phase in phases:
        end = start + phase["duration_s"]
        cursor = start
        while cursor < end - 1e-9:
            next_cursor = min(cursor + period_s, end)
            slots.append({"slot_index": len(slots), "phase_id": phase["phase_id"],
                "scheduled_start_s": round(cursor, 9),
                "scheduled_end_s": round(next_cursor, 9)})
            cursor = next_cursor
        start = end
    return slots


def prepare(output, mode, session_id, device="cuda"):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Prepared destination must be a new directory")
    if mode not in ("zero", "short") or device not in ("cpu", "cuda"):
        raise ValueError("Unknown pilot mode or inference device")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", session_id):
        raise ValueError("Session ID must be a bounded filesystem-safe identifier")

    sources = [ROOT / name for name in SOURCE_FILES]
    if any(not path.is_file() for path in sources):
        missing = [str(path.relative_to(ROOT)) for path in sources if not path.is_file()]
        raise ValueError("Declared source dependencies are not ready: " + ", ".join(missing))
    runtime_config = ROOT / "configs/t1_balloon_detector_runtime_cuda_v1.json"
    settings_source = ROOT / "configs/t1_continuous_identity_v1p3_20261004/settings.json"
    historical_source = ROOT / "configs/trajectory_timed_pilot_v6p2/center.json"
    detector = read_json(runtime_config)
    weights, receipt = absolute_artifact(detector["weights"]), absolute_artifact(detector["training_receipt"])
    if (detector.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256
        or not weights.is_file() or digest(weights) != EXPECTED_WEIGHTS_SHA256):
        raise ValueError("Frozen trained checkpoint binding changed")
    if (detector.get("training_receipt_sha256") != EXPECTED_TRAINING_RECEIPT_SHA256
        or not receipt.is_file() or digest(receipt) != EXPECTED_TRAINING_RECEIPT_SHA256):
        raise ValueError("Frozen training receipt binding changed")
    training = read_json(receipt)
    if (training.get("status") != "PASS"
        or training.get("best_weights_sha256") != EXPECTED_WEIGHTS_SHA256
        or absolute_artifact(training["best_weights"]) != weights
        or training.get("best_class_names") != {"0": "balloon"}
        or detector.get("class_names") != {"0": "balloon"}):
        raise ValueError("Training receipt does not support the configured balloon checkpoint")
    if (detector.get("confidence") != .25 or detector.get("nms_iou") != .5
        or detector.get("imgsz") != 640):
        raise ValueError("Detector thresholds changed from the accepted development protocol")
    settings = read_json(settings_source)
    if settings.get("ClockSpeed") != 1.0 or not {"DroneA", "DroneB"}.issubset(settings.get("Vehicles", {})):
        raise ValueError("Expected two-vehicle ClockSpeed=1 camera settings")
    historical = read_json(historical_source)["flight_source_sha256"]
    legacy_hashes = {name: historical[name] for name in LEGACY_NAMES}
    for name, expected in legacy_hashes.items():
        if digest(ROOT / "python" / name) != expected:
            raise ValueError("Historical V6.2 source lock mismatch: " + name)

    inventory = [{"relative_path": name, "path": str((ROOT / name).resolve()),
        "sha256": digest(ROOT / name)} for name in SOURCE_FILES]
    output.mkdir(parents=True)
    bundle = output / "source_bundle"
    for item in inventory:
        target = bundle / item["relative_path"]
        copy_verified(item["path"], target, item["sha256"])
        item["snapshot_path"] = str(target)
    settings_path = output / "settings.json"
    copy_verified(settings_source, settings_path, digest(settings_source))
    weights_path = bundle / "model/specialist_best.pt"
    receipt_path = output / "training_receipt.json"
    copy_verified(weights, weights_path, EXPECTED_WEIGHTS_SHA256)
    copy_verified(receipt, receipt_path, EXPECTED_TRAINING_RECEIPT_SHA256)
    detector.update(device="0" if device == "cuda" else "cpu", weights=str(weights_path),
        training_receipt=str(receipt_path), status="FROZEN_T1_IDENTITY_FLIGHT_PILOT_RUNTIME")
    config_path = output / "detector_config.json"
    save(config_path, detector)
    legacy_path = output / "legacy_v6p2_source_freeze.json"
    save(legacy_path, {"kind": "TWELVE_V6P2_MODULES_UNCHANGED", "reference": str(historical_source),
        "reference_sha256": digest(historical_source), "source_sha256": legacy_hashes})
    legacy_inventory = []
    for name, expected in legacy_hashes.items():
        path = ROOT / "python" / name
        snapshot = bundle / "legacy_v6p2" / name
        copy_verified(path, snapshot, expected)
        legacy_inventory.append({"path": str(path), "snapshot_path": str(snapshot), "sha256": expected})
    phases = ([{"phase_id": "visible", "duration_s": 8.0}] if mode == "zero" else [
        {"phase_id": "visible", "duration_s": 7.0},
        {"phase_id": "hidden", "duration_s": 2.0},
        {"phase_id": "return", "duration_s": 7.0}])
    packages = {}
    for package in ("airsim", "numpy", "opencv-python", "Pillow", "torch", "ultralytics", "requests"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "NOT_DISTRIBUTION_METADATA_AVAILABLE"
    plan = {"schema_version": 1, "kind": "T1_IDENTITY_LOW_SPEED_FLIGHT_PILOT",
        "status": "FROZEN_BEFORE_FLIGHT", "created_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id, "description": "带白色横条的红色气球", "target_generation": 1,
        "mode": "ZERO_GATE" if mode == "zero" else "SHORT_XY", "vehicle": "DroneA", "bystander": "DroneB",
        "recovery_observability": "VISUAL_ATTRIBUTE", "formal_efficacy_results": False,
        "planned_UE_sessions": 1, "planned_flights": 1, "period_s": .4,
        "world_z_ned_m": -3.0, "command_speed_cap_m_s": .03,
        "command_lease_s": .12, "self_monitor_period_s": .05,
        "phases": phases, "planned_duration_s": sum(p["duration_s"] for p in phases),
        "scene": {"target_world_ned_m": [5.0, -2.5, -3.05],
            "distractor_world_ned_m": [5.0, -3.8, -3.05],
            "occluder_world_ned_m": [4.0, -2.625, -3.05],
            "inactive_world_ned_m": [30.0, 100.0, -3.0],
            "target_scale_xyz": [1.2, 1.2, 1.5], "occluder_scale_xyz": [.2, 1.0, 2.0],
            "target_texture": "red_64x64_white_horizontal_rows_24_to_39",
            "distractor_texture": "plain_red_64x64"},
        "takeoff": {"world_z_ned_m": -3.0, "move_to_z_speed_m_s": .3,
            "xy_anchor": "ACTUAL_INITIAL_SELF_WORLD_POSITION", "scope": "Excluded from low-speed phase statistics"},
        "planned_slots": phase_grid(phases, .4), "timing_grid": "PHASE_BOUNDARY_CLIPPED_0P4",
        "detector_config_path": str(config_path), "detector_config_sha256": digest(config_path),
        "weights_path": str(weights_path), "weights_sha256": EXPECTED_WEIGHTS_SHA256,
        "training_receipt_path": str(receipt_path), "training_receipt_sha256": EXPECTED_TRAINING_RECEIPT_SHA256,
        "detector_bindings": {"weights_sha256": EXPECTED_WEIGHTS_SHA256,
            "training_receipt_sha256": EXPECTED_TRAINING_RECEIPT_SHA256, "class_names": {"0": "balloon"},
            "confidence": .25, "nms_iou": .5, "imgsz": 640, "device": detector["device"]},
        "settings_path": str(settings_path), "settings_sha256": digest(settings_path),
        "source_sha256": {item["relative_path"]: item["sha256"] for item in inventory},
        "source_inventory": inventory, "legacy_source_inventory": legacy_inventory,
        "legacy_freeze_path": str(legacy_path), "legacy_freeze_sha256": digest(legacy_path),
        "legacy_v6p2_source_sha256": legacy_hashes,
        "braking_model": {"horizon_s": 1.0, "ramp_acceleration_m_s2": .15, "position_reserve_m": .02,
            "scope": "Previously sampled low-speed candidate; no continuous physical safety certificate"},
        "observer_configuration": {"review_period_s": 3.0, "current_frame_max_age_s": .5,
            "semantic_max_age_s": 2.0, "response_timeout_s": 2.0, "min_request_interval_s": 2.0,
            "track_max_unreviewed_s": 8.0, "track_max_gap_s": .5, "confirmation_frames": 2},
        "artifact_origins": {"settings_path": str(settings_source), "settings_sha256": digest(settings_source),
            "detector_runtime_config_path": str(runtime_config), "detector_runtime_config_sha256": digest(runtime_config),
            "weights_path": str(weights), "training_receipt_path": str(receipt)},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages},
        "truth_access": "Scene construction and separate evaluation only; RGB identity and depth surface provide all planner target inputs",
        "flight_scope": "One development launch; zero or .03m/s XY phase requests; staging/return are separately recorded",
        "source_lock_scope": "Explicit project source inventory and historical twelve-module check; not a full dependency/build lock",
        "api_secret_snapshot": "NONE; .env, credentials and endpoints excluded",
        "missed_slot_policy": "Retain every predeclared slot, including NOT_CAPTURED, stale and interrupted slots",
        "ai_assistance": "AI assisted protocol, implementation and evidence preparation"}
    save(output / "plan.json", plan)
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("zero", "short"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = prepare(args.output, args.mode, args.session_id, args.device)
    print(json.dumps({"status": result["status"], "mode": result["mode"],
        "plan": str(args.output.resolve() / "plan.json"), "planned_slots": len(result["planned_slots"])}, ensure_ascii=False))
