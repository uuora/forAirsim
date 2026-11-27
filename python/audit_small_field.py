"""Recompute small-field acceptance from the frozen plan and AirSim samples."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dual_target_control.small_field import (BODY_PROTOCOL_REVISION, POINT_PROTOCOL_REVISION, expected_dynamics,
                                              flight_metrics, reference_checks, scaled_vehicle_bounds, total_tilt_deg)
from fast_reference import decode


def read(path):
    return json.loads(path.read_bytes())


def audit_trial(directory):
    summary = read(directory / "summary.json")
    manifest = read(directory / "live_manifest.json")
    trace = read(directory / "trace.json")
    states, commands = trace["states"], trace["commands"]
    cleanup_file = directory / "cleanup.json"
    cleanup = read(cleanup_file) if cleanup_file.is_file() else {}
    safe = all(cleanup.get(key) is True for key in ("landed", "disarmed", "api_control_released"))
    rules = summary["config"]["small_field"]
    received = dict(run=str(directory), method=summary["method"], outcome=summary["outcome"], error=summary["error"],
                    policy=summary["route"]["policy"], safe_landing=safe, passed=False)
    if not states:
        return received

    curve = decode(manifest["plan"])
    reference = reference_checks(curve)
    fault_path = directory / "tracking_fault.json"
    fault = read(fault_path) if fault_path.is_file() else None
    metrics = flight_metrics(states, curve, fault)
    clock = np.asarray([state["sim_timestamp_ns"] for state in states], dtype=np.int64)
    time = (clock - clock[0]) * 1e-9
    reference_positions = curve.sample(np.minimum(time, curve.duration_s))
    initial = read(directory / "curve_initial_state.json")
    yaw = np.r_[initial["actual_yaw_rad"], np.deg2rad([command["yaw_command_deg"] for command in commands])]
    yaw_changes = np.abs(np.rad2deg((np.diff(yaw) + np.pi) % (2*np.pi) - np.pi))
    yaw_limits = rules["max_yaw_rate_deg_s"] * np.maximum(np.diff(time, prepend=0.), .001)
    preparation = read(directory / "preparation.json")
    stages = [event["stage"] for event in preparation["events"]]
    settling = [row for row in preparation["samples"] if row["stage"] == "SETTLING" and row["new_state"]]
    n = rules["preparation"]["samples"]
    terminal_preparation = settling[-n:]
    limits = [rules["preparation"][key] for key in ("position_m", "speed_mps", "yaw_rad")]
    first = states[0]
    actual_start = np.asarray(first["position_zup_m"])
    position_ned = (actual_start * [1, 1, -1]).tolist()
    starting_errors = [np.linalg.norm(actual_start - rules["start"]),
                       np.linalg.norm(first["velocity_zup_mps"]), abs(first["yaw_rad"])]
    stamps = [row["sim_timestamp_ns"] for row in terminal_preparation if "sim_timestamp_ns" in row]
    spacing = np.diff(stamps) * 1e-9
    stage_order = ["MOVING_TO_START_ARRIVED", "PLANNING_STARTED", "REFERENCE_READY", "SETTLING_ARRIVED", "REFERENCE_STARTED"]
    recorded = summary["metrics"] or {}
    snapshots = directory / "curve_source"
    settings = manifest["conditions"].get("loaded_settings", {})
    dynamics = summary["dynamics"]
    body_geometry = "vehicle_bounds_frd_m" in rules
    expected_protocol = BODY_PROTOCOL_REVISION if body_geometry else POINT_PROTOCOL_REVISION
    checks = dict(
        reference_limits=reference["passed"],
        shared_clock=bool(np.allclose(time, [state["reference_elapsed_s"] for state in states], rtol=0., atol=1e-8)
                          and np.allclose(reference_positions, [state["reference_position_zup_m"] for state in states], rtol=0., atol=1e-8)),
        preparation=bool(all(stage in stages for stage in stage_order)
                         and [stages.index(stage) for stage in stage_order] == sorted(stages.index(stage) for stage in stage_order)
                         and len(stamps) == n and 0 <= (clock[0]-stamps[-1])*1e-9 <= rules["max_sample_gap_s"]
                         and np.all((spacing > 0.) & (spacing <= rules["max_sample_gap_s"]))
                         and np.all(np.asarray([row["errors"] for row in terminal_preparation]) <= limits)
                         and np.all(np.asarray(starting_errors) <= limits)),
        recorded_tilt=bool(np.allclose([state["tilt_deg"] for state in states],
                                       [total_tilt_deg(state["orientation_ned_xyzw"]) for state in states], atol=1e-8, rtol=0.)),
        yaw_slew=bool(len(commands) == len(states) and np.all(yaw_changes <= yaw_limits + 1e-7)),
        telemetry_acceptance=metrics["control_error_gates_passed"],
        metrics_match=all(recorded.get(key) == value for key, value in metrics.items()),
        source_snapshots=bool(manifest["source_sha256"]) and all(
            (snapshots / Path(path).name).is_file() and
            hashlib.sha256((snapshots / Path(path).name).read_bytes()).hexdigest() == digest
            for path, digest in manifest["source_sha256"].items()),
        configuration=curve.spec["small_field"] == rules and manifest["protocol_revision"] == expected_protocol,
        physics=expected_dynamics(dynamics)
                and settings.get("Vehicles", {}).get(summary["config"]["airsim"]["vehicle"], {}).get("VehicleType") == "SimpleFlight",
        recording_disabled=settings.get("Recording", {}).get("Enabled") is False
                           and summary["config"]["airsim"]["aim_camera"] is False,
    )
    if body_geometry:
        checks["vehicle_bounds_match_visual_scale"] = (
            rules["vehicle_bounds_frd_m"] == scaled_vehicle_bounds(summary["config"]["ue_visualization"]))
    if 'launch' in rules:
        field_frame = read(directory / 'field_frame.json')
        ground = field_frame.get('ground_start')
        takeoff = read(directory / 'takeoff.json')[0]
        launch_xy = np.asarray(rules['launch'][:2])
        checks['ground_launch'] = bool(
            ground and ground['contact'] == 'DC_CollisionFloor'
            and field_frame['launch_zup_m'] == rules['launch']
            and field_frame['reference_start_zup_m'] == rules['start']
            and ground['sim_timestamp_ns'] < clock[0]
            and all(np.linalg.norm(np.asarray(p)[:2] - launch_xy) <= limits[0]
                    for p in (ground['position_zup_m'], takeoff['position']))
            and all(np.linalg.norm(v) <= limits[1]
                    for v in (ground['velocity_zup_mps'], takeoff['velocity'])))
        received['measured_takeoff_zup_m'] = takeoff['position']
    return dict(received, checks=checks, measured_start_ned_m=position_ned, reference=reference, metrics=metrics,
                condition_fingerprint=manifest["condition_fingerprint"],
                passed=all(checks.values()) and safe and summary["outcome"] == "COMPLETED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path, nargs="+")
    args = parser.parse_args()
    reports = [audit_trial(path.resolve()) for path in args.run_directory]
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    raise SystemExit(not all(row["passed"] for row in reports))
