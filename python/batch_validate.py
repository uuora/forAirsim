"""Repeat fixed-scene missions and produce one auditable batch report."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime

from setup_scene import ROOT


def aggregate(rows):
    completed = len(rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    values = lambda key: [float(row[key]) for row in rows if row.get(key) is not None]
    separations = values("min_separation_m")
    drifts = values("max_waiting_drift_m")
    dispatch = values("miss_to_dispatch_s")
    visual_modes = [row.get("visual_final_mode") for row in rows
                    if row.get("visual_final_mode")]
    first_attempt_modes = [row.get("a_visual_final_mode") for row in rows
                           if row.get("a_visual_final_mode")]
    return {
        "completed": completed,
        "passed": passed,
        "failed": completed - passed,
        "pass_rate": passed / completed if completed else None,
        "minimum_separation_m": min(separations) if separations else None,
        "maximum_waiting_drift_m": max(drifts) if drifts else None,
        "mean_miss_to_dispatch_s": sum(dispatch) / len(dispatch) if dispatch else None,
        "visual_collision_runs": (sum(mode == "visual_collision" for mode in visual_modes)
                                  if visual_modes else None),
        "coordinate_contact_runs": (
            sum(mode == "visual_gate_coordinate_contact" for mode in visual_modes)
            if visual_modes else None),
        "a_visual_collision_runs": (sum(mode == "visual_collision" for mode in first_attempt_modes)
                                    if first_attempt_modes else None),
        "a_visual_handoff_runs": (sum(mode == "handoff_after_visual_failure" for mode in first_attempt_modes)
                                  if first_attempt_modes else None),
    }


def newest_report(previous):
    current = set((ROOT / "logs").glob("mission_*/report.json"))
    created = sorted(current - previous, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new mission report, got {len(created)}")
    return created[0]


def event_delay_s(mission, start_event="miss", end_event="fallback_ready"):
    """Return an event-to-event monotonic delay from a mission report."""
    timestamps = {}
    for event in mission.get("events", []):
        if event.get("event") in (start_event, end_event):
            timestamps.setdefault(event["event"], event.get("monotonic_s"))
    if timestamps.get(start_event) is None or timestamps.get(end_event) is None:
        return None
    return float(timestamps[end_event]) - float(timestamps[start_event])


def mission_row(mission, path, repeat, scenario):
    evaluation = mission.get("evaluation", {})
    attempts = mission.get("attempt_assessments", {})
    a_attempt = attempts.get("DroneA", {})
    return {
        "repeat": repeat,
        "scenario": scenario,
        "evaluation_mode": evaluation.get("mode", "UNKNOWN"),
        "policy_id": evaluation.get("policy_id", "legacy_unmarked"),
        "run": path.parent.name,
        "status": mission["status"],
        "hit_vehicle": mission.get("hit_evidence", {}).get("vehicle", ""),
        "landing_verified": mission.get("landing_verified", False),
        "min_separation_m": mission.get("min_separation_m"),
        "max_waiting_drift_m": mission.get("max_waiting_drift_m"),
        "miss_to_dispatch_s": mission.get("miss_to_dispatch_s", event_delay_s(mission)),
        "a_visual_frames": len(mission.get("a_visual_records", [])),
        "b_visual_frames": len(mission.get("b_visual_records", [])),
        "a_visual_final_mode": mission.get("a_visual_final_mode", ""),
        "b_visual_final_frames": len(mission.get("b_visual_final_records", [])),
        "visual_final_mode": mission.get("b_visual_final_mode", ""),
        "visual_fallback_verified": mission.get("visual_fallback_verified", False),
        "a_attempt_state": a_attempt.get("state", "UNKNOWN"),
        "failure_reason": mission.get("failure_reason", mission.get("error", "")),
        "error": mission.get("error", ""),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", choices=("a_hit", "fallback", "visual_fallback"),
                        default=("a_hit", "fallback"))
    parser.add_argument("--miss-offset-y", type=float,
                        help="Forward a tested miss offset to fallback mission runs")
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--attempt-mode", choices=("fault_injection", "observation"),
                        default="fault_injection",
                        help="For visual_fallback: use the injected or observation-assessed A attempt")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        raise ValueError("repeats must be between 1 and 100")
    if "visual_fallback" in args.scenarios and args.target_offset is not None:
        raise ValueError("visual_fallback currently supports only the calibrated center target")
    if "visual_fallback" in args.scenarios and args.miss_offset_y is not None:
        raise ValueError("visual_fallback uses its calibrated deterministic miss branch")

    batch_dir = ROOT / "logs" / datetime.now().strftime("batch_%Y%m%d_%H%M%S_%f")
    batch_dir.mkdir(parents=True)
    rows = []
    report = {
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "requested_repeats": args.repeats,
        "scenarios": list(args.scenarios),
        "miss_offset_y_m": args.miss_offset_y,
        "target_offset_ned_m": args.target_offset,
        "attempt_mode": args.attempt_mode,
        "configuration_sha256": hashlib.sha256((ROOT / "configs" / "mission.json").read_bytes()).hexdigest(),
        "runs": rows,
    }
    try:
        for repeat in range(1, args.repeats + 1):
            for scenario in args.scenarios:
                print(f"BATCH repeat={repeat}/{args.repeats} scenario={scenario}", flush=True)
                setup_command = [sys.executable, str(ROOT / "python" / "setup_scene.py")]
                if args.target_offset is not None:
                    setup_command.extend(["--target-offset", *map(str, args.target_offset)])
                subprocess.run(setup_command, cwd=ROOT, check=True)
                previous = set((ROOT / "logs").glob("mission_*/report.json"))
                if scenario == "visual_fallback":
                    command = [sys.executable, str(ROOT / "python" / "run_visual_fallback.py"),
                               "--attempt-mode", args.attempt_mode]
                else:
                    command = [sys.executable, str(ROOT / "python" / "run_balloon_mission.py"),
                               "--scenario", scenario]
                    if scenario == "fallback" and args.miss_offset_y is not None:
                        command.extend(["--miss-offset-y", str(args.miss_offset_y)])
                    if args.target_offset is not None:
                        command.extend(["--target-offset", *map(str, args.target_offset)])
                result = subprocess.run(command, cwd=ROOT)
                path = newest_report(previous)
                mission = json.loads(path.read_text(encoding="utf-8"))
                row = mission_row(mission, path, repeat, scenario)
                rows.append(row)
                report["aggregate"] = aggregate(rows)
                (batch_dir / "batch_report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                if result.returncode or mission["status"] != "PASS":
                    raise RuntimeError(f"Batch stopped after failed run: {path.parent.name}")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["aggregate"] = aggregate(rows)
        (batch_dir / "batch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if rows:
            with (batch_dir / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(f"Batch report: {batch_dir}", flush=True)


if __name__ == "__main__":
    main()
