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
    return {
        "completed": completed,
        "passed": passed,
        "failed": completed - passed,
        "pass_rate": passed / completed if completed else None,
        "minimum_separation_m": min(separations) if separations else None,
        "maximum_waiting_drift_m": max(drifts) if drifts else None,
        "mean_miss_to_dispatch_s": sum(dispatch) / len(dispatch) if dispatch else None,
    }


def newest_report(previous):
    current = set((ROOT / "logs").glob("mission_*/report.json"))
    created = sorted(current - previous, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"Expected exactly one new mission report, got {len(created)}")
    return created[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--scenarios", nargs="+", choices=("a_hit", "fallback"),
                        default=("a_hit", "fallback"))
    parser.add_argument("--miss-offset-y", type=float,
                        help="Forward a tested miss offset to fallback mission runs")
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"))
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        raise ValueError("repeats must be between 1 and 100")

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
                command = [sys.executable, str(ROOT / "python" / "run_balloon_mission.py"),
                           "--scenario", scenario]
                if scenario == "fallback" and args.miss_offset_y is not None:
                    command.extend(["--miss-offset-y", str(args.miss_offset_y)])
                if args.target_offset is not None:
                    command.extend(["--target-offset", *map(str, args.target_offset)])
                result = subprocess.run(command, cwd=ROOT)
                path = newest_report(previous)
                mission = json.loads(path.read_text(encoding="utf-8"))
                row = {
                    "repeat": repeat,
                    "scenario": scenario,
                    "run": path.parent.name,
                    "status": mission["status"],
                    "hit_vehicle": mission.get("hit_evidence", {}).get("vehicle", ""),
                    "landing_verified": mission.get("landing_verified", False),
                    "min_separation_m": mission.get("min_separation_m"),
                    "max_waiting_drift_m": mission.get("max_waiting_drift_m"),
                    "miss_to_dispatch_s": mission.get("miss_to_dispatch_s"),
                    "error": mission.get("error", ""),
                }
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
