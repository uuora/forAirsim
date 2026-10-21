"""Counterbalanced repeated single-drone stand-off comparison.

Runs one policy at a time and stops at the first failed or unsafe run. A PASS
means the aircraft landed safely; task_success is separately recorded.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

from setup_scene import ROOT
from single_drone_experiment import SingleDroneExperiment, CONFIG_PATH


def policy_order(repeats):
    if not 1 <= repeats <= 20:
        raise ValueError("repeats must be 1..20")
    return [policy for index in range(repeats)
            for policy in (("coordinate", "visual") if index % 2 == 0
                           else ("visual", "coordinate"))]


def main(repeats, output_dir=None, session_record=None, config_path=CONFIG_PATH):
    order = policy_order(repeats)
    folder = Path(output_dir).resolve() if output_dir else ROOT / "logs" / datetime.now().strftime("single_batch_%Y%m%d_%H%M%S_%f")
    if folder.exists():
        raise ValueError(f"Output already exists; preserve previous records: {folder}")
    session = json.loads(Path(session_record).read_text(encoding="utf-8-sig")) if session_record else None
    folder.mkdir(parents=True)
    report = {"status": "RUNNING", "repeats": repeats,
              "order": order, "runs": [], "session": session,
              "configuration_path": str(Path(config_path).resolve()),
              "configuration_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
              "batch_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scene": "Blocks; fixed red balloon; DroneA commanded; DroneB stationary"}
    report_file = folder / "batch_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    try:
        for index, policy in enumerate(report["order"], start=1):
            run = SingleDroneExperiment(policy, config_path=Path(config_path))
            report["active_run"] = {"run": index, "pair": (index + 1) // 2,
                "policy": policy, "report_path": str(run.folder / "report.json")}
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            failure = None
            try:
                run.run()
            except BaseException as exc:
                failure = exc
            result = json.loads((run.folder / "report.json").read_text(encoding="utf-8"))
            metrics = result.get("metrics", {})
            row = {"run": index, "pair": (index + 1) // 2,
                   "policy": policy, "status": result["status"],
                   "task_success": metrics.get("task_success"),
                   "approach_duration_s": metrics.get("approach_duration_s"),
                   "approach_path_m": metrics.get("approach_path_m"),
                   "observation_count": metrics.get("observation_count"),
                   "valid_detection_fraction": metrics.get("valid_detection_fraction"),
                   "mean_max_axis_error": metrics.get("mean_max_axis_error"),
                   "terminal_mean_max_axis_error": metrics.get("terminal_mean_max_axis_error"),
                   "last_surface_range_m": metrics.get("last_surface_range_m"),
                   "report_path": str(run.folder / "report.json")}
            rows.append(row)
            report["runs"].append(row)
            report["active_run"] = None
            report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if failure is not None:
                raise RuntimeError(f"Run {index} failed; inspect paused AirSim state") from failure
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=str(exc), stopped_after_runs=len(rows))
        raise
    finally:
        with (folder / "runs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=("run", "pair", "policy", "status",
                "task_success", "approach_duration_s", "approach_path_m",
                "observation_count", "valid_detection_fraction", "mean_max_axis_error",
                "terminal_mean_max_axis_error",
                "last_surface_range_m", "report_path"))
            writer.writeheader()
            writer.writerows(rows)
        (folder / "batch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        print(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1, help="1..20 paired repetitions")
    parser.add_argument("--output-dir", type=Path, help="New output directory; existing paths are refused")
    parser.add_argument("--session-record", type=Path, help="Unreal process and scene preflight provenance")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    main(args.repeats, args.output_dir, args.session_record, args.config)
