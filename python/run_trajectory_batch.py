"""Versioned paired trajectory interface trials; all started flights retained."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from setup_scene import ROOT
from trajectory_single_experiment import create_trajectory_experiment


def main(config, target_source, output_dir, session_record, reverse_order=False):
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    order = ["perception", "efficiency"] if reverse_order else ["efficiency", "perception"]
    report = {"status": "RUNNING", "kind": "TRAJECTORY_CONTROLLER_INTERFACE_PILOT",
              "target_source": target_source, "formal_thesis_effect_results": False,
              "planned_policies": order, "runs": [], "active_run": None,
              "configuration_path": str(Path(config).resolve()),
              "configuration_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
              "batch_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "session": json.loads(Path(session_record).read_text(encoding="utf-8-sig")),
              "started_local": datetime.now().astimezone().isoformat(),
              "ai_assistance": "experiment orchestration and reporting authored with AI assistance"}
    path = folder / "batch_report.json"

    def save():
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    save()
    try:
        for index, policy in enumerate(order, 1):
            run = create_trajectory_experiment(policy=policy, target_source=target_source, config_path=Path(config))
            entry = {"run": index, "policy": policy, "status": "STARTED",
                     "report_path": str(run.folder / "report.json")}
            report["runs"].append(entry)
            report["active_run"] = index
            save()
            failure = None
            try:
                run.run()
            except BaseException as exc:
                failure = exc
            if not Path(entry["report_path"]).exists():
                entry["status"] = "NO_TERMINAL_REPORT"
                raise RuntimeError("Started flight has no terminal report; preserve simulator") from failure
            result = json.loads(Path(entry["report_path"]).read_text(encoding="utf-8"))
            entry.update(status=result["status"], task_success=result.get("task_success", False),
                         report_sha256=hashlib.sha256(Path(entry["report_path"]).read_bytes()).hexdigest(),
                         metrics=result.get("metrics", {}))
            report["active_run"] = None
            save()
            landing = result.get("landing_evidence", {})
            if failure or result["status"] != "PASS" or not landing.get("landed_flag") or not landing.get("api_control_released"):
                raise RuntimeError(f"Unsafe or incomplete interface run {index}; preserve simulator") from failure
        report["status"] = "PASS"
        report["task_success_count"] = sum(item.get("task_success") is True for item in report["runs"])
    except BaseException as exc:
        report.update(status="FAIL", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["finished_local"] = datetime.now().astimezone().isoformat()
        save()
        print(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-source", choices=("geometry_oracle", "sensor_surface_estimate"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-record", type=Path, required=True)
    parser.add_argument("--reverse-order", action="store_true")
    args = parser.parse_args()
    main(args.config, args.target_source, args.output_dir, args.session_record, args.reverse_order)
