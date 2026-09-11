"""Run named target-position cases through the fixed-scene batch validator."""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from setup_scene import ROOT, validate_target_offset


def validate_cases(raw):
    cases = raw.get("cases", [])
    if not cases:
        raise ValueError("Validation matrix must contain at least one case")
    names = [case.get("name") for case in cases]
    if any(not name or not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError("Validation case names must be non-empty and unique")
    for case in cases:
        validate_target_offset(case["target_offset_ned_m"])
        if abs(float(case["miss_offset_y_m"])) < 1.5:
            raise ValueError(f"Unsafe miss offset in case {case['name']}")
        repeats = int(case.get("repeats", 1))
        if not 1 <= repeats <= 20:
            raise ValueError(f"Repeats outside 1..20 in case {case['name']}")
    return cases


def newest_batch(previous):
    created = sorted(set((ROOT / "logs").glob("batch_*/batch_report.json")) - previous,
                     key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"Expected one new batch report, got {len(created)}")
    return created[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "validation_matrix.json")
    parser.add_argument("--case", action="append", dest="selected_cases",
                        help="Run only this named case; repeat the option to select several")
    args = parser.parse_args()
    cases = validate_cases(json.loads(args.config.read_text(encoding="utf-8")))
    if args.selected_cases:
        requested = set(args.selected_cases)
        cases = [case for case in cases if case["name"] in requested]
        missing = requested - {case["name"] for case in cases}
        if missing:
            raise ValueError(f"Unknown validation cases: {sorted(missing)}")

    output_dir = ROOT / "logs" / datetime.now().strftime("matrix_%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
    report = {"status": "RUNNING", "configuration": str(args.config), "cases": []}
    try:
        for case in cases:
            print(f"MATRIX case={case['name']}", flush=True)
            previous = set((ROOT / "logs").glob("batch_*/batch_report.json"))
            offset = case["target_offset_ned_m"]
            command = [sys.executable, str(ROOT / "python" / "batch_validate.py"),
                       "--repeats", str(case.get("repeats", 1)), "--scenarios", "fallback",
                       "--miss-offset-y", str(case["miss_offset_y_m"]),
                       "--target-offset", *map(str, offset)]
            result = subprocess.run(command, cwd=ROOT)
            batch_path = newest_batch(previous)
            batch = json.loads(batch_path.read_text(encoding="utf-8"))
            report["cases"].append({"name": case["name"], "batch_report": str(batch_path.relative_to(ROOT)),
                                    "status": batch["status"], "aggregate": batch["aggregate"]})
            (output_dir / "matrix_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            if result.returncode or batch["status"] != "PASS":
                raise RuntimeError(f"Matrix stopped after failed case: {case['name']}")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        report["finished_at"] = datetime.now().astimezone().isoformat()
        (output_dir / "matrix_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Matrix report: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
