"""Archive verified local source and configuration bytes for one completed pilot."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SOURCES = {"single_drone_experiment.py", "visual_servo.py", "target_provider.py",
    "balloon_detector.py", "trajectory_single_experiment.py", "perception_trajectory_planner.py",
    "timestamped_target_provider.py", "matched_trajectory_start.py", "trajectory_execution_guard.py",
    "trajectory_timed_reference.py", "trajectory_timed_experiment.py", "trajectory_rgb_quality_reference.py"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(manifest_path):
    manifest_path = manifest_path.resolve()
    manifest = read(manifest_path)
    files = {}
    reports = []
    for session in manifest["sessions"]:
        batch_path = Path(session["batch_path"]) / "batch_report.json"
        batch = read(batch_path)
        for entry in batch["runs"]:
            report_path = Path(entry["report_path"])
            if digest(report_path) != entry["report_sha256"]:
                raise ValueError("Run hash changed")
            report = read(report_path)
            reports.append({"path": str(report_path), "sha256": digest(report_path)})
            for name, expected in report["source_sha256"].items():
                if name not in ALLOWED_SOURCES:
                    raise ValueError("Unexpected source filename")
                source = ROOT / "python" / name
                if digest(source) != expected:
                    raise ValueError(f"Live source differs from trial: {name}")
                files[source] = expected
    for path_key, hash_key in (("experiment_config", "experiment_config_sha256"), ("settings_path", "settings_sha256")):
        source = Path(manifest[path_key]).resolve()
        if source.suffix != ".json" or not source.is_relative_to(ROOT / "configs"):
            raise ValueError("Snapshot configuration is outside project configs")
        expected = manifest[hash_key]
        if digest(source) != expected:
            raise ValueError("Frozen configuration hash changed")
        files[source] = expected
    for relative in ("python/analyze_trajectory_pilot.py", "python/run_trajectory_batch.py",
                     "python/trajectory_timed_audit.py",
                     "python/audit_trajectory_matched_gate.py", "scripts/run-trajectory-interface-pilot.ps1",
                     "scripts/assert-trajectory-source-freeze.ps1",
                     "scripts/run-trajectory-matched-pilot.ps1"):
        source = ROOT / relative
        files[source] = digest(source)
    output = manifest_path.parent / "source_snapshot"
    output.mkdir(exist_ok=False)
    rows = []
    for source, expected in files.items():
        destination = output / source.name
        shutil.copy2(source, destination)
        if digest(destination) != expected:
            raise ValueError("Snapshot bytes differ")
        rows.append({"source": str(source), "snapshot": str(destination), "sha256": expected})
    receipt = {"manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
        "reports": reports, "files": rows, "scope": "flight source hashes verified; analysis and launcher bytes archived at analysis time"}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    main(parser.parse_args().manifest)
