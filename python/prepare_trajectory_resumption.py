"""Preserve an interrupted pilot and prepare a separately identified repeat.

This is a file-only reconciliation. It does not infer safe landing, repair raw
reports, or remove started flights from the original planned denominator.
"""
from copy import deepcopy
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def prepare(sweep_path, manifest_path, output):
    sweep_path, manifest_path, output = map(lambda p: Path(p).resolve(),
                                           (sweep_path, manifest_path, output))
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite: {output}")
    sweep, manifest = read(sweep_path), read(manifest_path)
    if sweep.get("status") != "RUNNING" or manifest.get("status") != "RUNNING":
        raise ValueError("Only an interrupted RUNNING ledger may be reconciled")
    pending = [entry for entry in sweep["sessions"] if entry.get("status") == "STARTED"]
    if len(pending) != 1 or pending[0] != sweep["sessions"][-1]:
        raise ValueError("Expected one final pending session")
    conditions_path = Path(sweep["conditions_path"])
    if digest(conditions_path) != sweep["conditions_sha256"]:
        raise ValueError("Frozen plan hash mismatch")
    conditions = read(conditions_path)
    pending = pending[0]
    plans = [p for p in conditions["session_plan"]
             if (p["condition_id"], p["repetition"]) ==
             (pending["condition_id"], pending["repetition"])]
    if len(plans) != 1 or plans[0]["method_order"] != pending["policy_order"]:
        raise ValueError("Interrupted method order does not match frozen plan")
    config = next(c for c in conditions["conditions"]
                  if c["condition_id"] == pending["condition_id"])
    if manifest.get("experiment_config_sha256") != config["configuration_sha256"]:
        raise ValueError("Interrupted manifest configuration differs")
    if digest(config["configuration_path"]) != config["configuration_sha256"]:
        raise ValueError("Frozen flight configuration hash mismatch")
    if len(manifest.get("sessions", [])) != 1:
        raise ValueError("Expected one interrupted Unreal session")
    inner = manifest["sessions"][0]
    batch_path = Path(inner["batch_path"]) / "batch_report.json"
    batch = read(batch_path)
    if batch.get("status") != "RUNNING" or batch.get("planned_policies") != plans[0]["method_order"]:
        raise ValueError("Interrupted batch does not match planned pair")
    started = batch.get("runs", [])
    if not started or [r["policy"] for r in started] != plans[0]["method_order"][:len(started)]:
        raise ValueError("Started flight records are not a planned prefix")
    if all(Path(r["report_path"]).exists() for r in started):
        raise ValueError("No missing terminal report; investigate separately")

    reconciled = deepcopy(sweep)
    reconciled.update(status="INTERRUPTED_WITH_MISSING_TERMINAL_REPORT",
        reconciliation_scope="new interpretation copy; original raw ledger unmodified",
        original_sweep_path=str(sweep_path), original_sweep_sha256=digest(sweep_path),
        reconciled_local=datetime.now().astimezone().isoformat(),
        terminal_safety_of_interrupted_run="UNKNOWN_NOT_INFERRED_FROM_PROCESS_EXIT")
    reconciled["sessions"][-1].update(status="INTERRUPTED_WITH_MISSING_TERMINAL_REPORT",
        manifest=str(manifest_path), manifest_sha256=digest(manifest_path))
    replacement = deepcopy(conditions)
    replacement.update(protocol_id=conditions["protocol_id"] + "_separate_center_repeat_after_interruption",
        research_scope="separately_registered_repeat_after_interruption_all_original_started_retained",
        prepared_local=datetime.now().astimezone().isoformat(),
        parent_sweep_path=str(sweep_path), parent_sweep_sha256=digest(sweep_path),
        parent_planned_session_index=plans[0]["planned_session_index"],
        preparation_status="PREPARED_NOT_EXECUTED", planned_condition_count=1,
        planned_independent_unreal_sessions=1, planned_flights=2,
        conditions=[deepcopy(config)], session_plan=[deepcopy(plans[0])])
    replacement["session_plan"][0]["planned_session_index"] = 1
    replacement["conditions"][0]["planned_repetitions"] = 1
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("original_reconciled_sweep.json", reconciled),
                        ("separate_repeat_conditions.json", replacement)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                             allow_nan=False) + "\n", encoding="utf-8")
    receipt = {"original_inputs": [{"path": str(p), "sha256": digest(p)}
                                   for p in (sweep_path, manifest_path, batch_path, conditions_path)],
        "derived_files": [{"path": str(p), "sha256": digest(p)}
                          for p in sorted(output.iterdir())],
        "all_original_started_flights_retained": True,
        "missing_terminal_report_count": sum(not Path(r["report_path"]).exists() for r in started),
        "started_flights_in_interrupted_session": len(started),
        "ai_assistance": "File reconciliation implemented with AI assistance; no terminal result fabricated"}
    (output / "reconciliation_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.sweep, args.manifest, args.output)
