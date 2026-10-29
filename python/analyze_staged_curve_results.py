"""Recompute the saved exploratory staged-schedule comparisons, read-only.

Example: python python/analyze_staged_curve_results.py --outputcampaign
reports/hermite_parameter_perturbation_20261027/staged_unlock
The archived staged_comparisons.json is checked and never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_parameter_search_budget import paired_change
import run_parameter_perturbation_research as protocol
from trajectory_research.parameterizations import default_parameters


METHODS = ("staged_4to4", "staged_4to7", "staged_7to7")
LAYOUTS = ("central", "early_offset", "two_obstacles")


def recompute(directory):
    directory = Path(directory).resolve()
    manifest, selection = protocol.load_selection(directory)
    if manifest["schema"] != "initial-curve-staged-perturbation/1" \
            or set(manifest["methods"]) != set(METHODS) \
            or set(manifest["layouts"]) != set(LAYOUTS) \
            or manifest["families"] != ["paired_bump"] \
            or manifest["validation"]["count"] != 32:
        raise ValueError("Expected the complete paired-bump three-schedule, three-layout, 32-condition study")
    # Frozen sources are checked against their own manifest, without requiring
    # a future checkout to have identical line endings or source revisions.
    for relative, expected in manifest["source_sha256"].items():
        if protocol.digest(directory / "source_snapshot" / relative) != expected:
            raise ValueError("Frozen staged source snapshot changed")
    validation = protocol.read(directory / "validation_summary.json")
    summary = protocol.read(directory / "summary.json")
    hashes = dict(manifest_sha256=protocol.digest(directory / "manifest.json"),
        selection_sha256=protocol.digest(directory / "selection.json"),
        stress_sha256=protocol.digest(directory / "stress_cases.json"))
    if any(validation[key] != value for key, value in hashes.items()) \
            or summary["manifest_sha256"] != hashes["manifest_sha256"] \
            or summary["validation_summary_sha256"] != protocol.digest(directory / "validation_summary.json"):
        raise ValueError("Staged summary/selection/stress hash binding mismatch")
    choices = {(c["family"], c["layout"], c["method"], c["seed"]): c for c in selection["incumbents"]}
    expected = set(choices) | {("paired_bump", layout, "initial", None) for layout in LAYOUTS}
    identities = [(r["family"], r["layout"], r["method"], r["seed"]) for r in validation["rows"]]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Staged validation rows lack complete unique seed/schedule coverage")
    for row, identity in zip(validation["rows"], identities):
        name = "validation/" + protocol.key(*identity) + ".json"
        if row["result_file"] != name or row["result_sha256"] != protocol.digest(directory / name):
            raise ValueError("Staged validation result file/hash mismatch")
        result = protocol.read(directory / name)
        if any(result[key] != value for key, value in hashes.items()) \
                or any(result[key] != row[key] for key in row if key not in ("result_file", "result_sha256")):
            raise ValueError("Staged validation summary differs from its result")
        parameters = choices[identity]["parameters"] if identity in choices else default_parameters("paired_bump").tolist()
        if result["parameters"] != parameters or len(result["cases"]) != 32 or result["conditions"] != 32 \
                or result["failures"] != sum(not case["feasible"] for case in result["cases"]) \
                or result["all_conditions_pass"] != all(case["feasible"] for case in result["cases"]):
            raise ValueError("Staged validation parameters/count/arithmetic mismatch")
    if summary["training_runs"] != len(choices) \
            or summary["training_candidates"] != len(choices) * manifest["budget"] \
            or summary["validation_conditions"] != sum(row["conditions"] for row in validation["rows"]) \
            or summary["failed_validation_conditions"] != sum(row["failures"] for row in validation["rows"]):
        raise ValueError("Staged aggregate arithmetic mismatch")
    comparisons = []
    for layout in LAYOUTS:
        for first, second in (("staged_4to7", "staged_4to4"), ("staged_4to7", "staged_7to7"),
                              ("staged_4to4", "staged_7to7")):
            a = {r["seed"]: r for r in validation["rows"] if r["layout"] == layout and r["method"] == first}
            b = {r["seed"]: r for r in validation["rows"] if r["layout"] == layout and r["method"] == second}
            if set(a) != set(manifest["seeds"]) or set(b) != set(manifest["seeds"]):
                raise ValueError("Paired staged optimizer-seed coverage mismatch")
            for field in ("quality", "duration_s", "jerk_energy_m2_s5"):
                comparisons.append(dict(layout=layout, first=first, second=second,
                                        **paired_change(a, b, field)))
    return dict(schema="staged-schedule-exploratory-comparisons/1", comparisons=comparisons,
        independent_unit="optimizer seed within fixed layout",
        note="post-development 10-seed study; paired bootstrap not multiplicity adjusted; restart and dimension both vary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputcampaign", type=Path, required=True)
    parser.add_argument("--destination", type=Path,
                        help="new JSON path; default is staged_comparisons_recomputed.json within the campaign")
    args = parser.parse_args()
    campaign = args.outputcampaign.resolve()
    report = recompute(campaign)
    archived = campaign / "staged_comparisons.json"
    if protocol.read(archived) != report:
        raise ValueError("Recomputed comparisons differ numerically from the archived report")
    destination = (args.destination or campaign / "staged_comparisons_recomputed.json").resolve()
    if destination == archived.resolve():
        raise ValueError("Choose a new output path; archived comparisons must remain unchanged")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", comparisons=len(report["comparisons"]),
        numerically_equal_to_archived=True, archived_sha256=protocol.digest(archived), output=str(destination)), indent=2))


if __name__ == "__main__":
    main()
