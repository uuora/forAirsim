"""Evaluate the audited starting-state gate before another development session."""
import argparse
import json
from pathlib import Path


def failures(summary):
    reasons = []
    runs = summary.get("runs", [])
    if len(runs) != 2 or summary.get("started_flights") != 2:
        reasons.append("REQUIRE_TWO_STARTED_RUNS")
    for row in runs:
        if row.get('approach_spatial_budget',{}).get('status') in ('OBSERVED_EXCEEDANCE','PARTIALLY_ASSESSED'):
            reasons.append(f"{row.get('policy')}:APPROACH_SPATIAL_GATE_FAILED")
        if row.get("timed_execution_audit", {}).get("status") == "INVALID":
            reasons.append(f"{row.get('policy')}:TIME_INDEXED_COMMAND_TRACE_GATE_FAILED")
        match = row.get("initial_state_match", {})
        if match.get("status") != "ACCEPTED_VERIFIED" or not match.get("acceptance_verified"):
            reasons.append(f"{row.get('policy')}:INITIAL_STATE_NOT_VERIFIED")
        if match.get("physical_sampled_budget_status") != "NO_EXCEEDANCE_IN_RECORDED_SAMPLES":
            reasons.append(f"{row.get('policy')}:POSITIONING_MOTION_GATE_FAILED")
        if match.get("command_xy_norm_violations") != 0 or match.get("command_xy_norm_invalid_count") != 0:
            reasons.append(f"{row.get('policy')}:POSITIONING_COMMAND_GATE_FAILED")
    pairs = summary.get("initial_state_pairs", {})
    if pairs.get("paired_session_sample_count") != 1:
        reasons.append("ACCEPTED_PAIR_NOT_VERIFIED")
    return reasons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.summary.read_text(encoding="utf-8-sig"))
    rejected = failures(result)
    print(json.dumps({"gate": "PASS" if not rejected else "FAIL", "reasons": rejected,
        "scope": "sampled initial-state, available approach spatial and request evidence; task completion assessed separately"}, ensure_ascii=False))
    raise SystemExit(1 if rejected else 0)
