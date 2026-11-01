"""Replay six-group frozen records without making new objective calls."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import math
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def near(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=False)


def audit(path):
    path = Path(path)
    config, manifest, completion = (read(path/name) for name in ("config.json", "manifest.json", "completion.json"))
    assert completion["status"] == "COMPLETE"
    for name, digest in completion["artifact_sha256"].items():
        assert hashlib.sha256((path/name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((path/"source_snapshot"/name).read_bytes()).hexdigest() == digest, name
    assert config["population"] == 20 and config["proposal_budget"] == 80 and config["dropout"] == .95
    rows, plans = read(path/"rows.json"), read(path/"airsim_replay_plan.json")["plans"]
    assert len(rows) == len(plans) == 6
    count = 0
    samples = 0
    groups = []
    for row, plan, spec in zip(rows, plans, config["groups"]):
        record = read(path/row["record_json"])
        assert record["group_id"] == row["group_id"] == plan["group_id"] == spec["group_id"]
        assert record["parameters"] == plan["parameters"]
        selected = record["selected_validation"]
        details, trace = selected["score_details"], selected["tracking_trace"]
        assert selected["feasible"] == row["selected_feasible"] == all(details["checks"].values())
        condition = trace["condition"]
        assert condition == config["tracking_condition"]
        t = np.asarray(trace["times_s"])
        q, qv = np.asarray(trace["reference_position_m"]), np.asarray(trace["reference_velocity_mps"])
        p, v = np.asarray(trace["actual_position_m"]), np.asarray(trace["actual_velocity_mps"])
        c, raw = np.asarray(trace["velocity_command_mps"]), np.asarray(trace["raw_velocity_command_mps"])
        T = record["parameters"][-1]
        assert t[0] == 0 and t[-1] == T and np.all(np.diff(t) > 0)
        near(p[0], q[0]); near(v[0], qv[0])
        error = np.linalg.norm(q-p, axis=1)
        rmse = float(np.sqrt(np.sum(.5*(error[:-1]**2+error[1:]**2)*np.diff(t))/T))
        near(rmse, row["tracking_rmse_m"])
        near(error.max(), row["peak_position_error_m"])
        near(error[-1], row["terminal_position_error_m"])
        near(np.linalg.norm(v[-1]-qv[-1]), row["terminal_velocity_error_mps"])
        near(raw, qv[:-1]+condition["position_gain"]*(q[:-1]-p[:-1]))
        previous = qv[0]
        for i, h in enumerate(np.diff(t)):
            speed = raw[i]*min(1., condition["max_command_speed_mps"]/max(np.linalg.norm(raw[i]),1e-300))
            change = speed-previous
            expected = previous+change*min(1., condition["max_command_slew_mps2"]*h/max(np.linalg.norm(change),1e-300))
            near(c[i], expected)
            beta = -math.expm1(-h/condition["velocity_lag_s"])
            near(v[i+1], v[i]+(c[i]-v[i])*beta)
            near(p[i+1], p[i]+c[i]*h+(v[i]-c[i])*condition["velocity_lag_s"]*beta)
            previous = c[i]
        quality = .7*T/(T+5)+.1*details["jerk_energy_m2_s5"]/(details["jerk_energy_m2_s5"]+30)+.2*rmse/(rmse+.25)
        near(quality, details["quality_term"])
        failures = sum(not x for x in details["checks"].values())
        score = quality if failures == 0 else 1+failures/len(details["checks"])+quality
        near(score, selected["score"]); near(score, row["selected_score"])
        search = record["search"]
        if search is None:
            assert row["search_objective_calls"] == 0 and record["parameters"] == config["anchor_parameters"]
        else:
            entries = search["evaluations"]
            assert len(entries) == row["search_objective_calls"] == 80
            assert search["population"] == 20 and search["screening_enabled"] is False
            best = min(entries, key=lambda r:r["score"])
            near(best["score"], selected["score"])
            near(best["parameters"], record["parameters"])
            assert record["selection_provenance"]["proposal_index"] == best["proposal_index"]
            assert entries[best["proposal_index"]-1] == best
            count += len(entries)
        assert row["selected_validation_calls"] == 1
        near(row["search_plus_selected_validation_wall_s"], row["search_wall_s"]+row["selected_validation_wall_s"])
        samples += len(t)
        groups.append(dict(group_id=row["group_id"], model_feasible=selected["feasible"],
            best_proposal_index=record["selection_provenance"]["proposal_index"]))
    assert count == 320
    result = dict(status="PASS", groups=6, search_calls_in_records=count, selected_revalidations=6,
        selected_state_samples=samples, group_provenance=groups,
        scope="Record hashes, source snapshots, selected ZOH/controller/metric replay and ledger consistency. "
              "Reference geometry verdicts are consumed as recorded; no independent geometry-certificate "
              "or AirSim verification, hidden-call/timing chronology proof, or efficacy conclusion.")
    (path/"audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
