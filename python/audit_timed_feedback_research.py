"""Independent replay of frozen timed-control feedback experiment records.

No optimizer, objective or tracking rollout is called. This checks recorded
arithmetic and provenance, not real-flight accuracy or generation chronology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NEURAL = ("scalar", "diagnostic", "zero", "shuffled")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13,
                               equal_nan=False, err_msg=label)


def features(details):
    m = details["tracking_metrics"]
    return [min(4., m["position_rmse_m"]/.25),
            min(4., m["max_position_error_m"]/.5),
            min(4., m["terminal_velocity_error_mps"]/.5),
            m["command_slew_clipped_time_fraction"], m["peak_position_error_u"]]


def check_result(score, feasible, details):
    checks = details["checks"]
    assert checks and all(isinstance(value, bool) for value in checks.values())
    failed = sum(not value for value in checks.values())
    assert failed == details["failed_checks"] and len(checks) == details["total_checks"]
    assert feasible == (failed == 0)
    t, j, error = (details[name] for name in ("duration_s", "jerk_energy_m2_s5", "tracking_rmse_m"))
    assert t > 0 and j >= 0 and error >= 0
    q = .7*t/(t+5.) + .1*j/(j+30.) + .2*error/(error+.25)
    close(details["quality_term"], q, "quality arithmetic")
    close(score, q if feasible else 1.+failed/len(checks)+q, "feasibility-first score")
    close(details["control_features"], features(details), "raw diagnostic normalization")
    m = details["tracking_metrics"]
    close(error, m["position_rmse_m"], "RMSE field")
    close(m["peak_position_error_time_s"]/t, m["peak_position_error_u"], "peak clock")
    for field in ("speed_clipped_time_fraction", "command_slew_clipped_time_fraction", "peak_position_error_u"):
        assert 0 <= m[field] <= 1.+1e-12
    assert checks["tracking_peak_error"] == (m["max_position_error_m"] <= .5)
    assert checks["terminal_position_error"] == (m["terminal_position_error_m"] <= .25)
    assert checks["terminal_velocity_error"] == (m["terminal_velocity_error_mps"] <= .5)


def replay_features(search):
    records, history = search["evaluations"], search["history"]
    n, budget, mode = search["population"], search["budget"], search["mode"]
    assert search["diagnostic_only_evaluations"] == 0 and search["full_evaluations"] == budget
    assert search["network_input_columns"] == (1 if mode == "scalar" else 6)
    assert len(history) == budget//n and len(search["feature_ledger"]) == budget//n-1
    parent_ids = sorted(range(1, n+1), key=lambda i: records[i-1]["score"])
    assert history[0]["parent_evaluation_ids"] == parent_ids
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([search["seed"], 8675309]))
    for generation, event in enumerate(search["feature_ledger"], 1):
        assert event["generation"] == generation
        assert event["parent_evaluation_ids"] == parent_ids
        assert all(i <= generation*n for i in parent_ids), "future evaluation enters features"
        parents = [records[i-1] for i in parent_ids]
        scores = [r["score"] for r in parents]
        diagnostics = np.array([r["control_features"] for r in parents])
        close(event["parent_normalized_parameters"], [r["normalized_parameters"] for r in parents], "parent positions")
        assert event["parent_scores"] == scores
        close(event["parent_control_features"], diagnostics, "parent diagnostics alignment")
        rank = np.arange(n, dtype=float)/(n-1)
        permutation = shuffle_rng.permutation(n) if mode == "shuffled" else np.arange(n)
        assert event["diagnostic_row_permutation"] == permutation.tolist()
        if mode == "scalar":
            expected = rank[:, None]
        else:
            extra = np.zeros_like(diagnostics) if mode == "zero" else diagnostics[permutation]
            expected = np.column_stack((rank, extra))
        close(event["consumed_features"], expected, "network-consumed features")
        children = list(range(generation*n+1, (generation+1)*n+1))
        old_best = scores[0]
        parent_ids = sorted(parent_ids+children, key=lambda i: records[i-1]["score"])[:n]
        h = history[generation]
        assert h["fe"] == (generation+1)*n and h["parent_evaluation_ids"] == parent_ids
        assert h["best"] == records[parent_ids[0]-1]["score"]
        close(h["success_rate"], np.mean([records[i-1]["score"] < old_best for i in children]), "success rate")
    assert search["final_parent_evaluation_ids"] == parent_ids
    assert search["final_scores"] == [records[i-1]["score"] for i in parent_ids]
    close(search["final_population_normalized"], [records[i-1]["normalized_parameters"] for i in parent_ids], "final parents")
    close(search["final_control_features"], [records[i-1]["control_features"] for i in parent_ids], "final diagnostics")
    assert search["best_evaluation_id"] == parent_ids[0]
    close(search["best_control_features"], records[parent_ids[0]-1]["control_features"], "incumbent diagnostics")


def check_trace(selected):
    trace = selected["tracking_trace"]
    times = np.array(trace["times_s"])
    q, p, v = (np.array(trace[key]) for key in ("reference_position_m", "actual_position_m", "actual_velocity_mps"))
    command = np.array(trace["velocity_command_mps"])
    h = np.diff(times)
    assert times[0] == 0 and times[-1] == selected["score_details"]["duration_s"]
    assert np.all(h > 0) and len(command) == len(h) and len(q) == len(h)+1
    error2 = np.sum((q-p)**2, axis=1)
    m = selected["score_details"]["tracking_metrics"]
    close(m["position_rmse_m"], np.sqrt(np.sum(.5*(error2[:-1]+error2[1:])*h)/times[-1]), "trace RMSE")
    close(m["max_position_error_m"], np.sqrt(np.max(error2)), "trace peak")
    close(m["peak_position_error_time_s"], times[np.argmax(error2)], "trace peak time")
    condition = selected["condition"]
    assert trace["condition"] == condition
    tau = condition["velocity_lag_s"]
    alpha = -np.expm1(-h/tau)
    close(v[1:], v[:-1]+(command-v[:-1])*alpha[:,None], "recorded ZOH velocity recurrence")
    close(p[1:], p[:-1]+command*h[:,None]+(v[:-1]-command)*tau*alpha[:,None], "recorded ZOH position recurrence")
    close(trace["reference_velocity_mps"][-1], [1.,0.,0.], "default nonzero terminal reference")


def audit(folder):
    folder = Path(folder)
    manifest, rows = load(folder/"manifest.json"), load(folder/"rows.json")
    args = manifest["args"]
    assert manifest["protocol"] == "independent-timed-control-feedback-pilot-v1"
    assert len(rows) == manifest["expected_runs"] == len(manifest["execution_plan"])*len(args["methods"])
    current_differences = []
    for source, digest in manifest["source_sha256"].items():
        name = str(source).replace("\\", "/").rsplit("/", 1)[-1]
        assert hashlib.sha256((folder/"source_snapshot"/name).read_bytes()).hexdigest() == digest
        current = ROOT/str(source).replace("\\", "/")
        if not current.exists() or hashlib.sha256(current.read_bytes()).hexdigest() != digest:
            current_differences.append(source)
    expected_order = [(b["family"], b["layout"], b["seed"], method, position)
                      for b in manifest["execution_plan"] for position, method in enumerate(b["method_order"])]
    assert [(r["family"], r["layout"], r["seed"], r["method"], r["execution_order_index"]) for r in rows] == expected_order
    train = manifest["training_condition"]
    assert train["velocity_lag_s"] == .30 and manifest["validation_lags_s"] == [.15,.45]
    initial, model_identity, indexed = {}, {}, {}
    observed_train = declared_train = neural_runs = feature_events = 0
    for row in rows:
        key = row["family"], row["layout"], row["seed"]
        raw = load(folder/("_".join(map(str, (row["family"],row["layout"],row["method"],row["seed"])))+".json"))
        assert raw["summary"] == row
        search, selected = raw["search"], raw["selected_training"]
        records, budget = search["evaluations"], args["budget"]
        assert row["observed_training_calls"] == len(records) == budget == search["budget"]
        assert [e["fe"] for e in records] == list(range(1,budget+1))
        assert row["best_score"] == search["best_score"] == min(r["score"] for r in records)
        assert row["feasible"] == selected["feasible"]
        assert selected["parameters"] == search["best_parameters"]
        assert selected["score"] == search["best_score"] and selected["condition"] == train
        initial_entry = (search["initial_population_normalized"], [e["parameters"] for e in records[:manifest["population"]]])
        if key in initial: assert initial[key] == initial_entry
        initial[key] = initial_entry
        for e in records:
            check_result(e["score"], e["feasible"], e["details"])
            assert e["evaluation_s"] >= 0
            condition = e["details"].get("tracking_condition")
            if condition is None:
                declared_train += 1
            else:
                assert condition == train, "validation model condition entered training"
                observed_train += 1
        assert row["failed_training_candidates"] == sum(not e["feasible"] for e in records)
        for h in search["history"]:
            assert h["best"] == min(e["score"] for e in records[:h["fe"]])
        selected_records = [e for e in records if e["parameters"] == selected["parameters"] and e["score"] == selected["score"]]
        assert selected_records
        for measured in selected_records:
            close(selected["score_details"]["control_features"], measured["details"]["control_features"], "selected diagnostic recheck")
            assert selected["score_details"]["tracking_metrics"] == measured["details"]["tracking_metrics"]
        check_result(selected["score"], selected["feasible"], selected["score_details"])
        check_trace(selected)
        assert len(raw["validation"]) == row["validation_conditions"] == 2
        for result, lag in zip(raw["validation"], manifest["validation_lags_s"]):
            expected_condition = dict(train, velocity_lag_s=lag)
            assert result["condition"] == expected_condition
            assert result["parameters"] == selected["parameters"]
            check_result(result["score"], result["feasible"], result["score_details"])
        assert row["validation_passes"] == sum(r["feasible"] for r in raw["validation"])
        assert row["post_selection_rollouts"] == 3 and row["screening_enabled"] is False
        if row["method"] in NEURAL:
            neural_runs += 1
            replay_features(search)
            feature_events += len(search["feature_ledger"])
            if row["method"] != "scalar":
                identity = search["parameter_count"], search["initial_model_sha256"]
                if key in model_identity: assert model_identity[key] == identity
                model_identity[key] = identity
        indexed[key+(row["method"],)] = row
    for group in load(folder/"summary.json"):
        chosen = [r for r in rows if (r["family"],r["layout"],r["method"]) == (group["family"],group["layout"],group["method"])]
        assert group["n"] == len(chosen)
        for target, source, operation in [("mean_selected_score","best_score",np.mean), ("median_selected_score","best_score",np.median),
                    ("median_wall_s","wall_s",np.median),("median_cpu_s","cpu_s",np.median),
                    ("mean_reference_duration_s","reference_duration_s",np.mean),("mean_tracking_rmse_m","tracking_rmse_m",np.mean)]:
            close(group[target],operation([r[source] for r in chosen]),target)
        feasible = [r for r in chosen if r["feasible"]]
        assert group["feasible_selected"] == len(feasible)
        if feasible: close(group["mean_feasible_selected_score"],np.mean([r["best_score"] for r in feasible]),"feasible score mean")
        else: assert group["mean_feasible_selected_score"] is None
        assert group["total_failed_training_candidates"] == sum(r["failed_training_candidates"] for r in chosen)
        assert group["validation_condition_passes"] == sum(r["validation_passes"] for r in chosen)
    for group in load(folder/"paired_comparisons.json"):
        delta = np.array([indexed[(group["family"],group["layout"],seed,"diagnostic")]["best_score"]-
                          indexed[(group["family"],group["layout"],seed,group["comparator"])]["best_score"] for seed in args["seeds"]])
        assert group["n"] == len(delta)
        close(group["mean_paired_score_difference"],np.mean(delta),"paired score difference")
        assert (group["better_runs"],group["equal_runs"],group["worse_runs"]) == (sum(delta<0),sum(delta==0),sum(delta>0))
        if len(delta) < 5: assert group["exploratory_paired_bootstrap95"] is None
        else:
            rng=np.random.default_rng(20261029)
            bootstrap=np.mean(delta[rng.integers(0,len(delta),(5000,len(delta)))],axis=1)
            close(group["exploratory_paired_bootstrap95"],np.quantile(bootstrap,[.025,.975]),"paired bootstrap")
    completion = load(folder/"completion.json")
    assert completion["status"] == "COMPLETE" and completion["runs"] == len(rows)
    assert completion["training_calls"] == sum(r["observed_training_calls"] for r in rows)
    assert completion["post_selection_rollouts"] == sum(r["post_selection_rollouts"] for r in rows)
    assert completion["feasible_selected"] == sum(r["feasible"] for r in rows)
    result = dict(status="PASS", runs=len(rows), training_evaluations=completion["training_calls"],
        neural_runs=neural_runs, feature_events_replayed=feature_events,
        per_query_training_condition_verified=observed_train, training_condition_declared_only=declared_train,
        source_snapshot_hashes_verified=True, current_source_differences=current_differences,
        producer_attests_source_bytes_unchanged=completion["source_bytes_unchanged"],
        raw_record_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.json") if p.name!="audit.json"},
        scope="Recorded budget, source bytes, stable selection, diagnostic consumption, score/trace arithmetic and declared post-selection validation; no optimizer or rollout executed.",
        limitations=["Record consistency cannot prove actual generation chronology or absence of unlogged external calls.",
                    "Queries without tracking_condition are checked against manifest and producer snapshot only; their individual model condition is not independently logged.",
                    "Current checkout may evolve; frozen snapshots remain authoritative. No real-flight or generalization verification."])
    (folder/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return {k:v for k,v in result.items() if k!="raw_record_sha256"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders",nargs="+")
    for folder in parser.parse_args().folders:
        print(json.dumps(dict(folder=str(folder),**audit(folder)),ensure_ascii=False,allow_nan=False))
