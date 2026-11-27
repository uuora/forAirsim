"""Export the small-field paired campaign and replay it from local files."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from segment_retiming import SegmentTimeMap


ROOT = Path(__file__).resolve().parents[1]
ROUTE_NAMES = {"s": "S 型", "u-turn": "U 型回转", "arch": "三维拱形"}
GEOMETRY_NAMES = {"hermite": "Hermite", "bezier": "Bézier", "bspline": "七次 B 样条"}
ARMS = ("ordinary", "toppquad")


def read(path):
    return json.loads(path.read_bytes())


def link(target, directory):
    return quote(Path(os.path.relpath(target, directory)).as_posix())


def load_flight(directory, trial):
    if trial is None:
        return dict(status="未运行", samples=[], passed=False, landed=False), None, None

    run = Path(trial["run"])
    summary = read(run / "summary.json")
    manifest = read(run / "live_manifest.json")
    states = read(run / "trace.json")["states"]
    assert len(states) == summary["samples"]
    events = {event["stage"]: event for event in summary["preparation_events"]}
    validation = Path(events["PLANNING_STARTED"]["output"]) / "validation.json" if "PLANNING_STARTED" in events else None
    plan = read(validation) if validation and validation.is_file() else None
    metrics = trial.get("metrics", {})
    timing = summary.get("planning_pipeline_timings") or {}
    planning_s = timing.get("parent_request_to_collect_s")
    if planning_s is None and "PLANNING_REJECTED" in events:
        planning_s = events["PLANNING_REJECTED"]["host_monotonic_s"] - events["PLANNING_STARTED"]["host_monotonic_s"]
    first_timestamp = states[0]["sim_timestamp_ns"] if states else 0
    samples = [
        [(state["sim_timestamp_ns"] - first_timestamp) * 1e-9,
         *state["position_zup_m"], *state["reference_position_zup_m"],
         math.hypot(*state["velocity_zup_mps"]), math.dist(state["position_zup_m"], state["reference_position_zup_m"]),
         state["tilt_deg"]]
        for state in states
    ]
    if trial["passed"]:
        assert plan and plan["validation"]["model_feasible"]
        assert metrics["tracking_time_s"] == samples[-1][0]
    status = ("通过" if trial["passed"] else
              "参考预检拒绝" if "PLANNING_REJECTED" in events else
              "跟踪未通过" if samples else "跟踪前结束")
    links = {name: link(path, directory) for name, path in {
        "状态 CSV": run / "states.csv", "原始遥测": run / "trace.json",
        "飞行回执": run / "summary.json", "规划回执": validation,
    }.items() if path and path.is_file()}
    return dict(status=status, error=trial["error"], passed=trial["passed"], landed=trial["safe_landing"],
                fingerprint=manifest["condition_fingerprint"], samples=samples, links=links,
                tracking_s=metrics.get("tracking_time_s"), planning_s=planning_s,
                reference_s=manifest.get("effective_duration_s"),
                speed_mps=metrics.get("measured_peak_speed_mps"), peak_error_m=metrics.get("max_position_error_m"),
                tilt_deg=metrics.get("peak_tilt_deg"), gap_s=metrics.get("max_sample_gap_s"),
                waypoints=metrics.get("waypoint_results", []), terminal=metrics.get("terminal_hold"),
                failures=[name for name, passed in trial.get("checks", {}).items() if not passed],
                planning_feasible=plan["validation"]["model_feasible"] if plan else None,
                candidate_count=len(plan["retiming_candidates"]) if plan else None,
                feasible_candidates=sum(candidate.get("model_feasible", False)
                                        for candidate in plan["retiming_candidates"]) if plan else None), manifest, plan


def segments(ordinary, adjusted):
    """Read executed-reference clocks from receipts and sample the selected blend."""
    if not (ordinary and adjusted and adjusted.get("retiming_profile")):
        return [], []
    profile = adjusted["retiming_profile"]
    assert profile["mode"] == "segmented"
    ordinary_times = np.array([0., *[row["arrival_time_s"] for row in ordinary["waypoint_times"]]])
    adjusted_times = np.array([0., *[row["arrival_time_s"] for row in adjusted["waypoint_times"]]])
    assert np.allclose(profile["phase_knots_s"], ordinary_times, rtol=0, atol=1e-9)
    assert len(ordinary_times) == len(adjusted_times)
    solution, = adjusted["solver"]
    phase_map = SegmentTimeMap(solution, profile["phase_knots_s"], profile["fractions"])
    legs, curve = [], []
    for i, (old, new) in enumerate(zip(np.diff(ordinary_times), np.diff(adjusted_times))):
        left, right = phase_map.knots[i:i+2]
        phase = np.linspace(left, right, 41)
        alpha = phase_map.blend(phase)
        curve.extend([[i + k / 40, float(a)] for k, a in enumerate(alpha) if not i or k])
        legs.append(dict(segment=i+1, baseline_s=float(old), adjusted_s=float(new),
                         delta_s=float(new-old), delta_pct=float((new/old-1)*100),
                         alpha_start=float(alpha[0]), alpha_end=float(alpha[-1])))
    assert math.isclose(sum(row["delta_s"] for row in legs), adjusted_times[-1] - ordinary_times[-1], abs_tol=1e-9)
    return legs, curve


def build(directory):
    design = read(directory / "design.json")
    assert hashlib.sha256(Path(design["source_design"]).read_bytes()).hexdigest() == design["source_design_sha256"]
    campaign_path = directory / "campaign.json"
    campaign = read(campaign_path) if campaign_path.is_file() else {}
    retries_path = directory / "retries.json"
    retries = read(retries_path) if retries_path.is_file() else []
    retry_reports = {(entry["route"], entry["geometry"]): entry["report"] for entry in retries}
    pairs = []
    for route, coords in design["routes"].items():
        receipt = campaign.get(route, {}).get("report")
        flights = read(Path(receipt) / "audit.json")["flights"] if receipt else []
        by_method = {flight["method"]: flight for flight in flights}
        assert len(by_method) == len(flights)
        for geometry, geometry_name in GEOMETRY_NAMES.items():
            trial_data = [load_flight(directory, by_method.get(f"{geometry}_{arm}")) for arm in ARMS]
            (ordinary, ordinary_manifest, ordinary_plan), (auto, auto_manifest, auto_plan) = trial_data
            for manifest, plan in ((ordinary_manifest, ordinary_plan), (auto_manifest, auto_plan)):
                if not manifest:
                    continue
                actual = manifest["plan"]["route"]
                assert actual["waypoints"] == coords["waypoints"] and actual["start"] == coords["start"]
                if plan:
                    assert plan["representation"] == geometry
                    assert plan["route"]["small_field"] == actual["small_field"]
            geometry_match = (ordinary_plan["timed_segments"] == auto_plan["base_timed_segments"]
                              if ordinary_plan and auto_plan else None)
            if geometry_match is False:
                raise ValueError(f"不同的基础路径 {route}/{geometry}")
            both_tracked = bool(ordinary["samples"] and auto["samples"])
            conditions_match = ordinary["fingerprint"] == auto["fingerprint"] if both_tracked else None
            legs, profile = segments(ordinary_plan, auto_plan)
            delta = (auto["tracking_s"] - ordinary["tracking_s"]
                     if conditions_match and ordinary["passed"] and auto["passed"] else None)
            pairs.append(dict(route=route, route_name=ROUTE_NAMES[route], geometry=geometry, geometry_name=geometry_name,
                              points=[coords["start"], *coords["waypoints"]], ordinary=ordinary, toppquad=auto,
                              same_geometry=geometry_match, same_tracking_conditions=conditions_match,
                              tracking_delta_s=delta, tracking_delta_pct=100*delta/ordinary["tracking_s"]
                              if delta is not None else None, segments=legs, alpha=profile))
            if report := retry_reports.get((route, geometry)):
                retest_trial, = read(Path(report) / "audit.json")["flights"]
                assert retest_trial["method"] == f"{geometry}_toppquad"
                retest, retest_manifest, retest_plan = load_flight(directory, retest_trial)
                retest_conditions = (retest["fingerprint"] == ordinary["fingerprint"]
                                     and retest_manifest["source_sha256"] == ordinary_manifest["source_sha256"])
                retest_geometry = bool(retest_plan and ordinary_plan["timed_segments"] == retest_plan["base_timed_segments"])
                retry_legs, retry_alpha = segments(ordinary_plan, retest_plan)
                retry_delta = (retest["tracking_s"] - ordinary["tracking_s"]
                               if retest_conditions and retest_geometry and ordinary["passed"] and retest["passed"] else None)
                pairs[-1]["retry"] = dict(toppquad=retest, same_tracking_conditions=retest_conditions,
                                          same_geometry=retest_geometry, segments=retry_legs, alpha=retry_alpha,
                                          tracking_delta_s=retry_delta,
                                          tracking_delta_pct=100*retry_delta/ordinary["tracking_s"]
                                          if retry_delta is not None else None)
    trials = [pair[arm] for pair in pairs for arm in ARMS]
    stats = dict(requested=len(design["routes"]) * len(design["jobs"]),
                 recorded=sum(trial["status"] != "未运行" for trial in trials),
                 tracked=sum(bool(trial["samples"]) for trial in trials),
                 passed=sum(trial["passed"] for trial in trials), landed=sum(trial["landed"] for trial in trials),
                 matched=sum(pair["tracking_delta_s"] is not None for pair in pairs),
                 retries=len(retries),
                 retries_passed=sum(pair["retry"]["toppquad"]["passed"] for pair in pairs if "retry" in pair))
    return dict(pairs=pairs, stats=stats, batch=directory.name)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def render(directory):
    data = build(directory)
    pairs = data["pairs"]
    flight_fields = ["route", "geometry", "arm", "attempt", "status", "passed", "landed", "reference_s", "tracking_s",
                     "planning_s", "speed_mps", "peak_error_m", "tilt_deg", "gap_s", "error", "fingerprint"]
    write_csv(directory / "flights.csv", flight_fields,
              [dict(route=p["route"], geometry=p["geometry"], arm=arm, attempt="initial", **p[arm])
               for p in pairs for arm in ARMS] +
              [dict(route=p["route"], geometry=p["geometry"], arm="toppquad", attempt="retry", **p["retry"]["toppquad"])
               for p in pairs if "retry" in p])
    segment_fields = ["route", "geometry", "attempt", "status", "segment", "baseline_s", "adjusted_s", "delta_s",
                      "delta_pct", "alpha_start", "alpha_end"]
    write_csv(directory / "segments.csv", segment_fields,
              [dict(route=p["route"], geometry=p["geometry"], attempt=attempt, status=variant["toppquad"]["status"], **leg)
               for p in pairs for attempt, variant in (("initial", p), ("retry", p.get("retry")))
               if variant for leg in variant["segments"]])
    audit = dict(data["stats"], pairs=[{key: pair[key] for key in (
        "route", "geometry", "same_geometry", "same_tracking_conditions", "tracking_delta_s", "tracking_delta_pct"
    )} for pair in pairs])
    audit["retry_pairs"] = [
        dict(route=p["route"], geometry=p["geometry"], passed=p["retry"]["toppquad"]["passed"],
             landed=p["retry"]["toppquad"]["landed"], **{key: p["retry"][key] for key in (
                 "same_geometry", "same_tracking_conditions", "tracking_delta_s", "tracking_delta_pct")})
        for p in pairs if "retry" in p
    ]
    (directory / "pair_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    web = Path(__file__).with_name("small_field_retiming_web")
    bundle = ROOT / ".runtime/route-replay-3d.js"
    template = (web / "index.html").read_text(encoding="utf-8")
    script = (web / "replay.js").read_text(encoding="utf-8")
    html = template.replace("/*REPLAY*/", script).replace("/*SCENE3D*/",
        bundle.read_text(encoding="utf-8") if bundle.is_file() else "")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    (directory / "index.html").write_text(html.replace("/*DATA*/", payload), encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(render(parser.parse_args().directory.resolve()), ensure_ascii=False))
