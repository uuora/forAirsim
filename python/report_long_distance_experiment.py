"""Export the long-distance campaign and package the evidence for replaying its audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.interpolate import CubicSpline

from dual_target_control.route_curve import decode_route
from flight_comparison import flight_metrics
from segment_retiming import SegmentTimeMap


def read_json(path):
    return json.loads(path.read_bytes())


def write_csv(path, rows):
    fields = dict.fromkeys(key for row in rows for key in row)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def describe_timing(route_name, method, plan):
    """Use the baseline phase as the common coordinate across all three timings."""
    base = decode_route(plan)
    phases = np.r_[0., base.ends]
    profile = plan.get("retiming_profile")
    raw = (plan.get("solver") or [None])[0]
    fixed = plan.get("retiming_fraction")
    if profile is not None:
        weights = np.asarray(profile["fractions"])
        alpha = SegmentTimeMap(raw, phases, weights).blend
    else:
        weights = np.full(len(phases), 0. if fixed is None else fixed)

        def alpha(s):
            return np.full_like(s, weights[0])

    summary = dict(alpha_mean=float(np.average((weights[:-1] + weights[1:]) / 2., weights=np.diff(phases))),
                   alpha_min=float(weights.min()), alpha_max=float(weights.max()),
                   knot_fractions=json.dumps(weights.tolist()), phase_knots_s=json.dumps(phases.tolist()))
    reference_times = np.r_[0., [row["arrival_time_s"] for row in plan.get("waypoint_times", base.waypoint_times)]]
    h_raw = CubicSpline(raw["anchors"], raw["h"], bc_type="natural") if raw is not None else None
    rows = []
    total_raw_area, total_adopted_area = 0., 0.
    for i, (lo, hi) in enumerate(zip(phases[:-1], phases[1:])):
        s = np.linspace(lo, hi, 513)
        piece = base.segments[i]
        path_speed = np.linalg.norm(piece.sample(np.clip(s-lo, 0., piece.duration_s), 1), axis=1)
        distance = float(np.trapz(path_speed, s))
        timing = dict(route=route_name, method=method, leg=i+1, distance_m=distance,
                      alpha_start=float(weights[i]), alpha_end=float(weights[i+1]),
                      alpha_mean=float((weights[i]+weights[i+1])/2.), baseline_reference_s=hi-lo,
                      elapsed_reference_s=reference_times[i+1]-reference_times[i])
        if h_raw is not None:
            raw_change = h_raw(s) - 1.
            raw_area = float(np.trapz(np.abs(raw_change), s))
            used_area = float(np.trapz(alpha(s)*np.abs(raw_change), s))
            total_raw_area += raw_area
            total_adopted_area += used_area
            timing.update(h_raw_start=float(h_raw(lo)), h_raw_end=float(h_raw(hi)),
                          adopted_fraction_of_abs_h_change=used_area/raw_area if raw_area else None,
                          mean_abs_delta_h=used_area/(hi-lo))
        rows.append(timing)
    summary["adopted_fraction_of_abs_h_change"] = total_adopted_area/total_raw_area if total_raw_area else None
    summary["mean_abs_delta_h"] = total_adopted_area/phases[-1] if raw is not None else None
    return summary, rows


def package_campaign(campaign, output):
    design = read_json(campaign/"design.json")
    progress = read_json(campaign/"progress.json")
    output.mkdir(parents=True, exist_ok=False)
    flights, legs, waypoints = [], [], []
    metadata = dict(campaign=campaign.name, route_fingerprints={}, entries={})
    with zipfile.ZipFile(output/"telemetry.zip", mode="x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def add(name, payload):
            archive.writestr(name, payload)
            metadata["entries"][name] = hashlib.sha256(payload).hexdigest()

        add("design.json", (campaign/"design.json").read_bytes())
        add("progress.json", (campaign/"progress.json").read_bytes())
        for route in dict.fromkeys(row["route"] for row in progress):
            selected = [row for row in progress if row["route"] == route]
            portable = dict(expected=len(selected), results=[], remaining=[])
            fingerprints = set()
            for row in selected:
                method = row["method"]
                item = dict(route=route, method=method, state=row["state"], passed=False)
                flights.append(item)
                if row["state"] != "RECORDED":
                    item["error"] = row.get("error")
                    portable["remaining"].append(["pass", "hermite", method])
                    continue
                run = Path(row["run"])
                audit = flight_metrics(run)
                report = read_json(run/"summary.json")
                manifest = read_json(run/"live_manifest.json")
                plan = manifest["plan"]
                preset = design["route_presets"]
                assert plan["route"]["start"] == preset["start"]
                assert plan["route"]["waypoints"] == preset["routes"][route]["waypoints"]
                assert audit["passed"] == row["passed"] and audit["arm"] == method
                fingerprints.add(audit["fingerprint"])
                item.update(run=run.name, batch=row["batch"], outcome=audit["outcome"], passed=audit["passed"],
                            safe_landing=audit["safe_landing"], error=audit.get("error"),
                            failed_checks=json.dumps(audit.get("failed_checks", []), ensure_ascii=False))
                keys = ("samples", "reference_s", "execution_s", "planning_s", "reference_peak_speed_mps",
                        "tracking_peak_speed_mps", "peak_error_m", "rmse_m", "max_command_mps",
                        "max_command_slew_mps2", "max_gap_s", "speed_clipped_fraction", "slew_clipped_fraction")
                item.update({key: audit.get(key) for key in keys})
                item["retiming_fraction"] = plan.get("retiming_fraction")
                item["fingerprint"] = audit["fingerprint"]
                for i, waypoint in enumerate(audit.get("waypoints", []), 1):
                    waypoints.append(dict(route=route, method=method, waypoint=i, **waypoint))
                if manifest.get("effective_duration_s") is not None:
                    summary, leg_rows = describe_timing(route, method, plan)
                    item.update(summary)
                    legs.extend(leg_rows)
                prefix = f"{route}/{run.name}/"
                portable["results"].append(dict(run=run.name, method=report["method"], policy="pass",
                                                samples=report["samples"], passed=audit["passed"], outcome=audit["outcome"]))
                for filename in ("states.csv", "summary.json", "live_manifest.json", "cleanup.json",
                                 "planning_request.json", "landing.json", "takeoff.json"):
                    target = run/filename
                    if target.is_file():
                        add(prefix+filename, target.read_bytes())
                solver = plan.get("solver", [])
                if solver:
                    item["ipopt_s"] = sum(s["solver_s"] for s in solver)
                section = report["config"]["curve_experiment"]
                if section.get("plans_file"):
                    validation_path = Path(section["plans_file"]).with_name("validation.json")
                    raw = validation_path.read_bytes()
                    validation = json.loads(raw)
                    assert hashlib.sha256(raw).hexdigest() == report["planning_evidence"]["source_sha256"]
                    item.update(validation["timings"])
                    item.update(validation["counts"])
                    item["proxy_peak_error_m"] = validation["validation"]["metrics"]["max_position_error_m"]
                    selected_profile = audit.get("segment_selection", {})
                    item["feasible_profiles"] = selected_profile.get("feasible")
                    add(prefix+"validation.json", raw)
                else:
                    rejection = next((event for event in report.get("preparation_events", [])
                                      if event["stage"] == "PLANNING_REJECTED"), None)
                    if rejection:
                        raw = Path(rejection["plan_path"]).with_name("validation.json").read_bytes()
                        add(prefix+"validation.json", raw)
            metadata["route_fingerprints"][route] = sorted(fingerprints)
            add(f"{route}/results.json", json.dumps(portable, ensure_ascii=False, indent=2).encode("utf-8"))
        archive.writestr("provenance.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    write_csv(output/"flights.csv", flights)
    write_csv(output/"leg_timing.csv", legs)
    write_csv(output/"waypoints.csv", waypoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package_campaign(args.campaign.resolve(), args.output.resolve())
