"""Frozen E0 control-interface pilot; no optimizer and no AirSim commands.

The 24 rollouts are six prescribed references crossed with four single-factor
conditions. They check a simplified fixed controller, not aircraft feasibility.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from trajectory_research.parameterizations import BoundaryStates, curve_from_parameters
from trajectory_research.timed_tracking import TrackingCondition, run_timed_tracking


ROOT = Path(__file__).resolve().parents[1]
SCOPE = (
    "E0 prescribed-reference control-interface pilot; deterministic 3D lagged point mass, "
    "fixed timed velocity tracker. No optimization, AirSim, real flight, obstacle or body "
    "collision check, attitude/rotor/thrust model, or aircraft-feasibility claim. "
    "Gates are sampled control-only research criteria. Bézier/Hermite is a basis check, "
    "not two independent experiments."
)
SOURCES = (
    "python/run_control_simulation_mvp.py", "python/audit_control_simulation_mvp.py",
    "python/trajectory_research/__init__.py", "python/trajectory_research/curves.py",
    "python/trajectory_research/parameterizations.py", "python/trajectory_research/timed_tracking.py",
)
METRIC_NAMES = (
    "position_rmse_m", "max_position_error_m", "peak_position_error_time_s",
    "peak_position_error_u", "terminal_position_error_m", "terminal_velocity_error_mps",
    "speed_clipped_time_fraction", "command_slew_clipped_time_fraction",
    "max_velocity_command_mps", "max_command_slew_mps2",
)
SUMMARY_FIELDS = (
    "record_id", "reference_id", "shape", "duration_s", "condition_id", "control_only_pass",
    "failed_gates", *METRIC_NAMES, "reference_build_wall_s", "candidate_evaluation_wall_s",
    "candidate_evaluation_cpu_s", "tracker_reported_wall_s", "trace_json", "samples_csv",
)
VECTOR_FIELDS = (
    ("reference_position_m", "q_m"), ("reference_velocity_mps", "qdot_mps"),
    ("actual_position_m", "p_m"), ("actual_velocity_mps", "v_mps"),
    ("raw_velocity_command_mps", "raw_cmd_mps"),
    ("speed_limited_velocity_command_mps", "speed_cmd_mps"),
    ("velocity_command_mps", "cmd_mps"),
)
SAMPLE_FIELDS = ("state_index", "time_s", *tuple(
    f"{prefix}_{axis}" for _, prefix in VECTOR_FIELDS for axis in "xyz"),
    "position_error_m", "command_interval_s", "command_slew_mps2",
    "speed_projection_difference_mps", "command_slew_projection_difference_mps",
    "speed_clipped", "command_slew_clipped")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def make_config():
    references = []
    for shape, displacement in (("straight", [0., 0., 0.]),
                                ("lateral", [0., 1.15, 0.]),
                                ("vertical", [0., 0., .65])):
        for label, duration in (("slow", 5.5), ("fast", 3.0)):
            references.append(dict(reference_id=f"{shape}_{label}", shape=shape,
                parameters=displacement+[duration], duration_label=label))
    conditions = [
        dict(condition_id="nominal", initial_position_offset_m=[0., 0., 0.], velocity_lag_s=.30),
        dict(condition_id="initial_y_plus_0p2", initial_position_offset_m=[0., .2, 0.], velocity_lag_s=.30),
        dict(condition_id="initial_y_minus_0p2", initial_position_offset_m=[0., -.2, 0.], velocity_lag_s=.30),
        dict(condition_id="lag_0p45", initial_position_offset_m=[0., 0., 0.], velocity_lag_s=.45),
    ]
    order = [f"{r['reference_id']}__{c['condition_id']}" for r in references for c in conditions]
    return dict(schema_version=1, experiment="control_simulation_mvp_e0", scope=SCOPE,
        family="single_bump", representation="hermite",
        boundary={key:list(value) for key,value in asdict(BoundaryStates()).items()},
        base_tracking_condition=asdict(TrackingCondition()), references=references, conditions=conditions,
        gates=dict(max_position_error_m=.5, terminal_position_error_m=.25,
                   terminal_velocity_error_mps=.5, command_numerical_tolerance=1e-10),
        initial_state_rule="p=q(0)+declared offset; v=qdot(0); previous command=qdot(0)",
        termination="Record state at physical T, no terminal hold or braking stage in this offline model",
        coordinates="metres, world xyz with z positive up", basis_grid_points=1001,
        basis_absolute_tolerance=1e-9, planned_rollouts=24, frozen_execution_order=order,
        timing_scope="Per-candidate curve construction, rollout and sampled gates; includes tracker list "
            "construction, excludes file JSON/CSV serialization, basis preflight, plotting and imports. "
            "Descriptive pilot timings, not optimizer planning-speed or real-time evidence.",
        validation_design="Single-factor nominal/position +/-0.2m/lag 0.45s; no full-factorial combinations")


def control_gates(trace, gates):
    m, c = trace["metrics"], trace["condition"]
    tol = gates["command_numerical_tolerance"]
    checks = {k: m[k] <= gates[k] for k in (
        "max_position_error_m", "terminal_position_error_m", "terminal_velocity_error_mps")}
    checks["command_speed_cap"] = m["max_velocity_command_mps"] <= c["max_command_speed_mps"]+tol
    checks["command_slew_cap"] = m["max_command_slew_mps2"] <= c["max_command_slew_mps2"]+tol
    return checks


def basis_check(reference, config):
    h = curve_from_parameters(reference["parameters"], family=config["family"], representation="hermite")
    b = curve_from_parameters(reference["parameters"], family=config["family"], representation="bezier")
    times = np.linspace(0., h.duration_s, config["basis_grid_points"])
    differences = {f"derivative_{d}_max_abs_difference": float(np.max(np.abs(h.sample(times, d)-b.sample(times, d))))
                   for d in (0, 1, 2)}
    passed = all(x <= config["basis_absolute_tolerance"] for x in differences.values())
    if not passed:
        raise AssertionError(f"basis equivalence check failed: {reference['reference_id']}")
    return dict(reference_id=reference["reference_id"], grid_points=len(times), passed=passed,
        hermite_power_coefficients=h.coefficients.tolist(), bezier_power_coefficients=b.coefficients.tolist(),
        bezier_control_points=b.bezier_control_points().tolist(), **differences)


def evaluate_reference(reference, condition_spec, config):
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    segment = curve_from_parameters(reference["parameters"], family=config["family"],
                                    representation=config["representation"])
    build_wall = time.perf_counter()-start_wall
    condition = replace(TrackingCondition(**config["base_tracking_condition"]),
                        velocity_lag_s=condition_spec["velocity_lag_s"])
    trace = run_timed_tracking(segment, condition,
        initial_position_offset_m=condition_spec["initial_position_offset_m"])
    gates = control_gates(trace, config["gates"])
    cpu, wall = time.process_time()-start_cpu, time.perf_counter()-start_wall
    return dict(record_id=f"{reference['reference_id']}__{condition_spec['condition_id']}",
        reference_id=reference["reference_id"], shape=reference["shape"],
        condition_id=condition_spec["condition_id"], parameters=reference["parameters"],
        curve_power_coefficients=segment.coefficients.tolist(), control_gates=gates,
        control_only_pass=all(gates.values()), failed_gates=[k for k,v in gates.items() if not v],
        timing=dict(reference_build_wall_s=build_wall, candidate_evaluation_wall_s=wall,
                    candidate_evaluation_cpu_s=cpu, tracker_reported_wall_s=trace["evaluation_wall_s"]),
        scope=SCOPE, trace=trace)


def sample_rows(trace):
    for i, t in enumerate(trace["times_s"]):
        row = dict(state_index=i, time_s=t, position_error_m=trace["position_error_m"][i])
        has_command = i < trace["intervals"]
        for name, prefix in VECTOR_FIELDS:
            values = trace[name][i] if i < len(trace[name]) else [None]*3
            row.update({f"{prefix}_{axis}": value for axis, value in zip("xyz", values)})
        for name in ("command_slew_mps2", "speed_projection_difference_mps",
                     "command_slew_projection_difference_mps", "speed_clipped", "command_slew_clipped"):
            row[name] = trace[name][i] if has_command else None
        row["command_interval_s"] = trace["command_intervals_s"][i] if has_command else None
        yield row


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary_row(record):
    name = record["record_id"]
    return dict(record_id=name, reference_id=record["reference_id"], shape=record["shape"],
        duration_s=record["trace"]["duration_s"], condition_id=record["condition_id"],
        control_only_pass=record["control_only_pass"], failed_gates=";".join(record["failed_gates"]),
        **record["trace"]["metrics"], **record["timing"], trace_json=f"traces/{name}.json",
        samples_csv=f"samples/{name}.csv")


def plot_records(output, config, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9, "axes.titlesize":10, "pdf.fonttype":42, "ps.fonttype":42})
    figure_dir = output/"figures"
    figure_dir.mkdir()
    grouped = {r["reference_id"]: [x for x in records if x["reference_id"] == r["reference_id"]]
               for r in config["references"]}
    colors = ("#1570A6", "#C56A12", "#458B45", "#A14678")
    labels = ("Nominal", "Initial y +0.2 m", "Initial y -0.2 m", "Lag 0.45 s")
    def finish(fig, name, legend_handles=None, legend_labels=None):
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="lower center", ncol=5, frameon=False)
        fig.savefig(figure_dir/f"{name}.png", dpi=180)
        fig.savefig(figure_dir/f"{name}.pdf")
        plt.close(fig)
    fig = plt.figure(figsize=(12,12), layout="constrained")
    handles = None
    for index, ref in enumerate(config["references"]):
        ax = fig.add_subplot(3,2,index+1,projection="3d")
        group = grouped[ref["reference_id"]]
        q = np.array(group[0]["trace"]["reference_position_m"])
        first, = ax.plot(*q.T, color="#222222", ls="--", lw=1.5, label="Reference")
        lines = [first]
        for record, color, label in zip(group, colors, labels):
            p = np.array(record["trace"]["actual_position_m"])
            line, = ax.plot(*p.T, color=color, lw=1.1, label=label)
            lines.append(line)
        ax.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
               title=f"{ref['reference_id']} | T={ref['parameters'][-1]:g} s")
        ax.view_init(elev=24,azim=-65)
        handles = lines
    fig.get_layout_engine().set(rect=(0,.035,1,.965))
    fig.suptitle("E0 point-mass pilot: reference and actual paths (no collision evaluation)")
    finish(fig,"trajectories",handles,["Reference",*labels])

    fig, axes = plt.subplots(3,2,figsize=(12,10),layout="constrained")
    handles = None
    for ax, ref in zip(axes.flat,config["references"]):
        lines = []
        for record,color,label in zip(grouped[ref["reference_id"]],colors,labels):
            trace = record["trace"]
            line, = ax.plot(trace["times_s"],trace["position_error_m"],color=color,lw=1.25,label=label)
            lines.append(line)
        cap = ax.axhline(config["gates"]["max_position_error_m"],color="#333333",ls="--",lw=.9)
        ax.set(title=ref["reference_id"],xlabel="Reference time (s)",ylabel="Synchronous position error (m)")
        ax.grid(alpha=.2)
        handles = [*lines,cap]
    fig.get_layout_engine().set(rect=(0,.045,1,.955))
    fig.suptitle("E0 point-mass pilot: sampled synchronous error")
    finish(fig,"tracking_errors",handles,[*labels,"Peak-error gate"])

    fig, axes = plt.subplots(6,2,figsize=(12,16),layout="constrained")
    handles = None
    for row,ref in enumerate(config["references"]):
        lines = []
        for record,color,label in zip(grouped[ref["reference_id"]],colors,labels):
            trace = record["trace"]
            cmd_norm = np.linalg.norm(trace["velocity_command_mps"],axis=1)
            line, = axes[row,0].plot(trace["command_times_s"],cmd_norm,color=color,lw=1.1,label=label)
            axes[row,1].plot(trace["command_times_s"],trace["command_slew_mps2"],color=color,lw=1.1)
            for col,values,key in ((0,cmd_norm,"speed_clipped"),(1,trace["command_slew_mps2"],"command_slew_clipped")):
                mask = np.array(trace[key],dtype=bool)
                axes[row,col].scatter(np.array(trace["command_times_s"])[mask],np.array(values)[mask],
                                      color=color,s=3,alpha=.45)
            lines.append(line)
        cap = axes[row,0].axhline(config["base_tracking_condition"]["max_command_speed_mps"],color="k",ls="--",lw=.8)
        axes[row,1].axhline(config["base_tracking_condition"]["max_command_slew_mps2"],color="k",ls="--",lw=.8)
        axes[row,0].set(title=ref["reference_id"],ylabel="Command speed (m/s)")
        axes[row,1].set(title=ref["reference_id"],ylabel="Command slew (m/s²)")
        for ax in axes[row]:
            ax.set(xlabel="Command time (s)",ylim=(0,4.25))
            ax.grid(alpha=.2)
        handles = [*lines,cap]
    fig.get_layout_engine().set(rect=(0,.035,1,.965))
    fig.suptitle("E0 point-mass pilot: limited commands; dots indicate clipping (not motor saturation)")
    finish(fig,"commands",handles,[*labels,"Command cap"])


def run(output, *, make_plots=True):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = make_config()
    write_json(output/"config.json",config)
    (output/"traces").mkdir()
    (output/"samples").mkdir()
    sources = {}
    for name in SOURCES:
        source = ROOT/name
        target = output/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        sources[name] = sha256(source)
    def git(*args):
        result = subprocess.run(["git",*args],cwd=ROOT,capture_output=True,text=True,
                                encoding="utf-8",check=True)
        return result.stdout.strip()
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        repo_head=git("rev-parse","HEAD"), working_tree_status=git("status","--short"),
        python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
        source_sha256=sources, config_sha256=sha256(output/"config.json"),
        planned_rollouts=config["planned_rollouts"], plots_requested=make_plots, scope=SCOPE,
        plotting="Post-rollout PNG and PDF; plotting/file serialization excluded from candidate timers")
    write_json(output/"manifest.json",manifest)
    basis = [basis_check(ref,config) for ref in config["references"]]
    write_json(output/"basis_checks.json",basis)
    rows, records = [], []
    for reference in config["references"]:
        for condition in config["conditions"]:
            record = evaluate_reference(reference,condition,config)
            records.append(record)
            row = summary_row(record)
            rows.append(row)
            write_json(output/row["trace_json"],record)
            write_csv(output/row["samples_csv"],SAMPLE_FIELDS,sample_rows(record["trace"]))
            print(json.dumps(dict(record_id=record["record_id"],control_only_pass=record["control_only_pass"],
                rmse_m=record["trace"]["metrics"]["position_rmse_m"],failed_gates=record["failed_gates"])),flush=True)
    write_json(output/"rows.json",rows)
    write_csv(output/"summary.csv",SUMMARY_FIELDS,rows)
    summary = dict(rollouts=len(records), prescribed_references=len(config["references"]),
        conditions_per_reference=len(config["conditions"]),control_only_passes=sum(r["control_only_pass"] for r in records),
        control_only_failures=sum(not r["control_only_pass"] for r in records),
        no_optimizer_calls=True, no_airsim_calls=True, no_obstacle_evaluation=True,
        basis_checks=len(basis),basis_checks_passed=all(x["passed"] for x in basis),
        gate_failure_counts={key:sum(not r["control_gates"][key] for r in records) for key in records[0]["control_gates"]},
        scope=SCOPE,timing_scope=config["timing_scope"],
        by_reference=[dict(reference_id=ref["reference_id"],
            control_only_passes=sum(r["control_only_pass"] for r in records if r["reference_id"]==ref["reference_id"]),
            conditions=4) for ref in config["references"]])
    write_json(output/"summary.json",summary)
    if make_plots:
        plot_records(output,config,records)
    changed = [name for name,digest in sources.items() if sha256(ROOT/name)!=digest]
    if changed:
        raise RuntimeError(f"covered source changed during campaign: {changed}")
    artifacts = {p.relative_to(output).as_posix():sha256(p)
                 for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"completion.json",dict(status="COMPLETE",rollouts=len(records),
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        producer_attests_source_bytes_unchanged=True,artifact_sha256=artifacts))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="New output directory; existing directory is refused")
    parser.add_argument("--no-plots",action="store_true",help="Skip postprocessing figures, keep all raw traces")
    args = parser.parse_args()
    print(json.dumps(run(args.output,make_plots=not args.no_plots),indent=2))


if __name__ == "__main__":
    main()
