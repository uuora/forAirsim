"""Independent arithmetic replay of frozen E0 traces; no rollout or optimizer.

PASS means consistent frozen records and the prescribed model equations. It
does not prove timing chronology, absence of unlogged calls or flight accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRIC_NAMES = (
    "position_rmse_m", "max_position_error_m", "peak_position_error_time_s",
    "peak_position_error_u", "terminal_position_error_m", "terminal_velocity_error_mps",
    "speed_clipped_time_fraction", "command_slew_clipped_time_fraction",
    "max_velocity_command_mps", "max_command_slew_mps2",
)
VECTOR_FIELDS = (
    ("reference_position_m", "q_m"), ("reference_velocity_mps", "qdot_mps"),
    ("actual_position_m", "p_m"), ("actual_velocity_mps", "v_mps"),
    ("raw_velocity_command_mps", "raw_cmd_mps"),
    ("speed_limited_velocity_command_mps", "speed_cmd_mps"),
    ("velocity_command_mps", "cmd_mps"),
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11,
                               equal_nan=False,err_msg=label)


def expected_coefficients(parameters,boundary):
    d, duration = np.array(parameters[:3]),parameters[-1]
    p0,v0,a0,p1,v1,a1 = (np.array(boundary[x]) for x in ("p0","v0","a0","p1","v1","a1"))
    c = np.zeros((7,3))
    c[0],c[1],c[2] = p0,duration*v0,.5*duration**2*a0
    r0 = p1-c[0]-c[1]-c[2]
    r1 = duration*v1-c[1]-2*c[2]
    r2 = duration**2*a1-2*c[2]
    c[3],c[4],c[5] = 10*r0-4*r1+.5*r2,-15*r0+7*r1-r2,6*r0-3*r1+.5*r2
    c[3:7] += np.array([64.,-192.,192.,-64.])[:,None]*d
    return c


def polynomial(coefficients,times,duration,derivative=0):
    c = np.array(coefficients,copy=True)
    for _ in range(derivative):
        c = c[1:]*np.arange(1,len(c))[:,None]/duration
    u = np.array(times)/duration
    # A direct sum is deliberately different from the producer's Horner evaluator.
    return sum(c[j]*u[...,None]**j for j in range(len(c)))


def verify_config(config):
    assert config["experiment"]=="control_simulation_mvp_e0" and config["schema_version"]==1
    assert config["family"]=="single_bump" and config["representation"]=="hermite"
    boundary = dict(p0=[0.,0.,5.],v0=[1.,0.,0.],a0=[0.,0.,0.],
                    p1=[8.,0.,5.],v1=[1.,0.,0.],a1=[0.,0.,0.])
    assert config["boundary"]==boundary
    expected_refs = []
    for shape,d in (("straight",[0.,0.,0.]),("lateral",[0.,1.15,0.]),("vertical",[0.,0.,.65])):
        for label,t in (("slow",5.5),("fast",3.)):
            expected_refs.append(dict(reference_id=f"{shape}_{label}",shape=shape,parameters=d+[t],duration_label=label))
    assert config["references"]==expected_refs
    assert config["base_tracking_condition"]==dict(velocity_lag_s=.3,timestep_s=.02,position_gain=2.,
        max_command_speed_mps=4.,max_command_slew_mps2=4.)
    expected_conditions = [
        dict(condition_id="nominal",initial_position_offset_m=[0.,0.,0.],velocity_lag_s=.3),
        dict(condition_id="initial_y_plus_0p2",initial_position_offset_m=[0.,.2,0.],velocity_lag_s=.3),
        dict(condition_id="initial_y_minus_0p2",initial_position_offset_m=[0.,-.2,0.],velocity_lag_s=.3),
        dict(condition_id="lag_0p45",initial_position_offset_m=[0.,0.,0.],velocity_lag_s=.45)]
    assert config["conditions"]==expected_conditions
    assert config["gates"]==dict(max_position_error_m=.5,terminal_position_error_m=.25,
        terminal_velocity_error_mps=.5,command_numerical_tolerance=1e-10)
    assert config["planned_rollouts"]==24
    expected_order = [f"{r['reference_id']}__{s['condition_id']}" for r in expected_refs for s in expected_conditions]
    assert config["frozen_execution_order"]==expected_order
    assert config["basis_grid_points"]==1001 and config["basis_absolute_tolerance"]==1e-9


def verify_record(record, reference, spec, config):
    trace = record["trace"]
    duration = reference["parameters"][-1]
    condition = dict(config["base_tracking_condition"],velocity_lag_s=spec["velocity_lag_s"])
    assert record["parameters"]==reference["parameters"]
    assert record["reference_id"]==reference["reference_id"] and record["condition_id"]==spec["condition_id"]
    assert record["shape"]==reference["shape"]
    assert record["record_id"]==f"{reference['reference_id']}__{spec['condition_id']}"
    assert trace["condition"]==condition and trace["duration_s"]==duration
    assert trace["initial_position_offset_m"]==spec["initial_position_offset_m"]
    count = math.ceil(duration/condition["timestep_s"])
    t = np.array(trace["times_s"])
    close(t,np.linspace(0.,duration,count+1),"physical time grid")
    assert t[0]==0 and t[-1]==duration and trace["intervals"]==count
    h = np.diff(t)
    assert np.all(h>0)
    close(trace["command_times_s"],t[:-1],"command interval start")
    close(trace["command_intervals_s"],h,"command interval lengths")
    close(trace["actual_max_timestep_s"],max(h),"maximum time step")
    coefficients = expected_coefficients(reference["parameters"],config["boundary"])
    close(record["curve_power_coefficients"],coefficients,"independent endpoint-preserving coefficients")
    q = polynomial(coefficients,t,duration)
    qdot = polynomial(coefficients,t,duration,1)
    close(trace["reference_position_m"],q,"q evaluated at same state times")
    close(trace["reference_velocity_mps"],qdot,"qdot evaluated at same state times")
    p,v = np.array(trace["actual_position_m"]),np.array(trace["actual_velocity_mps"])
    assert p.shape==q.shape and v.shape==q.shape and np.isfinite(p).all() and np.isfinite(v).all()
    close(p[0],q[0]+spec["initial_position_offset_m"],"initial actual position")
    close(v[0],qdot[0],"initial actual velocity")
    close(trace["initial_velocity_command_mps"],qdot[0],"initial previous command")
    previous = qdot[0].copy()
    commands,raws,speed_commands,slews,speed_diffs,slew_diffs,speed_clips,slew_clips = [],[],[],[],[],[],[],[]
    speed_cap,slew_cap = condition["max_command_speed_mps"],condition["max_command_slew_mps2"]
    for i,step in enumerate(h):
        raw = qdot[i]+condition["position_gain"]*(q[i]-p[i])
        raw_norm = math.hypot(*raw)
        limited = raw*min(1.,speed_cap/max(raw_norm,1e-300))
        delta = limited-previous
        delta_norm = math.hypot(*delta)
        cmd = previous+delta*min(1.,slew_cap*step/max(delta_norm,1e-300))
        raws.append(raw); speed_commands.append(limited); commands.append(cmd)
        slews.append(np.linalg.norm(cmd-previous)/step)
        speed_diffs.append(np.linalg.norm(limited-raw)); slew_diffs.append(np.linalg.norm(cmd-limited))
        speed_clips.append(raw_norm>speed_cap); slew_clips.append(delta_norm>slew_cap*step)
        beta = 1.-math.exp(-float(step)/condition["velocity_lag_s"])
        expected_v = v[i]+(cmd-v[i])*beta
        expected_p = p[i]+cmd*step+(v[i]-cmd)*condition["velocity_lag_s"]*beta
        close(v[i+1],expected_v,"independent exact ZOH velocity recursion")
        close(p[i+1],expected_p,"independent exact ZOH position recursion")
        previous = cmd
    for key,expected in (("raw_velocity_command_mps",raws),("speed_limited_velocity_command_mps",speed_commands),
            ("velocity_command_mps",commands),("command_slew_mps2",slews),
            ("speed_projection_difference_mps",speed_diffs),("command_slew_projection_difference_mps",slew_diffs)):
        close(trace[key],expected,key)
    # Projection flags use stored producer references to avoid threshold flips
    # from equivalent polynomial arithmetic at machine precision.
    q_stored,qdot_stored = np.array(trace["reference_position_m"]),np.array(trace["reference_velocity_mps"])
    flag_previous = qdot_stored[0]
    flag_speed,flag_slew = [],[]
    for i,step in enumerate(h):
        raw = qdot_stored[i]+condition["position_gain"]*(q_stored[i]-p[i])
        n = math.hypot(*raw)
        limited = raw*min(1.,speed_cap/max(n,1e-300))
        delta = limited-flag_previous
        dn = math.hypot(*delta)
        flag_speed.append(n>speed_cap); flag_slew.append(dn>slew_cap*step)
        flag_previous = np.array(trace["velocity_command_mps"])[i]
    assert trace["speed_clipped"]==flag_speed and trace["command_slew_clipped"]==flag_slew
    error = np.linalg.norm(q_stored-p,axis=1)
    close(trace["position_error_m"],error,"synchronous error norm")
    peak = int(np.argmax(error))
    command_array = np.array(trace["velocity_command_mps"])
    metrics = dict(position_rmse_m=math.sqrt(np.sum(.5*(error[:-1]**2+error[1:]**2)*h)/duration),
        max_position_error_m=float(error[peak]),peak_position_error_time_s=float(t[peak]),
        peak_position_error_u=float(t[peak]/duration),terminal_position_error_m=float(error[-1]),
        terminal_velocity_error_mps=float(np.linalg.norm(qdot_stored[-1]-v[-1])),
        speed_clipped_time_fraction=float(np.sum(h[np.array(flag_speed)])/duration),
        command_slew_clipped_time_fraction=float(np.sum(h[np.array(flag_slew)])/duration),
        max_velocity_command_mps=float(np.max(np.linalg.norm(command_array,axis=1))),
        max_command_slew_mps2=float(np.max(np.array(trace["command_slew_mps2"]))))
    assert set(trace["metrics"])==set(metrics)
    for name,value in metrics.items():
        close(trace["metrics"][name],value,name)
    gates = {name:metrics[name]<=config["gates"][name] for name in
        ("max_position_error_m","terminal_position_error_m","terminal_velocity_error_mps")}
    tol = config["gates"]["command_numerical_tolerance"]
    gates["command_speed_cap"] = metrics["max_velocity_command_mps"]<=speed_cap+tol
    gates["command_slew_cap"] = metrics["max_command_slew_mps2"]<=slew_cap+tol
    assert record["control_gates"]==gates
    assert record["control_only_pass"]==all(gates.values())
    assert record["failed_gates"]==[k for k,v in gates.items() if not v]
    assert record["scope"]==config["scope"]
    timing = record["timing"]
    assert all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in timing.values())
    assert timing["candidate_evaluation_wall_s"]>=timing["tracker_reported_wall_s"]
    assert timing["candidate_evaluation_wall_s"]>=timing["reference_build_wall_s"]
    close(timing["tracker_reported_wall_s"],trace["evaluation_wall_s"],"tracker timer field")
    return metrics,gates


def verify_basis(checks,config):
    assert len(checks)==6
    for check,ref in zip(checks,config["references"]):
        assert check["reference_id"]==ref["reference_id"] and check["grid_points"]==1001 and check["passed"]
        c = expected_coefficients(ref["parameters"],config["boundary"])
        close(check["hermite_power_coefficients"],c,"Hermite coefficient construction")
        close(check["bezier_power_coefficients"],c,"Bezier coefficient construction")
        controls = np.array([sum(c[j]*(math.comb(k,j)/math.comb(6,j)) for j in range(k+1)) for k in range(7)])
        close(check["bezier_control_points"],controls,"independent power to Bernstein mapping")
        t = np.linspace(0.,ref["parameters"][-1],1001)
        for d in (0,1,2):
            difference = np.max(np.abs(polynomial(check["hermite_power_coefficients"],t,t[-1],d)-
                polynomial(check["bezier_power_coefficients"],t,t[-1],d)))
            assert difference<=config["basis_absolute_tolerance"]
            assert 0<=check[f"derivative_{d}_max_abs_difference"]<=config["basis_absolute_tolerance"]


def verify_samples(path,record):
    trace = record["trace"]
    with path.open(encoding="utf-8",newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows)==len(trace["times_s"])
    for i,row in enumerate(rows):
        assert int(row["state_index"])==i
        close(float(row["time_s"]),trace["times_s"][i],"CSV state time")
        close(float(row["position_error_m"]),trace["position_error_m"][i],"CSV position error")
        for name,prefix in VECTOR_FIELDS:
            for axis,j in zip("xyz",range(3)):
                cell = row[f"{prefix}_{axis}"]
                if i<len(trace[name]):
                    close(float(cell),trace[name][i][j],f"CSV {name}")
                else:
                    assert cell=="", "terminal row invents a command"
        for name in ("command_slew_mps2","speed_projection_difference_mps",
                     "command_slew_projection_difference_mps","speed_clipped","command_slew_clipped"):
            if i==trace["intervals"]:
                assert row[name]==""
            elif name.endswith("clipped"):
                assert row[name]==str(trace[name][i])
            else:
                close(float(row[name]),trace[name][i],f"CSV {name}")
        if i==trace["intervals"]:
            assert row["command_interval_s"]==""
        else:
            close(float(row["command_interval_s"]),trace["command_intervals_s"][i],"CSV command interval")


def audit_directory(output,*,write=True):
    output = Path(output).resolve()
    config,manifest,completion = (load(output/name) for name in ("config.json","manifest.json","completion.json"))
    verify_config(config)
    assert completion["status"]=="COMPLETE" and completion["rollouts"]==24
    assert completion["producer_attests_source_bytes_unchanged"] is True
    assert manifest["planned_rollouts"]==24 and manifest["scope"]==config["scope"]
    assert sha256(output/"config.json")==manifest["config_sha256"]
    for name,digest in completion["artifact_sha256"].items():
        path = (output/name).resolve()
        assert output in path.parents and path.is_file(), f"invalid artifact path {name}"
        assert sha256(path)==digest, f"artifact changed: {name}"
    for name,digest in manifest["source_sha256"].items():
        assert sha256(output/"source_snapshot"/name)==digest,f"source snapshot changed: {name}"
    required = {"config.json","manifest.json","basis_checks.json","rows.json","summary.json","summary.csv"}
    rows = load(output/"rows.json")
    assert [r["record_id"] for r in rows]==config["frozen_execution_order"]
    with (output/"summary.csv").open(encoding="utf-8",newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(rows)==len(csv_rows)==24
    records = []
    for ref in config["references"]:
        for spec in config["conditions"]:
            index = len(records)
            record_id = f"{ref['reference_id']}__{spec['condition_id']}"
            json_name,csv_name = f"traces/{record_id}.json",f"samples/{record_id}.csv"
            required.update((json_name,csv_name))
            record = load(output/json_name)
            metrics,gates = verify_record(record,ref,spec,config)
            verify_samples(output/csv_name,record)
            row = rows[index]
            assert row["record_id"]==record_id and row["reference_id"]==ref["reference_id"]
            assert row["shape"]==ref["shape"] and row["duration_s"]==ref["parameters"][-1]
            assert row["condition_id"]==spec["condition_id"] and row["control_only_pass"]==all(gates.values())
            assert row["trace_json"]==json_name and row["samples_csv"]==csv_name
            assert row["failed_gates"]==";".join(record["failed_gates"])
            for name in METRIC_NAMES:
                close(row[name],metrics[name],f"summary {name}")
            for name,value in record["timing"].items():
                close(row[name],value,f"summary timing {name}")
            assert set(row)==set(csv_rows[index])
            for name,value in row.items():
                assert csv_rows[index][name]==str(value),f"summary CSV differs: {name}"
            records.append(record)
    verify_basis(load(output/"basis_checks.json"),config)
    summary = load(output/"summary.json")
    assert summary["rollouts"]==24 and summary["prescribed_references"]==6 and summary["conditions_per_reference"]==4
    passes = sum(r["control_only_pass"] for r in records)
    assert summary["control_only_passes"]==passes and summary["control_only_failures"]==24-passes
    assert summary["basis_checks"]==6 and summary["basis_checks_passed"]
    assert all(summary[x] is True for x in ("no_optimizer_calls","no_airsim_calls","no_obstacle_evaluation"))
    assert summary["scope"]==config["scope"] and summary["timing_scope"]==config["timing_scope"]
    expected_failures = {key:sum(not r["control_gates"][key] for r in records) for key in records[0]["control_gates"]}
    assert summary["gate_failure_counts"]==expected_failures
    assert summary["by_reference"]==[dict(reference_id=ref["reference_id"],
        control_only_passes=sum(r["control_only_pass"] for r in records if r["reference_id"]==ref["reference_id"]),
        conditions=4) for ref in config["references"]]
    if manifest["plots_requested"]:
        required.update(f"figures/{name}.{ext}" for name in ("trajectories","tracking_errors","commands") for ext in ("png","pdf"))
    required.update(f"source_snapshot/{name}" for name in manifest["source_sha256"])
    assert required.issubset(completion["artifact_sha256"]),"completion omits required evidence"
    current_diff = [name for name,digest in manifest["source_sha256"].items()
                    if not (ROOT/name).is_file() or sha256(ROOT/name)!=digest]
    result = dict(status="PASS",rollouts=24,state_samples=sum(len(r["trace"]["times_s"]) for r in records),
        command_intervals=sum(r["trace"]["intervals"] for r in records),control_only_passes=passes,
        control_only_failures=24-passes,basis_checks=6,source_snapshot_hashes_verified=True,
        artifact_hashes_verified=len(completion["artifact_sha256"]),current_source_differences=current_diff,
        scope="Frozen-record arithmetic replay only; no rollout or optimizer rerun. Does not prove absence "
              "of unlogged calls, timing chronology, real-time ability or physical flight validity.")
    if write:
        (output/"audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_directory(args.output),indent=2))


if __name__=="__main__":
    main()
