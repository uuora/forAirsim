"""Six-group curve/ABOM interface pilot under a fixed timed point-mass model.

Bezier/Hermite x prescribed anchor/ABOM/ABOM+paired parameter perturbation.
This single-seed pilot is not an efficacy study, an original-paper replication,
an AirSim run, or evidence that either curve basis is better.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
from itertools import product
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from trajectory_research.parameterizations import (
    BoundaryStates, curve_from_parameters, default_parameters, parameter_bounds,
)
from trajectory_research.perturbation_task import get_layout
from trajectory_research.screened_search import run_screened_search
from trajectory_research.search import UPSTREAM_COMMIT, UPSTREAM_NETWORK_BLOB
from trajectory_research.timed_control_task import evaluate_timed_parameters
from trajectory_research.timed_tracking import TrackingCondition


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT/"tmp/abom_deploy_probe/upstream"
SCOPE = (
    "Six-group single-seed development interface pilot in a deterministic lagged point-mass model. "
    "No AirSim or real flight; no efficacy/novelty/real-time or curve-basis superiority claim. "
    "The improved candidate is the existing generic paired parameter-perturbation operator. "
    "Pinned upstream network plus source-style adapter, not full original ABOM-paper replication. "
    "Reference geometry has its declared uncertainty envelope; actual point-path checks concern "
    "nominal static geometry only. No body, motor, attitude, wind or sensing model."
)
SOURCE_NAMES = (
    "python/run_curve_abom_six_group_pilot.py", "python/trajectory_research/__init__.py",
    "python/trajectory_research/screened_search.py", "python/trajectory_research/search.py",
    "python/trajectory_research/parameterizations.py", "python/trajectory_research/timed_control_task.py",
    "python/trajectory_research/timed_tracking.py", "python/trajectory_research/perturbation_task.py",
    "python/trajectory_research/quality_bound.py", "python/trajectory_research/curve_task.py",
    "python/trajectory_research/constraints.py", "python/trajectory_research/directional.py",
    "python/trajectory_research/curves.py",
)
SUMMARY_FIELDS = (
    "group_id", "representation", "method", "seed", "search_proposals", "search_objective_calls",
    "selected_validation_calls", "total_objective_calls", "selected_feasible", "selected_score",
    "duration_s", "tracking_rmse_m", "peak_position_error_m", "terminal_position_error_m",
    "terminal_velocity_error_mps", "jerk_energy_m2_s5", "speed_clipped_time_fraction",
    "command_slew_clipped_time_fraction", "actual_nominal_clearance_lower_bound_m",
    "search_wall_s", "selected_validation_wall_s", "search_plus_selected_validation_wall_s",
    "search_cpu_s", "selected_validation_cpu_s", "search_plus_selected_validation_cpu_s",
    "neural_generation_s", "neural_update_s", "search_objective_evaluation_s", "record_json",
)


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config():
    lower,upper = parameter_bounds("single_bump")
    groups = []
    # Paired representations are adjacent; fixed ordering is disclosed, not a
    # counterbalanced wall-time performance experiment.
    for label,method in (("ordinary",None),("abom","abom"),
                         ("abom_improved_candidate","abom_perturb")):
        for representation in ("bezier","hermite"):
            groups.append(dict(group_id=f"{representation}_{label}",representation=representation,
                               method_label=label,search_method=method))
    return dict(schema_version=1,experiment="curve_abom_six_group_pilot",scope=SCOPE,
        family="single_bump",layout=asdict(get_layout("central")),development_layout=True,
        boundary={k:list(v) for k,v in asdict(BoundaryStates()).items()},
        lower_bounds=lower.tolist(),upper_bounds=upper.tolist(),normalized_domain=[-1.,1.],
        anchor_parameters=default_parameters("single_bump").tolist(),seed=1101,
        tracking_condition=asdict(TrackingCondition()),population=20,proposal_budget=80,
        dropout=.95,initialization="uniform",learning_rate=1e-3,weight_decay=1e-5,
        learning_rate_source="AdamW hardcoded in screened_search.py; no caller override",
        injection_fraction=.5,screening_enabled=False,cache_enabled=False,
        groups=groups,frozen_execution_order=[g["group_id"] for g in groups],
        expected_search_objective_calls=320,expected_selected_validation_calls=6,
        expected_total_objective_calls=326,
        update_rule="Per generation: reset AdamW; SmoothL1 repeated best-elite teacher, upstream network architecture",
        selection_rule="ordinary=anchor; optimized=stable retained incumbent after all 80 proposals; no posthoc replacement",
        parameter_space_relation="same physical displacement/duration box in both bases; degree-six "
            "quintic-Hermite-plus-C2-bump equals the degree-six Bernstein construction; "
            "Bezier middle control P3 is shifted by 3.2*d, endpoints retain physical p/v/a",
        timing_scope="Search timer includes the adapter call, model initialization/lazy imports and all search "
            "evaluations. Selected complete revalidation is timed separately and added. Module imports, "
            "curve-only basis checks, file serialization and postprocessing are outside these timers. "
            "No warmup objective calls. Fixed order and single seed preclude runtime superiority claims.",
        airsim_status="NOT_RUN; exported plans only")


def basis_preflight(config):
    bounds = list(zip(config["lower_bounds"],config["upper_bounds"]))
    probes = [list(x) for x in product(*bounds)]+[config["anchor_parameters"]]
    rows = []
    for index,parameters in enumerate(probes):
        h=curve_from_parameters(parameters,family=config["family"],representation="hermite")
        b=curve_from_parameters(parameters,family=config["family"],representation="bezier")
        grid=np.linspace(0.,parameters[-1],1001)
        errors=[float(np.max(np.abs(h.sample(grid,d)-b.sample(grid,d)))) for d in (0,1,2)]
        if max(errors)>1e-9:
            raise AssertionError(f"basis construction differs outside tolerance at probe {index}")
        rows.append(dict(probe_index=index,parameters=parameters,position_max_abs_difference_m=errors[0],
            velocity_max_abs_difference_mps=errors[1],acceleration_max_abs_difference_mps2=errors[2]))
    return dict(status="PASS",probes=len(rows),time_samples_per_probe=1001,
        absolute_tolerance=1e-9,rows=rows,model_objective_calls=0,
        interpretation="Algebraically equivalent canonical curve spaces; finite numerical probes of two "
            "implementations, not an exhaustive floating-point equivalence proof or independent flight samples.")


def assert_search_record(search,calls,config,method):
    assert search["method"]==method and search["seed"]==config["seed"]
    assert search["population"]==config["population"] and search["proposal_budget"]==config["proposal_budget"]
    assert search["initialization"]==config["initialization"] and search["dropout"]==config["dropout"]
    assert search["injection_fraction"]==config["injection_fraction"]
    assert search["screening_enabled"] is False
    assert search["pruned_count"]==search["partial_evaluations"]==search["lower_bound_calls"]==0
    assert len(search["evaluations"])==search["proposals"]==search["full_evaluations"]==len(calls)==config["proposal_budget"]
    for index,(evaluation,call) in enumerate(zip(search["evaluations"],calls),1):
        assert evaluation["status"]=="EVALUATED" and evaluation["proposal_index"]==evaluation["fe"]==index
        assert evaluation["parameters"]==call["parameters"]
        assert evaluation["score"]==call["score"] and evaluation["feasible"]==call["feasible"]
    scores=[e["score"] for e in search["evaluations"]]
    assert search["best_score"]==min(scores)


def run_group(group,config,upstream,*,search_runner=run_screened_search):
    search_calls=[]
    phase_start=time.perf_counter()
    def objective(parameters):
        begin=time.perf_counter()
        result=evaluate_timed_parameters(parameters,family=config["family"],
            representation=group["representation"],layout=config["layout"]["name"],
            condition=TrackingCondition(**config["tracking_condition"]),include_trace=False)
        end=time.perf_counter()
        search_calls.append(dict(call_index=len(search_calls)+1,parameters=list(map(float,parameters)),
            score=result["score"],feasible=result["feasible"],start_relative_to_search_s=begin-phase_start,
            finish_relative_to_search_s=end-phase_start,callback_wall_s=end-begin))
        return result
    search=None
    search_wall=search_cpu=0.
    if group["search_method"] is None:
        selected_parameters=config["anchor_parameters"][:]
        selection=dict(kind="prescribed_anchor",method=None,proposal_index=None,score_before_revalidation=None)
    else:
        phase_start=time.perf_counter()
        cpu_start=time.process_time()
        search=search_runner(objective,config["lower_bounds"],config["upper_bounds"],
            method=group["search_method"],seed=config["seed"],population=config["population"],
            budget=config["proposal_budget"],anchor=config["anchor_parameters"],upstream=upstream,
            initialization=config["initialization"],injection_fraction=config["injection_fraction"],
            dropout=config["dropout"],lower_bound=None,bounded_objective=None)
        search_cpu=time.process_time()-cpu_start
        search_wall=time.perf_counter()-phase_start
        assert_search_record(search,search_calls,config,group["search_method"])
        selected_parameters=search["best_parameters"]
        matches=[e for e in search["evaluations"] if e["score"]==search["best_score"]
            and e["parameters"]==selected_parameters]
        assert matches,"selected incumbent absent from evaluation ledger"
        selection=dict(kind="stable_retained_incumbent",method=group["search_method"],
            proposal_index=matches[0]["proposal_index"],score_before_revalidation=search["best_score"])
    validation_start=time.perf_counter()
    validation_cpu_start=time.process_time()
    selected=evaluate_timed_parameters(selected_parameters,family=config["family"],
        representation=group["representation"],layout=config["layout"]["name"],
        condition=TrackingCondition(**config["tracking_condition"]),include_trace=True)
    validation_cpu=time.process_time()-validation_cpu_start
    validation_wall=time.perf_counter()-validation_start
    if search is not None:
        if not math.isclose(selected["score"],search["best_score"],rel_tol=1e-11,abs_tol=1e-12):
            raise AssertionError("selected full revalidation disagrees with search score")
        assert selected["feasible"]==matches[0]["feasible"]
    selection.update(result_json=f"groups/{group['group_id']}.json",family=config["family"],
        layout=config["layout"]["name"],model_feasible=selected["feasible"],model_score=selected["score"],
        timing="Selected revalidation is an additional complete model call, not a searched improvement",
        replication_scope="Pinned-network adapter; no full original-paper replication claim")
    counts=dict(search_proposals=search["proposals"] if search else 0,
        search_objective_calls=len(search_calls),selected_validation_calls=1,
        total_objective_calls=len(search_calls)+1,reference_evaluations=len(search_calls)+1,
        full_tracking_rollouts=len(search_calls)+1,screened_proposals=0,partial_evaluations=0,
        cache_hits=0)
    timings=dict(search_wall_s=search_wall,selected_validation_wall_s=validation_wall,
        search_plus_selected_validation_wall_s=search_wall+validation_wall,
        search_cpu_s=search_cpu,selected_validation_cpu_s=validation_cpu,
        search_plus_selected_validation_cpu_s=search_cpu+validation_cpu,
        neural_generation_s=search["neural_generation_s"] if search else 0.,
        neural_update_s=search["neural_update_s"] if search else 0.,
        search_objective_evaluation_s=search["evaluation_s"] if search else 0.)
    return dict(group_id=group["group_id"],representation=group["representation"],
        method=group["method_label"],family=config["family"],seed=config["seed"],scope=SCOPE,
        parameters=selected_parameters,selection_provenance=selection,counts=counts,timings=timings,
        search=search,search_call_ledger=search_calls,selected_validation=selected)


def summary_row(record):
    selected=record["selected_validation"]
    details=selected["score_details"]
    metrics=details["tracking_metrics"]
    obstacles=details["actual_nominal_path"]["nominal_spheres"]
    bounds=[s["surface_clearance_lower_bound_m"] for s in obstacles]
    clearance=min(bounds) if bounds and all(x is not None for x in bounds) else None
    row=dict(group_id=record["group_id"],representation=record["representation"],method=record["method"],
        seed=record["seed"],selected_feasible=selected["feasible"],selected_score=selected["score"],
        duration_s=details["duration_s"],tracking_rmse_m=metrics["position_rmse_m"],
        peak_position_error_m=metrics["max_position_error_m"],terminal_position_error_m=metrics["terminal_position_error_m"],
        terminal_velocity_error_mps=metrics["terminal_velocity_error_mps"],jerk_energy_m2_s5=details["jerk_energy_m2_s5"],
        speed_clipped_time_fraction=metrics["speed_clipped_time_fraction"],
        command_slew_clipped_time_fraction=metrics["command_slew_clipped_time_fraction"],
        actual_nominal_clearance_lower_bound_m=clearance,record_json=f"groups/{record['group_id']}.json",
        **{name:record["counts"][name] for name in ("search_proposals","search_objective_calls",
             "selected_validation_calls","total_objective_calls")},**record["timings"])
    assert set(row)==set(SUMMARY_FIELDS)
    return row


def representation_comparisons(records):
    by_id={r["group_id"]:r for r in records}
    initial_populations=[r["search"]["initial_population_normalized"] for r in records if r["search"] is not None]
    common_initial_population=all(p==initial_populations[0] for p in initial_populations[1:])
    if not common_initial_population:
        raise AssertionError("four search groups do not share the same initial population")
    pairs=[]
    for method in ("ordinary","abom","abom_improved_candidate"):
        b,h=(by_id[f"{rep}_{method}"] for rep in ("bezier","hermite"))
        bp,hp=np.array(b["parameters"]),np.array(h["parameters"])
        same_parameters=bool(np.array_equal(bp,hp))
        selected_curve_diffs=[]
        for derivative in (0,1,2):
            bc=curve_from_parameters(bp,family=b["family"],representation="bezier")
            hc=curve_from_parameters(hp,family=h["family"],representation="hermite")
            u=np.linspace(0.,1.,1001)
            selected_curve_diffs.append(float(np.max(np.abs(bc.sample(u*bp[-1],derivative)-hc.sample(u*hp[-1],derivative)))))
        stream_comparison=None
        if b["search"] is not None:
            be,he=b["search"]["evaluations"],h["search"]["evaluations"]
            stream_comparison=dict(proposals_compared=len(be),
                exact_parameter_stream_equal=bool(np.array_equal([e["parameters"] for e in be],[e["parameters"] for e in he])),
                parameter_stream_max_abs_difference=float(np.max(np.abs(np.array([e["parameters"] for e in be])-
                    np.array([e["parameters"] for e in he])))),
                score_stream_max_abs_difference=float(np.max(np.abs(np.array([e["score"] for e in be])-
                    np.array([e["score"] for e in he])))),
                feasibility_stream_equal=[e["feasible"] for e in be]==[e["feasible"] for e in he],
                initial_population_exactly_equal=b["search"]["initial_population_normalized"]==h["search"]["initial_population_normalized"])
        pairs.append(dict(method=method,selected_parameters_exactly_equal=same_parameters,
            selected_parameters_max_abs_difference=float(np.max(np.abs(bp-hp))),
            selected_score_abs_difference=abs(b["selected_validation"]["score"]-h["selected_validation"]["score"]),
            selected_curve_same_u_position_max_abs_difference_m=selected_curve_diffs[0],
            selected_curve_same_u_velocity_max_abs_difference_mps=selected_curve_diffs[1],
            selected_curve_same_u_acceleration_max_abs_difference_mps2=selected_curve_diffs[2],
            stream_comparison=stream_comparison))
    return dict(all_four_search_initial_populations_exactly_equal=common_initial_population,
        pairs=pairs,interpretation="Canonical curve spaces are algebraically equal. Numerical "
        "roundoff, discrete ranking or selected-parameter differences are recorded; no constructed curve "
        "superiority or independent replication claim. Same-u comparison is shape/time-law comparison, "
        "not a synchronous common-time tracking error when selected durations differ.")


def export_replay_plan(records):
    return dict(schema_version=1,plans=[dict(group_id=r["group_id"],representation=r["representation"],
        family=r["family"],parameters=r["parameters"],seed=r["seed"],
        selection_provenance=r["selection_provenance"]) for r in records])


def run(output,upstream=DEFAULT_UPSTREAM):
    output=Path(output).resolve()
    upstream=Path(upstream).resolve()
    # Verify pinned network before creating an output directory or calling any oracle.
    content=(upstream/"abom_net.py").read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()
    if blob!=UPSTREAM_NETWORK_BLOB:
        raise ValueError("upstream network is not the pinned source")
    output.mkdir(parents=True,exist_ok=False)
    (output/"groups").mkdir()
    config=make_config()
    write_json(output/"config.json",config)
    sources={name:ROOT/name for name in SOURCE_NAMES}
    sources["upstream/abom_net.py"]=upstream/"abom_net.py"
    digests={}
    for name,path in sources.items():
        target=output/"source_snapshot"/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        digests[name]=sha256(path)
    def git(*args):
        return subprocess.run(["git",*args],cwd=ROOT,capture_output=True,text=True,
                              encoding="utf-8",errors="replace",check=True).stdout.strip()
    try:
        torch_version=importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        torch_version=None
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),repo_head=git("rev-parse","HEAD"),
        working_tree_status=git("status","--short"),python=platform.python_version(),numpy=np.__version__,
        torch=torch_version,platform=platform.platform(),source_sha256=digests,
        config_sha256=sha256(output/"config.json"),upstream_commit=UPSTREAM_COMMIT,
        upstream_network_blob=UPSTREAM_NETWORK_BLOB,scope=SCOPE,
        planned_execution_order=config["frozen_execution_order"],execution_order_is_performance_counterbalanced=False)
    write_json(output/"manifest.json",manifest)
    write_json(output/"basis_preflight.json",basis_preflight(config))
    records=[]
    for group in config["groups"]:
        record=run_group(group,config,upstream)
        records.append(record)
        write_json(output/"groups"/f"{group['group_id']}.json",record)
        print(json.dumps(summary_row(record),ensure_ascii=False),flush=True)
    rows=[summary_row(r) for r in records]
    write_json(output/"rows.json",rows)
    with (output/"summary.csv").open("w",encoding="utf-8",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary=dict(groups=len(records),seed=config["seed"],development_layout=config["layout"]["name"],
        search_objective_calls=sum(r["counts"]["search_objective_calls"] for r in records),
        selected_validation_calls=sum(r["counts"]["selected_validation_calls"] for r in records),
        total_objective_calls=sum(r["counts"]["total_objective_calls"] for r in records),
        selected_model_feasible=sum(r["selected_validation"]["feasible"] for r in records),
        airsim="NOT_RUN",screening=False,paired_perturbation_is_unproven_candidate=True,scope=SCOPE,
        timing_scope=config["timing_scope"],no_warmup_objective_calls=True)
    assert summary["search_objective_calls"]==320 and summary["selected_validation_calls"]==6
    assert summary["total_objective_calls"]==326
    write_json(output/"summary.json",summary)
    write_json(output/"representation_comparisons.json",representation_comparisons(records))
    write_json(output/"airsim_replay_plan.json",export_replay_plan(records))
    changed=[name for name,path in sources.items() if sha256(path)!=digests[name]]
    if changed:
        raise RuntimeError(f"covered source changed during pilot: {changed}")
    artifacts={p.relative_to(output).as_posix():sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_json(output/"completion.json",dict(status="COMPLETE",finished_at_utc=datetime.now(timezone.utc).isoformat(),
        groups=6,total_objective_calls=326,producer_attests_source_bytes_unchanged=True,
        artifact_sha256=artifacts))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True,help="New directory, refuses overwrite")
    parser.add_argument("--upstream",type=Path,default=DEFAULT_UPSTREAM,help="Directory containing pinned abom_net.py")
    args=parser.parse_args()
    print(json.dumps(run(args.output,args.upstream),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
