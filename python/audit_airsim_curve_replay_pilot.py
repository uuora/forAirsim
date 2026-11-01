"""Read-only replay-record audit; never connects to or controls AirSim.

The optional audit.json is a verification receipt, not a new execution record.
PASS concerns consistency of all selected completed records, not passing control gates,
actual actuator acknowledgements, real-time ability or physical-flight truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

from trajectory_research.parameterizations import BoundaryStates,curve_from_parameters


ROOT = Path(__file__).resolve().parents[1]
SCOPE = (
    "Offline consistency audit of recorded AirSim API/state evidence; no RPC, replay or optimization. "
    "No proof of unlogged-call absence, host/simulator causality, actuator execution, continuous "
    "collision clearance, hard real time, simplified-model dynamics or real-flight validity. "
    "Audit PASS is distinct from control-error gates passing."
)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual,expected,label,*,atol=1e-9):
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=atol,equal_nan=False,err_msg=label)


def inside(root,path):
    result=(root/path).resolve()
    if root.resolve() not in result.parents:
        raise AssertionError(f"evidence path leaves directory: {path}")
    return result


def jsonlines(path):
    records=[]
    with Path(path).open(encoding="utf-8-sig") as file:
        for index,line in enumerate(file,1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise AssertionError(f"incomplete journal record at {path}:{index}") from exc
    hosts=[r["host_s"] for r in records]
    assert all(math.isfinite(x) for x in hosts)
    assert all(a<=b for a,b in zip(hosts,hosts[1:])),"journal host order decreases"
    return records


def verify_loaded_profile(config,probe):
    assert config["host"]=="127.0.0.1" and config["port"]==41461 and config["vehicle"]=="ResearchDrone"
    expected=config["settings"]
    assert expected["PhysicsEngineName"]=="FastPhysicsEngine" and expected["SimMode"]=="Multirotor"
    assert expected["ClockType"]=="SteppableClock" and expected["ClockSpeed"]==1.
    assert expected["ApiServerPort"]==41461
    assert list(expected["Vehicles"])==["ResearchDrone"]
    assert expected["Vehicles"]["ResearchDrone"]["VehicleType"]=="SimpleFlight"
    loaded=probe["loaded_settings"]
    for key in ("ApiServerPort","SimMode","PhysicsEngineName","ClockType","ClockSpeed"):
        assert loaded[key]==expected[key],f"loaded setting mismatch: {key}"
    assert list(loaded["Vehicles"])==["ResearchDrone"]
    vehicle=loaded["Vehicles"]["ResearchDrone"]
    assert vehicle["VehicleType"]=="SimpleFlight"
    assert vehicle["EnableCollisions"] is True and vehicle["EnableCollisionPassthrough"] is False
    assert probe["vehicles"]==["ResearchDrone"]
    assert probe["status"]=="PASS" and probe["checks"] and all(probe["checks"].values())
    assert probe["armed_by_probe"] is False and probe["movement_commands_sent"]==0
    assert probe["step"]["observed_sim_s"]>0
    assert config["controller"]["position_gain"]==2.
    assert config["controller"]["max_command_speed_mps"]==4.
    assert config["controller"]["max_command_slew_mps2"]==4.
    if config.get('protocol_revision',1)>=4:
        assert 0<=config['controller'].get('velocity_error_gain',0.)<=10
        assert 0<=config['controller'].get('acceleration_feedforward_s',0.)<=2
    assert config["execution"]["max_state_gap_s"]<=.10
    assert config["execution"]["terminal_interpolation_max_gap_s"]<=.10
    assert config["control_error_gates"]==dict(sampled_peak_position_m=.5,terminal_position_m=.25,terminal_velocity_mps=.5)


def collision_key(item):
    return (int(item.get("time_stamp",0)),item.get("object_name"),item.get("object_id"))


def is_new_collision(event,baseline,t0):
    if not event.get("has_collided",False):
        return False
    assert "time_stamp" in event,"contact flag has no event timestamp"
    if baseline.get("has_collided",False) and collision_key(event)==collision_key(baseline):
        return False
    return int(event["time_stamp"])>=t0


def independent_metrics(curve,states,commands,config):
    times=np.array([s["reference_elapsed_s"] for s in states],dtype=float)
    positions=np.array([s["position_zup_m"] for s in states],dtype=float)
    velocities=np.array([s["velocity_zup_mps"] for s in states],dtype=float)
    duration=curve.duration_s
    assert len(times)>=2 and times[0]==0 and np.all(np.diff(times)>0) and times[-1]>=duration
    after=int(np.searchsorted(times,duration))
    interpolated=times[after]!=duration
    bracket_gap=0.
    if interpolated:
        before=after-1
        bracket_gap=float(times[after]-times[before])
        assert bracket_gap<=config["execution"]["terminal_interpolation_max_gap_s"]
        fraction=(duration-times[before])/bracket_gap
        end_p=positions[before]*(1.-fraction)+positions[after]*fraction
        end_v=velocities[before]*(1.-fraction)+velocities[after]*fraction
    else:
        end_p,end_v=positions[after],velocities[after]
    keep=times<duration
    metric_times=np.concatenate((times[keep],[duration]))
    metric_positions=np.concatenate((positions[keep],end_p[None,:]))
    error=np.linalg.norm(curve.sample(metric_times)-metric_positions,axis=1)
    square=error**2
    rmse=math.sqrt(float(np.dot(.5*(square[:-1]+square[1:]),np.diff(metric_times))/duration))
    peak=float(error.max())
    p_error=float(np.linalg.norm(curve.sample(duration)-end_p))
    v_error=float(np.linalg.norm(curve.sample(duration,1)-end_v))
    gates=config["control_error_gates"]
    checks=dict(sampled_peak_position=peak<=gates["sampled_peak_position_m"],
                terminal_position=p_error<=gates["terminal_position_m"],
                terminal_velocity=v_error<=gates["terminal_velocity_mps"])
    assert len(commands)==len(times)-1
    weights=np.maximum(0.,np.minimum(duration,times[1:])-times[:-1])
    close(weights.sum(),duration,"clipping coverage spans [0,T]")
    speed_fraction=float(np.dot(weights,[c["speed_clipped"] for c in commands])/duration)
    slew_fraction=float(np.dot(weights,[c["slew_clipped"] for c in commands])/duration)
    # Eight-node Gauss-Legendre is exact in real arithmetic for squared jerk
    # of the supported <=9-degree polynomial, with floating-point tolerance.
    nodes,quadrature_weights=np.polynomial.legendre.leggauss(8)
    third=curve.sample(duration*(nodes+1.)/2.,derivative=3)
    jerk=float(duration/2.*np.dot(quadrature_weights,np.sum(third**2,axis=1)))
    quality=.7*duration/(duration+5.)+.1*jerk/(jerk+30.)+.2*rmse/(rmse+.25)
    result=dict(reference_duration_s=duration,position_rmse_m=rmse,max_position_error_m=peak,
        terminal_position_error_m=p_error,terminal_velocity_error_mps=v_error,
        terminal_interpolated=bool(interpolated),terminal_bracket_gap_s=bracket_gap,
        terminal_actual_position_zup_m=end_p.tolist(),terminal_actual_velocity_zup_mps=end_v.tolist(),
        max_observed_state_gap_s=float(np.diff(times).max()),control_error_checks=checks,
        control_error_gates_passed=all(checks.values()),speed_clipped_time_fraction=speed_fraction,
        command_slew_clipped_time_fraction=slew_fraction,jerk_energy_m2_s5=jerk,quality_term=quality)
    if config.get('sphere_scene'):
        scene=config['sphere_scene'];minimum=float('inf')
        for p1,p2 in zip(positions[:-1],positions[1:]):
            direction=p2-p1;den=float(direction@direction)
            for sphere in scene['obstacles']:
                centre=np.asarray(sphere['centre'])
                fraction=min(1.,max(0.,float((centre-p1)@direction)/den)) if den>0 else 0.
                minimum=min(minimum,float(np.linalg.norm(p1+fraction*direction-centre))-sphere['radius']-scene['vehicle_envelope_radius_m'])
        result.update(measured_chord_min_envelope_clearance_m=minimum,
            clearance_margin_passed=minimum>=scene['safety_margin_m'],
            task_success=all(checks.values()) and minimum>=scene['safety_margin_m'])
    elif config.get('target_scene'):
        result['task_success']=all(checks.values())
    if config.get('protocol_revision',1)>=12:
        chosen=[s for s in states if s['reference_elapsed_s']<=duration]
        acc=np.array([s['acceleration_zup_mps2'] for s in chosen],dtype=float)
        rates=np.array([s['angular_velocity_body'] for s in chosen],dtype=float)
        assert acc.shape==rates.shape==(len(chosen),3) and np.isfinite(acc).all() and np.isfinite(rates).all()
        result.update(sampled_peak_estimated_acceleration_mps2=float(np.sqrt(np.sum(acc*acc,axis=1)).max()),
            sampled_peak_angular_rate_radps=float(np.sqrt(np.sum(rates*rates,axis=1)).max()))
    return result


def verify_trace(plan,trace,report,config,events):
    curve=curve_from_parameters(plan["parameters"],family=plan["family"],representation=plan["representation"],boundary=BoundaryStates(**plan['boundary']) if plan.get('boundary') else None)
    states,commands=trace["states"],trace["commands"]
    assert report["state_samples"]==len(states) and report["command_requests"]==len(commands)
    assert report["plan"]==plan and report["group_id"]==plan["group_id"]
    assert report["status"] in ("COMPLETED","FAILED")
    completed=report["status"]=="COMPLETED"
    if completed:
        assert len(states)>=2 and len(commands)==len(states)-1
        assert "metrics" in report and "initialization" in report
    elif not states:
        assert not commands and not report.get("metrics"),"preparation failure invents execution metrics"
        assert report.get("failure_reason") and report.get("failure_type")
        return dict(group_id=plan["group_id"],classification="FAILED_BEFORE_REFERENCE",status=report["status"],
            state_samples=0,command_requests=0,new_collision_observations=0,new_collision_events=0,
            failure_reason=report["failure_reason"],metrics_recomputed=False)
    assert len(commands) in (len(states)-1,len(states)),"state/command trace length mismatch"
    timestamps=[s["sim_timestamp_ns"] for s in states]
    assert all(isinstance(x,int) and not isinstance(x,bool) and x>0 for x in timestamps)
    if completed:
        assert all(b>a for a,b in zip(timestamps,timestamps[1:])),"completed replay state timestamps do not strictly increase"
    t0=timestamps[0]
    times=np.array([(x-t0)*1e-9 for x in timestamps])
    ctrl=config["controller"]
    baseline=report.get("initial_collision_baseline")
    assert isinstance(baseline,dict),"reference trace lacks recorded collision baseline"
    baseline_events=[e for e in events if e["kind"]=="REFERENCE_COLLISION_BASELINE"]
    assert len(baseline_events)==1
    assert baseline_events[0]["sim_start_timestamp_ns"]==t0 and baseline_events[0]["collision"]==baseline
    previous=curve.sample(0.,1)
    new_contacts=[]
    sampled_outside=0
    lower,upper=(np.array(config["execution"][x]) for x in ("workspace_lower_zup_m","workspace_upper_zup_m"))
    for i,state in enumerate(states):
        close(state["reference_elapsed_s"],times[i],"elapsed time from integer simulator timestamps",atol=1e-12)
        ref_time=min(curve.duration_s,max(0.,times[i]))
        close(state["reference_sample_time_s"],ref_time,"clamped reference sample time",atol=1e-12)
        close(state["reference_position_zup_m"],curve.sample(ref_time),"z-up q(t)")
        close(state["reference_velocity_zup_mps"],curve.sample(ref_time,1),"z-up qdot(t)")
        p,v=np.array(state["position_zup_m"]),np.array(state["velocity_zup_mps"])
        assert p.shape==v.shape==(3,) and np.isfinite(p).all() and np.isfinite(v).all()
        assert state["read_requested_host_s"]<=state["read_returned_host_s"]
        if i:
            contact=state["collision"]
            fresh=is_new_collision(contact,baseline,t0)
            assert state["new_execution_collision"]==fresh
            if fresh:
                new_contacts.append(collision_key(contact))
            sampled_outside+=int(np.any(p<lower) or np.any(p>upper))
    gaps=np.diff(times)
    data_quality_violations=int(np.sum((gaps<=0)|(gaps>config["execution"]["max_state_gap_s"])))
    if completed:
        assert not data_quality_violations,"completed replay violates <=0.10s data-quality gate"
    for i,command in enumerate(commands):
        assert i<len(states),"command lacks its measured input state"
        state=states[i]
        assert times[i]<curve.duration_s,"command generated after reference horizon"
        assert command["command_index"]==i and command["read_sim_timestamp_ns"]==timestamps[i]
        close(command["reference_elapsed_s"],times[i],"command reference time")
        dt=ctrl["first_command_ramp_interval_s"] if i==0 else times[i]-times[i-1]
        close(command["ramp_interval_s"],dt,"observed previous-state ramp interval")
        raw=curve.sample(times[i],1)+ctrl["position_gain"]*(curve.sample(times[i])-state["position_zup_m"])
        raw+=ctrl.get('velocity_error_gain',0.)*(curve.sample(times[i],1)-np.asarray(state['velocity_zup_mps']))
        raw+=ctrl.get('acceleration_feedforward_s',0.)*curve.sample(times[i],2)
        raw_norm=float(np.linalg.norm(raw))
        speed=raw*min(1.,ctrl["max_command_speed_mps"]/max(raw_norm,1e-300))
        delta=speed-previous
        dn=float(np.linalg.norm(delta))
        limited=previous+delta*min(1.,ctrl["max_command_slew_mps2"]*dt/max(dn,1e-300))
        close(command["raw"],raw,"controller velocity feedforward plus position feedback")
        close(command["speed_limited"],speed,"command speed projection")
        close(command["limited"],limited,"command slew projection")
        assert command["speed_clipped"]==bool(raw_norm>ctrl["max_command_speed_mps"])
        assert command["slew_clipped"]==bool(dn>ctrl["max_command_slew_mps2"]*dt)
        close(command["command_slew_mps2"],np.linalg.norm(limited-previous)/dt,"observed command change rate")
        close(command["velocity_command_ned_mps"],limited*np.array([1.,1.,-1.]),"z-up to NED velocity request")
        assert np.linalg.norm(limited)<=ctrl["max_command_speed_mps"]+1e-9
        assert command["command_slew_mps2"]<=ctrl["max_command_slew_mps2"]+1e-9
        close(command["requested_command_duration_s"],ctrl["command_duration_s"],"requested API duration")
        assert command["send_requested_host_s"]<=command["async_returned_host_s"]
        if i<len(states)-1:
            close(command["actual_state_gap_sim_s"],gaps[i],"observed next-state simulation gap")
            close(command["requested_step_wall_s"],config["execution"]["step_wall_request_s"],"requested step wall time")
            assert command["step_started_host_s"]<=command["step_returned_host_s"]
        previous=limited
    command_events=[e["command"] for e in events if e["kind"]=="VELOCITY_REQUEST"]
    step_events=[e for e in events if e["kind"]=="COMPLETED_STEP"]
    assert len(command_events)==len(commands) and len(step_events)==len(states)-1
    for recorded,command in zip(command_events,commands):
        assert all(command[key]==value for key,value in recorded.items()),"command journal and trace differ"
    for index,event in enumerate(step_events):
        assert event["state"]==states[index+1] and event["command"]==commands[index]
    state_queries=[e["state"] for e in events if e["kind"]=="STATE_QUERY" and e["phase"]=="after_step"]
    assert len(state_queries)>=len(states)-1
    for observed,state in zip(state_queries,states[1:]):
        assert all(state[key]==value for key,value in observed.items()),"RPC state receipt and trace differ"
    metrics=None
    classification="FAILED_DURING_REFERENCE"
    if completed:
        initialization=report["initialization"]
        q0,v0=curve.sample(0.),curve.sample(0.,1)
        close(initialization["requested_position_zup_m"],q0,"requested start position")
        close(initialization["requested_velocity_zup_mps"],v0,"requested start velocity")
        close(states[0]["position_zup_m"],initialization["observed_state"]["position_zup_m"],"observed initialization position")
        close(states[0]["velocity_zup_mps"],initialization["observed_state"]["velocity_zup_mps"],"observed initialization velocity")
        p0_error=np.linalg.norm(np.array(states[0]["position_zup_m"])-q0)
        v0_error=np.linalg.norm(np.array(states[0]["velocity_zup_mps"])-v0)
        close(initialization["position_error_m"],p0_error,"initial position tolerance receipt")
        close(initialization["velocity_error_mps"],v0_error,"initial velocity tolerance receipt")
        assert p0_error<=config["initialization"]["position_tolerance_m"]
        assert v0_error<=config["initialization"]["velocity_tolerance_mps"]
        if config.get('protocol_revision',1)>=12:
            close(initialization['requested_acceleration_zup_mps2'],curve.sample(0.,2),'requested initial acceleration')
            a_error=float(np.linalg.norm(np.asarray(states[0]['acceleration_zup_mps2'])-curve.sample(0.,2)))
            close(initialization['acceleration_error_mps2'],a_error,'observed initial acceleration error')
            assert a_error<=config['initialization']['acceleration_tolerance_mps2']<=.05
        if config.get("protocol_revision",1)>=3:
            for name in ("lift_rpc","zero_velocity_rpc"):
                receipt=initialization[name]
                assert isinstance(receipt["raw_result"],(bool,int)) and receipt["raw_result"]==True
                actual_sim=(receipt["after_state"]["sim_timestamp_ns"]-receipt["before_state"]["sim_timestamp_ns"])*1e-9
                close(receipt["observed_sim_elapsed_s"],actual_sim,"preparation observed simulation interval")
                assert actual_sim>0 and receipt["rpc_elapsed_host_s"]>0
            lift_state=initialization["lift_rpc"]["after_state"]
            assert lift_state["landed_state"]==1
            assert lift_state["position_zup_m"][2]>config["initialization"]["minimum_lift_height_zup_m"]
        assert not new_contacts and not sampled_outside
        assert report["observed_collision_free"] is True and report["sampled_workspace_passed"] is True
        metrics=independent_metrics(curve,states,commands,config)
        for key,value in metrics.items():
            if isinstance(value,(bool,dict)):
                assert report["metrics"][key]==value,f"metric mismatch: {key}"
            else:
                close(report["metrics"][key],value,f"independent metric: {key}",atol=1e-7 if key=="jerk_energy_m2_s5" else 1e-9)
        classification="COMPLETED_CONTROL_PASS" if metrics["control_error_gates_passed"] else "COMPLETED_CONTROL_FAIL"
        assert "post_reference" in report,"completed record lacks separate post-T takeover receipt"
    else:
        assert report.get("failure_reason") and report.get("failure_type")
    return dict(group_id=plan["group_id"],classification=classification,status=report["status"],
        state_samples=len(states),command_requests=len(commands),
        max_observed_state_gap_s=float(max(gaps)) if len(gaps) else None,
        data_quality_violations=data_quality_violations,new_collision_observations=len(new_contacts),
        new_collision_events=len(set(new_contacts)),cached_initial_contact=bool(baseline.get("has_collided",False)),
        sampled_workspace_violations=sampled_outside,metrics_recomputed=metrics is not None,
        recomputed_metrics=metrics,failure_reason=report.get("failure_reason"))


def verify_selection_lineage(plans,source_record,manifest):
    if source_record is None:
        return dict(status="UNAVAILABLE",reason="manifest lacks original plan source pointer")
    path=Path(source_record["path"])
    original=path.parent
    if not path.is_file() or not (original/"completion.json").is_file():
        return dict(status="UNAVAILABLE",reason="original simplified-search package is absent")
    assert digest(path)==source_record["sha256"],"original replay-plan file changed"
    completion=load(original/"completion.json")
    hashes=completion["artifact_sha256"]
    assert completion["status"]=="COMPLETE" and completion["groups"]==6
    assert hashes[path.name]==digest(path)
    for plan in plans:
        relative=plan["selection_provenance"]["result_json"]
        result_path=inside(original,relative)
        assert relative in hashes and digest(result_path)==hashes[relative]
        result=load(result_path)
        for field in ("group_id","representation","family","parameters","seed"):
            assert result[field]==plan[field],f"plan differs from original selection: {field}"
        assert result.get('boundary')==plan.get('boundary'),'endpoint state provenance changed'
        assert result["selection_provenance"]==plan["selection_provenance"]
        selected=result["selected_validation"]
        assert selected["parameters"]==plan["parameters"]
        assert selected["feasible"]==plan["selection_provenance"]["model_feasible"]
        close(selected["score"],plan["selection_provenance"]["model_score"],"selected model score provenance")
    return dict(status="VERIFIED",plans=len(plans),original_package=str(original),
                original_plan_sha256=digest(path))


def verify_csv(directory,reports):
    with (directory/"metrics.csv").open(encoding="utf-8-sig",newline="") as file:
        rows=list(csv.DictReader(file))
    assert len(rows)==len(reports)
    for row,report in zip(rows,reports):
        assert row["group_id"]==report["group_id"] and row["status"]==report["status"]
        assert row["failure_reason"]==(report.get("failure_reason") or "")
        for key,cell in row.items():
            if key in ("group_id","status","failure_reason"):
                continue
            expected=report.get("metrics",{}).get(key)
            if expected is None:
                assert cell=="",f"failed/unmeasured metric fabricated in CSV: {key}"
            elif isinstance(expected,bool):
                assert cell==str(expected)
            else:
                close(float(cell),expected,f"metrics CSV {key}")


def manifest_plan_selection(manifest):
    """Independently bind selected slots to the complete frozen source plan.

    Older manifests lack selection fields and mean the full six-plan order.
    New manifests carry both fields; unselected plans are not failed/not-run
    selected execution slots. This check does not import the producer selector.
    """
    if bool(manifest["probe_only"]):
        assert manifest["planned_replays"] == 0
        assert manifest.get("selected_group_ids", []) == []
        assert manifest.get("unselected_group_ids", []) == []
        return [], [], [], []
    document = manifest["plans_document"]
    assert document["schema_version"] == 1
    full = document["plans"]
    assert isinstance(full, list) and len(full) == 6
    source_ids = [plan["group_id"] for plan in full]
    assert len(set(source_ids)) == 6
    assert {plan["representation"] for plan in full} == {"bezier", "hermite"}
    assert all(plan["family"] == "single_bump" and isinstance(plan['seed'],int) for plan in full)
    assert len({plan['seed'] for plan in full})==1
    if "selected_group_ids" in manifest or "unselected_group_ids" in manifest:
        assert "selected_group_ids" in manifest and "unselected_group_ids" in manifest, "incomplete selection metadata"
        selected_ids, unselected_ids = manifest["selected_group_ids"], manifest["unselected_group_ids"]
        assert isinstance(selected_ids, list) and 1 <= len(selected_ids) <= 6
        assert all(isinstance(group, str) and group.strip() for group in selected_ids)
        assert len(set(selected_ids)) == len(selected_ids), "duplicate selected group IDs"
        assert set(selected_ids) <= set(source_ids), "selected group is absent from frozen source"
        assert unselected_ids == [group for group in source_ids if group not in selected_ids], "unselected group metadata mismatch"
    else:
        selected_ids, unselected_ids = source_ids, []
    assert manifest["planned_replays"] == len(selected_ids), "planned count differs from selected slots"
    by_id = {plan["group_id"]: plan for plan in full}
    return full, [by_id[group] for group in selected_ids], selected_ids, unselected_ids


def audit_directory(directory,*,write=True):
    directory=Path(directory).resolve()
    if not (directory/"manifest.json").is_file() and (directory/"replay"/"manifest.json").is_file():
        directory=directory/"replay"
    manifest=load(directory/"manifest.json")
    config=manifest["config"]
    assert manifest["schema_version"]==1
    sources=manifest["sources"]
    assert len({s["snapshot"] for s in sources})==len(sources)
    source_plan=None
    core_sources_match=True
    current_differences=[]
    config_bound=False
    for source in sources:
        snapshot=inside(directory,source["snapshot"])
        assert digest(snapshot)==source["sha256"],f"frozen source SHA mismatch: {source['snapshot']}"
        name=Path(source["path"]).name
        if name=="airsim_replay_plan.json" or (name.endswith('.json') and manifest.get('plans_document') is not None and load(snapshot)==manifest['plans_document']):
            assert load(snapshot)==manifest["plans_document"]
            source_plan=source
        elif name.endswith(".json") and load(snapshot)==config:
            config_bound=True
        original=Path(source["path"])
        differs=not original.is_file() or digest(original)!=source["sha256"]
        if differs:
            current_differences.append(source["path"])
        if name in ("parameterizations.py","curves.py") and differs:
            core_sources_match=False
    assert config_bound,"embedded configuration has no matching frozen source snapshot"
    if config.get('target_scene') and not manifest.get('probe_only') and manifest.get('status')=='COMPLETED':
        receipt=manifest['target_scene_receipt'];scene=config['target_scene']
        assert receipt['status']=='PASS' and receipt['scene']==scene and scene['obstacles']==[]
        assert all(p['boundary']==scene['boundary'] for p in manifest['plans_document']['plans'])
        for observed,key in zip(receipt['objects'],('p0','p1')):
            close(observed['world_ned_m'],np.asarray(scene['boundary'][key])*[1,1,-1]+receipt['world_origin_offset_ned_m'],'03 target marker',atol=.002)
    if config.get('sphere_scene') and not manifest.get('probe_only') and manifest.get('status')=='COMPLETED':
        scene=config['sphere_scene'];receipt=manifest['scene_receipt']
        assert receipt['status']=='PASS' and receipt['scene']==scene
        import hashlib
        expected=hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert receipt['scene_sha256']==expected
        assert len(receipt['objects'])==len(scene['obstacles'])
        for configured,observed in zip(scene['obstacles'],receipt['objects']):
            assert observed['name']==configured['name']==observed['contact']['object_name']
            assert observed['occupied_region_voxels']>0
            close(observed['scale'],[2*configured['radius']]*3,'spawned sphere diameter')
            close(observed['position_world_ned_m'],np.asarray(configured['centre'])*[1,1,-1]+receipt['world_origin_offset_ned_m'],'spawned sphere centre',atol=.002)
    assert core_sources_match,"current curve construction differs from frozen source; cannot reuse it for replay"
    if manifest.get("sources_unchanged") is False:
        raise AssertionError("producer recorded source modifications during execution")
    probe_path=directory/"ground_probe.json"
    profile_verified=False
    if probe_path.is_file():
        verify_loaded_profile(config,load(probe_path))
        profile_verified=True
    probe_only=bool(manifest["probe_only"])
    full_plans, plans, selected_ids, unselected_ids = manifest_plan_selection(manifest)
    reports=load(directory/"rows.json") if (directory/"rows.json").is_file() else []
    assert len(reports)<=len(plans)
    attempted_ids=[r["group_id"] for r in reports]
    assert attempted_ids==[p["group_id"] for p in plans[:len(reports)]],"attempts are not a prefix of frozen plan slots"
    groups=[]
    all_hashes={"manifest.json":digest(directory/"manifest.json")}
    for plan,report in zip(plans,reports):
        group_dir=inside(directory,plan["group_id"])
        assert load(group_dir/"report.json")==report,"row copy differs from per-plan report"
        if 'planning_records' in manifest:
            expected=manifest['planning_records'].get(plan['group_id'])
            assert report.get('planning_evidence')==expected,'planning timing/count report copy differs'
            if expected:
                bound=[s for s in sources if s['sha256']==expected['source_sha256']]
                assert len(bound)==1,'planning evidence has no unique frozen model-result snapshot'
                model=load(inside(directory,bound[0]['snapshot']))
                assert model['counts']==expected['counts'] and model['timings']==expected['timings']
                assert model['parameters']==plan['parameters']
                if plan.get('boundary'):assert model['boundary']==plan['boundary']
                if 'search_call_ledger' in model:
                    assert model['counts']['search_objective_calls']==len(model['search_call_ledger'])
                    assert model['counts']['total_objective_calls']==len(model['search_call_ledger'])+1
                timing=expected['timings']
                if expected.get('model_config_sha256'):
                    configs=[s for s in sources if s['sha256']==expected['model_config_sha256']]
                    assert len(configs)==1,'planning model configuration snapshot missing'
                    model_config=load(inside(directory,configs[0]['snapshot']))
                    assert model_config.get('model_revision','legacy_position_feedback_surrogate')==expected['model_revision']
                assert all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in timing.values())
                close(timing['search_plus_selected_validation_wall_s'],timing['search_wall_s']+timing['selected_validation_wall_s'],'planning total wall timer')
        trace=load(group_dir/"trace.json")
        events=jsonlines(group_dir/"events.jsonl")
        groups.append(verify_trace(plan,trace,report,config,events))
        if trace["states"]:
            with (group_dir/"states.csv").open(encoding="utf-8-sig",newline="") as file:
                csv_rows=list(csv.DictReader(file))
            assert len(csv_rows)==len(trace["states"])
            for row,state in zip(csv_rows,trace["states"]):
                assert int(row["sim_timestamp_ns"])==state["sim_timestamp_ns"]
                for name in ("reference_elapsed_s","read_requested_host_s","read_returned_host_s"):
                    close(float(row[name]),state[name],f"state CSV {name}")
                for prefix,key in (("reference_p","reference_position_zup_m"),("reference_v","reference_velocity_zup_mps"),
                                   ("actual_p","position_zup_m"),("actual_v","velocity_zup_mps")):
                    for axis,value in zip("xyz",state[key]):
                        close(float(row[f"{prefix}_{axis}"]),value,f"state CSV {prefix}")
        for name in ("report.json","trace.json","events.jsonl","states.csv"):
            path=group_dir/name
            if path.is_file():
                all_hashes[path.relative_to(directory).as_posix()]=digest(path)
    if (directory/"metrics.csv").is_file():
        verify_csv(directory,reports)
    completed=sum(r["status"]=="COMPLETED" for r in reports)
    not_run=[p["group_id"] for p in plans[len(reports):]]
    if manifest["status"]!="RUNNING":
        assert manifest["attempted_replays"]==len(reports) and manifest["completed_replays"]==completed
        assert manifest["not_run_group_ids"]==not_run
    if completed:
        assert profile_verified,"completed execution lacks verified loaded physics/settings"
    if manifest["status"]=="COMPLETED":
        assert len(reports)==completed==len(plans) and not not_run
    # Always verify all six source selections, including those not replayed.
    # A subset execution must not weaken the original model-package binding.
    lineage=verify_selection_lineage(full_plans,source_plan,manifest) if full_plans else dict(status="NOT_APPLICABLE")
    complete=(not probe_only and manifest["status"]=="COMPLETED" and completed==len(plans)
              and profile_verified and lineage["status"]=="VERIFIED")
    status="PASS" if complete else "PARTIAL"
    if probe_only and manifest["status"]=="PASS" and profile_verified:
        status="PROBE_ONLY_PASS"
    for name in ("ground_probe.json","rows.json","metrics.csv","events.jsonl"):
        path=directory/name
        if path.is_file():
            all_hashes[name]=digest(path)
    result=dict(status=status,evidence_consistency="PASS",producer_status=manifest["status"],
        protocol_revision=config.get("protocol_revision",1),planned_replays=len(plans),attempted_replays=len(reports),
        source_plan_count=len(full_plans),selected_group_ids=selected_ids,unselected_group_ids=unselected_ids,
        all_six_completed=bool(complete and len(plans)==6),
        completion_scope="GROUND_PROBE_ONLY" if probe_only else ("FULL_SIX" if len(plans)==6 else "SELECTED_SUBSET"),
        completed_replays=completed,failed_replays=sum(r["status"]=="FAILED" for r in reports),
        not_run_group_ids=not_run,loaded_profile_verified=profile_verified,
        snapshot_sha256_verified=len(sources),current_source_differences=current_differences,
        original_selection_lineage=lineage,groups=groups,
        completed_control_passes=sum(g["classification"]=="COMPLETED_CONTROL_PASS" for g in groups),
        completed_control_failures=sum(g["classification"]=="COMPLETED_CONTROL_FAIL" for g in groups),
        audited_evidence_sha256=all_hashes,auditor_source_sha256=digest(Path(__file__)),scope=SCOPE)
    if write:
        (directory/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path,help="Replay evidence folder or launcher parent folder")
    parser.add_argument("--no-write",action="store_true",help="Print result without creating audit.json")
    args=parser.parse_args()
    try:
        result=audit_directory(args.directory,write=not args.no_write)
    except Exception as exc:
        result=dict(status="FAIL",evidence_consistency="FAIL",error_type=type(exc).__name__,
                    error=str(exc),traceback=traceback.format_exc(),scope=SCOPE)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
