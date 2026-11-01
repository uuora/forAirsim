"""Replay selected frozen curve plans in an isolated, configured AirSim.

The required ground probe never arms or sends a movement command. Flight mode
repeats that probe and selects one to six plans from a complete frozen six-plan
source. Omitting --groups retains the historical six-plan order. Plans were optimized elsewhere;
this program does not run ABOM or claim that AirSim was its optimization oracle.
No process is launched or killed here; the separate launcher owns its instance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
import traceback

import numpy as np

from trajectory_research.curve_task import jerk_energy
from trajectory_research.parameterizations import BoundaryStates,curve_from_parameters
from curve_airsim_sphere_scene import (validate_scene, setup_scene, scene_hash, swept_clearance,
                                       show_reference, show_step, capture_overview)


ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("AirSim replay of frozen curves found by separately declared simplified-model search; "
         "simulation-only initial-state injection, fixed 3D velocity-command outer loop; "
         "not direct AirSim optimization, real flight, motor-saturation verification or hard real-time evidence.")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "__dict__"):
        return plain(vars(value))
    return value


def vec(value):
    a = np.array([value.x_val, value.y_val, value.z_val], dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite AirSim state")
    return a


def zup(ned):
    a = np.asarray(ned, dtype=float).copy()
    a[2] *= -1
    return a


def limit_command(raw, previous, dt, speed_cap, slew_cap):
    raw, previous = np.asarray(raw, dtype=float), np.asarray(previous, dtype=float)
    if raw.shape != (3,) or previous.shape != (3,) or not np.isfinite(raw).all() \
            or not np.isfinite(previous).all() or not math.isfinite(dt) or dt <= 0:
        raise ValueError("finite 3D commands and positive observed ramp interval required")
    norm = float(np.linalg.norm(raw))
    speed = raw * min(1., speed_cap / max(norm, 1e-300))
    change = speed - previous
    change_norm = float(np.linalg.norm(change))
    limited = previous + change * min(1., slew_cap * dt / max(change_norm, 1e-300))
    return dict(raw=raw.tolist(), speed_limited=speed.tolist(), limited=limited.tolist(),
                speed_clipped=bool(norm > speed_cap), slew_clipped=bool(change_norm > slew_cap * dt),
                ramp_interval_s=float(dt), command_slew_mps2=float(np.linalg.norm(limited-previous)/dt))


def new_execution_collision(collision, baseline, reference_start_timestamp_ns):
    """AirSim's collision flag is cached: compare its recorded event identity.

    PawnSimApi::onCollision stores has_collided/time_stamp/object; ordinary
    getCollisionInfo returns that stored event rather than clearing its flag.
    Every newly observed contact timestamp at/after reference start fails,
    regardless of penetration or object type. A cached preparation contact is
    retained in the trace, not misclassified as a new flight-period contact.
    """
    if not collision.get("has_collided", False):
        return False
    if "time_stamp" not in collision:
        raise ValueError("collision flag lacks an event timestamp")
    timestamp = int(collision["time_stamp"])
    key = lambda item: (int(item.get("time_stamp", 0)), item.get("object_name"), item.get("object_id"))
    if baseline.get("has_collided", False) and key(collision) == key(baseline):
        return False
    if timestamp < reference_start_timestamp_ns:
        return False
    return True


def validate_config(config):
    if config.get("port") != 41461 or config.get("vehicle") != "ResearchDrone":
        raise ValueError("pilot must use its isolated 41461 / ResearchDrone configuration")
    s = config["settings"]
    if s.get("ApiServerPort") != config["port"] or list(s["Vehicles"]) != [config["vehicle"]] \
            or s.get("PhysicsEngineName") != "FastPhysicsEngine" or s.get("SimMode") != "Multirotor" \
            or s["Vehicles"][config["vehicle"]].get("VehicleType") != "SimpleFlight" \
            or s.get("ClockType") != "SteppableClock" or s.get("ClockSpeed") != 1.0:
        raise ValueError("isolated fixed SimpleFlight/FastPhysics/SteppableClock profile required")
    for section, names in (("controller", ("position_gain", "max_command_speed_mps", "max_command_slew_mps2",
                                          "first_command_ramp_interval_s", "command_duration_s")),
                           ("execution", ("step_wall_request_s", "max_state_gap_s", "terminal_interpolation_max_gap_s",
                                          "max_wall_s_per_plan", "max_steps_per_plan")),
                           ("initialization", ("lift_velocity_zup_mps", "lift_duration_s", "minimum_lift_height_zup_m",
                                               "zero_velocity_preparation_s", "position_tolerance_m", "velocity_tolerance_mps"))):
        for name in names:
            x = config[section][name]
            if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0:
                raise ValueError(f"invalid {section}.{name}")
    for name in ('velocity_error_gain','acceleration_feedforward_s'):
        value=config['controller'].get(name,0.)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('invalid controller.'+name)
    if config.get('sphere_scene'):validate_scene(config['sphere_scene'])
    if config.get('protocol_revision',1)>=12:
        tolerance=config['initialization'].get('acceleration_tolerance_mps2')
        if isinstance(tolerance,bool) or not isinstance(tolerance,(float,int)) or not 0<tolerance<=.05:
            raise ValueError('protocol12 requires an initial acceleration tolerance in (0,0.05] m/s2')
    return config


def tracking_command(curve,t,state,ctrl):
    q,v,a=(curve.sample(t,derivative=d) for d in (0,1,2))
    return (v+ctrl['position_gain']*(q-np.asarray(state['position_zup_m']))
            +ctrl.get('velocity_error_gain',0.)*(v-np.asarray(state['velocity_zup_mps']))
            +ctrl.get('acceleration_feedforward_s',0.)*a)


def plan_curve(plan):
    boundary=BoundaryStates(**plan['boundary']) if plan.get('boundary') else None
    return curve_from_parameters(plan['parameters'],family=plan['family'],representation=plan['representation'],boundary=boundary)


def setup_targets(client,config,airsim):
    name=config['vehicle'];local=vec(client.getMultirotorState(vehicle_name=name).kinematics_estimated.position)
    origin=vec(client.simGetObjectPose(name).position)-local
    scene=config['target_scene'];boundary=scene['boundary'];objects=[]
    for actor,key in (('DC_TargetBlue','p0'),('DC_TargetRed','p1')):
        point=zup(boundary[key])+origin
        if not client.simSetObjectPose(actor,airsim.Pose(airsim.Vector3r(*point),airsim.Quaternionr()),teleport=True):
            raise RuntimeError('03 target marker missing: '+actor)
        observed=vec(client.simGetObjectPose(actor).position)
        if not np.allclose(point,observed,atol=.002):raise RuntimeError('target marker readback mismatch')
        objects.append(dict(name=actor,world_ned_m=observed.tolist(),role='start' if key=='p0' else 'goal',collision_role='noncolliding target marker'))
    return dict(status='PASS',scene=scene,world_origin_offset_ned_m=origin.tolist(),objects=objects)


def set_goal_marker(client,config,airsim,receipt,hidden):
    """Display only: park or restore the 03 goal mesh; never a control input."""
    if not config.get('target_scene') or not receipt or not config.get('visualization',{}).get('hide_goal_on_arrival'):
        return None
    goal=[o for o in receipt.get('objects',[]) if o.get('role')=='goal']
    if len(goal)!=1:
        return None
    point=airsim.Vector3r(0.,0.,1000.) if hidden else airsim.Vector3r(*goal[0]['world_ned_m'])
    ok=bool(client.simSetObjectPose(goal[0]['name'],airsim.Pose(point,airsim.Quaternionr()),teleport=True))
    return dict(actor=goal[0]['name'],hidden=bool(hidden),set_ok=ok,
                world_ned_m=[point.x_val,point.y_val,point.z_val],
                scope='Display only: goal mesh parked off-map after arrival; not a control input, collision event, metric or gate.')


def load_plans(path):
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    plans = document.get("plans") if isinstance(document, dict) else None
    if not isinstance(plans, list) or len(plans) != 6:
        raise ValueError("plan file must contain exactly six plans")
    ids = []
    for item in plans:
        name = item.get("group_id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name):
            raise ValueError("group_id must be a unique simple ASCII file-safe identifier")
        ids.append(name)
        plan_curve(item)
        if isinstance(item.get("seed"), bool) or not isinstance(item.get("seed"), int):
            raise ValueError("each frozen plan requires its search seed")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate group_id")
    return document, plans


def select_plans(plans, groups=None):
    """Select only requested source IDs, preserving the user's explicit order.

    The full source document is never rewritten. Selection is resolved before
    output directory creation and before importing/connecting to AirSim.
    """
    by_id = {plan["group_id"]: plan for plan in plans}
    if groups is None:
        raise ValueError("explicit --groups or --all-groups is required; no flight selection is implicit")
    if not isinstance(groups, (list, tuple)) or not 1 <= len(groups) <= len(plans):
        raise ValueError("--groups must select between one and six source group IDs")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("--groups contains an empty or invalid group ID")
    if len(set(groups)) != len(groups):
        raise ValueError("--groups contains duplicate group IDs")
    unknown = [group for group in groups if group not in by_id]
    if unknown:
        raise ValueError("unknown --groups ID(s): " + ", ".join(unknown))
    return [by_id[group] for group in groups]


class Journal:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.stream = (self.folder / "events.jsonl").open("x", encoding="utf-8")
        self.origin = time.perf_counter()

    def event(self, kind, **values):
        record = dict(kind=kind, recorded_at_utc=stamp(), host_s=time.perf_counter(),
                      host_elapsed_s=time.perf_counter()-self.origin, **values)
        self.stream.write(json.dumps(plain(record), ensure_ascii=False, allow_nan=False)+"\n")
        self.stream.flush()
        return record

    def close(self):
        self.stream.close()


def read_state(client, vehicle, journal, phase):
    requested = time.perf_counter()
    state = client.getMultirotorState(vehicle_name=vehicle)
    received = time.perf_counter()
    kin = state.kinematics_estimated
    record = dict(sim_timestamp_ns=int(state.timestamp), read_requested_host_s=requested,
                  read_returned_host_s=received, position_zup_m=zup(vec(kin.position)).tolist(),
                  velocity_zup_mps=zup(vec(kin.linear_velocity)).tolist(),
                  orientation_ned=plain(kin.orientation), angular_velocity_body=vec(kin.angular_velocity).tolist(),
                  landed_state=int(state.landed_state))
    acceleration=getattr(kin,'linear_acceleration',None)
    record['acceleration_zup_mps2']=None if acceleration is None else zup(vec(acceleration)).tolist()
    journal.event("STATE_QUERY", phase=phase, state=record)
    return record


def paused_ground_probe(client, config, airsim, journal):
    """Only ground state/API inspection, reset and an unarmed time-step probe."""
    name = config["vehicle"]
    result = dict(started_at_utc=stamp(), armed_by_probe=False, movement_commands_sent=0,
                  checks={}, scope="ground/paused capability probe; no flight commands")
    result["checks"]["ping"] = bool(client.ping())
    result["vehicles"] = client.listVehicles()
    result["checks"]["isolated_vehicle"] = result["vehicles"] == [name]
    if not result["checks"]["isolated_vehicle"]:
        raise RuntimeError("endpoint vehicle list does not match owned research instance")
    loaded = json.loads(client.getSettingsString())
    result["loaded_settings"] = loaded
    result["checks"]["loaded_profile"] = all(loaded.get(key) == config["settings"][key]
        for key in ("ApiServerPort", "SimMode", "PhysicsEngineName", "ClockType", "ClockSpeed")) \
        and list(loaded.get("Vehicles", {})) == [name]
    if not result["checks"]["loaded_profile"]:
        raise RuntimeError("loaded AirSim profile differs from research configuration")
    client.simPause(True)
    before = read_state(client, name, journal, "probe_before_reset")
    if before["landed_state"] != int(airsim.LandedState.Landed):
        raise RuntimeError("ground probe refuses an airborne instance")
    client.reset()
    client.simPause(True)
    result["checks"]["paused_after_reset"] = bool(client.simIsPause())
    kin = client.simGetGroundTruthKinematics(vehicle_name=name)
    client.simSetKinematics(kin, False, vehicle_name=name)
    injected = read_state(client, name, journal, "probe_identity_kinematics_set")
    result["checks"]["identity_set_position"] = bool(np.linalg.norm(
        zup(vec(kin.position))-np.asarray(injected["position_zup_m"])) <= 1e-5)
    result["checks"]["identity_set_velocity"] = bool(np.linalg.norm(
        zup(vec(kin.linear_velocity))-np.asarray(injected["velocity_zup_mps"])) <= 1e-5)
    started = time.perf_counter()
    client.simContinueForTime(config["execution"]["ground_probe_step_wall_s"])
    returned = time.perf_counter()
    after = read_state(client, name, journal, "probe_after_unarmed_step")
    result["step"] = dict(requested_wall_s=config["execution"]["ground_probe_step_wall_s"],
                          observed_host_s=returned-started,
                          observed_sim_s=(after["sim_timestamp_ns"]-injected["sim_timestamp_ns"])*1e-9)
    result["checks"]["step_sim_time_advanced"] = result["step"]["observed_sim_s"] > 0
    result["checks"]["step_repaused"] = bool(client.simIsPause())
    result["checks"]["unarmed_step_remained_landed"] = after["landed_state"] == int(airsim.LandedState.Landed)
    client.reset()
    client.simPause(True)
    client.enableApiControl(False, vehicle_name=name)
    result["end_state"] = read_state(client, name, journal, "probe_end")
    result["checks"]["grounded_at_end"] = result["end_state"]["landed_state"] == int(airsim.LandedState.Landed)
    result["status"] = "PASS" if all(result["checks"].values()) else "FAIL"
    journal.event("GROUND_PROBE_RESULT", result=result)
    return result


def velocity_preparation_stage(client, lifecycle, velocity_zup, duration_s, config, airsim, journal, stage):
    """One uniform preparation request, with measured state and raw RPC result."""
    name = config["vehicle"]
    before = read_state(client, name, journal, f"before_{stage}")
    requested = time.perf_counter()
    journal.event("PREPARATION_VELOCITY_REQUEST", stage=stage, velocity_zup_mps=list(velocity_zup),
                  requested_duration_sim_s=duration_s, before_state=before, requested_host_s=requested)
    try:
        raw_result = lifecycle.moveByVelocityAsync(*zup(velocity_zup), duration_s,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, config["controller"]["yaw_deg"]), vehicle_name=name).get()
    except Exception as exc:
        journal.event("PREPARATION_VELOCITY_RPC_EXCEPTION", stage=stage, error_type=type(exc).__name__, error=str(exc),
                      elapsed_host_s=time.perf_counter()-requested)
        raise
    returned = time.perf_counter()
    after = read_state(client, name, journal, f"after_{stage}_rpc_return")
    sim_s = (after["sim_timestamp_ns"]-before["sim_timestamp_ns"])*1e-9
    host_s = returned-requested
    receipt = dict(stage=stage, requested_velocity_zup_mps=list(velocity_zup), requested_duration_sim_s=duration_s,
        raw_result=plain(raw_result), raw_result_repr=repr(raw_result),
        raw_result_type=type(raw_result).__name__, rpc_elapsed_host_s=host_s,
        observed_sim_elapsed_s=sim_s, observed_sim_per_host=sim_s/host_s if host_s > 0 else None,
        before_state=before, after_state=after,
        api_control_enabled_after=bool(client.isApiControlEnabled(vehicle_name=name)),
        note="Fixed preparation independent of optimized curve. False/void RPC results are not successful preparation.")
    try:
        receipt["after_rotor_states"] = plain(client.getRotorStates(vehicle_name=name))
    except Exception as exc:
        receipt["rotor_diagnostic_error"] = f"{type(exc).__name__}: {exc}"
    journal.event("PREPARATION_VELOCITY_RESULT", stage=stage, receipt=receipt)
    # RPC serialization normally yields bool; accept its integer-1 equivalent,
    # while preserving the actual type. Do not reinterpret None or False.
    succeeded = (isinstance(raw_result, (bool, int)) and raw_result == True)
    if not succeeded:
        raise RuntimeError(f"uniform {stage} velocity preparation did not complete: result={raw_result!r} "
                           f"type={type(raw_result).__name__}, host={host_s:.6g}s, sim={sim_s:.6g}s")
    return receipt


def initialize_curve(client, lifecycle, curve, config, airsim, journal):
    name = config["vehicle"]
    started = time.perf_counter()
    reset_result = client.reset()
    pause_result = client.simPause(False)
    enable_result = client.enableApiControl(True, vehicle_name=name)
    enabled = client.isApiControlEnabled(vehicle_name=name)
    arm_result = lifecycle.armDisarm(True, vehicle_name=name) if enabled else False
    journal.event("INITIALIZATION_CONTROL_RESULTS", reset_raw_result=plain(reset_result),
                  pause_raw_result=plain(pause_result), enable_raw_result=plain(enable_result),
                  api_control_enabled=bool(enabled), arm_raw_result=plain(arm_result), arm_result_type=type(arm_result).__name__,
                  state=read_state(client, name, journal, "after_reset_enable_and_arm"))
    if not enabled or not arm_result:
        raise RuntimeError("research vehicle rejected control or arming")
    init = config["initialization"]
    lift = velocity_preparation_stage(client, lifecycle, [0., 0., init["lift_velocity_zup_mps"]],
                                     init["lift_duration_s"], config, airsim, journal, "fixed_vertical_lift")
    lifted = lift["after_state"]
    if lifted["landed_state"] != int(airsim.LandedState.Flying) \
            or lifted["position_zup_m"][2] <= init["minimum_lift_height_zup_m"]:
        raise RuntimeError("uniform velocity lift failed Flying/height initialization checks")
    zero = velocity_preparation_stage(client, lifecycle, [0., 0., 0.], init["zero_velocity_preparation_s"],
                                     config, airsim, journal, "fixed_zero_velocity")
    client.simPause(True)
    prepared = read_state(client, name, journal, "after_uniform_velocity_lift_and_zero_velocity")
    kin = airsim.KinematicsState()
    p0, v0, a0 = (curve.sample(0., derivative=i) for i in (0, 1, 2))
    kin.position, kin.linear_velocity, kin.linear_acceleration = (airsim.Vector3r(*zup(v)) for v in (p0, v0, a0))
    kin.orientation = airsim.Quaternionr(0., 0., 0., 1.)
    kin.angular_velocity = airsim.Vector3r(0., 0., 0.)
    kin.angular_acceleration = airsim.Vector3r(0., 0., 0.)
    client.simSetKinematics(kin, False, vehicle_name=name)
    observed = read_state(client, name, journal, "after_simulation_only_initial_state_injection")
    p_error = float(np.linalg.norm(np.asarray(observed["position_zup_m"])-p0))
    v_error = float(np.linalg.norm(np.asarray(observed["velocity_zup_mps"])-v0))
    a_error=None if observed['acceleration_zup_mps2'] is None else float(np.linalg.norm(np.asarray(observed['acceleration_zup_mps2'])-a0))
    receipt = dict(initialization_wall_s=time.perf_counter()-started, lift_rpc=lift, zero_velocity_rpc=zero, preparation_state=prepared,
        requested_position_zup_m=p0.tolist(), requested_velocity_zup_mps=v0.tolist(),
        observed_state=observed, position_error_m=p_error, velocity_error_mps=v_error,
        requested_acceleration_zup_mps2=a0.tolist(),acceleration_error_mps2=a_error,
        state_injection_scope=config["initialization"]["state_injection"],
        pid_reset_method="AirSim reset before the same fixed-velocity-lift/zero-velocity preparation; hidden-state equality not observed")
    journal.event("INITIALIZATION_RECEIPT", receipt=receipt)
    if p_error > config["initialization"]["position_tolerance_m"] \
            or v_error > config["initialization"]["velocity_tolerance_mps"]:
        raise RuntimeError("simulator initial-state injection failed its p/v tolerances")
    if config.get('protocol_revision',1)>=12 and (a_error is None or a_error>config['initialization']['acceleration_tolerance_mps2']):
        raise RuntimeError('initial-state acceleration observation missing or outside declared tolerance')
    if not client.simIsPause():
        raise RuntimeError("initial reference state is not paused")
    return observed, receipt


def metrics_from_trace(curve, states, commands, config):
    """Recompute sampled metrics, inserting one declared terminal interpolation."""
    t = np.asarray([x["reference_elapsed_s"] for x in states])
    p = np.asarray([x["position_zup_m"] for x in states])
    v = np.asarray([x["velocity_zup_mps"] for x in states])
    T = curve.duration_s
    if len(t) < 2 or t[0] != 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0) or t[-1] < T:
        raise ValueError("complete strictly ordered state trace bracketing T required")
    right = int(np.searchsorted(t, T, side="left"))
    if t[right] == T:
        terminal_p, terminal_v = p[right], v[right]
        gap, interpolation = 0., False
    else:
        left = right-1
        gap = float(t[right]-t[left])
        if gap > config["execution"]["terminal_interpolation_max_gap_s"]:
            raise ValueError("terminal state bracketing gap exceeds declared data-quality limit")
        fraction = (T-t[left])/gap
        terminal_p, terminal_v = p[left]+fraction*(p[right]-p[left]), v[left]+fraction*(v[right]-v[left])
        interpolation = True
    inside = t < T
    metric_t = np.r_[t[inside], T]
    metric_p = np.vstack((p[inside], terminal_p))
    error = np.linalg.norm(curve.sample(metric_t)-metric_p, axis=1)
    rmse = float(np.sqrt(np.sum(.5*(error[:-1]**2+error[1:]**2)*np.diff(metric_t))/T))
    peak = float(np.max(error))
    p_end = float(np.linalg.norm(curve.sample(T)-terminal_p))
    v_end = float(np.linalg.norm(curve.sample(T, derivative=1)-terminal_v))
    gates = config["control_error_gates"]
    checks = dict(sampled_peak_position=peak <= gates["sampled_peak_position_m"],
                  terminal_position=p_end <= gates["terminal_position_m"],
                  terminal_velocity=v_end <= gates["terminal_velocity_mps"])
    weights = [max(0., min(T, t[i+1])-t[i]) for i in range(len(commands))]
    clipped = {key: float(sum(w for w, c in zip(weights, commands) if c[key])/T)
               for key in ("speed_clipped", "slew_clipped")}
    energy = float(jerk_energy(curve))
    result=dict(reference_duration_s=T, position_rmse_m=rmse, max_position_error_m=peak,
        terminal_position_error_m=p_end, terminal_velocity_error_mps=v_end,
        terminal_interpolated=interpolation, terminal_bracket_gap_s=gap,
        terminal_actual_position_zup_m=terminal_p.tolist(), terminal_actual_velocity_zup_mps=terminal_v.tolist(),
        max_observed_state_gap_s=float(np.max(np.diff(t))), control_error_checks=checks,
        control_error_gates_passed=all(checks.values()), speed_clipped_time_fraction=clipped["speed_clipped"],
        command_slew_clipped_time_fraction=clipped["slew_clipped"], jerk_energy_m2_s5=energy,
        quality_term=float(.7*T/(T+5)+.1*energy/(energy+30)+.2*rmse/(rmse+.25)),
        rmse_scope="sampled synchronous errors with trapezoidal quadrature; linear actual-state interpolation at T",
        feasibility_scope="control-error gates only; observed collision/workspace checks reported separately, no continuous entity-clearance guarantee")
    if config.get('protocol_revision',1)>=12:
        chosen=[s for s in states if s['reference_elapsed_s']<=T]
        acc=np.asarray([s['acceleration_zup_mps2'] for s in chosen],dtype=float)
        rates=np.asarray([s['angular_velocity_body'] for s in chosen],dtype=float)
        if acc.shape!=(len(chosen),3) or not np.isfinite(acc).all() or not np.isfinite(rates).all():
            raise ValueError('complete finite estimated acceleration and angular-rate samples required')
        result.update(sampled_peak_estimated_acceleration_mps2=float(np.linalg.norm(acc,axis=1).max()),
            sampled_peak_angular_rate_radps=float(np.linalg.norm(rates,axis=1).max()))
    return result


def stop_after_reference(client, lifecycle, config, airsim, journal):
    """Separate post-T braking; no post-T samples enter reference metrics."""
    started = time.perf_counter()
    client.simPause(False)
    result = lifecycle.moveByVelocityAsync(0., 0., 0., config["execution"]["post_reference_zero_velocity_s"],
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, 0.),
        vehicle_name=config["vehicle"]).get()
    client.cancelLastTask(vehicle_name=config["vehicle"])
    client.simPause(True)
    client.enableApiControl(False, vehicle_name=config["vehicle"])
    return journal.event("POST_REFERENCE_BRAKING", duration_host_s=time.perf_counter()-started,
        request_result=plain(result), paused=bool(client.simIsPause()),
        end_state=read_state(client, config["vehicle"], journal, "post_reference_brake_end"),
        scope="separate simulation takeover and zero-velocity braking; landing is not claimed")


def replay_one(client, lifecycle, plan, config, airsim, folder,scene_receipt=None):
    folder.mkdir()
    journal = Journal(folder)
    report = dict(group_id=plan["group_id"], plan=plan, scope=SCOPE, started_at_utc=stamp(), status="RUNNING")
    states, commands = [], []
    started = time.perf_counter()
    try:
        curve = plan_curve(plan)
        observed, initialization = initialize_curve(client, lifecycle, curve, config, airsim, journal)
        report["initialization"] = initialization
        origin=np.asarray(scene_receipt['world_origin_offset_ned_m']) if scene_receipt else np.zeros(3)
        show_reference(client,curve,plan,config,airsim,origin)
        if config.get('sphere_scene'):
            report['scene_id']=config['sphere_scene']['scene_id']
            report['scene_sha256']=scene_hash(config['sphere_scene'])
            for sphere in config['sphere_scene']['obstacles']:
                world=zup(vec(client.simGetObjectPose(sphere['name']).position)-origin)
                if not np.allclose(world,sphere['centre'],atol=.002):raise RuntimeError('sphere moved after vehicle reset')
        if config.get('visualization',{}).get('capture_overview'):
            capture_overview(client,config,airsim,folder/'overview_start.png')
        t0 = observed["sim_timestamp_ns"]
        collision_baseline = plain(client.simGetCollisionInfo(vehicle_name=config["vehicle"]))
        report["initial_collision_baseline"] = collision_baseline
        journal.event("REFERENCE_COLLISION_BASELINE", sim_start_timestamp_ns=t0, collision=collision_baseline)
        observed.update(reference_elapsed_s=0., reference_sample_time_s=0.,
                        reference_position_zup_m=curve.sample(0.).tolist(),
                        reference_velocity_zup_mps=curve.sample(0., derivative=1).tolist())
        states.append(observed)
        previous = curve.sample(0., derivative=1)
        ctrl, execution = config["controller"], config["execution"]
        lower, upper = np.array(execution["workspace_lower_zup_m"]), np.array(execution["workspace_upper_zup_m"])
        pending = []
        while states[-1]["reference_elapsed_s"] < curve.duration_s:
            if time.perf_counter()-started > execution["max_wall_s_per_plan"] or len(commands) >= execution["max_steps_per_plan"]:
                raise TimeoutError("replay exceeded predeclared wall/step limit")
            current = states[-1]
            t = current["reference_elapsed_s"]
            if len(states) == 1:
                dt = ctrl["first_command_ramp_interval_s"]
            else:
                dt = t-states[-2]["reference_elapsed_s"]
            raw = tracking_command(curve,t,current,ctrl)
            command = limit_command(raw, previous, dt, ctrl["max_command_speed_mps"], ctrl["max_command_slew_mps2"])
            requested = time.perf_counter()
            future = client.moveByVelocityAsync(*zup(command["limited"]), ctrl["command_duration_s"],
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom, yaw_mode=airsim.YawMode(False, ctrl["yaw_deg"]),
                vehicle_name=config["vehicle"])
            pending.append(future)
            returned = time.perf_counter()
            # A synchronous read pumps this msgpack client's outgoing request loop.
            # It is not an acknowledgment of physical actuator execution.
            read_state(client, config["vehicle"], journal, "paused_after_command_request")
            command.update(command_index=len(commands), reference_elapsed_s=t,
                read_sim_timestamp_ns=current["sim_timestamp_ns"], send_requested_host_s=requested,
                async_returned_host_s=returned, requested_command_duration_s=ctrl["command_duration_s"],
                velocity_command_ned_mps=zup(command["limited"]).tolist())
            journal.event("VELOCITY_REQUEST", command=command)
            commands.append(command)
            step_started = time.perf_counter()
            client.simContinueForTime(execution["step_wall_request_s"])
            step_returned = time.perf_counter()
            following = read_state(client, config["vehicle"], journal, "after_step")
            elapsed = (following["sim_timestamp_ns"]-t0)*1e-9
            actual_gap = elapsed-t
            sample_time = max(0., min(elapsed, curve.duration_s))
            following.update(reference_elapsed_s=float(elapsed), reference_sample_time_s=sample_time,
                reference_position_zup_m=curve.sample(sample_time).tolist(),
                reference_velocity_zup_mps=curve.sample(sample_time, derivative=1).tolist())
            collision = client.simGetCollisionInfo(vehicle_name=config["vehicle"])
            following["collision"] = plain(collision)
            following["new_execution_collision"] = new_execution_collision(following["collision"], collision_baseline, t0)
            command.update(step_started_host_s=step_started, step_returned_host_s=step_returned,
                requested_step_wall_s=execution["step_wall_request_s"], actual_state_gap_sim_s=float(actual_gap))
            states.append(following)
            journal.event("COMPLETED_STEP", state=following, command=command)
            if not client.simIsPause() or actual_gap <= 0 or actual_gap > execution["max_state_gap_s"]:
                raise ValueError("step pause/time increment violated declared data-quality gate")
            if following["new_execution_collision"]:
                raise RuntimeError("collision observed during research replay")
            p = np.asarray(following["position_zup_m"])
            if np.any(p < lower) or np.any(p > upper):
                raise RuntimeError("sampled actual position left declared research workspace")
            show_step(client,current,following,config,airsim,origin)
            for f in pending[:]:
                if getattr(f, "_set_flag", False):
                    f.get()
                    pending.remove(f)
            previous = np.array(command["limited"])
        report["metrics"] = metrics_from_trace(curve, states, commands, config)
        if config.get('target_scene'):
            report['scene_id']=config['target_scene']['scene_id']
            report['metrics']['task_success']=bool(report['metrics']['control_error_gates_passed'])
        if config.get('sphere_scene'):
            report['metrics'].update(swept_clearance([r['position_zup_m'] for r in states],config['sphere_scene']))
            report['metrics']['task_success']=bool(report['metrics']['control_error_gates_passed'] and report['metrics']['clearance_margin_passed'])
        report["observed_collision_free"] = True
        report["collision_scope"] = "No newly observed contact events since the recorded initial collision baseline; old cached preparation contact is retained."
        report["sampled_workspace_passed"] = True
        report["reference_execution_wall_s"] = time.perf_counter()-started-initialization["initialization_wall_s"]
        report["status"] = "COMPLETED"
        report["post_reference"] = stop_after_reference(client, lifecycle, config, airsim, journal)
        if config.get('visualization',{}).get('capture_overview'):
            capture_overview(client,config,airsim,folder/'overview_end.png')
        hold=config.get('visualization',{}).get('final_display_hold_s',0.)
        if hold:time.sleep(min(hold,30.))
    except Exception as exc:
        report.update(status="FAILED", failure_type=type(exc).__name__, failure_reason=str(exc), traceback=traceback.format_exc())
        journal.event("REPLAY_FAILURE", failure=report["failure_reason"], failure_type=report["failure_type"])
        try:
            client.simPause(True)
            client.enableApiControl(False, vehicle_name=config["vehicle"])
            report["failure_end_paused"] = bool(client.simIsPause())
            report["failure_end_state"] = read_state(client, config["vehicle"], journal, "failure_after_pause_and_api_release")
        except Exception as cleanup:
            report["cleanup_error"] = f"{type(cleanup).__name__}: {cleanup}"
    finally:
        report.update(total_wall_s=time.perf_counter()-started, finished_at_utc=stamp(), state_samples=len(states), command_requests=len(commands))
        save(folder/"trace.json", dict(states=states, commands=commands, scope=SCOPE))
        save(folder/"report.json", report)
        if states:
            with (folder/"states.csv").open("w", newline="", encoding="utf-8") as stream:
                names = ["sim_timestamp_ns", "reference_elapsed_s", "read_requested_host_s", "read_returned_host_s"]
                names += [f"{kind}_{axis}" for kind in ("reference_p", "reference_v", "actual_p", "actual_v") for axis in "xyz"]
                if config.get('protocol_revision',1)>=12:names += ['actual_a_'+axis for axis in 'xyz']
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                for row in states:
                    item = {key: row[key] for key in names[:4]}
                    for label, key in (("reference_p", "reference_position_zup_m"), ("reference_v", "reference_velocity_zup_mps"),
                                       ("actual_p", "position_zup_m"), ("actual_v", "velocity_zup_mps")):
                        item.update({f"{label}_{axis}":value for axis, value in zip("xyz", row[key])})
                    if config.get('protocol_revision',1)>=12:
                        acceleration=row.get('acceleration_zup_mps2')
                        # Keep a failed/missing sensor observation blank in CSV;
                        # saving the failure must not invent a zero measurement.
                        if acceleration is None:acceleration=[None]*3
                        item.update({'actual_a_'+axis:value for axis,value in zip('xyz',acceleration)})
                    writer.writerow(item)
        journal.close()
    return report


def plots(output, reports, config,trace_root=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    if config.get('sphere_scene'):
        for sphere in config['sphere_scene']['obstacles']:
            u,w=np.meshgrid(np.linspace(0,2*np.pi,32),np.linspace(0,np.pi,20))
            c=np.asarray(sphere['centre']);r=sphere['radius']
            ax.plot_surface(c[0]+r*np.cos(u)*np.sin(w),c[1]+r*np.sin(u)*np.sin(w),c[2]+r*np.cos(w),color='crimson',alpha=.4)
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    fig3, axes3 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig4,axes4=plt.subplots(3,1,figsize=(11,9),sharex=True)
    fig5,axes5=plt.subplots(3,1,figsize=(11,9),sharex=True)
    all_positions=[]
    for i, report in enumerate(reports):
        path = (Path(trace_root) if trace_root else output)/report["group_id"]/"trace.json"
        trace = json.loads(path.read_text(encoding="utf-8"))
        rows, cmds = trace["states"], trace["commands"]
        if not rows:
            continue
        t = np.array([r["reference_elapsed_s"] for r in rows])
        p = np.array([r["position_zup_m"] for r in rows])
        q = np.array([r["reference_position_zup_m"] for r in rows])
        all_positions.extend([p,q])
        label = report["group_id"]+(' [FAILED]' if report["status"] != "COMPLETED" else '')
        color = f"C{i%10}"
        ax.plot(*q.T, '--', color=color, alpha=.55)
        ax.plot(*p.T, color=color, label=label)
        valid = t <= report["plan"]["parameters"][-1]
        ax2.plot(t[valid], np.linalg.norm(q[valid]-p[valid], axis=1), color=color, label=label)
        if cmds:
            ct = [r["reference_elapsed_s"] for r in cmds]
            axes3[0].plot(ct, np.linalg.norm([r["limited"] for r in cmds], axis=1), color=color, label=label)
            axes3[0].plot(t,np.linalg.norm([r['reference_velocity_zup_mps'] for r in rows],axis=1),'--',color=color,alpha=.6)
            axes3[0].plot(t,np.linalg.norm([r['velocity_zup_mps'] for r in rows],axis=1),':',color=color,alpha=.8)
            axes3[1].plot(ct, [r["command_slew_mps2"] for r in cmds], color=color)
            for axis in range(3):
                axes4[axis].plot(t,q[:,axis],'--',color=color,alpha=.5)
                axes4[axis].plot(t,p[:,axis],color=color,label=label)
                axes5[axis].plot(t,np.asarray([r['reference_velocity_zup_mps'] for r in rows])[:,axis],'--',color=color,alpha=.5)
                axes5[axis].plot(t,np.asarray([r['velocity_zup_mps'] for r in rows])[:,axis],':',color=color)
                axes5[axis].plot(ct,np.asarray([r['limited'] for r in cmds])[:,axis],color=color,label=label)
    ax.set(xlabel="Local x (m)", ylabel="Local y (m)", zlabel="Height above local origin (m)", title="AirSim replay: dashed reference, solid actual")
    ax.legend(fontsize=8)
    if all_positions:
        points=np.vstack(all_positions);span=np.maximum(np.ptp(points,axis=0),[1.,1.,1.])
        ax.set_box_aspect(span)
    ax2.set(xlabel="Actual elapsed simulation time (s)", ylabel="Synchronous position error (m)", title="Frozen-plan AirSim execution pilot")
    ax2.axhline(config["control_error_gates"]["sampled_peak_position_m"], color='black', linestyle=':')
    ax2.legend(fontsize=8)
    axes3[0].axhline(config["controller"]["max_command_speed_mps"], color='black', linestyle=':')
    axes3[1].axhline(config["controller"]["max_command_slew_mps2"], color='black', linestyle=':')
    axes3[0].set(ylabel="Speed norm (m/s)", title="Solid command / dashed reference / dotted actual; not motor signals")
    axes3[1].set(xlabel="Actual elapsed simulation time (s)", ylabel="Command change / ramp dt (m/s2)")
    axes3[0].legend(fontsize=8)
    for index,axis in enumerate('xyz'):
        axes4[index].set(ylabel=f'{axis} position (m)');axes5[index].set(ylabel=f'{axis} velocity (m/s)')
    axes4[0].set_title('Position components: dashed reference / solid actual')
    axes5[0].set_title('Velocity components: solid command / dashed reference / dotted actual')
    axes4[-1].set_xlabel('Actual elapsed simulation time (s)');axes5[-1].set_xlabel('Actual elapsed simulation time (s)')
    axes4[0].legend(fontsize=8);axes5[0].legend(fontsize=8)
    for name, figure in (("reference_actual", fig), ("tracking_error", fig2), ("velocity_commands", fig3),
                         ('position_components',fig4),('velocity_components',fig5)):
        figure.tight_layout()
        figure.savefig(output/(name+".png"), dpi=170)
        figure.savefig(output/(name+".pdf"))
        plt.close(figure)


def run(args):
    config_path = Path(args.config).resolve()
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8-sig")))
    requested_groups = getattr(args, "groups", None)
    all_groups = getattr(args, "all_groups", False)
    if all_groups and requested_groups is not None:
        raise ValueError("choose --groups or --all-groups")
    if args.probe_only:
        if requested_groups is not None or all_groups:
            raise ValueError("--groups is only valid for replay, not --probe-only")
        plans_doc, all_plans, plans = None, [], []
    else:
        plans_doc, all_plans = load_plans(args.plans)
        if all_groups:requested_groups=[p['group_id'] for p in all_plans]
        plans = select_plans(all_plans, requested_groups)
    selected_ids = [plan["group_id"] for plan in plans]
    unselected_ids = [plan["group_id"] for plan in all_plans if plan["group_id"] not in selected_ids]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshots = output/"source_snapshot"
    snapshots.mkdir()
    sources = [Path(__file__), ROOT/'python/curve_airsim_sphere_scene.py', ROOT/'python/setup_scene.py', ROOT/"python/trajectory_research/parameterizations.py", ROOT/"python/trajectory_research/curves.py",
               ROOT/"python/trajectory_research/curve_task.py", config_path]
    if not args.probe_only:
        sources.append(Path(args.plans).resolve())
        if config.get('sphere_scene'):
            sources.append(ROOT/config['sphere_scene']['asset_receipt'])
            sources.append(ROOT/'external/AirSim/Unreal/Environments/Blocks/Content/CurveControl/Meshes/SM_CurveControlSphere_v1.uasset')
    planning_records={}
    if not args.probe_only:
        package=Path(args.plans).resolve().parent
        completion_path=package/'completion.json'
        if completion_path.is_file():
            completion=json.loads(completion_path.read_text(encoding='utf-8'))
            model_config=json.loads((package/'config.json').read_text(encoding='utf-8'))
            if sha(package/'config.json')!=completion['artifact_sha256']['config.json']:
                raise ValueError('original model configuration binding failed')
            sources.extend([package/'config.json',completion_path])
            for plan in plans:
                relative=plan['selection_provenance']['result_json'];model=(package/relative).resolve()
                if package not in model.parents or sha(model)!=completion['artifact_sha256'][relative]:
                    raise ValueError('original simplified-search result binding failed')
                record=json.loads(model.read_text(encoding='utf-8'))
                if record['parameters']!=plan['parameters']:raise ValueError('model selection parameters changed')
                planning_records[plan['group_id']]=dict(source_sha256=sha(model),counts=record['counts'],timings=record['timings'],
                    model_revision=model_config.get('model_revision','legacy_position_feedback_surrogate'),
                    model_config_sha256=sha(package/'config.json'),
                    scope='Historical frozen simplified-model search plus selected validation; not AirSim optimization or current planning latency')
                sources.append(model)
    source_records = []
    for i, source in enumerate(sources):
        target = snapshots/f"{i:02d}_{source.name}"
        target.write_bytes(source.read_bytes())
        source_records.append(dict(path=str(source), snapshot=target.relative_to(output).as_posix(), sha256=sha(source)))
    manifest = dict(schema_version=1, scope=SCOPE, started_at_utc=stamp(), probe_only=args.probe_only,
                    config=config, plans_document=plans_doc, sources=source_records,
                    python=sys.executable, python_version=platform.python_version(), platform=platform.platform(),
                    planned_replays=len(plans), selected_group_ids=selected_ids, unselected_group_ids=unselected_ids,
                    selection_policy="explicit all-groups flag" if all_groups else "explicit input order",
                    status="RUNNING", ai_assistance="Code and protocol prepared with AI assistance.")
    manifest['planning_records']=planning_records
    save(output/"manifest.json", manifest)
    journal = Journal(output)
    reports = []
    client = None
    endpoint_verified = False
    try:
        import airsim
        client = airsim.MultirotorClient(ip=config["host"], port=config["port"], timeout_value=config["rpc_timeout_s"])
        lifecycle = airsim.MultirotorClient(ip=config["host"], port=config["port"], timeout_value=config["lifecycle_rpc_timeout_s"])
        # Establish the research identity with read-only calls before any global
        # pause/reset or cleanup operation, including when run without launcher.
        loaded = json.loads(client.getSettingsString())
        endpoint_verified = client.listVehicles() == [config["vehicle"]] and all(
            loaded.get(key) == config["settings"][key]
            for key in ("ApiServerPort", "SimMode", "PhysicsEngineName", "ClockType", "ClockSpeed"))
        if not endpoint_verified:
            raise RuntimeError("read-only endpoint identity check failed; refusing simulator mutation")
        probe = paused_ground_probe(client, config, airsim, journal)
        save(output/"ground_probe.json", probe)
        if probe["status"] != "PASS":
            raise RuntimeError("ground/paused capability probe did not pass; no flight allowed")
        if config.get('target_scene') and not args.probe_only:
            if any(p.get('boundary')!=config['target_scene']['boundary'] for p in plans):
                raise ValueError('target scene does not match frozen plan boundaries')
            manifest['target_scene_receipt']=setup_targets(client,config,airsim)
            save(output/'manifest.json',manifest)
        if config.get('sphere_scene') and not args.probe_only:
            if any(p['selection_provenance'].get('layout')!='central' for p in plans):
                raise ValueError('scene currently requires frozen central-layout plans')
            if [(s['centre'],s['radius']) for s in config['sphere_scene']['obstacles']]!=[([4.,0.,5.],.4)]:
                raise ValueError('scene geometry differs from frozen central planner geometry')
            manifest['scene_receipt']=setup_scene(client,config,airsim,output/'scene')
            save(output/'manifest.json',manifest)
        for plan in plans:
            set_goal_marker(client,config,airsim,manifest.get('target_scene_receipt'),False)
            report = replay_one(client, lifecycle, plan, config, airsim, output/plan["group_id"],scene_receipt=manifest.get('scene_receipt') or manifest.get('target_scene_receipt'))
            parked=set_goal_marker(client,config,airsim,manifest.get('target_scene_receipt'),
                                   bool(report.get('metrics',{}).get('control_error_gates_passed')))
            if parked:
                report['goal_marker_display']=parked
                manifest.setdefault('goal_marker_display',[]).append(dict(group_id=plan['group_id'],**parked))
            report['planning_evidence']=planning_records.get(plan['group_id'])
            report['airsim_reference_execution_calls']=1
            (output/plan['group_id']).mkdir(exist_ok=True)
            save(output/plan['group_id']/'report.json',report)
            reports.append(report)
            save(output/"rows.json", reports)
            print(json.dumps(dict(group_id=report["group_id"], status=report["status"], metrics=report.get("metrics"),
                                  failure_reason=report.get("failure_reason")), ensure_ascii=True, allow_nan=False), flush=True)
            if report["status"] != "COMPLETED":
                raise RuntimeError("pilot stopped after failed replay; later slots remain NOT_RUN")
        if reports:
            plots(output, reports, config)
        manifest["status"] = "PASS" if args.probe_only else "COMPLETED"
    except Exception as exc:
        manifest.update(status="FAIL", failure_type=type(exc).__name__, failure_reason=str(exc), traceback=traceback.format_exc())
        journal.event("PILOT_FAILURE", failure_type=type(exc).__name__, failure_reason=str(exc))
    finally:
        if client is not None and endpoint_verified:
            try:
                client.simPause(True)
                client.enableApiControl(False, vehicle_name=config["vehicle"])
                manifest["end_paused"] = bool(client.simIsPause())
                manifest["end_api_control_enabled"] = bool(client.isApiControlEnabled(vehicle_name=config["vehicle"]))
            except Exception as exc:
                manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
        manifest.update(finished_at_utc=stamp(), completed_replays=sum(r["status"] == "COMPLETED" for r in reports),
                        attempted_replays=len(reports), not_run_group_ids=[p["group_id"] for p in plans[len(reports):]],
                        sources_unchanged=all(sha(r["path"]) == r["sha256"] for r in source_records))
        if not manifest["sources_unchanged"]:
            manifest["status"] = "FAIL"
            manifest["source_change_failure"] = True
        save(output/"manifest.json", manifest)
        with (output/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            keys = ["group_id", "status", "reference_duration_s", "position_rmse_m", "max_position_error_m",
                    "terminal_position_error_m", "terminal_velocity_error_mps", "quality_term", "control_error_gates_passed",
                    "speed_clipped_time_fraction", "command_slew_clipped_time_fraction", "failure_reason"]
            if config.get('sphere_scene'):
                keys+=['measured_chord_min_envelope_clearance_m','clearance_margin_passed','task_success']
            if config.get('protocol_revision',1)>=12:keys+=['sampled_peak_estimated_acceleration_mps2','sampled_peak_angular_rate_radps']
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            for r in reports:
                row = {key:r.get(key) for key in ("group_id", "status", "failure_reason")}
                row.update({key:r.get("metrics", {}).get(key) for key in keys if key not in row})
                writer.writerow(row)
        journal.close()
    print(json.dumps(dict(status=manifest["status"], output_dir=str(output), completed_replays=manifest["completed_replays"],
                          failure_reason=manifest.get("failure_reason")), ensure_ascii=True), flush=True)
    return 0 if manifest["status"] in ("PASS", "COMPLETED") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT/"configs/curve_abom_airsim_control_targets_v12.json"))
    parser.add_argument("--plans")
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument("--groups", nargs="+", metavar="GROUP_ID",
                        help="Replay only these explicitly selected source group IDs, in this order.")
    selection.add_argument("--all-groups",action="store_true",help="Explicitly request all six groups.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-only", action="store_true")
    options = parser.parse_args()
    if not options.probe_only and not options.plans:
        parser.error("--plans is required unless --probe-only")
    if not options.probe_only and not options.groups and not options.all_groups:
        parser.error("select --groups or --all-groups; no methods run by default")
    raise SystemExit(run(options))
