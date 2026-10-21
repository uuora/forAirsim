"""AirSim interface experiment for a finite-candidate single-drone planner.

The inherited lifecycle supplies inspected-scene launch/return/landing checks.
Planning is a fixed-yaw, sampled-geometry prototype, not a safety certificate.
The sensor estimate is a visible surface point, never a balloon-center label.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np

from perception_trajectory_planner import PlanningParams, evaluate_plan, plan_trajectory
from matched_trajectory_start import (
    MatchedStartGate, initial_state_check, reposition_xy_command,
    validate_matched_start_config,
)
from single_drone_experiment import SingleDroneExperiment, summarize, terminal_success
from setup_scene import ROOT, vector
from trajectory_execution_guard import projected_bbox_support_offsets, boundary_command, command_window


class ProtectionRequired(RuntimeError):
    """An unsupported planning input needs controller protection and return."""


def finite_vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ProtectionRequired(f"{name} must be three finite values")
    return result


def quaternion_rpy_deg(quaternion_xyzw):
    q = np.asarray(quaternion_xyzw, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ProtectionRequired("Invalid actual orientation quaternion")
    norm = float(np.linalg.norm(q))
    if not 0.99 <= norm <= 1.01:
        raise ProtectionRequired("Actual orientation quaternion is not normalized")
    x, y, z, w = q / norm
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x))))
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return np.degrees([roll, pitch, yaw])


def require_fixed_forward_attitude(body_xyzw, camera_xyzw, limit_deg):
    limit = float(limit_deg)
    if not math.isfinite(limit) or not 0 < limit <= 10:
        raise ValueError("attitude_limit_deg must be in (0, 10]")
    body = quaternion_rpy_deg(body_xyzw)
    camera = quaternion_rpy_deg(camera_xyzw)
    if np.max(np.abs(body)) > limit or np.max(np.abs(camera)) > limit:
        raise ProtectionRequired("Actual body/camera attitude exceeds fixed +X model range")
    return {"body_rpy_deg": body.tolist(), "camera_rpy_deg": camera.tolist(),
            "allowed_max_abs_axis_deg": limit,
            "model": "fixed_yaw_zero_approximation_with_measured_attitude_gate"}


def frame_translation(body_world, body_local, recorded_origin=None, tolerance_m=0.05):
    inferred = finite_vector(body_world, "body world position") - finite_vector(body_local, "body local position")
    if recorded_origin is not None:
        error = float(np.linalg.norm(inferred - finite_vector(recorded_origin, "recorded origin")))
        if error > float(tolerance_m):
            raise ProtectionRequired("Local/world origin translation changed beyond tolerance")
    else:
        error = 0.0
    return inferred, error


def checked_motion_state(velocity, acceleration, max_speed, max_acceleration):
    v = finite_vector(velocity, "measured velocity")
    a = finite_vector(acceleration, "measured acceleration")
    if float(np.linalg.norm(v)) > float(max_speed) + 1e-9:
        raise ProtectionRequired("Measured initial velocity exceeds planner budget")
    if float(np.linalg.norm(a)) > float(max_acceleration) + 1e-9:
        raise ProtectionRequired("Measured initial acceleration exceeds planner budget")
    return v, a


def candidate_endpoints(position, planning):
    p = finite_vector(position, "candidate origin")
    axes = []
    for axis in ("x", "y", "z"):
        values = np.asarray(planning[f"candidate_steps_{axis}_m"], dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Candidate step axes must be nonempty finite arrays")
        axes.append(values)
    durations = planning["durations_s"]
    count = math.prod(len(axis) for axis in axes) * len(durations)
    if not durations or count > 60:
        raise ValueError("At most 60 endpoint-duration candidates are allowed")
    endpoints = [(p + np.asarray(delta)).tolist() for delta in itertools.product(*axes)]
    if planning.get("execution_mode") == "fixed_world_height_preview":
        anchor = float(planning["fixed_world_height_ned_m"])
        if not math.isfinite(anchor) or axes[2].tolist() != [0.0]:
            raise ValueError("Fixed-height execution requires one zero vertical candidate step")
        for endpoint in endpoints:
            endpoint[2] = anchor
    return endpoints


def world_height_to_local(world_height_ned_m, origin_translation):
    origin = finite_vector(origin_translation, "world/local origin")
    height = float(world_height_ned_m)
    if not math.isfinite(height):
        raise ProtectionRequired("World height must be finite")
    return height - float(origin[2])


def execution_sample(position, state, planning, workspace_min, workspace_max):
    """Actual self-state guard, independent of predicted polynomial feasibility."""
    p = finite_vector(position, "execution position")
    velocity = finite_vector(vector(state.kinematics_estimated.linear_velocity), "execution velocity")
    acceleration = finite_vector(vector(state.kinematics_estimated.linear_acceleration), "execution acceleration")
    speed, acceleration_norm = float(np.linalg.norm(velocity)), float(np.linalg.norm(acceleration))
    rpy = quaternion_rpy_deg(_quaternion_tuple(state.kinematics_estimated.orientation))
    margin = float(np.min(np.minimum(p - workspace_min, np.asarray(workspace_max) - p)))
    height_error = abs(float(p[2]) - float(planning["fixed_world_height_ned_m"]))
    record = {"world_ned_m": p.tolist(), "velocity_ned_m_s": velocity.tolist(),
              "acceleration_ned_m_s2": acceleration.tolist(), "speed_norm_m_s": speed,
              "acceleration_norm_m_s2": acceleration_norm, "actual_body_rpy_deg": rpy.tolist(),
              "query_state_timestamp_ns": int(state.timestamp), "workspace_margin_m": margin,
              "height_error_m": height_error,
              "over_planning_speed_budget": speed > planning["execution_speed_cap_m_s"] + 1e-9,
              "over_planning_acceleration_budget": acceleration_norm > planning["max_acceleration_m_s2"] + 1e-9}
    reasons = []
    if margin < planning["execution_guard_margin_m"]:
        reasons.append("ACTUAL_POSITION_ENTERED_PROTECTION_MARGIN")
    if height_error > planning["fixed_height_max_error_m"]:
        reasons.append("ACTUAL_HEIGHT_ERROR_EXCEEDED")
    if record["over_planning_speed_budget"]:
        reasons.append("ACTUAL_SPEED_BUDGET_EXCEEDED")
    if record["over_planning_acceleration_budget"]:
        reasons.append("ACTUAL_ACCELERATION_BUDGET_EXCEEDED")
    if float(np.max(np.abs(rpy))) > planning["attitude_limit_deg"]:
        reasons.append("ACTUAL_ATTITUDE_MODEL_LIMIT_EXCEEDED")
    record["protection_reasons"] = reasons
    return record


def bounded_reference_command(plan, measured_position, previous_command,
                              lookahead_s, position_gain, speed_cap, acceleration_cap, command_duration_s):
    """Track a positive-time reference; never silently alter the planner input."""
    duration = float(plan["trajectory"]["duration_s"])
    lookahead = min(float(lookahead_s), duration)
    if not math.isfinite(lookahead) or lookahead <= 0:
        raise ValueError("Reference lookahead must be positive")
    reference = evaluate_plan(plan, lookahead)
    measured = finite_vector(measured_position, "measured body position")
    previous = finite_vector(previous_command, "previous velocity command")
    speed, acceleration, dt = float(speed_cap), float(acceleration_cap), float(command_duration_s)
    if min(speed, acceleration, dt) <= 0 or not all(math.isfinite(x) for x in (speed, acceleration, dt)):
        raise ValueError("Command budgets must be finite and positive")
    if np.linalg.norm(previous) > speed + 1e-9:
        raise ProtectionRequired("Previous velocity command exceeds current budget")
    raw = reference["velocity"] + float(position_gain) * (reference["position"] - measured)
    clipped = raw * min(1.0, speed / max(float(np.linalg.norm(raw)), 1e-12))
    change = clipped - previous
    applied = previous + change * min(1.0, acceleration * dt / max(float(np.linalg.norm(change)), 1e-12))
    return {"reference_time_s": lookahead,
            "reference_position_world_ned_m": reference["position"].tolist(),
            "reference_velocity_ned_m_s": reference["velocity"].tolist(),
            "reference_acceleration_ned_m_s2": reference["acceleration"].tolist(),
            "raw_tracking_velocity_ned_m_s": raw.tolist(),
            "command_velocity_ned_m_s": applied.tolist(),
            "command_velocity_norm_m_s": float(np.linalg.norm(applied)),
            "velocity_norm_clipped": bool(np.linalg.norm(raw) > speed),
            "command_change_limited": bool(np.linalg.norm(change) > acceleration * dt),
            "command_change_norm_m_s": float(np.linalg.norm(applied - previous)),
            "command_change_reference_duration_s": dt,
            "command_budget_scope": "velocity RPC commands only; hover autopilot and physical acceleration are separately observed"}


def _quaternion_tuple(value):
    return (value.x_val, value.y_val, value.z_val, value.w_val)


class TrajectorySingleExperiment(SingleDroneExperiment):
    def __init__(self, policy, target_source, config_path, client=None, provider=None):
        if policy not in ("efficiency", "perception"):
            raise ValueError("policy must be efficiency or perception")
        if target_source not in ("geometry_oracle", "sensor_surface_estimate"):
            raise ValueError("Explicit target_source is required")
        super().__init__("visual", client=client, config_path=Path(config_path),
                         record_prefix=f"trajectory_{policy}_{target_source}")
        self.planning_mode = policy
        self.target_source = target_source
        self.planning = self.cfg["planning"]
        quality_mode=self.planning.get('image_center_reference_mode','surface_representative_point')
        if quality_mode not in ('surface_representative_point','rgb_detection_centroid_axial_plane'):
            raise ValueError('Unknown image-centering reference mode')
        if quality_mode=='rgb_detection_centroid_axial_plane' and target_source!='sensor_surface_estimate':
            raise ValueError('RGB centroid quality reference requires explicit sensor surface input')
        self.fixed_height_execution = self.planning.get("execution_mode") == "fixed_world_height_preview"
        self.active_xy_command = np.zeros(2)
        self.active_observation = None
        self.freshness_replans = 0
        self.guard_cfg = self.planning.get("execution_envelope", {})
        self.match_params = (validate_matched_start_config(self.cfg)
                             if self.cfg.get("initial_state_match", {}).get("enabled") else None)
        if self.match_params is not None and (not self.fixed_height_execution or not self.planning.get("continuous_goal_keepalive_s")):
            raise ValueError("Measured initial-state matching requires continuous fixed-height control")
        candidate_endpoints(self.cfg["observation_world_ned_m"], self.planning)
        positive_fields = ("max_acceleration_m_s2", "command_duration_s", "lookahead_s", "position_gain",
            "min_body_target_distance_m", "desired_geometry_body_center_m", "desired_surface_body_distance_m",
            "settle_speed_m_s", "settle_acceleration_m_s2", "settle_stable_s", "settle_timeout_s",
            "origin_translation_tolerance_m", "execution_speed_cap_m_s", "sample_dt_s", "max_rgb_depth_skew_s")
        if any(not math.isfinite(float(self.planning[name])) or self.planning[name] <= 0 for name in positive_fields):
            raise ValueError("Planning interface budgets must be finite and positive")
        if self.planning["command_duration_s"] > .5 or self.planning["execution_speed_cap_m_s"] > .8:
            raise ValueError("Planning interface command exceeds inspected development range")
        if self.fixed_height_execution:
            for name in ("execution_guard_margin_m", "fixed_height_max_error_m", "execution_sample_dt_s", "execution_ramp_acceleration_m_s2"):
                if not math.isfinite(float(self.planning[name])) or self.planning[name] <= 0:
                    raise ValueError(f"Invalid fixed-height parameter: {name}")
            if self.planning["execution_sample_dt_s"] > .1 or self.planning["execution_ramp_acceleration_m_s2"] > self.planning["max_acceleration_m_s2"]:
                raise ValueError("Execution sampling/ramp exceeds development budget")
            if not self.cfg["workspace_min_ned_m"][2] + self.planning["execution_guard_margin_m"] < self.planning["fixed_world_height_ned_m"] < self.cfg["workspace_max_ned_m"][2] - self.planning["execution_guard_margin_m"]:
                raise ValueError("Height anchor must lie inside the common protected workspace")
            keepalive = self.planning.get("continuous_goal_keepalive_s", 0)
            if keepalive and (not math.isfinite(keepalive) or not 0 < keepalive <= self.cfg["visual"]["frame_max_age_s"]):
                raise ValueError("Continuous goal lifetime must fit the observation freshness budget")
        if self.guard_cfg:
            if not self.fixed_height_execution or not self.match_params:
                raise ValueError("Execution envelope requires matched fixed-height control")
            for name in ("horizon_s", "position_reserve_m", "freshness_reserve_s", "minimum_command_s", "brake_timeout_s", "image_border_margin_fraction"):
                if not math.isfinite(float(self.guard_cfg[name])) or self.guard_cfg[name] <= 0:
                    raise ValueError(f"Invalid execution envelope: {name}")
            if self.guard_cfg["horizon_s"] < self.planning["continuous_goal_keepalive_s"] or self.guard_cfg["minimum_command_s"] > self.planning["execution_sample_dt_s"]:
                raise ValueError("Envelope must cover maximum lease; command quantum must fit samples")
            if not self.guard_cfg["image_border_margin_fraction"] < 1 or not isinstance(self.guard_cfg["max_freshness_replans"],int) or not 1 <= self.guard_cfg["max_freshness_replans"] <= 20:
                raise ValueError("Invalid footprint margin or bounded replan allowance")
        from timestamped_target_provider import TimestampedTargetProvider
        self.timestamped_camera = provider or TimestampedTargetProvider(
            self.client, coordinate_frame="VEHICLE_LOCAL_NED",
            max_rgb_depth_skew_s=self.planning["max_rgb_depth_skew_s"],
            max_observation_age_s=self.cfg["visual"]["frame_max_age_s"])
        self.origin_translation = None
        self.report.update(policy=policy, target_source=target_source,
                           episode_id=self.folder.name, session_id=self.folder.name,
                           experiment_scope="single_drone_planner_interface_development",
                           research_result_ready=False, planning_records=[], settling_records=[], control_iterations=[],
                           protective_return_reason=None,
                           planning_model="finite_candidate_fixed_yaw_sampled_geometry",
                           target_estimate_semantics=("simulator_balloon_center_oracle" if target_source == "geometry_oracle"
                                                      else "visible_surface_representative_point_not_balloon_center"))
        self.report["truth_access"].update(
            target_pose=("preflight_and_explicit_geometry_oracle_planning" if target_source == "geometry_oracle"
                         else "preflight_only; never used to repair missing sensor target"),
            vehicle_pose="world/local_origin_calibration_staging_return_safety_and_planner_body_state",
            depth_camera="single_RPC_RGB_DepthPerspective_surface_estimation",
            actual_camera_pose="ImageResponse_vehicle_local_NED_translated_with_explicit_recorded_origin")
        self.report["truth_access"]["self_kinematics"] = "SimpleFlight kinematics_estimated currently exposes simulator truth; explicitly allowed self state"
        self.report["time_measurement_scope"] = ("ClockSpeed=1 development freshness proxy: simulator render-readback/query age plus host processing delay; "
            "not an exact exposure-to-command or kinematics-measurement latency")
        self.report["limitations"] = ["No continuous safety proof", "No occlusion prediction",
            "Finite candidate minimum only", "Hover/return uses inherited controller, not planner-certified fallback",
            "Surface-point estimate cannot be compared as balloon-center estimation accuracy",
            "Two policies use the same candidates and constraints; only image-center cost weight differs"]
        self.report["body_execution_mode"] = ("moveByVelocityZAsync_constant_world_height" if self.fixed_height_execution else "moveByVelocityAsync_hover_separated_preview")
        self.report["reference_execution_scope"] = "Positive-time preview feedback; not time-indexed execution of the quintic segment"
        self.report["ai_assistance"] = "Controller and reporting authored with AI assistance"
        self.report["goal_keepalive_records"] = []
        self.report["continuous_goal_scope"] = (
            "Background same-mode RPC refreshes previous XY velocity and fixed local Z during sensing/planning; "
            "not proof of runtime firmware timeout behavior")
        for name in ("trajectory_single_experiment.py", "perception_trajectory_planner.py", "timestamped_target_provider.py"):
            self.report["source_sha256"][name] = hashlib.sha256((ROOT / "python" / name).read_bytes()).hexdigest()
        self.report["configuration_sha256"] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        self.report["condition_id"] = self.cfg.get("condition_id")
        self.report['image_center_reference_mode']=quality_mode
        if quality_mode=='rgb_detection_centroid_axial_plane':
            self.report['source_sha256']['trajectory_rgb_quality_reference.py']=hashlib.sha256(
                (ROOT/'python/trajectory_rgb_quality_reference.py').read_bytes()).hexdigest()
        if self.match_params is not None:
            self.report["source_sha256"]["matched_trajectory_start.py"] = hashlib.sha256(
                (ROOT / "python" / "matched_trajectory_start.py").read_bytes()).hexdigest()
        if self.guard_cfg:
            self.report["source_sha256"]["trajectory_execution_guard.py"] = hashlib.sha256(
                (ROOT / "python" / "trajectory_execution_guard.py").read_bytes()).hexdigest()
            self.report["execution_envelope_scope"] = "Shared measured-box footprint and command delay/braking envelope; model approximation, no physical safety proof"
            self.report["freshness_replan_events"] = []
            self.report["execution_guard_samples"] = []

    def _bystander_preflight(self):
        result = super()._bystander_preflight()
        position, speed, state = self.state()
        if speed > self.planning["settle_speed_m_s"]:
            raise ProtectionRequired("Origin calibration requires stationary preflight body")
        local = vector(state.kinematics_estimated.position)
        inferred, _ = frame_translation(position, local)
        self.origin_translation = inferred.copy()
        self.report["origin_calibration"] = {"world_from_vehicle_local_translation_ned_m": inferred.tolist(),
            "body_world_ned_m": position.tolist(), "body_local_ned_m": local,
            "self_state_query_timestamp_ns": int(state.timestamp), "speed_m_s": speed,
            "method": "stationary_ground_preflight_world_object_pose_minus_vehicle_local_self_state",
            "time_scope": "sequential near-stationary queries, not simultaneous exposure"}
        return result

    def _record_path(self, position):
        if self.last_position is not None:
            self.path_m += float(np.linalg.norm(position - self.last_position))
        self.last_position = position.copy()

    def _save_guard_state(self, sample, reason):
        sample.update(host_monotonic_s=time.monotonic(), guard_query_reason=reason)
        self.report["execution_guard_samples"].append(sample)
        with (self.folder / "execution_guard_samples.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, allow_nan=False) + "\n")

    def _boundary_command(self, position, desired_xy, previous_xy, dt):
        margin = self.planning["execution_guard_margin_m"]
        try:
            decision = boundary_command(position, desired_xy, previous_xy,
                np.asarray(self.cfg["workspace_min_ned_m"]) + margin,
                np.asarray(self.cfg["workspace_max_ned_m"]) - margin,
                self.planning["execution_speed_cap_m_s"], self.planning["execution_ramp_acceleration_m_s2"],
                dt, self.guard_cfg["horizon_s"], self.guard_cfg["position_reserve_m"])
        except ValueError as error:
            raise ProtectionRequired("Execution boundary envelope: " + str(error)) from error
        if not decision["accepted"]:
            self.report.setdefault("rejected_boundary_envelopes", []).append(decision)
            raise ProtectionRequired(decision["reason"])
        return decision

    def _freshness_brake_and_reobserve(self, reason, age):
        """Discard target and decelerate with self state, retaining all evidence."""
        self.active_observation = None
        self.freshness_replans += 1
        event = {"reason": reason, "trigger_age_proxy_s": age, "status": "BRAKING",
                 "started_host_monotonic_s": time.monotonic(), "target_used_for_braking": False,
                 "planner_certified_safe_fallback": False}
        self.report["freshness_replan_events"].append(event)
        entry = {"reason": "FRESHNESS_REPLAN_SELF_STATE_BRAKE", "samples": [],
                 "fallback_certified_safe": False, "status": "RUNNING"}
        self.report["settling_records"].append(entry)
        stable = None
        local_z = world_height_to_local(self.planning["fixed_world_height_ned_m"], self.origin_translation)
        while time.monotonic()-event["started_host_monotonic_s"] <= self.guard_cfg["brake_timeout_s"]:
            position, _, state = self.state(check_workspace=True)
            sample = execution_sample(position, state, self.planning,
                self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
            sample["host_monotonic_s"] = time.monotonic()
            entry["samples"].append(sample)
            self._record_path(position)
            with (self.folder / "brake_samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(sample, allow_nan=False)+"\n")
            if sample["protection_reasons"]:
                entry["status"] = event["status"] = "PROTECTED_RETURN"
                raise ProtectionRequired("Self-state brake: "+";".join(sample["protection_reasons"]))
            stopped = (np.linalg.norm(self.active_xy_command) <= 1e-9 and
                sample["speed_norm_m_s"] <= self.planning["settle_speed_m_s"] and
                sample["acceleration_norm_m_s2"] <= self.planning["settle_acceleration_m_s2"])
            stable = (stable if stable is not None else time.monotonic()) if stopped else None
            if stable is not None and time.monotonic()-stable >= self.planning["settle_stable_s"]:
                entry["status"] = event["status"] = "BRAKED_READY_TO_REOBSERVE"
                if self.freshness_replans > self.guard_cfg["max_freshness_replans"]:
                    raise ProtectionRequired("Freshness replan allowance exhausted after self-state brake")
                return position, state
            dt = self.planning["execution_sample_dt_s"]
            envelope = self._boundary_command(position, [0.,0.], self.active_xy_command, dt)
            applied = np.asarray(envelope["command_velocity_xy_m_s"])
            rpc_started = time.monotonic()
            self.client.moveByVelocityZAsync(float(applied[0]),float(applied[1]),local_z,duration=dt,
                yaw_mode=airsim.YawMode(False,0),vehicle_name=self.vehicle).join()
            command = {"reason": "SELF_STATE_BRAKE_WITHOUT_TARGET", "phase": "APPROACH",
                "command_velocity_xy_m_s": applied.tolist(), "requested_duration_s": dt,
                "join_called": True, "rpc_elapsed_host_s": time.monotonic()-rpc_started,
                "boundary_envelope": envelope, "target_observation_live": False}
            self.report["goal_keepalive_records"].append(command)
            self._on_brake_command(command, rpc_started, time.monotonic())
            with (self.folder/"goal_keepalive_records.jsonl").open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(command,allow_nan=False)+"\n")
            self.active_xy_command = applied.copy()
        entry["status"] = event["status"] = "BRAKE_TIMEOUT"
        raise ProtectionRequired("Self-state brake timed out")

    def _on_brake_command(self, command, started, finished):
        """Optional trace hook; legacy fixed-preview behavior is unchanged."""

    def _initial_reference_command(self, plan, position, previous, record):
        return bounded_reference_command(plan, position, previous,
            self.planning["lookahead_s"], self.planning["position_gain"],
            self.planning["execution_speed_cap_m_s"], self.planning["max_acceleration_m_s2"],
            self.planning["command_duration_s"])

    def _maintain_fixed_goal(self, reason):
        duration = self.planning.get("continuous_goal_keepalive_s", 0)
        if not self.fixed_height_execution or not duration:
            return
        xy = np.asarray(self.active_xy_command, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or np.linalg.norm(xy) > self.planning["execution_speed_cap_m_s"] + 1e-9:
            raise ProtectionRequired("Invalid continuous XY command")
        envelope = None
        if self.planning.get("execution_envelope") and getattr(self, "phase", "APPROACH") == "APPROACH":
            if self.active_observation is not None:
                window = command_window(self.active_observation.age_at(), self.cfg["visual"]["frame_max_age_s"], duration,
                    self.guard_cfg["minimum_command_s"], self.guard_cfg["freshness_reserve_s"])
                if not window["accepted"]:
                    self._freshness_brake_and_reobserve("LEASE_WINDOW_EXHAUSTED", window["age_proxy_s"])
                    xy = self.active_xy_command.copy()
                else:
                    duration = window["duration_s"]
            elif np.linalg.norm(xy) > 1e-9:
                raise ProtectionRequired("Moving goal has no live target observation")
            position, _, state = self.state(check_workspace=True)
            sample = execution_sample(position, state, self.planning,
                self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
            self._save_guard_state(sample, "keepalive_pre_dispatch")
            if sample["protection_reasons"]:
                raise ProtectionRequired(";".join(sample["protection_reasons"]))
            envelope = self._boundary_command(position, xy, self.active_xy_command, self.planning["execution_sample_dt_s"])
            xy = np.asarray(envelope["command_velocity_xy_m_s"])
            self.active_xy_command = xy.copy()
            if self.active_observation is not None:
                final_window = command_window(self.active_observation.age_at(),self.cfg["visual"]["frame_max_age_s"],duration,
                    self.guard_cfg["minimum_command_s"],self.guard_cfg["freshness_reserve_s"])
                if not final_window["accepted"]:
                    self._freshness_brake_and_reobserve("KEEPALIVE_STATE_QUERY_CONSUMED_FRESHNESS",final_window["age_proxy_s"])
                    return self._maintain_fixed_goal(reason+"_after_self_state_brake")
                duration = final_window["duration_s"]
        local_z = world_height_to_local(self.planning["fixed_world_height_ned_m"], self.origin_translation)
        started = time.monotonic()
        self.client.moveByVelocityZAsync(float(xy[0]), float(xy[1]), local_z, duration=duration,
            yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle)
        entry = {"reason": reason, "command_velocity_xy_m_s": xy.tolist(),
            "world_height_setpoint_ned_m": self.planning["fixed_world_height_ned_m"],
            "local_height_setpoint_ned_m": local_z, "requested_duration_s": duration,
            "host_monotonic_s": started, "rpc_enqueue_return_host_s": time.monotonic(),
            "join_called": False, "scope": "previous command maintained by background RPC; may be superseded"}
        entry["phase"] = getattr(self, "phase", "APPROACH")
        if envelope is not None:
            entry.update(boundary_envelope=envelope, target_observation_live=self.active_observation is not None)
        matching = entry["phase"] == "INITIAL_STATE_MATCH" and "initial_state_match" in self.report
        if matching:
            self.report["initial_state_match"]["command_records"].append(entry)
        else:
            self.report["goal_keepalive_records"].append(entry)
        log_name = "initial_state_commands.jsonl" if matching else "goal_keepalive_records.jsonl"
        with (self.folder / log_name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")

    def _settle(self, record_destination=None):
        self.client.cancelLastTask(vehicle_name=self.vehicle)
        if not self.fixed_height_execution:
            self.client.hoverAsync(vehicle_name=self.vehicle).join()
        start, stable = time.monotonic(), None
        records = []
        entry = {"started_at": datetime.now().astimezone().isoformat(), "samples": records,
                 "fallback_certified_safe": False, "status": "RUNNING"}
        destination = self.report["settling_records"] if record_destination is None else record_destination
        destination.append(entry)
        while time.monotonic() - start <= self.planning["settle_timeout_s"]:
            if self.fixed_height_execution:
                local_z = world_height_to_local(self.planning["fixed_world_height_ned_m"], self.origin_translation)
                if self.planning.get("continuous_goal_keepalive_s", 0):
                    self._maintain_fixed_goal("initial_settle")
                else:
                    self.client.moveByVelocityZAsync(0.0, 0.0, local_z, duration=self.planning["command_duration_s"],
                        yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle).join()
            position, _, state = self.state(check_workspace=True)
            self._record_path(position)
            v = finite_vector(vector(state.kinematics_estimated.linear_velocity), "settling measured velocity")
            a = finite_vector(vector(state.kinematics_estimated.linear_acceleration), "settling measured acceleration")
            speed, acceleration = float(np.linalg.norm(v)), float(np.linalg.norm(a))
            records.append({"host_monotonic_s": time.monotonic(), "world_ned_m": position.tolist(),
                "velocity_ned_m_s": v.tolist(), "acceleration_ned_m_s2": a.tolist(),
                "speed_m_s": speed, "acceleration_norm_m_s2": acceleration,
                "over_planning_speed_budget": speed > self.planning["execution_speed_cap_m_s"],
                "over_planning_acceleration_budget": acceleration > self.planning["max_acceleration_m_s2"]})
            ready = speed <= self.planning["settle_speed_m_s"] and acceleration <= self.planning["settle_acceleration_m_s2"]
            if self.fixed_height_execution:
                ready = ready and abs(position[2] - self.planning["fixed_world_height_ned_m"]) <= self.planning["fixed_height_max_error_m"]
            stable = (stable if stable is not None else time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable >= self.planning["settle_stable_s"]:
                entry.update(status="SETTLED", elapsed_s=time.monotonic() - start)
                return position, state
            time.sleep(0.10)
        entry.update(status="TIMEOUT", elapsed_s=time.monotonic() - start)
        raise ProtectionRequired("Measured body state did not settle before timeout")

    def _prepare_matched_start(self):
        """Pre-approach positioning; performance timing starts after acceptance."""
        self.phase = "INITIAL_STATE_MATCH"
        started = time.monotonic()
        desired = np.asarray(self.cfg["observation_world_ned_m"], dtype=float)
        entry = {"status": "RUNNING", "specification": self.cfg["initial_state_match"],
            "desired_world_ned_m": desired.tolist(), "samples": [], "accepted_state": None,
            "command_records": [], "warmup_settling_records": [], "protective_reason": None,
            "scope": "preparation excluded from approach duration and path",
            "started_host_monotonic_s": started}
        self.report["initial_state_match"] = entry
        gate = MatchedStartGate(self.match_params, start_time_s=started)
        previous_dispatch = started
        try:
            self._settle(record_destination=entry["warmup_settling_records"])
            warmup = [s for r in entry["warmup_settling_records"] for s in r["samples"]]
            if any(s["over_planning_speed_budget"] or s["over_planning_acceleration_budget"] for s in warmup):
                raise ProtectionRequired("Initial-state warmup exceeded actual motion budget")
            while time.monotonic() - started < self.match_params.timeout_s:
                self._maintain_fixed_goal("matched_start_positioning")
                position, _, state = self.state(check_workspace=True)
                now = time.monotonic()
                check = initial_state_check(position, vector(state.kinematics_estimated.linear_velocity),
                    vector(state.kinematics_estimated.linear_acceleration),
                    _quaternion_tuple(state.kinematics_estimated.orientation), desired, self.match_params)
                check.update(host_monotonic_s=now, query_state_timestamp_ns=int(state.timestamp),
                    world_ned_m=position.tolist(), actual_body_rpy_deg=check["body_rpy_deg"],
                    rejection_reasons=list(check["reasons"]),
                    previous_command_velocity_xy_m_s=self.active_xy_command.tolist())
                if np.linalg.norm(self.active_xy_command) > 1e-9:
                    check["accepted"] = False
                    check["rejection_reasons"].append("POSITIONING_COMMAND_NOT_ZERO")
                result = gate.update(check, now)
                check.update(result)
                entry["samples"].append(check)
                with (self.folder / "initial_state_samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(check, allow_nan=False) + "\n")
                physical = execution_sample(position, state, self.planning,
                    self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
                if check["protection_required"] or physical["protection_reasons"]:
                    raise ProtectionRequired("Initial-state matching protection: " +
                        ";".join(check["rejection_reasons"] + physical["protection_reasons"]))
                if result["gate_accepted"]:
                    entry.update(status="ACCEPTED", accepted_state=check,
                                 continuous_stable_s=result["continuous_stable_s"])
                    self._maintain_fixed_goal("matched_start_accepted_zero_xy")
                    return
                if result["timed_out"]:
                    break
                command = reposition_xy_command(position, desired, self.active_xy_command,
                    now - previous_dispatch, self.match_params)
                check["next_positioning_command"] = command
                self.active_xy_command = np.asarray(command["command_velocity_xy_m_s"], dtype=float)
                self._maintain_fixed_goal("matched_start_bounded_reposition")
                previous_dispatch = time.monotonic()
                time.sleep(self.match_params.command_dt_s)
            raise ProtectionRequired("Initial-state matching timed out before continuous acceptance")
        except (ProtectionRequired, ValueError) as error:
            entry.update(status="PROTECTED_RETURN", protective_reason=str(error))
            self.client.cancelLastTask(vehicle_name=self.vehicle)
            raise ProtectionRequired(str(error)) from error
        finally:
            entry["elapsed_s"] = time.monotonic() - started

    def _sample_timestamped(self):
        self._maintain_fixed_goal("before_capture_and_planning")
        if self.fixed_height_execution:
            pre_capture_position, _, pre_capture_state = self.state(check_workspace=True)
            pre_capture_host = time.monotonic()
        observation = self.timestamped_camera.capture(self.vehicle)
        position, _, state = self.state(check_workspace=True)
        self._record_path(position)
        age_error = None
        try:
            age = observation.age_at()
        except ValueError as error:
            age, age_error = None, str(error)
        track = self.tracker.update(observation.valid and age is not None
                                    and age <= self.cfg["visual"]["frame_max_age_s"])
        row = {"time": datetime.now().astimezone().isoformat(), "world_ned_m": position.tolist(),
            "detected": bool(observation.detected), "truncated": bool(observation.truncated),
            "tracking_state": track.state.value, "candidate_count": observation.candidate_count,
            "shape_score_not_probability": observation.confidence,
            "error_xy": observation.normalised_error_xy, "depth_m": observation.depth_m,
            "frame_age_s": age, "timestamped_observation": observation.metadata(),
            "provider_valid": bool(observation.valid), "target_source": self.target_source,
            "freshness_proxy_error": age_error, "episode_id": self.folder.name,
            "frame_id": f"{self.folder.name}:frame:{len(self.report['observations']) + 1:04d}"}
        if self.fixed_height_execution:
            row["pre_capture_body_state"] = execution_sample(pre_capture_position, pre_capture_state,
                self.planning, self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
            row["pre_capture_body_state"]["host_monotonic_s"] = pre_capture_host
            row["planning_input_body_state"] = execution_sample(position, state,
                self.planning, self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
            row["planning_input_body_state"]["host_monotonic_s"] = time.monotonic()
            row["capture_query_elapsed_host_s"] = row["planning_input_body_state"]["host_monotonic_s"] - pre_capture_host
            if observation.state_sample is not None:
                capture_state = observation.state_sample
                row["provider_query_motion_state"] = {
                    "velocity_ned_m_s": vector(capture_state.kinematics_estimated.linear_velocity),
                    "acceleration_ned_m_s2": vector(capture_state.kinematics_estimated.linear_acceleration),
                    "query_state_timestamp_ns": int(capture_state.timestamp)}
        self.report["observations"].append(row)
        if observation.image is not None:
            raw_path = self.folder / f"raw_frame_{len(self.report['observations']):04d}.png"
            if not cv2.imwrite(str(raw_path), observation.image):
                raise RuntimeError("Could not persist original RGB frame")
            row.update(raw_image=str(raw_path), raw_image_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest())
        if observation.depth is not None:
            depth_path = self.folder / f"raw_depth_{len(self.report['observations']):04d}.npy"
            np.save(str(depth_path), observation.depth, allow_pickle=False)
            row.update(raw_depth=str(depth_path), raw_depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
                       raw_depth_semantics="AirSim_DepthPerspective_camera_radial_range_metres")
        return observation, row, position, state

    def _planning_inputs(self, observation, row, position, state):
        if row["frame_age_s"] is None:
            raise ProtectionRequired("Invalid simulator/host freshness proxy: " + str(row.get("freshness_proxy_error")))
        try:
            row["frame_age_s"] = observation.age_at()
        except ValueError as error:
            raise ProtectionRequired("Invalid freshness proxy at planning input check: " + str(error)) from error
        if not observation.valid:
            raise ProtectionRequired("Timestamped RGB/depth observation rejected: " + str(observation.rejection_reasons))
        if row["frame_age_s"] > self.cfg["visual"]["frame_max_age_s"]:
            raise ProtectionRequired("Observation expired before planning")
        inferred, translation_error = frame_translation(position,
            vector(state.kinematics_estimated.position), self.origin_translation,
            self.planning["origin_translation_tolerance_m"])
        if self.origin_translation is None:
            self.origin_translation = inferred.copy()
            self.report["origin_calibration"] = {"world_from_vehicle_local_translation_ned_m": inferred.tolist(),
                "body_world_ned_m": position.tolist(),
                "body_local_ned_m": vector(state.kinematics_estimated.position),
                "method": "world_object_pose_minus_same_base_state_local_position_after_low_motion_settle"}
        camera_local = finite_vector(observation.camera_position_local_ned_m, "actual camera local position")
        camera_world = camera_local + self.origin_translation
        attitude = require_fixed_forward_attitude(_quaternion_tuple(state.kinematics_estimated.orientation),
            observation.camera_orientation_xyzw, self.planning["attitude_limit_deg"])
        if observation.state_sample is not None:
            capture_attitude = require_fixed_forward_attitude(
                _quaternion_tuple(observation.state_sample.kinematics_estimated.orientation),
                observation.camera_orientation_xyzw, self.planning["attitude_limit_deg"])
            row["capture_state_attitude"] = capture_attitude
        velocity, acceleration = checked_motion_state(vector(state.kinematics_estimated.linear_velocity),
            vector(state.kinematics_estimated.linear_acceleration), self.planning["execution_speed_cap_m_s"],
            self.planning["max_acceleration_m_s2"])
        if self.target_source == "geometry_oracle":
            target = finite_vector(vector(self.client.simGetObjectPose(self.cfg["target"]).position), "explicit oracle target")
        else:
            target = finite_vector(observation.estimate_world(
                world_from_local_translation_ned_m=self.origin_translation), "sensor surface target")
        row.update(actual_camera_world_ned_m=camera_world.tolist(),
                   actual_camera_offset_from_body_ned_m=(camera_world - position).tolist(),
                   origin_translation_error_m=translation_error, attitude_gate=attitude,
                   measured_velocity_ned_m_s=velocity.tolist(), measured_acceleration_ned_m_s2=acceleration.tolist(),
                   target_estimate_world_ned_m=target.tolist(), target_velocity_estimate_ned_m_s=[0, 0, 0],
                   target_velocity_model="zero_velocity_static_target_surface_snapshot")
        supports = [[0., 0., 0.]]
        if self.planning.get("execution_envelope"):
            try:
                footprint = projected_bbox_support_offsets(observation.bbox_xywh,
                    (observation.image.shape[1], observation.image.shape[0]), target-camera_world)
                supports = footprint["offsets_ned_m"]
            except (ValueError, TypeError) as error:
                raise ProtectionRequired("Invalid observed target footprint: "+str(error)) from error
            row.update(target_support_offsets_ned_m=supports,
                       footprint_anchor_pixel_xy=footprint["anchor_pixel_xy"],
                       footprint_anchor_scope=footprint["anchor_scope"],
                footprint_scope="observed bbox backprojected at representative axial depth; not true 3D hull")
            self.active_observation = observation
        desired = self.planning["desired_geometry_body_center_m" if self.target_source == "geometry_oracle"
                                else "desired_surface_body_distance_m"]
        height, width = observation.image.shape[:2]
        quality_offset=[0.,0.,0.]
        if self.planning.get('image_center_reference_mode')=='rgb_detection_centroid_axial_plane':
            from trajectory_rgb_quality_reference import rgb_quality_reference
            try:
                quality=rgb_quality_reference(target,camera_world,observation.centre_px,(width,height))
            except (ValueError,TypeError) as error:
                raise ProtectionRequired('Invalid RGB centering reference: '+str(error)) from error
            row['image_center_reference']=quality
            quality_offset=quality['offset_from_surface_ned_m']
        guard_margin = self.planning["execution_guard_margin_m"] if self.fixed_height_execution else 0.0
        params = PlanningParams(durations=tuple(self.planning["durations_s"]),
            sample_dt_s=self.planning["sample_dt_s"], position_min=tuple(np.asarray(self.cfg["workspace_min_ned_m"]) + guard_margin),
            position_max=tuple(np.asarray(self.cfg["workspace_max_ned_m"]) - guard_margin),
            max_speed_mps=self.planning["execution_speed_cap_m_s"],
            max_acceleration_mps2=self.planning["max_acceleration_m_s2"],
            image_border_margin_fraction=self.planning.get("execution_envelope", {}).get("image_border_margin_fraction", 0.),
            horizontal_fov_deg=90.0, vertical_fov_deg=math.degrees(2 * math.atan(height / width)),
            camera_offset_ned_m=tuple(camera_world - position),
            min_target_distance_m=self.planning["min_body_target_distance_m"],
            desired_observation_distance_m=desired,
            max_observation_age_s=self.cfg["visual"]["frame_max_age_s"])
        return dict(position=position, velocity=velocity, acceleration=acceleration,
            target_position=target, target_velocity=[0, 0, 0],
            candidate_endpoints=candidate_endpoints(position, self.planning), params=params,
            obstacles=self.planning.get("sphere_obstacles", []), observation_age_s=observation.age_at(),
            target_support_offsets=supports,
            image_center_reference_offset_ned_m=quality_offset,
            mode=self.planning_mode)

    def _persist_planning_record(self, record):
        self.report["planning_records"].append(record)
        with (self.folder / "planning_records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    def _start_iteration(self):
        sequence = len(self.report["control_iterations"])
        item = {"episode_id": self.folder.name, "iteration_id": f"{self.folder.name}:iteration:{sequence:04d}",
                "sequence": sequence, "status": "STARTED", "started_host_monotonic_s": time.monotonic(),
                "planned_frame_id": f"{self.folder.name}:frame:{len(self.report['observations']) + 1:04d}"}
        self.report["control_iterations"].append(item)
        with (self.folder / "control_iterations.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        return item

    def _execute_fixed_height(self, command, record, start_state, observation=None):
        local_z = world_height_to_local(self.planning["fixed_world_height_ned_m"], self.origin_translation)
        command.update(body_execution_mode="moveByVelocityZAsync_constant_world_height",
                       requested_preview_velocity_ned_m_s=command["command_velocity_ned_m_s"],
                       world_height_setpoint_ned_m=self.planning["fixed_world_height_ned_m"],
                       local_height_setpoint_ned_m=local_z,
                       command_budget_scope="XY velocity ramps; Z is position control; actual 3D motion separately guarded")
        desired = np.asarray(command["command_velocity_ned_m_s"], dtype=float)[:2]
        previous = np.asarray(vector(start_state.kinematics_estimated.linear_velocity), dtype=float)[:2]
        guarded = bool(self.planning.get("execution_envelope"))
        if guarded and observation is None:
            raise ProtectionRequired("Guarded target execution requires a timestamped observation")
        if guarded:
            previous = self.active_xy_command.copy()
        command["command_velocity_ned_m_s"] = [float(desired[0]), float(desired[1]), 0.0]
        command["command_velocity_norm_m_s"] = float(np.linalg.norm(desired))
        record["execution_samples"] = []
        remaining = float(self.planning["command_duration_s"])
        while remaining > 1e-8:
            dt = min(remaining, self.planning["execution_sample_dt_s"])
            if guarded and dt < self.guard_cfg["minimum_command_s"]:
                break
            if observation is not None:
                age = observation.age_at()
                record["latest_execution_age_proxy_s"] = age
                if guarded:
                    window = command_window(age,self.cfg["visual"]["frame_max_age_s"],dt,
                        self.guard_cfg["minimum_command_s"],self.guard_cfg["freshness_reserve_s"])
                    record.setdefault("freshness_windows",[]).append(window)
                    if not window["accepted"]:
                        position,state = self._freshness_brake_and_reobserve("EXECUTION_WINDOW_EXHAUSTED",age)
                        record["execution_status"] = "SELF_STATE_BRAKE_AND_REOBSERVE"
                        return position,state
                    dt = window["duration_s"]
                if age > self.cfg["visual"]["frame_max_age_s"]:
                    record["execution_status"] = "PROTECTED_STALE_DURING_EXECUTION"
                    self.client.cancelLastTask(vehicle_name=self.vehicle)
                    raise ProtectionRequired("Observation expired during execution")
            delta = desired - previous
            ramp_cap = self.planning["execution_ramp_acceleration_m_s2"] * dt
            applied = previous + delta * min(1.0, ramp_cap / max(float(np.linalg.norm(delta)), 1e-12))
            envelope = None
            if guarded:
                command_position,_,command_state = self.state(check_workspace=True)
                guard_sample = execution_sample(command_position,command_state,self.planning,
                    self.cfg["workspace_min_ned_m"],self.cfg["workspace_max_ned_m"])
                self._save_guard_state(guard_sample,"segment_pre_dispatch")
                if guard_sample["protection_reasons"]:
                    raise ProtectionRequired(";".join(guard_sample["protection_reasons"]))
                envelope = self._boundary_command(command_position,desired,previous,dt)
                applied = np.asarray(envelope["command_velocity_xy_m_s"])
                # Recheck after state RPC. Its elapsed time consumes the same lease.
                window = command_window(observation.age_at(),self.cfg["visual"]["frame_max_age_s"],dt,
                    self.guard_cfg["minimum_command_s"],self.guard_cfg["freshness_reserve_s"])
                record["freshness_windows"].append(window)
                if not window["accepted"] or window["duration_s"] < dt-1e-9:
                    position,state = self._freshness_brake_and_reobserve("PRE_DISPATCH_WINDOW_CHANGED",window["age_proxy_s"])
                    record["execution_status"] = "SELF_STATE_BRAKE_AND_REOBSERVE"
                    return position,state
            rpc_started = time.monotonic()
            self.client.moveByVelocityZAsync(float(applied[0]), float(applied[1]), local_z, duration=dt,
                yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle).join()
            rpc_elapsed = time.monotonic()-rpc_started
            if guarded:
                self.active_xy_command = applied.copy()
            position, _, state = self.state(check_workspace=True)
            sample = execution_sample(position, state, self.planning, self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
            sample.update(command_velocity_xy_m_s=applied.tolist(), command_duration_s=dt,
                          host_monotonic_s=time.monotonic(),rpc_elapsed_host_s=rpc_elapsed)
            if envelope is not None:
                sample["boundary_envelope"] = envelope
                sample["observation_age_after_rpc_and_state_s"] = observation.age_at()
            record["execution_samples"].append(sample)
            self._record_path(position)
            with (self.folder / "execution_samples.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"plan_id": record["plan_id"], "sample": sample}, allow_nan=False) + "\n")
            if sample["protection_reasons"]:
                record["execution_status"] = "PROTECTED_DURING_EXECUTION"
                self.client.cancelLastTask(vehicle_name=self.vehicle)
                raise ProtectionRequired(";".join(sample["protection_reasons"]))
            previous = applied
            remaining -= dt
            if guarded and observation.age_at() >= self.cfg["visual"]["frame_max_age_s"]-self.guard_cfg["freshness_reserve_s"]:
                position,state = self._freshness_brake_and_reobserve("RPC_OR_QUERY_CONSUMED_FRESHNESS",observation.age_at())
                record["execution_status"] = "SELF_STATE_BRAKE_AND_REOBSERVE"
                return position,state
        self.active_xy_command = previous.copy()
        self._maintain_fixed_goal("after_execution_before_reporting")
        record["executed_body_state"] = record["execution_samples"][-1]
        return position, state

    def approach(self):
        self.phase = "APPROACH"
        self.path_m = 0.0
        self.last_position = self.state(check_workspace=True)[0]
        matching_enabled = self.cfg.get("initial_state_match", {}).get("enabled", False)
        start = None if matching_enabled else time.monotonic()
        deadline = None if start is None else start + self.cfg["visual"]["max_duration_s"]
        previous_command = np.zeros(3)
        unsafe_execution_error = False
        try:
            if matching_enabled:
                self._prepare_matched_start()
                self.phase = "APPROACH"
                self.last_position = self.state(check_workspace=True)[0]
                self.path_m = 0.0
                start = time.monotonic()
                deadline = start + self.cfg["visual"]["max_duration_s"]
            while time.monotonic() < deadline:
                iteration = self._start_iteration()
                if not self.fixed_height_execution or (not self.report["observations"] and not matching_enabled):
                    self._settle()
                observation, row, position, state = self._sample_timestamped()
                iteration["frame_id"] = row.get("frame_id")
                row["iteration_id"] = iteration["iteration_id"]
                if self.planning.get("execution_envelope") and observation.valid and observation.age_at() > self.cfg["visual"]["frame_max_age_s"]-self.guard_cfg["freshness_reserve_s"]:
                    self._freshness_brake_and_reobserve("CAPTURE_CONSUMED_FRESHNESS",observation.age_at())
                    iteration["status"] = "FRESH_OBSERVATION_REQUIRED_AFTER_SELF_STATE_BRAKE"
                    continue
                inputs = self._planning_inputs(observation, row, position, state)
                if terminal_success(self.report["observations"], self.cfg["visual"]):
                    self.report["approach_stop_reason"] = "common_fresh_aligned_surface_standoff_criterion"
                    iteration["status"] = "TERMINAL_CRITERION_MET"
                    break
                if row["tracking_state"] != "DETECTED":
                    row["control_reason"] = "WAIT_FOR_FRESH_TRACK_CONFIRMATION"
                    iteration["status"] = "AWAITING_TRACK_CONFIRMATION"
                    continue
                if len(self.report["planning_records"]) >= self.planning.get("max_interface_planning_steps", 5000):
                    self.report["approach_stop_reason"] = "configured_interface_step_limit_reached"
                    iteration["status"] = "INTERFACE_STEP_LIMIT_REACHED"
                    break
                planned_at = time.monotonic()
                step_index = len(self.report["planning_records"])
                plan_id = f"{self.report.get('episode_id', 'test_episode')}:plan:{step_index:04d}"
                iteration.update(plan_id=plan_id, status="PLANNING_STARTED")
                plan = plan_trajectory(**inputs)
                record = {"step_index": step_index, "plan_id": plan_id,
                    "episode_id": self.report.get("episode_id"), "frame_id": row.get("frame_id"),
                    "iteration_id": iteration["iteration_id"],
                    "started_host_monotonic_s": planned_at, "solver_elapsed_s": time.monotonic() - planned_at,
                    "observation_index": len(self.report["observations"]) - 1,
                    "target_source": self.target_source, "policy": self.planning_mode, "plan": plan}
                if plan["status"] != "OK":
                    self._persist_planning_record(record)
                    raise ProtectionRequired(f"Planner {plan['status']}: {plan['reason']}")
                age_before_command = observation.age_at()
                record["observation_age_before_command_s"] = age_before_command
                if age_before_command > self.cfg["visual"]["frame_max_age_s"]:
                    record["execution_status"] = "REJECTED_STALE_AFTER_PLANNING"
                    self._persist_planning_record(record)
                    if self.planning.get("execution_envelope"):
                        self._freshness_brake_and_reobserve("PLANNING_CONSUMED_FRESHNESS",age_before_command)
                        iteration["status"] = "FRESH_OBSERVATION_REQUIRED_AFTER_SELF_STATE_BRAKE"
                        continue
                    raise ProtectionRequired("Observation expired during planning")
                command_position, _, command_state = self.state(check_workspace=True)
                if self.fixed_height_execution:
                    drift = float(np.linalg.norm(command_position - position))
                    record["command_start_position_drift_m"] = drift
                    record["command_start_world_ned_m"] = command_position.tolist()
                    record["command_start_actual_body_state"] = execution_sample(command_position, command_state,
                        self.planning, self.cfg["workspace_min_ned_m"], self.cfg["workspace_max_ned_m"])
                    record["command_start_actual_body_state"]["host_monotonic_s"] = time.monotonic()
                    if drift > self.planning["plan_start_position_tolerance_m"]:
                        record["execution_status"] = "REJECTED_STATE_MOVED_AFTER_PLANNING"
                        self._persist_planning_record(record)
                        raise ProtectionRequired("Body moved beyond plan-start tolerance during planning")
                    previous_command = finite_vector(vector(command_state.kinematics_estimated.linear_velocity), "command-start actual velocity")
                    try:
                        checked_motion_state(previous_command, vector(command_state.kinematics_estimated.linear_acceleration),
                            self.planning["execution_speed_cap_m_s"], self.planning["max_acceleration_m_s2"])
                    except ProtectionRequired:
                        record["execution_status"] = "REJECTED_COMMAND_START_MOTION_BUDGET"
                        self._persist_planning_record(record)
                        raise
                    record["command_start_actual_velocity_ned_m_s"] = previous_command.tolist()
                command = self._initial_reference_command(plan, command_position, previous_command, record)
                record["observation_age_before_command_s"] = observation.age_at()
                if record["observation_age_before_command_s"] > self.cfg["visual"]["frame_max_age_s"]:
                    record["execution_status"] = "REJECTED_STALE_AT_DISPATCH"
                    self._persist_planning_record(record)
                    if self.planning.get("execution_envelope"):
                        self._freshness_brake_and_reobserve("FINAL_DISPATCH_WINDOW_EXHAUSTED",record["observation_age_before_command_s"])
                        iteration["status"] = "FRESH_OBSERVATION_REQUIRED_AFTER_SELF_STATE_BRAKE"
                        continue
                    raise ProtectionRequired("Observation expired at final dispatch check")
                record.update(command=command, execution_status="COMMAND_ABOUT_TO_SEND")
                self._persist_planning_record(record)
                applied = command["command_velocity_ned_m_s"]
                row["command_velocity_ned_m_s"] = applied
                if self.fixed_height_execution:
                    position, executed_state = self._execute_fixed_height(command, record, command_state, observation)
                    if record.get("execution_status") == "SELF_STATE_BRAKE_AND_REOBSERVE":
                        iteration["status"] = "FRESH_OBSERVATION_REQUIRED_AFTER_SELF_STATE_BRAKE"
                        continue
                else:
                    self.client.moveByVelocityAsync(*applied, duration=self.planning["command_duration_s"],
                        yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle).join()
                previous_command = np.asarray(applied)
                if record.get("execution_status") != "YIELDED_FOR_NEW_OBSERVATION":
                    record["execution_status"] = "RPC_COMPLETED"
                iteration["status"] = "RPC_COMPLETED"
                position, _, executed_state = self.state(check_workspace=True)
                record["executed_body_state"] = {**record.get("executed_body_state", {}), "world_ned_m": position.tolist(),
                    "velocity_ned_m_s": vector(executed_state.kinematics_estimated.linear_velocity),
                    "acceleration_ned_m_s2": vector(executed_state.kinematics_estimated.linear_acceleration),
                    "actual_body_rpy_deg": quaternion_rpy_deg(_quaternion_tuple(executed_state.kinematics_estimated.orientation)).tolist(),
                    "query_state_timestamp_ns": int(executed_state.timestamp)}
                record["executed_body_state"].update(
                    speed_norm_m_s=float(np.linalg.norm(record["executed_body_state"]["velocity_ned_m_s"])),
                    acceleration_norm_m_s2=float(np.linalg.norm(record["executed_body_state"]["acceleration_ned_m_s2"])))
                record["executed_body_state"].update(
                    over_planning_speed_budget=record["executed_body_state"]["speed_norm_m_s"] > self.planning["execution_speed_cap_m_s"],
                    over_planning_acceleration_budget=record["executed_body_state"]["acceleration_norm_m_s2"] > self.planning["max_acceleration_m_s2"])
                with (self.folder / "command_results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step_index": record["step_index"], "plan_id": plan_id,
                        "episode_id": self.report["episode_id"], "frame_id": row["frame_id"],
                        "execution_status": record["execution_status"], "executed_body_state": record["executed_body_state"]},
                        ensure_ascii=False, allow_nan=False) + "\n")
                self._record_path(position)
                if np.max(np.abs(record["executed_body_state"]["actual_body_rpy_deg"])) > self.planning["attitude_limit_deg"]:
                    raise ProtectionRequired("Actual executed body attitude left fixed-yaw model range")
            else:
                raise ProtectionRequired("Trajectory interface approach duration expired")
        except ProtectionRequired as error:
            if self.report.get("control_iterations"):
                self.report["control_iterations"][-1].update(status="PROTECTED_RETURN", reason=str(error))
            self.report["protective_return_reason"] = str(error)
            self.report["approach_stop_reason"] = "unsupported_input_or_no_valid_plan_controller_protection"
            self.report["planner_certified_fallback"] = False
        except BaseException:
            unsafe_execution_error = True
            if self.report.get("control_iterations"):
                self.report["control_iterations"][-1]["status"] = "UNSAFE_EXECUTION_ERROR"
            raise
        finally:
            if not unsafe_execution_error:
                self.client.cancelLastTask(vehicle_name=self.vehicle)
                self.client.hoverAsync(vehicle_name=self.vehicle).join()
            self.report["metrics"] = summarize(self.report["observations"], None if start is None else self.path_m,
                                                None if start is None else time.monotonic() - start,
                                                self.cfg["visual"])
            self.report["task_success"] = (self.report["metrics"]["task_success"]
                                           and self.report["protective_return_reason"] is None
                                           and not unsafe_execution_error)
            self.report["metrics"]["task_success"] = self.report["task_success"]
            self.report["metrics"]["planning_step_count"] = len(self.report["planning_records"])
            self.report["metrics"]["control_iterations_started"] = len(self.report.get("control_iterations", []))
            self.report["metrics"]["command_completion_count"] = sum(
                r.get("execution_status") == "RPC_COMPLETED" for r in self.report["planning_records"])
            measured = [sample for settling in self.report.get("settling_records", []) for sample in settling["samples"]]
            measured.extend(r["executed_body_state"] for r in self.report["planning_records"] if "executed_body_state" in r)
            self.report["metrics"]["observed_approach_speed_budget_exceedances"] = sum(
                bool(sample.get("over_planning_speed_budget")) for sample in measured)
            self.report["metrics"]["observed_approach_acceleration_budget_exceedances"] = sum(
                bool(sample.get("over_planning_acceleration_budget")) for sample in measured)


def create_trajectory_experiment(policy, target_source, config_path, **kwargs):
    config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    mode = config["planning"].get("execution_reference_mode", "fixed_positive_time_preview")
    if mode == "time_indexed":
        from trajectory_timed_experiment import TimedTrajectoryExperiment
        return TimedTrajectoryExperiment(policy, target_source, config_path, **kwargs)
    if mode != "fixed_positive_time_preview":
        raise ValueError("Unsupported execution_reference_mode")
    return TrajectorySingleExperiment(policy, target_source, config_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("efficiency", "perception"), required=True)
    parser.add_argument("--target-source", choices=("geometry_oracle", "sensor_surface_estimate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    create_trajectory_experiment(args.policy, args.target_source, args.config).run()


if __name__ == "__main__":
    main()
