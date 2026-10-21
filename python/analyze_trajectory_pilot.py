"""Audit and describe one trajectory interface pilot, without efficacy claims."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from trajectory_timed_audit import audit_timed_execution


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_scalar(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def finite_vector(value):
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return vector if vector.shape == (3,) and np.isfinite(vector).all() else None


def measured_norm(sample, vector_field, scalar_fields):
    # Recompute from the raw vector when available instead of trusting a flag.
    if vector_field in sample:
        vector = finite_vector(sample[vector_field])
        return finite_scalar(float(np.linalg.norm(vector))) if vector is not None else None
    for field in scalar_fields:
        value = finite_scalar(sample.get(field))
        if value is not None and value >= 0:
            return value
    return None


def summarize_physical_samples(samples, speed_cap, acceleration_cap):
    speeds = [value for sample in samples
              if (value := measured_norm(sample, "velocity_ned_m_s", ("speed_norm_m_s", "speed_m_s"))) is not None]
    accelerations = [value for sample in samples
                     if (value := measured_norm(sample, "acceleration_ned_m_s2", ("acceleration_norm_m_s2",))) is not None]
    return {
        "sample_count": len(samples),
        "speed_assessed_sample_count": len(speeds),
        "acceleration_assessed_sample_count": len(accelerations),
        "max_speed_norm_m_s": max(speeds) if speeds else None,
        "max_acceleration_norm_m_s2": max(accelerations) if accelerations else None,
        "speed_budget_exceedances": sum(value > speed_cap for value in speeds) if speed_cap is not None else None,
        "acceleration_budget_exceedances": sum(value > acceleration_cap for value in accelerations)
                                             if acceleration_cap is not None else None,
    }


def execution_samples(planning):
    samples, sources = [], {"execution_samples": 0, "distinct_execution_end_query": 0,
                            "legacy_executed_body_state": 0}
    for row in planning:
        subsegments = row.get("execution_samples")
        if subsegments:
            samples.extend(subsegments)
            sources["execution_samples"] += len(subsegments)
            endpoint = row.get("executed_body_state")
            # V3 endpoints inherit a host time from the last subsegment. Only
            # their own simulator query timestamp can establish a new query.
            last_timestamp = query_timestamp(subsegments[-1])
            endpoint_timestamp = query_timestamp(endpoint)
            if endpoint_timestamp is not None and last_timestamp is not None and endpoint_timestamp != last_timestamp:
                samples.append(endpoint)
                sources["distinct_execution_end_query"] += 1
        elif row.get("executed_body_state"):
            samples.append(row["executed_body_state"])
            sources["legacy_executed_body_state"] += 1
    return samples, sources


def query_timestamp(sample):
    value = sample.get("query_state_timestamp_ns") if isinstance(sample, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


MATCH_THRESHOLDS = {
    "position_tolerance_m": "position_error_norm_m",
    "speed_tolerance_m_s": "speed_norm_m_s",
    "acceleration_tolerance_m_s2": "acceleration_norm_m_s2",
    "attitude_tolerance_deg": "max_abs_body_axis_deg",
}


def match_sample_values(sample, desired):
    if not isinstance(sample, dict):
        return None
    position = finite_vector(sample.get("world_ned_m"))
    velocity = finite_vector(sample.get("velocity_ned_m_s"))
    acceleration = finite_vector(sample.get("acceleration_ned_m_s2"))
    attitude = finite_vector(sample.get("actual_body_rpy_deg"))
    if desired is None or any(value is None for value in (position, velocity, acceleration, attitude)):
        return None
    return {"position_error_norm_m": float(np.linalg.norm(position - desired)),
            "speed_norm_m_s": float(np.linalg.norm(velocity)),
            "acceleration_norm_m_s2": float(np.linalg.norm(acceleration)),
            "max_abs_body_axis_deg": float(np.max(np.abs(attitude)))}


def zero_previous_xy_command(sample):
    values = sample.get("previous_command_velocity_xy_m_s") if isinstance(sample, dict) else None
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return False
    components = [finite_scalar(value) for value in values]
    return all(value is not None for value in components) and float(np.linalg.norm(components)) <= 1e-9


def summarize_initial_match(report, speed_cap, acceleration_cap):
    record = report.get("initial_state_match")
    if not isinstance(record, dict):
        return {"status": "LEGACY_NOT_EVALUATED", "reported_status": None,
                "acceptance_verified": None, "sample_count": 0,
                "accepted_state": None, "first_observation_drift": None}
    specification = record.get("specification")
    specification = specification if isinstance(specification, dict) else {}
    thresholds = {key: finite_scalar(specification.get(key)) for key in MATCH_THRESHOLDS}
    complete = all(value is not None and value > 0 for value in thresholds.values())
    desired = finite_vector(record.get("desired_world_ned_m"))
    raw_samples = record.get("samples")
    samples = [sample for sample in raw_samples if isinstance(sample, dict)] if isinstance(raw_samples, list) else []
    accepted_state = record.get("accepted_state")
    final_values = match_sample_values(accepted_state, desired)
    reasons = []
    if record.get("status") != "ACCEPTED":
        reasons.append("REPORTED_STATUS_NOT_ACCEPTED")
    if not complete:
        reasons.append("MISSING_OR_INVALID_MATCH_THRESHOLDS")
    if final_values is None:
        reasons.append("MISSING_OR_INVALID_ACCEPTED_STATE_VECTORS")
    elif complete:
        for key, field in MATCH_THRESHOLDS.items():
            if final_values[field] > thresholds[key] + 1e-9:
                reasons.append("ACCEPTED_STATE_EXCEEDS_" + key.upper())
    if not isinstance(accepted_state, dict) or accepted_state.get("accepted") is not True:
        reasons.append("FINAL_STATE_NOT_MARKED_ACCEPTABLE")
    if not zero_previous_xy_command(accepted_state):
        reasons.append("FINAL_PREVIOUS_XY_COMMAND_NOT_VERIFIED_ZERO")
    if not samples or query_timestamp(accepted_state) is None or query_timestamp(samples[-1]) != query_timestamp(accepted_state):
        reasons.append("FINAL_ACCEPTED_QUERY_NOT_RECORDED_AS_LAST_SAMPLE")
    elif any(accepted_state.get(key) != samples[-1].get(key)
             for key in ("world_ned_m", "velocity_ned_m_s", "acceleration_ned_m_s2", "actual_body_rpy_deg")):
        reasons.append("FINAL_ACCEPTED_QUERY_VECTORS_DIFFER_FROM_LAST_SAMPLE")
    suffix = []
    for sample in reversed(samples):
        values = match_sample_values(sample, desired)
        if not complete or sample.get("accepted") is not True or not zero_previous_xy_command(sample) or values is None or any(
                values[field] > thresholds[key] + 1e-9 for key, field in MATCH_THRESHOLDS.items()):
            break
        suffix.append(sample)
    suffix.reverse()
    stable_required = finite_scalar(specification.get("stable_s"))
    gap_cap = finite_scalar(specification.get("max_sample_gap_s"))
    host_times = [finite_scalar(sample.get("host_monotonic_s")) for sample in suffix]
    stable_elapsed = None
    max_gap = None
    stable_verified = None
    if stable_required is not None and stable_required > 0 and gap_cap is not None and gap_cap > 0 and suffix and all(
            value is not None for value in host_times):
        gaps = [right - left for left, right in zip(host_times, host_times[1:])]
        # A stale gap resets the gate at the later query. Earlier acceptable
        # samples must not invalidate a subsequently complete fresh interval.
        reset_index = max((index + 1 for index, gap in enumerate(gaps) if not 0 < gap <= gap_cap + 1e-9), default=0)
        suffix, host_times = suffix[reset_index:], host_times[reset_index:]
        gaps = [right - left for left, right in zip(host_times, host_times[1:])]
        stable_elapsed = host_times[-1] - host_times[0]
        max_gap = max(gaps) if gaps else 0.0
        stable_verified = stable_elapsed + 1e-9 >= stable_required and all(0 < gap <= gap_cap + 1e-9 for gap in gaps)
    if stable_verified is not True:
        reasons.append("STABLE_ACCEPTANCE_INTERVAL_NOT_VERIFIED")
    verified = not reasons
    warmup = [sample for row in record.get("warmup_settling_records", []) if isinstance(row, dict)
              for sample in row.get("samples", []) if isinstance(sample, dict)]
    motion = summarize_physical_samples(warmup + samples, speed_cap, acceleration_cap)
    motion_status = physical_sample_status(motion, speed_cap, acceleration_cap)
    positioning_commands = record.get("command_records", [])
    _, _, _, _, positioning_norms, positioning_invalid = command_samples([], positioning_commands)
    positioning_command_cap = finite_scalar(specification.get("reposition_speed_cap_m_s"))
    observations = report.get("observations", [])
    first_input = observations[0].get("planning_input_body_state") if observations else None
    first_input = first_input if isinstance(first_input, dict) else None
    drift = None
    if verified and first_input is not None:
        drift = vector_differences(accepted_state, first_input)
        drift["query_timestamp_delta_ns"] = (query_timestamp(first_input) - query_timestamp(accepted_state)
                                              if query_timestamp(first_input) is not None else None)
        drift["scope"] = "first recorded planning-input query minus final accepted positioning query; no exact-state identity claim"
    return {"status": "ACCEPTED_VERIFIED" if verified else ("PROTECTED_RETURN" if record.get("status") == "PROTECTED_RETURN" else "INVALID_ACCEPTANCE_EVIDENCE"),
            "reported_status": record.get("status"), "acceptance_verified": verified,
            "acceptance_evidence_reasons": reasons, "specification": specification,
            "thresholds_complete": complete,
            "position_tolerance_norm_m": thresholds["position_tolerance_m"],
            "speed_tolerance_norm_m_s": thresholds["speed_tolerance_m_s"],
            "acceleration_tolerance_norm_m_s2": thresholds["acceleration_tolerance_m_s2"],
            "attitude_tolerance_max_abs_axis_deg": thresholds["attitude_tolerance_deg"],
            "desired_world_ned_m": desired.tolist() if desired is not None else None,
            "accepted_state": accepted_state if verified else None,
            "accepted_state_recomputed_values": final_values, "sample_count": len(warmup) + len(samples),
            "matching_attempt_sample_count": len(samples), "warmup_settling_sample_count": len(warmup),
            "sampled_motion": motion, "physical_sampled_budget_status": motion_status,
            "command_record_count": len(positioning_commands),
            "command_xy_norm_assessed_count": len(positioning_norms),
            "command_xy_norm_invalid_count": positioning_invalid,
            "max_command_xy_norm_m_s": max(positioning_norms) if positioning_norms else None,
            "command_xy_speed_budget_m_s": positioning_command_cap,
            "command_xy_norm_violations": sum(value > positioning_command_cap + 1e-9 for value in positioning_norms)
                                          if positioning_command_cap is not None and positioning_command_cap > 0 else None,
            "stable_suffix_sample_count": len(suffix), "stable_suffix_elapsed_s": stable_elapsed,
            "stable_suffix_max_host_gap_s": max_gap, "stable_interval_verified": stable_verified,
            "elapsed_s": finite_scalar(record.get("elapsed_s")), "protective_reason": record.get("protective_reason"),
            "first_observation_drift": drift,
            "scope": "positioning acceptance and sampled motion are assessed separately from approach, task outcome, and landing; telemetry samples are not independent experimental units"}


def physical_sample_status(motion, speed_cap, acceleration_cap):
    if any((motion[key] or 0) > 0 for key in ("speed_budget_exceedances", "acceleration_budget_exceedances")):
        return "OBSERVED_EXCEEDANCE"
    if not motion["sample_count"]:
        return "NOT_RECORDED"
    if speed_cap is None or acceleration_cap is None or any(motion[key] != motion["sample_count"]
            for key in ("speed_assessed_sample_count", "acceleration_assessed_sample_count")):
        return "PARTIALLY_ASSESSED"
    return "NO_EXCEEDANCE_IN_RECORDED_SAMPLES"


def vector_differences(left, right):
    result = {}
    for field, name, unit in (("world_ned_m", "position", "m"), ("velocity_ned_m_s", "velocity", "m_s"),
                             ("acceleration_ned_m_s2", "acceleration", "m_s2")):
        left_vector, right_vector = finite_vector(left.get(field)), finite_vector(right.get(field))
        if left_vector is None or right_vector is None:
            result[f"{name}_difference_vector_{unit}"] = None
            result[f"{name}_difference_norm_{unit}"] = None
        else:
            delta = right_vector - left_vector
            result[f"{name}_difference_vector_{unit}"] = delta.tolist()
            result[f"{name}_difference_norm_{unit}"] = float(np.linalg.norm(delta))
    return result


def summarize_matched_pairs(rows):
    sessions = sorted({row.get("session") for row in rows if row.get("session") is not None})
    pairs = []
    for session in sessions:
        grouped = [row for row in rows if row.get("session") == session]
        by_policy = {policy: [row for row in grouped if row.get("policy") == policy]
                     for policy in ("efficiency", "perception")}
        pair = {"session": session, "status": "NOT_EVALUATED", "within_shared_acceptance_envelope": None}
        if any(len(value) != 1 for value in by_policy.values()):
            pair["reason"] = "SESSION_DOES_NOT_HAVE_EXACTLY_ONE_STARTED_RUN_PER_POLICY"
        else:
            matches = [by_policy[policy][0].get("initial_state_match", {}) for policy in ("efficiency", "perception")]
            if not all(value.get("acceptance_verified") is True for value in matches):
                pair["reason"] = "BOTH_RUNS_NEED_VERIFIED_INITIAL_ACCEPTANCE"
            elif matches[0].get("specification") != matches[1].get("specification") or matches[0].get("desired_world_ned_m") != matches[1].get("desired_world_ned_m"):
                pair["status"], pair["reason"] = "MISMATCHED_SPECIFICATION", "PAIR_REQUIRES_IDENTICAL_ACCEPTANCE_SPECIFICATION_AND_DESIRED_POSITION"
            else:
                pair.update(vector_differences(matches[0]["accepted_state"], matches[1]["accepted_state"]))
                limits = {"position_m": 2 * matches[0]["position_tolerance_norm_m"],
                          "velocity_m_s": 2 * matches[0]["speed_tolerance_norm_m_s"],
                          "acceleration_m_s2": 2 * matches[0]["acceleration_tolerance_norm_m_s2"]}
                checks = {name: pair[name.replace("_", "_difference_norm_", 1)] <= value + 1e-9 for name, value in limits.items()}
                pair["pair_difference_norm_limits"] = limits
                pair["component_within_shared_acceptance_envelope"] = checks
                pair["within_shared_acceptance_envelope"] = all(checks.values())
                pair["status"] = "WITHIN_SHARED_ACCEPTANCE_ENVELOPE" if all(checks.values()) else "PAIR_DIFFERENCE_EXCEEDS_ACCEPTANCE_ENVELOPE"
                pair["scope"] = "perception minus efficiency accepted state; the two accepted states lie in a common tolerance range, not an identical state"
        pairs.append(pair)
    return {"paired_session_sample_count": sum(pair["within_shared_acceptance_envelope"] is True for pair in pairs),
            "assessed_pair_count": sum(pair["within_shared_acceptance_envelope"] is not None for pair in pairs),
            "started_session_count": len(sessions), "pairs": pairs,
            "statistical_unit": "one policy pair per separately launched simulator session; state queries and adjacent frames are not independent repetitions",
            "independent_run_generation_verified_by_this_analysis": False}


def protection_evidence(report, stage_samples, speed_cap, acceleration_cap):
    """Keep reported rejections separate from numeric observations.

    A legacy initial-motion rejection cannot be quantified using an earlier
    settling sample or later execution state. New sampled guard codes are tied
    to their own sample; report-level codes can use matching sampled codes.
    """
    def sample_reason_text(sample):
        values = sample.get("protection_reasons", [])
        if isinstance(values, str):
            return values.lower()
        return ";".join(value for value in (values or []) if isinstance(value, str)).lower()

    reasons = []
    for field in ("protective_return_reason", "error"):
        value = report.get(field)
        if isinstance(value, str) and value:
            reasons.append({"source": field, "reason": value, "stage": None, "sample": None})
    for index, iteration in enumerate(report.get("control_iterations", [])):
        value = iteration.get("reason")
        if isinstance(value, str) and value:
            reasons.append({"source": f"control_iterations[{index}].reason", "reason": value, "stage": None, "sample": None})
    for stage, samples in stage_samples.items():
        for index, sample in enumerate(samples):
            values = sample.get("protection_reasons", [])
            values = [values] if isinstance(values, str) else (values or [])
            for value in values:
                if isinstance(value, str) and value:
                    reasons.append({"source": f"{stage}[{index}].protection_reasons", "reason": value,
                                    "stage": stage, "sample": sample})
    patterns = (
        ("measured initial velocity exceeds planner budget", "speed", "planning_input"),
        ("measured initial acceleration exceeds planner budget", "acceleration", "planning_input"),
        ("actual_speed_budget_exceeded", "speed", None),
        ("actual_acceleration_budget_exceeded", "acceleration", None),
    )
    budget_records = []
    for entry in reasons:
        for pattern, component, expected_stage in patterns:
            if pattern not in entry["reason"].lower():
                continue
            resolved_stage = expected_stage
            if entry["sample"] is not None:
                candidates = [entry["sample"]]
            elif expected_stage is not None:
                # The initial-input guard is associated with the last captured
                # observation, or the explicitly rejected command-start query.
                # Earlier high samples cannot fill a missing rejection query.
                planning = report.get("planning_records", [])
                if planning and planning[-1].get("execution_status") == "REJECTED_COMMAND_START_MOTION_BUDGET":
                    resolved_stage = "command_start"
                    last_input = planning[-1].get("command_start_actual_body_state")
                else:
                    observations = report.get("observations", [])
                    last_input = observations[-1].get("planning_input_body_state") if observations else None
                candidates = [last_input] if last_input else []
            else:
                candidates = [sample for samples in stage_samples.values() for sample in samples
                              if pattern in sample_reason_text(sample)]
            if component == "speed":
                cap, vector, scalar = speed_cap, "velocity_ned_m_s", ("speed_norm_m_s", "speed_m_s")
            else:
                cap, vector, scalar = acceleration_cap, "acceleration_ned_m_s2", ("acceleration_norm_m_s2",)
            quantified = cap is not None and any(
                value > cap for sample in candidates
                if (value := measured_norm(sample, vector, scalar)) is not None)
            budget_records.append({"source": entry["source"], "reason": entry["reason"], "component": component,
                                   "stage": entry["stage"] or resolved_stage or "reported_body_guard",
                                   "numeric_exceedance_support_recorded": quantified})
    return {
        "recorded_protection_reasons": sorted({entry["reason"] for entry in reasons}),
        "reported_budget_rejection_reasons": sorted({entry["reason"] for entry in budget_records}),
        "unquantified_budget_rejection_reasons": sorted({entry["reason"] for entry in budget_records
                                                         if not entry["numeric_exceedance_support_recorded"]}),
        "budget_rejection_evidence": budget_records,
    }


def command_samples(planning, keepalive_records=()):
    norms, invalid = [], 0
    sources = {"actual_xy_velocity_components": 0, "legacy_reference_vector": 0,
               "goal_keepalive_actual_xy_velocity_components": len(keepalive_records)}
    for row in planning:
        subsegments = row.get("execution_samples") or []
        actual = [sample for sample in subsegments if "command_velocity_xy_m_s" in sample]
        if actual:
            sources["actual_xy_velocity_components"] += len(actual)
            for sample in actual:
                try:
                    vector = np.asarray(sample["command_velocity_xy_m_s"], dtype=float)
                except (TypeError, ValueError):
                    vector = np.asarray([])
                norm = finite_scalar(float(np.linalg.norm(vector))) if vector.shape == (2,) and np.isfinite(vector).all() else None
                if norm is None:
                    invalid += 1
                else:
                    norms.append(norm)
        elif row.get("command"):
            sources["legacy_reference_vector"] += 1
            norm = measured_norm(row["command"], "command_velocity_ned_m_s", ("command_velocity_norm_m_s",))
            if norm is None:
                invalid += 1
            else:
                norms.append(norm)
    keepalive_norms, keepalive_invalid = [], 0
    for row in keepalive_records:
        try:
            vector = np.asarray(row.get("command_velocity_xy_m_s"), dtype=float)
        except (TypeError, ValueError):
            vector = np.asarray([])
        norm = finite_scalar(float(np.linalg.norm(vector))) if vector.shape == (2,) and np.isfinite(vector).all() else None
        if norm is None:
            keepalive_invalid += 1
        else:
            keepalive_norms.append(norm)
    norms.extend(keepalive_norms)
    invalid += keepalive_invalid
    has_actual_xy = sources["actual_xy_velocity_components"] or sources["goal_keepalive_actual_xy_velocity_components"]
    if has_actual_xy and sources["legacy_reference_vector"]:
        scope = "MIXED_ACTUAL_XY_COMPONENTS_AND_LEGACY_REFERENCE_VECTOR"
    elif has_actual_xy:
        scope = "ACTUAL_XY_VELOCITY_COMPONENTS_ONLY_NOT_3D_VELOCITY_COMMAND"
    elif sources["legacy_reference_vector"]:
        scope = "LEGACY_RECORDED_REFERENCE_VECTOR_ONLY"
    else:
        scope = "NOT_RECORDED"
    return norms, invalid, sources, scope, keepalive_norms, keepalive_invalid


def summarize_spatial_samples(samples, configuration):
    planning=configuration.get('planning',{})
    margin=finite_scalar(planning.get('execution_guard_margin_m'))
    low=finite_vector(configuration.get('workspace_min_ned_m'))
    high=finite_vector(configuration.get('workspace_max_ned_m'))
    if margin is None or low is None or high is None:
        return {'status':'NOT_CONFIGURED','scope':'approach spatial samples only'}
    margins=[]
    for sample in samples:
        position=finite_vector(sample.get('world_ned_m'))
        if position is not None:
            margins.append(float(np.min(np.minimum(position-low,high-position))))
    violations=sum(value<margin-1e-9 for value in margins)
    return {'status':'OBSERVED_EXCEEDANCE' if violations else (
                'NO_EXCEEDANCE_IN_RECORDED_SAMPLES' if len(margins)==len(samples) and margins else 'PARTIALLY_ASSESSED'),
            'required_inner_margin_m':margin,'sample_count':len(samples),'assessed_count':len(margins),
            'minimum_recorded_margin_m':min(margins,default=None),'entered_inner_margin_samples':violations,
            'outside_outer_workspace_samples':sum(value<0 for value in margins),
            'scope':'saved approach queries; stage duplicates preserved, not independent repetitions or continuous safety proof'}


def summarize_run(report):
    planning = report.get("planning_records", [])
    observations = report.get("observations", [])
    planning_config = report.get("configuration", {}).get("planning", {})
    cap = finite_scalar(planning_config.get("execution_speed_cap_m_s"))
    acceleration_cap = finite_scalar(planning_config.get("max_acceleration_m_s2"))
    cap = cap if cap is not None and cap > 0 else None
    acceleration_cap = acceleration_cap if acceleration_cap is not None and acceleration_cap > 0 else None
    reference_norms = [value for row in planning if row.get("command")
                     if (value := measured_norm(row["command"], "command_velocity_ned_m_s", ("command_velocity_norm_m_s",))) is not None]
    keepalive = report.get("goal_keepalive_records", [])
    command_norms, invalid_commands, command_sources, command_scope, keepalive_norms, keepalive_invalid = command_samples(planning, keepalive)
    solver_times = [value for row in planning if (value := finite_scalar(row.get("solver_elapsed_s"))) is not None]
    settling = [sample for entry in report.get("settling_records", []) for sample in entry.get("samples", [])]
    executed, sample_sources = execution_samples(planning)
    planning_input = [row["planning_input_body_state"] for row in observations if row.get("planning_input_body_state")]
    command_start = [row["command_start_actual_body_state"] for row in planning if row.get("command_start_actual_body_state")]
    guard_input = report.get("execution_guard_samples", [])
    execution_motion = summarize_physical_samples(executed, cap, acceleration_cap)
    settling_motion = summarize_physical_samples(settling, cap, acceleration_cap)
    planning_input_motion = summarize_physical_samples(planning_input, cap, acceleration_cap)
    command_start_motion = summarize_physical_samples(command_start, cap, acceleration_cap)
    guard_input_motion = summarize_physical_samples(guard_input, cap, acceleration_cap)
    all_physical_samples = planning_input + command_start + executed + settling + guard_input
    combined_motion = summarize_physical_samples(all_physical_samples, cap, acceleration_cap)
    evidence = protection_evidence(report, {"planning_input": planning_input, "command_start": command_start,
                                           "execution": executed, "settling": settling, "guard_input": guard_input}, cap, acceleration_cap)
    before_dispatch_ages = [value for row in planning
                            if (value := finite_scalar(row.get("observation_age_before_command_s"))) is not None]
    observed_exceedance = any((combined_motion[key] or 0) > 0 for key in ("speed_budget_exceedances", "acceleration_budget_exceedances"))
    if evidence["unquantified_budget_rejection_reasons"]:
        physical_status = "INCOMPLETE_WITH_REPORTED_BUDGET_REJECTION"
    elif observed_exceedance:
        physical_status = "OBSERVED_EXCEEDANCE"
    elif not combined_motion["sample_count"]:
        physical_status = "NOT_RECORDED"
    elif cap is None or acceleration_cap is None or any(
            combined_motion[key] != combined_motion["sample_count"]
            for key in ("speed_assessed_sample_count", "acceleration_assessed_sample_count")):
        physical_status = "PARTIALLY_ASSESSED"
    else:
        physical_status = "NO_EXCEEDANCE_IN_RECORDED_SAMPLES"
    z_values = [float(position[2]) for sample in observations + all_physical_samples
                if (position := finite_vector(sample.get("world_ned_m"))) is not None]
    modes = sorted({value for value in [report.get("body_execution_mode"), planning_config.get("body_execution_mode")]
                    + [row.get("body_execution_mode") or (row.get("command") or {}).get("body_execution_mode") for row in planning]
                    if isinstance(value, str) and value})
    metrics = report.get("metrics", {})
    landing = report.get("landing_evidence", {})
    initial_match = summarize_initial_match(report, cap, acceleration_cap)
    reported_task_success = report.get("task_success") is True
    task_success = reported_task_success and initial_match["acceptance_verified"] is not False
    result = {
        "policy": report["policy"], "target_source": report["target_source"],
        "execution_reference_mode": report.get("execution_reference_mode", "fixed_positive_time_preview"),
        "command_ramp_time_basis": report.get("command_ramp_time_basis", "fixed_sample_quantum"),
        "image_center_reference_mode": report.get("image_center_reference_mode", "surface_representative_point"),
        "approach_spatial_budget": summarize_spatial_samples(all_physical_samples,report.get('configuration',{})),
        "status": report["status"], "task_success": task_success,
        "reported_task_success": reported_task_success,
        "initial_state_match": initial_match,
        "task_success_inconsistent_with_initial_acceptance": reported_task_success and initial_match["acceptance_verified"] is False,
        "safe_landing_and_release": report["status"] == "PASS" and landing.get("landed_flag") is True
                                    and landing.get("api_control_released") is True,
        "approach_duration_s": metrics.get("approach_duration_s"),
        "sampled_approach_path_m": metrics.get("approach_path_m"),
        "observation_count": len(observations), "planning_step_count": len(planning),
        "ok_plan_count": sum(row.get("plan", {}).get("status") == "OK" for row in planning),
        "solver_p50_s": float(np.median(solver_times)) if solver_times else None,
        "solver_p95_s": float(np.percentile(solver_times, 95)) if solver_times else None,
        "max_command_norm_m_s": max(command_norms) if command_norms else None,
        "command_norm_assessed_count": len(command_norms),
        "command_norm_invalid_count": invalid_commands,
        "command_norm_sample_sources": command_sources,
        "command_norm_scope": command_scope,
        "command_norm_violations": sum(norm > cap + 1e-9 for norm in command_norms) if cap is not None else None,
        "goal_keepalive_command_count": len(keepalive),
        "goal_keepalive_command_norm_assessed_count": len(keepalive_norms),
        "goal_keepalive_command_norm_invalid_count": keepalive_invalid,
        "max_goal_keepalive_command_xy_norm_m_s": max(keepalive_norms) if keepalive_norms else None,
        "goal_keepalive_command_norm_violations": sum(norm > cap + 1e-9 for norm in keepalive_norms) if cap is not None else None,
        "goal_keepalive_requested_duration_is_executed_time": False,
        "recorded_command_norm_within_budget": all(norm <= cap + 1e-9 for norm in command_norms)
                                                if command_norms and not invalid_commands and cap is not None else None,
        "max_reference_command_norm_m_s": max(reference_norms) if reference_norms else None,
        "reference_command_norm_violations": sum(norm > cap + 1e-9 for norm in reference_norms) if cap is not None else None,
        "recorded_reference_command_norm_within_budget": all(norm <= cap + 1e-9 for norm in reference_norms)
                                                       if reference_norms and cap is not None else None,
        "execution_speed_budget_m_s": cap,
        "planning_acceleration_budget_m_s2": acceleration_cap,
        "physical_sampled_budget_status": physical_status,
        "observed_numeric_physical_budget_exceedance_present": observed_exceedance,
        "reported_motion_budget_rejection_present": bool(evidence["reported_budget_rejection_reasons"]),
        "unquantified_motion_budget_rejection_present": bool(evidence["unquantified_budget_rejection_reasons"]),
        "body_execution_mode": modes[0] if len(modes) == 1 else ("MIXED:" + ",".join(modes) if modes else "NOT_RECORDED"),
        "execution_mode": report.get("execution_mode") or planning_config.get("execution_mode") or "NOT_RECORDED",
        "execution_sample_sources": sample_sources,
        "rpc_completed_count": sum(row.get("execution_status") == "RPC_COMPLETED" for row in planning),
        "max_settling_sample_speed_m_s": settling_motion["max_speed_norm_m_s"],
        "max_settling_sample_acceleration_m_s2": settling_motion["max_acceleration_norm_m_s2"],
        "min_sampled_approach_world_z_ned_m": min(z_values) if z_values else None,
        "max_sampled_approach_world_z_ned_m": max(z_values) if z_values else None,
        "min_sampled_approach_world_height_m": -max(z_values) if z_values else None,
        "max_sampled_approach_world_height_m": -min(z_values) if z_values else None,
        "max_dispatch_age_proxy_s": max(before_dispatch_ages) if before_dispatch_ages else None,
        "bystander_max_drift_m": report.get("bystander_max_drift_m"),
        "protective_return_reason": report.get("protective_return_reason"),
        "approach_stop_reason": report.get("approach_stop_reason"),
        "error": report.get("error"),
        "freshness_replan_event_count": len(report.get("freshness_replan_events", [])),
        "freshness_braked_ready_count": sum(e.get("status") == "BRAKED_READY_TO_REOBSERVE" for e in report.get("freshness_replan_events", [])),
        "rejected_boundary_envelope_count": len(report.get("rejected_boundary_envelopes", [])),
        "timed_execution_audit": audit_timed_execution(report),
        "freshness_reobserved_valid_count": sum(e.get("status") == "REOBSERVED_VALID" for e in report.get("freshness_replan_events", [])),
        "freshness_limit_rejected_after_brake_count": sum(e.get("status") == "BRAKED_REOBSERVATION_LIMIT_REJECTED" for e in report.get("freshness_replan_events", [])),
        "execution_yield_count": sum(e.get("execution_status") == "YIELDED_FOR_NEW_OBSERVATION" for e in planning),
    }
    for prefix, values in (("execution", execution_motion), ("settling", settling_motion),
                           ("planning_input", planning_input_motion), ("command_start", command_start_motion),
                           ("guard_input", guard_input_motion),
                           ("observed_approach", combined_motion)):
        result.update({f"{prefix}_{key}": value for key, value in values.items()})
    result.update(evidence)
    return result


def build_overview(rows):
    heights_min = [value for row in rows
                   if (value := finite_scalar(row.get("min_sampled_approach_world_height_m"))) is not None]
    heights_max = [value for row in rows
                   if (value := finite_scalar(row.get("max_sampled_approach_world_height_m"))) is not None]
    return {
        "body_execution_modes": sorted({row["body_execution_mode"] for row in rows if "body_execution_mode" in row}),
        "execution_modes": sorted({row["execution_mode"] for row in rows if "execution_mode" in row}),
        "execution_reference_modes": sorted({row["execution_reference_mode"] for row in rows if "execution_reference_mode" in row}),
        "command_ramp_time_bases": sorted({row["command_ramp_time_basis"] for row in rows if "command_ramp_time_basis" in row}),
        "command_norm_scopes": sorted({row["command_norm_scope"] for row in rows if "command_norm_scope" in row}),
        "min_sampled_approach_world_height_m": min(heights_min) if heights_min else None,
        "max_sampled_approach_world_height_m": max(heights_max) if heights_max else None,
        "height_scope": "negative world NED Z from recorded approach observations/execution/settling; not ground clearance",
        "runs_with_recorded_command_norm_violation": sum((row.get("command_norm_violations") or 0) > 0 for row in rows),
        "runs_with_observed_physical_budget_exceedance": sum(row.get("observed_numeric_physical_budget_exceedance_present") is True for row in rows),
        "runs_with_reported_motion_budget_rejection": sum(row.get("reported_motion_budget_rejection_present") is True for row in rows),
        "runs_with_unquantified_motion_budget_rejection": sum(row.get("unquantified_motion_budget_rejection_present") is True for row in rows),
        "runs_without_recorded_physical_samples": sum(not row.get("observed_approach_sample_count", 0) for row in rows),
        "runs_with_partially_assessed_physical_samples": sum(row.get("physical_sampled_budget_status") == "PARTIALLY_ASSESSED" for row in rows),
        "runs_without_terminal_report": sum(row.get("status") == "NO_TERMINAL_REPORT" for row in rows),
        "runs_with_verified_initial_acceptance": sum(row.get("initial_state_match", {}).get("acceptance_verified") is True for row in rows),
        "runs_with_failed_or_invalid_initial_acceptance": sum(row.get("initial_state_match", {}).get("acceptance_verified") is False for row in rows),
        "runs_without_initial_acceptance_assessment": sum(row.get("initial_state_match", {}).get("acceptance_verified") is None for row in rows),
        "recorded_initial_positioning_sample_count": sum(row.get("initial_state_match", {}).get("sample_count", 0) for row in rows),
        "runs_with_initial_positioning_budget_exceedance": sum(row.get("initial_state_match", {}).get("physical_sampled_budget_status") == "OBSERVED_EXCEEDANCE" for row in rows),
        "runs_with_task_success_initial_acceptance_contradiction": sum(row.get("task_success_inconsistent_with_initial_acceptance") is True for row in rows),
        "recorded_execution_physical_sample_count": sum(row.get("execution_sample_count", 0) for row in rows),
        "recorded_settling_physical_sample_count": sum(row.get("settling_sample_count", 0) for row in rows),
        "recorded_planning_input_physical_sample_count": sum(row.get("planning_input_sample_count", 0) for row in rows),
        "recorded_command_start_physical_sample_count": sum(row.get("command_start_sample_count", 0) for row in rows),
        "goal_keepalive_command_count": sum(row.get("goal_keepalive_command_count", 0) for row in rows),
        "goal_keepalive_command_norm_violations": sum(row.get("goal_keepalive_command_norm_violations") or 0 for row in rows),
        "command_scope": "actual sent XY components when recorded, otherwise legacy reference vector; altitude-position controller output is not a 3D velocity command",
        "physical_scope": "planning-input and command-start queries plus execution subsegments, independently timestamped execution-end queries (legacy endpoint fallback), and settling; positioning samples excluded; distinct query stages are not deduplicated; no unsampled peak guarantee",
        "keepalive_scope": "actual XY RPC command records only, not physical state samples; requested duration may be superseded and is not summed as elapsed or executed time",
        "initial_positioning_scope": "separate positioning acceptance and physical-budget queries; a protected positioning return remains a started failed task even when landing succeeds; legacy acceptance is unassessed",
    }


def main(manifest_path, output_dir):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    rows, runs = [], []
    same_sources = None
    for session in manifest["sessions"]:
        batch_path = Path(session["batch_path"]) / "batch_report.json"
        if not batch_path.exists():
            continue
        if session.get("batch_report_sha256") and sha256(batch_path) != session["batch_report_sha256"]:
            raise ValueError("Batch hash mismatch")
        batch = read_json(batch_path)
        if batch["target_source"] != manifest["target_source"]:
            raise ValueError("Do not mix target source groups")
        for entry in batch["runs"]:
            row = {"session": session["session_index"], "run": entry["run"],
                   "report_path": entry["report_path"]}
            path = Path(entry["report_path"])
            if not path.exists():
                row.update(policy=entry["policy"], target_source=batch["target_source"], status="NO_TERMINAL_REPORT",
                           task_success=False, safe_landing_and_release=False)
            else:
                if entry.get("report_sha256") and sha256(path) != entry["report_sha256"]:
                    raise ValueError("Run hash mismatch")
                report = read_json(path)
                if report["target_source"] != manifest["target_source"]:
                    raise ValueError("Run target source differs from manifest")
                if report.get("configuration_sha256") != manifest["experiment_config_sha256"]:
                    raise ValueError("Run configuration differs from frozen manifest")
                if same_sources is None:
                    same_sources = report["source_sha256"]
                elif same_sources != report["source_sha256"]:
                    raise ValueError("Changed source versions require separate analyses")
                row.update(summarize_run(report))
                runs.append(report)
            rows.append(row)
    summary = {
        "kind": "TRAJECTORY_INTERFACE_PILOT_DESCRIPTIVE_ANALYSIS", "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path), "manifest_status": manifest["status"],
        "target_source": manifest["target_source"], "formal_thesis_effect_results": False,
        "planned_flights": manifest["planned_sessions"] * manifest["planned_flights_per_session"],
        "started_flights": len(rows), "task_success_count": sum(row["task_success"] for row in rows),
        "safe_landing_and_release_count": sum(row["safe_landing_and_release"] for row in rows),
        "overview": build_overview(rows),
        "initial_state_pairs": summarize_matched_pairs(rows),
        "same_run_source_versions": same_sources, "runs": rows,
        "ai_assistance": "analysis code and narrative authored with AI assistance",
        "limits": ["Interface development pilot with very small sample", "No inference of method superiority",
                   "Path is from sampled simulator self states", "Planning-input, command-start, execution and settling queries do not cover all physical velocity or acceleration peaks",
                   "Reported motion-budget rejection without a matching quantified query remains incomplete; numeric exceedance is never reconstructed from a reason string",
                   "Velocity-command norm compliance does not imply physical speed/acceleration budget compliance",
                   "Reference feedback and command clipping do not prove exact continuous polynomial tracking",
                   "Height is relative to world NED origin, not clearance above ground; launch/return/landing are outside this approach summary",
                   "Timestamp age is a protection proxy, not exact exposure-to-command delay",
                   "geometry_oracle is not sensor target estimation; surface_estimate is not balloon center"],
        "generated_local": datetime.now().astimezone().isoformat(),
    }
    summary["analysis_source"] = {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__)),
        "timed_audit_path": str(Path(__file__).with_name("trajectory_timed_audit.py").resolve()),
        "timed_audit_sha256": sha256(Path(__file__).with_name("trajectory_timed_audit.py"))}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if rows:
        with (output_dir / "runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# 单机候选轨迹 AirSim 接口先导", "", f"目标输入：`{summary['target_source']}`；批次状态：`{manifest['status']}`。", "",
             f"计划飞行{summary['planned_flights']}次，已开始{len(rows)}次；任务达标{summary['task_success_count']}次，确认落地并释放控制{summary['safe_landing_and_release_count']}次。", "",
             "| 方法 | 任务达标 | 安全收尾 | 接近秒 | 采样路径米 | 规划步数 | 停止原因 |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in rows:
        duration = row.get("approach_duration_s")
        path = row.get("sampled_approach_path_m")
        lines.append("| " + " | ".join(str(value) for value in (
            row["policy"], row["task_success"], row["safe_landing_and_release"],
            f"{duration:.3f}" if duration is not None else "未完成",
            f"{path:.3f}" if path is not None else "未完成", row.get("planning_step_count", 0),
            row.get("protective_return_reason") or row.get("approach_stop_reason") or row.get("error") or row["status"])) + " |")
    lines += ["", "本批用于验证观测、坐标、规划、执行和起降接口。失败、中断、未达标保留；不能据此声称新方法更优或完成硕士论文实验。", "",
              "## 命令、实际状态和落地证据", "",
              "命令 norm 合规、实际状态采样未超限、任务达标和确认落地释放控制分别统计。命令合规不能推导动力学合规；落地成功也不能消除接近阶段的实际速度/加速度采样超限。", "",
              "| 方法 | 机体执行模式 | 命令分量 norm 超限 | 规划输入速度/加速度超限 | 命令前速度/加速度超限 | 执行速度/加速度超限 | 保持速度/加速度超限 | 实际状态采样结论 | 高度 min/max 米 |", 
              "|---|---|---:|---|---|---|---|---|---|"]
    for row in rows:
        low, high = row.get("min_sampled_approach_world_height_m"), row.get("max_sampled_approach_world_height_m")
        height_range = f"{low:.4f} / {high:.4f}" if low is not None and high is not None else "未记录"
        def count_text(key):
            return str(row[key]) if row.get(key) is not None else "未评估"
        lines.append("| " + " | ".join((row["policy"], row.get("body_execution_mode", "未记录"),
            count_text("command_norm_violations"),
            count_text("planning_input_speed_budget_exceedances") + " / " + count_text("planning_input_acceleration_budget_exceedances"),
            count_text("command_start_speed_budget_exceedances") + " / " + count_text("command_start_acceleration_budget_exceedances"),
            count_text("execution_speed_budget_exceedances") + " / " + count_text("execution_acceleration_budget_exceedances"),
            count_text("settling_speed_budget_exceedances") + " / " + count_text("settling_acceleration_budget_exceedances"),
            row.get("physical_sampled_budget_status", "未评估"), height_range)) + " |")
    lines += ["", "命令预算优先检查 execution_samples 实际发出的 XY 速度分量 norm；高度位置模式不等于发送完整三维速度命令。没有实际分量记录时，只核验旧版参考向量，并在 command_norm_scope 中标明范围。参考 preview 不计作已执行命令。", "",
              "goal_keepalive_records 的维持 RPC 指令独立计数并合并检查 XY norm，不作为实际状态采样。requested_duration_s 可能被后续任务替代，不累加为实际执行或飞行时间。", "",
              "规划输入、命令前状态、执行及保持查询分别统计后并入接近总数；不同阶段的查询不跨阶段去重。执行采样优先使用 execution_samples；末态只有在其 AirSim query 时间戳与最后分段时间戳均已知且不同的情况下单独计数。缺时间戳或相同时间戳不重复末态；无分段时使用旧版 executed_body_state。继承的末态 host 时间不用于判定新查询或时序。定位验收阶段单独统计，排除于接近指标。预算值来自冻结配置。", "",
              "旧报告如果明确记载速度/加速度预算保护拒绝，但对应查询没有量化记录，结论为 INCOMPLETE_WITH_REPORTED_BUDGET_REJECTION；已保存样本的超限计数仍按实际数据计算，不能从拒绝文字伪造加速度数值，也不能将其他样本未超限写成全程合规。", "",
              "高度定义为负的 world NED Z，相对于世界坐标原点，不是离地高度。上述高度范围仅覆盖接近阶段已记录状态，不覆盖起飞、返航、落地或未采样峰值。", "",
              "效率/感知规划仅改变图像居中代价权重；原始RGB、深度、候选日志和实际状态可供复算。实际身份关联、遮挡质量和正式独立重复仍待后续阶段。", "",
              "AI使用说明：分析脚本和报告由AI辅助编写；飞行结果来自所链接的AirSim原始记录。"]
    reason_rows = [row for row in rows if row.get("recorded_protection_reasons")]
    if reason_rows:
        lines += ["", "## 保护拒绝与证据缺口", ""]
        for row in reason_rows:
            lines.append(f"- `{row['policy']}`：" + "；".join(row["recorded_protection_reasons"]))
            if row.get("unquantified_budget_rejection_reasons"):
                lines.append("  - 对应运动预算拒绝未量化：" + "；".join(row["unquantified_budget_rejection_reasons"]))
    if any(row.get("goal_keepalive_command_count", 0) for row in rows):
        lines += ["", "## 同模式维持指令", "", "| 方法 | 已记录维持 RPC 指令 | XY norm 已评估 | 无效/缺失 norm | XY norm 超限 |",
                  "|---|---:|---:|---:|---:|"]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in (
                row["policy"], row.get("goal_keepalive_command_count", 0),
                row.get("goal_keepalive_command_norm_assessed_count", 0),
                row.get("goal_keepalive_command_norm_invalid_count", 0),
                row.get("goal_keepalive_command_norm_violations", "未评估"))) + " |")
    lines += ["", "## 实测起点验收与配对范围", "",
              "定位阶段核验最终位置误差 norm、速度 norm、加速度 norm和固定零姿态的各轴角度误差，并复核末尾连续稳定区间。ACCEPTED 字符串不替代原始查询证据；定位保护返回仍计入已开始任务失败，落地结果单独判断。旧版缺验收记录标记 LEGACY_NOT_EVALUATED。定位采样不并入接近时间、路径、高度或接近运动预算样本。", "",
              "| 方法 | 初始验收结论 | 定位查询数 | 定位运动预算采样结论 | 位置/速度/加速度 norm 容差 | 稳定区间秒 | 首次观测位置漂移米 |",
              "|---|---|---:|---|---|---:|---:|"]
    for row in rows:
        match = row.get("initial_state_match", {})
        drift = match.get("first_observation_drift") or {}
        def format_number(value):
            return f"{value:.6f}" if finite_scalar(value) is not None else "未评估"
        tolerance = " / ".join(format_number(match.get(key)) for key in (
            "position_tolerance_norm_m", "speed_tolerance_norm_m_s", "acceleration_tolerance_norm_m_s2"))
        lines.append("| " + " | ".join((row["policy"], match.get("status", "未评估"), str(match.get("sample_count", 0)),
            match.get("physical_sampled_budget_status", "未评估"), tolerance,
            format_number(match.get("stable_suffix_elapsed_s")), format_number(drift.get("position_difference_norm_m")))) + " |")
    pair_summary = summary["initial_state_pairs"]
    lines += ["", f"已开始会话数 {pair_summary['started_session_count']}；已核验的完整配对会话数 {pair_summary['paired_session_sample_count']}。每个会话须恰好有一条 efficiency 和一条 perception 已开始记录；定位失败和缺报告不从分母删除。独立统计单位为独立启动仿真的方法配对会话，遥测查询和相邻帧不是独立重复；分析器不单独证明启动流程的独立性。", "",
              "| 会话 | 配对结论 | 位置差 norm 米 | 速度差 norm m/s | 加速度差 norm m/s² |",
              "|---|---|---:|---:|---:|"]
    for pair in pair_summary["pairs"]:
        lines.append("| " + " | ".join((str(pair["session"]), pair["status"],
            *(format_number(pair.get(key)) for key in ("position_difference_norm_m", "velocity_difference_norm_m_s", "acceleration_difference_norm_m_s2")))) + " |")
    lines += ["", "配对差向量定义为 perception 验收状态减 efficiency 验收状态。共同许可差值分别为位置、速度、加速度验收 norm 阈值的两倍；符合共同接受范围不表示实际状态绝对相同。完整向量、阈值、首次观测速度/加速度漂移及验收保护理由保存在 summary.json。", ""]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render(runs, output_dir)
    print(json.dumps({key: summary[key] for key in ("planned_flights", "started_flights", "task_success_count", "safe_landing_and_release_count")}, ensure_ascii=False))
    return summary


def render(runs, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for run in runs:
        observations = run.get("observations", [])
        if not observations:
            continue
        points = np.array([row["world_ned_m"] for row in observations])
        label = run["policy"]
        color = {"efficiency": "#2166ac", "perception": "#d6604d"}[label]
        axes[0].plot(points[:, 0], points[:, 1], ".-", color=color, label=label)
        for axis_index, field in ((1, "depth_m"), (2, "frame_age_s")):
            values = [row.get(field) for row in observations]
            axes[axis_index].plot(range(len(values)), values, color=color, label=label)
    axes[0].set(xlabel="World NED x (m)", ylabel="World NED y (m)", title="Sampled actual approach positions")
    axes[1].set(xlabel="Observation index", ylabel="Surface ray range (m)", title="Observed surface range")
    axes[2].set(xlabel="Observation index", ylabel="Readback + host age proxy (s)", title="Development time gate input")
    for axis in axes:
        axis.grid(alpha=.2)
        if axis.lines:
            axis.legend(fontsize=8)
    figure.suptitle("AirSim controller interface pilot — descriptive only")
    figure.savefig(output_dir / "interface_observations.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.manifest, args.output)
