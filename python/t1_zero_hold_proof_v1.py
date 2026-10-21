"""Keep only actual samples in the declared established stationary interval."""
import copy
import math


def stable_interval_proof(proof):
    result=copy.deepcopy(proof)
    start=result.get("sampled_stop_started_monotonic_s")
    config=result["configured_settlement"]
    if not isinstance(start,(int,float)) or not math.isfinite(start):
        raise ValueError("DECLARED_STABLE_INTERVAL_REQUIRED")
    samples=[s for s in result["samples"] if s["receipt_monotonic_s"]>=start-1e-9]
    retained=samples[-5:]
    if (len(retained)<config["minimum_distinct_state_timestamps"] or
        len({s["state_timestamp_ns"] for s in retained})<config["minimum_distinct_state_timestamps"] or
        retained[-1]["receipt_monotonic_s"]-retained[0]["receipt_monotonic_s"]<config["stationary_span_s"]-1e-9):
        raise ValueError("ACTUAL_STABLE_INTERVAL_DWELL_REQUIRED")
    for sample in retained:
        values=sample["velocity_xy_m_s"]+sample["actual_last_command_xy_m_s"]
        if (not all(math.isfinite(v) for v in values) or
            not sample["stopped_at_stricter_threshold"] or
            math.hypot(*sample["velocity_xy_m_s"])>config["stationary_xy_threshold_m_s"] or
            math.hypot(*sample["actual_last_command_xy_m_s"])>1e-12):
            raise ValueError("ACTUAL_STATIONARY_ZERO_SAMPLES_REQUIRED")
    current=result["current_self_state"]
    if (retained[-1]["receipt_monotonic_s"]!=current["host_receipt_monotonic_s"] or
        retained[-1]["state_timestamp_ns"]!=current["state_timestamp_ns"] or
        retained[-1]["velocity_xy_m_s"]!=current["velocity_ned_m_s"][:2] or
        retained[-1]["receipt_monotonic_s"]-result["target_setpoint_cleared_monotonic_s"]<config["minimum_post_clear_observation_s"]-1e-9):
        raise ValueError("CURRENT_STABLE_ZERO_PROOF_BINDING_REQUIRED")
    result.update(samples=retained,
        proof_scope="last up to five actual zero-command samples within the declared stable interval",
        excluded_pre_stable_history_count=len(proof["samples"])-len(samples),
        stability_limits_unchanged=True)
    return result
