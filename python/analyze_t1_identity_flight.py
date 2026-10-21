"""Independent, readonly analysis of the developmental identity/depth XY pilots.

No runner, AirSim, model, cloud, or motion module is imported. Frozen sources,
saved media, segmentation IDs and actual execution receipts are audited here.
All declared slots and failed runs remain in the denominator. Startup/landing
are excluded from the measured XY stage, and two pilots are not efficacy trials.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

SOURCE_PATTERN = re.compile(r"^(python/[A-Za-z0-9_]+\.py|scripts/[A-Za-z0-9_-]+\.ps1)$")
TARGET_PURPOSE = "CURRENT_TARGET_TRAJECTORY"
STAGE = "PERCEPTION_XY"


def digest(path):
    path = Path(path)
    if path.name.casefold().startswith(".env"):
        raise ValueError("Credential files are outside analysis inputs")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, size):
    return isinstance(value, (tuple, list)) and len(value) == size and all(finite(x) for x in value)


def verify_file(path, expected, label):
    result = {"label": label, "path": str(path), "expected_sha256": expected,
        "actual_sha256": None, "status": "MISSING"}
    try:
        result["actual_sha256"] = digest(path)
        result["status"] = "PASS" if expected == result["actual_sha256"] else "FAIL"
    except (OSError, ValueError):
        pass
    return result


def mask_truth(segmentation_bgr, palette):
    """Recompute exclusive visible instance-ID boxes from saved media."""
    if not isinstance(segmentation_bgr, np.ndarray) or segmentation_bgr.ndim != 3 or segmentation_bgr.shape[2] != 3:
        raise ValueError("Invalid saved segmentation image")
    truth = {}
    for label, instance_id in (("target", "42"), ("distractor", "43")):
        rgb = palette[instance_id]
        if not vector(rgb, 3) or any(x < 0 or x > 255 or int(x) != x for x in rgb):
            raise ValueError("Invalid saved instance palette")
        mask = np.all(segmentation_bgr == np.asarray(rgb[::-1], np.uint8), axis=2)
        ys, xs = np.nonzero(mask)
        truth[label] = {"instance_id": int(instance_id), "visible_pixel_count": int(len(xs)),
            "bbox_xyxy_pixels": [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None}
    if palette["42"] == palette["43"]:
        raise ValueError("Target/distractor palette colors must differ")
    return truth


def box_iou(left, right):
    if not vector(left, 4) or not vector(right, 4) or any(box[2] <= box[0] or box[3] <= box[1] for box in (left, right)):
        return 0.
    inter = max(0., min(left[2], right[2])-max(left[0], right[0])) * max(0., min(left[3], right[3])-max(left[1], right[1]))
    union = (left[2]-left[0])*(left[3]-left[1]) + (right[2]-right[0])*(right[3]-right[1]) - inter
    return inter/union if union > 0 else 0.


def identity_score(selection, truth, minimum=.3, margin=.15):
    if isinstance(selection, dict) and selection.get("status") == "NO_TARGET_OUTPUT":
        return {"status": "NO_OUTPUT", "selected_identity": None, "ious": {}}
    if not isinstance(selection, dict) or selection.get("status") != "TEMPORAL_LOCAL_ASSOCIATION_VALID":
        return {"status": "UNKNOWN_OUTPUT_STATE", "selected_identity": None, "ious": {}}
    if truth is None:
        return {"status": "UNKNOWN_MEDIA", "selected_identity": None, "ious": {}}
    box = selection.get("selected_candidate", {}).get("bbox_xyxy_pixels")
    scores = {key: box_iou(box, item["bbox_xyxy_pixels"]) for key, item in truth.items()}
    ordered = sorted(scores, key=scores.get, reverse=True)
    top, other = ordered
    unique = scores[top] >= minimum and scores[other] < minimum and scores[top]-scores[other] >= margin
    return {"status": ("CORRECT" if top == "target" else "WRONG_DISTRACTOR") if unique else "AMBIGUOUS_OR_UNMATCHED",
        "selected_identity": top if unique else None, "ious": scores,
        "minimum_iou": minimum, "exclusion_margin": margin,
        "scope": "visible mask-box association; not calibrated identity probability"}


def current_phase(report, capture_started, readback):
    """Use actual transition receipts; nominal slot label is not authoritative."""
    if not finite(capture_started) or not finite(readback):
        return "UNKNOWN_TIME"
    transitions = report.get("phase_transitions", [])
    for transition in transitions:
        start, end = transition.get("started_monotonic_s"), transition.get("completed_monotonic_s")
        if finite(start) and finite(end) and capture_started < end and readback >= start:
            return "TRANSITION_OVERLAP"
    eligible = [row for row in transitions if finite(row.get("completed_monotonic_s"))
        and row["completed_monotonic_s"] <= capture_started]
    return max(eligible, key=lambda row: row["completed_monotonic_s"])["phase_id"] if eligible else "BEFORE_COMPLETED_TRANSITION"


def binding_issues(frame, report):
    issues = []
    result = frame.get("observer_result", {})
    saved = result.get("semantic_frame", {})
    selection = result.get("temporal_identity", {})
    rgb_sha = frame.get("media", {}).get("rgb", {}).get("sha256")
    expected = {"session_id": report.get("session_id"), "target_generation": 1,
        "frame_id": frame.get("frame_id"), "frame_sequence": frame.get("frame_sequence"),
        "image_sha256": rgb_sha, "readback_receipt_monotonic_s": frame.get("capture_received_host_s")}
    if not saved:
        return ["SEMANTIC_FRAME_NOT_RETURNED"]
    if any(saved.get(key) != value or isinstance(saved.get(key), bool) for key, value in expected.items()):
        issues.append("SEMANTIC_ACQUISITION_BINDING_MISMATCH")
    description_sha = hashlib.sha256(report.get("configuration", {}).get("description", "").encode()).hexdigest()
    expected_binding = dict(expected, description_sha256=description_sha,
        candidates_ready_monotonic_s=saved.get("candidates_ready_monotonic_s"))
    if selection.get("current_binding") != expected_binding:
        issues.append("TEMPORAL_CURRENT_BINDING_MISMATCH")
    if selection.get("status") == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
        if selection.get("state") != "TRACKING":
            issues.append("TEMPORAL_VALID_WITHOUT_TRACKING")
        outer = {"session_id": report.get("session_id"), "target_generation": 1,
            "description_sha256": description_sha, "current_frame_id": frame.get("frame_id"),
            "current_frame_sequence": frame.get("frame_sequence"), "current_image_sha256": rgb_sha}
        if any(selection.get(key) != value or isinstance(selection.get(key), bool) for key, value in outer.items()):
            issues.append("TEMPORAL_OUTER_CURRENT_BINDING_MISMATCH")
        candidates = [item for item in saved.get("candidates", [])
            if item.get("candidate_id") == selection.get("selected_candidate_id")]
        if len(candidates) != 1 or candidates[0] != selection.get("selected_candidate"):
            issues.append("TEMPORAL_SELECTED_CANDIDATE_CONTENT_MISMATCH")
        if any(selection.get(key) is not False for key in (
                "flight_authorized", "identity_probability_calibrated", "instance_identity_guaranteed")):
            issues.append("TEMPORAL_EXCEEDS_IDENTITY_EVIDENCE")
    surface = frame.get("surface", {})
    if surface.get("status") == "SURFACE_GEOMETRY_ACCEPTED":
        expected_surface = {"session_id": expected["session_id"], "target_generation": 1,
            "frame_id": expected["frame_id"], "frame_sequence": expected["frame_sequence"],
            "image_sha256": rgb_sha, "description_sha256": description_sha,
            "selection_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID",
            "selected_candidate_id": selection.get("selected_candidate_id"),
            "selected_bbox_xyxy_pixels": selection.get("selected_candidate", {}).get("bbox_xyxy_pixels"),
            "current_binding": expected_binding}
        if any(surface.get(key) != value for key, value in expected_surface.items()):
            issues.append("SURFACE_CURRENT_BINDING_MISMATCH")
        try:
            if surface.get("selection_certificate_sha256") != canonical_sha(selection):
                issues.append("SURFACE_TEMPORAL_CERTIFICATE_HASH_MISMATCH")
        except (ValueError, TypeError):
            issues.append("TEMPORAL_CERTIFICATE_NONFINITE")
    gate = frame.get("motion_gate", {})
    if gate.get("current_binding") is not None and gate["current_binding"] != expected_binding:
        issues.append("GATE_CURRENT_BINDING_MISMATCH")
    return issues


def command_audit(command, frame, report):
    issues = []
    xy, dispatch, duration = command.get("velocity_xy_m_s"), command.get("dispatch_monotonic_s"), command.get("duration_s")
    if not vector(xy, 2) or not finite(dispatch) or not finite(duration) or duration <= 0:
        return ["INVALID_EXECUTED_COMMAND_RECORD"]
    config = report.get("configuration", {})
    cap = config.get("command_speed_cap_m_s", .03)
    if math.hypot(*xy) > cap + 1e-8:
        issues.append("EXECUTED_XY_COMMAND_CAP_EXCEEDED")
    if duration > config.get("command_lease_s", .12) + 1e-9:
        issues.append("EXECUTED_COMMAND_LEASE_EXCEEDED")
    if command.get("purpose") != TARGET_PURPOSE:
        return issues
    if frame is None:
        return issues + ["TARGET_COMMAND_NO_MATCHING_CURRENT_FRAME"]
    issues.extend(binding_issues(frame, report))
    selection = frame.get("observer_result", {}).get("temporal_identity", {})
    gate = frame.get("motion_gate", {})
    if selection.get("status") != "TEMPORAL_LOCAL_ASSOCIATION_VALID" or selection.get("state") != "TRACKING":
        issues.append("TARGET_COMMAND_WITHOUT_VALID_SELECTION")
    if frame.get("surface", {}).get("status") != "SURFACE_GEOMETRY_ACCEPTED":
        issues.append("TARGET_COMMAND_WITHOUT_SURFACE")
    if gate.get("status") != "ALLOW_BOUNDED_COMMAND" or gate.get("target_command_authorized") is not True:
        issues.append("TARGET_COMMAND_WITHOUT_GATE_AUTHORIZATION")
    if xy != gate.get("command_velocity_xy_m_s") or duration != gate.get("command_duration_s"):
        issues.append("EXECUTED_COMMAND_DIFFERS_FROM_GATE")
    end_deadline = gate.get("command_end_deadline_monotonic_s")
    latest_dispatch = gate.get("latest_dispatch_monotonic_s")
    if not finite(end_deadline) or dispatch+duration > end_deadline+1e-9:
        issues.append("TARGET_COMMAND_END_AFTER_GATE_DEADLINE")
    if not finite(latest_dispatch) or dispatch > latest_dispatch+1e-9:
        issues.append("TARGET_COMMAND_DISPATCH_AFTER_GATE_DEADLINE")
    if command.get("target_end_deadline_monotonic_s") != end_deadline or command.get("target_frame_id") != frame.get("frame_id"):
        issues.append("EXECUTED_TARGET_BINDING_MISMATCH")
    receipt = frame.get("capture_received_host_s")
    review = selection.get("semantic_review_deadline_monotonic_s")
    self_receipt = frame.get("command_self_state", {}).get("host_receipt_monotonic_s")
    envelope = gate.get("configured_envelope", {})
    age = envelope.get("current_frame_max_age_s", .5)
    reserve = envelope.get("freshness_reserve_s", .02)
    self_age = envelope.get("self_state_max_age_s", .12)
    if not all(finite(value) for value in (receipt, review, self_receipt, age, reserve, self_age)):
        issues.append("UNKNOWN_EXECUTION_FRESHNESS_EVIDENCE")
    else:
        if age > .5 or reserve < .02 or self_age > .12:
            issues.append("GATE_FRESHNESS_CONFIG_EXCEEDS_FROZEN_LIMITS")
        if dispatch+duration > min(receipt+min(age, .5), review)-max(reserve, .02)+1e-9:
            issues.append("TARGET_COMMAND_OUTLIVES_CURRENT_OBSERVATION")
        if dispatch < self_receipt-1e-9 or dispatch > self_receipt+min(self_age, .12)+1e-9:
            issues.append("TARGET_COMMAND_AFTER_FRESH_SELF_STATE_DEADLINE")
    return sorted(set(issues))


def lease_clear_audit(command, frame, events):
    """Audit explicit zero replacement; duration never implies physical stop."""
    result = {"status": "NOT_NONZERO_TARGET_COMMAND", "issues": [],
        "declared_lease_end_s": None, "actual_zero_clear_s": None,
        "first_own_state_replacement_s": None, "clear_after_declared_lease_s": None,
        "clear_after_observation_deadline_s": None, "max_stop_sample_gap_s": None}
    if command.get("purpose") != TARGET_PURPOSE or not vector(command.get("velocity_xy_m_s"), 2) or math.hypot(*command["velocity_xy_m_s"]) <= 1e-6:
        return result
    result["status"] = "UNKNOWN_EXPLICIT_STOP"
    if frame is None:
        result["issues"].append("TARGET_LEASE_STOP_FRAME_MISSING")
        return result
    dispatch, duration = command.get("dispatch_monotonic_s"), command.get("duration_s")
    if not finite(dispatch) or not finite(duration):
        result["issues"].append("TARGET_LEASE_STOP_COMMAND_TIME_INVALID")
        return result
    declared_end = dispatch+duration
    result["declared_lease_end_s"] = declared_end
    stop = frame.get("lease_stop", {})
    event_stops = [row for row in events if row.get("kind") == "TARGET_LEASE_STOP"
        and row.get("target_frame_id") == command.get("target_frame_id")]
    if len(event_stops) != 1 or any(event_stops[0].get(key) != value for key, value in stop.items()):
        result["issues"].append("TARGET_LEASE_STOP_EVENT_RECEIPT_MISMATCH")
    clear = stop.get("target_setpoint_cleared_monotonic_s")
    completed = stop.get("completed_monotonic_s")
    if not finite(clear) or not finite(completed) or clear < dispatch or completed < clear:
        result["issues"].append("TARGET_EXPLICIT_ZERO_CLEAR_TIME_UNKNOWN_OR_INVALID")
        return result
    next_target = min((row["dispatch_monotonic_s"] for row in events if row.get("kind") == "COMMAND"
        and row.get("purpose") == TARGET_PURPOSE and finite(row.get("dispatch_monotonic_s"))
        and row["dispatch_monotonic_s"] > dispatch), default=completed+1e-8)
    replacements = [row for row in events if row.get("kind") == "COMMAND"
        and row.get("purpose", "").startswith("OWN_STATE") and finite(row.get("dispatch_monotonic_s"))
        and dispatch <= row["dispatch_monotonic_s"] < next_target]
    if replacements:
        result["first_own_state_replacement_s"] = min(row["dispatch_monotonic_s"] for row in replacements)
    matching_zero = [row for row in replacements if vector(row.get("velocity_xy_m_s"), 2)
        and math.hypot(*row["velocity_xy_m_s"]) <= 1e-12 and abs(row["dispatch_monotonic_s"]-clear) <= 1e-8]
    # Windows host monotonic receipts can share one millisecond tick. Multiple
    # equivalent zero dispatch records at that tick establish the same clear;
    # the log lacks unique command IDs, so timestamp multiplicity is retained.
    result["actual_zero_dispatch_records_at_clear_timestamp"] = len(matching_zero)
    if not matching_zero:
        result["issues"].append("TARGET_CLEAR_NOT_BOUND_TO_ACTUAL_OWN_ZERO_COMMAND")
    result.update(status="EXPLICIT_ZERO_CLEAR_RECORDED", actual_zero_clear_s=clear,
        clear_after_declared_lease_s=clear-declared_end)
    if not finite(stop.get("declared_lease_end_monotonic_s")) or abs(stop["declared_lease_end_monotonic_s"]-declared_end) > 1e-8:
        result["issues"].append("TARGET_LEASE_DECLARED_END_MISMATCH")
    end_deadline = frame.get("motion_gate", {}).get("command_end_deadline_monotonic_s")
    receipt = frame.get("capture_received_host_s")
    review = frame.get("observer_result", {}).get("temporal_identity", {}).get("semantic_review_deadline_monotonic_s")
    reserve = frame.get("motion_gate", {}).get("configured_envelope", {}).get("freshness_reserve_s", .02)
    if all(finite(value) for value in (receipt, review, reserve)):
        computed_deadline = min(receipt+.5, review)-max(reserve, .02)
        result["independently_computed_observation_end_deadline_s"] = computed_deadline
        if finite(end_deadline):
            end_deadline = min(end_deadline, computed_deadline)
        else:
            end_deadline = computed_deadline
    if finite(end_deadline):
        result["clear_after_observation_deadline_s"] = clear-end_deadline
        if clear > end_deadline+1e-9:
            result["issues"].append("TARGET_EXPLICIT_CLEAR_AFTER_OBSERVATION_DEADLINE")
    else:
        result["issues"].append("TARGET_CLEAR_OBSERVATION_DEADLINE_UNKNOWN")
    samples = stop.get("samples", [])
    stamps = [row.get("receipt_monotonic_s") for row in samples]
    if not samples or any(not finite(stamp) for stamp in stamps) or any(right <= left for left, right in zip(stamps, stamps[1:])):
        result["issues"].append("TARGET_STOP_SAMPLE_TIMES_MISSING_OR_REGRESSED")
    else:
        result["max_stop_sample_gap_s"] = max((right-left for left, right in zip(stamps, stamps[1:])), default=0.)
        result["stop_sample_count"] = len(samples)
        result["last_sample_xy_speed_m_s"] = math.hypot(*samples[-1]["velocity_xy_m_s"]) if vector(samples[-1].get("velocity_xy_m_s"), 2) else None
    result["scope"] = "actual logged own-state zero dispatch and sampled velocity; declared duration is not automatic setpoint expiry"
    return result


def stage_metrics(report, events):
    states = [row for row in events if row.get("kind") == "SELF_STATE" and row.get("stage") == STAGE]
    states += [row["command_self_state"] for row in report.get("frames", []) if "command_self_state" in row]
    for key in ("low_speed_stage_start_state", "low_speed_stage_end_state"):
        if key in report:
            states.append(report[key])
    known, rejected = {}, []
    for row in states:
        stamp = row.get("host_receipt_monotonic_s")
        if not finite(stamp) or not all(vector(row.get(key), 3) for key in (
                "position_world_ned_m", "velocity_ned_m_s", "acceleration_ned_m_s2")):
            rejected.append("INVALID_STAGE_SELF_STATE")
            continue
        known[(stamp, row.get("state_timestamp_ns"))] = row
    states = sorted(known.values(), key=lambda row: row["host_receipt_monotonic_s"])
    def maxima(function):
        return max((function(row) for row in states), default=None)
    z = report.get("configuration", {}).get("world_z_ned_m", -3.)
    path = sum(np.linalg.norm(np.asarray(right["position_world_ned_m"][:2])-left["position_world_ned_m"][:2])
        for left, right in zip(states, states[1:]))
    finite_diff = []
    for left, right in zip(states, states[1:]):
        dt = right["host_receipt_monotonic_s"]-left["host_receipt_monotonic_s"]
        if dt > 0:
            finite_diff.append(float(np.linalg.norm(np.asarray(right["velocity_ned_m_s"])-left["velocity_ned_m_s"])/dt))
    return {"status": "SAMPLED_LOW_SPEED_STAGE" if states else "NO_STAGE_STATE_EVIDENCE",
        "self_state_samples": len(states), "rejected_state_records": rejected,
        "max_actual_xy_speed_m_s": maxima(lambda row: math.hypot(*row["velocity_ned_m_s"][:2])),
        "max_actual_total_speed_m_s": maxima(lambda row: float(np.linalg.norm(row["velocity_ned_m_s"]))),
        "max_estimated_acceleration_m_s2": maxima(lambda row: float(np.linalg.norm(row["acceleration_ned_m_s2"]))),
        "max_host_receipt_finite_difference_acceleration_m_s2": max(finite_diff, default=None),
        "max_fixed_height_error_m": maxima(lambda row: abs(row["position_world_ned_m"][2]-z)),
        "sampled_xy_path_length_m": float(path) if states else None,
        "measured_host_receipt_span_s": states[-1]["host_receipt_monotonic_s"]-states[0]["host_receipt_monotonic_s"] if states else None,
        "max_self_state_sample_gap_s": max((right["host_receipt_monotonic_s"]-left["host_receipt_monotonic_s"] for left, right in zip(states, states[1:])), default=None),
        "command_speed_cap_m_s": report.get("configuration", {}).get("command_speed_cap_m_s"),
        "stage_sample_rows": states,
        "scope": "PERCEPTION_XY and explicitly recorded stage bounds/current self state; startup and landing excluded",
        "limitations": "sampled positions can underestimate path; estimated acceleration and host timing are not continuous physical safety proof",
        "state_record_coverage": "execution SELF_STATE plus per-frame command_self_state; pre-planning query is not saved by this runner",
        "sample_gap_interpretation": "gap between saved query receipts; not a complete query census or proof of a 50ms controller deadline"}


def recovery_metrics(report, frame_rows):
    transitions = {row["phase_id"]: row.get("completed_monotonic_s") for row in report.get("phase_transitions", [])}
    result = {"hidden_transition_completed_s": transitions.get("hidden"), "return_transition_completed_s": transitions.get("return"),
        "loss_receipt_delay_from_actual_hidden_s": None, "recovery_completion_delay_from_actual_return_s": None,
        "recovery_identity_availability_delay_from_actual_return_s": None,
        "pre_hide_track_established": False, "pre_hide_track_qualification": "TRANSITIONS_NOT_VERIFIED",
        "status": "NOT_APPLICABLE_NO_HIDE_RETURN"}
    if "hidden" not in transitions or "return" not in transitions:
        if report.get("configuration", {}).get("mode") == "SHORT_XY":
            result["status"] = "INTERRUPTED_OR_TRANSITION_EVIDENCE_MISSING"
        return result
    hidden, returned = transitions["hidden"], transitions["return"]
    if not finite(hidden) or not finite(returned) or returned <= hidden:
        result["status"] = "INVALID_ACTUAL_TRANSITION_TIMES"
        return result
    # A hidden-phase no-output is a loss only if a correct chain was established
    # immediately before hiding. Source availability is the observer completion
    # receipt, not later command execution/full-frame completion. Older correct
    # output cannot override a subsequent lost output or a stale source frame.
    gap = report.get("observer_close", {}).get("tracker_config", {}).get("max_link_gap_s", .5)
    gap = min(gap, .5) if finite(gap) and gap > 0 else None
    available = [row for row in frame_rows if finite(row.get("readback_s"))
        and finite(row.get("observer_completed_s"))
        and row["readback_s"] <= row["observer_completed_s"] <= hidden]
    def correct_track(row):
        return row.get("identity", {}).get("status") == "CORRECT" \
            and row.get("temporal_status") == "TEMPORAL_LOCAL_ASSOCIATION_VALID"
    prior = max(available, key=lambda row: (row["readback_s"], row["observer_completed_s"]), default=None)
    earlier_correct = any(correct_track(row) for row in available)
    result.update(pre_hide_available_output_frames=len(available), pre_hide_max_link_gap_s=gap,
        pre_hide_reference_frame_id=prior.get("frame_id") if prior else None,
        pre_hide_reference_readback_s=prior.get("readback_s") if prior else None,
        pre_hide_reference_observer_completed_s=prior.get("observer_completed_s") if prior else None,
        pre_hide_source_age_at_hidden_s=hidden-prior["readback_s"] if prior else None)
    qualification = "NO_CORRECT_PRE_HIDE_TRACK"
    if prior and correct_track(prior):
        if gap is None:
            qualification = "PRE_HIDE_LINK_GAP_CONFIGURATION_UNKNOWN"
        elif hidden-prior["readback_s"] > gap+1e-9:
            qualification = "PRE_HIDE_TRACK_STALE_AT_HIDE"
        else:
            qualification = "CORRECT_FRESH_TRACK_ESTABLISHED_BEFORE_HIDE"
            result["pre_hide_track_established"] = True
    elif prior and earlier_correct:
        qualification = "LOSS_ALREADY_PRESENT_BEFORE_HIDE" if prior.get("temporal_status") == "NO_TARGET_OUTPUT" \
            else "LATEST_PRE_HIDE_TRACK_NOT_CORRECT_OR_UNVERIFIED"
    result["pre_hide_track_qualification"] = qualification
    losses = [row for row in frame_rows if row.get("actual_phase") == "hidden"
        and row.get("temporal_status") != "TEMPORAL_LOCAL_ASSOCIATION_VALID" and finite(row.get("readback_s"))]
    recoveries = [row for row in frame_rows if row.get("actual_phase") == "return"
        and row.get("identity", {}).get("status") == "CORRECT"
        and row.get("temporal_status") == "TEMPORAL_LOCAL_ASSOCIATION_VALID"
        and finite(row.get("decision_completed_s"))]
    established = result["pre_hide_track_established"]
    if established and losses:
        result["loss_receipt_delay_from_actual_hidden_s"] = min(row["readback_s"] for row in losses)-hidden
    if established and losses and recoveries:
        result["recovery_completion_delay_from_actual_return_s"] = min(row["decision_completed_s"] for row in recoveries)-returned
        available = [row["observer_completed_s"] for row in recoveries if finite(row.get("observer_completed_s"))]
        if available:
            result["recovery_identity_availability_delay_from_actual_return_s"] = min(available)-returned
    if not established:
        result["status"] = "CORRECT_INITIALIZATION_OBSERVED_AFTER_RETURN" if recoveries and not earlier_correct \
            else "CORRECT_RETURN_OUTPUT_WITHOUT_PRE_HIDE_TRACK" if recoveries else "RECOVERY_NOT_APPLICABLE_NO_ESTABLISHED_PRE_HIDE_TRACK"
    elif not losses:
        result["status"] = "CORRECT_RETURN_TRACK_WITHOUT_OBSERVED_HIDDEN_LOSS" if recoveries else "NO_OBSERVED_HIDDEN_LOSS_OR_CORRECT_RETURN"
    else:
        result["status"] = "RECOVERY_OBSERVED" if recoveries else "NO_OBSERVED_CORRECT_RECOVERY"
    return_rows = [row for row in frame_rows if row.get("actual_phase") == "return"]
    correct_rows = [row for row in return_rows if row.get("identity", {}).get("status") == "CORRECT"
        and row.get("temporal_status") == "TEMPORAL_LOCAL_ASSOCIATION_VALID"]
    result["return_correct_output_frames"] = len(correct_rows)
    result["return_other_frames"] = len(return_rows)-len(correct_rows)
    lengths, span, count, last, previously_correct, post_loss = [], 0., 0, None, False, 0
    for row in return_rows:
        correct = row in correct_rows and finite(row.get("readback_s"))
        if correct and (last is None or row["readback_s"]-last <= .5):
            if last is not None:
                span += row["readback_s"]-last
            count += 1
            last = row["readback_s"]
        else:
            if count:
                lengths.append((count, span))
            if previously_correct and not correct:
                post_loss += 1
            count, span, last = (1, 0., row["readback_s"]) if correct else (0, 0., None)
        previously_correct = correct
    if count:
        lengths.append((count, span))
    result["longest_consecutive_correct_return_frames"] = max((count for count, span in lengths), default=0)
    result["longest_correct_return_readback_span_s"] = max((span for count, span in lengths), default=0.)
    result["post_recovery_lost_output_transitions"] = post_loss
    result["return_continuity_scope"] = "consecutive saved correct outputs, readback gaps <=.5s; brief recovery is not sustained tracking"
    result["time_scope"] = "same-host actual object transition receipt to captured readback/decision completion; not exposure time"
    result["recovery_qualification_scope"] = "fresh last available correct pre-hide output, then observed hidden loss; never-established or already-lost output is not a newly caused loss"
    return result


def source_checks(plan, launcher_dir, report):
    checks = []
    checks.append(verify_file(report["plan"], report["plan_sha256"], "runner_plan"))
    snapshot = launcher_dir/"source_snapshot"
    checks.append(verify_file(snapshot/"plan.json", report["plan_sha256"], "launch_plan_snapshot"))
    inventory = plan.get("source_inventory", [])
    mapped = {row.get("relative_path"): row.get("sha256") for row in inventory}
    checks.append({"label": "source_inventory_map", "status": "PASS" if mapped == plan.get("source_sha256") and len(mapped) == len(inventory) else "FAIL"})
    for item in inventory:
        relative = item.get("relative_path", "")
        if not SOURCE_PATTERN.fullmatch(relative):
            checks.append({"label": "source_path_allowlist", "status": "FAIL"})
            continue
        for label, path in (("prepared_source_snapshot", item["snapshot_path"]),
            ("launch_source_snapshot", snapshot/relative)):
            checks.append(verify_file(path, item["sha256"], label))
    legacy = plan.get("legacy_source_inventory", [])
    checks.append({"label": "legacy_twelve_count", "status": "PASS" if len(legacy) == 12 else "FAIL"})
    for item in legacy:
        name = Path(item["path"]).name
        if not name.endswith(".py"):
            checks.append({"label": "legacy_path_allowlist", "status": "FAIL"})
            continue
        for label, path in (("legacy_prepared_snapshot", item["snapshot_path"]),
            ("legacy_launch_snapshot", snapshot/"legacy_v6p2"/name)):
            checks.append(verify_file(path, item["sha256"], label))
    for key in ("settings", "detector_config", "training_receipt", "weights", "legacy_freeze"):
        path, expected = plan[key+"_path"], plan[key+"_sha256"]
        checks.append(verify_file(path, expected, "prepared_"+key))
        launch_name = "specialist_best.pt" if key == "weights" else Path(path).name
        checks.append(verify_file(snapshot/launch_name, expected, "launch_"+key))
    if canonical_sha(report.get("configuration")) != canonical_sha(plan):
        checks.append({"label": "runner_config_vs_frozen_plan", "status": "FAIL"})
    else:
        checks.append({"label": "runner_config_vs_frozen_plan", "status": "PASS"})
    return checks


def discover_media_without_ledger(flight_dir, plan, known_frame_ids):
    """Recover file existence only; never reconstruct lost observer decisions."""
    rows = []
    for slot in plan.get("planned_slots", []):
        frame_id = f"{plan['session_id']}_f{slot['slot_index']:03d}"
        if frame_id in known_frame_ids:
            continue
        paths = {kind: flight_dir/"frames"/(frame_id+suffix) for kind, suffix in (
            ("rgb", "_rgb.png"), ("depth", "_depth.npy"), ("segmentation", "_seg.png"))}
        existing = {kind: path for kind, path in paths.items() if path.exists()}
        if existing:
            rows.append({"frame_id": frame_id, "slot_index": slot["slot_index"],
                "frame_sequence": slot["slot_index"], "session_id": plan["session_id"],
                "phase_id": slot["phase_id"], "status": "MEDIA_SAVED_DECISION_LEDGER_LOST",
                "media": {kind: {"path": str(path), "sha256": digest(path),
                    "hash_scope": "computed_at_independent_analysis_no_original_seal"} for kind, path in existing.items()}})
    return rows


def analyze_run(run_path):
    path = Path(run_path).resolve()
    run_path_resolved = path
    launcher_dir = path if path.is_dir() else path.parent
    if path.is_file() and path.name == "report.json":
        report_path = path
        if path.parent.name == "flight":
            launcher_dir = path.parent.parent
    else:
        report_path = path/"flight/report.json" if (path/"manifest.json").exists() else path/"report.json"
    manifest_path = launcher_dir/"manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not report_path.exists():
        plan = read_json(manifest["plan"]) if manifest.get("plan") else {}
        return {"run": str(path), "status": "RUNNER_REPORT_MISSING", "launcher_status": manifest.get("status"),
            "new_flights": None, "formal_efficacy_results": False, "source_integrity_status": "UNKNOWN",
            "planned_slots": len(plan.get("planned_slots", [])), "captured_frames": 0,
            "slot_status_counts": {"INTERRUPTED_NOT_REACHED": len(plan.get("planned_slots", []))},
            "unknowns": ["No runner report; simulator launch is not evidence of a flight"],
            "audit_issues": [], "frame_rows": [], "command_rows": [], "planned_slot_rows": [],
            "low_speed_stage": stage_metrics({}, []), "source_receipts": []}
    report = read_json(report_path)
    plan = read_json(report["plan"])
    checks = source_checks(plan, launcher_dir, report)
    current_source_differences = [row["relative_path"] for row in plan.get("source_inventory", [])
        if SOURCE_PATTERN.fullmatch(row["relative_path"]) and verify_file(row["path"], row["sha256"], "live_after_freeze")["status"] != "PASS"]
    if manifest:
        checks.extend([verify_file(report_path, manifest.get("report_sha256"), "launcher_runner_report"),
            verify_file(report["plan"], manifest.get("plan_sha256"), "launcher_plan")])
        if manifest.get("fresh_ground_closure_path"):
            checks.append(verify_file(manifest["fresh_ground_closure_path"], manifest.get("fresh_ground_closure_sha256"), "fresh_ground_closure"))
    execution = report.get("execution_log", {})
    log_path = Path(execution.get("path", report_path.parent/"execution.jsonl"))
    log_check = verify_file(log_path, execution.get("sha256"), "execution_jsonl")
    if log_path.exists() and execution.get("sha256") is None:
        log_check["status"] = "OBSERVED_BYTES_WITHOUT_FINAL_RUNNER_HASH_SEAL"
    checks.append(log_check)
    events, unknowns = [], []
    if log_check["status"] in {"PASS", "OBSERVED_BYTES_WITHOUT_FINAL_RUNNER_HASH_SEAL"}:
        for number, line in enumerate(log_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            try:
                events.append(json.loads(line))
            except (ValueError, TypeError):
                unknowns.append(f"MALFORMED_EXECUTION_LINE_{number}")
    else:
        unknowns.append("UNVERIFIED_EXECUTION_LOG_NOT_USED")
    if log_check["status"] == "OBSERVED_BYTES_WITHOUT_FINAL_RUNNER_HASH_SEAL":
        unknowns.append("FINAL_EXECUTION_HASH_SEAL_MISSING_SAVED_BYTES_ANALYZED_WITH_EXPLICIT_LIMIT")
    frames, frame_rows, issues = list(report.get("frames", [])), [], []
    ledger_path = report_path.parent/"frame_results.jsonl"
    if ledger_path.exists():
        ledger_rows = []
        for number, line in enumerate(ledger_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            try:
                ledger_rows.append(json.loads(line))
            except ValueError:
                unknowns.append(f"MALFORMED_FRAME_LEDGER_LINE_{number}")
        known = {row.get("frame_id"): row for row in frames}
        for row in ledger_rows:
            frame_id = row.get("frame_id")
            if frame_id in known:
                if canonical_sha(known[frame_id]) != canonical_sha(row):
                    issues.append({"frame_id": frame_id, "issue": "FINAL_REPORT_VS_FRAME_LEDGER_CONTENT_MISMATCH"})
            else:
                frames.append(row)
                known[frame_id] = row
                unknowns.append("FRAME_FROM_APPEND_LEDGER_WITHOUT_FINAL_REPORT_COPY")
        checks.append({"label": "frame_results_ledger", "path": str(ledger_path),
            "actual_sha256": digest(ledger_path), "status": "PASS" if not any(item["issue"] == "FINAL_REPORT_VS_FRAME_LEDGER_CONTENT_MISMATCH" for item in issues) else "FAIL"})
    orphans = discover_media_without_ledger(report_path.parent, plan, {row.get("frame_id") for row in frames})
    frames.extend(orphans)
    if orphans:
        unknowns.append("SAVED_MEDIA_WITHOUT_OBSERVER_DECISION_LEDGER_IDENTITY_UNKNOWN")
    seen = set()
    previous_sequence = previous_receipt = None
    for frame in frames:
        frame_id = frame.get("frame_id")
        duplicate = frame_id in seen
        seen.add(frame_id)
        media_status, truth = {}, None
        for kind in ("rgb", "depth", "segmentation"):
            item = frame.get("media", {}).get(kind, {})
            check = verify_file(item.get("path", "__missing_media__"), item.get("sha256"), "frame_"+kind)
            if frame.get("status") == "MEDIA_SAVED_DECISION_LEDGER_LOST" and check["status"] == "PASS":
                check["status"] = "CURRENT_BYTES_LOCKED_WITHOUT_ORIGINAL_FRAME_HASH_SEAL"
            checks.append(check)
            media_status[kind] = check["status"]
        if all(value == "PASS" for value in media_status.values()) and frame.get("status") != "MEDIA_SAVED_DECISION_LEDGER_LOST":
            rgb = cv2.imread(frame["media"]["rgb"]["path"], cv2.IMREAD_COLOR)
            seg = cv2.imread(frame["media"]["segmentation"]["path"], cv2.IMREAD_COLOR)
            depth = np.load(frame["media"]["depth"]["path"], allow_pickle=False)
            dims = tuple(frame.get("image_size_wh", [0, 0])[::-1])
            if rgb is None or seg is None or rgb.shape != seg.shape or rgb.shape[:2] != dims or depth.shape != dims:
                media_status["dimensions"] = "FAIL"
            else:
                try:
                    truth = mask_truth(seg, report["segmentation_palette_rgb"])
                except (ValueError, KeyError):
                    media_status["palette"] = "FAIL"
        selection = frame.get("observer_result", {}).get("temporal_identity", {})
        row_issues = [] if frame.get("status") == "MEDIA_SAVED_DECISION_LEDGER_LOST" else binding_issues(frame, report)
        if frame.get("frame_sequence") != frame.get("slot_index"):
            row_issues.append("ACQUISITION_SEQUENCE_DIFFERS_FROM_PLANNED_SLOT")
        if frame_id != f"{plan['session_id']}_f{frame.get('slot_index', -1):03d}":
            row_issues.append("ACQUISITION_FRAME_ID_DIFFERS_FROM_PLANNED_SLOT")
        sequence, receipt = frame.get("frame_sequence"), frame.get("capture_received_host_s")
        if finite(sequence) and finite(receipt):
            if previous_sequence is not None and (sequence <= previous_sequence or receipt <= previous_receipt):
                row_issues.append("FRAME_SEQUENCE_OR_READBACK_RECEIPT_REGRESSION")
            previous_sequence, previous_receipt = sequence, receipt
        if duplicate:
            row_issues.append("DUPLICATE_FRAME_ID")
        identity = identity_score(selection, truth)
        row = {"frame_id": frame_id, "frame_sequence": frame.get("frame_sequence"),
            "slot_index": frame.get("slot_index"), "saved_frame_status": frame.get("status"),
            "capture_started_s": frame.get("capture_started_host_s"), "readback_s": frame.get("capture_received_host_s"),
            "decision_completed_s": frame.get("decision_completed_monotonic_s"),
            "observer_completed_s": frame.get("observer_result", {}).get("process_completed_host_monotonic_s"),
            "planning_wall_s": frame.get("planning_wall_s"),
            "gate_decision_s": frame.get("motion_gate", {}).get("decision_monotonic_s"),
            "decision_ledger_missing": frame.get("status") == "MEDIA_SAVED_DECISION_LEDGER_LOST",
            "nominal_phase": frame.get("phase_id"), "actual_phase": current_phase(report,
                frame.get("capture_started_host_s"), frame.get("capture_received_host_s")),
            "media_status": media_status, "truth_recomputed": truth, "identity": identity,
            "temporal_status": selection.get("status", "UNKNOWN"), "tracker_state": selection.get("state"),
            "surface_status": frame.get("surface", {}).get("status", "UNKNOWN"),
            "gate_status": frame.get("motion_gate", {}).get("status", "UNKNOWN"),
            "binding_issues": row_issues}
        if truth is not None and truth["target"]["visible_pixel_count"] == 0 and selection.get("status") == "TEMPORAL_LOCAL_ASSOCIATION_VALID":
            row_issues.append("TEMPORAL_SELECTION_WHEN_TARGET_MASK_HIDDEN")
        for issue in row_issues:
            issues.append({"frame_id": frame_id, "issue": issue})
        frame_rows.append(row)
    raw_by_id = {frame.get("frame_id"): frame for frame in frames}
    analyzed_by_id = {row["frame_id"]: row for row in frame_rows}
    command_rows = []
    commands = [row for row in events if row.get("kind") == "COMMAND" and row.get("stage") == STAGE]
    for index, command in enumerate(commands):
        frame = raw_by_id.get(command.get("target_frame_id"))
        if frame is not None and frame.get("status") == "MEDIA_SAVED_DECISION_LEDGER_LOST":
            command_issues = []
            unknowns.append("TARGET_COMMAND_IDENTITY_GATE_AND_STOP_BINDING_LOST_WITH_FRAME_LEDGER")
        else:
            command_issues = command_audit(command, frame, report)
        xy = command.get("velocity_xy_m_s")
        magnitude = math.hypot(*xy) if vector(xy, 2) else None
        target = command.get("purpose") == TARGET_PURPOSE
        truth = analyzed_by_id.get(command.get("target_frame_id"), {}).get("truth_recomputed")
        if target and magnitude is not None and magnitude > 1e-6 and truth is not None and truth["target"]["visible_pixel_count"] == 0:
            command_issues.append("NONZERO_TARGET_COMMAND_WHEN_TARGET_MASK_HIDDEN")
        if target and magnitude is not None and magnitude > 1e-6:
            identity_status = analyzed_by_id.get(command.get("target_frame_id"), {}).get("identity", {}).get("status")
            if identity_status in {"WRONG_DISTRACTOR", "AMBIGUOUS_OR_UNMATCHED", "NO_OUTPUT"}:
                command_issues.append("NONZERO_TARGET_COMMAND_WITHOUT_INDEPENDENT_CORRECT_TARGET")
            elif identity_status != "CORRECT":
                unknowns.append("NONZERO_TARGET_COMMAND_INDEPENDENT_IDENTITY_UNKNOWN")
        if target and frame is not None and frame.get("status") != "MEDIA_SAVED_DECISION_LEDGER_LOST" and frame.get("command") != command:
            command_issues.append("FRAME_EXECUTION_COMMAND_RECORD_MISMATCH")
        row = dict(command, command_index=index, command_norm_m_s=magnitude, audit_issues=sorted(set(command_issues)))
        row["lease_clear_audit"] = lease_clear_audit(command, frame, events)
        row["audit_issues"] = sorted(set(row["audit_issues"] + row["lease_clear_audit"]["issues"]))
        command_rows.append(row)
        issues += [{"command_index": index, "issue": issue} for issue in row["audit_issues"]]
    frame_by_slot = {row.get("slot_index"): row for row in frame_rows}
    observed_slots = {row["slot_index"]: row for row in report.get("slots", [])}
    slot_rows = []
    for declared in plan.get("planned_slots", []):
        index = declared["slot_index"]
        observed, frame = observed_slots.get(index, {}), frame_by_slot.get(index)
        status = "INTERRUPTED_NOT_REACHED"
        if frame is not None:
            status = "CAPTURED_COMPLETED" if frame["saved_frame_status"] == "FRAME_COMPLETED" else "SAVED_MEDIA_DECISION_UNKNOWN" if frame["decision_ledger_missing"] else "FRAME_STARTED_INCOMPLETE"
        elif str(observed.get("status", "")).startswith("NOT_CAPTURED"):
            status = "NOT_CAPTURED_SLOT_DEADLINE"
        late = None
        deadline = observed.get("deadline_monotonic_s")
        if frame is not None and finite(deadline) and finite(frame["decision_completed_s"]):
            late = frame["decision_completed_s"] > deadline
        observer_late = None
        if frame is not None and finite(deadline) and finite(frame["observer_completed_s"]):
            observer_late = frame["observer_completed_s"] > deadline
        gate_late = None
        if frame is not None and finite(deadline) and finite(frame["gate_decision_s"]):
            gate_late = frame["gate_decision_s"] > deadline
        timed_commands = []
        epoch = report.get("epoch_monotonic_s")
        if finite(epoch):
            timed_commands = [row for row in command_rows if finite(row.get("dispatch_monotonic_s"))
                and epoch+declared["scheduled_start_s"] <= row["dispatch_monotonic_s"] < epoch+declared["scheduled_end_s"]]
        slot_rows.append(dict(declared, observed_status=status, frame_id=frame.get("frame_id") if frame else None,
            decision_late=late, observer_late=observer_late, gate_decision_late=gate_late,
            command_counts={"status": "KNOWN_NOMINAL_TIME_GRID" if finite(epoch) else "UNKNOWN_TIME_GRID",
                "target_commands": sum(row.get("purpose") == TARGET_PURPOSE for row in timed_commands),
                "nonzero_target_commands": sum(row.get("purpose") == TARGET_PURPOSE and row["command_norm_m_s"] is not None and row["command_norm_m_s"] > 1e-6 for row in timed_commands),
                "zero_commands": sum(row["command_norm_m_s"] is not None and row["command_norm_m_s"] <= 1e-6 for row in timed_commands),
                "own_state_commands": sum(row.get("purpose", "").startswith("OWN_STATE") for row in timed_commands)},
            identity_status=frame["identity"]["status"] if frame else "UNKNOWN_UNCAPTURED_OR_INTERRUPTED"))
    stage = stage_metrics(report, events)
    targets = [row for row in command_rows if row.get("purpose") == TARGET_PURPOSE]
    clear_audits = [row["lease_clear_audit"] for row in targets if row["lease_clear_audit"]["status"] != "NOT_NONZERO_TARGET_COMMAND"]
    nonzero = [row for row in command_rows if row["command_norm_m_s"] is not None and row["command_norm_m_s"] > 1e-6]
    source_statuses = Counter(row["status"] for row in checks)
    correct = sum(row["identity_status"] == "CORRECT" for row in slot_rows)
    unknown = sum(row["identity_status"] in {"UNKNOWN_MEDIA", "UNKNOWN_OUTPUT_STATE", "UNKNOWN_UNCAPTURED_OR_INTERRUPTED"} for row in slot_rows)
    supplemental = []
    for name in ("supplemental_ground_recovery.json", "supplemental_ground_recovery_v2.json", "supplemental_owned_closure.json"):
        path = launcher_dir/name
        if path.exists():
            receipt = read_json(path)
            supplemental.append({"path": str(path), "sha256": digest(path), "status": receipt.get("status"),
                "kind": receipt.get("kind"), "engine_closed_after_supplemental_ground_recovery": receipt.get("engine_closed_after_supplemental_ground_recovery"),
                "original_attempt_status": receipt.get("original_attempt_status")})
    runner_status = report.get("status")
    status = "FAIL_INCOMPLETE_FINAL_RUNNER_REPORT" if manifest.get("status") == "FAIL" and runner_status == "RUNNING" else runner_status
    source_integrity = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL_OR_MISSING"
    evidence_reasons = []
    if status != "PASS":
        evidence_reasons.append("ATTEMPT_NOT_SUCCESSFULLY_FINALIZED")
    if source_integrity != "PASS":
        evidence_reasons.append("FROZEN_SOURCE_OR_RUN_OUTPUT_INTEGRITY_INCOMPLETE")
    if issues:
        evidence_reasons.append("RECORDED_BINDING_OR_EXECUTION_ISSUES")
    if unknowns:
        evidence_reasons.append("UNRESOLVED_EVIDENCE_GAPS")
    if not any(row["surface_status"] == "SURFACE_GEOMETRY_ACCEPTED" for row in frame_rows) or not any(row["identity"]["status"] == "CORRECT" for row in frame_rows):
        evidence_reasons.append("NO_INDEPENDENT_CORRECT_SELECTED_SURFACE")
    if plan.get("mode") == "SHORT_XY" and not any(row["command_norm_m_s"] is not None and row["command_norm_m_s"] > 1e-6 for row in targets):
        evidence_reasons.append("NO_NONZERO_TARGET_TRAJECTORY_DISPATCH")
    if report.get("safe_end", {}).get("grounded_and_released") is not True or report.get("safe_end", {}).get("paused") is not True:
        evidence_reasons.append("ORIGINAL_ATTEMPT_SAFE_END_NOT_CONFIRMED")
    if manifest.get("fresh_ground_closure_verified") is not True or manifest.get("engine_closed_after_safe_completion") is not True:
        evidence_reasons.append("FRESH_CLOSURE_OR_ENGINE_COMPLETION_NOT_CONFIRMED")
    for key, ceiling in (("max_actual_xy_speed_m_s", .06), ("max_actual_total_speed_m_s", .15),
        ("max_estimated_acceleration_m_s2", .3), ("max_fixed_height_error_m", .12)):
        if stage[key] is not None and stage[key] > ceiling+1e-9:
            evidence_reasons.append("MEASURED_STAGE_ENVELOPE_EXCEEDED_"+key.upper())
    airborne_events = [row for row in events if row.get("kind") == "SELF_STATE"
        and row.get("stage") == STAGE and row.get("landed") is False]
    return {"run": str(run_path_resolved), "report_path": str(report_path), "report_sha256": digest(report_path),
        "session_id": report.get("session_id"), "mode": plan.get("mode"), "status": status, "original_runner_status": runner_status,
        "launcher_status": manifest.get("status"), "new_flights": report.get("new_flights"),
        "observed_airborne_from_execution": bool(airborne_events),
        "observed_airborne_self_state_samples": len(airborne_events),
        "declared_vs_observed_flight_scope": "new_flights preserves original report declaration; airborne evidence comes from saved PERCEPTION_XY landed=False self states, including failed attempts",
        "safe_end": report.get("safe_end"), "fresh_ground_closure_verified": manifest.get("fresh_ground_closure_verified"),
        "engine_closed_after_safe_completion": manifest.get("engine_closed_after_safe_completion"),
        "errors_preserved": report.get("errors", []), "formal_efficacy_results": False,
        "supplemental_recovery_receipts": supplemental,
        "planned_slots": len(slot_rows), "captured_frames": len(frames),
        "captured_frame_count_scope": "complete/partial frame ledger plus saved media groups; lost decisions remain unknown",
        "media_files_original_hash_verified": sum(row["label"] in {"frame_rgb", "frame_depth", "frame_segmentation"} and row["status"] == "PASS" for row in checks),
        "media_files_locked_at_analysis_without_original_seal": sum(row["status"] == "CURRENT_BYTES_LOCKED_WITHOUT_ORIGINAL_FRAME_HASH_SEAL" for row in checks),
        "slot_status_counts": dict(Counter(row["observed_status"] for row in slot_rows)),
        "late_completed_decisions": sum(row["decision_late"] is True for row in slot_rows),
        "full_frame_completion_timing_scope": "includes command lease and explicit own-state stopping; not the motion-gate decision timestamp",
        "late_completed_decision_unknown_slots": sum(row["decision_late"] is None for row in slot_rows),
        "late_observer_returns": sum(row["observer_late"] is True for row in slot_rows),
        "late_observer_return_unknown_slots": sum(row["observer_late"] is None for row in slot_rows),
        "gate_decisions_late": sum(row["gate_decision_late"] is True for row in slot_rows),
        "gate_decisions_timely": sum(row["gate_decision_late"] is False for row in slot_rows),
        "gate_decision_unknown_slots": sum(row["gate_decision_late"] is None for row in slot_rows),
        "temporal_status_counts": dict(Counter(row["temporal_status"] for row in frame_rows)),
        "surface_status_counts": dict(Counter(row["surface_status"] for row in frame_rows)),
        "planning_wall_seconds": {"recorded_frames": sum(finite(row["planning_wall_s"]) for row in frame_rows),
            "maximum": max((row["planning_wall_s"] for row in frame_rows if finite(row["planning_wall_s"])), default=None)},
        "identity_point_counts": dict(Counter(row["identity"]["status"] for row in frame_rows)),
        "planned_identity_bounds": {"denominator": len(slot_rows), "correct_slots": correct,
            "unknown_slots": unknown, "point_fraction": correct/len(slot_rows) if slot_rows and unknown == 0 else None,
            "lower_bound": correct/len(slot_rows) if slot_rows else None,
            "upper_bound": (correct+unknown)/len(slot_rows) if slot_rows else None,
            "scope": "discrete planned slots; later than deadline decisions separately counted"},
        "command_counts": {"experimental_commands_all": len(command_rows), "target_commands_all": len(targets),
            "target_commands_nonzero": sum(row["command_norm_m_s"] is not None and row["command_norm_m_s"] > 1e-6 for row in targets),
            "zero_commands_all": sum(row["command_norm_m_s"] is not None and row["command_norm_m_s"] <= 1e-6 for row in command_rows),
            "unknown_command_values": sum(row["command_norm_m_s"] is None for row in command_rows),
            "own_state_brake_or_hold_commands": sum(row.get("purpose", "").startswith("OWN_STATE") for row in command_rows),
            "own_state_nonzero_brake_commands": sum(row.get("purpose", "").startswith("OWN_STATE") for row in nonzero),
            "max_commanded_xy_speed_m_s": max((row["command_norm_m_s"] for row in command_rows if row["command_norm_m_s"] is not None), default=None)},
        "target_lease_clear_summary": {"nonzero_target_commands": len(clear_audits),
            "explicit_clear_recorded_count": sum(row["status"] == "EXPLICIT_ZERO_CLEAR_RECORDED" for row in clear_audits),
            "max_clear_after_declared_lease_s": max((row["clear_after_declared_lease_s"] for row in clear_audits if row["clear_after_declared_lease_s"] is not None), default=None),
            "max_clear_after_observation_deadline_s": max((row["clear_after_observation_deadline_s"] for row in clear_audits if row["clear_after_observation_deadline_s"] is not None), default=None),
            "max_stop_sample_gap_s": max((row["max_stop_sample_gap_s"] for row in clear_audits if row["max_stop_sample_gap_s"] is not None), default=None)},
        "source_integrity_status": source_integrity,
        "source_check_status_counts": dict(source_statuses), "source_checks": checks,
        "current_source_differences_after_freeze": current_source_differences,
        "current_source_difference_scope": "later development changes do not invalidate frozen source_bundle/source_snapshot bytes",
        "unsealed_media_frames_with_unknown_identity": len(orphans),
        "evidence_gate_status": "PASS_RECORDED_DEVELOPMENT_INTEGRATION" if not evidence_reasons else "FAIL_OR_INCOMPLETE_DEVELOPMENT_EVIDENCE",
        "evidence_gate_reasons": evidence_reasons,
        "audit_status": "NO_DETECTED_RECORDED_VIOLATION" if not issues else "RECORDED_ISSUES_REQUIRE_REVIEW",
        "audit_issues": issues, "unknowns": sorted(set(unknowns)),
        "actual_transition_recovery": recovery_metrics(report, frame_rows), "low_speed_stage": stage,
        "planned_slot_rows": slot_rows, "frame_rows": frame_rows, "command_rows": command_rows,
        "source_receipts": [{"path": str(report_path), "sha256": digest(report_path)},
            {"path": str(report["plan"]), "sha256": digest(report["plan"])},
            {"path": str(log_path), "sha256": digest(log_path)}] if log_path.exists() else []}


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def plot_runs(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(max(1, len(runs)), 1, figsize=(12, 3.4*max(1, len(runs))), squeeze=False)
    for ax, run in zip(axes.flat, runs):
        for row in run["planned_slot_rows"]:
            index = row["slot_index"]
            status = row["identity_status"]
            color = "#168582" if status == "CORRECT" else "#777777" if status == "NO_OUTPUT" else "#c47729"
            y = 1 if status == "CORRECT" else 0
            ax.scatter(index, y, color=color, marker="s" if row["observed_status"] == "CAPTURED_COMPLETED" else "x", s=22)
            if row["decision_late"] is True:
                ax.scatter(index, 1.3, color="#b34242", marker="v", s=22)
        for row in run["command_rows"]:
            if row.get("purpose") == TARGET_PURPOSE and row["command_norm_m_s"] is not None and row["command_norm_m_s"] > 1e-6:
                frame = next((frame for frame in run["frame_rows"] if frame["frame_id"] == row.get("target_frame_id")), None)
                if frame:
                    ax.scatter(frame["slot_index"], 1.6, marker="+", color="#2155ac", s=35)
        ax.set_title(f"{run.get('session_id', Path(run['run']).name)} | {run['status']}", fontsize=10)
        ax.set_yticks([0, 1, 1.3, 1.6], ["No/unknown output", "Correct mask association", "Late full completion", "Nonzero target command"])
        ax.set_ylim(-.25, 1.9)
        ax.set_xlabel("All frozen planned slot indices (interrupted and missed retained)")
        ax.grid(alpha=.18)
    fig.suptitle("Identity/depth/short XY development pilots; no efficacy inference")
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"pilot_timeline.{suffix}", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, max(1, len(runs)), figsize=(6*max(1, len(runs)), 5), squeeze=False)
    for ax, run in zip(axes.flat, runs):
        states = run["low_speed_stage"]["stage_sample_rows"] if "stage_sample_rows" in run["low_speed_stage"] else []
        if states:
            positions = np.asarray([row["position_world_ned_m"] for row in states])
            ax.plot(positions[:, 0], positions[:, 1], "-", color="#168582", lw=1.2)
            ax.scatter(*positions[0, :2], marker="o", label="stage start", s=35)
            ax.scatter(*positions[-1, :2], marker="x", label="stage end", s=35)
            ax.legend()
        ax.set_title(f"{run.get('mode', 'UNKNOWN')} | {run['status']}")
        ax.set_xlabel("World NED X (m)")
        ax.set_ylabel("World NED Y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.2)
    fig.suptitle("Actual sampled onboard self-position in PERCEPTION_XY only\nStartup/landing excluded; not target truth or planned trajectory")
    fig.tight_layout(rect=(0, 0, 1, .9))
    for suffix in ("png", "pdf"):
        fig.savefig(output/f"pilot_xy.{suffix}", dpi=150)
    plt.close(fig)


def analyze(runs, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = [analyze_run(path) for path in runs]
    report = {"schema_version": 1, "kind": "INDEPENDENT_T1_IDENTITY_DEPTH_XY_PILOT_ANALYSIS",
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_count_all": len(results),
        "status": "DEVELOPMENT_ANALYZED_ALL_ATTEMPTS_RETAINED", "formal_efficacy_results": False,
        "new_analysis_UE_sessions": 0, "new_analysis_flights": 0, "new_API_or_model_calls": 0,
        "runs": results, "limits": ["zero-gate and short-XY integration pilots, not independent efficacy trials",
            "segmentation is used only in this postflight independent evaluation",
            "all planned slots, interruptions, no outputs, late results and failures are retained",
            "readback and self-state timestamps are host receipts, not exact exposure times",
            "sampled motion protection checks do not prove continuous safety",
            "selected visible-depth surfaces are not physical balloon centers"],
        "ai_assistance": "AI assisted readonly independent analysis, tests and scientific plots"}
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for name in ("analyze_t1_identity_flight.py", "test_analyze_t1_identity_flight.py"):
        source = Path(__file__).parent/name
        (snapshot/name).write_bytes(source.read_bytes())
    report["analysis_source_sha256"] = {path.name: digest(path) for path in snapshot.iterdir()}
    save(output/"report.json", report)
    lines = ["# 任务一身份—深度—低速 XY 飞行开发验证", "", "本报告保留本轮全部运行尝试，零指令与短 XY 指令属于集成开发验证。尚未形成四方法对照或独立效能结论。", ""]
    for run in results:
        stage = run["low_speed_stage"]
        lines += [f"## {run.get('session_id', Path(run['run']).name)}", "",
            f"运行状态：{run['status']}；来源审计：{run['source_integrity_status']}。预定 {run['planned_slots']} 个时隙，记录 {run['captured_frames']} 个采集帧。",
            f"时隙结果：{json.dumps(run['slot_status_counts'], ensure_ascii=False)}。",
            f"记录审计问题：{len(run['audit_issues'])} 项；未知项：{len(run['unknowns'])} 项。", ""]
        if "command_counts" in run:
            lines += [f"独立开发证据门：{run['evidence_gate_status']}；原因：{json.dumps(run['evidence_gate_reasons'], ensure_ascii=False)}。",
                f"保存媒体按原始摘要验证 {run['media_files_original_hash_verified']} 个；原决策未封口、仅在本分析锁定字节的媒体 {run['media_files_locked_at_analysis_without_original_seal']} 个。",
                f"帧完成晚于计划截止 {run['late_completed_decisions']} 个，完成时刻未知 {run['late_completed_decision_unknown_slots']} 个；observer 本身晚返回 {run['late_observer_returns']} 个，返回时刻未知 {run['late_observer_return_unknown_slots']} 个。",
                f"实际 motion gate 决策按时 {run['gate_decisions_timely']} 个，晚决策 {run['gate_decisions_late']} 个，时刻未知或漏采 {run['gate_decision_unknown_slots']} 个。上述整帧完成还包含指令租期和自状态停稳，不把整帧晚完成数当作晚 gate 决策数。",
                f"实际指令：{json.dumps(run['command_counts'], ensure_ascii=False)}。",
                f"当前身份点计数：{json.dumps(run['identity_point_counts'], ensure_ascii=False)}。",
                f"低速阶段采样速度最大值 XY={stage['max_actual_xy_speed_m_s']} m/s，总速度={stage['max_actual_total_speed_m_s']} m/s；估计加速度最大值={stage['max_estimated_acceleration_m_s2']} m/s²。",
                f"命令限幅={stage['command_speed_cap_m_s']} m/s，与实测响应分列；高度偏差最大值={stage['max_fixed_height_error_m']} m，采样 XY 航程={stage['sampled_xy_path_length_m']} m。",
                f"显式清除指令审计：{json.dumps(run['target_lease_clear_summary'], ensure_ascii=False)}；低速阶段最大自状态采样间隙={stage['max_self_state_sample_gap_s']} s。",
                f"实际遮挡/返回转换后的失锁与恢复：{json.dumps(run['actual_transition_recovery'], ensure_ascii=False)}。", ""]
    lines += ["启动、起飞稳定和落地恢复不进入上述低速阶段速度、加速度及航程统计。完整失败原因、无输出与未知状态保留在 report.json。", "",
        "自状态间隙按 execution SELF_STATE 与每帧 command_self_state 的保存读回记录计算，规划前那次未保存的查询不在清单中；不据此声称达到 50 ms 控制周期或连续安全。", "",
        "图表由保存的独立解析记录生成，报告、分析源文件与图件均通过摘要收据绑定。未读取凭证文件，也未调用 AirSim、云服务或模型。", "",
        "AI 辅助说明：AI 辅助独立分析代码、契约测试和图表制作，原始运行数据保持不变。", ""]
    (output/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    plot_runs(results, output)
    save(output/"artifact_receipt.json", {"artifacts": [{"path": str(path), "sha256": digest(path)} for path in (
        output/"report.json", output/"summary.md", output/"pilot_timeline.png", output/"pilot_timeline.pdf",
        output/"pilot_xy.png", output/"pilot_xy.pdf")], "analysis_source_sha256": report["analysis_source_sha256"],
        "source_run_report_sha256": [run.get("report_sha256") for run in results]})
    print(json.dumps({"run_count": len(results), "runs": [{"mode": run.get("mode"), "status": run["status"],
        "captured": run["captured_frames"], "planned": run["planned_slots"],
        "source_integrity": run["source_integrity_status"], "audit_issues": len(run["audit_issues"])} for run in results]}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.runs, args.output)
