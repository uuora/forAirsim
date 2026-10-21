"""Independent, denominator-preserving evaluation of grounded identity pilots.

The observer is never invoked. Instance labels come exclusively from saved
evaluation-only visible masks, independently reproduced from segmentation.
Scheduled-slot metrics and point observations are separate: late observations
cannot retrospectively fill a missed interval. Host receipts are not exposures.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("initial_pair", "target_hidden_with_distractor_at_old_target", "return_target_right")
OBSERVABILITY = ("VISUAL_ATTRIBUTE", "UNRESOLVABLE_IDENTICAL")
UNKNOWN = {"UNKNOWN_TRUTH", "UNRESOLVED"}


def digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Report must be a JSON object")
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _box(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    result = tuple(_number(item) for item in value)
    if any(item is None for item in result) or not (result[0] < result[2] and result[1] < result[3]):
        return None
    return result


def _iou(first, second) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0., x2-x1) * max(0., y2-y1)
    union = ((first[2]-first[0])*(first[3]-first[1])
             + (second[2]-second[0])*(second[3]-second[1]) - intersection)
    return intersection / union if union else 0.


def expected_slots(configuration: dict) -> list[dict]:
    """Reconstruct all registered intervals rather than trusting captured rows."""
    if configuration.get("timing_grid") == "BOUNDARY_CLIPPED_0P4":
        from t1_identity_schedule import boundary_clipped_slots
        return [{key:value for key,value in row.items() if key not in
            {"capture_status","not_captured_reason"}} for row in boundary_clipped_slots(configuration)]
    period = _number(configuration.get("minimum_period_s"))
    phases = configuration.get("phases")
    if period is None or period < .2 or not isinstance(phases, list):
        raise ValueError("Invalid scheduled sequence protocol")
    if [item.get("phase_id") for item in phases] != list(PHASES):
        raise ValueError("Three registered phases required")
    rows, start = [], 0.
    for phase, expected_duration in zip(phases, (5., 2., 5.)):
        duration = _number(phase.get("host_duration_s"))
        if duration != expected_duration:
            raise ValueError("Registered phase durations changed")
        count = round(duration / period)
        if count <= 0 or not math.isclose(count*period, duration, abs_tol=1e-9):
            raise ValueError("Intervals must cover each whole phase")
        for index in range(count):
            rows.append({"frame_sequence": len(rows)+1, "phase_id": phase["phase_id"],
                         "planned_start_relative_s": start+index*period,
                         "planned_end_relative_s": start+(index+1)*period})
        start += duration
    return rows


def _mask(segmentation, rgb) -> dict:
    if (not isinstance(rgb, list) or len(rgb) != 3 or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in rgb)):
        raise ValueError("Invalid saved segmentation palette")
    mask = np.all(segmentation == np.asarray(rgb[::-1], np.uint8), axis=2)
    ys, xs = np.nonzero(mask)
    return {"visible_pixel_count": int(mask.sum()), "bbox_xyxy_pixels":
            [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None}


def _verify_frame(frame: dict, report: dict, phase: str) -> tuple[dict | None, list[str], int]:
    problems, verified, media = [], 0, frame.get("media", {})
    loaded = {}
    for kind in ("rgb", "depth", "segmentation"):
        record = media.get(kind)
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            problems.append("MISSING_MEDIA:"+kind)
            continue
        path = Path(record["path"])
        try:
            if digest(path) != record.get("sha256"):
                problems.append("MEDIA_HASH_MISMATCH:"+kind)
                continue
            verified += 1
            loaded[kind] = (np.load(path, allow_pickle=False) if kind == "depth"
                            else cv2.imread(str(path), cv2.IMREAD_COLOR))
        except (OSError, ValueError):
            problems.append("MEDIA_READ_ERROR:"+kind)
    dimensions = frame.get("image_size_wh")
    if (not isinstance(dimensions, list) or len(dimensions) != 2 or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in dimensions)):
        problems.append("INVALID_IMAGE_DIMENSIONS")
    elif len(loaded) == 3:
        expected = (dimensions[1], dimensions[0])
        for kind, array in loaded.items():
            if array is None or array.shape[:2] != expected or (kind != "depth" and array.shape != (*expected, 3)):
                problems.append("MEDIA_DIMENSIONS_MISMATCH:"+kind)
    if problems:
        return None, problems, verified
    palette = report.get("segmentation_palette_rgb", {})
    truth = {}
    try:
        for name, key in (("target", "42"), ("distractor", "43"), ("occluder", "44")):
            truth[name] = _mask(loaded["segmentation"], palette.get(key))
            saved = frame.get("evaluation_only", {}).get(name)
            if not isinstance(saved, dict) or any(saved.get(field) != truth[name][field]
                    for field in ("visible_pixel_count", "bbox_xyxy_pixels")):
                problems.append("SAVED_VISIBLE_MASK_MISMATCH:"+name)
    except ValueError:
        problems.append("INVALID_INSTANCE_PALETTE")
        return None, problems, verified
    target, distractor, occluder = (truth[name]["visible_pixel_count"] for name in ("target", "distractor", "occluder"))
    scene_valid = distractor >= 500 and ((target == 0) if phase == PHASES[1] else (target >= 500 and occluder == 0))
    if not scene_valid:
        problems.append("INTENDED_SCENE_ACCEPTANCE_FAILED")
    if frame.get("phase_id") != phase:
        problems.append("FRAME_PHASE_MISMATCH")
    if frame.get("scene_acceptance", {}).get("status") != ("PASS" if scene_valid else "FAIL"):
        problems.append("SAVED_SCENE_ACCEPTANCE_MISMATCH")
    return (truth if not problems else None), problems, verified


def _decision_label(frame: dict, report: dict, truth: dict) -> tuple[str, dict]:
    observer = frame.get("observer_record", {})
    if observer.get("status") != "RETURNED":
        return "NO_OUTPUT", {"reason": "OBSERVER_NOT_RETURNED", "association_ious": {}}
    decision = observer.get("decision")
    if not isinstance(decision, dict) or not isinstance(decision.get("temporal_identity"), dict):
        return "UNRESOLVED", {"reason": "OBSERVER_SCHEMA_UNRESOLVED", "association_ious": {}}
    tracker, semantic_frame = decision["temporal_identity"], decision.get("semantic_frame", {})
    bindings = {"session_id": report.get("session_id"), "target_generation": 1,
                "frame_id": frame["frame_id"], "frame_sequence": frame["frame_sequence"],
                "image_sha256": frame["media"]["rgb"]["sha256"]}
    if any(semantic_frame.get(key) != value for key, value in bindings.items()):
        return "UNRESOLVED", {"reason": "SEMANTIC_FRAME_BINDING_MISMATCH", "association_ious": {}}
    if semantic_frame.get("readback_receipt_monotonic_s") != frame.get("capture_received_host_s"):
        return "UNRESOLVED", {"reason": "SEMANTIC_RECEIPT_BINDING_MISMATCH", "association_ious": {}}
    status = tracker.get("status")
    if status == "NO_TARGET_OUTPUT" and tracker.get("selected_candidate") is None:
        pending = (tracker.get("state") == "RECOVERY_PENDING" or
                   decision.get("semantic_poll", {}).get("status") in {"PENDING", "WAITING"})
        return ("PENDING" if pending else "NO_OUTPUT"), {"reason": tracker.get("reason"), "association_ious": {}}
    if status != "TEMPORAL_LOCAL_ASSOCIATION_VALID":
        return "UNRESOLVED", {"reason": "UNRECOGNIZED_TRACKER_OUTPUT", "association_ious": {}}
    candidate = tracker.get("selected_candidate")
    bbox = _box(candidate.get("bbox_xyxy_pixels")) if isinstance(candidate, dict) else None
    binding = tracker.get("current_binding", {})
    if bbox is None or any(binding.get(key) != value for key, value in bindings.items()):
        return "UNRESOLVED", {"reason": "SELECTED_CURRENT_BINDING_INVALID", "association_ious": {}}
    selected_id = candidate.get("candidate_id")
    matches = [item for item in semantic_frame.get("candidates", []) if item.get("candidate_id") == selected_id]
    if (len(matches) != 1 or matches[0] != candidate or tracker.get("state") != "TRACKING"
            or tracker.get("flight_authorized") is not False
            or tracker.get("instance_identity_guaranteed") is not False):
        return "UNRESOLVED", {"reason": "SELECTED_CANDIDATE_NOT_IN_BOUND_FRAME", "association_ious": {}}
    ious = {name: _iou(bbox, box) for name in ("target", "distractor")
            if (box := _box(truth[name]["bbox_xyxy_pixels"])) is not None}
    matched = [name for name, score in ious.items() if score >= .3]
    if len(matched) != 1:
        return "UNRESOLVED", {"reason": "NO_UNIQUE_VISIBLE_INSTANCE_MATCH", "association_ious": ious}
    return ("CORRECT" if matched[0] == "target" else "WRONG"), {
        "reason": "UNIQUE_VISIBLE_MASK_MATCH", "matched_instance": matched[0], "association_ious": ious}


def rate_bounds(rows: list[dict], field: str) -> dict:
    counts = Counter(row[field] for row in rows)
    total, correct = len(rows), counts["CORRECT"]
    unknown = sum(counts[key] for key in UNKNOWN)
    return {"status": "COMPLETE" if not unknown else "UNKNOWN_WITH_BOUNDS", "planned_slot_denominator": total,
            "correct_slots": correct, "unknown_slots": unknown, "scoreable_slots": total-unknown,
            "fraction": correct/total if total and not unknown else None,
            "lower_bound": correct/total if total else None,
            "upper_bound": (correct+unknown)/total if total else None, "counts": dict(counts)}


def receipt_events(rows: list[dict], report: dict, observability: str) -> dict:
    """Events are supported observed transitions, not continuous truth spans."""
    actual = [row for row in rows if row["identity_point_label"] not in UNKNOWN
              and row["process_completed_host_s"] is not None]
    actual.sort(key=lambda row: row["process_completed_host_s"])
    first = next((row for row in actual if row["identity_point_label"] == "CORRECT"), None)
    origin = _number(report.get("schedule_begin_host_monotonic_s"))
    transition = next((item for item in report.get("phase_transition_records", []) if item.get("phase_id") == PHASES[2]), {})
    return_setup = _number(transition.get("setup_completed_host_monotonic_s"))
    return_correct = next((row for row in actual if row["phase_id"] == PHASES[2] and row["identity_point_label"] == "CORRECT"), None)
    loss = None
    if first is not None:
        loss = next((row for row in actual if row["process_completed_host_s"] > first["process_completed_host_s"]
                     and row["identity_point_label"] != "CORRECT"), None)
    recovery = next((row for row in actual if loss is not None and
                     row["process_completed_host_s"] > loss["process_completed_host_s"] and
                     row["identity_point_label"] == "CORRECT"), None)
    return_rows = [row for row in rows if row["phase_id"] == PHASES[2]]
    loss_time = loss["process_completed_host_s"] if loss else None
    recovery_time = recovery["process_completed_host_s"] if recovery else None
    missing_between = bool(loss and any(row["identity_point_label"] in UNKNOWN and
        (row["planned_start_host_s"] is None or loss_time <= row["planned_start_host_s"] <= (
            recovery_time if recovery_time is not None else float("inf"))) for row in rows))
    return {"event_clock": "actual observer completion host monotonic; saved readback receipts also retained",
            "initial_acquisition_observed": first is not None,
            "initial_acquisition_elapsed_host_s": first["process_completed_host_s"]-origin if first and origin is not None else None,
            "first_observed_loss_host_s": loss_time,
            "first_observed_recovery_host_s": recovery_time,
            "observed_loss_to_recovery_span_s": recovery_time-loss_time if recovery and not missing_between else None,
            "loss_to_recovery_coverage": "UNKNOWN_INTERVALS" if missing_between else "OBSERVED_RECEIPT_TRANSITIONS",
            "return_setup_completed_host_s": return_setup,
            "first_correct_return_available_after_setup_s": return_correct["process_completed_host_s"]-return_setup
                if return_correct and return_setup is not None else None,
            "attribute_cued_recovery_observed": bool(loss and return_correct) if observability == "VISUAL_ATTRIBUTE" else None,
            "identical_post_loss_refusal": ({"planned_return_slots": len(return_rows),
                "captured_return_slots": sum(row["captured"] for row in return_rows),
                "refusal_flag_slots": sum(row["unobservable_recovery_blocked"] for row in return_rows),
                "no_target_output_or_pending_slots": sum(row["identity_point_label"] in {"NO_OUTPUT", "PENDING"} for row in return_rows),
                "correct_visible_mask_matches": sum(row["identity_point_label"] == "CORRECT" for row in return_rows),
                "interpretation": "Refusal after loss is not successful reidentification; identical instances lack a visible recovery cue"}
                if observability == "UNRESOLVABLE_IDENTICAL" else None),
            "limits": ["Acquisition/recovery availability includes processing and cloud wait, not exposure-to-response delay",
                       "Observed transitions bound sampled behavior; missing scheduled observations prevent full continuous loss-duration claims"]}


def analyze_report(report: dict | None, configuration: dict, *, episode_id: str) -> dict:
    expected = expected_slots(configuration)
    observability = configuration.get("recovery_observability")
    if observability not in OBSERVABILITY:
        raise ValueError("Recovery observability must be registered")
    report = report or {}
    report_problems = []
    if report and (report.get("configuration") != configuration or report.get("kind") != "T1_GROUND_CONTINUOUS_IDENTITY_DEVELOPMENT"):
        report_problems.append("REPORT_PROTOCOL_OR_KIND_MISMATCH")
    origin = _number(report.get("schedule_begin_host_monotonic_s"))
    declared, frames = {}, {}
    for item in report.get("scheduled_slots", []):
        key = item.get("frame_sequence")
        if key in declared:
            raise ValueError("Duplicate scheduled slot")
        declared[key] = item
    for frame in report.get("frames", []):
        key = frame.get("frame_sequence")
        if key in frames:
            raise ValueError("Duplicate frame sequence")
        frames[key] = frame
    expected_ids = {row["frame_sequence"] for row in expected}
    if (set(declared) | set(frames))-expected_ids:
        raise ValueError("Unexpected frame/slot sequence")
    rows, requests = [], {}
    for planned in expected:
        number, phase = planned["frame_sequence"], planned["phase_id"]
        slot, frame = declared.get(number, {}), frames.get(number)
        start = origin+planned["planned_start_relative_s"] if origin is not None else None
        end = origin+planned["planned_end_relative_s"] if origin is not None else None
        row = {"episode_id": episode_id, "recovery_observability": observability, **planned,
               "planned_start_host_s": start, "planned_end_host_s": end,
               "captured": frame is not None, "declared_capture_status": slot.get("capture_status", "MISSING_SLOT_RECORD"),
               "not_captured_reason": slot.get("not_captured_reason"), "frame_id": frame.get("frame_id") if frame else None,
               "truth_verified": False, "scene_valid": False, "media_hashes_verified": 0,
               "identity_point_label": "UNKNOWN_TRUTH", "scheduled_identity_label": "UNKNOWN_TRUTH",
               "scheduled_effective_observation_label": "UNKNOWN_TRUTH", "capture_received_host_s": None,
               "process_completed_host_s": None, "capture_receipt_late": None, "output_late": None,
               "target_visible": None, "tracker_status": None, "tracker_state": None,
               "semantic_poll_status": None, "semantic_request_status": None,
               "semantic_source_age_proxy_s": None, "current_frame_age_proxy_s": None,
               "unobservable_recovery_blocked": False, "request_id_submitted": None, "problems": list(report_problems)}
        if slot and any(slot.get(key) != planned[key] for key in ("phase_id", "planned_start_relative_s", "planned_end_relative_s")):
            row["problems"].append("DECLARED_SLOT_GRID_MISMATCH")
        if frame is None:
            row["problems"].append("SCHEDULED_FRAME_MISSING")
            rows.append(row)
            continue
        if slot.get("capture_status") != "CAPTURED" or slot.get("frame_id") != frame.get("frame_id"):
            row["problems"].append("CAPTURE_RECORD_BINDING_MISMATCH")
        truth, problems, verified = _verify_frame(frame, report, phase)
        row["problems"].extend(problems)
        row["media_hashes_verified"] = verified
        row["capture_received_host_s"] = receipt = _number(frame.get("capture_received_host_s"))
        observer = frame.get("observer_record", {})
        row["process_completed_host_s"] = completed = _number(observer.get("process_completed_host_monotonic_s"))
        decision = observer.get("decision", {})
        tracker, poll, request = (decision.get(key, {}) for key in ("temporal_identity", "semantic_poll", "semantic_request"))
        row.update(tracker_status=tracker.get("status"), tracker_state=tracker.get("state"),
                   semantic_poll_status=poll.get("status"), semantic_request_status=request.get("status"),
                   semantic_source_age_proxy_s=_number(poll.get("semantic_source_age_proxy_s")),
                   current_frame_age_proxy_s=_number(poll.get("current_frame_age_proxy_s")),
                   unobservable_recovery_blocked=decision.get("unobservable_recovery_blocked") is True)
        if request.get("status") == "SUBMITTED":
            binding = request.get("request_binding", {})
            request_id = binding.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                row["problems"].append("SUBMITTED_REQUEST_BINDING_MISSING")
            else:
                if request_id in requests:
                    row["problems"].append("DUPLICATE_REQUEST_SUBMISSION")
                requests[request_id] = binding
                row["request_id_submitted"] = request_id
        row["capture_receipt_late"] = None if receipt is None or end is None else receipt >= end
        row["output_late"] = None if completed is None or end is None else completed >= end
        if truth is not None and not row["problems"]:
            row.update(truth_verified=True, scene_valid=True, target_visible=truth["target"]["visible_pixel_count"] > 0,
                       target_visible_pixels=truth["target"]["visible_pixel_count"],
                       distractor_visible_pixels=truth["distractor"]["visible_pixel_count"])
            label, audit = _decision_label(frame, report, truth)
            row["identity_point_label"] = label
            row.update(audit)
            if receipt is not None and start is not None and start <= receipt < end:
                if completed is not None and completed < receipt:
                    row["problems"].append("OUTPUT_CLOCK_REGRESSION")
                else:
                    scheduled = label if completed is not None and completed < end else "NO_OUTPUT"
                    row["scheduled_identity_label"] = scheduled
                    row["scheduled_effective_observation_label"] = ("CORRECT" if scheduled == "CORRECT" and row["target_visible"]
                        else "UNRESOLVED" if scheduled in UNKNOWN and row["target_visible"] else "NO_OUTPUT")
            else:
                row["problems"].append("RECEIPT_OUTSIDE_SCHEDULED_INTERVAL_OR_CLOCK_UNKNOWN")
        rows.append(row)
    return {"episode_id": episode_id, "recovery_observability": observability,
            "execution_status": report.get("status", "NO_REPORT"), "planned_slots": len(rows),
            "captured_slots": sum(row["captured"] for row in rows),
            "valid_scene_slots": sum(row["scene_valid"] for row in rows),
            "identity_point_counts": dict(Counter(row["identity_point_label"] for row in rows)),
            "scheduled_identity": rate_bounds(rows, "scheduled_identity_label"),
            "scheduled_effective_observation": rate_bounds(rows, "scheduled_effective_observation_label"),
            "media_hashes_verified": sum(row["media_hashes_verified"] for row in rows),
            "late_readback_receipts": sum(row["capture_receipt_late"] is True for row in rows),
            "late_observer_completions": sum(row["output_late"] is True for row in rows),
            "request_submissions": len(requests), "submitted_requests": list(requests.values()),
            "semantic_poll_counts": dict(Counter(row["semantic_poll_status"] or "NO_POLL" for row in rows)),
            "tracker_state_counts": dict(Counter(row["tracker_state"] or "NO_TRACKER" for row in rows)),
            "receipt_events": receipt_events(rows, report, observability),
            "ground_only_declared_and_recorded": bool(report) and configuration.get("ground_only") is True and
                all(report.get(key) is False for key in ("flight_commands_issued", "arm_commands_issued", "api_control_acquired"))
                and report.get("new_flights") == 0,
            "safe_ground_closure": report.get("safe_ground_closure"), "observer_close_receipt": report.get("observer_close_receipt"),
            "rows": rows}


def _find_plan(campaign: Path, plan_path: Path | None) -> Path:
    if plan_path is not None:
        return plan_path.resolve()
    candidate = campaign/"plan.json"
    if candidate.is_file():
        return candidate
    matches = []
    for candidate in (ROOT/"configs").glob("t1_identity_sequence*/plan.json"):
        try:
            if Path(_read(candidate).get("campaign_root", "")).resolve() == campaign:
                matches.append(candidate)
        except (OSError, ValueError):
            pass
    if len(matches) != 1:
        raise ValueError("An explicit frozen --plan is required")
    return matches[0]


def analyze_campaign(campaign: Path | str, output: Path | str, *, plan_path: Path | str | None = None) -> dict:
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    plan_path = _find_plan(campaign, Path(plan_path) if plan_path is not None else None)
    plan = _read(plan_path)
    if Path(plan.get("campaign_root", "")).resolve() != campaign:
        raise ValueError("Frozen plan belongs to another campaign")
    episodes, sources, identities = [], [{"path": str(plan_path), "sha256": digest(plan_path)}], set()
    episode_ids = set()
    for item in plan.get("episodes", []):
        episode_id = item["episode_id"]
        if episode_id in episode_ids:
            raise ValueError("Duplicate planned episode")
        episode_ids.add(episode_id)
        protocol_path = Path(item["protocol_path"])
        if digest(protocol_path) != item.get("protocol_sha256"):
            raise ValueError("Frozen episode protocol digest changed")
        configuration = _read(protocol_path)
        sources.append({"path": str(protocol_path), "sha256": digest(protocol_path)})
        report_path, manifest_path = Path(item["report_path"]), Path(item["manifest_path"])
        report = _read(report_path) if report_path.is_file() else None
        if report is not None:
            sources.append({"path": str(report_path), "sha256": digest(report_path)})
        result = analyze_report(report, configuration, episode_id=episode_id)
        result.update(report_path=str(report_path), manifest_path=str(manifest_path))
        manifest = _read(manifest_path) if manifest_path.is_file() else {}
        if manifest:
            sources.append({"path": str(manifest_path), "sha256": digest(manifest_path)})
        identity = (manifest.get("engine_process_id"), manifest.get("engine_started_utc"))
        if identity[0] is not None and identity[1] is not None:
            identities.add(identity)
        result["owned_UE_identity"] = list(identity)
        result["manifest_status"] = manifest.get("status", "NO_MANIFEST")
        result["engine_closed"] = manifest.get("engine_closed_after_safe_completion")
        if report is not None and manifest.get("report_sha256") != digest(report_path):
            raise ValueError("Episode report changed after launcher closeout")
        episodes.append(result)
    if not episodes:
        raise ValueError("Frozen campaign has no planned episodes")
    all_rows = [row for episode in episodes for row in episode["rows"]]
    result = {"schema_version": 1, "kind": "INDEPENDENT_T1_GROUND_IDENTITY_SEQUENCE_EVALUATION",
              "status": "DEVELOPMENT_ANALYZED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "campaign": str(campaign), "frozen_plan": str(plan_path), "formal_efficacy_results": False,
              "planned_UE_sessions": len(episodes), "identified_UE_processes": len(identities),
              "planned_slots_all": len(all_rows), "captured_slots_all": sum(row["captured"] for row in all_rows),
              "valid_scene_slots_all": sum(row["scene_valid"] for row in all_rows),
              "scheduled_identity": rate_bounds(all_rows, "scheduled_identity_label"),
              "scheduled_effective_observation": rate_bounds(all_rows, "scheduled_effective_observation_label"),
              "counts_by_observability": {condition: {"planned_sessions": sum(ep["recovery_observability"] == condition for ep in episodes),
                  "captured_slots": sum(ep["captured_slots"] for ep in episodes if ep["recovery_observability"] == condition)} for condition in OBSERVABILITY},
              "request_submissions_all": sum(ep["request_submissions"] for ep in episodes),
              "all_sessions_safely_closed": all(ep["safe_ground_closure"] is True
                  and ep["engine_closed"] is True for ep in episodes),
              "media_hashes_verified_all": sum(ep["media_hashes_verified"] for ep in episodes),
              "identity_point_counts_all": dict(Counter(row["identity_point_label"] for row in all_rows)),
              "late_readback_receipts_all": sum(row["capture_receipt_late"] is True for row in all_rows),
              "late_observer_completions_all": sum(row["output_late"] is True for row in all_rows),
              "all_reports_ground_only": all(ep["ground_only_declared_and_recorded"] for ep in episodes),
              "new_flights": 0 if all(ep["ground_only_declared_and_recorded"] for ep in episodes) else None,
              "episodes": episodes, "sources": sources,
              "analyzer_source_sha256": digest(Path(__file__)),
              "limits": ["Two cue and two identical sessions are development checks; report actual available sessions and missing reports",
                  "Frames, phases and slots are nested within independently launched UE sessions, not independent method repetitions",
                  "Instance scoring uses unique IoU >= .3 against independently reproduced saved evaluation-only visible boxes",
                  "No/multiple instance match remains UNRESOLVED; missing or invalid scene/media remains UNKNOWN_TRUTH",
                  "Planned-slot fractions describe sampled available output; do not establish continuous target identity or visibility",
                  "Actual readback and observer completion clocks are host times, not exposure or exact actuation delay",
                  "Identical-instance refusal after loss is not successful reidentification",
                  "Ground virtual-camera scene; no aircraft flight, trajectory efficacy, safety or map/appearance generalization"],
              "ai_assistance": "AI assisted independent evaluator implementation; counts derived from archived scheduled slots and media"}
    output.mkdir(parents=True, exist_ok=False)
    snapshot=output/"source_snapshot"
    snapshot.mkdir()
    for source in (Path(__file__),Path(__file__).with_name("t1_identity_schedule.py")):
        destination=snapshot/source.name
        shutil.copyfile(source,destination)
        if digest(destination)!=digest(source):
            raise ValueError("Evaluation source snapshot changed")
    result["analyzer_source_snapshot"]={name:digest(snapshot/name) for name in
        ("analyze_t1_identity_sequence.py","t1_identity_schedule.py")}
    (output/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output/"timeline_rows.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    fields = [key for key in all_rows[0] if key not in {"problems", "association_ious"}]
    with (output/"timeline_rows.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({key: row.get(key) for key in fields})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    result = analyze_campaign(args.campaign, args.output, plan_path=args.plan)
    print(json.dumps({key: result[key] for key in ("status", "planned_UE_sessions", "planned_slots_all",
        "captured_slots_all", "valid_scene_slots_all", "request_submissions_all", "new_flights")}, ensure_ascii=False))
