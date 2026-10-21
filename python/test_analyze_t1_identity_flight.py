"""Synthetic contracts exercise independent parsing and failure preservation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from analyze_t1_identity_flight import (TARGET_PURPOSE, analyze_run, binding_issues,
    canonical_sha, command_audit, current_phase, discover_media_without_ledger,
    identity_score, lease_clear_audit, mask_truth, recovery_metrics, stage_metrics)


class IndependentFlightAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.report = {"session_id": "s", "configuration": {"description": "striped balloon",
            "command_speed_cap_m_s": .03, "command_lease_s": .12, "world_z_ned_m": -3., "mode": "SHORT_XY"}}
        sha = hashlib.sha256(b"rgb").hexdigest()
        desc = hashlib.sha256(b"striped balloon").hexdigest()
        binding = {"session_id": "s", "target_generation": 1, "description_sha256": desc,
            "frame_id": "f", "frame_sequence": 1, "image_sha256": sha,
            "readback_receipt_monotonic_s": 10., "candidates_ready_monotonic_s": 10.05}
        candidate = {"candidate_id": "a", "bbox_xyxy_pixels": [2., 2., 6., 6.], "detector_score": .8}
        selection = {"status": "TEMPORAL_LOCAL_ASSOCIATION_VALID", "state": "TRACKING",
            "session_id": "s", "target_generation": 1, "description_sha256": desc,
            "current_frame_id": "f", "current_frame_sequence": 1, "current_image_sha256": sha,
            "current_binding": binding, "selected_candidate_id": "a", "selected_candidate": candidate,
            "semantic_review_deadline_monotonic_s": 18., "flight_authorized": False,
            "identity_probability_calibrated": False, "instance_identity_guaranteed": False}
        saved = {key: value for key, value in binding.items() if key != "description_sha256"}
        saved["candidates"] = [candidate]
        surface = {"status": "SURFACE_GEOMETRY_ACCEPTED", "session_id": "s", "target_generation": 1,
            "frame_id": "f", "frame_sequence": 1, "image_sha256": sha, "description_sha256": desc,
            "selection_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID", "selected_candidate_id": "a",
            "selected_bbox_xyxy_pixels": candidate["bbox_xyxy_pixels"], "current_binding": binding,
            "selection_certificate_sha256": canonical_sha(selection)}
        self.frame = {"frame_id": "f", "frame_sequence": 1, "capture_received_host_s": 10.,
            "media": {"rgb": {"sha256": sha}}, "observer_result": {"semantic_frame": saved,
            "temporal_identity": selection}, "surface": surface,
            "command_self_state": {"host_receipt_monotonic_s": 10.08},
            "motion_gate": {"status": "ALLOW_BOUNDED_COMMAND", "target_command_authorized": True,
                "current_binding": binding, "command_velocity_xy_m_s": [.02, 0.], "command_duration_s": .12,
                "command_end_deadline_monotonic_s": 10.48, "latest_dispatch_monotonic_s": 10.2,
                "configured_envelope": {"current_frame_max_age_s": .5, "freshness_reserve_s": .02, "self_state_max_age_s": .12}}}
        self.command = {"kind": "COMMAND", "stage": "PERCEPTION_XY", "purpose": TARGET_PURPOSE,
            "velocity_xy_m_s": [.02, 0.], "duration_s": .12, "dispatch_monotonic_s": 10.1,
            "target_frame_id": "f", "target_end_deadline_monotonic_s": 10.48}

    def test_recomputed_masks_use_saved_rgb_palette_not_runner_truth(self):
        palette = {"42": [200, 20, 10], "43": [0, 80, 100]}
        seg = np.zeros((10, 16, 3), np.uint8)
        seg[2:6, 2:6] = palette["42"][::-1]
        seg[2:6, 10:14] = palette["43"][::-1]
        truth = mask_truth(seg, palette)
        self.assertEqual(truth["target"]["visible_pixel_count"], 16)
        self.assertEqual(truth["target"]["bbox_xyxy_pixels"], [2, 2, 6, 6])
        self.assertEqual(identity_score(self.frame["observer_result"]["temporal_identity"], truth)["status"], "CORRECT")
        selected = deepcopy(self.frame["observer_result"]["temporal_identity"])
        selected["selected_candidate"]["bbox_xyxy_pixels"] = [10, 2, 14, 6]
        self.assertEqual(identity_score(selected, truth)["status"], "WRONG_DISTRACTOR")

    def test_overlap_requires_exclusive_iou_not_only_top_score(self):
        selection = self.frame["observer_result"]["temporal_identity"]
        truth = {"target": {"bbox_xyxy_pixels": [2, 2, 6, 6]}, "distractor": {"bbox_xyxy_pixels": [3, 2, 7, 6]}}
        self.assertEqual(identity_score(selection, truth)["status"], "AMBIGUOUS_OR_UNMATCHED")

    def test_unknown_decision_is_not_no_output_or_zero_identity(self):
        self.assertEqual(identity_score({}, None)["status"], "UNKNOWN_OUTPUT_STATE")
        self.assertEqual(identity_score({"status": "NO_TARGET_OUTPUT"}, None)["status"], "NO_OUTPUT")
        self.assertEqual(identity_score(self.frame["observer_result"]["temporal_identity"], None)["status"], "UNKNOWN_MEDIA")

    def test_correct_same_frame_binding_and_gate_command_pass(self):
        self.assertEqual(binding_issues(self.frame, self.report), [])
        self.assertEqual(command_audit(self.command, self.frame, self.report), [])

    def test_reused_hash_sequence_or_candidate_payload_detected(self):
        for key, value in (("frame_sequence", 99), ("image_sha256", "0"*64)):
            frame = deepcopy(self.frame)
            frame["observer_result"]["semantic_frame"][key] = value
            self.assertIn("SEMANTIC_ACQUISITION_BINDING_MISMATCH", binding_issues(frame, self.report))
        frame = deepcopy(self.frame)
        frame["observer_result"]["temporal_identity"]["selected_candidate"]["detector_score"] = .9
        self.assertIn("SURFACE_TEMPORAL_CERTIFICATE_HASH_MISMATCH", binding_issues(frame, self.report))

    def test_target_command_without_selection_is_detected(self):
        frame = deepcopy(self.frame)
        frame["observer_result"]["temporal_identity"]["status"] = "NO_TARGET_OUTPUT"
        self.assertIn("TARGET_COMMAND_WITHOUT_VALID_SELECTION", command_audit(self.command, frame, self.report))
        self.assertIn("TARGET_COMMAND_NO_MATCHING_CURRENT_FRAME", command_audit(self.command, None, self.report))

    def test_target_end_and_fresh_self_dispatch_checked_independently(self):
        late = dict(self.command, dispatch_monotonic_s=10.39)
        issues = command_audit(late, self.frame, self.report)
        self.assertIn("TARGET_COMMAND_END_AFTER_GATE_DEADLINE", issues)
        self.assertIn("TARGET_COMMAND_AFTER_FRESH_SELF_STATE_DEADLINE", issues)
        self.assertIn("TARGET_COMMAND_OUTLIVES_CURRENT_OBSERVATION", issues)
        # A forged generous gate cannot extend the frozen observation/self age.
        frame = deepcopy(self.frame)
        frame["motion_gate"]["command_end_deadline_monotonic_s"] = 20.
        frame["motion_gate"]["latest_dispatch_monotonic_s"] = 20.
        frame["motion_gate"]["configured_envelope"]["current_frame_max_age_s"] = 10.
        forged = dict(late, target_end_deadline_monotonic_s=20.)
        self.assertIn("GATE_FRESHNESS_CONFIG_EXCEEDS_FROZEN_LIMITS", command_audit(forged, frame, self.report))

    def test_declared_duration_does_not_imply_actual_zero_clear(self):
        result = lease_clear_audit(self.command, self.frame, [])
        self.assertEqual(result["status"], "UNKNOWN_EXPLICIT_STOP")
        self.assertIn("TARGET_EXPLICIT_ZERO_CLEAR_TIME_UNKNOWN_OR_INVALID", result["issues"])

    def lease_fixture(self, clear):
        frame = deepcopy(self.frame)
        stop = {"status": "OWN_STATE_SAMPLED_STOPPED_BEFORE_BLOCKING_WORK",
            "declared_lease_end_monotonic_s": 10.22, "target_setpoint_cleared_monotonic_s": clear,
            "clear_after_declared_lease_s": clear-10.22, "completed_monotonic_s": clear+.06,
            "samples": [{"receipt_monotonic_s": clear, "velocity_xy_m_s": [.004, 0.]},
                {"receipt_monotonic_s": clear+.03, "velocity_xy_m_s": [.001, 0.]}]}
        frame["lease_stop"] = stop
        events = [self.command, {"kind": "COMMAND", "purpose": "OWN_STATE_BRAKE_OR_HOLD",
            "dispatch_monotonic_s": clear, "velocity_xy_m_s": [0., 0.]},
            dict(stop, kind="TARGET_LEASE_STOP", target_frame_id="f")]
        return frame, events

    def test_actual_clear_after_declared_lease_inside_fresh_deadline_retained(self):
        frame, events = self.lease_fixture(10.235)
        result = lease_clear_audit(self.command, frame, events)
        self.assertEqual(result["issues"], [])
        self.assertAlmostEqual(result["clear_after_declared_lease_s"], .015)
        self.assertLess(result["clear_after_observation_deadline_s"], 0)
        self.assertAlmostEqual(result["max_stop_sample_gap_s"], .03)

    def test_actual_clear_after_observation_deadline_fails(self):
        frame, events = self.lease_fixture(10.5)
        result = lease_clear_audit(self.command, frame, events)
        self.assertIn("TARGET_EXPLICIT_CLEAR_AFTER_OBSERVATION_DEADLINE", result["issues"])
        self.assertAlmostEqual(result["clear_after_observation_deadline_s"], .02)

    def test_generous_gate_end_cannot_hide_actual_clear_after_frame_freshness(self):
        frame, events = self.lease_fixture(10.5)
        frame["motion_gate"]["command_end_deadline_monotonic_s"] = 20.
        result = lease_clear_audit(self.command, frame, events)
        self.assertIn("TARGET_EXPLICIT_CLEAR_AFTER_OBSERVATION_DEADLINE", result["issues"])
        self.assertAlmostEqual(result["independently_computed_observation_end_deadline_s"], 10.48)

    def test_stop_record_without_matching_zero_dispatch_fails(self):
        frame, events = self.lease_fixture(10.235)
        events[1]["velocity_xy_m_s"] = [.01, 0.]
        self.assertIn("TARGET_CLEAR_NOT_BOUND_TO_ACTUAL_OWN_ZERO_COMMAND", lease_clear_audit(self.command, frame, events)["issues"])

    def test_equivalent_zero_dispatches_sharing_host_tick_are_retained(self):
        frame, events = self.lease_fixture(10.235)
        events.insert(2, deepcopy(events[1]))
        result = lease_clear_audit(self.command, frame, events)
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["actual_zero_dispatch_records_at_clear_timestamp"], 2)

    def test_startup_landing_outliers_do_not_enter_low_speed_metrics(self):
        def state(t, x, speed, stage):
            return {"kind": "SELF_STATE", "stage": stage, "host_receipt_monotonic_s": t,
                "state_timestamp_ns": int(t*1e9), "position_world_ned_m": [x, 0., -3.],
                "velocity_ned_m_s": [speed, 0., 0.], "acceleration_ned_m_s2": [.02, 0., 0.]}
        events = [state(1., 100., 10., "SETTLING"), state(10., 0., .02, "PERCEPTION_XY"),
            state(10.7, .014, .025, "PERCEPTION_XY"), state(20., 100., 20., "LANDING_RECOVERY")]
        metrics = stage_metrics(self.report, events)
        self.assertEqual(metrics["self_state_samples"], 2)
        self.assertAlmostEqual(metrics["max_actual_xy_speed_m_s"], .025)
        self.assertAlmostEqual(metrics["sampled_xy_path_length_m"], .014)
        self.assertAlmostEqual(metrics["max_self_state_sample_gap_s"], .7)

    def test_actual_transition_completion_owns_phase_and_recovery_timing(self):
        report = deepcopy(self.report)
        report["phase_transitions"] = [{"phase_id": "visible", "started_monotonic_s": 1., "completed_monotonic_s": 1.1},
            {"phase_id": "hidden", "started_monotonic_s": 7., "completed_monotonic_s": 7.35},
            {"phase_id": "return", "started_monotonic_s": 9., "completed_monotonic_s": 9.4}]
        self.assertEqual(current_phase(report, 7.1, 7.4), "TRANSITION_OVERLAP")
        self.assertEqual(current_phase(report, 7.5, 7.6), "hidden")
        rows = [{"frame_id": "pre-hide", "actual_phase": "visible", "readback_s": 6.95,
             "observer_completed_s": 6.98, "temporal_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID",
             "identity": {"status": "CORRECT"}},
            {"actual_phase": "hidden", "temporal_status": "NO_TARGET_OUTPUT", "readback_s": 7.6},
            {"actual_phase": "return", "temporal_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID",
             "identity": {"status": "CORRECT"}, "readback_s": 9.9,
             "observer_completed_s": 10.2, "decision_completed_s": 10.3}]
        metrics = recovery_metrics(report, rows)
        self.assertTrue(metrics["pre_hide_track_established"])
        self.assertEqual(metrics["pre_hide_track_qualification"], "CORRECT_FRESH_TRACK_ESTABLISHED_BEFORE_HIDE")
        self.assertAlmostEqual(metrics["loss_receipt_delay_from_actual_hidden_s"], .25)
        self.assertAlmostEqual(metrics["recovery_completion_delay_from_actual_return_s"], .9)
        self.assertAlmostEqual(metrics["recovery_identity_availability_delay_from_actual_return_s"], .8)

    def recovery_fixture(self):
        report = deepcopy(self.report)
        report["phase_transitions"] = [
            {"phase_id": "hidden", "completed_monotonic_s": 10.},
            {"phase_id": "return", "completed_monotonic_s": 12.}]
        before = {"frame_id": "pre", "actual_phase": "visible", "readback_s": 9.8,
            "observer_completed_s": 9.85, "temporal_status": "NO_TARGET_OUTPUT",
            "identity": {"status": "NO_OUTPUT"}}
        hidden = {"actual_phase": "hidden", "readback_s": 10.1,
            "observer_completed_s": 10.15, "temporal_status": "NO_TARGET_OUTPUT",
            "identity": {"status": "NO_OUTPUT"}}
        returned = {"actual_phase": "return", "readback_s": 12.2,
            "observer_completed_s": 12.25, "decision_completed_s": 12.3,
            "temporal_status": "TEMPORAL_LOCAL_ASSOCIATION_VALID", "identity": {"status": "CORRECT"}}
        return report, before, hidden, returned

    def test_never_established_track_cannot_be_lost_by_occlusion(self):
        report, before, hidden, returned = self.recovery_fixture()
        returned.update(temporal_status="NO_TARGET_OUTPUT", identity={"status": "NO_OUTPUT"})
        metrics = recovery_metrics(report, [before, hidden, returned])
        self.assertEqual(metrics["pre_hide_track_qualification"], "NO_CORRECT_PRE_HIDE_TRACK")
        self.assertFalse(metrics["pre_hide_track_established"])
        self.assertIsNone(metrics["loss_receipt_delay_from_actual_hidden_s"])
        self.assertIsNone(metrics["recovery_identity_availability_delay_from_actual_return_s"])
        self.assertIsNone(metrics["recovery_completion_delay_from_actual_return_s"])

    def test_first_correct_after_return_is_initialization_not_recovery(self):
        report, before, hidden, returned = self.recovery_fixture()
        metrics = recovery_metrics(report, [before, hidden, returned])
        self.assertEqual(metrics["status"], "CORRECT_INITIALIZATION_OBSERVED_AFTER_RETURN")
        self.assertIsNone(metrics["loss_receipt_delay_from_actual_hidden_s"])
        self.assertIsNone(metrics["recovery_identity_availability_delay_from_actual_return_s"])
        self.assertIsNone(metrics["recovery_completion_delay_from_actual_return_s"])
        self.assertEqual(metrics["return_correct_output_frames"], 1)
        self.assertEqual(metrics["longest_consecutive_correct_return_frames"], 1)

    def test_earlier_correct_cannot_override_lost_output_before_hide(self):
        report, before, hidden, returned = self.recovery_fixture()
        earlier = dict(returned, frame_id="earlier", actual_phase="visible", readback_s=9.4,
            observer_completed_s=9.45, decision_completed_s=9.5)
        metrics = recovery_metrics(report, [earlier, before, hidden, returned])
        self.assertEqual(metrics["pre_hide_track_qualification"], "LOSS_ALREADY_PRESENT_BEFORE_HIDE")
        self.assertEqual(metrics["pre_hide_reference_frame_id"], "pre")
        self.assertFalse(metrics["pre_hide_track_established"])
        self.assertIsNone(metrics["loss_receipt_delay_from_actual_hidden_s"])
        self.assertIsNone(metrics["recovery_completion_delay_from_actual_return_s"])

    def test_stale_or_not_yet_available_pre_hide_correct_output_cannot_qualify(self):
        report, _, hidden, returned = self.recovery_fixture()
        before = dict(returned, frame_id="stale", actual_phase="visible", readback_s=9.4,
            observer_completed_s=9.45, decision_completed_s=9.5)
        metrics = recovery_metrics(report, [before, hidden, returned])
        self.assertEqual(metrics["pre_hide_track_qualification"], "PRE_HIDE_TRACK_STALE_AT_HIDE")
        self.assertIsNone(metrics["loss_receipt_delay_from_actual_hidden_s"])
        pending = dict(before, frame_id="pending", readback_s=9.8, observer_completed_s=10.01)
        metrics = recovery_metrics(report, [pending, hidden, returned])
        self.assertEqual(metrics["pre_hide_track_qualification"], "NO_CORRECT_PRE_HIDE_TRACK")
        self.assertEqual(metrics["pre_hide_available_output_frames"], 0)

    def test_fresh_correct_return_without_observed_hidden_loss_is_not_recovery(self):
        report, _, _, returned = self.recovery_fixture()
        before = dict(returned, frame_id="pre", actual_phase="visible", readback_s=9.8,
            observer_completed_s=9.85, decision_completed_s=9.9)
        metrics = recovery_metrics(report, [before, returned])
        self.assertTrue(metrics["pre_hide_track_established"])
        self.assertEqual(metrics["status"], "CORRECT_RETURN_TRACK_WITHOUT_OBSERVED_HIDDEN_LOSS")
        self.assertIsNone(metrics["recovery_completion_delay_from_actual_return_s"])

    def test_saved_media_survives_lost_ledger_without_fabricated_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            flight = Path(directory)
            (flight/"frames").mkdir()
            cv2.imwrite(str(flight/"frames/s_f000_rgb.png"), np.zeros((8, 8, 3), np.uint8))
            plan = {"session_id": "s", "planned_slots": [{"slot_index": 0, "phase_id": "visible"}, {"slot_index": 1, "phase_id": "hidden"}]}
            rows = discover_media_without_ledger(flight, plan, set())
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "MEDIA_SAVED_DECISION_LEDGER_LOST")
            self.assertNotIn("observer_result", rows[0])
            self.assertEqual(identity_score(rows[0].get("observer_result"), None)["status"], "UNKNOWN_OUTPUT_STATE")

    def test_failed_no_report_run_retains_planned_denominator_and_unknown_flight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root/"plan.json"
            plan.write_text(json.dumps({"planned_slots": [{"slot_index": i} for i in range(5)]}))
            (root/"manifest.json").write_text(json.dumps({"status": "FAIL", "plan": str(plan)}))
            result = analyze_run(root)
            self.assertEqual(result["planned_slots"], 5)
            self.assertIsNone(result["new_flights"])
            self.assertEqual(result["slot_status_counts"], {"INTERRUPTED_NOT_REACHED": 5})

    def test_supplemental_receipt_does_not_shadow_run_root_and_airborne_is_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flight = root/"flight"
            flight.mkdir()
            plan = root/"plan.json"
            config = {"session_id": "s", "source_inventory": [], "planned_slots": [], "world_z_ned_m": -3.}
            plan.write_text(json.dumps(config))
            event = {"kind": "SELF_STATE", "stage": "PERCEPTION_XY", "landed": False,
                "host_receipt_monotonic_s": 10., "state_timestamp_ns": 10,
                "position_world_ned_m": [0., 0., -3.], "velocity_ned_m_s": [0., 0., 0.],
                "acceleration_ned_m_s2": [0., 0., 0.]}
            log = flight/"execution.jsonl"
            log.write_text(json.dumps(event)+"\n")
            report = {"plan": str(plan), "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                "configuration": config, "session_id": "s", "status": "RUNNING", "new_flights": 0,
                "frames": [], "slots": [], "execution_log": {"path": str(log),
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest()}}
            (flight/"report.json").write_text(json.dumps(report))
            (root/"manifest.json").write_text(json.dumps({"status": "FAIL"}))
            (root/"supplemental_owned_closure.json").write_text(json.dumps({"status": "PASS"}))
            with patch("analyze_t1_identity_flight.source_checks", return_value=[]):
                result = analyze_run(root)
            self.assertEqual(result["run"], str(root.resolve()))
            self.assertEqual(result["new_flights"], 0)
            self.assertTrue(result["observed_airborne_from_execution"])
            self.assertEqual(result["observed_airborne_self_state_samples"], 1)


if __name__ == "__main__":
    unittest.main()
