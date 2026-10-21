"""Timing provenance, interval coverage and clock separation regressions."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from analyze_t1_async_timing import (TARGET_PURPOSE, analyze_run, digest, distribution,
    frame_deadlines, interval_evidence, receipt_cadence, timing_metrics)


class AsyncTimingTests(unittest.TestCase):
    def state(self, stamp):
        return {"kind": "SELF_STATE", "stage": "PERCEPTION_XY", "host_receipt_monotonic_s": stamp}

    def fixture(self):
        clear = {"kind": "TARGET_LEASE_STOP", "stage": "PERCEPTION_XY", "target_frame_id": "f",
            "declared_lease_end_monotonic_s": 10.22, "target_setpoint_cleared_monotonic_s": 10.23,
            "completed_monotonic_s": 10.33,
            "samples": [{"receipt_monotonic_s": 10.28, "velocity_xy_m_s": [0., 0.]},
                {"receipt_monotonic_s": 10.33, "velocity_xy_m_s": [0., 0.]}]}
        frame = {"frame_id": "f", "slot_index": 0, "capture_received_host_s": 10.,
            "observer_result": {"process_completed_host_monotonic_s": 10.08,
                "temporal_identity": {"semantic_review_deadline_monotonic_s": 18.}},
            "motion_gate": {"decision_monotonic_s": 10.09,
                "command_end_deadline_monotonic_s": 10.48,
                "configured_envelope": {"freshness_reserve_s": .02}},
            "decision_completed_monotonic_s": 10.5, "lease_stop": clear}
        report = {"configuration": {"self_monitor_period_s": .05},
            "low_speed_stage_start_state": self.state(10.), "low_speed_stage_end_state": self.state(10.6),
            "frames": [frame], "slots": [{"slot_index": 0, "deadline_monotonic_s": 10.4},
                {"slot_index": 1, "deadline_monotonic_s": 10.6, "status": "NOT_CAPTURED"}],
            "injected_delays": [{"slot_index": 1, "started_monotonic_s": 10.16,
                "completed_monotonic_s": 10.51, "duration_s": .35,
                "boundary": "AFTER_CAPTURE_BEFORE_OBSERVER"}]}
        target = {"kind": "COMMAND", "stage": "PERCEPTION_XY", "purpose": TARGET_PURPOSE,
            "velocity_xy_m_s": [.02, 0.], "duration_s": .12,
            "dispatch_monotonic_s": 10.1, "target_frame_id": "f"}
        zero = {"kind": "COMMAND", "stage": "PERCEPTION_XY", "purpose": "OWN_STATE_HOLD",
            "velocity_xy_m_s": [0., 0.], "duration_s": .12, "dispatch_monotonic_s": 10.23}
        events = [self.state(stamp) for stamp in (10., 10.05, 10.1, 10.15, 10.2, 10.25, 10.3, 10.35, 10.4, 10.45, 10.5, 10.55, 10.6)]
        events += [target, zero, clear]
        return report, events

    def test_statistics_filter_unknowns_and_use_explicit_quantile_rule(self):
        result = distribution([None, False, float("nan"), .1, .2, .4])
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["median_s"], .2)
        self.assertAlmostEqual(result["p95_s"], .38)
        self.assertIsNone(distribution([])["max_s"])

    def test_duplicate_ticks_and_logging_order_do_not_manufacture_gap(self):
        result = receipt_cadence([self.state(stamp) for stamp in (10., 10.1, 10.1, 10.05, 10.5)])
        self.assertEqual(result["unique_host_receipt_count"], 4)
        self.assertEqual(result["same_host_tick_extra_records"], 1)
        self.assertEqual(result["raw_event_order_regressions"], 1)
        self.assertAlmostEqual(result["gap_distribution"]["max_s"], .4)
        self.assertEqual(result["gaps_over_0p4_s"], 0)
        self.assertEqual(result["gaps_over_0p12_s"], 1)

    def test_cadence_stage_edges_are_visible_and_outside_states_excluded(self):
        result = receipt_cadence([self.state(stamp) for stamp in (9., 10.1, 10.2, 11.)], 10., 10.4)
        self.assertEqual(result["in_stage_record_count"], 2)
        self.assertAlmostEqual(result["stage_head_gap_s"], .1)
        self.assertAlmostEqual(result["stage_tail_gap_s"], .2)

    def test_zero_samples_block_retains_unknown_coverage_and_edge_gap(self):
        cadence = receipt_cadence([self.state(10.), self.state(11.)])
        interval = {"started_monotonic_s": 10.1, "completed_monotonic_s": 10.8}
        result = interval_evidence(interval, cadence, [], {})
        self.assertEqual(result["self_state_receipts_inside"], 0)
        self.assertAlmostEqual(result["max_boundary_clipped_receipt_gap_s"], .7)
        self.assertAlmostEqual(result["max_whole_receipt_gap_touching_interval_s"], 1.)
        self.assertTrue(result["bracketed_by_saved_self_receipts"])
        empty = interval_evidence(interval, receipt_cadence([]), [], {})
        self.assertFalse(empty["bracketed_by_saved_self_receipts"])

    def test_delay_boundary_counts_and_prior_active_target_are_distinct(self):
        report, events = self.fixture()
        metrics = timing_metrics(report, events)
        row = metrics["injected_delay_rows"][0]
        self.assertEqual(row["own_state_command_dispatches_inside"], 1)
        self.assertEqual(row["nonzero_target_dispatches_inside"], 0)
        self.assertEqual(len(row["nonzero_target_active_overlap_rows"]), 1)
        self.assertAlmostEqual(row["nonzero_target_active_overlap_rows"][0]["duration_s"], .07)
        self.assertEqual(row["self_state_receipts_inside"], 7)
        self.assertAlmostEqual(row["max_boundary_clipped_receipt_gap_s"], .05)

    def test_real_gate_timely_full_completion_late_and_miss_unknown(self):
        report, _ = self.fixture()
        metrics = frame_deadlines(report)
        self.assertEqual(metrics["gate_deadline_counts"], {"TIMELY": 1, "UNKNOWN": 1})
        self.assertEqual(metrics["full_completion_deadline_counts"], {"LATE": 1, "UNKNOWN": 1})
        self.assertAlmostEqual(metrics["rows"][0]["full_completion_after_nominal_deadline_s"], .1)

    def test_declared_lease_end_is_not_automatic_target_clear(self):
        report, events = self.fixture()
        metrics = timing_metrics(report, events)
        self.assertEqual(metrics["missing_or_unverified_explicit_target_clears"], 0)
        self.assertAlmostEqual(metrics["lease_clear_after_declared_end_distribution"]["max_s"], .01)
        self.assertAlmostEqual(metrics["observation_deadline_margin_distribution"]["min_s"], .25)
        broken = deepcopy(report)
        broken["frames"][0].pop("lease_stop")
        metrics = timing_metrics(broken, events)
        self.assertEqual(metrics["missing_or_unverified_explicit_target_clears"], 1)
        self.assertIn("TARGET_EXPLICIT_ZERO_CLEAR_TIME_UNKNOWN_OR_INVALID", metrics["lease_clear_audit_issue_counts"])

    def test_command_query_union_is_separate_from_monitor_census(self):
        report, events = self.fixture()
        report["frames"][0]["command_self_state"] = self.state(10.075)
        result = timing_metrics(report, events)
        monitor = result["monitor_event_receipt_cadence"]
        combined = result["all_saved_stage_receipt_cadence"]
        self.assertEqual(combined["unique_host_receipt_count"], monitor["unique_host_receipt_count"]+1)
        self.assertFalse(result["hard_real_time_demonstrated"])
        self.assertFalse(result["formal_efficacy_results"])

    def test_invalid_block_is_not_silently_counted_as_coverage(self):
        result = interval_evidence({"started_monotonic_s": 11., "completed_monotonic_s": 10.}, receipt_cadence([]), [], {})
        self.assertEqual(result["status"], "INVALID_ACTUAL_BLOCK_INTERVAL")

    def sealed_fixture(self, root):
        """A complete miniature frozen bundle, with no media or model loading."""
        snapshot = root / "source_snapshot"
        flight = root / "flight"
        snapshot.mkdir()
        flight.mkdir()
        prepared = root / "prepared"
        prepared.mkdir()
        def put(path, content):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return digest(path)
        source_path = prepared / "source.py"
        sha = put(source_path, "# frozen synthetic source\n")
        frozen_path = prepared / "frozen_source.py"
        put(frozen_path, "# frozen synthetic source\n")
        put(snapshot / "python/synthetic.py", "# frozen synthetic source\n")
        plan = {"session_id": "synthetic", "source_sha256": {"python/synthetic.py": sha},
            "source_inventory": [{"relative_path": "python/synthetic.py", "path": str(source_path),
                "snapshot_path": str(frozen_path), "sha256": sha}], "legacy_source_inventory": []}
        for index in range(12):
            original = prepared / f"legacy_{index}.py"
            frozen = prepared / f"legacy_{index}_frozen.py"
            content = f"# frozen legacy {index}\n"
            legacy_sha = put(original, content)
            put(frozen, content)
            put(snapshot / "legacy_v6p2" / original.name, content)
            plan["legacy_source_inventory"].append({"path": str(original), "snapshot_path": str(frozen), "sha256": legacy_sha})
        for key in ("settings", "detector_config", "training_receipt", "weights", "legacy_freeze"):
            path = prepared / (key+".json")
            item_sha = put(path, "{}\n")
            plan[key+"_path"], plan[key+"_sha256"] = str(path), item_sha
            put(snapshot / ("specialist_best.pt" if key == "weights" else path.name), "{}\n")
        plan_path = prepared / "plan.json"
        plan_content = json.dumps(plan)
        plan_sha = put(plan_path, plan_content)
        put(snapshot / "plan.json", plan_content)
        report, events = self.fixture()
        report.update(plan=str(plan_path), plan_sha256=plan_sha, configuration=plan, session_id="synthetic")
        execution = flight / "execution.jsonl"
        execution_sha = put(execution, "\n".join(json.dumps(row) for row in events)+"\n")
        report["execution_log"] = {"path": str(execution), "sha256": execution_sha}
        report_path = flight / "report.json"
        report_sha = put(report_path, json.dumps(report))
        put(root / "manifest.json", json.dumps({"status": "FAIL_RETAINED_SYNTHETIC", "runner_status": "FAIL",
            "report": str(report_path), "report_sha256": report_sha, "plan_sha256": plan_sha}))
        return report_path, plan_path, execution

    def test_sealed_failed_pilot_remains_failed_but_timing_can_be_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.sealed_fixture(root)
            result = analyze_run(root)
            self.assertEqual(result["status"], "SEALED_TIMING_ANALYZED")
            self.assertEqual(result["launcher_status"], "FAIL_RETAINED_SYNTHETIC")
            self.assertEqual(result["runner_status"], "FAIL")
            self.assertEqual(result["timing"]["nonzero_target_command_count"], 1)
            self.assertTrue(all(row["status"] == "PASS" for row in result["source_receipts"]))

    def test_report_plan_and_execution_hash_changes_each_refuse_statistics(self):
        for index in range(3):
            with self.subTest(component=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths = self.sealed_fixture(root)
                with paths[index].open("a", encoding="utf-8") as stream:
                    stream.write(" ")
                result = analyze_run(root)
                self.assertEqual(result["status"], "INPUT_HASH_VERIFICATION_FAILED")
                self.assertIsNone(result["timing"])

    def test_credentials_cannot_become_analysis_hash_inputs(self):
        with self.assertRaises(ValueError):
            digest(Path(".env"))


if __name__ == "__main__":
    unittest.main()
