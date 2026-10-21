from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_trajectory_matched_sweep import audit_sweep, main, sha256
from analyze_trajectory_pilot import summarize_run


class MatchedSweepTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.conditions_path = self.root / "conditions.json"
        self.sweep_path = self.root / "sweep_report.json"
        self.match_specification = {"enabled": True, "position_tolerance_m": .015,
            "speed_tolerance_m_s": .01, "acceleration_tolerance_m_s2": .05,
            "attitude_tolerance_deg": 1., "stable_s": .5, "max_sample_gap_s": .25,
            "reposition_speed_cap_m_s": .05}
        conditions, plans = [], []
        for condition in ("center", "left", "right"):
            config = {"condition_id": condition, "planning": {"execution_speed_cap_m_s": .15,
                "max_acceleration_m_s2": .3}, "initial_state_match": self.match_specification}
            path = self.root / f"{condition}.json"
            self.write(path, config)
            conditions.append({"condition_id": condition, "configuration_path": str(path),
                               "configuration_sha256": sha256(path)})
        for repetition in (1, 2):
            for condition in conditions:
                plans.append({"planned_session_index": len(plans) + 1, "condition_id": condition["condition_id"],
                    "configuration_path": condition["configuration_path"], "repetition": repetition,
                    "target_source": "sensor_surface_estimate",
                    "method_order": ["efficiency", "perception"] if repetition == 1 else ["perception", "efficiency"]})
        self.conditions = {"conditions": conditions, "session_plan": plans}
        self.write(self.conditions_path, self.conditions)
        self.sweep = {"status": "STOPPED_FOR_INSPECTION", "conditions_path": str(self.conditions_path),
            "conditions_sha256": sha256(self.conditions_path), "planned_sessions": 6, "planned_flights": 12, "sessions": []}
        self.save_sweep()

    def write(self, path, value):
        Path(path).write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

    def save_sweep(self):
        self.write(self.sweep_path, self.sweep)

    def report(self, policy, config_hash, source, positioning_failure=False):
        sample = {"world_ned_m": [-5., -.5, -3.], "velocity_ned_m_s": [0., 0., 0.],
            "acceleration_ned_m_s2": [0., 0., 0.], "actual_body_rpy_deg": [0., 0., 0.],
            "previous_command_velocity_xy_m_s": [0., 0.], "accepted": True}
        samples = [dict(deepcopy(sample), host_monotonic_s=10. + index * .2, query_state_timestamp_ns=100 + index)
                   for index in range(4)]
        report = {"policy": policy, "target_source": "sensor_surface_estimate", "status": "PASS",
            "task_success": not positioning_failure, "source_sha256": source, "configuration_sha256": config_hash,
            "configuration": {"planning": {"execution_speed_cap_m_s": .15, "max_acceleration_m_s2": .3}},
            "landing_evidence": {"landed_flag": True, "api_control_released": True},
            "initial_state_match": {"status": "PROTECTED_RETURN" if positioning_failure else "ACCEPTED",
                "specification": deepcopy(self.match_specification), "desired_world_ned_m": [-5., -.5, -3.],
                "samples": samples, "accepted_state": None if positioning_failure else deepcopy(samples[-1]),
                "elapsed_s": .7, "protective_reason": "POSITIONING_TIMEOUT" if positioning_failure else None},
            "metrics": {"approach_duration_s": None if positioning_failure else 2., "approach_path_m": 0. if positioning_failure else .2,
                        "terminal_mean_max_axis_error": None if positioning_failure else .01},
            "observations": [] if positioning_failure else [{"detected": True, "error_xy": [.03, .02],
                "world_ned_m": [-4.9, -.5, -3.], "planning_input_body_state": dict(deepcopy(sample), query_state_timestamp_ns=105)}],
            "planning_records": [] if positioning_failure else [{"solver_elapsed_s": .03,
                "execution_samples": [{"command_velocity_xy_m_s": [.1, 0], "speed_norm_m_s": .1,
                                       "acceleration_norm_m_s2": .1, "query_state_timestamp_ns": 106}]}]}
        return report

    def add_session(self, index, fail_policy=None, source=None, started_count=2):
        plan = self.conditions["session_plan"][index - 1]
        condition = next(row for row in self.conditions["conditions"] if row["condition_id"] == plan["condition_id"])
        source = source or {"controller.py": "frozen-source"}
        folder = self.root / f"session_{index}"
        folder.mkdir()
        batch_folder = folder / "batch"
        batch_folder.mkdir()
        reports, entries = [], []
        for position, policy in enumerate(plan["method_order"][:started_count], 1):
            report = self.report(policy, condition["configuration_sha256"], source, policy == fail_policy)
            report_path = folder / f"{policy}_report.json"
            self.write(report_path, report)
            reports.append(report)
            entries.append({"run": position, "policy": policy, "status": "PASS", "report_path": str(report_path),
                            "report_sha256": sha256(report_path)})
        batch_path = batch_folder / "batch_report.json"
        self.write(batch_path, {"configuration_sha256": condition["configuration_sha256"],
            "target_source": "sensor_surface_estimate", "planned_policies": plan["method_order"], "runs": entries})
        manifest = {"status": "PASS", "planned_sessions": 1, "planned_flights_per_session": 2,
            "target_source": "sensor_surface_estimate", "experiment_config_sha256": condition["configuration_sha256"],
            "sessions": [{"engine_process_id": 1000 + index, "engine_started": f"2026-10-03T03:{index:02d}:00Z",
                "engine_closed_after_safe_completion": started_count == 2, "batch_path": str(batch_folder),
                "batch_report_sha256": sha256(batch_path)}]}
        manifest_path = folder / "manifest.json"
        self.write(manifest_path, manifest)
        analysis = {"manifest_sha256": sha256(manifest_path), "started_flights": len(entries),
            "same_run_source_versions": source, "runs": [dict(summarize_run(report), report_path=entry["report_path"])
                                                        for report, entry in zip(reports, entries)]}
        analysis_path = folder / "summary.json"
        self.write(analysis_path, analysis)
        entry = {"planned_session_index": index, "condition_id": plan["condition_id"], "repetition": plan["repetition"],
            "policy_order": plan["method_order"], "configuration": plan["configuration_path"], "status": "COMPLETED",
            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "analysis": str(analysis_path), "analysis_sha256": sha256(analysis_path)}
        self.sweep["sessions"].append(entry)
        self.save_sweep()
        return entry, batch_path

    def test_positioning_failure_is_started_failed_task_with_independent_landing(self):
        self.add_session(1, fail_policy="perception")
        summary, points = audit_sweep(self.sweep_path)
        self.assertEqual(summary["planned_flights"], 12)
        self.assertEqual(summary["known_started_flights"], 2)
        self.assertEqual(summary["verified_not_started_flights"], 10)
        self.assertEqual(summary["task_success_count"], 1)
        self.assertEqual(summary["safe_landing_and_release_count"], 2)
        perception = next(row for row in summary["condition_policy_descriptions"] if row["condition_id"] == "center" and row["policy"] == "perception")
        self.assertEqual(perception["initial_acceptance_failure_count"], 1)
        self.assertEqual(perception["task_failure_reason_counts"], {"POSITIONING_TIMEOUT": 1})
        self.assertEqual(perception["task_rate_known_started"], 0.)
        self.assertEqual(perception["approach_duration_s"]["assessed_count"], 0)
        self.assertEqual(len(points), 1)

    def test_unstarted_second_policy_is_preserved_without_inflating_started_count(self):
        self.add_session(1, started_count=1)
        summary, _ = audit_sweep(self.sweep_path)
        self.assertEqual(summary["known_started_flights"], 1)
        self.assertEqual(summary["verified_not_started_flights"], 11)
        self.assertFalse(summary["runs"][1]["started"])
        self.assertEqual(summary["initial_state_pairs"]["paired_session_sample_count"], 0)

    def test_started_session_missing_manifest_keeps_flight_starts_unknown(self):
        plan = self.conditions["session_plan"][0]
        self.sweep["sessions"] = [{"planned_session_index": 1, "condition_id": "center", "repetition": 1,
            "policy_order": plan["method_order"], "configuration": plan["configuration_path"], "status": "STARTED"}]
        self.save_sweep()
        summary, _ = audit_sweep(self.sweep_path)
        self.assertEqual(summary["known_started_flights"], 0)
        self.assertEqual(summary["unknown_start_status_flights"], 2)
        self.assertEqual(summary["verified_not_started_flights"], 10)
        self.assertIsNone(summary["runs"][0]["started"])
        self.assertFalse(summary["condition_policy_descriptions"][0]["started_denominator_complete"])

    def test_started_missing_terminal_report_remains_in_denominator(self):
        entry, batch_path = self.add_session(1)
        batch = json.loads(batch_path.read_text())
        Path(batch["runs"][1]["report_path"]).unlink()
        summary, _ = audit_sweep(self.sweep_path)
        self.assertEqual(summary["known_started_flights"], 2)
        self.assertEqual(summary["missing_terminal_report_count"], 1)
        self.assertEqual(summary["task_success_count"], 1)
        self.assertEqual(summary["independent_launch_evidence_verified_session_count"], 0)

    def test_changed_source_version_refuses_mixed_sweep(self):
        self.add_session(1)
        self.add_session(2, source={"controller.py": "changed-source"})
        with self.assertRaisesRegex(ValueError, "Different source versions"):
            audit_sweep(self.sweep_path)

    def test_method_order_mismatch_refuses_unplanned_comparison(self):
        entry, _ = self.add_session(1)
        entry["policy_order"] = ["perception", "efficiency"]
        self.save_sweep()
        with self.assertRaisesRegex(ValueError, "method order"):
            audit_sweep(self.sweep_path)

    def test_summary_hash_mismatch_is_rejected(self):
        entry, _ = self.add_session(1)
        Path(entry["analysis"]).write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "analysis hash mismatch"):
            audit_sweep(self.sweep_path)

    def test_reused_engine_pid_and_start_time_cannot_count_as_independent_launch(self):
        first, _ = self.add_session(1)
        second, _ = self.add_session(2)
        first_manifest = json.loads(Path(first["manifest"]).read_text())
        second_manifest = json.loads(Path(second["manifest"]).read_text())
        for key in ("engine_process_id", "engine_started"):
            second_manifest["sessions"][0][key] = first_manifest["sessions"][0][key]
        self.write(second["manifest"], second_manifest)
        second["manifest_sha256"] = sha256(second["manifest"])
        self.save_sweep()
        with self.assertRaisesRegex(ValueError, "Repeated Unreal PID/StartTime"):
            audit_sweep(self.sweep_path)

    def test_all_six_sessions_have_distinct_launch_evidence_and_balanced_order(self):
        for index in range(1, 7):
            self.add_session(index)
        summary, _ = audit_sweep(self.sweep_path)
        self.assertEqual(summary["known_started_flights"], 12)
        self.assertEqual(summary["task_success_count"], 12)
        self.assertEqual(summary["independent_launch_evidence_verified_session_count"], 6)
        self.assertEqual(summary["initial_state_pairs"]["paired_session_sample_count"], 6)
        self.assertEqual(summary["runs"][6]["policy"], "perception")
        for group in summary["condition_policy_descriptions"]:
            self.assertEqual(group["known_started_flights"], 2)
            self.assertAlmostEqual(group["per_run_mean_solver_s"]["mean"], .03)
        with patch("analyze_trajectory_matched_sweep.render"):
            main(self.sweep_path, self.root / "output")
        self.assertTrue((self.root / "output" / "provenance.json").exists())
        with self.assertRaises(FileExistsError):
            main(self.sweep_path, self.root / "output")


if __name__ == "__main__":
    unittest.main()
