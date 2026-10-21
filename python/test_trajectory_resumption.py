import json
from pathlib import Path

from prepare_trajectory_resumption import prepare
from summarize_trajectory_resumption import audit_startup_failure, combine
from analyze_trajectory_matched_sweep import audit_sweep, sha256
from test_trajectory_matched_sweep import MatchedSweepTests


class ResumptionTests(MatchedSweepTests):
    # Use the existing realistic frozen-ledger fixture, not its test methods.
    __unittest_skip__ = False

    def interrupted(self):
        self.conditions["protocol_id"] = "fixture_protocol"
        self.write(self.conditions_path, self.conditions)
        self.sweep["conditions_sha256"] = sha256(self.conditions_path)
        entry, batch_path = self.add_session(1, started_count=1)
        batch = json.loads(batch_path.read_text())
        batch["status"] = "RUNNING"
        Path(batch["runs"][0]["report_path"]).unlink()
        batch["runs"][0].pop("report_sha256")
        self.write(batch_path, batch)
        manifest = json.loads(Path(entry["manifest"]).read_text())
        manifest["status"] = "RUNNING"
        manifest["sessions"][0].pop("batch_report_sha256")
        self.write(entry["manifest"], manifest)
        entry.update(status="STARTED")
        for key in ("manifest", "manifest_sha256", "analysis", "analysis_sha256"):
            entry.pop(key, None)
        self.sweep["status"] = "RUNNING"
        self.save_sweep()
        return self.root / "session_1" / "manifest.json", batch_path

    def test_reconcile_keeps_original_bytes_and_missing_terminal_start(self):
        manifest, _ = self.interrupted()
        original_hash = sha256(self.sweep_path)
        manifest_hash = sha256(manifest)
        output = self.root / "reconcile"
        receipt = prepare(self.sweep_path, manifest, output)
        self.assertEqual(sha256(self.sweep_path), original_hash)
        self.assertEqual(sha256(manifest), manifest_hash)
        self.assertEqual(receipt["missing_terminal_report_count"], 1)
        summary, _ = audit_sweep(output / "original_reconciled_sweep.json")
        self.assertEqual(summary["known_started_flights"], 1)
        self.assertEqual(summary["missing_terminal_report_count"], 1)
        self.assertEqual(summary["safe_landing_and_release_count"], 0)
        replacement = json.loads((output / "separate_repeat_conditions.json").read_text())
        self.assertEqual(replacement["planned_flights"], 2)
        self.assertEqual(replacement["session_plan"][0]["method_order"], self.conditions["session_plan"][0]["method_order"])
        self.assertEqual(replacement["conditions"][0]["configuration_sha256"], self.conditions["conditions"][0]["configuration_sha256"])
        with self.assertRaises(FileExistsError):
            prepare(self.sweep_path, manifest, output)

    def test_reconcile_rejects_changed_frozen_configuration(self):
        manifest, _ = self.interrupted()
        Path(self.conditions["conditions"][0]["configuration_path"]).write_text("{}")
        with self.assertRaisesRegex(ValueError, "configuration hash mismatch"):
            prepare(self.sweep_path, manifest, self.root / "invalid")

    def test_reconcile_does_not_label_existing_terminal_reports_interrupted(self):
        manifest, batch_path = self.interrupted()
        batch = json.loads(batch_path.read_text())
        self.write(batch["runs"][0]["report_path"], {})
        with self.assertRaisesRegex(ValueError, "No missing terminal report"):
            prepare(self.sweep_path, manifest, self.root / "invalid")

    def test_combine_retains_interrupted_start_and_unknown_landing(self):
        self.add_session(1, started_count=1)
        first, first_points = audit_sweep(self.sweep_path)
        first["runs"][0].update(ledger_status="STARTED_NO_TERMINAL_REPORT", task_success=False, safe_landing_and_release=False)
        first["same_source_versions"] = None
        # Separate second input with a different report path, retaining all rows.
        self.add_session(2)
        second, second_points = audit_sweep(self.sweep_path)
        second["runs"] = [r for r in second["runs"] if r["session"] == 2]
        second["sessions"] = [r for r in second["sessions"] if r["planned_session_index"] == 2]
        second_points = [r for r in second_points if r["planned_session_index"] == 2]
        result, _ = combine([(first, first_points), (second, second_points)])
        self.assertEqual(result["known_started_flights"], 3)
        self.assertEqual(result["missing_terminal_report_count"], 1)
        self.assertEqual(result["safe_landing_and_release_count"], 2)
        self.assertEqual(result["task_success_count"], 2)

    def test_combine_rejects_duplicate_started_report_path(self):
        self.add_session(1)
        audited = audit_sweep(self.sweep_path)
        with self.assertRaisesRegex(ValueError, "Duplicate started report"):
            combine([audited, audited])

    def test_combine_rejects_different_terminal_source_versions(self):
        self.add_session(1)
        first = audit_sweep(self.sweep_path)
        second = audit_sweep(self.sweep_path)
        second[0]["same_source_versions"] = {"controller.py": "different"}
        with self.assertRaisesRegex(ValueError, "Different terminal-report source versions"):
            combine([first, second])

    def startup_fixture(self):
        manifest = self.root / "failed_startup_manifest.json"
        batch_folder = self.root / "unstarted_batch"
        self.write(manifest, {"status": "FAIL", "error": "Unreal startup timed out: log",
            "experiment_config_sha256": self.conditions["conditions"][0]["configuration_sha256"],
            "sessions": [{"batch_path": str(batch_folder)}]})
        sweep = self.root / "failed_startup_sweep.json"
        self.write(sweep, {"status": "STOPPED_FOR_INSPECTION",
            "conditions_path": str(self.conditions_path), "conditions_sha256": sha256(self.conditions_path), "sessions": [{
            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
            "condition_id": "center", "repetition": 2,
            "policy_order": ["perception", "efficiency"]}]})
        return sweep, manifest, batch_folder

    def test_rpc_timeout_before_dispatch_is_two_unstarted_flights(self):
        sweep, _, _ = self.startup_fixture()
        summary, points = audit_startup_failure(sweep)
        self.assertEqual(len(summary["runs"]), 2)
        self.assertTrue(all(row["started"] is False for row in summary["runs"]))
        self.assertFalse(summary["sessions"][0]["independent_launch_evidence_verified"])
        self.assertEqual(points, [])

    def test_existing_batch_prevents_no_flights_startup_claim(self):
        sweep, _, batch_folder = self.startup_fixture()
        batch_folder.mkdir()
        self.write(batch_folder / "batch_report.json", {"runs": []})
        with self.assertRaisesRegex(ValueError, "beyond RPC startup"):
            audit_startup_failure(sweep)

    def test_combine_rejects_reused_engine_identity_between_attempts(self):
        self.add_session(1)
        first = audit_sweep(self.sweep_path)
        second = audit_sweep(self.sweep_path)
        second[0]["runs"] = []
        with self.assertRaisesRegex(ValueError, "Repeated engine PID/StartTime"):
            combine([first, second])


# Inherited test methods would duplicate the main suite. Keep only the fixture
# and the six interruption-specific checks in this module's class.
for name in dir(MatchedSweepTests):
    if name.startswith("test_"):
        setattr(ResumptionTests, name, None)
del MatchedSweepTests
