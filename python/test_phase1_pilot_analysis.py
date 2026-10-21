"""Guard against dropping failures or pooling changed controllers in the pilot."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analyze_phase1_pilot import analyze
from run_single_batch import main as run_batch
from release_grounded_single import grounded


class PilotAnalysisTests(unittest.TestCase):
    def fixture(self, root, failed=False, changed=False):
        batch_folder = root / "batch"
        batch_folder.mkdir()
        rows = []
        for index, policy in enumerate(("coordinate", "visual"), 1):
            status = "FAIL" if failed and policy == "visual" else "PASS"
            metrics = {} if status == "FAIL" else {
                "task_success": True, "approach_duration_s": index * 10.0,
                "approach_path_m": 3.0 - index * 0.5}
            path = root / f"{policy}.json"
            path.write_text(json.dumps({"status": status, "configuration": {"fixed": True},
                "source_sha256": {"control": "changed" if changed and index == 2 else "v1"},
                "metrics": metrics}), encoding="utf-8")
            rows.append({"run": index, "pair": 1, "policy": policy,
                         "status": status, "task_success": metrics.get("task_success"),
                         "report_path": str(path)})
        batch_path = batch_folder / "batch_report.json"
        batch_path.write_text(json.dumps({"status": "FAIL" if failed else "PASS", "runs": rows}), encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps({"status": "FAIL" if failed else "PASS",
            "planned_sessions": 1, "sessions": [{"session_index": 1,
            "engine_process_id": 123, "status": "PASS", "batch_path": str(batch_folder),
            "batch_sha256": hashlib.sha256(batch_path.read_bytes()).hexdigest()}]}), encoding="utf-8")
        return manifest_path

    def test_failed_run_stays_in_denominator_and_pair_is_not_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = analyze(self.fixture(Path(folder), failed=True))
            self.assertEqual(summary["complete_pair_count"], 0)
            self.assertEqual(summary["policies"]["visual"]["started"], 1)
            self.assertEqual(summary["policies"]["visual"]["unassessed"], 1)

    def test_completed_pair_difference_keeps_visual_minus_coordinate_sign(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = analyze(self.fixture(Path(folder)))
            pair = summary["pairs"][0]
            self.assertEqual(pair["visual_minus_coordinate_approach_duration_s"], 10.0)
            self.assertEqual(pair["visual_minus_coordinate_approach_path_m"], -0.5)

    def test_modified_controller_versions_refuse_pooled_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Mixed controller"):
                analyze(self.fixture(Path(folder), changed=True))

    def test_batch_hash_detects_changed_record(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.fixture(root)
            with (root / "batch" / "batch_report.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "hash no longer matches"):
                analyze(manifest)

    def test_batch_refuses_existing_output_before_connecting_to_simulator(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Output already exists"):
                run_batch(2, output_dir=folder)

    def test_started_flight_without_terminal_report_is_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.fixture(root)
            batch_path = root / "batch" / "batch_report.json"
            batch = json.loads(batch_path.read_text())
            batch["active_run"] = {"run": 3, "pair": 2, "policy": "visual",
                                   "report_path": str(root / "missing.json")}
            batch_path.write_text(json.dumps(batch))
            data = json.loads(manifest.read_text())
            data["status"] = "FAIL"
            data["sessions"][0]["batch_sha256"] = hashlib.sha256(batch_path.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(data))
            result = analyze(manifest)
            self.assertEqual(result["policies"]["visual"]["started"], 2)
            self.assertEqual(result["policies"]["visual"]["unassessed"], 1)
            self.assertEqual(result["sessions"][0]["started_runs"], 3)

    def test_grounded_release_rejects_airborne_or_controlled_bystander(self):
        a = {"position": [-5, -4, 0.817], "velocity": [0, 0, 0], "api_control": True,
             "landed_state": 1, "collision_object": "Ground_5", "collision_timestamp": 1}
        b = {"position": [0, 3, 0.808], "velocity": [0, 0, 0], "api_control": False,
             "landed_state": 0}
        self.assertTrue(grounded({"DroneA": a, "DroneB": b}, [-5, -4, -3]))
        self.assertFalse(grounded({"DroneA": dict(a, position=[-5, -4, -1]), "DroneB": b}, [-5, -4, -3]))
        self.assertFalse(grounded({"DroneA": a, "DroneB": dict(b, api_control=True)}, [-5, -4, -3]))


if __name__ == "__main__":
    unittest.main()
