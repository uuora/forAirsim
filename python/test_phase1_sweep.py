import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from analyze_phase1_sweep import analyze_sweep
import test_phase1_pilot_analysis as pilot_fixtures


class SweepTests(unittest.TestCase):
    def fixture(self, root, changed=False):
        entries = []
        helper = pilot_fixtures.PilotAnalysisTests()
        for index in range(2):
            folder = root / str(index)
            folder.mkdir()
            manifest = helper.fixture(folder)
            config = folder / "config.json"
            config.write_text(json.dumps({"target_world_ned_m": [0, index * .6, -3],
                "condition_id": str(index), "fixed_waypoint": index if changed else 0}))
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            pilot = json.loads(manifest.read_text())
            pilot["experiment_config_sha256"] = config_hash
            manifest.write_text(json.dumps(pilot))
            entries.append({"condition_id": str(index), "manifest": str(manifest),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "config": str(config), "config_sha256": config_hash})
        path = root / "sweep.json"
        path.write_text(json.dumps({"status": "PASS", "pilots": entries}))
        return path

    def test_target_variations_remain_separate_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            result = analyze_sweep(self.fixture(Path(folder)))
            self.assertEqual(len(result["conditions"]), 2)
            self.assertEqual(result["conditions"][0]["policies"]["visual"]["started"], 1)

    def test_changed_waypoint_baseline_refuses_condition_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "baseline/controller change"):
                analyze_sweep(self.fixture(Path(folder), changed=True))


if __name__ == "__main__":
    unittest.main()
