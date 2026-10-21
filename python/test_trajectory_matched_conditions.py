"""Offline invariants for the frozen matched-start lateral pilot inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from prepare_trajectory_matched_conditions import (
    CONDITIONS, DEFAULT_BASE, DEFAULT_SETTINGS, INITIAL_STATE_MATCH,
    build_configs, prepare_conditions, session_plan,
)


class MatchedConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = json.loads(DEFAULT_BASE.read_text(encoding="utf-8-sig"))

    def test_three_inputs_change_only_target_y_and_condition_metadata(self):
        configs = build_configs(self.base, 20261003)
        self.assertEqual(len(configs), 3)
        for condition_id, offset in CONDITIONS:
            config = deepcopy(configs[condition_id])
            self.assertEqual(config["target_world_ned_m"], [0.0, offset, -3.0])
            self.assertEqual(config["target_offset_ned_m"], [0.0, offset, 0.0])
            config["target_world_ned_m"][1] = 0.0
            for name in ("condition_id", "target_offset_ned_m", "condition_metadata"):
                config.pop(name)
            if condition_id == "center":
                reference = config
            else:
                self.assertEqual(config, reference)

    def test_matching_geometry_candidates_and_budgets_are_preserved(self):
        for config in build_configs(self.base, 20261003).values():
            self.assertEqual(config["initial_state_match"], INITIAL_STATE_MATCH)
            for field in ("observation_world_ned_m", "workspace_min_ned_m", "workspace_max_ned_m",
                          "safety", "visual", "planning"):
                self.assertEqual(config[field], self.base[field])
            self.assertIsInstance(config["planning"]["fixed_world_height_ned_m"], (int, float))
            self.assertFalse(config["formal_thesis_effect_results"])

    def test_seeded_plan_has_two_independent_repetitions_and_balanced_orders(self):
        paths = {name: Path(f"/{name}.json") for name, _ in CONDITIONS}
        first = session_plan(20261003, paths)
        self.assertEqual(first, session_plan(20261003, paths))
        self.assertEqual(len(first), 6)
        self.assertEqual(first[0]["condition_id"], "center")
        self.assertEqual([item["condition_id"] for item in first[:3]],
                         list(reversed([item["condition_id"] for item in first[3:]])))
        for name in paths:
            sessions = [item for item in first if item["condition_id"] == name]
            self.assertEqual([item["repetition"] for item in sessions], [1, 2])
            self.assertEqual(sessions[0]["method_order"], ["efficiency", "perception"])
            self.assertEqual(sessions[1]["method_order"], ["perception", "efficiency"])
            self.assertTrue(all(item["independent_unreal_session"] for item in sessions))
            self.assertTrue(all(item["execution_status"] == "NOT_RUN" for item in sessions))

    def test_preparation_hashes_exact_config_bytes_and_never_invents_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "prepared"
            manifest_path = prepare_conditions(DEFAULT_BASE, output, settings_path=DEFAULT_SETTINGS)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["preparation_status"], "PREPARED_NOT_EXECUTED")
            self.assertEqual(manifest["planned_flights"], 12)
            self.assertEqual(manifest["planned_independent_unreal_sessions"], 6)
            self.assertEqual(manifest["execution_results"], [])
            for row in manifest["conditions"]:
                raw = Path(row["configuration_path"]).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["configuration_sha256"])
            for source in manifest["source_inputs"].values():
                self.assertEqual(hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest(), source["sha256"])

    def test_existing_directory_and_file_are_rejected_without_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "prepared"
            prepare_conditions(DEFAULT_BASE, output)
            before = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaises(FileExistsError):
                prepare_conditions(DEFAULT_BASE, output)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, before)
            existing_file = Path(folder) / "file"
            existing_file.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                prepare_conditions(DEFAULT_BASE, existing_file)
            self.assertEqual(existing_file.read_bytes(), b"preserve")

    def test_changed_budgets_and_string_height_rejected_before_creation(self):
        cases = []
        changed = deepcopy(self.base)
        changed["planning"]["max_acceleration_m_s2"] = 0.31
        cases.append(changed)
        changed = deepcopy(self.base)
        changed["planning"]["fixed_world_height_ned_m"] = "-3.0"
        cases.append(changed)
        changed = deepcopy(self.base)
        changed["workspace_max_ned_m"][1] = 0.7
        cases.append(changed)
        with tempfile.TemporaryDirectory() as folder:
            for index, config in enumerate(cases):
                base = Path(folder) / f"base{index}.json"
                base.write_text(json.dumps(config), encoding="utf-8")
                output = Path(folder) / f"out{index}"
                with self.assertRaises(ValueError):
                    prepare_conditions(base, output)
                self.assertFalse(output.exists())

    def test_mutation_of_one_condition_cannot_change_another_or_the_source(self):
        original = deepcopy(self.base)
        configs = build_configs(self.base, 20261003)
        configs["target_y_minus_0p4"]["initial_state_match"]["stable_s"] = 99
        configs["target_y_minus_0p4"]["planning"]["candidate_steps_x_m"].append(99)
        self.assertEqual(configs["center"]["initial_state_match"], INITIAL_STATE_MATCH)
        self.assertEqual(configs["center"]["planning"], original["planning"])
        self.assertEqual(self.base, original)


if __name__ == "__main__":
    unittest.main()
