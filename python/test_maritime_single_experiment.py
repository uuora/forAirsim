import json
import unittest

from maritime_experiments import ROOT, balloon_locations
from maritime_single_experiment import associate_selected, box_iou, load_config


class MaritimeSingleTests(unittest.TestCase):
    def test_config_covers_fixed_observation_and_perturbations(self):
        cfg = load_config()
        self.assertEqual(cfg["observation_local_ned_m"], [0.0, 0.0, -6.0])
        self.assertIn("similar_red", cfg["profiles"])
        self.assertIn("target_occluded", cfg["profiles"])

    def test_selected_box_distinguishes_target_from_similar_instance(self):
        boxes = {"balloon_001": [10, 10, 30, 30],
                 "balloon_003": [40, 10, 60, 30]}
        selected = {"bbox_xywh": [41, 11, 18, 18]}
        self.assertEqual(associate_selected(selected, boxes, "balloon_001")[0],
                         "OTHER_INSTANCE")
        self.assertEqual(associate_selected(None, boxes, "balloon_001")[0],
                         "NO_CANDIDATE")
        self.assertEqual(box_iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_profile_locations_keep_instances_separate(self):
        profiles = json.loads((ROOT / "configs" / "maritime_experiments.json").read_text(
            encoding="utf-8"))["profiles"]
        for profile in profiles.values():
            self.assertEqual(len(balloon_locations(profile)), 3)
        bad = dict(profiles["similar_red"])
        bad["balloon_positions_relative_hull_m"] = [[4, 9.5, -1.5],
            [11, -10, -1.7], [4, 9.7, -1.45]]
        with self.assertRaisesRegex(ValueError, "at least 2 m"):
            balloon_locations(bad)

    def test_layout_pilot_preregisters_balanced_target_sides(self):
        cfg = load_config(ROOT / "configs" / "maritime_single_layout_pilot.json")
        sides = [cfg["tasks"][profile]["expected_target_side"]
                 for profile in cfg["profiles"]]
        self.assertEqual(sides.count("left"), 2)
        self.assertEqual(sides.count("right"), 2)
        profiles = json.loads((ROOT / "configs" / "maritime_experiments.json").read_text(
            encoding="utf-8"))["profiles"]
        narrow_right = balloon_locations(profiles["similar_red"])
        narrow_left = balloon_locations(profiles["similar_red_left"])
        wide_right = balloon_locations(profiles["similar_red_wide_right"])
        wide_left = balloon_locations(profiles["similar_red_wide_left"])
        self.assertEqual(narrow_right[0][:2], narrow_left[2][:2])
        self.assertEqual(narrow_left[0][:2], narrow_right[2][:2])
        self.assertEqual(wide_right[0][:2], wide_left[2][:2])
        self.assertEqual(wide_left[0][:2], wide_right[2][:2])
        narrow_gap = abs(narrow_right[0][1] - narrow_right[2][1])
        wide_gap = abs(wide_right[0][1] - wide_right[2][1])
        self.assertGreater(wide_gap, narrow_gap)

    def test_session_config_preserves_pilot_tasks(self):
        pilot = load_config(ROOT / "configs" / "maritime_single_layout_pilot.json")
        sessions = load_config(ROOT / "configs" / "maritime_single_layout_sessions.json")
        self.assertEqual(sessions["profiles"], pilot["profiles"])
        self.assertEqual(sessions["tasks"], pilot["tasks"])
        self.assertEqual(sessions["frames_per_profile"], 3)


if __name__ == "__main__":
    unittest.main()
