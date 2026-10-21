import unittest
from prepare_phase1_conditions import build
from single_drone_experiment import load_config


class ConditionTests(unittest.TestCase):
    def test_only_target_and_condition_metadata_change(self):
        base = load_config()
        original = base["target_world_ned_m"][:]
        conditions = build(base, original)
        for cfg in conditions.values():
            for key in base:
                if key not in ("target_world_ned_m", "protocol_id"):
                    self.assertEqual(cfg[key], base[key])
        self.assertEqual(base["target_world_ned_m"], original)
        self.assertEqual(conditions["target_y_plus_06"]["target_world_ned_m"][1], 0.6)
        self.assertEqual(conditions["target_y_minus_06"]["target_world_ned_m"][1], -0.6)


if __name__ == "__main__":
    unittest.main()
