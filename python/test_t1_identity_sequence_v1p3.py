"""New cadence is frozen before collection and cannot rewrite historical grids."""
import unittest
from prepare_t1_identity_sequence import build_protocol
from t1_identity_schedule import boundary_clipped_slots
from analyze_t1_identity_sequence import expected_slots


class CadenceTests(unittest.TestCase):
    def config(self):
        value=build_protocol("x",6100410,"VISUAL_ATTRIBUTE")
        value.update(timing_grid="BOUNDARY_CLIPPED_0P4",minimum_period_s=.4)
        return value
    def test_all_twelve_seconds_and_phase_boundaries(self):
        rows=boundary_clipped_slots(self.config())
        self.assertEqual(len(rows),31)
        self.assertEqual(rows[0]["planned_start_relative_s"],0)
        self.assertEqual(rows[-1]["planned_end_relative_s"],12)
        for left,right in zip(rows,rows[1:]):
            self.assertEqual(left["planned_end_relative_s"],right["planned_start_relative_s"])
        self.assertEqual(sum(abs(row["planned_end_relative_s"]-row["planned_start_relative_s"]-.2)<1e-8 for row in rows),2)
        self.assertEqual([sum(row["phase_id"]==name for row in rows) for name in
            ("initial_pair","target_hidden_with_distractor_at_old_target","return_target_right")],[13,5,13])
    def test_evaluator_reconstructs_registered_grid(self):
        self.assertEqual(len(expected_slots(self.config())),31)
    def test_original_point_two_grid_stays_sixty(self):
        self.assertEqual(len(expected_slots(build_protocol("old",6100410,"VISUAL_ATTRIBUTE"))),60)
    def test_cannot_silently_change_cadence(self):
        config=self.config();config["minimum_period_s"]=.5
        with self.assertRaises(ValueError):
            boundary_clipped_slots(config)


if __name__=="__main__":
    unittest.main()
