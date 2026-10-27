import unittest
import numpy as np
from trajectory_research.directional_task import (
    FACTOR, evaluate_directional_parameters, make_directional_stress_cases,
)
from trajectory_research.curve_task import evaluate_curve_parameters


class DirectionalTaskTests(unittest.TestCase):
    def test_profiles_keep_quality_and_end_states_fixed(self):
        vector = [0, .7, 0, 5]
        rows = [evaluate_directional_parameters(vector, profile=p) for p in
                ("ellipsoid_nominal", "ellipsoid_isotropic", "ellipsoid_directional")]
        self.assertEqual(len({r["score_details"]["quality_term"] for r in rows}), 1)
        self.assertTrue(rows[0]["feasible"])
        self.assertFalse(rows[1]["feasible"])
        self.assertTrue(rows[2]["feasible"])

    def test_stress_cases_are_replayable_and_within_declared_ellipsoid(self):
        cases = make_directional_stress_cases(9182, count=32)
        self.assertEqual(cases, make_directional_stress_cases(9182, count=32))
        for case in cases:
            delta = np.asarray(case.centre_m) - [4, 0, 5]
            self.assertLessEqual(np.linalg.norm(np.linalg.solve(FACTOR, delta)), 1 + 1e-12)
            self.assertTrue(.35 <= case.radius_m <= .45)
        self.assertTrue(evaluate_curve_parameters([0, .7, 0, 5], cases)["feasible"])

    def test_invalid_profile_and_stress_arguments(self):
        with self.assertRaises(ValueError):
            evaluate_directional_parameters([0, .7, 0, 5], profile="wind")
        for seed, count in ((True, 2), (-1, 2), (1, 0), (1, 10001)):
            with self.assertRaises(ValueError):
                make_directional_stress_cases(seed, count=count)


if __name__ == "__main__":
    unittest.main()
