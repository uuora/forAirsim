"""Budget and exploration properties, independently of UAV effect estimates."""
import unittest
import numpy as np
from trajectory_research.search import (
    PerturbationState, paired_perturbation, reflect_box, run_search,
)


def sphere(x):
    return dict(score=float(np.dot(x, x)), feasible=True)


class SearchTests(unittest.TestCase):
    def test_reflection_is_identity_inside_and_periodic_outside(self):
        x = np.array([-1., -.3, 0., .8, 1.])
        np.testing.assert_allclose(reflect_box(x), x, atol=1e-15)
        np.testing.assert_allclose(reflect_box([1.2, -1.2, 5.2]), [.8, -.8, .8])
        with self.assertRaises(ValueError):
            reflect_box([np.nan])

    def test_exact_budget_shared_initial_population_and_replay(self):
        kwargs = dict(lower=[-3., -2.], upper=[4., 5.], seed=7, population=8,
                      budget=40, anchor=[1., 1.])
        a = run_search(sphere, method="random", **kwargs)
        b = run_search(sphere, method="gaussian", **kwargs)
        c = run_search(sphere, method="gaussian", **kwargs)
        self.assertEqual(a["initial_population_normalized"], b["initial_population_normalized"])
        self.assertEqual(b["best_parameters"], c["best_parameters"])
        for result in (a, b, c):
            self.assertEqual(len(result["evaluations"]), 40)
            self.assertTrue(all(x["best"] >= y["best"] for x, y in zip(result["history"], result["history"][1:])))

    def test_antithetic_pairs_preserve_centres_before_reflection(self):
        parents = np.zeros((4, 3))
        state = PerturbationState(.025)
        points = paired_perturbation(parents, 8, np.random.default_rng(13), state)
        np.testing.assert_allclose(points[:4] + points[4:], 0, atol=1e-15)
        self.assertGreater(np.linalg.norm(points), 0)
        for _ in range(5):
            state.update(0.)
        self.assertGreaterEqual(state.scale, .2)

    def test_independent_and_clipped_ablations_have_declared_bounds(self):
        parents = np.full((4, 3), .99)
        state = PerturbationState(.5)
        for paired in (True, False):
            for boundary in ("clip", "reflect"):
                points = paired_perturbation(parents, 8, np.random.default_rng(3), state,
                                             antithetic=paired, boundary=boundary)
                self.assertTrue(np.all(np.abs(points) <= 1.))

    def test_invalid_budget_bounds_and_scores_fail_visibly(self):
        for budget in (7, 15, True):
            with self.assertRaises(ValueError):
                run_search(sphere, [-1], [1], population=8, budget=budget)
        with self.assertRaises(ValueError):
            run_search(sphere, [1], [1])
        with self.assertRaises(ValueError):
            run_search(lambda x: dict(score=np.nan, feasible=True), [-1], [1])


if __name__ == "__main__":
    unittest.main()
