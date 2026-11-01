import unittest

import numpy as np

from trajectory_research.parameterizations import default_parameters
from trajectory_research.perturbation_task import (
    GEOMETRY_FACTOR, Layout, Obstacle, evaluate_parameters, get_layout, layouts, shape_rms, stress_obstacles,
)


class PerturbationTaskTests(unittest.TestCase):
    def test_shared_warmstart_is_feasible_for_every_layout_and_family(self):
        for family in ("single_bump", "paired_bump"):
            for layout in layouts():
                result = evaluate_parameters(default_parameters(family), family=family, layout=layout)
                self.assertTrue(result["feasible"], (family, layout.name, result))
                self.assertLess(result["score"], 1.)

    def test_nested_subspace_has_same_cost_and_certificate_bounds(self):
        for d in ([0., .8, 0.], [.2, 1., -.3], [-.5, -.6, .4]):
            for duration in (3., 5.5, 7.):
                a = evaluate_parameters([*d, duration], family="single_bump")
                b = evaluate_parameters([*d, 0., 0., 0., duration], family="paired_bump")
                self.assertEqual(a["feasible"], b["feasible"])
                self.assertAlmostEqual(a["score"], b["score"], places=11)
                np.testing.assert_allclose(a["score_details"]["derivative_bounds"],
                                           b["score_details"]["derivative_bounds"], atol=1e-10)

    def test_shape_rms_matches_independent_gaussian_quadrature(self):
        u, weights = np.polynomial.legendre.leggauss(8)
        u, weights = (u + 1.) / 2., weights / 2.
        d, e = np.array([.3, -.5, .2]), np.array([1.7, -.8, .5])
        deformation = (64 * u ** 3 * (1 - u) ** 3)[:, None] * (d + (u[:, None] - .5) * e)
        actual = np.sqrt(np.sum(weights * np.sum(deformation ** 2, axis=1)))
        self.assertAlmostEqual(shape_rms([*d, *e, 5.]), actual, places=12)

    def test_equivalent_representations_have_same_decisions_and_cost(self):
        for layout in layouts():
            for family in ("single_bump", "paired_bump"):
                vector = default_parameters(family)
                if family == "paired_bump":
                    vector[3:6] = [.8, -.4, .2]
                a = evaluate_parameters(vector, family=family, layout=layout, representation="hermite")
                b = evaluate_parameters(vector, family=family, layout=layout, representation="bezier")
                self.assertEqual(a["feasible"], b["feasible"])
                self.assertAlmostEqual(a["score"], b["score"], places=9)

    def test_stress_replays_and_all_obstacles_respect_the_envelope(self):
        layout = get_layout("two_obstacles")
        scenes = stress_obstacles(layout, seed=77, count=12)
        self.assertEqual(scenes, stress_obstacles(layout, seed=77, count=12))
        factor = np.asarray(GEOMETRY_FACTOR)
        for scene in scenes:
            self.assertEqual(len(scene), 2)
            for nominal, actual in zip(layout.obstacles, scene):
                delta = np.asarray(actual.centre) - nominal.centre
                self.assertLessEqual(np.linalg.norm(np.linalg.solve(factor, delta)), 1. + 1e-12)
                self.assertTrue(.35 <= actual.radius <= .45)
            result = evaluate_parameters(default_parameters("paired_bump"), layout=layout,
                                         actual_obstacles=scene, robust=False)
            self.assertTrue(result["feasible"])

    def test_invalid_layout_and_validation_mode(self):
        for name in ("", "wind"):
            with self.assertRaises(ValueError):
                evaluate_parameters(default_parameters(), layout=name)
        with self.assertRaises(ValueError):
            evaluate_parameters(default_parameters(), actual_obstacles=get_layout("central").obstacles)
        for seed, count in ((-1, 2), (True, 2), (2, 0), (2, 10001)):
            with self.assertRaises(ValueError):
                stress_obstacles("central", seed=seed, count=count)

    def test_custom_obstacle_radius_is_perturbed_about_its_own_radius(self):
        scene = Layout("custom_radius", (Obstacle((4., 0., 5.), .8), Obstacle((2., 0., 5.), .02)))
        for actual in stress_obstacles(scene, seed=4, count=12):
            self.assertTrue(.75 <= actual[0].radius <= .85)
            self.assertTrue(0. < actual[1].radius <= .07)


if __name__ == "__main__":
    unittest.main()
