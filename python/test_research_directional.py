"""Checks of directional geometric uncertainty certificates, without a vehicle."""
import unittest

import numpy as np

from trajectory_research.constraints import sphere_clearance_certificate
from trajectory_research.curves import bezier_segment, c2_bump, quintic_hermite
from trajectory_research.directional import ellipsoid_sphere_certificate


class DirectionalTests(unittest.TestCase):
    def setUp(self):
        self.base = quintic_hermite([0, 0, 5], [2, 0, 0], [0, 0, 0],
                                    [8, 0, 5], [2, 0, 0], [0, 0, 0], 4)
        self.centre = np.array([4., 0., 5.])

    def test_zero_factor_matches_known_sphere_cases(self):
        for displacement in (0., .8):
            curve = c2_bump(self.base, [0, displacement, 0])
            reference = sphere_clearance_certificate(curve, self.centre, .4, clearance=.1)
            result = ellipsoid_sphere_certificate(curve, self.centre, .4, np.zeros((3, 3)),
                                                   radius_increment=0., clearance=.1)
            self.assertEqual(result["status"], reference["status"])

    def test_isotropic_factor_respects_inflated_sphere(self):
        for displacement in (.2, .5, .8, 1.1):
            curve = c2_bump(self.base, [0, displacement, 0])
            inflated = sphere_clearance_certificate(curve, self.centre, .75, clearance=.1)
            result = ellipsoid_sphere_certificate(curve, self.centre, .4, .3 * np.eye(3))
            if inflated["status"] == "VIOLATED":
                self.assertNotEqual(result["status"], "CERTIFIED")
            if result["status"] == "CERTIFIED":
                positions = curve.evaluate(np.linspace(0, 1, 1001))
                self.assertGreater(float(np.min(np.linalg.norm(positions - self.centre, axis=1))), .85)

    def test_directional_bound_reduces_isotropic_conservatism(self):
        curve = c2_bump(self.base, [0, .65, 0])
        factor = np.diag([.3, .05, .05])
        directional = ellipsoid_sphere_certificate(curve, self.centre, .4, factor)
        isotropic = sphere_clearance_certificate(curve, self.centre, .75, clearance=.1)
        self.assertEqual(directional["status"], "CERTIFIED")
        self.assertEqual(isotropic["status"], "VIOLATED")
        self.assertGreater(directional["minimum_separating_margin_m"], 0.)

    def test_certification_survives_bounded_random_centre_crosscheck(self):
        curve = c2_bump(self.base, [0, .7, 0])
        factor = np.array([[.25, .07, 0.], [.03, .04, 0.], [0., .02, .05]])
        result = ellipsoid_sphere_certificate(curve, self.centre, .4, factor)
        self.assertEqual(result["status"], "CERTIFIED")
        rng = np.random.default_rng(2311)
        z = rng.normal(size=(128, 3))
        z /= np.linalg.norm(z, axis=1)[:, None]
        z *= rng.random((128, 1)) ** (1 / 3)
        points = curve.evaluate(np.linspace(0, 1, 1001))
        for actual_centre in self.centre + z @ factor.T:
            # This diagnostic sampling checks the certificate; it is not its proof.
            self.assertGreater(float(np.min(np.linalg.norm(points - actual_centre, axis=1))), .55)
        for leaf in result["leaf_certificates"]:
            normal = np.array(leaf["normal"])
            self.assertAlmostEqual(np.linalg.norm(normal), 1., places=13)
            self.assertLessEqual(float(np.max((z @ factor.T) @ normal)),
                                 leaf["uncertainty_support_m"] + 1e-13)

    def test_violation_has_an_explicit_realizable_uncertainty_witness(self):
        curve = c2_bump(self.base, [0, .65, 0])
        factor = .3 * np.eye(3)
        result = ellipsoid_sphere_certificate(curve, self.centre, .4, factor)
        self.assertEqual(result["status"], "VIOLATED")
        witness = result["violation_witness"]
        z = np.array(witness["uncertainty_coordinates"])
        self.assertLessEqual(np.linalg.norm(z), 1. + 1e-13)
        np.testing.assert_allclose(self.centre + factor @ z, witness["obstacle_centre"], atol=1e-14)
        np.testing.assert_allclose(curve.evaluate(witness["u"]), witness["position"], atol=2e-12)
        self.assertLess(witness["distance_to_centre"], .55)

    def test_touch_and_depth_exhaustion_are_unresolved(self):
        tangent = bezier_segment([[-1., 1.], [1., 1.]])
        result = ellipsoid_sphere_certificate(tangent, [0, 0], 1., np.zeros((2, 2)),
                                               radius_increment=0., clearance=0., max_depth=5)
        self.assertEqual(result["status"], "UNRESOLVED")
        self.assertIsNone(result["violation_witness"])
        arch = c2_bump(self.base, [0, .65, 0])
        coarse = ellipsoid_sphere_certificate(arch, self.centre, .4, np.diag([.3, .05, .05]), max_depth=0)
        self.assertEqual(coarse["status"], "UNRESOLVED")
        self.assertIsNone(coarse["violation_witness"])

    def test_rank_deficient_rectangular_factor_and_invalid_inputs(self):
        curve = c2_bump(self.base, [0, .7, 0])
        result = ellipsoid_sphere_certificate(curve, self.centre, .4, [[.3], [0.], [0.]])
        self.assertEqual(result["status"], "CERTIFIED")
        for factor in (np.eye(2), np.full((3, 3), np.nan), [[True], [False], [False]], []):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                ellipsoid_sphere_certificate(curve, self.centre, .4, factor)
        for name, value in (("radius_increment", -.1), ("clearance", True),
                            ("numerical_margin", np.inf), ("max_depth", 21), ("max_depth", 2.5)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                ellipsoid_sphere_certificate(curve, self.centre, .4, np.eye(3), **{name: value})


if __name__ == "__main__":
    unittest.main()
