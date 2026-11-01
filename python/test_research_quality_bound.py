"""Closed-form quality is cross-checked against the existing curve oracle."""
import unittest

import numpy as np

from trajectory_research.curve_task import jerk_energy
from trajectory_research.curves import PolynomialSegment
from trajectory_research.parameterizations import curve_from_parameters, parameter_bounds
from trajectory_research.perturbation_task import evaluate_parameters, shape_rms
from trajectory_research.quality_bound import NUMERICAL_CUSHION, quality_lower_bound, quality_terms


def oracle_terms(vector, family):
    segment = curve_from_parameters(vector, family=family)
    # Match the task's uniform stored degree before independent integration.
    if segment.degree < 7:
        segment = PolynomialSegment(np.pad(segment.coefficients,
            ((0, 7-segment.degree), (0, 0))), segment.duration_s)
    energy = jerk_energy(segment)
    rms = shape_rms(vector, family=family)
    t = segment.duration_s
    q = .9*t/(t+5.) + .05*energy/(energy+30.) + .05*rms/(rms+1.)
    return energy, rms, q


class QualityBoundTests(unittest.TestCase):
    def assert_matches_independent_curve(self, vector, family):
        expected_energy, expected_rms, expected_quality = oracle_terms(vector, family)
        actual = quality_terms(vector, family=family)
        # Power-basis integration includes floating cancellation; the closed
        # form avoids it. These tolerances compare implementations, not safety.
        self.assertAlmostEqual(actual['jerk_energy_m2_s5'], expected_energy,
                               delta=2e-8 * max(1., expected_energy))
        self.assertAlmostEqual(actual['deformation_rms_m'], expected_rms, places=14)
        self.assertAlmostEqual(actual['quality_term'], expected_quality, delta=2e-11)
        self.assertLessEqual(quality_lower_bound(vector, family=family), expected_quality)

    def test_thousand_random_points_per_family_match_curve_integrals(self):
        rng = np.random.default_rng(20261029)
        for family in ('single_bump', 'paired_bump'):
            lower, upper = parameter_bounds(family)
            for _ in range(1000):
                vector = rng.uniform(lower, upper)
                self.assert_matches_independent_curve(vector, family)

    def test_canonical_box_corners_and_duration_endpoints(self):
        for family in ('single_bump', 'paired_bump'):
            lower, upper = parameter_bounds(family)
            for duration in (3., 7.):
                for shape in (lower[:-1], upper[:-1], np.zeros(len(lower)-1)):
                    vector = np.r_[shape, duration]
                    self.assert_matches_independent_curve(vector, family)

    def test_zero_asymmetric_mode_has_identical_closed_form(self):
        for displacement in ([0.,0.,0.], [.8,-1.4,.8], [-.3,.7,.2]):
            for duration in (3.,5.5,7.):
                a = [*displacement,duration]
                b = [*displacement,0.,0.,0.,duration]
                self.assertEqual(quality_terms(a,family='single_bump'),
                                 quality_terms(b,family='paired_bump'))
                self.assertEqual(quality_lower_bound(a,family='single_bump'),
                                 quality_lower_bound(b,family='paired_bump'))

    def test_bound_against_complete_objective_including_failures(self):
        rng = np.random.default_rng(7193)
        passed_flags = []
        for family in ('single_bump', 'paired_bump'):
            lower, upper = parameter_bounds(family)
            vectors = [np.r_[np.zeros(len(lower)-1),3.],
                       np.r_[lower[:-1],7.]]
            vectors.extend(rng.uniform(lower,upper) for _ in range(5))
            for vector in vectors:
                result = evaluate_parameters(vector,family=family,layout='central')
                bound = quality_lower_bound(vector,family=family)
                self.assertLessEqual(bound,result['score'])
                self.assertLessEqual(bound,result['score_details']['quality_term'])
                passed_flags.append(result['feasible'])
        self.assertIn(False,passed_flags)

    def test_numerical_cushion_is_explicit(self):
        vector = [0.,1.15,0.,5.5]
        q = quality_terms(vector,family='single_bump')['quality_term']
        self.assertEqual(quality_lower_bound(vector,family='single_bump'),q-NUMERICAL_CUSHION)
        self.assertGreater(NUMERICAL_CUSHION,0.)

    def test_bad_parameter_requests_fail_visibly(self):
        bad = ([0.,0.,0.], [0.,0.,0.,2.99], [0.,0.,0.,7.01],
               [.81,0.,0.,5.], [0.,0.,0.,np.nan], [0.,0.,0.,np.inf],
               ['0','0','0','5'], [False,False,False,True],
               np.zeros((4,1)), [0j,0j,0j,5+0j])
        for vector in bad:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                quality_lower_bound(vector,family='single_bump')
        for family in ('other', '', None):
            with self.subTest(family=family), self.assertRaises(ValueError):
                quality_lower_bound([0.,0.,0.,5.],family=family)

    def test_custom_boundaries_are_not_an_accepted_argument(self):
        with self.assertRaises(TypeError):
            quality_lower_bound([0.,0.,0.,5.],family='single_bump',boundary=object())


if __name__ == '__main__':
    unittest.main()
