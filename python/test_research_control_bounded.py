"""Partial evaluation is elimination evidence, never an imputed objective."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from dual_target_control import trajectory_objective as original
from trajectory_research.control_bounded import (
    default_parameters, evaluate_control_bounded, parameter_bounds, partial_score_lower_bound,
)


def row(time=2., jerk=1., passed=True, name='synthetic'):
    return dict(passed=passed, elapsed_s=time, scenario=name,
        maxima={'command_jerk_mps3':jerk}, control_config={'max_acceleration':4.},
        intervals_s=[.05], yaw_lag_s=.25, legs=[],
        checks={'arrival':passed, 'commands':True})


class ControlBoundedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = original.make_training_cases()

    def test_bound_valid_for_random_mixed_prefixes_and_orders(self):
        rng = np.random.default_rng(739)
        for _ in range(200):
            rows = [row(float(rng.uniform(0,100)), float(rng.uniform(0,300)),
                        bool(rng.integers(0,2))) for _ in range(4)]
            score = original._score_rows(rows)[0]
            order = rng.permutation(4)
            for k in range(5):
                prefix = [rows[i] for i in order[:k]]
                self.assertLessEqual(partial_score_lower_bound(prefix,4), score)

    def test_initial_failure_prunes_without_imputing_remaining_cases(self):
        rows = [row(passed=False), row(), row()]
        with patch.object(original,'run_scenario',side_effect=rows) as runner:
            result = evaluate_control_bounded(default_parameters(), .8, self.cases)
        self.assertEqual(runner.call_count,1)
        self.assertEqual(result['status'],'PRUNED')
        self.assertIsNone(result['score'])
        self.assertIsNone(result['feasible'])
        self.assertEqual(result['completed_cases'],1)
        self.assertEqual(result['total_cases'],3)
        self.assertGreaterEqual(result['lower_bound'],.8)
        self.assertEqual(result['case_observations'][0]['case_index'],0)
        self.assertFalse(result['case_observations'][0]['passed'])
        self.assertEqual(len(result['cases']),1)
        json.dumps(result,allow_nan=False)

    def test_no_failure_yet_does_not_assert_feasibility(self):
        rows = [row(time=100), row(), row()]
        with patch.object(original,'run_scenario',side_effect=rows):
            result = evaluate_control_bounded(default_parameters(),.3,self.cases)
        self.assertEqual(result['status'],'PRUNED')
        self.assertIsNone(result['feasible'])
        self.assertEqual(result['score_details']['observed_failed_cases'],0)

    def test_complete_case_order_restores_original_score_order(self):
        by_name = {case.scenario.name:row(time=2+i,passed=i!=1,name=case.scenario.name)
                   for i,case in enumerate(self.cases)}
        def run(_,scenario,**kwargs):
            return deepcopy(by_name[scenario.name])
        with patch.object(original,'run_scenario',side_effect=run):
            baseline = original.evaluate_parameters(default_parameters(),self.cases)
            result = evaluate_control_bounded(default_parameters(),None,self.cases,
                                             case_order=[2,0,1])
        for key in ('score','feasible','score_details','summary','cases'):
            self.assertEqual(result[key],baseline[key])
        self.assertEqual([r['case_index'] for r in result['case_observations']],[2,0,1])
        self.assertEqual(result['completed_cases'],3)

    def test_high_cutoff_or_single_case_returns_measured_score(self):
        with patch.object(original,'run_scenario',side_effect=[row(),row(),row()]):
            result = evaluate_control_bounded(default_parameters(),10.,self.cases)
        self.assertEqual(result['status'],'EVALUATED')
        self.assertTrue(result['feasible'])
        with patch.object(original,'run_scenario',return_value=row(passed=False)):
            result = evaluate_control_bounded(default_parameters(),0.,self.cases[:1])
        self.assertEqual(result['status'],'EVALUATED')
        self.assertFalse(result['feasible'])

    def test_invalid_requests_fail_before_rollout(self):
        with patch.object(original,'run_scenario') as runner:
            for cutoff in (True,np.nan,np.inf,'1',1+2j):
                with self.subTest(cutoff=cutoff),self.assertRaises(ValueError):
                    evaluate_control_bounded(default_parameters(),cutoff,self.cases)
            for order in ([0,1], [0,0,1], [0,1,3], [False,1,2], [0.,1,2],1):
                with self.subTest(order=order),self.assertRaises(ValueError):
                    evaluate_control_bounded(default_parameters(),cases=self.cases,case_order=order)
            with self.assertRaises(ValueError):
                evaluate_control_bounded(default_parameters(),cases=[])
            runner.assert_not_called()

    def test_invalid_nonnegative_bound_assumptions_fail_visibly(self):
        for value in (-1.,np.nan,np.inf,True):
            with self.subTest(value=value),self.assertRaises(ValueError):
                partial_score_lower_bound([row(time=value)],3)
        with self.assertRaises(ValueError):
            partial_score_lower_bound([row(jerk=-1)],3)
        with self.assertRaises(ValueError):
            partial_score_lower_bound([row()],0)
        with self.assertRaises(ValueError):
            partial_score_lower_bound([row(),row()],1)

    def test_exception_is_not_recast_as_failed_or_pruned_score(self):
        with patch.object(original,'run_scenario',side_effect=RuntimeError('simulation failed')):
            with self.assertRaisesRegex(RuntimeError,'simulation failed'):
                evaluate_control_bounded(default_parameters(),.1,self.cases)

    def test_real_controller_matches_original_objective(self):
        cases = self.cases[:1]
        baseline = original.evaluate_parameters(default_parameters(),cases,dt=.05)
        result = evaluate_control_bounded(default_parameters(),cases=cases,dt=.05)
        for key in ('score','feasible','score_details','summary','cases'):
            self.assertEqual(result[key],baseline[key])
        self.assertIn('not AirSim',result['model'])
        self.assertEqual(result['completed_cases'],1)
        lower,upper = parameter_bounds()
        self.assertTrue(np.all(default_parameters() >= lower))
        self.assertTrue(np.all(default_parameters() <= upper))
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main()
