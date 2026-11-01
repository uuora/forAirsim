"""Independent arithmetic, enclosure and feedback tests for the timed task."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import numpy as np

from trajectory_research.parameterizations import default_parameters, parameter_bounds
from trajectory_research.perturbation_task import Layout, Obstacle
from trajectory_research.timed_tracking import TrackingCondition
from trajectory_research.timed_control_task import (
    DISTANCE_CUSHION_M, _clipping_intervals, actual_nominal_path_checks, control_features,
    evaluate_timed_parameters, timed_quality_lower_bound,
)


def simple_trace(*, cap=2., velocity=2.):
    """A constant-speed two-sample trace whose missed interior crosses x=1."""
    return dict(times_s=[0.,1.], actual_position_m=[[0.,0.,5.],[2.,0.,5.]],
                actual_velocity_mps=[[velocity,0.,0.],[velocity,0.,0.]],
                velocity_command_mps=[[cap,0.,0.]])


class TimedControlTaskTests(unittest.TestCase):
    def test_quality_and_penalty_reconstruct_and_trace_is_explicit(self):
        parameters=default_parameters('single_bump')
        result=evaluate_timed_parameters(parameters, include_trace=True)
        details=result['score_details']
        energy=details['jerk_energy_m2_s5']
        rmse=details['tracking_metrics']['position_rmse_m']
        duration=parameters[-1]
        expected=.7*duration/(duration+5.)+.1*energy/(energy+30.)+.2*rmse/(rmse+.25)
        self.assertEqual(details['quality_term'],expected)
        self.assertLess(expected,1.)
        failed=sum(not value for value in details['checks'].values())
        self.assertEqual(details['failed_checks'],failed)
        self.assertEqual(result['feasible'],failed==0)
        self.assertEqual(result['score'],expected if failed==0 else 1.+failed/len(details['checks'])+expected)
        trace=result['tracking_trace']
        self.assertEqual(trace['times_s'][-1],duration)
        np.testing.assert_allclose(trace['reference_velocity_mps'][-1],[1.,0.,0.],atol=2e-12,rtol=0)
        json.dumps(result,allow_nan=False)
        self.assertNotIn('tracking_trace',evaluate_timed_parameters(parameters))

    def test_geometry_representation_does_not_change_the_same_tracking_curve(self):
        parameters=default_parameters('paired_bump')
        parameters[3:6]=[.2,-.4,.3]
        options=dict(family='paired_bump',condition=TrackingCondition(timestep_s=.037))
        first=evaluate_timed_parameters(parameters,representation='hermite',**options)
        second=evaluate_timed_parameters(parameters,representation='bezier',**options)
        self.assertEqual(first['feasible'],second['feasible'])
        self.assertEqual(first['score_details']['checks'],second['score_details']['checks'])
        self.assertAlmostEqual(first['score'],second['score'],delta=2e-10)
        for name,value in first['score_details']['tracking_metrics'].items():
            self.assertAlmostEqual(value,second['score_details']['tracking_metrics'][name],delta=2e-9)

    def test_cheap_bound_holds_on_declared_random_candidates(self):
        rng=np.random.default_rng(17031)
        for family in ['single_bump','paired_bump']:
            lower,upper=parameter_bounds(family)
            points=[default_parameters(family),lower,upper]
            points.extend(rng.uniform(lower,upper,size=(4,len(lower))))
            for point in points:
                with self.subTest(family=family,parameters=point.tolist()):
                    measured=evaluate_timed_parameters(point,family=family,
                        condition=TrackingCondition(timestep_s=.05))
                    bound=timed_quality_lower_bound(point,family=family)
                    self.assertLessEqual(bound,measured['score_details']['quality_term'])
                    self.assertLessEqual(bound,measured['score'])

    def test_sampling_gap_catches_a_sphere_missed_by_both_samples(self):
        trace=simple_trace()
        condition=TrackingCondition(timestep_s=1.,max_command_speed_mps=2.)
        scene=Layout('missed_interior',(Obstacle((1.,0.,5.),radius=.4),))
        result=actual_nominal_path_checks(trace,layout=scene,condition=condition)
        sphere=result['nominal_spheres'][0]
        # At both endpoints the distance to the sphere surface is .6m, but the
        # constant-speed trace crosses the centre halfway through the interval.
        self.assertAlmostEqual(sphere['min_sampled_surface_clearance_m'],.6,places=14)
        self.assertEqual(result['sampling_gap_margin_m'],1.)
        self.assertAlmostEqual(sphere['surface_clearance_lower_bound_m'],
                               -.4-DISTANCE_CUSHION_M,places=14)
        self.assertFalse(sphere['passed'])
        self.assertTrue(result['checks']['model_velocity_bound'])
        # Independently reconstruct one workspace face enclosure.
        self.assertAlmostEqual(result['workspace_lower_face_clearance_bounds_m'][0],
                               .5-1.-DISTANCE_CUSHION_M,places=14)

    def test_unresolved_speed_assumption_never_mints_clearance_bounds(self):
        trace=simple_trace(velocity=3.)
        result=actual_nominal_path_checks(trace,condition=TrackingCondition(max_command_speed_mps=2.))
        self.assertEqual(result['status'],'UNRESOLVED_SPEED_ASSUMPTION')
        self.assertFalse(result['checks']['model_velocity_bound'])
        self.assertFalse(result['checks']['actual_nominal_workspace'])
        self.assertIsNone(result['workspace_lower_face_clearance_bounds_m'])
        self.assertTrue(all(sphere['surface_clearance_lower_bound_m'] is None
                            and not sphere['passed'] for sphere in result['nominal_spheres']))

    def test_error_gate_has_priority_over_a_good_scalar_quality(self):
        parameters=default_parameters('single_bump')
        baseline=evaluate_timed_parameters(parameters,include_trace=True)
        trace=deepcopy(baseline['tracking_trace'])
        trace['metrics']['max_position_error_m']=.5001
        with patch('trajectory_research.timed_control_task.run_timed_tracking',return_value=trace) as tracker:
            result=evaluate_timed_parameters(parameters)
        tracker.assert_called_once()
        self.assertFalse(result['score_details']['checks']['tracking_peak_error'])
        self.assertFalse(result['feasible'])
        self.assertGreaterEqual(result['score'],1.)
        self.assertLess(result['score_details']['quality_term'],1.)

    def test_control_features_match_raw_metrics_and_cap_only_declared_terms(self):
        result=evaluate_timed_parameters(default_parameters('single_bump'))
        metrics=result['score_details']['tracking_metrics']
        expected=[min(4.,metrics['position_rmse_m']/.25),
                  min(4.,metrics['max_position_error_m']/.5),
                  min(4.,metrics['terminal_velocity_error_mps']/.5),
                  metrics['command_slew_clipped_time_fraction'],metrics['peak_position_error_u']]
        self.assertEqual(result['score_details']['control_features'],expected)
        injected=dict(position_rmse_m=2.,max_position_error_m=3.,terminal_velocity_error_mps=3.,
                      command_slew_clipped_time_fraction=.8,peak_position_error_u=.3)
        self.assertEqual(control_features(injected),[4.,4.,4.,.8,.3])

    def test_clipping_time_intervals_compress_without_adding_evaluations(self):
        synthetic=dict(times_s=[0.,.1,.2,.3,.4,.5],speed_clipped=[True,True,False,False,True])
        self.assertEqual(_clipping_intervals(synthetic,'speed_clipped'),[[0.,.2],[.4,.5]])
        result=evaluate_timed_parameters(default_parameters('single_bump'),include_trace=True)
        trace=result['tracking_trace']
        for key,fraction in [('speed_clip_intervals_s','speed_clipped_time_fraction'),
                             ('command_slew_clip_intervals_s','command_slew_clipped_time_fraction')]:
            spans=result['score_details']['tracking_diagnostics'][key]
            proportion=sum(end-start for start,end in spans)/trace['duration_s']
            self.assertAlmostEqual(proportion,trace['metrics'][fraction],places=12)

    def test_invalid_inputs_and_evaluator_exceptions_remain_visible(self):
        parameters=default_parameters('single_bump')
        with patch('trajectory_research.timed_control_task.run_timed_tracking') as tracker:
            for condition in ({},True,0):
                with self.subTest(condition=condition),self.assertRaises(ValueError):
                    evaluate_timed_parameters(parameters,condition=condition)
            for include in (1,None,'true'):
                with self.subTest(include=include),self.assertRaises(ValueError):
                    evaluate_timed_parameters(parameters,include_trace=include)
            tracker.assert_not_called()
        for invalid in ([0.,0.,3.],[0.,0.,0.,np.nan]):
            with self.assertRaises(ValueError): evaluate_timed_parameters(invalid)
        with patch('trajectory_research.timed_control_task.run_timed_tracking',side_effect=RuntimeError('rollout error')):
            with self.assertRaisesRegex(RuntimeError,'rollout error'):
                evaluate_timed_parameters(parameters)
        bad=simple_trace();bad['times_s']=[0.,0.]
        with self.assertRaises(ValueError): actual_nominal_path_checks(bad)
        bad=simple_trace();bad['condition']={'wrong':'condition'}
        with self.assertRaises(ValueError): actual_nominal_path_checks(bad)
        metrics=dict(position_rmse_m=np.nan,max_position_error_m=0.,terminal_velocity_error_mps=0.,
                     command_slew_clipped_time_fraction=0.,peak_position_error_u=0.)
        with self.assertRaises(ValueError): control_features(metrics)


if __name__=='__main__': unittest.main()
