"""Analytic and diagnostic checks for the independent time-indexed tracker."""
from dataclasses import replace
import json
import math
import unittest

import numpy as np

from trajectory_research.curves import PolynomialSegment
from trajectory_research.parameterizations import curve_from_parameters, default_parameters
from trajectory_research.timed_tracking import TrackingCondition, run_timed_tracking


class TimedTrackingTests(unittest.TestCase):
    def test_constant_velocity_is_tracked_at_synchronized_times(self):
        duration=1.3
        start=np.array([.1,-.2,5.])
        speed=np.array([1.,.25,-.1])
        segment=PolynomialSegment(np.array([start,duration*speed]),duration)
        result=run_timed_tracking(segment)
        expected=start+np.array(result['times_s'])[:,None]*speed
        np.testing.assert_allclose(result['actual_position_m'],expected,atol=2e-14,rtol=0)
        np.testing.assert_allclose(result['actual_velocity_mps'],np.tile(speed,(len(expected),1)),atol=2e-14,rtol=0)
        self.assertLess(result['metrics']['position_rmse_m'],2e-14)
        self.assertEqual(result['metrics']['speed_clipped_time_fraction'],0.)
        self.assertEqual(result['metrics']['command_slew_clipped_time_fraction'],0.)
        self.assertNotIn('feasible',result)
        json.dumps(result,allow_nan=False)

    def test_one_step_matches_independently_computed_lag_solution(self):
        duration=.2
        lag=.3
        # Stationary q, initial offset -.5m and kp=2 request a +1m/s command.
        segment=PolynomialSegment([[0.,0.,0.]],duration)
        condition=TrackingCondition(velocity_lag_s=lag,timestep_s=duration,position_gain=2.,
                                    max_command_slew_mps2=100.)
        result=run_timed_tracking(segment,condition,initial_position_offset_m=[-.5,0.,0.])
        decay=math.exp(-duration/lag)
        expected_velocity=1.-decay
        expected_position=-.5+duration-lag*(1.-decay)
        self.assertEqual(result['intervals'],1)
        self.assertAlmostEqual(result['velocity_command_mps'][0][0],1.,places=15)
        self.assertAlmostEqual(result['actual_velocity_mps'][-1][0],expected_velocity,places=15)
        self.assertAlmostEqual(result['actual_position_m'][-1][0],expected_position,places=15)
        expected_rmse=math.sqrt((.5**2+expected_position**2)/2.)
        self.assertAlmostEqual(result['metrics']['position_rmse_m'],expected_rmse,places=15)
        self.assertEqual(result['metrics']['peak_position_error_time_s'],0.)

    def test_nondivisible_horizon_ends_exactly_without_extrapolation(self):
        segment=PolynomialSegment([[0.,0.,0.],[1.,0.,0.]],1.)
        result=run_timed_tracking(segment,TrackingCondition(timestep_s=.3))
        self.assertEqual(result['times_s'],[0.,.25,.5,.75,1.])
        self.assertEqual(result['intervals'],4)
        self.assertLessEqual(max(result['command_intervals_s']),.3)
        self.assertEqual(len(result['reference_position_m']),5)
        self.assertEqual(len(result['velocity_command_mps']),4)
        self.assertEqual(result['command_times_s'],result['times_s'][:-1])

    def test_speed_and_slew_clipping_have_distinct_observed_diagnostics(self):
        segment=PolynomialSegment([[0.,0.,0.]],1.)
        condition=TrackingCondition(timestep_s=.2,max_command_speed_mps=1.,max_command_slew_mps2=.5)
        result=run_timed_tracking(segment,condition,initial_position_offset_m=[-10.,0.,0.])
        self.assertTrue(all(result['speed_clipped']))
        self.assertTrue(all(result['command_slew_clipped']))
        self.assertEqual(result['metrics']['speed_clipped_time_fraction'],1.)
        self.assertEqual(result['metrics']['command_slew_clipped_time_fraction'],1.)
        self.assertGreater(result['speed_projection_difference_mps'][0],0.)
        self.assertGreater(result['command_slew_projection_difference_mps'][0],0.)
        commands=np.array(result['velocity_command_mps'])
        previous=np.vstack(([0.,0.,0.],commands[:-1]))
        actual_slew=np.linalg.norm(commands-previous,axis=1)/np.array(result['command_intervals_s'])
        np.testing.assert_allclose(actual_slew,result['command_slew_mps2'],atol=1e-15,rtol=0)
        self.assertLessEqual(np.linalg.norm(commands,axis=1).max(),1.+1e-14)
        self.assertLessEqual(actual_slew.max(),.5+1e-14)

    def test_nonzero_terminal_reference_velocity_is_not_replaced_by_hold(self):
        segment=curve_from_parameters(default_parameters('single_bump'),family='single_bump')
        result=run_timed_tracking(segment)
        self.assertEqual(result['times_s'][-1],segment.duration_s)
        np.testing.assert_allclose(result['reference_velocity_mps'][-1],[1.,0.,0.],atol=2e-13,rtol=0)
        self.assertGreater(np.linalg.norm(result['actual_velocity_mps'][-1]),.1)
        self.assertEqual(result['command_times_s'][-1]+result['command_intervals_s'][-1],segment.duration_s)
        self.assertIn('not production RollingController',result['scope'])
        self.assertNotIn('collision_passed',result)

    def test_rmse_peak_time_and_terminal_errors_recompute_from_trace(self):
        segment=curve_from_parameters(default_parameters('paired_bump'),family='paired_bump')
        result=run_timed_tracking(segment,TrackingCondition(timestep_s=.07))
        t=np.array(result['times_s'])
        error=np.array(result['reference_position_m'])-np.array(result['actual_position_m'])
        squared=np.sum(error**2,axis=1)
        independent_rmse=np.sqrt(np.sum((squared[1:]+squared[:-1])*.5*np.diff(t))/t[-1])
        self.assertAlmostEqual(result['metrics']['position_rmse_m'],independent_rmse,places=15)
        self.assertEqual(result['metrics']['peak_position_error_time_s'],t[np.argmax(squared)])
        self.assertAlmostEqual(result['metrics']['terminal_position_error_m'],np.linalg.norm(error[-1]),places=15)
        velocity_error=np.array(result['reference_velocity_mps'][-1])-np.array(result['actual_velocity_mps'][-1])
        self.assertAlmostEqual(result['metrics']['terminal_velocity_error_mps'],np.linalg.norm(velocity_error),places=15)

    def test_invalid_model_parameters_or_initial_states_are_rejected(self):
        default=TrackingCondition()
        for name in ('velocity_lag_s','timestep_s','max_command_speed_mps','max_command_slew_mps2'):
            for bad in (0.,-1.,np.nan,np.inf,True,'1',1+0j):
                with self.subTest(name=name,bad=bad),self.assertRaises(ValueError):
                    replace(default,**{name:bad})
        for bad in (-1.,np.nan,np.inf,True,'1',1+0j):
            with self.subTest(gain=bad),self.assertRaises(ValueError):
                TrackingCondition(position_gain=bad)
        self.assertEqual(TrackingCondition(position_gain=0.).position_gain,0.)
        segment=PolynomialSegment([[0.,0.,0.]],1.)
        for bad in (None,[0.,0.],[0.,0.,np.nan],[False,False,True],['0','0','0'],[0j,0j,0j]):
            with self.subTest(offset=bad),self.assertRaises(ValueError):
                run_timed_tracking(segment,initial_position_offset_m=bad)
        for bad in (None,object(),PolynomialSegment([[0.,0.]],1.)):
            with self.subTest(segment=bad),self.assertRaises(ValueError):
                run_timed_tracking(bad)
        with self.assertRaises(ValueError):
            run_timed_tracking(segment,condition={})
        with self.assertRaises(ValueError):
            run_timed_tracking(segment,TrackingCondition(timestep_s=1e-8))
        with self.assertRaises(ValueError):
            run_timed_tracking(PolynomialSegment([[0.,0.,0.],[5.,0.,0.]],1.))


if __name__=='__main__':
    unittest.main()
