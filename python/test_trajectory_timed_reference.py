import unittest
import numpy as np
from perception_trajectory_planner import PlanningParams, plan_trajectory
from trajectory_timed_reference import command_ramp_interval, scheduled_quantum, timed_reference


class TimedReferenceTests(unittest.TestCase):
    def plan(self):
        return plan_trajectory(position=[-5,-.5,-3],velocity=[0,0,0],acceleration=[0,0,0],
            target_position=[0,0,-3],target_velocity=[0,0,0],candidate_endpoints=[[-4.8,-.38,-3]],
            params=PlanningParams(durations=(3,),max_speed_mps=.15,max_acceleration_mps2=.3))

    def test_reference_uses_elapsed_time_and_never_repeats_fixed_preview(self):
        p=self.plan()
        a=timed_reference(p,[-5,-.5,-3],100,100.1,None,.8,.15)
        b=timed_reference(p,[-5,-.5,-3],100,100.5,a['reference_time_s'],.8,.15)
        self.assertAlmostEqual(a['reference_time_s'],.1)
        self.assertAlmostEqual(b['reference_time_s'],.5)
        self.assertGreater(b['reference_position_world_ned_m'][0],a['reference_position_world_ned_m'][0])

    def test_clock_rollback_and_extrapolation_rejected(self):
        for now,last in ((99.9,None),(100.1,.2),(103.1,None)):
            with self.subTest(now=now),self.assertRaises(ValueError):
                timed_reference(self.plan(),[-5,-.5,-3],100,now,last,.8,.15)

    def test_zero_elapsed_and_xy_norm_budget(self):
        result=timed_reference(self.plan(),[-8,-8,-3],100,100,None,10,.15)
        self.assertEqual(result['reference_time_s'],0.)
        self.assertLessEqual(np.linalg.norm(result['desired_velocity_xy_m_s']),.15+1e-12)
        self.assertTrue(result['xy_norm_clipped'])

    def slot(self,age):
        return scheduled_quantum(age,.5,.05,.03,.06,.17,.035,.015)

    def test_next_cycle_budget_yields_without_braking_live_frame(self):
        a=self.slot(.14)
        self.assertEqual(a['status'],'EXECUTE')
        self.assertAlmostEqual(a['duration_s'],.05)
        b=self.slot(.25)
        self.assertEqual(b['status'],'YIELD_FOR_NEW_OBSERVATION')
        self.assertGreater(b['live_command_window_s'],.03)

    def test_nearly_expired_frame_requires_brake(self):
        self.assertEqual(self.slot(.49)['status'],'BRAKE')

    def test_invalid_clock_and_scheduler_values_rejected(self):
        for age in (-.1,float('nan'),float('inf'),True):
            with self.subTest(age=age),self.assertRaises(ValueError):
                self.slot(age)
        with self.assertRaises(ValueError):
            timed_reference(self.plan(),[-5,-.5,-3],True,100,None,.8,.15)

    def test_ramp_time_uses_previous_request_not_future_duration(self):
        self.assertAlmostEqual(command_ramp_interval(100.,100.14,.2)['effective_interval_s'],.14)
        self.assertAlmostEqual(command_ramp_interval(100.,101.,.2)['effective_interval_s'],.2)
        self.assertEqual(command_ramp_interval(None,100.,.2)['effective_interval_s'],0.)
        self.assertEqual(command_ramp_interval(100.,100.,.2)['effective_interval_s'],0.)

    def test_ramp_time_rejects_clock_rollback_and_invalid_values(self):
        for previous,current,cap in ((101,100,.2),(True,100,.2),(100,float('nan'),.2),(100,100,0)):
            with self.subTest(previous=previous,current=current),self.assertRaises(ValueError):
                command_ramp_interval(previous,current,cap)


if __name__=='__main__':
    unittest.main()
