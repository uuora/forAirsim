"""Recovery guards under delayed evidence and fixed-height bias."""
import unittest
from unittest.mock import MagicMock,patch
import numpy as np
import test_fixed_map_perception_execution as fixtures


class RecoveryV18Tests(unittest.TestCase):
    def owner(self):
        helpers=fixtures.ExecutionTests();e=helpers.stationary_owner()
        e.service.cfg.update(stationary_height_integral_v18=True,recovery_plane_gate_v18=True,plan_fresh_handoff_timeout_s=2.)
        e.resume_required=True;e.recovery_plane_ready=False;e.recovery_plane_since=None;e.recovery_plane_deadline=None
        return e
    def test_integral_is_bounded_and_does_not_follow_old_target_xy(self):
        e=self.owner();e.p[2]=3.06;e.v=np.zeros(3);e.active_plan=dict(task=dict(goal_xyz_m=[38,8,3.1]))
        for tick in range(80):
            e.state['state_timestamp_ns']+=100000000
            with patch('fixed_map_perception_execution.time.monotonic',return_value=10.+tick*.1):e.stationary_recovery_hold()
            self.assertLessEqual(np.linalg.norm(e.vcmd),.25000001);self.assertEqual(e.vcmd[0],0.);self.assertEqual(e.vcmd[1],0.)
        self.assertAlmostEqual(e.stationary_height_integral,.15)
        self.assertGreater(e.vcmd[2],.7*.04)
        e.stage='RETURN';e.stationary_recovery_hold();self.assertEqual(e.stationary_height_integral,0.)
    def test_plane_gate_requires_three_sim_seconds_and_original_height(self):
        e=self.owner();e.p[2]=3.066
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):self.assertFalse(e.recovery_plane_gate())
        e.p[2]=3.095;e.state['state_timestamp_ns']=1000000000
        with patch('fixed_map_perception_execution.time.monotonic',return_value=11.):self.assertFalse(e.recovery_plane_gate())
        e.state['state_timestamp_ns']=3900000000
        with patch('fixed_map_perception_execution.time.monotonic',return_value=13.9):self.assertFalse(e.recovery_plane_gate())
        e.state['state_timestamp_ns']=4100000000
        with patch('fixed_map_perception_execution.time.monotonic',return_value=14.1):self.assertTrue(e.recovery_plane_gate())
        self.assertEqual(e.recovery_plane_deadline,35.);self.assertEqual(e.service.cfg['plan_fresh_handoff_timeout_s'],2.)
    def test_plane_excursion_resets_dwell_and_deadline_is_not_extended(self):
        e=self.owner();e.p[2]=3.095;e.state['state_timestamp_ns']=1000000000
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):e.recovery_plane_gate()
        e.p[2]=3.066;e.state['state_timestamp_ns']=4000000000
        with patch('fixed_map_perception_execution.time.monotonic',return_value=13.):self.assertFalse(e.recovery_plane_gate())
        self.assertIsNone(e.recovery_plane_since)
        with patch('fixed_map_perception_execution.time.monotonic',return_value=35.):self.assertFalse(e.recovery_plane_gate())
        self.assertEqual(e.preview_result['reason'],'RECOVERY_PLANE_STABILITY_DEADLINE');self.assertFalse(e.resume_required);e.c.moveByVelocityAsync.assert_not_called()
    def test_fast_motion_cannot_pass_plane_gate(self):
        e=self.owner();e.p[2]=3.1;e.state['speed_m_s']=.13
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.):self.assertFalse(e.recovery_plane_gate())
        self.assertIsNone(e.recovery_plane_since)
    def test_new_recovery_plan_cannot_use_expired_source(self):
        e=self.owner();e.stage='ABORTED';e.pending_plan=None;e.baseline_mode=False;e.active_plan=dict(estimate=fixtures.row()['position_estimate']);e.service.latest=fixtures.row();e.service.cfg.update(fixtures.CFG);e.planner=MagicMock()
        with patch('fixed_map_perception_execution.time.monotonic',return_value=10.6):
            with self.assertRaisesRegex(ValueError,'EXPIRED'):e.request_plan(True,'recovery')
        e.planner.submit.assert_not_called()


if __name__=='__main__':unittest.main()
