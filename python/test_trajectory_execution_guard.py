import unittest
import numpy as np
from trajectory_execution_guard import bbox_support_offsets, projected_bbox_support_offsets, boundary_command, command_window


class ExecutionGuardTests(unittest.TestCase):
    def boundary(self, **changes):
        args=dict(position=[-5.,0.,-3.],desired_xy=[0.,.15],previous_xy=[0.,0.],
            low=[-5.45,-1.15,-3.45],high=[-2.55,.45,-2.55],speed_cap=.15,
            ramp_acceleration=.15,ramp_dt=.1,horizon_s=.5,reserve_m=.02)
        return boundary_command(**(args|changes))

    def test_observed_box_corners_use_axial_depth_not_radial_range(self):
        corners=bbox_support_offsets([310,190,100,100],[360,240],[640,480],4.)
        np.testing.assert_allclose(corners,[[0,-.625,-.625],[0,.625,-.625],[0,-.625,.625],[0,.625,.625]])

    def test_missing_or_outside_observed_box_rejected(self):
        for box in ([600,190,100,100],[310,190,0,100],[310,190,float('nan'),100]):
            with self.subTest(box=box),self.assertRaises(ValueError):
                bbox_support_offsets(box,[360,240],[640,480],4.)

    def test_off_centre_surface_anchor_reprojects_measured_box_without_translation(self):
        depth, focal = 2., 320.
        point = np.array([depth, (517.-319.5)*depth/focal, (233.-239.5)*depth/focal])
        result = projected_bbox_support_offsets([401,145,183,185], [640,480], point)
        np.testing.assert_allclose(result['anchor_pixel_xy'], [517,233])
        supports = point + np.asarray(result['offsets_ned_m'])
        pixels = np.array([319.5,239.5]) + focal*supports[:,1:]/supports[:,0,None]
        np.testing.assert_allclose(pixels, [[401,145],[584,145],[401,330],[584,330]])
        # Using the detection centroid would shift the entire support rectangle.
        old = point + bbox_support_offsets([401,145,183,185],[491.967,240.739],[640,480],depth)
        old_pixels = np.array([319.5,239.5]) + focal*old[:,1:]/old[:,0,None]
        self.assertGreater(np.max(np.abs(old_pixels-pixels)), 25.)

    def test_invalid_or_unobservable_surface_anchor_is_rejected(self):
        for point in ([0,0,0],[-1,0,0],[2,10,0],[2,float('nan'),0]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                projected_bbox_support_offsets([401,145,183,185],[640,480],point)

    def test_outward_command_stops_inside_reserve_and_inward_remains_allowed(self):
        stop=self.boundary(position=[-5,.435,-3])
        self.assertTrue(stop['accepted'])
        np.testing.assert_allclose(stop['command_velocity_xy_m_s'],[0,0])
        inward=self.boundary(position=[-5,.435,-3],desired_xy=[0,-.15])
        self.assertTrue(inward['accepted'])
        self.assertLess(inward['command_velocity_xy_m_s'][1],0)

    def test_boundary_and_ramp_incompatible_is_rejected_without_forcing_zero(self):
        decision=self.boundary(position=[-5,.42,-3],previous_xy=[0,.05],ramp_dt=.05)
        self.assertFalse(decision['accepted'])
        self.assertIsNone(decision['command_velocity_xy_m_s'])
        self.assertEqual(decision['previous_velocity_xy_m_s'],[0,.05])

    def test_longer_delay_model_shrinks_outward_velocity_limit(self):
        a=self.boundary(position=[-5,.35,-3],horizon_s=.2,ramp_dt=1.)
        b=self.boundary(position=[-5,.35,-3],horizon_s=.5,ramp_dt=1.)
        self.assertLess(b['upper_velocity_xy_m_s'][1],a['upper_velocity_xy_m_s'][1])
        v=b['command_velocity_xy_m_s'][1]
        self.assertLessEqual(.35+.5*v+v*v/(2*.15)+.02,.45+1e-10)

    def test_vector_norm_and_command_delta_budget(self):
        decision=self.boundary(desired_xy=[.15,.15],previous_xy=[.1,.1],ramp_dt=.05)
        self.assertTrue(decision['accepted'])
        v=np.array(decision['command_velocity_xy_m_s'])
        self.assertLessEqual(np.linalg.norm(v),.15+1e-10)
        self.assertLessEqual(np.linalg.norm(v-[.1,.1]),.15*.05+1e-10)

    def test_actual_outside_inner_workspace_not_repaired_by_clipping(self):
        with self.assertRaisesRegex(ValueError,'outside protected workspace'):
            self.boundary(position=[-5,.453156,-3])

    def test_window_trims_command_without_extending_original_freshness_budget(self):
        d=command_window(.4,.5,.1,.03,.06)
        self.assertTrue(d['accepted'])
        self.assertAlmostEqual(d['duration_s'],.04)
        self.assertLessEqual(.4+d['duration_s']+.06,.5+1e-10)

    def test_expiring_or_expired_observation_requests_reobservation(self):
        for age in (.43,.5,.998):
            d=command_window(age,.5,.05,.03,.06)
            self.assertFalse(d['accepted'])
            self.assertEqual(d['duration_s'],0)

    def test_invalid_freshness_age_is_not_recoded_zero(self):
        for age in (-.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):command_window(age,.5,.05,.03,.06)
