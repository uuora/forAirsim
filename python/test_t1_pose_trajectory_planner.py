import unittest
import numpy as np
from dataclasses import replace
from t1_pose_geometry import SceneCameraModel, CameraPose, bbox_plane_corners_world
from t1_pose_trajectory_planner import PosePlannerConfig, plan_pose_trajectory, preview_command


class PosePlannerTests(unittest.TestCase):
    def setUp(self):
        self.camera = SceneCameraModel(640,480,320.,320.,320.,240.)
        self.pose = CameraPose([1.,0.,-.05],[0.,0.,0.,1.],[0.,-3.,-1.])
        self.position = [0.,-3.,-1.]
        self.config = replace(PosePlannerConfig(),fixed_world_z_m=-1.,
                              position_min_ned_m=(-.3,-3.3,-1.15),
                              position_max_ned_m=(.3,-2.7,-.85))
        self.surface = {"status":"SURFACE_GEOMETRY_ACCEPTED",
            "rgb_reference_world_ned_m":[5.,-2.5,-1.05],
            "bbox_plane_corners_world_ned_m":bbox_plane_corners_world(
                [315,205,90,70],4.,self.pose,self.camera).tolist()}

    def plan(self,**extra):
        args=dict(surface=self.surface,camera=self.camera,camera_pose=self.pose,
            position_ned_m=self.position,velocity_ned_m_s=[0.,0.,0.],
            acceleration_ned_m_s2=[0.,0.,0.],config=self.config)
        args.update(extra)
        return plan_pose_trajectory(**args)

    def test_toward_off_center_bearing_and_bounded_samples(self):
        plan=self.plan()
        self.assertEqual(plan['status'],'POSE_TRAJECTORY_SELECTED')
        self.assertGreater(plan['trajectory']['endpoint_ned_m'][1],self.position[1])
        self.assertLessEqual(plan['trajectory']['max_sampled_speed_m_s'],.03+1e-9)
        self.assertEqual(plan['trajectory']['endpoint_ned_m'][2],-1.)
        self.assertFalse(plan['flight_authorized'])
        self.assertGreater(preview_command(plan,.4)[1],0.)

    def test_over_speed_initial_state_rejects(self):
        plan=self.plan(velocity_ned_m_s=[.04,0.,0.])
        self.assertEqual(plan['status'],'NO_FEASIBLE_CANDIDATE')
        self.assertEqual(preview_command(plan),[0.,0.])

    def test_unaccepted_surface_and_height_reject(self):
        self.assertEqual(self.plan(surface={})['reason'],'CURRENT_SURFACE_REQUIRED')
        self.assertEqual(self.plan(position_ned_m=[0.,-3.,-.8])['reason'],'FIXED_HEIGHT_NOT_SETTLED')

    def test_rotated_camera_changes_world_direction(self):
        q=[0.,0.,np.sin(np.pi/4),np.cos(np.pi/4)]
        pose=CameraPose([0.,0.,0.],q,[0.,-3.,-1.])
        ref=pose.camera_to_world([4.,.5,0.]).tolist()
        surface=dict(self.surface,rgb_reference_world_ned_m=ref,
            bbox_plane_corners_world_ned_m=bbox_plane_corners_world([315,205,90,70],4.,pose,self.camera).tolist())
        plan=self.plan(camera_pose=pose,surface=surface)
        self.assertEqual(plan['status'],'POSE_TRAJECTORY_SELECTED')
        self.assertLess(plan['trajectory']['endpoint_ned_m'][0],0.)

    def test_workspace_blocks_positive_y_candidates(self):
        config=replace(self.config,position_max_ned_m=(.3,-3.,-.85))
        plan=self.plan(config=config)
        self.assertTrue(any('SAMPLED_WORKSPACE' in r['rejection_reasons'] for r in plan['candidates']))
        self.assertLessEqual(plan['trajectory']['endpoint_ned_m'][1],-3.)

if __name__=='__main__':unittest.main()
