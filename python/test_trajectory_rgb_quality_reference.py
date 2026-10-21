import unittest
import numpy as np
from perception_trajectory_planner import PlanningParams,plan_trajectory
from trajectory_rgb_quality_reference import rgb_quality_reference,replay_centering_ranking


class RGBQualityReferenceTests(unittest.TestCase):
    def test_reference_projects_to_rgb_centroid_while_surface_is_preserved(self):
        surface=[4.,.8,-.1];camera=[0,0,0]
        result=rgb_quality_reference(surface,camera,[319.5,239.5],[640,480])
        np.testing.assert_allclose(result['reference_world_ned_m'],[4,0,0],atol=1e-12)
        np.testing.assert_allclose(np.array(result['reference_world_ned_m'])-np.array(result['offset_from_surface_ned_m']),surface)
        self.assertEqual(surface,[4.,.8,-.1])
        self.assertIn('not_physical_balloon_center',result['semantics'])

    def test_invalid_centroid_and_plane_cannot_be_used(self):
        for surface,pixel in (([0,0,0],[320,240]),([4,0,0],[-1,240]),([4,0,0],[640,240]),([4,0,0],[float('nan'),240])):
            with self.subTest(surface=surface,pixel=pixel),self.assertRaises(ValueError):
                rgb_quality_reference(surface,[0,0,0],pixel,[640,480])

    def test_replay_preserves_feasibility_and_efficiency_ranking(self):
        for mode in ('efficiency','perception'):
            plan=plan_trajectory(position=[-5,-.5,-3],velocity=[0,0,0],acceleration=[0,0,0],
                target_position=[0,.4,-3],target_velocity=[0,0,0],candidate_endpoints=[[-4.8,-.5,-3],[-4.8,-.38,-3]],
                params=PlanningParams(durations=(3,),max_speed_mps=.15,max_acceleration_mps2=.3),mode=mode)
            replay=replay_centering_ranking(plan,[0,-.2,0])
            self.assertTrue(replay['feasible_set_unchanged'])
            self.assertEqual(replay['feasible_candidate_count'],sum(c['feasible'] for c in plan['candidates']))
            if mode=='efficiency':
                self.assertEqual(replay['proposed_selected_index'],replay['original_selected_index'])


if __name__=='__main__':unittest.main()
