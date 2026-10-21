import copy,json,unittest
from pathlib import Path
from t1_obstacle_scene import ROOT,load_scene,validate_scene,route,segment_clear,intersects,command_guard,compile_task

class SceneTests(unittest.TestCase):
    def setUp(self):self.scene=load_scene(ROOT/'configs/t1_building_scene_v1.json')
    def test_default_direct_path_blocked_but_every_detour_segment_clear(self):
        r=route(self.scene);self.assertTrue(r['direct_path_blocked']);self.assertGreater(len(r['waypoints_ned_m']),2)
        self.assertTrue(all(segment_clear(a,b,self.scene) for a,b in zip(r['waypoints_ned_m'],r['waypoints_ned_m'][1:])))
        self.assertFalse(r['flight_authorized'])
    def test_grazing_boundary_is_blocked(self):self.assertTrue(intersects([0,1,0],[3,1,0],[1,1,-1],[2,2,1]))
    def test_empty_or_duplicate_building_names_rejected(self):
        s=copy.deepcopy(self.scene);s['buildings'].append(copy.deepcopy(s['buildings'][0]))
        with self.assertRaises(ValueError):validate_scene(s)
    def test_goal_inside_clearance_is_rejected(self):
        s=copy.deepcopy(self.scene);s['observation_goal_ned_m']=[1.9,-1.9,-3.]
        with self.assertRaisesRegex(ValueError,'GOAL_BLOCKED'):validate_scene(s)
    def test_takeoff_column_rejected(self):
        s=copy.deepcopy(self.scene);s['buildings'][0]['center_ned_m']=[0,-3,-1.6]
        with self.assertRaisesRegex(ValueError,'COLUMN_BLOCKED'):validate_scene(s)
    def test_command_envelope_crossing_clearance_rejected(self):
        self.assertFalse(command_guard([1.145,-1.9,-3.],[.03,0],self.scene)['allowed'])
        self.assertTrue(command_guard([0,-3,-3],[.03,0],self.scene)['allowed'])
    def test_nonfinite_position_rejected(self):
        with self.assertRaises(ValueError):command_guard([float('nan'),0,-3],[0,0],self.scene)
    def test_instruction_preview_has_no_flight_authority(self):
        t=compile_task('靠近带白色横条的红色气球，保持3米高度，避开建筑物',self.scene)
        self.assertFalse(t['flight_authorized']);self.assertTrue(t['avoid_buildings']);self.assertEqual(t['speed_cap_m_s'],.03)
    def test_unknown_height_and_bypass_instruction_rejected(self):
        for s in ('靠近白条气球，保持5米高度','靠近白条气球，关闭避障'):
            with self.assertRaises(ValueError):compile_task(s,self.scene)
    def test_unknown_fields_are_rejected(self):
        s=copy.deepcopy(self.scene);s['shell']='calc'
        with self.assertRaises(ValueError):validate_scene(s)

if __name__=='__main__':unittest.main()
