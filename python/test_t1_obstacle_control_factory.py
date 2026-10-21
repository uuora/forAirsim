"""Exercise the child-owned command boundary, including feedback rejection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from t1_obstacle_control_factory import make_obstacle_airsim
from t1_obstacle_scene import ROOT,load_scene

class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.scene=load_scene(ROOT/'configs/t1_building_scene_v1.json');self.position=[0,-3,-3]
        self.commands=[];self.events=[]
        self.flight=SimpleNamespace(state=lambda:dict(position_world_ned_m=self.position),
            issue=lambda xy,duration,purpose,**kw:self.commands.append(dict(xy=xy,purpose=purpose)),
            protection=lambda state:[],client=SimpleNamespace(simGetCollisionInfo=lambda **kw:SimpleNamespace(has_collided=False,object_name='')))
    def factory(self):
        with patch('t1_obstacle_control_factory.make_airsim',return_value=(self.flight,'policy')):
            return make_obstacle_airsim(dict(obstacle_scene=self.scene),self.events.append)[0]
    def test_safe_target_command_is_forwarded_once(self):
        self.factory().issue([.03,0],.12,'CURRENT_TARGET_TRAJECTORY')
        self.assertEqual(len(self.commands),1);self.assertTrue(self.events[-1]['allowed'])
    def test_target_crossing_building_emits_zero_and_rejects(self):
        flight=self.factory();self.position=[1.145,-1.9,-3]
        with self.assertRaisesRegex(RuntimeError,'REJECTED'):flight.issue([.03,0],.12,'CURRENT_TARGET_TRAJECTORY')
        self.assertEqual(self.commands,[dict(xy=[0.,0.],purpose='STATIC_OBSTACLE_REJECTION_ZERO')])
    def test_feedback_command_has_same_obstacle_guard(self):
        flight=self.factory();self.position=[1.145,-1.9,-3]
        with self.assertRaisesRegex(RuntimeError,'REJECTED'):flight.issue([.03,0],.12,'OWN_STATE_FEEDBACK_BRAKE')
        self.assertFalse(self.events[-1]['allowed']);self.assertEqual(self.commands[0]['xy'],[0.,0.])
    def test_position_inside_clearance_is_protection_failure(self):
        flight=self.factory();self.position=[1.9,-1.9,-3]
        self.assertIn('STATIC_BUILDING_CLEARANCE',flight.protection(dict(position_world_ned_m=self.position)))
    def test_building_collision_record_is_protection_failure(self):
        flight=self.factory();self.flight.client.simGetCollisionInfo=lambda **kw:SimpleNamespace(has_collided=True,object_name='T1Building_A')
        self.assertIn('BUILDING_COLLISION',flight.protection(dict(position_world_ned_m=self.position)))

if __name__=='__main__':unittest.main()
