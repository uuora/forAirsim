import unittest
import numpy as np
from fixed_map_mission import Geometry,Executor
from types import SimpleNamespace
import queue

class SafetyGeometryTests(unittest.TestCase):
    def test_rotated_box_uses_local_axes_and_envelope(self):
        t=np.pi/4;a=[[np.cos(t),np.sin(t),0],[-np.sin(t),np.cos(t),0],[0,0,1]]
        g=Geometry([dict(id='rotated',shape='obb',center_xyz_m=[0,0,0],local_size_xyz_m=[4,1,2],orientation_axes=a,group='obstacle')],.75)
        self.assertAlmostEqual(g.clearance([3/np.sqrt(2),3/np.sqrt(2),0])[0],.25)
        self.assertFalse(g.segment([-3,-3,0],[3,3,0],.75)[0])
    def test_tiny_obstacle_between_samples_is_rejected(self):
        g=Geometry([dict(id='thin',shape='sphere',center_xyz_m=[.023,0,0],radius_m=.001,group='obstacle')],0)
        self.assertFalse(g.segment([0,0,0],[.049,0,0],.002)[0])
    def test_real_geometry_planner_avoids_blocked_direct_route(self):
        from fixed_map_mission import read,AUDIT
        g=Geometry(read(AUDIT/'planning_geometry.json')['primitives'])
        s=[0,-1.8,3.1];end=[44.5,-2,3.1]
        self.assertFalse(g.segment(s,end)[0])
        path=g.plan(s,end)
        self.assertGreater(len(path),2)
        self.assertTrue(all(g.segment(a,b)[0] for a,b in zip(path,path[1:])))
    def test_contact_exemption_does_not_exempt_other_obstacles(self):
        g=Geometry([dict(id='ceiling',shape='sphere',center_xyz_m=[0,0,2],radius_m=.4,group='obstacle')])
        self.assertFalse(g.segment([0,0,.8],[0,0,3.1],.75,True)[0])
    def test_adapter_cannot_change_frozen_goal(self):
        e=Executor.__new__(Executor);e.task={'goal_xyz_m':[1,2,3]};e.commands=queue.Queue()
        with self.assertRaisesRegex(ValueError,'PROPOSED_TASK_REQUIRES_NEW_VALIDATION'):e.submit_task({'goal_xyz_m':[1,2,4]},'LLM_adapter')
        self.assertTrue(e.commands.empty())
    def test_measured_speed_violation_stops_before_control_rpc(self):
        e=Executor.__new__(Executor);e.prev=None;e.stage='APPROACH';e.ground_b=[0,1.8,.8]
        e.state=dict(DroneB_api_control=False,DroneB_landed=True,DroneB_position_xyz_m=e.ground_b,speed_m_s=.81)
        e.task={'max_speed_measured_m_s':.8}
        with self.assertRaisesRegex(ValueError,'MEASURED_SPEED_LIMIT_EXCEEDED'):e.control()
    def test_drone_b_movement_stops_before_control_rpc(self):
        e=Executor.__new__(Executor);e.prev=None;e.stage='APPROACH';e.ground_b=[0,1.8,.8]
        e.state=dict(DroneB_api_control=False,DroneB_landed=False,DroneB_position_xyz_m=e.ground_b)
        with self.assertRaisesRegex(ValueError,'DRONE_B_GROUND_INVARIANT_FAILED'):e.control()

if __name__=='__main__':unittest.main()
