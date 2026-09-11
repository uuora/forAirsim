import unittest
from target_provider import TargetObservation
from visual_range_gate import range_command


class RangeGateTests(unittest.TestCase):
    def test_old_frame_cannot_advance(self):
        obs=TargetObservation('DroneA',True,'camera',normalised_error_xy=(0,0),depth_m=4)
        for age in [0.6,-1,float('nan')]:
            self.assertEqual(range_command(obs,'DETECTED',age).velocity,(0,0,0))
        self.assertEqual(range_command(obs,'DETECTED',0.1).velocity,(0.15,0,0))

    def test_near_or_missing_depth_stops(self):
        for depth in [None,1.9]:
            obs=TargetObservation('DroneA',True,'camera',normalised_error_xy=(0,0),depth_m=depth)
            self.assertEqual(range_command(obs,'DETECTED',0.1).velocity,(0,0,0))
