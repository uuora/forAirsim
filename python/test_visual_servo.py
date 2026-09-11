import unittest
from visual_servo import command, stop_forward_at_range


class ServoTests(unittest.TestCase):
    def test_range_stop_does_not_deadlock_alignment(self):
        velocity=command(True,(0.084,0.007),2.37).velocity
        stopped=stop_forward_at_range(velocity,True)
        self.assertEqual(stopped[0],0)
        self.assertGreater(stopped[1],0)
        self.assertEqual(stop_forward_at_range((0.15,0,0),True),(0,0,0))

    def test_invalid_or_lost_never_moves(self):
        for detected, error, depth, clipped in [(False,(0,0),3,False),
                (True,(float('nan'),0),3,False),(True,(0,0),float('inf'),False),
                (True,(0,0),-1,False),(True,(0,0),3,True)]:
            self.assertEqual(command(detected,error,depth,clipped).velocity,(0,0,0))

    def test_distance_stop_precedes_alignment(self):
        self.assertEqual(command(True,(0.8,0.8),1.9).reason,'DISTANCE_STOP')

    def test_final_approach_allows_close_forward_motion(self):
        self.assertEqual(command(True,(0,0),1.9,stop_depth=0.35).velocity,(0.15,0,0))
        self.assertEqual(command(True,(0,0),0.2,stop_depth=0.35).reason,'DISTANCE_STOP')

    def test_body_axes_and_limits(self):
        self.assertEqual(command(True,(-0.9,0.9),4).velocity,(0,-0.15,0.15))
        self.assertEqual(command(True,(0,0),4).velocity,(0.15,0,0))
        self.assertEqual(command(True,(0,0),4).duration_s,0.2)
