import unittest
from unittest.mock import Mock
import numpy as np

from run_visual_fallback import committed_dash_velocity, send_final_velocity


class FinalVelocityTests(unittest.TestCase):
    def test_rpc_receives_logged_vertical_correction(self):
        client = Mock()
        for vz in (-0.15, 0.0, 0.15):
            vector = (0.08, -0.03, vz)
            send_final_velocity(client, 'DroneB', vector)
            args, kwargs = client.moveByVelocityAsync.call_args
            self.assertEqual(args, (*vector, 0.5))
            self.assertEqual(kwargs['vehicle_name'], 'DroneB')
            client.moveByVelocityAsync.return_value.join.assert_called()

    def test_committed_dash_continues_through_bounded_occlusion(self):
        velocity, distance = committed_dash_velocity(
            np.array([-2.3, 0.2, -3.0]), np.array([-1.5, 0.2, -3.0]))
        self.assertEqual(velocity, (0.35, 0.0, 0.0))
        self.assertAlmostEqual(distance, 0.8)

    def test_committed_dash_refuses_excess_travel(self):
        with self.assertRaises(TimeoutError):
            committed_dash_velocity((0, 0, 0), (1.35, 0, 0))
