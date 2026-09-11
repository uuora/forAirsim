import unittest
from unittest.mock import Mock
from run_visual_fallback import send_final_velocity


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
