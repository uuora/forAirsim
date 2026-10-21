import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import airsim
from braking_startup_readiness import vehicle_sample,probe


class StartupTests(unittest.TestCase):
    def client(self):
        client=Mock();client.getMultirotorState.return_value=SimpleNamespace(timestamp=10,landed_state=airsim.LandedState.Landed,
            kinematics_estimated=SimpleNamespace(linear_velocity=airsim.Vector3r(),linear_acceleration=airsim.Vector3r()))
        client.simGetObjectPose.return_value=airsim.Pose(airsim.Vector3r(0,-3,.825))
        client.simGetVehiclePose.return_value=airsim.Pose(airsim.Vector3r(0,0,1.675))
        client.isApiControlEnabled.return_value=False;client.simGetCollisionInfo.return_value=SimpleNamespace(object_name='Ground_6')
        return client

    def test_readiness_uses_self_world_pose_and_never_vehicle_local_height(self):
        client=self.client();self.assertTrue(vehicle_sample(client,'DroneA')['ground_readiness'])
        client.simGetObjectPose.assert_called_once_with('DroneA');client.simGetVehiclePose.assert_not_called()

    def test_external_control_or_initial_motion_is_not_ready(self):
        client=self.client();client.isApiControlEnabled.return_value=True
        self.assertFalse(vehicle_sample(client,'DroneA')['ground_readiness'])

    def test_probe_uses_methods_present_in_local_airsim_client(self):
        client=Mock(spec_set=airsim.MultirotorClient);fixture=self.client()
        for name in ('getMultirotorState','simGetObjectPose','isApiControlEnabled','simGetCollisionInfo'):
            getattr(client,name).return_value=getattr(fixture,name).return_value
        client.simIsPause.return_value=False
        with TemporaryDirectory() as folder, patch('braking_startup_readiness.airsim.MultirotorClient',return_value=client), \
            patch('braking_startup_readiness.time.monotonic',side_effect=[10.,10.,10.1,10.2,10.7]), \
            patch('braking_startup_readiness.time.sleep'):
            self.assertTrue(probe(Path(folder)/'receipt.json'))
        client.simIsPause.assert_called_once()
        client=self.client();client.getMultirotorState.return_value.kinematics_estimated.linear_velocity=airsim.Vector3r(.02,0,0)
        self.assertFalse(vehicle_sample(client,'DroneA')['ground_readiness'])


if __name__=='__main__':unittest.main()
