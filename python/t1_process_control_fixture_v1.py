"""Offline RPC substitute for actual Windows spawn ownership tests.

Uses the production timed Flight, motion policy and controller. It performs
no AirSim/network/provider calls. It asserts every RPC and callback owner.
"""
import os
import threading
import time
from types import SimpleNamespace


class Future:
    def __init__(self):
        self.error = None
        self.result = True
    def attach_callback(self, callback):
        callback(self)


class FixtureRPC:
    def __init__(self, mode):
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.mode = mode
        self.velocity = [0., 0., 0.]
        self.reads = 0
    def owned(self):
        if (os.getpid(), threading.get_ident()) != (self.pid, self.thread):
            raise AssertionError('RPC_OWNER_CHANGED')
    def moveByVelocityZAsync(self, vx, vy, z, duration, **kwargs):
        self.owned()
        if self.mode == 'FAIL_RPC':
            raise RuntimeError('private RPC body must not leak')
        self.velocity = [vx, vy, 0.]
        return Future()
    def getMultirotorState(self, **kwargs):
        self.owned()
        self.reads += 1
        if self.mode == 'FAIL_STATE' and self.reads > 3:
            raise RuntimeError('FIXTURE_STATE_RPC_FAILURE')
        import airsim
        return SimpleNamespace(timestamp=self.reads, landed_state=airsim.LandedState.Flying,
            kinematics_estimated=SimpleNamespace(position=airsim.Vector3r(0., 0., -3.),
                linear_velocity=airsim.Vector3r(*self.velocity), linear_acceleration=airsim.Vector3r(),
                orientation=airsim.Quaternionr()))


def make_fixture(config, log):
    from t1_feedback_dispatch_v1 import TimedFeedbackFlight
    from t1_identity_motion_policy import MotionPolicyConfig
    from t1_identity_motion_policy_v1p1 import IdentityMotionPolicyV1P1
    # The test mode uses the public session field, not additional private config.
    mode = config['session_id'] if config['session_id'] in ('FAIL_RPC', 'FAIL_STATE') else 'NORMAL'
    flight = TimedFeedbackFlight(FixtureRPC(mode), config['translation'], config['world_z'],
                                config['low'], config['high'], log)
    flight.stage = 'PERCEPTION_XY'
    policy = IdentityMotionPolicyV1P1(config['session_id'], 1, config['description'],
        config=MotionPolicyConfig(workspace_low_ned_m=tuple(config['low']),
                                 workspace_high_ned_m=tuple(config['high'])))
    return flight, policy
