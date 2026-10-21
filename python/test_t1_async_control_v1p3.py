"""Zero holds preserve policy/freshness and cannot counterfeit dispatch or renewal."""
import unittest

from test_t1_async_control import ManualClock
from test_t1_async_control_v1p2 import FeedbackFake
import test_t1_async_control_v1p1 as fixtures
from t1_async_control_v1p3 import AsyncControlV1P3
from t1_identity_motion_policy import IdentityMotionPolicy
from test_t1_identity_motion_policy import DESCRIPTION


class ZeroHoldTests(unittest.TestCase):
    def ready(self):
        clock=ManualClock(); logs=[]; f=FeedbackFake(logs,clock=clock)
        c=AsyncControlV1P3(lambda:f,lambda:None,clock=clock)
        c._idle_tick(f,c._read(f))
        for _ in range(8): clock.advance(.05); c._idle_tick(f,c._read(f))
        policy=IdentityMotionPolicy('session',1,DESCRIPTION,clock=clock)
        return clock,f,c,policy

    def proposal(self, clock, sequence=1):
        frame,selection,surface=fixtures.IdleZeroTests.real_inputs(clock(),frame_id='frame-'+str(sequence),sequence=sequence)
        return dict(ticket_id='manual-'+str(sequence),enqueued_monotonic_s=clock(),frame=frame,
            selection=selection,surface=surface,desired_xy_m_s=[0.,0.])

    def test_zero_hold_has_current_validation_but_no_new_command_or_lease(self):
        clock,f,c,policy=self.ready(); count=len(f.commands); dispatch=f.last_command
        r=c._consume(f,policy,c._read(f),self.proposal(clock))
        self.assertEqual(r['motion_gate']['reason'],'ALREADY_ZERO_WITH_STRICT_SAMPLED_STOP')
        self.assertIsNone(r['command']); self.assertIsNone(c._active)
        self.assertEqual((len(f.commands),f.last_command),(count,dispatch))
        self.assertFalse(r['zero_held_stop_proof']['new_target_lease'])

    def test_repeated_frame_validation_is_not_bypassed(self):
        clock,f,c,policy=self.ready(); proposal=self.proposal(clock)
        c._consume(f,policy,c._read(f),proposal)
        clock.advance(.02)
        r=c._consume(f,policy,c._read(f),proposal)
        self.assertEqual(r['motion_gate']['reason'],'TARGET_FRAME_ALREADY_USED_FOR_COMMAND')
        self.assertFalse(r['motion_gate']['target_command_authorized'])

    def test_residual_or_old_state_cannot_be_zero_held(self):
        clock,f,c,policy=self.ready(); f.stuck_velocity=True; f.velocity=[.0036,0.,0.]
        r=c._consume(f,policy,c._read(f),self.proposal(clock))
        self.assertNotIn('zero_held_stop_proof',r)
        self.assertEqual(r['motion_gate']['reason'],'SELF_STATE_RESIDUAL_REQUIRES_SETTLEMENT')


if __name__=='__main__': unittest.main()
