"""Reproduce tied receipt clocks with different actual simulator state stamps."""
import unittest
from test_t1_async_control import ManualClock
from test_t1_async_control_v1p2 import FeedbackFake
import test_t1_async_control_v1p3 as prior_zero_hold
from test_t1_identity_motion_policy import DESCRIPTION
from t1_identity_motion_policy import IdentityMotionPolicy
from t1_async_control_v1p6 import AsyncControlV1P6
from t1_async_control_v1p5 import AsyncControlV1P5

class RefusalTests(unittest.TestCase):
    proposal=prior_zero_hold.ZeroHoldTests.proposal
    def ready(self,kind=AsyncControlV1P6):
        clock=ManualClock();logs=[];flight=FeedbackFake(logs,clock=clock)
        controller=kind(lambda:flight,lambda:None,clock=clock)
        controller._idle_tick(flight,controller._read(flight))
        for _ in range(8):clock.advance(.05);controller._idle_tick(flight,controller._read(flight))
        return clock,flight,controller,IdentityMotionPolicy('session',1,DESCRIPTION,clock=clock)

    def test_tied_clock_with_new_stamp_refuses_proof_without_fatal_or_command(self):
        clock,f,c,policy=self.ready();before=len(f.commands);dispatch=f.last_command
        state=c._read(f);last=c._idle_window.samples[-1]
        self.assertEqual(state['host_receipt_monotonic_s'],last['receipt_monotonic_s'])
        self.assertNotEqual(state['state_timestamp_ns'],last['state_timestamp_ns'])
        result=c._consume(f,policy,state,self.proposal(clock))
        self.assertNotIn('zero_held_stop_proof',result);self.assertIsNone(result['command']);self.assertIsNone(c._fatal)
        self.assertEqual(result['zero_held_stop_refusal']['reason'],'CURRENT_STABLE_ZERO_PROOF_BINDING_REQUIRED')
        self.assertEqual((len(f.commands),f.last_command),(before,dispatch));self.assertIsNone(c._active)
        self.assertFalse(result['motion_gate']['target_command_authorized'])
        self.assertEqual(result['motion_gate']['reason'],'CURRENT_ZERO_HOLD_PROOF_UNAVAILABLE')

    def test_new_actual_sample_establishes_exact_current_proof_after_refusal(self):
        clock,f,c,policy=self.ready();c._consume(f,policy,c._read(f),self.proposal(clock))
        clock.advance(.02);state=c._read(f);result=c._consume(f,policy,state,self.proposal(clock,2))
        self.assertEqual(result['zero_held_stop_proof']['current_self_state'],state)
        self.assertEqual(result['zero_held_stop_proof']['samples'][-1]['state_timestamp_ns'],state['state_timestamp_ns'])
        self.assertNotIn('zero_held_stop_refusal',result);self.assertIsNone(result['command'])

    def test_source_expiry_is_still_rejected_and_never_rebased(self):
        clock,f,c,policy=self.ready();proposal=self.proposal(clock);clock.advance(.7)
        result=c._consume(f,policy,c._read(f),proposal)
        self.assertNotEqual(result['motion_gate']['reason'],'ALREADY_ZERO_WITH_STRICT_SAMPLED_STOP')
        self.assertFalse(result['motion_gate']['target_command_authorized'])

    def test_residual_velocity_still_requires_actual_feedback_settlement(self):
        clock,f,c,policy=self.ready();f.stuck_velocity=True;f.velocity=[.0036,0.,0.]
        result=c._consume(f,policy,c._read(f),self.proposal(clock))
        self.assertEqual(result['motion_gate']['reason'],'SELF_STATE_RESIDUAL_REQUIRES_SETTLEMENT')
        self.assertNotIn('zero_held_stop_refusal',result);self.assertNotIn('zero_held_stop_proof',result)
        self.assertIsNotNone(result['command']);self.assertIsNotNone(c._active)

    def test_frozen_prior_controller_still_exposes_the_original_failure(self):
        clock,f,c,policy=self.ready(AsyncControlV1P5)
        with self.assertRaisesRegex(ValueError,'CURRENT_STABLE_ZERO_PROOF_BINDING_REQUIRED'):
            c._consume(f,policy,c._read(f),self.proposal(clock))

if __name__=='__main__':unittest.main()
