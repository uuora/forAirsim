"""New worker ownership, early inertial stopping rejection and busy/no-renew tests."""
import threading
import time
import unittest

from test_t1_async_control import FakeFlight, RecordingPolicy, ManualClock
from t1_async_control_v1p2 import AsyncControlV1P2
from t1_control_settlement import own_state_feedback


class FeedbackFake(FakeFlight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_anchor_xy = None
        self.stamp = 0

    def state(self):
        s = super().state()
        self.stamp += 1
        s['state_timestamp_ns'] = self.stamp
        return s

    def clear_target_tick(self, state):
        row = super().brake_tick(state)
        if row['velocity_xy_m_s'] == [0., 0.]:
            self.stop_anchor_xy = list(state['position_world_ned_m'][:2])
        return row

    def brake_tick(self, state=None):
        state = state or self.state()
        self.stop_anchor_xy = self.stop_anchor_xy or list(state['position_world_ned_m'][:2])
        plan = own_state_feedback(state, self.stop_anchor_xy, self.last_xy, self.last_command,
            self.clock(), [-.3, -.3, -3.15], [.3, .3, -2.85])
        return self.issue(plan['command_velocity_xy_m_s'], .12, 'OWN_STATE_FEEDBACK_BRAKE')


class StrictControlTests(unittest.TestCase):
    def direct(self):
        clock = ManualClock(); logs = []
        f = FeedbackFake(logs, clock=clock)
        c = AsyncControlV1P2(lambda: f, lambda: RecordingPolicy(), clock=clock)
        command = f.issue([.009, 0.], .12, 'CURRENT_TARGET_TRAJECTORY')
        c._activate_stop('manual-1', command, 'frame-1')
        return clock, f, c, logs

    def test_before_expiry_never_clears_or_renews_target(self):
        clock, f, c, _ = self.direct()
        clock.advance(.1); c._stop_tick(f, c._read(f))
        self.assertEqual(len(f.commands), 1)
        self.assertFalse(c.receipts())

    def test_old_residual_threshold_cannot_complete_new_stop(self):
        clock, f, c, _ = self.direct()
        f.stuck_velocity = True
        for _ in range(15):
            f.velocity = [.0036, 0., 0.]; clock.advance(.05)
            c._stop_tick(f, c._read(f))
        self.assertFalse(c.receipts())
        self.assertTrue(any(x['purpose'] == 'OWN_STATE_FEEDBACK_BRAKE' and x['velocity_xy_m_s'][0] < 0
            for x in f.commands))
        self.assertFalse(any(x['purpose'] == 'CURRENT_TARGET_TRAJECTORY' for x in f.commands[1:]))

    def test_busy_proposal_cannot_renew_any_lease(self):
        clock, f, c, _ = self.direct()
        before = len(f.commands)
        r = c._consume(f, RecordingPolicy(), c._read(f), dict(ticket_id='manual-2',
            enqueued_monotonic_s=clock()))
        self.assertEqual(r['motion_gate']['reason'], 'CONTROLLER_BUSY_UNTIL_STRICT_SAMPLED_STOP')
        self.assertEqual(len(f.commands), before)

    def test_actual_zero_then_dwell_and_distinct_states_prove_stop(self):
        clock, f, c, logs = self.direct()
        for _ in range(14):
            clock.advance(.05); c._stop_tick(f, c._read(f))
        r = c.receipts()['manual-1']
        self.assertGreaterEqual(r['samples'][-1]['receipt_monotonic_s']-
            r['target_setpoint_cleared_monotonic_s'], .3-1e-9)
        self.assertGreaterEqual(r['distinct_stable_simulator_state_timestamps'], 3)
        self.assertFalse(r['target_lease_renewed'])
        self.assertTrue(any(x['kind']=='TARGET_LEASE_STOP' for x in logs))

    def test_idle_omission_requires_strict_history(self):
        clock = ManualClock(); logs=[]; f = FeedbackFake(logs, clock=clock)
        c = AsyncControlV1P2(lambda:f, lambda:None, clock=clock)
        c._idle_tick(f, c._read(f))
        for _ in range(4): clock.advance(.05); c._idle_tick(f, c._read(f))
        self.assertFalse(any(x['kind']=='IDLE_ZERO_OMITTED' for x in logs))
        for _ in range(4): clock.advance(.05); c._idle_tick(f, c._read(f))
        self.assertTrue(any(x['kind']=='IDLE_ZERO_OMITTED' for x in logs))
        f.stuck_velocity=True; f.velocity=[.0036,0.,0.]
        clock.advance(.05); c._idle_tick(f, c._read(f))
        self.assertLess(f.commands[-1]['velocity_xy_m_s'][0], 0.)

    def test_thread_keeps_ownership_during_main_stall_and_closes(self):
        logs=[]; holder={}; owner=threading.get_ident()
        def make():
            holder['flight']=FeedbackFake(logs)
            return holder['flight']
        c = AsyncControlV1P2(make, RecordingPolicy, period_s=.02)
        try:
            c.start()
            now=time.monotonic()
            ticket=c.submit({'frame_id':'one','receipt':now-.05}, {'mutable':'original'}, {}, [.009,0.])
            decision=c.await_decision(ticket)
            self.assertEqual(decision['motion_gate']['status'], 'ALLOW_BOUNDED_COMMAND')
            time.sleep(.7)
            self.assertIn(ticket, c.receipts())
            self.assertNotEqual(holder['flight'].owner, owner)
            self.assertEqual(set(holder['flight'].call_threads), {holder['flight'].owner})
            self.assertEqual(c.wait_until_settled()['velocity_ned_m_s'], [0.,0.,0.])
        finally:
            r=c.close(timeout=3.)
        self.assertEqual(r['status'], 'OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE')
        self.assertFalse(c.is_alive())


if __name__ == '__main__': unittest.main()
