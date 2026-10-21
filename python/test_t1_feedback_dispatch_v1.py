"""Deadline stalls use actual owned Flight calls and independently audited rows."""
import copy
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import airsim

from audit_t1_feedback_dispatch_v1 import audit
from t1_control_settlement import own_state_feedback
from t1_feedback_dispatch_v1 import TimedFeedbackFlight, DispatchDeadlineExpired
from t1_async_control_v1p7 import AsyncControlV1P7
from test_t1_async_control import ManualClock
from test_t1_identity_flight_v2 import Future


class Client:
    def __init__(self):
        self.calls, self.threads, self.reads = [], [], 0
        self.owner = threading.get_ident()
        self.velocity = [.002, 0., 0.]
        self.future = None

    def owned(self):
        self.threads.append(threading.get_ident())
        if threading.get_ident() != self.owner:
            raise AssertionError('RPC_OWNER_CHANGED')

    def moveByVelocityZAsync(self, *args, **kwargs):
        self.owned(); self.calls.append((args, kwargs)); self.future = Future(); return self.future

    def getMultirotorState(self, **kwargs):
        self.owned(); self.reads += 1
        zero = airsim.Vector3r()
        return SimpleNamespace(timestamp=self.reads, landed_state=airsim.LandedState.Flying,
            kinematics_estimated=SimpleNamespace(position=airsim.Vector3r(0., 0., -3.),
                linear_velocity=airsim.Vector3r(*self.velocity), linear_acceleration=zero,
                orientation=airsim.Quaternionr()))


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.clock = ManualClock(); self.client = Client(); self.logs = []
        self.mock_clock = patch('time.monotonic', self.clock); self.mock_clock.start()
        self.addCleanup(self.mock_clock.stop)
        self.f = TimedFeedbackFlight(self.client, [0., 0., 0.], -3., [-.3, -.3, -3.15], [.3, .3, -2.85], self.logs.append)
        self.f.stage = 'PERCEPTION_XY'
        self.f.issue([0., 0.], .12, 'OWN_STATE_INITIAL_ZERO')
        self.c = AsyncControlV1P7(lambda: self.f, lambda: None, clock=self.clock)

    def state(self):
        return self.c._read(self.f)

    def stall(self, times=1):
        count = [0]
        def delayed(*args, **kwargs):
            plan = own_state_feedback(*args, **kwargs)
            if count[0] < times:
                count[0] += 1; self.clock.advance(.13)
            return plan
        return patch('t1_feedback_dispatch_v1.own_state_feedback', delayed)

    def test_fresh_feedback_records_exact_check_without_extra_state_query(self):
        self.clock.advance(.05); row = self.f.brake_tick(self.state())
        self.assertEqual(self.client.reads, 1)
        self.assertEqual(row['dispatch_monotonic_s'], row['deadline_check_monotonic_s'])
        self.assertEqual(self.logs[-1]['attempt'], 0)
        self.assertEqual(self.f.feedback_recoveries, 0)

    def test_preparation_stall_discards_nonzero_vector_then_zero_and_fresh_query(self):
        self.clock.advance(.05); s = self.state()
        with self.stall(): row = self.f.brake_tick(s)
        refusal = next(e for e in self.logs if e['kind'] == 'FEEDBACK_DISPATCH_REFUSAL')
        self.assertNotEqual(refusal['rejected_feedback']['command_velocity_xy_m_s'], [0., 0.])
        self.assertEqual(self.client.calls[-2][0][:2], (0., 0.))
        self.assertEqual(self.client.reads, 2)
        self.assertEqual(self.logs[-1]['attempt'], 1)
        fresh = self.logs[-1]['command_self_state']
        self.assertNotEqual(s['state_timestamp_ns'], fresh['state_timestamp_ns'])
        self.assertLessEqual(row['dispatch_monotonic_s'] - fresh['host_receipt_monotonic_s'], .12)
        self.assertEqual(set(self.client.threads), {self.client.owner})

    def test_already_expired_state_is_never_rebased_or_computed(self):
        s = self.state(); original = copy.deepcopy(s); self.clock.advance(.13)
        self.f.brake_tick(s)
        r = next(e for e in self.logs if e['kind'] == 'FEEDBACK_DISPATCH_REFUSAL')
        self.assertIsNone(r['rejected_feedback']); self.assertEqual(s, original)
        self.assertEqual(r['timing']['rejection_boundary'], 'PREPARATION_ENTRY')

    def test_two_expirations_stop_after_one_refresh_and_leave_zero(self):
        self.clock.advance(.05)
        with self.stall(2), self.assertRaisesRegex(RuntimeError, 'AFTER_ONE_FRESH_RETRY'):
            self.f.brake_tick(self.state())
        self.assertEqual(self.client.reads, 2)
        self.assertEqual(self.f.last_xy, [0., 0.])
        self.assertEqual(len([e for e in self.logs if e['kind'] == 'FEEDBACK_DISPATCH_REFUSAL']), 2)
        self.assertFalse(any(e.get('purpose') == 'OWN_STATE_FEEDBACK_BRAKE' for e in self.logs))

    def test_recovery_clears_target_without_new_target_lease(self):
        self.f.issue([.01, 0.], .12, 'CURRENT_TARGET_TRAJECTORY')
        self.clock.advance(.05)
        with self.stall(): self.f.brake_tick(self.state())
        self.assertIsNone(self.f.target_lease_end)
        self.assertEqual(len([e for e in self.logs if e.get('purpose') == 'CURRENT_TARGET_TRAJECTORY']), 1)

    def test_future_state_is_fatal_without_feedback_rpc_or_refresh(self):
        s = self.state(); s['host_receipt_monotonic_s'] += .1
        before = len(self.client.calls)
        with self.assertRaisesRegex(ValueError, 'EXPIRED_OR_FUTURE'): self.f.brake_tick(s)
        self.assertEqual(len(self.client.calls), before); self.assertEqual(self.client.reads, 1)

    def test_refresh_protection_violation_stops_after_actual_zero(self):
        self.clock.advance(.05)
        with self.stall(), patch.object(self.f, 'protection', side_effect=[[], ['MEASURED_WORKSPACE']]):
            with self.assertRaisesRegex(RuntimeError, 'PROTECTION_DURING_FEEDBACK_RETRY'):
                self.f.brake_tick(self.state())
        self.assertEqual(self.f.last_xy, [0., 0.])

    def test_final_dispatch_validation_stall_is_caught_at_exact_clock(self):
        original = self.f.issue; held = [False]
        def stalled(xy, duration, purpose, **kwargs):
            if purpose == 'OWN_STATE_FEEDBACK_BRAKE' and not held[0]:
                held[0] = True; self.clock.advance(.13)
            return original(xy, duration, purpose, **kwargs)
        self.clock.advance(.05)
        with patch.object(self.f, 'issue', stalled): self.f.brake_tick(self.state())
        r = next(e for e in self.logs if e['kind'] == 'FEEDBACK_DISPATCH_REFUSAL')
        self.assertEqual(r['timing']['rejection_checked_monotonic_s'], r['actual_zero_dispatch_monotonic_s'])

    def test_expired_target_dispatch_remains_refused_without_rpc(self):
        before = len(self.client.calls)
        with self.assertRaises(DispatchDeadlineExpired):
            self.f.issue([.01, 0.], .12, 'CURRENT_TARGET_TRAJECTORY', gate={'latest_dispatch_monotonic_s': self.clock() - .01})
        self.assertEqual(len(self.client.calls), before)

    def test_actual_clear_and_original_stop_deadline_survive_recovery(self):
        target = self.f.issue([.01, 0.], .12, 'CURRENT_TARGET_TRAJECTORY')
        self.c._activate_stop('original', target, 'original-frame')
        self.clock.advance(.13); self.c._stop_tick(self.f, self.state())
        clear, deadline = self.c._active['clear'], self.c._active['stop_deadline']
        self.clock.advance(.05)
        with self.stall(): self.c._stop_tick(self.f, self.state())
        self.assertEqual(self.c._active['clear'], clear)
        self.assertEqual(self.c._active['stop_deadline'], deadline)
        self.assertEqual(deadline, target['dispatch_monotonic_s'] + .12 + 2.)
        self.assertFalse(self.c.receipts())
        self.client.velocity = [0., 0., 0.]
        for _ in range(12):
            self.clock.advance(.05); self.c._stop_tick(self.f, self.state())
        proof = self.c.receipts()['original']
        self.assertFalse(proof['target_lease_renewed'])
        self.assertGreaterEqual(proof['distinct_stable_simulator_state_timestamps'], 3)
        self.assertLessEqual(proof['completed_monotonic_s'], deadline)

    def test_rpc_callback_preserves_replaced_false_and_private_error_category(self):
        self.f.issue([0., 0.], .12, 'OWN_STATE_CALLBACK_TEST')
        self.client.future.finish(False)
        self.assertEqual(self.logs[-1]['status'], 'RETURNED_FALSE')
        self.assertFalse(self.f.rpc_fault)
        self.client.future.finish(error=RuntimeError('private response secret'))
        self.assertTrue(self.f.rpc_fault)
        self.assertNotIn('private', str(self.logs[-1]))

    def test_second_expired_event_admission_cannot_trigger_third_query(self):
        original = self.f.log
        def stalled(row):
            original(copy.deepcopy(row))
            if row['kind'] == 'SELF_STATE': self.clock.advance(.13)
        self.f.log = stalled
        with self.assertRaisesRegex(RuntimeError, 'AFTER_ONE_FRESH_RETRY'):
            self.f.brake_tick(self.state())
        self.assertEqual(self.client.reads, 2)
        self.assertEqual(self.f.last_xy, [0., 0.])

    def test_condition_wait_is_measured_separately_from_rpc_and_log(self):
        class DelayedCondition:
            def __enter__(inner): self.clock.advance(.13)
            def __exit__(inner, *args): pass
        self.c._condition = DelayedCondition()
        self.f.brake_tick(self.state())
        t = next(e['timing'] for e in self.logs if e['kind'] == 'OWN_STATE_READ_TIMING')
        self.assertAlmostEqual(t['condition_acquired_monotonic_s'] - t['condition_entered_monotonic_s'], .13)
        self.assertEqual(t['query_completed_monotonic_s'] - t['query_started_monotonic_s'], 0.)
        self.assertEqual(self.logs[-1]['attempt'], 1)

    def test_state_event_admission_stall_is_saved_then_recovers_fresh(self):
        original = self.f.log; held = [False]
        def log(row):
            original(copy.deepcopy(row))
            if row['kind'] == 'SELF_STATE' and not held[0]:
                held[0] = True; self.clock.advance(.13)
        self.f.log = log
        self.f.brake_tick(self.state())
        t = next(e['timing'] for e in self.logs if e['kind'] == 'OWN_STATE_READ_TIMING')
        self.assertAlmostEqual(t['state_event_admission_completed_monotonic_s'] - t['state_event_admission_started_monotonic_s'], .13)
        self.assertEqual(self.logs[-1]['attempt'], 1)

    def recovered_events(self):
        self.clock.advance(.05)
        with self.stall(): self.f.brake_tick(self.state())
        return dict(low_speed_workspace={'low_ned_m': self.f.low.tolist(), 'high_ned_m': self.f.high.tolist()}), copy.deepcopy(self.logs)

    def test_independent_audit_accepts_bound_fresh_retry(self):
        report, events = self.recovered_events(); r = audit(report, events)
        self.assertEqual(r['status'], 'PASS_STRICT_FEEDBACK_DISPATCH_AUDIT', r['issues'])
        self.assertEqual(r['recovered_ticks'], 1)

    def test_audit_rejects_clock_vector_zero_and_new_state_tampering(self):
        report, events = self.recovered_events()
        for mutation in ('clock', 'vector', 'zero', 'state'):
            tampered = copy.deepcopy(events)
            refusal = next(e for e in tampered if e['kind'] == 'FEEDBACK_DISPATCH_REFUSAL')
            if mutation == 'clock': refusal['timing']['rejection_checked_monotonic_s'] -= .1
            if mutation == 'vector': refusal['rejected_feedback']['command_velocity_xy_m_s'][0] += .01
            if mutation == 'zero':
                next(e for e in tampered if e.get('command_id') == refusal['actual_zero_command_id'])['velocity_xy_m_s'] = [.01, 0.]
            if mutation == 'state':
                tampered = [e for e in tampered if e.get('query_scope') != 'ONE_FRESH_OWN_STATE_FEEDBACK_RETRY']
            with self.subTest(mutation=mutation): self.assertEqual(audit(report, tampered)['status'], 'FAIL_FEEDBACK_DISPATCH_EVIDENCE')


if __name__ == '__main__':
    unittest.main()
