"""Thread ownership, measured leases, stalls and conservative failure tests."""
import copy
import math
import threading
import time
import unittest

from t1_async_control import AsyncControl, closed_category
from t1_identity_motion_policy import IdentityMotionPolicy
from test_t1_identity_motion_policy import DESCRIPTION, bound_fixture, rebind_surface


class FakeFlight:
    def __init__(self, logs, *, clock=time.monotonic):
        self.clock = clock
        self.log = logs.append
        self.last_xy = [0., 0.]
        self.last_command = clock()-.1
        self.stage = 'PERCEPTION_XY'
        self.owner = threading.get_ident()
        self.call_threads = []
        self.velocity = [0., 0., 0.]
        self.fault = False
        self.fail_state = False
        self.stuck_velocity = False
        self.expire_next_dispatch = False
        self.read_delay_s = 0.
        self.protection_reasons = []
        self.read_entered = threading.Event()
        self.read_release = None
        self.commands = []

    def owned(self):
        self.call_threads.append(threading.get_ident())
        if threading.get_ident() != self.owner:
            raise AssertionError('FAKE_CLIENT_CROSSED_THREAD')

    def state(self):
        self.owned()
        self.read_entered.set()
        if self.read_release is not None:
            self.read_release.wait(2.)
        if self.read_delay_s:
            time.sleep(self.read_delay_s)
        if self.fail_state:
            raise RuntimeError('FAKE_STATE_RPC_FAULT')
        return dict(host_receipt_monotonic_s=self.clock(), state_timestamp_ns=1,
                    position_world_ned_m=[0., 0., -3.], velocity_ned_m_s=self.velocity[:],
                    acceleration_ned_m_s2=[0., 0., 0.], orientation_xyzw=[0., 0., 0., 1.], landed=False)

    def protection(self, state):
        self.owned()
        return ['FAKE_SELF_STATE_FAULT'] if self.fault else self.protection_reasons[:]

    def issue(self, xy, duration, purpose, *, gate=None):
        self.owned()
        if gate and self.expire_next_dispatch:
            self.expire_next_dispatch = False
            raise RuntimeError('SELF_STATE_OR_TARGET_DISPATCH_DEADLINE_EXPIRED')
        now = self.clock()
        if gate and now > gate['latest_dispatch_monotonic_s']:
            raise RuntimeError('SELF_STATE_OR_TARGET_DISPATCH_DEADLINE_EXPIRED')
        if gate and 'command_end_deadline_monotonic_s' in gate and now+duration > gate['command_end_deadline_monotonic_s']:
            raise RuntimeError('COMMAND_DISPATCH_DEADLINE_EXPIRED')
        self.last_xy, self.last_command = list(xy), now
        if not self.stuck_velocity:
            self.velocity = list(xy)+[0.]
        row = dict(kind='COMMAND', purpose=purpose, stage=self.stage,
                   dispatch_monotonic_s=now, returned_monotonic_s=self.clock(),
                   velocity_xy_m_s=list(xy), duration_s=duration,
                   target_frame_id=(gate.get('current_binding') or {}).get('frame_id')
                   if gate and purpose == 'CURRENT_TARGET_TRAJECTORY' else None)
        self.commands.append(copy.deepcopy(row))
        self.log(row)
        return row

    def brake_tick(self, state=None):
        self.owned()
        elapsed = max(0., min(self.clock()-self.last_command, .4))
        norm = math.hypot(*self.last_xy)
        factor = max(0., norm-.15*elapsed)/norm if norm > 0 else 0.
        return self.issue([x*factor for x in self.last_xy], .12, 'OWN_STATE_BRAKE_OR_HOLD')


class RecordingPolicy:
    def __init__(self):
        self.owner = threading.get_ident()
        self.calls = []
        self.closed = False
        self.used = set()

    def decide(self, **kwargs):
        assert threading.get_ident() == self.owner
        self.calls.append(copy.deepcopy(kwargs))
        frame = kwargs['frame']
        now = kwargs['now']
        gate = dict(status='ALLOW_BOUNDED_COMMAND', reason='FAKE_FRESH',
                    decision_monotonic_s=now, target_command_authorized=True,
                    current_binding={'frame_id': frame['frame_id']},
                    command_velocity_xy_m_s=kwargs['desired_xy_m_s'], command_duration_s=.12,
                    latest_dispatch_monotonic_s=now+.1, command_end_deadline_monotonic_s=now+.3)
        if now-frame['receipt'] > .5:
            gate.update(status='WAIT_NO_TARGET', reason='CURRENT_FRAME_EXPIRED', target_command_authorized=False)
        elif frame['frame_id'] in self.used:
            gate.update(status='REJECT', reason='TARGET_FRAME_ALREADY_USED_FOR_COMMAND', target_command_authorized=False)
        elif kwargs['selection'].get('mutable') != 'original':
            gate.update(status='REJECT', reason='MUTATED_INPUT', target_command_authorized=False)
        else:
            self.used.add(frame['frame_id'])
        return gate

    def close(self):
        assert threading.get_ident() == self.owner
        self.closed = True


class ManualClock:
    def __init__(self):
        self._lock = threading.Lock()
        self.value = 10.

    def __call__(self):
        with self._lock:
            return self.value

    def advance(self, step):
        with self._lock:
            self.value += step


class AsyncControlTests(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.instances = {}
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            if controller.is_alive():
                flight = self.instances.get(id(controller), {}).get('flight')
                if flight:
                    flight.fault = flight.fail_state = flight.stuck_velocity = False
                    flight.read_release = None
                    flight.velocity = [0., 0., 0.]
                try:
                    controller.close(timeout=2.)
                except RuntimeError:
                    pass

    def create(self, *, clock=time.monotonic, period_s=.01, real_policy=False):
        holder = {}

        def flight_factory():
            holder['factory_thread'] = threading.get_ident()
            holder['flight'] = FakeFlight(self.logs, clock=clock)
            return holder['flight']

        def policy_factory():
            holder['policy'] = IdentityMotionPolicy('session', 1, DESCRIPTION, clock=clock) if real_policy else RecordingPolicy()
            return holder['policy']

        control = AsyncControl(flight_factory, policy_factory, period_s=period_s, clock=clock)
        self.instances[id(control)] = holder
        self.controllers.append(control)
        control.start()
        return control, holder

    @staticmethod
    def frame(name='frame-1', receipt=None):
        return {'frame_id': name, 'receipt': time.monotonic() if receipt is None else receipt}

    def submit(self, control, frame=None):
        return control.submit(frame or self.frame(), {'mutable': 'original'}, {}, [.01, 0.])

    @staticmethod
    def wait_for(predicate, timeout=1.0):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.002)
        raise AssertionError('CONDITION_NOT_REACHED')

    def test_factories_and_client_policy_calls_stay_on_worker(self):
        control, holder = self.create()
        ticket = self.submit(control)
        decision = control.await_decision(ticket)
        self.assertEqual(decision['motion_gate']['status'], 'ALLOW_BOUNDED_COMMAND')
        self.assertNotEqual(holder['factory_thread'], threading.get_ident())
        self.assertTrue(all(x == holder['factory_thread'] for x in holder['flight'].call_threads))
        self.assertEqual(holder['policy'].owner, holder['factory_thread'])
        self.assertFalse(control._thread.daemon)

    def test_main_thread_stall_does_not_renew_or_delay_lease_clear(self):
        control, holder = self.create()
        ticket = self.submit(control)
        decision = control.await_decision(ticket)
        # Represents a .35 second capture/model/file wait on the calling thread.
        time.sleep(.35)
        receipt = control.receipts()[ticket]
        self.assertEqual(len([x for x in holder['flight'].commands if x['purpose'] == 'CURRENT_TARGET_TRAJECTORY']), 1)
        self.assertLess(receipt['clear_after_declared_lease_s'], .04)
        self.assertGreater(len([x for x in self.logs if x['kind'] == 'SELF_STATE']), 20)
        self.assertLess(control.scheduling_receipt()['maximum_observed_query_gap_s'], .04)
        self.assertGreaterEqual(receipt['samples'][-1]['receipt_monotonic_s']-receipt['samples'][0]['receipt_monotonic_s'], .05)

    def test_busy_proposal_cannot_renew_active_lease(self):
        control, holder = self.create()
        first = self.submit(control)
        control.await_decision(first)
        second = self.submit(control, self.frame('frame-2'))
        rejected = control.await_decision(second)
        self.assertEqual(rejected['motion_gate']['reason'], 'CONTROLLER_BUSY_UNTIL_SAMPLED_STOP')
        self.assertIsNone(rejected['command'])
        self.assertNotIn(second, control.receipts())
        self.assertEqual(len(holder['policy'].calls), 1)

    def test_submission_and_latest_state_are_detached_deepcopies(self):
        control, holder = self.create()
        selection = {'mutable': 'original'}
        frame = self.frame()
        ticket = control.submit(frame, selection, {}, [.01, 0.])
        selection['mutable'] = 'changed'
        frame['frame_id'] = 'changed'
        decision = control.await_decision(ticket)
        self.assertEqual(decision['motion_gate']['status'], 'ALLOW_BOUNDED_COMMAND')
        own = control.latest_state()
        own['position_world_ned_m'][0] = 100.
        self.assertEqual(control.latest_state()['position_world_ned_m'][0], 0.)

    def test_replayed_frame_rejects_after_first_stop(self):
        control, holder = self.create()
        frame = self.frame()
        first = self.submit(control, frame)
        control.await_decision(first)
        self.wait_for(lambda: first in control.receipts())
        second = self.submit(control, frame)
        decision = control.await_decision(second)
        self.assertEqual(decision['motion_gate']['reason'], 'TARGET_FRAME_ALREADY_USED_FOR_COMMAND')
        self.wait_for(lambda: second in control.receipts())
        self.assertEqual(len([x for x in holder['flight'].commands if x['purpose'] == 'CURRENT_TARGET_TRAJECTORY']), 1)

    def test_stale_observation_gets_only_own_state_zero_receipt(self):
        control, holder = self.create()
        ticket = self.submit(control, self.frame(receipt=time.monotonic()-.7))
        decision = control.await_decision(ticket)
        self.assertEqual(decision['motion_gate']['reason'], 'CURRENT_FRAME_EXPIRED')
        self.assertEqual(decision['command']['velocity_xy_m_s'], [0., 0.])
        self.wait_for(lambda: ticket in control.receipts())
        self.assertFalse(any(x['purpose'] == 'CURRENT_TARGET_TRAJECTORY' for x in holder['flight'].commands))

    def test_dispatch_expiry_rejects_conservatively_and_remeasures(self):
        control, holder = self.create()
        holder['flight'].expire_next_dispatch = True
        before = len([x for x in self.logs if x['kind'] == 'SELF_STATE'])
        ticket = self.submit(control)
        decision = control.await_decision(ticket)
        self.assertEqual(decision['motion_gate']['reason'], 'ACTUAL_DISPATCH_DEADLINE_EXPIRED')
        self.assertFalse(decision['motion_gate']['target_command_authorized'])
        self.assertEqual(decision['command']['purpose'], 'OWN_STATE_BRAKE_OR_HOLD')
        self.assertGreaterEqual(len([x for x in self.logs if x['kind'] == 'SELF_STATE'])-before, 2)
        control.check()

    def test_protection_fault_immediately_commands_zero_and_fails_visibly(self):
        control, holder = self.create()
        holder['flight'].fault = True
        self.wait_for(lambda: control.scheduling_receipt()['fatal'] is not None)
        with self.assertRaisesRegex(RuntimeError, 'SELF_STATE_PROTECTION'):
            control.check()
        self.wait_for(lambda: not control.is_alive())
        self.assertTrue(any(x['purpose'] == 'OWN_STATE_PROTECTION_ZERO' for x in holder['flight'].commands))
        with self.assertRaisesRegex(RuntimeError, 'NO_VERIFIED_SAMPLED_STOP'):
            control.close()

    def test_state_rpc_fault_still_attempts_explicit_zero_and_no_false_stop(self):
        control, holder = self.create()
        holder['flight'].fail_state = True
        self.wait_for(lambda: not control.is_alive())
        self.assertTrue(any(x['purpose'] == 'OWN_STATE_CONTROLLER_FATAL_ZERO' for x in holder['flight'].commands))
        self.assertIsNone(control.scheduling_receipt()['shutdown_stop'])
        with self.assertRaisesRegex(RuntimeError, 'NO_VERIFIED_SAMPLED_STOP'):
            control.close()

    def test_close_interrupts_lease_with_explicit_zero_and_joined_worker(self):
        control, holder = self.create()
        ticket = self.submit(control)
        control.await_decision(ticket)
        stop = control.close()
        self.assertFalse(control.is_alive())
        self.assertTrue(stop['explicit_zero'])
        self.assertTrue(control.receipts()[ticket]['lease_interrupted_by_close'])
        self.assertLess(control.receipts()[ticket]['clear_after_declared_lease_s'], 0.)
        self.assertTrue(holder['policy'].closed)
        self.assertEqual(holder['flight'].last_xy, [0., 0.])

    def test_stuck_velocity_prevents_false_shutdown_success(self):
        control, holder = self.create()
        holder['flight'].velocity = [.02, 0., 0.]
        holder['flight'].stuck_velocity = True
        with self.assertRaisesRegex(RuntimeError, 'NO_VERIFIED_SAMPLED_STOP'):
            control.close(timeout=1.5)
        self.assertIsNone(control.scheduling_receipt()['shutdown_stop'])

    def test_slow_controller_rpc_reports_measured_gap_without_hard_realtime_claim(self):
        control, holder = self.create()
        holder['flight'].read_delay_s = .08
        time.sleep(.2)
        metrics = control.scheduling_receipt()
        self.assertGreater(metrics['maximum_observed_query_gap_s'], .075)
        self.assertFalse(metrics['hard_real_time_established'])
        self.assertTrue(metrics['query_gap_not_a_continuous_safety_guarantee'])

    def test_pending_mailbox_replacement_is_explicit(self):
        control, holder = self.create()
        blocker = threading.Event()
        holder['flight'].read_entered.clear()
        holder['flight'].read_release = blocker
        with control._condition:
            control._condition.notify_all()
        self.wait_for(holder['flight'].read_entered.is_set)
        first = self.submit(control, self.frame('frame-1'))
        second = self.submit(control, self.frame('frame-2'))
        overwritten = control.await_decision(first)
        self.assertEqual(overwritten['motion_gate']['reason'], 'PENDING_PROPOSAL_REPLACED')
        self.assertIsNone(overwritten['command'])
        blocker.set()
        self.assertEqual(control.await_decision(second)['motion_gate']['status'], 'ALLOW_BOUNDED_COMMAND')

    def test_real_policy_rejects_stale_bound_semantic_frame(self):
        control, holder = self.create(real_policy=True)
        now = time.monotonic()
        frame, selection, surface = bound_fixture(receipt=now-.7, ready=now-.6)
        selection['semantic_review_deadline_monotonic_s'] = now+8.
        rebind_surface(selection, surface)
        ticket = control.submit(frame, selection, surface, [.01, 0.])
        decision = control.await_decision(ticket)
        self.assertEqual(decision['motion_gate']['reason'], 'CURRENT_FRAME_EXPIRED')
        self.assertFalse(any(x['purpose'] == 'CURRENT_TARGET_TRAJECTORY' for x in holder['flight'].commands))

    def test_manual_clock_lease_clear_is_not_assumed_automatic(self):
        clock = ManualClock()
        control, holder = self.create(clock=clock, period_s=.05)
        ticket = control.submit(self.frame(receipt=clock()), {'mutable': 'original'}, {}, [.01, 0.])
        decision = control.await_decision(ticket)
        self.assertEqual(decision['command']['velocity_xy_m_s'], [.01, 0.])
        # Advancing beyond the declared lease must produce an explicit zero.
        clock.advance(.12)
        with control._condition:
            control._condition.notify_all()
        self.wait_for(lambda: any(x['purpose'] == 'OWN_STATE_BRAKE_OR_HOLD' and
                                 x['dispatch_monotonic_s'] >= 10.12 and
                                 x['velocity_xy_m_s'] == [0., 0.] for x in holder['flight'].commands))
        for _ in range(3):
            clock.advance(.05)
            with control._condition:
                control._condition.notify_all()
            time.sleep(.01)
        self.assertIn(ticket, control.receipts())
        # Shutdown itself also requires distinct, post-zero measured times.
        closer = threading.Thread(target=lambda: control.close(timeout=1.))
        closer.start()
        for _ in range(5):
            clock.advance(.05)
            with control._condition:
                control._condition.notify_all()
            time.sleep(.01)
        closer.join(1.)
        self.assertFalse(control.is_alive())
        times = [x['receipt_monotonic_s'] for x in control.receipts()[ticket]['samples']]
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))

    def test_period_lifecycle_and_unknown_ticket_validation(self):
        with self.assertRaises(ValueError):
            AsyncControl(lambda: None, lambda: None, period_s=.1)
        control, holder = self.create()
        with self.assertRaises(ValueError):
            control.await_decision('control-999')
        with self.assertRaises(RuntimeError):
            control.start()
        control.close()
        with self.assertRaises(RuntimeError):
            self.submit(control)

    def test_exception_categories_never_copy_unexpected_rpc_bodies(self):
        self.assertEqual(closed_category(RuntimeError('SELF_STATE_PROTECTION')), 'RuntimeError: SELF_STATE_PROTECTION')
        self.assertEqual(closed_category(RuntimeError('secret=abc https://provider.invalid/body')), 'RuntimeError')
        self.assertEqual(closed_category(RuntimeError('UPPERCASE_'+'X'*160)), 'RuntimeError')
        control, holder = self.create()

        def private_body(state):
            raise RuntimeError('key=PRIVATE_FAKE_TEST_VALUE; response body https://example.invalid')

        holder['flight'].protection = private_body
        self.wait_for(lambda: not control.is_alive())
        self.assertEqual(control.scheduling_receipt()['fatal'], 'RuntimeError; CLOSE: RuntimeError')
        with self.assertRaises(RuntimeError) as caught:
            control.close()
        self.assertNotIn('PRIVATE_FAKE_TEST_VALUE', str(caught.exception))

    def test_dispatched_decision_survives_later_fault_as_readonly_copy(self):
        control, holder = self.create()
        ticket = self.submit(control)
        decision = control.await_decision(ticket)
        holder['flight'].fail_state = True
        self.wait_for(lambda: not control.is_alive())
        salvaged = control.decisions()
        self.assertEqual(salvaged[ticket]['command'], decision['command'])
        salvaged[ticket]['command']['velocity_xy_m_s'][0] = 100.
        self.assertEqual(control.decisions()[ticket]['command']['velocity_xy_m_s'][0], .01)
        with self.assertRaisesRegex(RuntimeError, 'NO_VERIFIED_SAMPLED_STOP'):
            control.close()


if __name__ == '__main__':
    unittest.main()
