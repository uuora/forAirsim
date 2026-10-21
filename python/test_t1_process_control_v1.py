"""Actual spawn tests using production gates and an owner-checking RPC fixture."""
import copy
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

from t1_process_control_v1 import ProcessControlV1, PROPOSAL_BYTES, MAX_TICKETS
from test_t1_identity_motion_policy import DESCRIPTION, bound_fixture, rebind_surface
from audit_t1_feedback_dispatch_v1 import audit as dispatch_audit
from audit_t1_process_control_v1 import audit as process_audit


def config(session='session'):
    return dict(session_id=session, description=DESCRIPTION, translation=[0., 0., 0.],
                world_z=-3., low=[-.3, -.3, -3.15], high=[.3, .3, -2.85])


def proposal(sequence=1, age=.03):
    now = time.monotonic()
    frame, selection, surface = bound_fixture(frame_id='frame-'+str(sequence), sequence=sequence,
        receipt=now-age, ready=now)
    selection['semantic_review_deadline_monotonic_s'] = now + 8.
    rebind_surface(selection, surface)
    return frame, selection, surface


def abrupt_parent(output):
    control = ProcessControlV1(config(), Path(output) / 'control',
                              factory='t1_process_control_fixture_v1:make_fixture')
    control.start()
    time.sleep(.36)
    ticket = control.submit(*proposal(), [.009, 0.])
    decision = control.await_decision(ticket)
    (Path(output) / 'parent.json').write_text(json.dumps(dict(pid=os.getpid(),
        child_pid=control._process.pid, ticket=ticket, command=decision.get('command'),
        exit_requested_monotonic_s=time.monotonic())), encoding='utf-8')
    # This owned test process skips atexit child joins to model abrupt loss.
    os._exit(31)


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='t1-spawn-')
        self.folder = Path(self.temp.name)
        self.controls = []

    def tearDown(self):
        for c in self.controls:
            if c._process is not None:
                try:
                    c.close(timeout=6.)
                except (RuntimeError, TimeoutError):
                    self.assertFalse(c.is_alive(), 'fixture must leave no active RPC owner')
                c.proposals.cancel_join_thread()
                c.proposals.close()
                c.responses.close()
        self.temp.cleanup()

    def create(self, mode='session'):
        c = ProcessControlV1(config(mode), self.folder / ('control-'+str(len(self.controls))),
                            factory='t1_process_control_fixture_v1:make_fixture')
        self.controls.append(c)
        c.start()
        return c

    @staticmethod
    def events(c):
        return [json.loads(s) for s in (c.output / 'execution.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_actual_spawn_rpc_callbacks_and_state_have_one_child_owner(self):
        c = self.create()
        c.submit(*proposal(), [0., 0.])
        c.wait_until_settled()
        c.close()
        receipt = c.scheduling_receipt()
        self.assertEqual(receipt['spawn_method'], 'spawn')
        self.assertNotEqual(receipt['pid'], os.getpid())
        self.assertEqual(receipt['ownership']['pid'], receipt['pid'])
        events = self.events(c)
        owned = [r for r in events if 'rpc_owner_pid' in r]
        self.assertTrue(any(r['kind']=='COMMAND_RPC_RESULT' for r in owned))
        self.assertEqual({(r['rpc_owner_pid'], r['rpc_owner_thread_id']) for r in owned},
                         {(receipt['pid'], receipt['ownership']['thread_id'])})
        self.assertTrue(c.can_transfer_rpc())
        self.assertFalse(receipt['safe_landing_verified'])

    def test_parent_cpu_block_cannot_renew_target_and_saved_proofs_pass(self):
        c = self.create()
        time.sleep(.36)
        ticket = c.submit(*proposal(), [.009, 0.])
        command = c.await_decision(ticket)['command']
        self.assertEqual(command['purpose'], 'CURRENT_TARGET_TRAJECTORY')
        # Hold the parent interpreter's GIL for the entire injection window.
        import sys
        previous = sys.getswitchinterval()
        sys.setswitchinterval(1.)
        try:
            deadline = time.monotonic()+.65
            while time.monotonic() < deadline:
                pass
        finally:
            sys.setswitchinterval(previous)
        c.wait_until_settled()
        c.close()
        events = self.events(c)
        targets = [e for e in events if e.get('purpose')=='CURRENT_TARGET_TRAJECTORY' and e['kind']=='COMMAND']
        self.assertEqual(len(targets), 1)
        proof = c.receipts()[ticket]
        self.assertFalse(proof['target_lease_renewed'])
        self.assertLessEqual(proof['completed_monotonic_s'], command['dispatch_monotonic_s']+.12+2.)
        self.assertLessEqual(proof['target_setpoint_cleared_monotonic_s'],
                             command['target_end_deadline_monotonic_s'])
        self.assertGreaterEqual(proof['distinct_stable_simulator_state_timestamps'], 3)
        checked = dispatch_audit({}, events)
        self.assertTrue(checked['status'].startswith('PASS'), checked)

    def test_mailbox_lock_cannot_hold_child_zero_or_shutdown(self):
        c = self.create()
        time.sleep(.36)
        ticket = c.submit(*proposal(), [.009, 0.])
        command = c.await_decision(ticket)['command']
        lock = c.mailbox[2]
        lock.acquire()
        try:
            time.sleep(.65)
            c.close()
        finally:
            lock.release()
        self.assertTrue(c.can_transfer_rpc())
        self.assertGreater(c.scheduling_receipt()['publication_skips'], 0)
        self.assertLessEqual(c.receipts()[ticket]['target_setpoint_cleared_monotonic_s'],
                             command['target_end_deadline_monotonic_s'])

    def test_expired_source_is_rejected_with_original_clock_after_ipc(self):
        c = self.create()
        time.sleep(.36)
        frame, selection, surface = proposal(age=.7)
        ticket = c.submit(frame, selection, surface, [.009, 0.])
        decision = c.await_decision(ticket)
        self.assertFalse(decision['motion_gate']['target_command_authorized'])
        c.close()
        accepted = next(e for e in self.events(c) if e['kind']=='PROCESS_PROPOSAL_ACCEPTED')
        self.assertEqual(accepted['source_receipt_monotonic_s'], frame.readback_receipt_monotonic_s)
        self.assertFalse(accepted['source_clock_rebased'])
        self.assertFalse(any(e.get('purpose')=='CURRENT_TARGET_TRAJECTORY' for e in self.events(c)))

    def test_message_size_and_private_config_rejected_before_enqueue(self):
        with self.assertRaisesRegex(ValueError, 'PUBLIC_GEOMETRY'):
            ProcessControlV1(dict(config(), api_key='never sent'), self.folder/'private')
        c = self.create()
        frame, selection, surface = proposal()
        selection['too_large'] = 'x'*PROPOSAL_BYTES
        with self.assertRaisesRegex(ValueError, 'BYTE_LIMIT'):
            c.submit(frame, selection, surface, [.009, 0.])
        self.assertEqual(c._tickets, 0)
        with self.assertRaisesRegex(ValueError, 'CERTIFICATE_ONLY'):
            c.submit({}, {}, {}, [0.,0.])

    def test_proposal_capacity_rejects_explicitly_without_new_lease(self):
        c = self.create()
        with c._lock:
            c._outstanding = 'control-pending'
        with self.assertRaisesRegex(RuntimeError, 'BUFFER_FULL'):
            c.submit(*proposal(), [.009, 0.])
        with c._lock:
            c._outstanding = None
            c._tickets = MAX_TICKETS
        with self.assertRaisesRegex(RuntimeError, 'TICKET_CAPACITY'):
            c.submit(*proposal(), [.009, 0.])
        c.close()
        self.assertFalse(any(e.get('purpose')=='CURRENT_TARGET_TRAJECTORY' for e in self.events(c)))

    def test_response_overload_stops_and_keeps_local_command_and_proof(self):
        c = self.create()
        time.sleep(.36)
        c._receiver_stop.set()
        c._receiver.join(.5)
        for _ in range(8):
            c.responses.put_nowait(b'{}')
        c.submit(*proposal(), [.009, 0.])
        deadline = time.monotonic()+5.
        while c.is_alive() and time.monotonic()<deadline:
            time.sleep(.02)
        self.assertFalse(c.is_alive())
        with self.assertRaisesRegex(RuntimeError, 'FATAL_AFTER'):
            c.close()
        receipt = c.scheduling_receipt()
        self.assertTrue(receipt['sampled_stop_verified'])
        self.assertTrue(c.can_transfer_rpc())
        self.assertEqual(receipt['close_reason'], 'PROCESS_RESPONSE_CHANNEL_OVERLOAD')
        self.assertEqual(len(c.decisions()), 1)
        self.assertTrue(any(e['kind']=='CONTROLLER_SHUTDOWN_STOP' for e in self.events(c)))

    def test_rpc_failure_never_turns_process_exit_into_transfer_or_landing_proof(self):
        c = ProcessControlV1(config('FAIL_RPC'), self.folder/'failed',
                            factory='t1_process_control_fixture_v1:make_fixture')
        self.controls.append(c)
        with self.assertRaisesRegex(RuntimeError, 'START_FAILED'):
            c.start()
        with self.assertRaisesRegex(RuntimeError, 'NO_VERIFIED_STOP'):
            c.close()
        self.assertFalse(c.can_transfer_rpc())
        receipt = c.scheduling_receipt()
        self.assertFalse(receipt['sampled_stop_verified'])
        self.assertFalse(receipt['safe_landing_verified'])
        self.assertNotIn('private RPC body', json.dumps(receipt))

    def test_input_mutation_after_submit_cannot_change_child_certificate(self):
        c = self.create()
        time.sleep(.36)
        frame, selection, surface = proposal()
        expected = copy.deepcopy(selection)
        ticket = c.submit(frame, selection, surface, [.009, 0.])
        selection.clear()
        surface.clear()
        decision = c.await_decision(ticket)
        self.assertEqual(decision['motion_gate']['current_binding'], expected['current_binding'])
        self.assertTrue(decision['motion_gate']['target_command_authorized'])

    def test_settlement_read_cannot_use_snapshot_from_before_proposal(self):
        c = self.create()
        time.sleep(.36)
        ticket = c.submit(*proposal(), [.009, 0.])
        c.await_decision(ticket)
        state = c.wait_until_settled()
        self.assertEqual(c._status()['accepted_total'], 1)
        self.assertEqual(c._status()['decided_total'], 1)
        self.assertIn(ticket, c.receipts())
        self.assertLessEqual(sum(abs(v) for v in state['velocity_ned_m_s']), .0005)

    def test_changed_event_bytes_are_rejected_by_independent_audit(self):
        c = self.create()
        c.close()
        path = c.output/'execution.jsonl'
        original = path.read_bytes()
        try:
            path.write_bytes(original.replace(b'PERCEPTION_XY', b'ALTERED_STAGE', 1))
            self.assertEqual(process_audit(c.output)['status'], 'FAIL_PROCESS_CONTROL_EVIDENCE')
        finally:
            path.write_bytes(original)

    def test_false_stop_boolean_cannot_override_missing_actual_proof(self):
        c = self.create()
        c.close()
        path = c.output/'final.json'
        original = path.read_bytes()
        value = json.loads(original)
        value['shutdown_stop']['samples'] = []
        value['sampled_stop_verified'] = True
        try:
            path.write_text(json.dumps(value), encoding='utf-8')
            self.assertEqual(process_audit(c.output)['status'], 'FAIL_PROCESS_CONTROL_EVIDENCE')
        finally:
            path.write_bytes(original)

    def test_post_close_trace_replay_preserves_actual_state_and_command_binding(self):
        from t1_process_trace_v1 import replay_closed_trace
        from t1_bounded_evidence_v1 import BoundedEventJournal
        from audit_t1_feedback_stop_v1p1 import feedback_audit
        c=self.create()
        time.sleep(.36)
        c.await_decision(c.submit(*proposal(),[.009,0.]))
        c.wait_until_settled();c.close()
        target=self.folder/'parent'
        target.mkdir()
        journal=BoundedEventJournal(target/'execution.jsonl')
        journal.submit(dict(kind='PARENT_LIFECYCLE_END',stage='LANDING_RECOVERY',clock=time.monotonic()))
        try:
            receipt=replay_closed_trace(c,journal)
        finally:
            journal.close()
        events=[json.loads(s) for s in (target/'execution.jsonl').read_text(encoding='utf-8').splitlines()]
        original=self.events(c)
        replayed=events[1:]
        self.assertEqual(receipt['events'],len(original))
        for before,after in zip(original,replayed):
            for key,value in before.items():
                renamed={'journal_sequence':'child_journal_sequence',
                         'journal_enqueued_monotonic_s':'child_journal_enqueued_monotonic_s'}.get(key,key)
                self.assertEqual(value,after[renamed])
        checked=feedback_audit({},dict(low_speed_workspace=dict(low_ned_m=config()['low'],
                    high_ned_m=config()['high'])),events)
        self.assertTrue(checked['status'].startswith('PASS'),checked)

    def test_actual_parent_process_exit_triggers_owned_zero_and_sampled_stop(self):
        output = self.folder/'orphan'
        output.mkdir()
        parent = mp.get_context('spawn').Process(target=abrupt_parent, args=(str(output),))
        parent.start()
        parent.join(15.)
        self.assertFalse(parent.is_alive())
        self.assertEqual(parent.exitcode, 31)
        final_path = output/'control/final.json'
        deadline = time.monotonic()+8.
        while not final_path.exists() and time.monotonic()<deadline:
            time.sleep(.02)
        self.assertTrue(final_path.exists())
        final = json.loads(final_path.read_text(encoding='utf-8'))
        self.assertEqual(final['close_reason'], 'PARENT_PROCESS_EXITED')
        self.assertTrue(final['sampled_stop_verified'], final)
        self.assertFalse(final['safe_landing_verified'])
        events = [json.loads(s) for s in (output/'control/execution.jsonl').read_text(encoding='utf-8').splitlines()]
        detected = final['close_detected_monotonic_s']
        zero = [e for e in events if e['kind']=='COMMAND' and e.get('purpose')=='OWN_STATE_CONTROLLER_CLOSE_ZERO'][-1]
        self.assertGreaterEqual(zero['dispatch_monotonic_s'], detected)
        self.assertEqual(zero['velocity_xy_m_s'], [0.,0.])
        self.assertNotEqual(zero['rpc_owner_pid'], os.getpid())
        self.assertLessEqual(final['shutdown_stop']['completed_monotonic_s'], zero['dispatch_monotonic_s']+2.)


if __name__ == '__main__':
    unittest.main()
