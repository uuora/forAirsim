"""Windows spawn control owner with bounded certificate-only IPC.

The child preserves first-clear clocks with V1P8 stopping and V10P2 deadline checks. All control events
are written inside the child, so a stalled/dead perception parent cannot
hold an evidence queue needed by the zero command. A latest-state mailbox
may skip publications, but never replaces the authoritative event journal.
No hard real-time guarantee or landing proof follows from process exit.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import threading
import time

PROPOSAL_BYTES = 65536
RESPONSE_BYTES = 131072
STATE_BYTES = 16384
MAX_TICKETS = 1024
RESPONSE_CAPACITY = 8
FINAL_BYTES = 64 * 1024 * 1024
BRIDGE_PERIOD_S = .01
MAX_RUN_S = 380.
FACTORIES = {'t1_process_control_factories_v1:make_airsim',
             't1_process_control_fixture_v1:make_fixture',
             't1_obstacle_control_factory:make_obstacle_airsim'}
CONFIG_KEYS = {'session_id', 'description', 'translation', 'world_z', 'low', 'high'}


def _bytes(value, maximum):
    # Serialize primitives before enqueue; multiprocessing's feeder sees bytes,
    # never an RPC client, future, mutable certificate or private configuration.
    data = json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')
    if len(data) > maximum:
        raise ValueError('PROCESS_MESSAGE_BYTE_LIMIT')
    return data


def _decode(data, maximum):
    if not isinstance(data, bytes) or len(data) > maximum:
        raise ValueError('PROCESS_MESSAGE_BYTE_LIMIT')
    return json.loads(data.decode('utf-8'))


def _category(error):
    from t1_async_control import closed_category
    return closed_category(error)


def _save_new(path, value):
    data = _bytes(value, FINAL_BYTES)
    with Path(path).open('xb') as stream:
        stream.write(data + b'\n')
        stream.flush()


def _publish(mailbox, value):
    data = _bytes(value, STATE_BYTES)
    buffer, length, lock = mailbox
    # The parent can die holding this lock. Never wait on it in the child.
    if not lock.acquire(False):
        return False
    try:
        buffer[:len(data)] = data
        length.value = len(data)
    finally:
        lock.release()
    return True


def _snapshot(mailbox):
    buffer, length, lock = mailbox
    if not lock.acquire(False):
        return None
    try:
        size = length.value
        return _decode(bytes(buffer[:size]), STATE_BYTES) if size else None
    finally:
        lock.release()


def _child(config, factory_name, folder, proposals, responses, mailbox, closing):
    # This entry point receives only JSON configuration and bounded channels.
    from t1_async_control_v1p8 import AsyncControlV1P8
    from t1_bounded_evidence_v1 import BoundedEventJournal

    parent = mp.parent_process()
    journal = controller = None
    ipc_fault = threading.Event()
    ownership = {}
    reason = 'NOT_STARTED'
    detected = None
    final = dict(status='RUNNING', pid=os.getpid(), parent_pid=parent.pid,
                 spawn_method=mp.get_start_method(), hard_real_time_established=False,
                 safe_landing_verified=False, total_proposals=0, publication_skips=0,
                 errors=[], shutdown_stop=None, decisions={}, receipts={})

    def log(row):
        if threading.get_ident() != ownership.get('thread_id'):
            raise RuntimeError('CONTROL_EVENT_RPC_OWNER_CHANGED')
        event = dict(row, rpc_owner_pid=os.getpid(), rpc_owner_thread_id=threading.get_ident())
        journal.submit(event)
        kind = row.get('kind')
        if kind not in ('CONTROL_DECISION', 'TARGET_LEASE_STOP') or ipc_fault.is_set():
            return
        try:
            responses.put_nowait(_bytes(dict(kind=kind, value=row), RESPONSE_BYTES))
        except (queue.Full, ValueError):
            # The first failure stops control. Subsequent emergency proof rows
            # remain in the local journal instead of repeatedly failing IPC.
            ipc_fault.set()
            raise RuntimeError('PROCESS_RESPONSE_CHANNEL_OVERLOAD')

    try:
        journal = BoundedEventJournal(Path(folder) / 'execution.jsonl')
        module, name = factory_name.split(':')
        factory = getattr(importlib.import_module(module), name)

        def make_flight():
            ownership.update(pid=os.getpid(), thread_id=threading.get_ident())
            flight, policy = factory(config, log)
            ownership['policy'] = policy
            journal.submit(dict(kind='PROCESS_RPC_OWNER_CREATED',
                recorded_monotonic_s=time.monotonic(), pid=os.getpid(),
                thread_id=threading.get_ident(), parent_pid=parent.pid))
            return flight

        controller = AsyncControlV1P8(make_flight, lambda: ownership['policy'],
            period_s=.05, idle_zero_refresh_s=.30)
        controller.start(timeout=2.)
        runtime_deadline = time.monotonic()+MAX_RUN_S
        final['status'] = 'READY'
        _publish(mailbox, dict(status='READY', pid=os.getpid(),
            state=controller.latest_state(), settled=False, published_monotonic_s=time.monotonic()))
        while True:
            # The spawn Process handle guards against PID reuse. This detects
            # actual parent termination; a blocked but live parent is allowed.
            if parent is None or not parent.is_alive():
                reason = 'PARENT_PROCESS_EXITED'
                break
            if closing.is_set() or (config.get('stop_file') and Path(config['stop_file']).exists()):
                reason = 'PARENT_REQUESTED_CLOSE'
                break
            if ipc_fault.is_set():
                reason = 'PROCESS_RESPONSE_CHANNEL_OVERLOAD'
                break
            if time.monotonic() >= runtime_deadline:
                reason = 'PROCESS_ABSOLUTE_RUNTIME_LIMIT'
                final['errors'].append(reason)
                break
            controller.check()
            journal.check()
            try:
                data = proposals.get_nowait()
            except queue.Empty:
                data = None
            if data is not None:
                received = time.monotonic()
                proposal = _decode(data, PROPOSAL_BYTES)
                if (set(proposal) != {'frame', 'selection', 'surface', 'desired_xy_m_s',
                        'ticket_id', 'parent_enqueued_monotonic_s'}
                        or final['total_proposals'] >= MAX_TICKETS):
                    raise ValueError('PROCESS_PROPOSAL_CONTRACT_MISMATCH')
                expected = 'control-' + str(final['total_proposals'] + 1)
                if proposal['ticket_id'] != expected:
                    raise ValueError('PROCESS_TICKET_SEQUENCE_MISMATCH')
                enqueued = proposal['parent_enqueued_monotonic_s']
                if (isinstance(enqueued, bool) or not isinstance(enqueued, (int, float))
                        or not math.isfinite(enqueued) or enqueued > time.monotonic()):
                    raise ValueError('PROCESS_FUTURE_ENQUEUE_CLOCK')
                from t1_process_control_factories_v1 import certificate_frame
                frame = certificate_frame(proposal['frame'])
                journal.submit(dict(kind='PROCESS_PROPOSAL_ACCEPTED', ticket_id=expected,
                    parent_enqueued_monotonic_s=enqueued, child_received_monotonic_s=received,
                    child_admitted_monotonic_s=time.monotonic(), frame_id=frame.frame_id,
                    source_receipt_monotonic_s=frame.readback_receipt_monotonic_s,
                    proposal_sha256=hashlib.sha256(data).hexdigest(),
                    source_clock_rebased=False, pid=os.getpid()))
                ticket = controller.submit(frame, proposal['selection'],
                    proposal['surface'], proposal['desired_xy_m_s'])
                if ticket != expected:
                    raise RuntimeError('PROCESS_CONTROLLER_TICKET_MISMATCH')
                final['total_proposals'] += 1
            with controller._condition:
                # Small fixed-size snapshot; never copy growing histories here.
                state = copy.deepcopy(controller._latest)
                settled = bool(controller._settled and controller._active is None
                               and controller._pending is None)
                decided_total = len(controller._decisions)
            published = _publish(mailbox, dict(status='READY', pid=os.getpid(), state=state,
                settled=settled, accepted_total=final['total_proposals'], decided_total=decided_total,
                published_monotonic_s=time.monotonic()))
            final['publication_skips'] += int(not published)
            closing.wait(BRIDGE_PERIOD_S)
    except BaseException as error:
        reason = _category(error)
        final['errors'].append(reason)
    finally:
        detected = time.monotonic()
        if journal:
            try:
                journal.submit(dict(kind='PROCESS_CLOSE_DETECTED', reason=reason,
                    detected_monotonic_s=detected, pid=os.getpid(), monitor_period_s=BRIDGE_PERIOD_S,
                    continuous_parent_monitoring_guaranteed=False))
            except BaseException as error:
                final['errors'].append(_category(error))
        if controller:
            try:
                controller.close(timeout=3.)
            except BaseException as error:
                final['errors'].append(_category(error))
                # Never terminate an RPC owner or let the parent land concurrently.
                while controller.is_alive():
                    try:
                        controller.close(timeout=1.)
                    except BaseException:
                        pass
            final['decisions'] = controller.decisions()
            final['receipts'] = controller.receipts()
            final['scheduling'] = controller.scheduling_receipt()
            final['shutdown_stop'] = final['scheduling'].get('shutdown_stop')
        ownership.pop('policy', None)
        final.update(ownership=ownership, close_reason=reason,
            public_config=config,
            maximum_active_runtime_s=MAX_RUN_S,
            close_detection_scope='bridge observation; owner may have already entered failure shutdown',
            close_detected_monotonic_s=detected, completed_monotonic_s=time.monotonic())
        if journal:
            try:
                final['journal'] = journal.close(timeout=5.)
            except BaseException as error:
                final['errors'].append(_category(error))
                final['journal'] = journal.receipt()
        final['sampled_stop_verified'] = bool(final['shutdown_stop'] and
            final['shutdown_stop'].get('status') == 'OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE')
        final['status'] = 'CLOSED' if final['sampled_stop_verified'] and not final['errors'] else 'FAILED'
        final['receipt_persistence_scope'] = 'Python write and flush; not fsync or power-loss durability'
        try:
            _save_new(Path(folder) / 'final.json', final)
        except BaseException:
            # Exit without a usable final receipt stays unknown to the parent.
            final['status'] = 'FAILED_FINAL_RECEIPT_WRITE'
        _publish(mailbox, dict(status=final['status'], pid=os.getpid(),
            fatal=reason if final['errors'] else None, published_monotonic_s=time.monotonic()))
        # Feeder thread must not hold child exit when the parent is dead/stalled.
        # Exact decisions/receipts are already in final.json and the journal.
        responses.cancel_join_thread()
        responses.close()
        proposals.close()


class ProcessControlV1P1:
    """Parent facade. It has no simulator object or direct motion authority."""
    def __init__(self, config, output, *, factory='t1_process_control_factories_v1:make_airsim'):
        if factory not in FACTORIES or set(config) - {'stop_file','obstacle_scene'} != CONFIG_KEYS:
            raise ValueError('PUBLIC_GEOMETRY_CONTROL_CONFIG_REQUIRED')
        if ('obstacle_scene' in config)!=(factory=='t1_obstacle_control_factory:make_obstacle_airsim'):
            raise ValueError('OBSTACLE_SCENE_AND_FACTORY_MUST_MATCH')
        if 'obstacle_scene' in config:
            from t1_obstacle_scene import validate_scene
            validate_scene(config['obstacle_scene'])
        if (not isinstance(config['session_id'], str)
                or not re.fullmatch(r'[A-Za-z0-9_-]{1,256}', config['session_id'])
                or config['description'] != '带白色横条的红色气球'):
            raise ValueError('PUBLIC_SESSION_AND_T1_DESCRIPTION_REQUIRED')
        for key in ('translation', 'low', 'high'):
            if (not isinstance(config[key], (list, tuple)) or len(config[key]) != 3
                    or any(isinstance(v, bool) or not isinstance(v, (int, float))
                           or not math.isfinite(v) for v in config[key])):
                raise ValueError('FINITE_THREE_VECTOR_GEOMETRY_REQUIRED')
        z = config['world_z']
        if (isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z)
                or any(a >= b for a, b in zip(config['low'], config['high']))
                or not config['low'][2] < z < config['high'][2]):
            raise ValueError('VALID_FIXED_HEIGHT_WORKSPACE_REQUIRED')
        if config.get('stop_file'):
            stop_path=Path(config['stop_file']).resolve()
            logs_root=Path(__file__).resolve().parents[1]/'logs'
            if not stop_path.is_relative_to(logs_root) or stop_path.name!='manual_stop.request':
                raise ValueError('OWNED_MANUAL_STOP_PATH_REQUIRED')
        self.config = _decode(_bytes(config, PROPOSAL_BYTES), PROPOSAL_BYTES)
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.factory = factory
        self.ctx = mp.get_context('spawn')
        self.proposals = self.ctx.Queue(maxsize=1)
        self.responses = self.ctx.Queue(maxsize=RESPONSE_CAPACITY)
        self.mailbox = (self.ctx.RawArray('B', STATE_BYTES), self.ctx.RawValue('I', 0), self.ctx.Lock())
        self.closing = self.ctx.Event()
        self._lock = threading.Lock()
        self._process = self._receiver = None
        self._receiver_stop = threading.Event()
        self._decisions, self._receipts = {}, {}
        self._tickets, self._outstanding = 0, None
        self._receive_error = None
        self._final = None
        self._final_audit = None
        self._cached = None

    def start(self, timeout=15.):
        self._positive_timeout(timeout)
        if self._process is not None:
            raise RuntimeError('PROCESS_CONTROL_ALREADY_STARTED')
        self._process = self.ctx.Process(target=_child, name='T1-process-control', daemon=False,
            args=(self.config, self.factory, str(self.output), self.proposals, self.responses,
                  self.mailbox, self.closing))
        self._process.start()
        self._receiver = threading.Thread(target=self._receive, name='T1-process-receipts', daemon=True)
        self._receiver.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self._status()
            if value and value.get('status') == 'READY':
                self.check()
                return self
            if not self.is_alive():
                self._load_final()
                raise RuntimeError('PROCESS_CONTROL_START_FAILED')
            time.sleep(.01)
        self.closing.set()
        raise TimeoutError('PROCESS_CONTROL_START_TIMEOUT_OWNER_MAY_REMAIN_ACTIVE')

    @staticmethod
    def _positive_timeout(timeout):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('POSITIVE_FINITE_TIMEOUT_REQUIRED')

    def _status(self):
        value = _snapshot(self.mailbox)
        if value is not None:
            self._cached = value
        return self._cached

    def _receive(self):
        while not self._receiver_stop.is_set():
            try:
                message = _decode(self.responses.get(timeout=.02), RESPONSE_BYTES)
                row = message['value']
                ticket = row['ticket_id']
                with self._lock:
                    if message['kind'] == 'CONTROL_DECISION':
                        if ticket in self._decisions:
                            raise ValueError('DUPLICATE_PROCESS_DECISION')
                        self._decisions[ticket] = row
                        if self._outstanding == ticket:
                            self._outstanding = None
                    elif message['kind'] == 'TARGET_LEASE_STOP':
                        if ticket in self._receipts:
                            raise ValueError('DUPLICATE_PROCESS_STOP_RECEIPT')
                        self._receipts[ticket] = row
                    else:
                        raise ValueError('UNKNOWN_PROCESS_RESPONSE')
                    if len(self._decisions) > MAX_TICKETS or len(self._receipts) > MAX_TICKETS:
                        raise ValueError('PROCESS_RECEIPT_CAPACITY')
            except queue.Empty:
                continue
            except BaseException as error:
                self._receive_error = _category(error)
                self.closing.set()
                return

    def check(self):
        if self._process is None:
            raise RuntimeError('PROCESS_CONTROL_NOT_STARTED')
        if self._receive_error:
            raise RuntimeError('PROCESS_RECEIVER_FAILED')
        value = self._status()
        if value and value.get('status') not in ('READY', 'RUNNING'):
            raise RuntimeError('PROCESS_CONTROL_FATAL')
        if not self.is_alive():
            raise RuntimeError('PROCESS_CONTROL_EXITED_NO_ACTIVE_AUTHORITY')

    def latest_state(self):
        self.check()
        state = self._status().get('state')
        if state is None:
            raise RuntimeError('PROCESS_STATE_NOT_READY')
        # Consumers must retain the actual receipt; never restamp IPC arrival.
        return copy.deepcopy(state)

    def submit(self, frame, selection, surface, desired_xy_m_s):
        self.check()
        from t1_semantic_perception import SemanticFrame
        if not isinstance(frame, SemanticFrame) or frame.encoded_image:
            raise ValueError('CERTIFICATE_ONLY_SEMANTIC_FRAME_REQUIRED')
        frozen_frame = asdict(frame)
        # Image bytes and detector payload are excluded; all identity bindings,
        # frame clocks and candidates remain unchanged.
        frozen_frame.pop('encoded_image')
        frozen_frame.pop('detector_report_json')
        with self._lock:
            if self.closing.is_set():
                raise RuntimeError('PROCESS_CONTROL_CLOSING')
            if self._outstanding is not None:
                raise RuntimeError('PROCESS_PROPOSAL_BUFFER_FULL')
            if self._tickets >= MAX_TICKETS:
                raise RuntimeError('PROCESS_TICKET_CAPACITY')
            ticket = 'control-' + str(self._tickets + 1)
            data = _bytes(dict(frame=frozen_frame, selection=selection, surface=surface,
                desired_xy_m_s=desired_xy_m_s, ticket_id=ticket,
                parent_enqueued_monotonic_s=time.monotonic()), PROPOSAL_BYTES)
            try:
                self.proposals.put_nowait(data)
            except queue.Full:
                raise RuntimeError('PROCESS_PROPOSAL_BUFFER_FULL') from None
            self._tickets += 1
            self._outstanding = ticket
        return ticket

    def await_decision(self, ticket, timeout=.5):
        self._positive_timeout(timeout)
        if not isinstance(ticket, str) or not ticket.startswith('control-') or not ticket[8:].isdigit() or not 1 <= int(ticket[8:]) <= self._tickets:
            raise ValueError('UNKNOWN_CONTROL_TICKET')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if ticket in self._decisions:
                    return copy.deepcopy(self._decisions[ticket])
            self.check()
            time.sleep(.002)
        raise TimeoutError('PROCESS_CONTROL_DECISION_TIMEOUT')

    def wait_until_settled(self, timeout=2.5):
        self._positive_timeout(timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check()
            value = self._status()
            with self._lock:
                outstanding = self._outstanding
            if (value.get('settled') and outstanding is None
                    and value.get('accepted_total') == self._tickets
                    and value.get('decided_total') == self._tickets):
                return copy.deepcopy(value['state'])
            time.sleep(.005)
        raise TimeoutError('PROCESS_CONTROL_SETTLEMENT_TIMEOUT')

    def is_alive(self):
        return self._process is not None and self._process.is_alive()

    def _load_final(self):
        if self.is_alive() or self._final is not None:
            return
        path = self.output / 'final.json'
        if path.is_file() and path.stat().st_size <= FINAL_BYTES + 1:
            self._final = json.loads(path.read_text(encoding='utf-8'))
            from audit_t1_process_control_v1p1 import audit
            self._final_audit = audit(self.output)
            with self._lock:
                self._decisions.update(self._final.get('decisions', {}))
                self._receipts.update(self._final.get('receipts', {}))

    def receipts(self):
        self._load_final()
        with self._lock:
            return copy.deepcopy(self._receipts)

    def decisions(self):
        self._load_final()
        with self._lock:
            return copy.deepcopy(self._decisions)

    def scheduling_receipt(self):
        self._load_final()
        if self._final is None:
            return dict(status='NO_FINAL_PROCESS_RECEIPT', owner_alive=self.is_alive(),
                shutdown_stop=None, hard_real_time_established=False)
        return copy.deepcopy(self._final)

    def can_transfer_rpc(self):
        self._load_final()
        return bool(not self.is_alive() and self._final and self._final.get('sampled_stop_verified')
            and self._final['pid'] == self._process.pid
            and self._final_audit and self._final_audit['status'] == 'PASS_PROCESS_CONTROL_EVIDENCE')

    def close(self, timeout=3.):
        self._positive_timeout(timeout)
        if self._process is None:
            raise RuntimeError('PROCESS_CONTROL_NOT_STARTED')
        self.closing.set()
        self._process.join(timeout)
        if self.is_alive():
            raise TimeoutError('PROCESS_JOIN_TIMEOUT_RPC_OWNER_MAY_REMAIN_ACTIVE')
        self._receiver_stop.set()
        if self._receiver:
            self._receiver.join(.5)
        self._load_final()
        if not self.can_transfer_rpc():
            raise RuntimeError('PROCESS_CLOSE_NO_VERIFIED_STOP_PRESERVE_SIM')
        if self._final['status'] != 'CLOSED':
            raise RuntimeError('PROCESS_FATAL_AFTER_VERIFIED_SAMPLED_CLOSE')
        return copy.deepcopy(self._final['shutdown_stop'])
