"""Own-state controller isolated from blocking T1 perception and persistence.

The injected Flight and policy are constructed and used only by the worker.
Its receipts report measured host timing; a Python thread and a nominal
period do not establish a hard real-time or continuous safety guarantee.
"""
from __future__ import annotations

import copy
import math
import re
import threading
import time
from typing import Callable


def _xy_norm(value):
    return math.hypot(float(value[0]), float(value[1]))


def closed_category(error):
    """Keep bounded categorical diagnostics; never retain arbitrary RPC bodies."""
    name = type(error).__name__
    raw = str(error)
    if len(raw) <= 150 and re.fullmatch(r'[A-Z0-9_: ,.;-]+', raw):
        return name+': '+raw
    return name


class AsyncControl:
    """One pending proposal, one executed lease, fresh own-state dispatch.

    Busy proposals and overwritten pending proposals are explicitly rejected.
    They cannot renew a lease. ``await_decision`` returns at dispatch/rejection,
    while ``receipts`` supplies a later, separately measured stopping receipt.
    The worker never calls Flight.wait or Flight.complete_target_lease.
    """

    def __init__(self, flight_factory: Callable, policy_factory: Callable,
                 period_s=.05, clock=time.monotonic):
        if not callable(flight_factory) or not callable(policy_factory) or not callable(clock):
            raise ValueError('CONTROL_FACTORIES_AND_CLOCK_REQUIRED')
        if isinstance(period_s, bool) or not math.isfinite(period_s) or not 0 < period_s <= .05:
            raise ValueError('CONTROL_PERIOD_MUST_BE_POSITIVE_AND_AT_MOST_005')
        self.flight_factory, self.policy_factory = flight_factory, policy_factory
        self.period_s, self.clock = float(period_s), clock
        self._condition = threading.Condition()
        self._thread = None
        self._ready = False
        self._closing = False
        self._closed = False
        self._fatal = None
        self._latest = None
        self._pending = None
        self._active = None
        self._decisions = {}
        self._receipts = {}
        self._tickets = 0
        self._last_query = None
        self._gaps = []
        self._lateness = []
        self._shutdown_receipt = None

    def start(self, timeout=2.0):
        self._positive_timeout(timeout)
        with self._condition:
            if self._thread is not None:
                raise RuntimeError('CONTROL_ALREADY_STARTED')
            # Process exit must not silently discard a worker owning motion.
            self._thread = threading.Thread(target=self._run, name='T1-own-state-controller', daemon=False)
            self._thread.start()
            deadline = time.monotonic() + timeout
            while not self._ready and not self._fatal and not self._closed:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    self._closing = True
                    self._condition.notify_all()
                    raise TimeoutError('CONTROL_START_TIMEOUT')
                self._condition.wait(remaining)
        self.check()
        return self

    @staticmethod
    def _positive_timeout(timeout):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('POSITIVE_FINITE_TIMEOUT_REQUIRED')

    def check(self):
        with self._condition:
            if self._fatal:
                raise RuntimeError('ASYNC_CONTROL_FATAL: '+self._fatal)
            if self._thread is None:
                raise RuntimeError('CONTROL_NOT_STARTED')
            if self._closed and not self._closing:
                raise RuntimeError('CONTROL_UNEXPECTEDLY_STOPPED')

    def latest_state(self):
        self.check()
        with self._condition:
            if self._latest is None:
                raise RuntimeError('CONTROL_SELF_STATE_NOT_READY')
            return copy.deepcopy(self._latest)

    def submit(self, frame, selection, surface, desired_xy_m_s):
        # Freeze external mutable certificates before the mailbox takes ownership.
        frozen = copy.deepcopy(dict(frame=frame, selection=selection, surface=surface,
                                    desired_xy_m_s=desired_xy_m_s))
        self.check()
        with self._condition:
            if self._closing or self._closed:
                raise RuntimeError('CONTROL_CLOSING')
            self._tickets += 1
            ticket = 'control-'+str(self._tickets)
            proposal = dict(frozen, ticket_id=ticket, enqueued_monotonic_s=self.clock())
            if self._pending is not None:
                old = self._pending
                self._decisions[old['ticket_id']] = self._rejection(old, 'PENDING_PROPOSAL_REPLACED')
            self._pending = proposal
            self._condition.notify_all()
            return ticket

    def await_decision(self, ticket, timeout=.5):
        self._positive_timeout(timeout)
        deadline = time.monotonic()+timeout
        with self._condition:
            if ticket not in self._decisions and (not isinstance(ticket, str) or
                    not ticket.startswith('control-') or not ticket[8:].isdigit() or
                    not 1 <= int(ticket[8:]) <= self._tickets):
                raise ValueError('UNKNOWN_CONTROL_TICKET')
            while ticket not in self._decisions:
                if self._fatal:
                    raise RuntimeError('ASYNC_CONTROL_FATAL: '+self._fatal)
                if self._closed:
                    raise RuntimeError('CONTROL_CLOSED_WITHOUT_DECISION')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('CONTROL_DECISION_TIMEOUT')
                self._condition.wait(remaining)
            return copy.deepcopy(self._decisions[ticket])

    def receipts(self):
        with self._condition:
            return copy.deepcopy(self._receipts)

    def decisions(self):
        """Salvage already dispatched/rejected decisions after a later failure."""
        with self._condition:
            return copy.deepcopy(self._decisions)

    def is_alive(self):
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def scheduling_receipt(self):
        with self._condition:
            return dict(configured_period_s=self.period_s,
                        observed_query_gap_s=copy.deepcopy(self._gaps),
                        observed_tick_lateness_s=copy.deepcopy(self._lateness),
                        maximum_observed_query_gap_s=max(self._gaps, default=None),
                        maximum_observed_tick_lateness_s=max(self._lateness, default=None),
                        query_gap_not_a_continuous_safety_guarantee=True,
                        hard_real_time_established=False, fatal=self._fatal,
                        shutdown_stop=copy.deepcopy(self._shutdown_receipt))

    def close(self, timeout=2.0):
        """Join after explicit zero plus sampled stopping; fail visibly otherwise."""
        self._positive_timeout(timeout)
        with self._condition:
            if self._thread is None:
                raise RuntimeError('CONTROL_NOT_STARTED')
            self._closing = True
            self._condition.notify_all()
            thread = self._thread
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError('CONTROL_JOIN_TIMEOUT_WORKER_MAY_STILL_OWN_CLIENT')
        with self._condition:
            receipt = copy.deepcopy(self._shutdown_receipt)
            fatal = self._fatal
        if receipt is None or receipt.get('status') != 'OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE':
            raise RuntimeError('CONTROL_CLOSE_HAS_NO_VERIFIED_SAMPLED_STOP'+(': '+fatal if fatal else ''))
        if fatal:
            raise RuntimeError('ASYNC_CONTROL_FATAL_AFTER_SAMPLED_CLOSE: '+fatal)
        return receipt

    def _rejection(self, proposal, reason, state=None, **details):
        now = self.clock()
        return dict(ticket_id=proposal['ticket_id'], enqueued_monotonic_s=proposal['enqueued_monotonic_s'],
                    decision_completed_monotonic_s=now, command_self_state=copy.deepcopy(state),
                    motion_gate=dict(status='REJECT', reason=reason, decision_monotonic_s=now,
                                     target_command_authorized=False, flight_authorized=False,
                                     execution_guard_checked=False, current_binding=None, **details), command=None)

    def _log(self, flight, row):
        flight.log(copy.deepcopy(row))

    def _read(self, flight, *, scheduled=None):
        query_start = self.clock()
        state = flight.state()
        now = self.clock()
        # Reject malformed/future state before it enters a gate or shutdown proof.
        if (not isinstance(state, dict) or
                not math.isfinite(state.get('host_receipt_monotonic_s', float('nan'))) or
                state['host_receipt_monotonic_s'] > now+1e-9):
            raise RuntimeError('INVALID_CONTROLLER_SELF_STATE_RECEIPT')
        for key, count in (('position_world_ned_m', 3), ('velocity_ned_m_s', 3)):
            if key not in state or len(state[key]) != count or not all(math.isfinite(float(x)) for x in state[key]):
                raise RuntimeError('INVALID_CONTROLLER_SELF_STATE_VECTOR')
        reasons = list(flight.protection(state))
        state = dict(state, kind='SELF_STATE', stage=flight.stage,
                     protection_reasons=reasons, controller_query_started_monotonic_s=query_start,
                     controller_query_completed_monotonic_s=now,
                     controller_ticket_id=self._active['ticket_id'] if self._active else None)
        if scheduled is not None:
            state['controller_scheduled_tick_monotonic_s'] = scheduled
            state['controller_tick_lateness_s'] = max(0., query_start-scheduled)
        with self._condition:
            if self._last_query is not None:
                self._gaps.append(state['host_receipt_monotonic_s']-self._last_query)
            self._last_query = state['host_receipt_monotonic_s']
            if scheduled is not None:
                self._lateness.append(max(0., query_start-scheduled))
            self._latest = copy.deepcopy(state)
        self._log(flight, state)
        if reasons:
            flight.issue([0., 0.], .12, 'OWN_STATE_PROTECTION_ZERO')
            raise RuntimeError('SELF_STATE_PROTECTION: '+','.join(reasons))
        return state

    def _activate_stop(self, ticket, command, frame_id=None, *, already_cleared=None):
        now = self.clock()
        nonzero = _xy_norm(command['velocity_xy_m_s']) > 1e-12
        self._active = dict(ticket_id=ticket, command=copy.deepcopy(command), target_frame_id=frame_id,
                           declared_end=command['dispatch_monotonic_s']+command['duration_s'],
                           wait_until=command['dispatch_monotonic_s']+command['duration_s'] if nonzero else now,
                           clear=already_cleared, stable=None, samples=[], stop_deadline=None)

    def _stop_tick(self, flight, state):
        active = self._active
        if active is None or self.clock() < active['wait_until']:
            return
        now = self.clock()
        if active['stop_deadline'] is None:
            active['stop_deadline'] = now+1.0
        if now > active['stop_deadline']:
            raise RuntimeError('OWN_STATE_STOP_TIMEOUT')
        command = flight.brake_tick(state)
        if _xy_norm(command['velocity_xy_m_s']) < 1e-12 and active['clear'] is None:
            active['clear'] = command['dispatch_monotonic_s']
        receipt = state['host_receipt_monotonic_s']
        # A pre-clear measurement cannot establish post-clear stable stopping.
        stopped = (active['clear'] is not None and receipt >= active['clear'] and
                   _xy_norm(state['velocity_ned_m_s']) <= .005)
        if stopped:
            if active['stable'] is None:
                active['stable'] = receipt
        else:
            active['stable'] = None
        if not active['samples'] or receipt > active['samples'][-1]['receipt_monotonic_s']:
            active['samples'].append(dict(receipt_monotonic_s=receipt,
                                         velocity_xy_m_s=state['velocity_ned_m_s'][:2]))
        if active['stable'] is not None and receipt-active['stable'] >= .05-1e-9:
            result = dict(status='OWN_STATE_SAMPLED_STOPPED_WITH_INDEPENDENT_CONTROL',
                          ticket_id=active['ticket_id'], declared_lease_end_monotonic_s=active['declared_end'],
                          target_setpoint_cleared_monotonic_s=active['clear'],
                          clear_after_declared_lease_s=active['clear']-active['declared_end'],
                          completed_monotonic_s=self.clock(), samples=active['samples'],
                          independent_from_main_thread_blocking_work=True)
            self._log(flight, dict(result, kind='TARGET_LEASE_STOP', stage=flight.stage,
                                   target_frame_id=active['target_frame_id']))
            with self._condition:
                self._receipts[active['ticket_id']] = copy.deepcopy(result)
                self._active = None
                self._condition.notify_all()

    def _consume(self, flight, policy, state, proposal):
        if self._active is not None:
            return self._rejection(proposal, 'CONTROLLER_BUSY_UNTIL_SAMPLED_STOP', state,
                                   active_ticket_id=self._active['ticket_id'])
        now = self.clock()
        gate = policy.decide(frame=proposal['frame'], selection=proposal['selection'], surface=proposal['surface'],
                             desired_xy_m_s=proposal['desired_xy_m_s'],
                             actual_velocity_xy_m_s=state['velocity_ned_m_s'][:2],
                             self_position_ned_m=state['position_world_ned_m'],
                             previous_command_xy_m_s=flight.last_xy,
                             previous_command_monotonic_s=flight.last_command,
                             self_state_receipt_monotonic_s=state['host_receipt_monotonic_s'], now=now)
        if not isinstance(gate, dict) or gate.get('status') not in (
                'ALLOW_BOUNDED_COMMAND', 'REQUIRE_SELF_STATE_BRAKE', 'WAIT_NO_TARGET', 'REJECT'):
            raise RuntimeError('MALFORMED_CONTROLLER_MOTION_GATE')
        command = None
        if gate['status'] in ('ALLOW_BOUNDED_COMMAND', 'REQUIRE_SELF_STATE_BRAKE'):
            purpose = ('CURRENT_TARGET_TRAJECTORY' if gate['status'] == 'ALLOW_BOUNDED_COMMAND'
                       else 'OWN_STATE_LOSS_BRAKE')
            try:
                command = flight.issue(gate['command_velocity_xy_m_s'], gate['command_duration_s'],
                                       purpose, gate=gate)
            except RuntimeError as error:
                if str(error) not in ('SELF_STATE_OR_TARGET_DISPATCH_DEADLINE_EXPIRED',
                                      'COMMAND_DISPATCH_DEADLINE_EXPIRED'):
                    raise
                gate = dict(gate, status='REJECT', reason='ACTUAL_DISPATCH_DEADLINE_EXPIRED',
                            expired_gate_status=gate['status'], target_command_authorized=False,
                            self_state_brake_authorized=False, command_velocity_xy_m_s=[0., 0.],
                            command_duration_s=0.)
                # Remeasure before own-state braking; target data cannot authorize it.
                state = self._read(flight)
                command = flight.brake_tick(state)
        else:
            command = flight.brake_tick(state)
        frame_id = getattr(proposal['frame'], 'frame_id', None)
        clear = command['dispatch_monotonic_s'] if _xy_norm(command['velocity_xy_m_s']) < 1e-12 else None
        self._activate_stop(proposal['ticket_id'], command, frame_id, already_cleared=clear)
        return dict(ticket_id=proposal['ticket_id'], enqueued_monotonic_s=proposal['enqueued_monotonic_s'],
                    decision_completed_monotonic_s=self.clock(), command_self_state=copy.deepcopy(state),
                    motion_gate=gate, command=command)

    def _shutdown(self, flight):
        # Shutdown is bounded, uses own state only, and never creates a target command.
        zero = flight.issue([0., 0.], .12, 'OWN_STATE_CONTROLLER_CLOSE_ZERO')
        stable = None
        samples = []
        deadline = self.clock()+1.0
        while self.clock() <= deadline:
            state = self._read(flight)
            flight.issue([0., 0.], .12, 'OWN_STATE_CONTROLLER_CLOSE_HOLD')
            receipt = state['host_receipt_monotonic_s']
            stopped = receipt >= zero['dispatch_monotonic_s'] and _xy_norm(state['velocity_ned_m_s']) <= .005
            stable = receipt if stopped and stable is None else stable if stopped else None
            if not samples or receipt > samples[-1]['receipt_monotonic_s']:
                samples.append(dict(receipt_monotonic_s=receipt, velocity_xy_m_s=state['velocity_ned_m_s'][:2]))
            if stable is not None and receipt-stable >= .05-1e-9:
                result = dict(status='OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE',
                              target_setpoint_cleared_monotonic_s=zero['dispatch_monotonic_s'],
                              completed_monotonic_s=self.clock(), samples=samples,
                              explicit_zero=True, continuous_safety_established=False)
                self._log(flight, dict(result, kind='CONTROLLER_SHUTDOWN_STOP', stage=flight.stage))
                # If close interrupted an active lease, attach the actual premature
                # clear and stopping receipt instead of pretending normal expiry.
                if self._active is not None:
                    active = self._active
                    lease = dict(result, status='OWN_STATE_SAMPLED_STOPPED_BY_CONTROLLER_CLOSE',
                                 ticket_id=active['ticket_id'], declared_lease_end_monotonic_s=active['declared_end'],
                                 clear_after_declared_lease_s=zero['dispatch_monotonic_s']-active['declared_end'],
                                 lease_interrupted_by_close=True)
                    self._log(flight, dict(lease, kind='TARGET_LEASE_STOP', stage=flight.stage,
                                          target_frame_id=active['target_frame_id']))
                    with self._condition:
                        self._receipts[active['ticket_id']] = copy.deepcopy(lease)
                        self._active = None
                with self._condition:
                    self._shutdown_receipt = result
                return
            with self._condition:
                self._condition.wait(self.period_s)
        raise RuntimeError('CONTROLLER_SHUTDOWN_STOP_TIMEOUT')

    def _run(self):
        flight = policy = None
        try:
            flight = self.flight_factory()
            policy = self.policy_factory()
            state = self._read(flight)
            flight.brake_tick(state)
            with self._condition:
                self._ready = True
                self._condition.notify_all()
            next_tick = self.clock()+self.period_s
            while True:
                with self._condition:
                    if self._closing:
                        break
                    now = self.clock()
                    lease_event = self._active['wait_until'] if self._active and self._active['stop_deadline'] is None else None
                    due = now >= next_tick or (lease_event is not None and now >= lease_event)
                    # Pending work is sampled freshly; lease expiry gets an earlier
                    # wake even if it falls between nominal .05 second ticks.
                    if self._pending is None and not due:
                        wake = min(next_tick, lease_event) if lease_event is not None else next_tick
                        self._condition.wait(max(0., wake-now))
                        continue
                    proposal = self._pending
                    self._pending = None
                scheduled = next_tick if self.clock() >= next_tick else None
                state = self._read(flight, scheduled=scheduled)
                if scheduled is not None:
                    # Skip elapsed periods without inventing missing measurements.
                    next_tick += (math.floor(max(0., self.clock()-next_tick)/self.period_s)+1)*self.period_s
                self._stop_tick(flight, state)
                if proposal is not None:
                    decision = self._consume(flight, policy, state, proposal)
                    with self._condition:
                        self._decisions[proposal['ticket_id']] = copy.deepcopy(decision)
                        self._condition.notify_all()
                    self._log(flight, dict(decision, kind='CONTROL_DECISION', stage=flight.stage))
                elif self._active is None:
                    flight.brake_tick(state)
        except BaseException as error:
            with self._condition:
                self._fatal = closed_category(error)
                self._condition.notify_all()
            if flight is not None:
                try:
                    flight.issue([0., 0.], .12, 'OWN_STATE_CONTROLLER_FATAL_ZERO')
                except BaseException:
                    pass
        finally:
            if flight is not None:
                try:
                    self._shutdown(flight)
                except BaseException as error:
                    with self._condition:
                        details = closed_category(error)
                        self._fatal = self._fatal+'; CLOSE: '+details if self._fatal else details
            if policy is not None:
                try:
                    policy.close()
                except BaseException as error:
                    with self._condition:
                        self._fatal = self._fatal or 'POLICY_CLOSE: '+closed_category(error)
            with self._condition:
                if self._pending is not None:
                    self._decisions[self._pending['ticket_id']] = self._rejection(
                        self._pending, 'CONTROLLER_CLOSED_BEFORE_DECISION')
                    self._pending = None
                self._closed = True
                self._condition.notify_all()
