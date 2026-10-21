"""Strict sampled stopping with own-state feedback, retaining independent control."""
from __future__ import annotations

import copy
from dataclasses import asdict
import time

from t1_async_control import AsyncControl, _xy_norm
from t1_async_control_v1p1 import AsyncControlV1P1
from t1_control_settlement import SampledStopWindow, SettlementConfig


class AsyncControlV1P2(AsyncControlV1P1):
    revision = 'OWN_STATE_FEEDBACK_STRICT_SAMPLED_STOP_V1P2'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settlement = SettlementConfig()
        self._idle_window = None
        self._settled = False

    def wait_until_settled(self, timeout=2.5):
        self._positive_timeout(timeout)
        deadline = time.monotonic()+timeout
        with self._condition:
            while not self._settled or self._active is not None or self._pending is not None:
                if self._fatal or self._closed:
                    raise RuntimeError('CONTROL_NOT_AVAILABLE_FOR_STRICT_SETTLEMENT')
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('STRICT_SETTLEMENT_WAIT_TIMEOUT')
                self._condition.wait(remaining)
            return copy.deepcopy(self._latest)

    def scheduling_receipt(self):
        row = super().scheduling_receipt()
        row['settlement_config'] = asdict(self.settlement)
        row['feedback_target_truth_access'] = False
        return row

    def _consume(self, flight, policy, state, proposal):
        if self._active is not None:
            return self._rejection(proposal, 'CONTROLLER_BUSY_UNTIL_STRICT_SAMPLED_STOP', state,
                active_ticket_id=self._active['ticket_id'])
        with self._condition:
            self._settled = False
        if _xy_norm(state['velocity_ned_m_s']) > self.settlement.stationary_xy_threshold_m_s:
            command = flight.brake_tick(state)
            self._activate_stop(proposal['ticket_id'], command)
            self._idle_window = self._idle_basis = None
            result = self._rejection(proposal, 'SELF_STATE_RESIDUAL_REQUIRES_SETTLEMENT', state)
            result['command'] = command
            return result
        result = AsyncControl._consume(self, flight, policy, state, proposal)
        self._idle_window = self._idle_basis = None
        return result

    def _stop_tick(self, flight, state):
        active = self._active
        if active is None or self.clock() < active['wait_until']:
            return
        if active['stop_deadline'] is None:
            # The bound starts at the declared lease end, so a late wake cannot
            # silently extend it. Zero/own-state leases use the same clock.
            active['stop_deadline'] = active['declared_end']+self.settlement.maximum_stop_after_lease_s
        if self.clock() > active['stop_deadline']:
            raise RuntimeError('STRICT_OWN_STATE_STOP_TIMEOUT')
        previous_xy, previous_dispatch = list(flight.last_xy), flight.last_command
        if active['clear'] is None:
            command = flight.clear_target_tick(state)
            if _xy_norm(command['velocity_xy_m_s']) <= 1e-12:
                active['clear'] = command['dispatch_monotonic_s']
        else:
            flight.brake_tick(state)
        if active['clear'] is None:
            return
        if 'window' not in active:
            active['window'] = SampledStopWindow(active['clear'],
                flight.stop_anchor_xy or state['position_world_ned_m'][:2], self.settlement)
        window = active['window']
        if _xy_norm(flight.last_xy) > 1e-12:
            previous_xy, previous_dispatch = list(flight.last_xy), flight.last_command
        if not window.observe(state, previous_xy, self.clock(), previous_dispatch):
            return
        result = dict(window.proof(), status='OWN_STATE_SAMPLED_STOPPED_WITH_INDEPENDENT_CONTROL',
            ticket_id=active['ticket_id'], declared_lease_end_monotonic_s=active['declared_end'],
            clear_after_declared_lease_s=active['clear']-active['declared_end'],
            completed_monotonic_s=self.clock(), independent_from_main_thread_blocking_work=True,
            feedback_after_target_clear=True, target_lease_renewed=False,
            settlement_deadline_monotonic_s=active['stop_deadline'])
        self._log(flight, dict(result, kind='TARGET_LEASE_STOP', stage=flight.stage,
            target_frame_id=active['target_frame_id']))
        with self._condition:
            self._receipts[active['ticket_id']] = copy.deepcopy(result)
            self._active = None
            self._settled = True
            self._condition.notify_all()
        self._idle_window = window
        self._idle_basis = dict(origin='STRICT_INDEPENDENT_TARGET_LEASE_STOP_RECEIPT')

    def _idle_tick(self, flight, state):
        now = self.clock()
        if self._idle_window is None:
            row = flight.clear_target_tick(state)
            if _xy_norm(row['velocity_xy_m_s']) <= 1e-12:
                self._idle_window = SampledStopWindow(row['dispatch_monotonic_s'],
                    flight.stop_anchor_xy or state['position_world_ned_m'][:2], self.settlement)
                self._idle_basis = dict(origin='ACTUAL_IDLE_ZERO_DISPATCH')
            return row
        window = self._idle_window
        established = window.observe(state, flight.last_xy, now, flight.last_command)
        with self._condition:
            self._settled = established
            if established:
                self._condition.notify_all()
        elapsed = now-flight.last_command
        if not established or elapsed >= self.idle_zero_refresh_s:
            row = flight.brake_tick(state)
            if established and elapsed >= self.idle_zero_refresh_s:
                self._log(flight, dict(kind='IDLE_ZERO_REFRESH', stage=flight.stage,
                    actual_zero_dispatch_monotonic_s=row['dispatch_monotonic_s'],
                    actual_command_velocity_xy_m_s=row['velocity_xy_m_s'],
                    actual_command_id=row.get('command_id'), real_command_dispatched=True,
                    target_observation_used=False))
                with self._condition:
                    self._idle_refreshes += 1
            return row
        row = dict(kind='IDLE_ZERO_OMITTED', stage=flight.stage, decision_monotonic_s=now,
            self_state_receipt_monotonic_s=state['host_receipt_monotonic_s'],
            actual_velocity_xy_m_s=state['velocity_ned_m_s'][:2],
            last_actual_command_velocity_xy_m_s=list(flight.last_xy),
            last_actual_zero_dispatch_monotonic_s=flight.last_command,
            elapsed_since_last_actual_zero_dispatch_s=elapsed,
            configured_real_zero_refresh_s=self.idle_zero_refresh_s,
            sampled_stop_clear_monotonic_s=window.clear_s,
            sampled_stop_started_monotonic_s=window.stable_s,
            sampled_stop_span_s=state['host_receipt_monotonic_s']-window.stable_s,
            distinct_post_clear_sample_count=len(window.stable_state_timestamps),
            sampled_stop_basis=self._idle_basis['origin'],
            stationary_xy_threshold_m_s=self.settlement.stationary_xy_threshold_m_s,
            active_target_lease=False, zero_command_omitted=True, command_dispatched=False,
            command_receipt_created=False, target_observation_used=False, flight_authorized=False,
            continuous_safety_established=False, configured_settlement=asdict(self.settlement))
        self._log(flight, row)
        with self._condition:
            self._idle_omissions += 1

    def _shutdown(self, flight):
        # A genuine zero cancels a target lease; following feedback is own-state
        # braking and cannot extend a target observation's authority.
        zero = flight.issue([0., 0.], .12, 'OWN_STATE_CONTROLLER_CLOSE_ZERO')
        state = self._read(flight)
        flight.stop_anchor_xy = list(state['position_world_ned_m'][:2])
        window = SampledStopWindow(zero['dispatch_monotonic_s'], flight.stop_anchor_xy, self.settlement)
        deadline = zero['dispatch_monotonic_s']+self.settlement.maximum_stop_after_lease_s
        while self.clock() <= deadline:
            state = self._read(flight)
            previous_xy, previous_dispatch = list(flight.last_xy), flight.last_command
            flight.brake_tick(state)
            if _xy_norm(flight.last_xy) > 1e-12:
                previous_xy, previous_dispatch = list(flight.last_xy), flight.last_command
            if window.observe(state, previous_xy, self.clock(), previous_dispatch):
                result = dict(window.proof(), status='OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE',
                    completed_monotonic_s=self.clock(), explicit_zero=True,
                    feedback_after_target_clear=True, target_lease_renewed=False)
                self._log(flight, dict(result, kind='CONTROLLER_SHUTDOWN_STOP', stage=flight.stage))
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
                    self._settled = True
                return
            with self._condition:
                self._condition.wait(self.period_s)
        raise RuntimeError('STRICT_CONTROLLER_SHUTDOWN_STOP_TIMEOUT')
