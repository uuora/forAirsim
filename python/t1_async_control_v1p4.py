"""Strict stop receipts retain the command actually present before each state query."""
import copy
from t1_async_control import _xy_norm
from t1_async_control_v1p3 import AsyncControlV1P3
from t1_control_settlement import SampledStopWindow


class AsyncControlV1P4(AsyncControlV1P3):
    revision = 'STRICT_STOP_PRE_QUERY_ACTUAL_COMMAND_RECEIPTS_V1P4'

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
        established = window.observe(state, previous_xy, self.clock(), previous_dispatch)
        if _xy_norm(flight.last_xy) > 1e-12:
            window.stable_s = None
            window.stable_state_timestamps.clear()
            established = False
        if not established:
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
            established = window.observe(state, previous_xy, self.clock(), previous_dispatch)
            if _xy_norm(flight.last_xy) > 1e-12:
                window.stable_s = None
                window.stable_state_timestamps.clear()
                established = False
            if established:
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
