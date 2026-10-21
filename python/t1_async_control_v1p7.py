"""Expose own-state validation, condition lock and event-admission boundaries."""
import copy
import math

from t1_async_control_v1p6 import AsyncControlV1P6


class AsyncControlV1P7(AsyncControlV1P6):
    revision = 'STRICT_STOP_TIMED_READ_ONE_FRESH_FEEDBACK_RETRY_V1P7'

    def _read(self, flight, *, scheduled=None):
        started = self.clock()
        state = flight.state()
        completed = self.clock()
        if (not isinstance(state, dict)
                or not math.isfinite(state.get('host_receipt_monotonic_s', float('nan')))
                or state['host_receipt_monotonic_s'] > completed + 1e-9):
            raise RuntimeError('INVALID_CONTROLLER_SELF_STATE_RECEIPT')
        for key, count in (('position_world_ned_m', 3), ('velocity_ned_m_s', 3)):
            if key not in state or len(state[key]) != count or not all(math.isfinite(float(x)) for x in state[key]):
                raise RuntimeError('INVALID_CONTROLLER_SELF_STATE_VECTOR')
        reasons = list(flight.protection(state))
        validated = self.clock()
        state = dict(state, kind='SELF_STATE', stage=flight.stage, protection_reasons=reasons,
            controller_query_started_monotonic_s=started, controller_query_completed_monotonic_s=completed,
            controller_ticket_id=self._active['ticket_id'] if self._active else None)
        if scheduled is not None:
            state.update(controller_scheduled_tick_monotonic_s=scheduled,
                controller_tick_lateness_s=max(0., started - scheduled))
        lock_entered = self.clock()
        with self._condition:
            lock_acquired = self.clock()
            if self._last_query is not None:
                self._gaps.append(state['host_receipt_monotonic_s'] - self._last_query)
            self._last_query = state['host_receipt_monotonic_s']
            if scheduled is not None:
                self._lateness.append(max(0., started - scheduled))
            self._latest = copy.deepcopy(state)
        lock_released = self.clock()
        event_started = self.clock()
        self._log(flight, state)
        event_completed = self.clock()
        timing = dict(query_started_monotonic_s=started, query_completed_monotonic_s=completed,
            validation_completed_monotonic_s=validated, condition_entered_monotonic_s=lock_entered,
            condition_acquired_monotonic_s=lock_acquired, condition_released_monotonic_s=lock_released,
            state_event_admission_started_monotonic_s=event_started,
            state_event_admission_completed_monotonic_s=event_completed,
            state_receipt_monotonic_s=state['host_receipt_monotonic_s'],
            state_timestamp_ns=state['state_timestamp_ns'])
        flight.last_read_timing = timing
        self._log(flight, dict(kind='OWN_STATE_READ_TIMING', stage=flight.stage, timing=timing))
        if reasons:
            flight.issue([0., 0.], .12, 'OWN_STATE_PROTECTION_ZERO')
            raise RuntimeError('SELF_STATE_PROTECTION: ' + ','.join(reasons))
        return state
