"""Preserve target first-clear clock when rebuilding a shutdown stop window.

The closing window still requires its own actual zero, new state samples and
unchanged dwell/thresholds. An active lease retains its original 2s deadline.
This new revision is offline tested; no V11 flight record is rewritten.
"""
import copy
from t1_async_control import _xy_norm
from t1_async_control_v1p7 import AsyncControlV1P7
from t1_control_settlement import SampledStopWindow


class AsyncControlV1P8(AsyncControlV1P7):
    revision='PRESERVED_FIRST_TARGET_CLEAR_AND_DISTINCT_SHUTDOWN_WINDOW_V1P8'

    def _shutdown(self,flight):
        active=self._active
        first_clear=active.get('clear') if active else None
        zero=flight.issue([0.,0.],.12,'OWN_STATE_CONTROLLER_CLOSE_ZERO')
        state=self._read(flight)
        flight.stop_anchor_xy=list(state['position_world_ned_m'][:2])
        window=SampledStopWindow(zero['dispatch_monotonic_s'],flight.stop_anchor_xy,self.settlement)
        closing_deadline=zero['dispatch_monotonic_s']+self.settlement.maximum_stop_after_lease_s
        original_deadline=active['declared_end']+self.settlement.maximum_stop_after_lease_s if active else None
        deadline=min(closing_deadline,original_deadline) if active else closing_deadline
        while self.clock()<=deadline:
            state=self._read(flight)
            previous_xy,previous_dispatch=list(flight.last_xy),flight.last_command
            flight.brake_tick(state)
            established=window.observe(state,previous_xy,self.clock(),previous_dispatch)
            if _xy_norm(flight.last_xy)>1e-12:
                window.stable_s=None;window.stable_state_timestamps.clear();established=False
            if established:
                result=dict(window.proof(),status='OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE',
                    completed_monotonic_s=self.clock(),explicit_zero=True,feedback_after_target_clear=True,
                    target_lease_renewed=False,shutdown_window_clear_monotonic_s=zero['dispatch_monotonic_s'],
                    shutdown_stop_deadline_monotonic_s=deadline,original_active_lease_stop_deadline_monotonic_s=original_deadline)
                self._log(flight,dict(result,kind='CONTROLLER_SHUTDOWN_STOP',stage=flight.stage))
                if active is not None:
                    target_clear=first_clear if first_clear is not None else zero['dispatch_monotonic_s']
                    lease=dict(result,status='OWN_STATE_SAMPLED_STOPPED_BY_CONTROLLER_CLOSE',
                        ticket_id=active['ticket_id'],declared_lease_end_monotonic_s=active['declared_end'],
                        target_setpoint_cleared_monotonic_s=target_clear,
                        clear_after_declared_lease_s=target_clear-active['declared_end'],
                        first_target_clear_preserved=first_clear is not None,
                        clear_clock_scope='first actual target clear; shutdown window clock retained separately',
                        lease_interrupted_by_close=True,settlement_deadline_monotonic_s=original_deadline)
                    self._log(flight,dict(lease,kind='TARGET_LEASE_STOP',stage=flight.stage,
                        target_frame_id=active['target_frame_id']))
                    with self._condition:
                        self._receipts[active['ticket_id']]=copy.deepcopy(lease);self._active=None
                with self._condition:
                    self._shutdown_receipt=result;self._settled=True
                return
            with self._condition:self._condition.wait(self.period_s)
        raise RuntimeError('STRICT_ORIGINAL_TARGET_CLOSE_STOP_TIMEOUT')
