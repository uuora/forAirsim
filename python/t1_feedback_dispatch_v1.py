"""Measured feedback deadlines; one fresh own-state retry after an actual zero.

The retry cannot renew a target lease. Both attempts use the original 120 ms
limit, and a second expiration remains fatal. RPC clients stay on their owner.
"""
from __future__ import annotations

import time

import airsim
import numpy as np

from t1_control_settlement import own_state_feedback
from t1_feedback_flight import FeedbackFlight


class DispatchDeadlineExpired(RuntimeError):
    def __init__(self, checked, deadline):
        super().__init__('SELF_STATE_OR_TARGET_DISPATCH_DEADLINE_EXPIRED')
        self.checked, self.deadline = checked, deadline


class TimedFeedbackFlight(FeedbackFlight):
    """Retain actual command receipts and save the precise local deadline check."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feedback_sequence = 0
        self.feedback_recoveries = 0
        self.last_read_timing = None

    def issue(self, xy, duration, purpose, *, gate=None):
        # The frozen Flight.issue validation, RPC callback and receipt semantics
        # are retained; the deadline check clock is additionally observable.
        validation_started = time.monotonic()
        xy = np.asarray(xy, float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not 0 < duration <= .12:
            raise ValueError('INVALID_SHORT_COMMAND')
        if np.linalg.norm(xy) > .03000001:
            raise ValueError('SHORT_COMMAND_CAP')
        dispatched = time.monotonic()
        if gate:
            if dispatched > gate['latest_dispatch_monotonic_s'] + 1e-9:
                raise DispatchDeadlineExpired(dispatched, gate['latest_dispatch_monotonic_s'])
            if ('command_end_deadline_monotonic_s' in gate
                    and dispatched + duration > gate['command_end_deadline_monotonic_s'] + 1e-9):
                raise RuntimeError('COMMAND_DISPATCH_DEADLINE_EXPIRED')
        future = self.client.moveByVelocityZAsync(float(xy[0]), float(xy[1]), self.local_z, float(duration),
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0.), vehicle_name='DroneA')
        self.last_xy, self.last_command = xy.tolist(), dispatched
        self.target_lease_end = dispatched + duration if purpose == 'CURRENT_TARGET_TRAJECTORY' else None
        row = dict(kind='COMMAND', purpose=purpose, stage=self.stage, dispatch_monotonic_s=dispatched,
            returned_monotonic_s=time.monotonic(), velocity_xy_m_s=xy.tolist(), duration_s=float(duration),
            local_z_ned_m=self.local_z,
            target_frame_id=(gate.get('current_binding') or {}).get('frame_id')
                if gate and purpose == 'CURRENT_TARGET_TRAJECTORY' else None,
            target_end_deadline_monotonic_s=gate.get('command_end_deadline_monotonic_s') if gate else None,
            validation_started_monotonic_s=validation_started,
            deadline_check_monotonic_s=dispatched,
            latest_dispatch_monotonic_s=gate.get('latest_dispatch_monotonic_s') if gate else None)
        self.command_sequence += 1
        row['command_id'] = f'{self.stage}_{self.command_sequence}'

        def completed(done):
            error = done.error
            self.rpc_fault = self.rpc_fault or error is not None
            try:
                self.log(dict(kind='COMMAND_RPC_RESULT', stage=row['stage'], command_id=row['command_id'],
                    purpose=purpose, received_monotonic_s=time.monotonic(),
                    status='ERROR' if error is not None else 'RETURNED_TRUE' if done.result is True else
                        'RETURNED_FALSE' if done.result is False else 'OTHER_RESULT',
                    error_type=type(error).__name__ if error is not None else None))
            except BaseException:
                self.rpc_fault = True

        future.attach_callback(completed)
        self.emit(row)
        if purpose == 'CURRENT_TARGET_TRAJECTORY':
            self.stop_anchor_xy = None
        return row

    def _expired(self, state, plan, timing, attempt):
        # Discard the old vector and clear the actual setpoint first. This zero
        # has no source frame, feedback vector or target-observation authority.
        zero = self.issue([0., 0.], .12, 'OWN_STATE_FEEDBACK_EXPIRED_ZERO')
        self.emit(dict(kind='FEEDBACK_DISPATCH_REFUSAL', stage=self.stage,
            feedback_sequence=self.feedback_sequence, attempt=attempt,
            reason='FEEDBACK_SELF_STATE_EXPIRED_BEFORE_DISPATCH',
            rejected_feedback=plan, rejected_self_state=state, timing=timing,
            actual_zero_command_id=zero['command_id'], actual_zero_dispatch_monotonic_s=zero['dispatch_monotonic_s'],
            rejected_vector_dispatched=False, target_lease_renewed=False,
            original_self_state_limit_s=self.settlement_config.self_state_max_age_s,
            retry_allowed=attempt == 0))
        if attempt == 1:
            raise RuntimeError('FEEDBACK_SELF_STATE_EXPIRED_AFTER_ONE_FRESH_RETRY')
        self.feedback_recoveries += 1
        query_started = time.monotonic()
        fresh = self.state()
        query_completed = time.monotonic()
        reasons = self.protection(fresh)
        self.emit(dict(fresh, kind='SELF_STATE', stage=self.stage, protection_reasons=reasons,
            controller_query_started_monotonic_s=query_started,
            controller_query_completed_monotonic_s=query_completed,
            query_scope='ONE_FRESH_OWN_STATE_FEEDBACK_RETRY',
            feedback_sequence=self.feedback_sequence))
        if reasons:
            raise RuntimeError('SELF_STATE_PROTECTION_DURING_FEEDBACK_RETRY')
        self.last_read_timing = dict(query_started_monotonic_s=query_started,
            query_completed_monotonic_s=query_completed, refresh_only=True,
            state_event_admission_completed_monotonic_s=time.monotonic())
        return fresh

    def brake_tick(self, state=None):
        state = state or self.state()
        if self.stop_anchor_xy is None:
            self.stop_anchor_xy = list(state['position_world_ned_m'][:2])
        self.feedback_sequence += 1
        for attempt in range(2):
            started = time.monotonic()
            timing = dict(preparation_started_monotonic_s=started,
                original_state_receipt_monotonic_s=state['host_receipt_monotonic_s'],
                original_state_timestamp_ns=state['state_timestamp_ns'],
                read_timing=self.last_read_timing)
            try:
                plan = own_state_feedback(state, self.stop_anchor_xy, self.last_xy,
                    self.last_command, started, self.low, self.high, self.settlement_config)
            except ValueError as error:
                if (str(error) != 'FEEDBACK_SELF_STATE_EXPIRED_OR_FUTURE'
                        or started - state['host_receipt_monotonic_s'] <= self.settlement_config.self_state_max_age_s):
                    raise
                timing.update(preparation_completed_monotonic_s=time.monotonic(),
                    rejection_checked_monotonic_s=started, rejection_boundary='PREPARATION_ENTRY')
                state = self._expired(state, None, timing, attempt)
                continue
            timing['preparation_completed_monotonic_s'] = time.monotonic()
            try:
                command = self.issue(plan['command_velocity_xy_m_s'], .12, 'OWN_STATE_FEEDBACK_BRAKE', gate=plan)
            except DispatchDeadlineExpired as error:
                timing.update(rejection_checked_monotonic_s=error.checked, rejection_boundary='ACTUAL_LOCAL_DISPATCH_CHECK')
                state = self._expired(state, plan, timing, attempt)
                continue
            timing.update(deadline_check_monotonic_s=command['deadline_check_monotonic_s'],
                command_receipt_admission_completed_monotonic_s=time.monotonic())
            self.emit(dict(kind='OWN_STATE_FEEDBACK_PROOF', stage=self.stage, command_id=command['command_id'],
                dispatch_monotonic_s=command['dispatch_monotonic_s'], feedback=plan,
                command_self_state=state, target_observation_used=False,
                feedback_sequence=self.feedback_sequence, attempt=attempt, timing=timing))
            return command
        raise AssertionError('BOUNDED_FEEDBACK_ATTEMPTS_EXHAUSTED')
