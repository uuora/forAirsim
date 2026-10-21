"""Own-state feedback and sampled stopping; no simulator or target-truth access."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from trajectory_execution_guard import boundary_command


@dataclass(frozen=True)
class SettlementConfig:
    stationary_xy_threshold_m_s: float = .0005
    stationary_span_s: float = .10
    minimum_post_clear_observation_s: float = .30
    minimum_distinct_state_timestamps: int = 3
    maximum_anchor_error_m: float = .002
    anchor_deadband_m: float = .001
    feedback_velocity_gain: float = 4.
    feedback_position_gain_s_inv: float = .5
    feedback_speed_cap_m_s: float = .015
    command_acceleration_cap_m_s2: float = .15
    self_state_max_age_s: float = .12
    maximum_stop_after_lease_s: float = 2.

    def __post_init__(self):
        expected = dict(stationary_xy_threshold_m_s=.0005, stationary_span_s=.10,
            minimum_post_clear_observation_s=.30, minimum_distinct_state_timestamps=3,
            maximum_anchor_error_m=.002, anchor_deadband_m=.001,
            feedback_velocity_gain=4., feedback_position_gain_s_inv=.5,
            feedback_speed_cap_m_s=.015, command_acceleration_cap_m_s2=.15,
            self_state_max_age_s=.12, maximum_stop_after_lease_s=2.)
        if asdict(self) != expected or any(isinstance(x, bool) for x in asdict(self).values()):
            raise ValueError('SETTLEMENT_CONFIG_DIFFERS_FROM_PREDECLARED_V1')


def own_state_feedback(state, anchor_xy, last_xy, last_command_s, now, low, high,
                       config=None):
    c = config or SettlementConfig()
    receipt = state.get('host_receipt_monotonic_s')
    if (not isinstance(receipt, (float, int)) or isinstance(receipt, bool)
            or not math.isfinite(receipt) or not 0 <= now-receipt <= c.self_state_max_age_s):
        raise ValueError('FEEDBACK_SELF_STATE_EXPIRED_OR_FUTURE')
    if (not math.isfinite(now) or not math.isfinite(last_command_s)
            or last_command_s > now):
        raise ValueError('FEEDBACK_ACTUAL_COMMAND_CLOCK_INVALID')
    p = np.asarray(state['position_world_ned_m'], dtype=float)
    v = np.asarray(state['velocity_ned_m_s'], dtype=float)
    anchor = np.asarray(anchor_xy, dtype=float)
    if p.shape != (3,) or v.shape != (3,) or anchor.shape != (2,) or not all(
            np.isfinite(x).all() for x in (p, v, anchor)):
        raise ValueError('FEEDBACK_SELF_STATE_VECTOR_INVALID')
    error = p[:2]-anchor
    # Deadband prevents a tiny positional correction from keeping a sampled
    # stop perpetually alive. A new target command resets the own-state anchor.
    position_term = error if np.linalg.norm(error) > c.anchor_deadband_m else np.zeros(2)
    desired = -c.feedback_velocity_gain*v[:2]-c.feedback_position_gain_s_inv*position_term
    if np.linalg.norm(v[:2]) <= c.stationary_xy_threshold_m_s and not position_term.any():
        desired[:] = 0.
    # Use the actual last dispatch clock. At zero elapsed time return the old
    # setpoint; never manufacture a nominal acceleration interval.
    elapsed = min(max(now-last_command_s, 0.), .4)
    envelope = boundary_command(p, desired, last_xy, low, high, .03,
        c.command_acceleration_cap_m_s2, max(elapsed, 1e-12), 1., .02)
    if np.linalg.norm(desired) > c.feedback_speed_cap_m_s:
        desired *= c.feedback_speed_cap_m_s/np.linalg.norm(desired)
        envelope = boundary_command(p, desired, last_xy, low, high, .03,
            c.command_acceleration_cap_m_s2, max(elapsed, 1e-12), 1., .02)
    if not envelope['accepted']:
        raise ValueError('FEEDBACK_BOUNDARY_AND_RAMP_REFUSED')
    if elapsed == 0.:
        envelope['command_velocity_xy_m_s'] = list(last_xy)
    return dict(envelope, self_state_receipt_monotonic_s=receipt,
        self_state_age_at_decision_s=now-receipt, actual_previous_dispatch_monotonic_s=last_command_s,
        elapsed_since_actual_previous_dispatch_s=elapsed, anchor_xy_world_ned_m=anchor.tolist(),
        anchor_error_xy_m=error.tolist(), measured_velocity_xy_m_s=v[:2].tolist(),
        latest_dispatch_monotonic_s=receipt+c.self_state_max_age_s,
        target_observation_used=False, configured_settlement=asdict(c))


class SampledStopWindow:
    """Post-clear dwell plus distinct simulation states, never a physical guarantee."""

    def __init__(self, clear_s, anchor_xy, config=None):
        self.config = config or SettlementConfig()
        self.clear_s = float(clear_s)
        self.anchor_xy = tuple(anchor_xy)
        self.stable_s = None
        self.stable_state_timestamps = set()
        self.samples = []
        self.last_receipt = None

    def observe(self, state, last_xy, now, last_dispatch_s=None):
        c = self.config
        receipt = state['host_receipt_monotonic_s']
        stamp = state.get('state_timestamp_ns')
        valid = (isinstance(stamp, int) and not isinstance(stamp, bool)
            and isinstance(receipt, (float, int)) and math.isfinite(receipt)
            and self.clear_s <= receipt <= now and now-receipt <= c.self_state_max_age_s
            and (self.last_receipt is None or receipt >= self.last_receipt)
            and (last_dispatch_s is None or last_dispatch_s <= receipt))
        speed = math.hypot(*state['velocity_ned_m_s'][:2])
        error = math.hypot(*(state['position_world_ned_m'][i]-self.anchor_xy[i] for i in range(2)))
        stopped = (valid and math.isfinite(speed) and math.isfinite(error)
            and speed <= c.stationary_xy_threshold_m_s and error <= c.maximum_anchor_error_m
            and math.hypot(*last_xy) <= 1e-12 and not state.get('protection_reasons'))
        if not stopped:
            self.stable_s = None
            self.stable_state_timestamps.clear()
        elif self.last_receipt is None or receipt > self.last_receipt:
            if self.stable_s is None:
                self.stable_s = receipt
            self.stable_state_timestamps.add(stamp)
        if valid and (self.last_receipt is None or receipt > self.last_receipt):
            self.samples.append(dict(receipt_monotonic_s=receipt, state_timestamp_ns=stamp,
                velocity_xy_m_s=state['velocity_ned_m_s'][:2],
                position_world_ned_m=state['position_world_ned_m'],
                actual_last_command_xy_m_s=list(last_xy),
                actual_last_command_dispatch_monotonic_s=last_dispatch_s,
                stopped_at_stricter_threshold=stopped))
            self.last_receipt = receipt
        return bool(stopped and self.stable_s is not None
            and receipt-self.clear_s >= c.minimum_post_clear_observation_s-1e-9
            and receipt-self.stable_s >= c.stationary_span_s-1e-9
            and len(self.stable_state_timestamps) >= c.minimum_distinct_state_timestamps)

    def proof(self):
        return dict(target_setpoint_cleared_monotonic_s=self.clear_s,
            sampled_stop_started_monotonic_s=self.stable_s,
            distinct_stable_simulator_state_timestamps=len(self.stable_state_timestamps),
            own_state_anchor_xy_world_ned_m=list(self.anchor_xy), samples=self.samples,
            configured_settlement=asdict(self.config), continuous_safety_established=False)


def completion_timing(frame):
    """Keep the historical max clock and report perception and settlement separately."""
    due, deadline = frame.get('due_monotonic_s'), frame.get('deadline_monotonic_s')
    processing = frame.get('frame_processing_completed_monotonic_s')
    command, stop = frame.get('command'), frame.get('lease_stop')
    stopped = stop.get('completed_monotonic_s') if isinstance(stop, dict) else None
    def number(x):
        return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)
    processing_status = ('TIMELY' if processing <= deadline+1e-9 else 'LATE') if all(
        number(x) for x in (processing, deadline)) else 'UNKNOWN'
    stop_status = 'NOT_APPLICABLE_NO_COMMAND' if not command else 'UNKNOWN_MISSING_STOP_RECEIPT'
    stop_deadline = None
    if command and stopped is not None:
        end = command.get('dispatch_monotonic_s', 0)+command.get('duration_s', 0)
        stop_deadline = end+SettlementConfig().maximum_stop_after_lease_s
        stop_status = 'WITHIN_PREDECLARED_SETTLEMENT_BOUND' if stopped <= stop_deadline+1e-9 else 'LATE_SETTLEMENT'
    full = max(processing, stopped) if all(number(x) for x in (processing, stopped)) else (
        processing if not command and number(processing) else None)
    return dict(frame_processing_timing=processing_status, settlement_timing=stop_status,
        settlement_deadline_monotonic_s=stop_deadline,
        historical_full_completion_monotonic_s=full,
        historical_full_completion_timing=('TIMELY' if full <= deadline+1e-9 else 'LATE')
            if number(full) and number(deadline) else 'UNKNOWN',
        full_completion_limiter=('CONTROL_SETTLEMENT' if stopped >= processing else 'FRAME_PROCESSING')
            if all(number(x) for x in (processing, stopped)) else 'UNKNOWN',
        full_completion_after_due_s=full-due if number(full) and number(due) else None,
        timing_policy_changed=True, historical_full_clock_preserved=True)
