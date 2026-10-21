"""Independent low-speed Flight adapter: feedback uses only current own state."""
from __future__ import annotations

import time

from t1_control_settlement import SettlementConfig, own_state_feedback
from t1_identity_flight_v2 import Flight


class FeedbackFlight(Flight):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_anchor_xy = None
        self.settlement_config = SettlementConfig()

    def issue(self, xy, duration, purpose, *, gate=None):
        row = super().issue(xy, duration, purpose, gate=gate)
        if purpose == 'CURRENT_TARGET_TRAJECTORY':
            self.stop_anchor_xy = None
        return row

    def clear_target_tick(self, state):
        # First actually replace the target setpoint with zero under the frozen
        # boundary/ramp envelope; feedback is a separately identified command.
        row = super().brake_tick(state)
        if sum(x*x for x in row['velocity_xy_m_s']) <= 1e-24:
            self.stop_anchor_xy = list(state['position_world_ned_m'][:2])
        return row

    def brake_tick(self, state=None):
        state = state or self.state()
        if self.stop_anchor_xy is None:
            self.stop_anchor_xy = list(state['position_world_ned_m'][:2])
        plan = own_state_feedback(state, self.stop_anchor_xy, self.last_xy,
            self.last_command, time.monotonic(), self.low, self.high, self.settlement_config)
        if time.monotonic() > plan['latest_dispatch_monotonic_s']:
            raise RuntimeError('FEEDBACK_SELF_STATE_EXPIRED_BEFORE_DISPATCH')
        row = self.issue(plan['command_velocity_xy_m_s'], .12, 'OWN_STATE_FEEDBACK_BRAKE', gate=plan)
        self.emit(dict(kind='OWN_STATE_FEEDBACK_PROOF', stage=self.stage, command_id=row['command_id'],
            dispatch_monotonic_s=row['dispatch_monotonic_s'], feedback=plan,
            command_self_state=state, target_observation_used=False))
        return row
