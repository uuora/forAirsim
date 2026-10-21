"""Reserve measured control-clear budget without extending image/review freshness."""
import math
from t1_identity_motion_policy import IdentityMotionPolicy


class IdentityMotionPolicyV1P1(IdentityMotionPolicy):
    actual_clear_reserve_s = .12

    def decide(self, **kwargs):
        gate=super().decide(**kwargs)
        if gate['status']!='ALLOW_BOUNDED_COMMAND' or math.hypot(*gate['command_velocity_xy_m_s'])<=1e-12:
            return gate
        observation_deadline=gate['command_end_deadline_monotonic_s']
        declared_end_limit=observation_deadline-self.actual_clear_reserve_s
        available=declared_end_limit-gate['decision_monotonic_s']
        duration=min(gate['command_duration_s'],available)
        gate=dict(gate,actual_clear_reserve_s=self.actual_clear_reserve_s,
            original_observation_end_deadline_monotonic_s=observation_deadline,
            latest_declared_lease_end_monotonic_s=declared_end_limit,
            reserve_is_not_hard_realtime_guarantee=True)
        if duration<self.config.minimum_command_lease_s:
            return dict(gate,status='WAIT_NO_TARGET',reason='INSUFFICIENT_CURRENT_FRAME_BUDGET_FOR_LEASE_AND_ACTUAL_CLEAR',
                target_command_authorized=False,command_velocity_xy_m_s=[0.,0.],command_duration_s=0.)
        return dict(gate,command_duration_s=duration,
            latest_dispatch_monotonic_s=min(gate['latest_dispatch_monotonic_s'],declared_end_limit-duration),
            # The original deadline still governs actual clear. Only admission
            # and declared duration shrink, so the old independent audit applies.
            command_end_deadline_monotonic_s=observation_deadline)
