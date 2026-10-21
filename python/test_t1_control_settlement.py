"""Residual motion, delayed response, duplicate states and separate timing tests."""
import unittest

from t1_control_settlement import SettlementConfig, SampledStopWindow, own_state_feedback, completion_timing


def state(t=10., speed=0., stamp=None, x=0.):
    return dict(host_receipt_monotonic_s=t, state_timestamp_ns=int(t*1e9) if stamp is None else stamp,
        position_world_ned_m=[x, 0., -3.], velocity_ned_m_s=[speed, 0., 0.], protection_reasons=[])


class SettlementTests(unittest.TestCase):
    def feedback(self, s=None, previous=(0., 0.), previous_t=9.95, now=10.):
        return own_state_feedback(s or state(), [0., 0.], previous, previous_t, now,
            [-.3, -.3, -3.15], [.3, .3, -2.85])

    def test_strict_config_cannot_silently_relax(self):
        with self.assertRaises(ValueError): SettlementConfig(stationary_xy_threshold_m_s=.005)

    def test_residual_speed_receives_opposite_feedback(self):
        r = self.feedback(state(speed=.0036))
        self.assertLess(r['command_velocity_xy_m_s'][0], 0.)
        self.assertLessEqual(abs(r['command_velocity_xy_m_s'][0]), .15*.05+1e-9)
        self.assertFalse(r['target_observation_used'])

    def test_feedback_uses_actual_dispatch_elapsed(self):
        r = self.feedback(state(speed=.02), previous=(.009, 0.), previous_t=10.)
        self.assertEqual(r['command_velocity_xy_m_s'], [.009, 0.])
        r = self.feedback(state(speed=.02), previous=(.009, 0.), previous_t=9.99)
        self.assertAlmostEqual(abs(r['command_velocity_xy_m_s'][0]-.009), .0015)

    def test_static_deadband_returns_zero(self):
        self.assertEqual(self.feedback(state(speed=.0002, x=.0002))['command_velocity_xy_m_s'], [0., 0.])

    def test_position_feedback_is_own_anchor(self):
        self.assertLess(self.feedback(state(x=.01))['command_velocity_xy_m_s'][0], 0.)

    def test_stale_future_state_and_command_clock_refused(self):
        for s, prev in ((state(t=9.87), 9.95), (state(t=10.1), 9.95), (state(), 10.1)):
            with self.subTest(s=s, prev=prev), self.assertRaises(ValueError):
                self.feedback(s, previous_t=prev)

    def test_outside_workspace_refused(self):
        with self.assertRaises(ValueError): self.feedback(state(x=.31))

    def test_old_threshold_residual_cannot_be_stopped(self):
        w = SampledStopWindow(10., [0., 0.])
        for t in (10., 10.1, 10.2, 10.3, 10.4):
            self.assertFalse(w.observe(state(t=t, speed=.0036), [0., 0.], t))
        self.assertIsNone(w.stable_s)

    def test_post_clear_dwell_requires_three_distinct_states(self):
        w = SampledStopWindow(10., [0., 0.])
        self.assertFalse(w.observe(state(t=10.05), [0., 0.], 10.05))
        self.assertFalse(w.observe(state(t=10.15), [0., 0.], 10.15))
        self.assertTrue(w.observe(state(t=10.3), [0., 0.], 10.3))

    def test_repeated_simulator_timestamp_cannot_establish_stop(self):
        w = SampledStopWindow(10., [0., 0.])
        for t in (10.05, 10.15, 10.3, 10.4):
            self.assertFalse(w.observe(state(t=t, stamp=1), [0., 0.], t))

    def test_late_inertial_peak_resets_stable_window(self):
        w = SampledStopWindow(10., [0., 0.])
        for t in (10.05, 10.1): w.observe(state(t=t, speed=.0003), [0., 0.], t)
        self.assertFalse(w.observe(state(t=10.2, speed=.0008), [0., 0.], 10.2))
        self.assertFalse(w.observe(state(t=10.3), [0., 0.], 10.3))
        self.assertFalse(w.observe(state(t=10.35), [0., 0.], 10.35))
        self.assertTrue(w.observe(state(t=10.4), [0., 0.], 10.4))

    def test_nonzero_last_goal_and_pre_dispatch_measurement_refused(self):
        for xy, dispatch in (([.001, 0.], 9.), ([0., 0.], 11.)):
            w = SampledStopWindow(10., [0., 0.])
            for t in (10.1, 10.2, 10.3):
                self.assertFalse(w.observe(state(t=t), xy, t, dispatch))

    def test_anchor_drift_and_protection_refused(self):
        for s in (state(t=10.3, x=.003), dict(state(t=10.3), protection_reasons=['FAULT'])):
            self.assertFalse(SampledStopWindow(10., [0., 0.]).observe(s, [0., 0.], 10.3))

    def test_split_timing_does_not_relabel_historical_late(self):
        f = dict(due_monotonic_s=10., deadline_monotonic_s=10.4,
            frame_processing_completed_monotonic_s=10.3,
            command=dict(dispatch_monotonic_s=10.25, duration_s=.12),
            lease_stop=dict(completed_monotonic_s=10.7))
        r = completion_timing(f)
        self.assertEqual(r['frame_processing_timing'], 'TIMELY')
        self.assertEqual(r['historical_full_completion_timing'], 'LATE')
        self.assertEqual(r['settlement_timing'], 'WITHIN_PREDECLARED_SETTLEMENT_BOUND')
        self.assertEqual(r['full_completion_limiter'], 'CONTROL_SETTLEMENT')

    def test_command_without_stop_is_unknown_not_timely(self):
        f = dict(deadline_monotonic_s=10.4, frame_processing_completed_monotonic_s=10.3,
            command=dict(dispatch_monotonic_s=10.25, duration_s=.12))
        self.assertEqual(completion_timing(f)['historical_full_completion_timing'], 'UNKNOWN')
        f.pop('command')
        self.assertEqual(completion_timing(f)['settlement_timing'], 'NOT_APPLICABLE_NO_COMMAND')


if __name__ == '__main__': unittest.main()
