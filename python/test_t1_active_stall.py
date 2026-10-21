"""Require actual bounded motion receipts, without extending a target lease."""
import copy
import unittest
from t1_active_stall import active_stall_eligibility


class StallEligibilityTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(frame_id='f1', ticket_id='control-1',
            motion_gate=dict(status='ALLOW_BOUNDED_COMMAND', target_command_authorized=True),
            command=dict(purpose='CURRENT_TARGET_TRAJECTORY', target_frame_id='f1',
                velocity_xy_m_s=[.004, 0.], dispatch_monotonic_s=10., duration_s=.12,
                command_id='cmd1'))

    def test_real_bounded_dispatch_eligible_without_modifying_receipt(self):
        before = copy.deepcopy(self.row)
        audit = active_stall_eligibility(self.row, 10.02)
        self.assertTrue(audit['eligible'])
        self.assertAlmostEqual(audit['remaining_declared_lease_s'], .10)
        self.assertEqual(self.row, before)

    def test_zero_self_hold_brake_and_unauthorized_command_do_not_trigger(self):
        for changes in ({'velocity_xy_m_s': [0., 0.]}, {'purpose': 'OWN_STATE_ZERO'},
                        {'duration_s': .13}, {'velocity_xy_m_s': [.031, 0.]},
                        {'target_frame_id': 'different'}):
            row = copy.deepcopy(self.row); row['command'].update(changes)
            self.assertFalse(active_stall_eligibility(row, 10.02)['eligible'])
        row = copy.deepcopy(self.row); row['motion_gate']['target_command_authorized'] = False
        self.assertFalse(active_stall_eligibility(row, 10.02)['eligible'])

    def test_ended_future_short_remaining_or_cleared_lease_does_not_trigger(self):
        for now in (9.99, 10.09, 10.12, float('nan'), True):
            self.assertFalse(active_stall_eligibility(self.row, now)['eligible'])
        self.assertFalse(active_stall_eligibility(self.row, 10.02, cleared_tickets=['control-1'])['eligible'])

    def test_single_predeclared_stall(self):
        self.assertFalse(active_stall_eligibility(self.row, 10.02, already_triggered=True)['eligible'])


if __name__ == '__main__':
    unittest.main()
