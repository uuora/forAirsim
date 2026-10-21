"""Late perception cannot consume the budget required for actual target clearing."""
import unittest
from test_t1_identity_motion_policy import DESCRIPTION,bound_fixture,rebind_surface
from t1_identity_motion_policy_v1p1 import IdentityMotionPolicyV1P1


class DispatchReserveTests(unittest.TestCase):
    def decide(self, now):
        frame,selection,surface=bound_fixture(receipt=10.,ready=10.02)
        selection['semantic_review_deadline_monotonic_s']=18.;rebind_surface(selection,surface)
        return IdentityMotionPolicyV1P1('session',1,DESCRIPTION).decide(frame=frame,selection=selection,surface=surface,
            desired_xy_m_s=[.009,0.],actual_velocity_xy_m_s=[0.,0.],self_position_ned_m=[0.,0.,-3.],
            previous_command_xy_m_s=[0.,0.],previous_command_monotonic_s=now-.1,
            self_state_receipt_monotonic_s=now,now=now)

    def test_typical_frame_keeps_lease_and_original_expiry(self):
        g=self.decide(10.15)
        self.assertEqual(g['status'],'ALLOW_BOUNDED_COMMAND');self.assertEqual(g['command_duration_s'],.12)
        self.assertAlmostEqual(g['command_end_deadline_monotonic_s'],10.48)
        self.assertAlmostEqual(g['latest_dispatch_monotonic_s']+g['command_duration_s'],10.36)

    def test_smaller_budget_shortens_lease_before_original_deadline(self):
        g=self.decide(10.32)
        self.assertEqual(g['status'],'ALLOW_BOUNDED_COMMAND');self.assertAlmostEqual(g['command_duration_s'],.04)
        self.assertAlmostEqual(g['latest_declared_lease_end_monotonic_s'],10.36)

    def test_observed_late_frame_is_withdrawn_with_freshness_unchanged(self):
        g=self.decide(10.406)
        self.assertEqual(g['status'],'WAIT_NO_TARGET')
        self.assertEqual(g['reason'],'INSUFFICIENT_CURRENT_FRAME_BUDGET_FOR_LEASE_AND_ACTUAL_CLEAR')
        self.assertFalse(g['target_command_authorized'])
        self.assertAlmostEqual(g['command_end_deadline_monotonic_s'],10.48)


if __name__=='__main__':unittest.main()
