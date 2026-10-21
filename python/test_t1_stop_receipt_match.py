import unittest
from t1_stop_receipt_match import match_state_sample


class OrderedMatchTests(unittest.TestCase):
    def fixture(self):
        state=dict(host_receipt_monotonic_s=10.,state_timestamp_ns=1,velocity_ned_m_s=[0.,0.,0.],position_world_ned_m=[0.,0.,-3.])
        sample=dict(receipt_monotonic_s=10.,state_timestamp_ns=1,velocity_xy_m_s=[0.,0.],position_world_ned_m=[0.,0.,-3.],
            actual_last_command_dispatch_monotonic_s=10.,actual_last_command_xy_m_s=[0.,0.])
        commands=[(0,dict(dispatch_monotonic_s=9.95,velocity_xy_m_s=[0.,0.])),(2,dict(dispatch_monotonic_s=10.,velocity_xy_m_s=[0.,0.]))]
        return sample,[(1,dict(state)),(3,dict(state))],commands

    def test_same_clock_same_state_resolved_by_actual_pre_query_command(self):
        sample,states,commands=self.fixture();r=match_state_sample(sample,states,commands)
        self.assertEqual(r['candidate_event_indexes'],[1,3]);self.assertEqual(r['compatible_event_indexes'],[3])
        self.assertFalse(r['time_or_velocity_changed'])

    def test_recorded_dispatch_without_actual_pre_query_command_stays_unverified(self):
        sample,states,commands=self.fixture();sample['actual_last_command_dispatch_monotonic_s']=10.1
        self.assertEqual(match_state_sample(sample,states,commands)['status'],'NO_MATCHING_ACTUAL_PRE_QUERY_COMMAND')

    def test_state_or_velocity_difference_cannot_be_patched_into_a_match(self):
        sample,states,commands=self.fixture();sample['velocity_xy_m_s']=[.001,0.]
        self.assertFalse(match_state_sample(sample,states,commands)['compatible_event_indexes'])


if __name__=='__main__':unittest.main()
