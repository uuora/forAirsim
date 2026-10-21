import copy
import unittest
from braking_response_v1p2 import summarize_pulse
from test_braking_response import fixture


class OriginSelectionTests(unittest.TestCase):
    def fixture(self):
        m,p,model=fixture();p['samples'].insert(0,p['brake_first_request']['origin_sample'])
        p['brake_first_request']['dispatch_host_s']=p['brake_first_request']['origin_sample']['host_monotonic_s']
        return m,p,model

    def test_equal_origin_and_request_host_times_do_not_repeat_origin(self):
        m,p,model=self.fixture();r=summarize_pulse(p,m,model)
        self.assertTrue(r['counts_as_measured_stopping']);self.assertTrue(r['chronology_verified'])
        self.assertAlmostEqual(r['maximum_recorded_forward_displacement_m'],.0034)

    def test_missing_or_duplicate_origin_never_verified(self):
        for kind in ('missing','duplicate'):
            m,p,model=self.fixture()
            if kind=='missing':p['samples'].pop(0)
            else:p['samples'].insert(0,copy.deepcopy(p['samples'][0]))
            with self.subTest(kind=kind):self.assertFalse(summarize_pulse(p,m,model)['counts_as_measured_stopping'])

    def test_post_origin_rollback_or_duplicate_query_never_verified(self):
        for kind in ('before_request','same_host','same_state'):
            m,p,model=self.fixture()
            if kind=='before_request':p['samples'][1]['host_monotonic_s']=9.99
            if kind=='same_host':p['samples'][2]['host_monotonic_s']=p['samples'][1]['host_monotonic_s']
            if kind=='same_state':p['samples'][2]['query_state_timestamp_ns']=p['samples'][1]['query_state_timestamp_ns']
            with self.subTest(kind=kind):self.assertFalse(summarize_pulse(p,m,model)['counts_as_measured_stopping'])


if __name__=='__main__':unittest.main()
