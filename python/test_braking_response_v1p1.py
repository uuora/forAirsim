import copy
import json
from pathlib import Path
import unittest
from braking_response_v1p1 import acquired_velocity_interval,validate_acquisition


class AcquisitionTests(unittest.TestCase):
    def fixture(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/braking_response_v1/low_gate.json').read_text(encoding='utf-8'))
        cfg['braking_measurement'].update(entry_acquisition_timeout_s=3.,entry_stable_s=.25)
        m=validate_acquisition(cfg); pulse=m['pulses'][0]
        rows=[dict(host_monotonic_s=10+i*.1,query_state_timestamp_ns=int((10+i*.1)*1e9),
            velocity_ned_m_s=[.025,0,0],protection_reasons=[]) for i in range(4)]
        return cfg,m,pulse,rows

    def test_fresh_stable_suffix_is_required(self):
        _,m,p,rows=self.fixture()
        self.assertFalse(acquired_velocity_interval(rows[:1],p,m)['accepted'])
        self.assertTrue(acquired_velocity_interval(rows,p,m)['accepted'])
        rows[-1]['velocity_ned_m_s']=[.0206757765263319,0,0]
        self.assertFalse(acquired_velocity_interval(rows,p,m)['accepted'])

    def test_gap_protection_and_clock_rollback_reset_acquisition(self):
        for kind in ('gap','protection','clock'):
            _,m,p,rows=self.fixture()
            if kind=='gap':rows[-1]['host_monotonic_s'] += .3
            if kind=='protection':rows[-1]['protection_reasons']=['REJECTED']
            if kind=='clock':rows[-1]['query_state_timestamp_ns']=1
            with self.subTest(kind=kind):self.assertFalse(acquired_velocity_interval(rows,p,m)['accepted'])

    def test_timeout_and_stable_range_cannot_be_relaxed(self):
        for name,value in [('entry_acquisition_timeout_s',4.),('entry_stable_s',.1)]:
            cfg,_,_,_=self.fixture();cfg['braking_measurement'][name]=value
            with self.assertRaises(ValueError):validate_acquisition(cfg)


if __name__=='__main__':unittest.main()
