import unittest
from unittest.mock import Mock,patch
from braking_response_experiment_v1p3 import SavedSpanBrakingExperiment


class SavedSpanTests(unittest.TestCase):
    def test_loop_time_cannot_substitute_for_query_span(self):
        run=object.__new__(SavedSpanBrakingExperiment);run.measurement=dict(mode='zero_hold_gate',stop_stable_s=.5,brake_timeout_s=6.)
        run.cfg={'initial_state_match':{'speed_tolerance_m_s':.01}};run.report={'zero_gate_samples':[]};run.vehicle='DroneA'
        run.prepare=Mock();run.dispatch=Mock();run.client=Mock();run.save=Mock()
        values=iter([10.,10.2,10.4,10.6])
        def query(stage):
            row=dict(host_monotonic_s=next(values),speed_norm_m_s=0.,protection_reasons=[])
            run.report['zero_gate_samples'].append(row);return row
        run.query=query
        with patch('braking_response_experiment_v1p3.time.monotonic',side_effect=[20.,20.8,21.,21.2]):run.approach()
        self.assertEqual(run.dispatch.call_count,3);self.assertGreaterEqual(run.report['zero_gate_saved_query_span_s'],.5)
        self.assertTrue(run.report['measurement_passed'])


if __name__=='__main__':unittest.main()
