"""Phase scopes cannot mix equal local slot indexes or acquisition movement."""
import unittest
from analyze_t1_delayed_identity import phase_evidence,scoped_events


def state(t,x):return dict(kind='SELF_STATE',stage='PERCEPTION_XY',host_receipt_monotonic_s=t,
    position_world_ned_m=[x,0,-3],velocity_ned_m_s=[0,0,0],acceleration_ned_m_s2=[0,0,0],state_timestamp_ns=int(t*1e9))


class PhaseAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.events=[state(10,0),state(11,10),state(12,20),state(13,21)]
        self.report=dict(configuration={},injected_delays=[dict(started_monotonic_s=12.1,completed_monotonic_s=12.7)],
            acquisition_stage_start_state=self.events[0],acquisition_stage_end_state=self.events[1],
            low_speed_stage_start_state=self.events[2],low_speed_stage_end_state=self.events[3],
            acquisition_frames=[dict(frame_id='a0',slot_index=0)],acquisition_slots=[dict(slot_index=0)],
            frames=[dict(frame_id='m0',slot_index=0)],slots=[dict(slot_index=0)])
    def test_overlapping_local_slots_stay_separate(self):
        acquired,_=phase_evidence(self.report,self.events,acquisition=True)
        measured,_=phase_evidence(self.report,self.events)
        self.assertEqual(acquired['frame_timing']['rows'][0]['frame_id'],'a0')
        self.assertEqual(measured['frame_timing']['rows'][0]['frame_id'],'m0')
        self.assertEqual(acquired['frame_timing']['saved_frames'],1)
        self.assertEqual(measured['frame_timing']['saved_frames'],1)
    def test_acquisition_path_is_excluded_from_measured_path(self):
        _,acquired=phase_evidence(self.report,self.events,acquisition=True)
        _,measured=phase_evidence(self.report,self.events)
        self.assertEqual(acquired['sampled_xy_path_length_m'],10.)
        self.assertEqual(measured['sampled_xy_path_length_m'],1.)
        self.assertEqual(measured['measured_host_receipt_span_s'],1.)
    def test_measured_stall_not_counted_in_acquisition(self):
        acquired,_=phase_evidence(self.report,self.events,acquisition=True)
        measured,_=phase_evidence(self.report,self.events)
        self.assertEqual(acquired['injected_delay_rows'],[])
        self.assertEqual(len(measured['injected_delay_rows']),1)
    def test_unentered_phase_is_not_fabricated_from_acquisition_queries(self):
        self.report.pop('low_speed_stage_start_state');self.report.pop('low_speed_stage_end_state')
        self.assertEqual(phase_evidence(self.report,self.events),(None,None))
        self.assertEqual(scoped_events(self.events,None,None),[])


if __name__=='__main__':unittest.main()
