"""An audit keeps missing proof/failed scale and the historical late deadline."""
import json
from pathlib import Path
import unittest

from analyze_t1_settlement import timing_rows, feedback_audit
from t1_settlement_plan_v1 import ARCHITECTURE


class SettlementAuditTests(unittest.TestCase):
    def test_real_old_full_deadline_and_limiter_are_preserved(self):
        path=Path(__file__).resolve().parents[1]/'logs/t1_motion_visible20_v6p1_20261006/flight/report.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        rows=timing_rows(report)
        self.assertEqual(len(rows),50)
        self.assertTrue(all(r['full_limiter']=='CONTROL_SETTLEMENT' for r in rows))
        self.assertTrue(all(r['recorded_timing']['frame_processing']['status']=='TIMELY' for r in rows))
        self.assertTrue(all(r['recorded_timing']['historical_full']['status']=='LATE' for r in rows))
        self.assertFalse(any('new_predeclared_timing' in r for r in rows))

    def test_new_timing_keeps_late_historical_clock(self):
        row=dict(frame_id='one',slot_index=0,due_monotonic_s=10.,deadline_monotonic_s=10.4,
            frame_processing_completed_monotonic_s=10.3,decision_completed_monotonic_s=10.7,
            command=dict(dispatch_monotonic_s=10.25,duration_s=.12),lease_stop=dict(completed_monotonic_s=10.7))
        r=timing_rows(dict(control_architecture=ARCHITECTURE,frames=[row]))[0]
        self.assertEqual(r['new_predeclared_timing']['historical_full_completion_timing'],'LATE')

    def test_missing_feedback_and_stop_proofs_cannot_pass(self):
        events=[dict(kind='COMMAND',stage='PERCEPTION_XY',purpose='OWN_STATE_FEEDBACK_BRAKE',
            command_id='one',dispatch_monotonic_s=10.,velocity_xy_m_s=[0.,0.])]
        r=feedback_audit({}, {}, events)
        self.assertEqual(r['status'],'FAIL_STRICT_FEEDBACK_STOP_AUDIT')
        self.assertGreaterEqual(len(r['issues']),2)


if __name__=='__main__': unittest.main()
