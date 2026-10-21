"""A moving sample cannot claim a command issued after its state query."""
import unittest
from test_t1_async_control import ManualClock
from test_t1_async_control_v1p2 import FeedbackFake
from t1_async_control_v1p4 import AsyncControlV1P4
from test_t1_settlement_plan import plan as prior_plan
from t1_settlement_plan_v1p1 import TRANSITION
from t1_settlement_plan_v1p2 import ARCHITECTURE, validate_plan


class ReceiptTests(unittest.TestCase):
    def test_every_saved_sample_matches_actual_pre_query_command_including_clock_ties(self):
        clock=ManualClock(); logs=[]; f=FeedbackFake(logs,clock=clock)
        c=AsyncControlV1P4(lambda:f,lambda:None,clock=clock)
        command=f.issue([.009,0.],.12,'CURRENT_TARGET_TRAJECTORY'); c._activate_stop('one',command,'frame')
        f.stuck_velocity=True
        for _ in range(14):
            f.velocity=[.0036,0.,0.]; clock.advance(.05); c._stop_tick(f,c._read(f))
        self.assertFalse(c.receipts())
        self.assertTrue(c._active['window'].samples)
        for s in c._active['window'].samples:
            index=next(i for i,e in enumerate(logs) if e.get('kind')=='SELF_STATE'
                and e['host_receipt_monotonic_s']==s['receipt_monotonic_s'] and e['state_timestamp_ns']==s['state_timestamp_ns'])
            before=[e for e in logs[:index] if e.get('kind')=='COMMAND'][-1]
            self.assertEqual(before['velocity_xy_m_s'],s['actual_last_command_xy_m_s'])
            self.assertEqual(before['dispatch_monotonic_s'],s['actual_last_command_dispatch_monotonic_s'])

    def test_complete_longer_grid_keeps_all_original_thresholds(self):
        p=prior_plan(); p.update(control_architecture=ARCHITECTURE,acquisition_transition_policy=TRANSITION,
            state_command_receipt_policy='actual last command before SELF_STATE query; later feedback cannot be relabeled as prior')
        self.assertEqual(validate_plan(p)['expected_main_slots'],50)
        p.update(planned_duration_s=60.,phases=[dict(phase_id='visible',duration_s=60.)],
            planned_slots=[dict(slot_index=i,phase_id='visible',scheduled_start_s=round(i*.4,9),
                scheduled_end_s=round((i+1)*.4,9)) for i in range(150)])
        p['motion_scale_validation'].update(planned_measured_slots=150,continuous_visible_duration_s=60.)
        self.assertEqual(validate_plan(p)['expected_main_slots'],150)
        p['motion_scale_validation']['minimum_endpoint_net_xy_displacement_m']=.008
        with self.assertRaises(ValueError): validate_plan(p)


if __name__=='__main__': unittest.main()
