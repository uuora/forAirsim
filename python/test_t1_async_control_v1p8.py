import copy
import unittest
from test_t1_async_control import ManualClock
from test_t1_async_control_v1p2 import FeedbackFake
from t1_async_control_v1p8 import AsyncControlV1P8
from t1_interrupted_clear_audit_v1 import audit


class CloseTests(unittest.TestCase):
    def setup_control(self):
        clock=ManualClock();logs=[];flight=FeedbackFake(logs,clock=clock)
        controller=AsyncControlV1P8(lambda:flight,lambda:None,clock=clock)
        original=flight.state
        def state():clock.advance(.05);return original()
        flight.state=state
        return clock,logs,flight,controller
    def active(self):
        clock,logs,f,c=self.setup_control()
        command=f.issue([.009,0.],.12,'CURRENT_TARGET_TRAJECTORY')
        c._activate_stop('control-1',command,'frame-1')
        clock.advance(.125)
        zero=f.issue([0.,0.],.12,'OWN_STATE_BRAKE_OR_HOLD')
        c._active['clear']=zero['dispatch_monotonic_s']
        return clock,logs,f,c,command,zero
    def test_first_target_zero_survives_later_close_window_reset(self):
        clock,logs,f,c,target,first=self.active();clock.advance(.45)
        c._shutdown(f);lease=c.receipts()['control-1'];close=c._shutdown_receipt
        self.assertEqual(lease['target_setpoint_cleared_monotonic_s'],first['dispatch_monotonic_s'])
        self.assertGreater(lease['shutdown_window_clear_monotonic_s'],first['dispatch_monotonic_s'])
        self.assertEqual(close['target_setpoint_cleared_monotonic_s'],lease['shutdown_window_clear_monotonic_s'])
        self.assertTrue(lease['first_target_clear_preserved'])
    def test_original_lease_stop_bound_is_preserved(self):
        clock,logs,f,c,target,first=self.active();clock.advance(.45);c._shutdown(f)
        lease=c.receipts()['control-1']
        self.assertEqual(lease['settlement_deadline_monotonic_s'],target['dispatch_monotonic_s']+.12+2.)
        self.assertLessEqual(lease['completed_monotonic_s'],lease['settlement_deadline_monotonic_s'])
    def test_new_window_still_observes_030_seconds_and_distinct_states(self):
        clock,logs,f,c,target,first=self.active();c._shutdown(f);r=c._shutdown_receipt
        self.assertGreaterEqual(r['samples'][-1]['receipt_monotonic_s']-r['shutdown_window_clear_monotonic_s'],.3-1e-9)
        self.assertGreaterEqual(r['distinct_stable_simulator_state_timestamps'],3)
        self.assertEqual(r['configured_settlement']['stationary_xy_threshold_m_s'],.0005)
    def test_no_prior_clear_uses_actual_close_zero(self):
        clock,logs,f,c=self.setup_control();command=f.issue([.009,0.],.12,'CURRENT_TARGET_TRAJECTORY')
        c._activate_stop('control-1',command,'frame-1');c._shutdown(f);r=c.receipts()['control-1']
        self.assertFalse(r['first_target_clear_preserved'])
        self.assertEqual(r['target_setpoint_cleared_monotonic_s'],r['shutdown_window_clear_monotonic_s'])
    def test_already_expired_original_bound_cannot_be_extended_by_close(self):
        clock,logs,f,c,target,first=self.active();clock.advance(2.1)
        with self.assertRaisesRegex(RuntimeError,'ORIGINAL_TARGET_CLOSE_STOP_TIMEOUT'):c._shutdown(f)
        self.assertIsNone(c._shutdown_receipt);self.assertFalse(c.receipts())
    def test_residual_0036_velocity_cannot_pass_close(self):
        clock,logs,f,c,target,first=self.active();f.stuck_velocity=True;f.velocity=[.0036,0.,0.]
        with self.assertRaises(RuntimeError):c._shutdown(f)
        self.assertIsNone(c._shutdown_receipt)
    def test_empty_active_close_keeps_its_own_strict_receipt(self):
        clock,logs,f,c=self.setup_control();c._shutdown(f)
        self.assertEqual(c._shutdown_receipt['status'],'OWN_STATE_SAMPLED_STOPPED_AT_CONTROLLER_CLOSE')
        self.assertIsNone(c._shutdown_receipt['original_active_lease_stop_deadline_monotonic_s'])
    def test_no_new_target_command_during_close(self):
        clock,logs,f,c,target,first=self.active();c._shutdown(f)
        self.assertEqual(sum(e.get('kind')=='COMMAND' and e.get('purpose')=='CURRENT_TARGET_TRAJECTORY' for e in logs),1)


class SupplementTests(unittest.TestCase):
    def fixture(self):
        target=dict(kind='COMMAND',stage='PERCEPTION_XY',command_id='target',purpose='CURRENT_TARGET_TRAJECTORY',
            velocity_xy_m_s=[.009,0.],dispatch_monotonic_s=10.,target_end_deadline_monotonic_s=10.48)
        first=dict(kind='COMMAND',stage='PERCEPTION_XY',command_id='first',purpose='OWN_STATE_BRAKE_OR_HOLD',
            velocity_xy_m_s=[0.,0.],dispatch_monotonic_s=10.125)
        close=dict(first,command_id='close',purpose='OWN_STATE_CONTROLLER_CLOSE_ZERO',dispatch_monotonic_s=10.765)
        row=dict(frame_id='frame-1',slot_index=387,ticket_id='control-1',command=copy.deepcopy(target),
            lease_stop=dict(lease_interrupted_by_close=True,target_setpoint_cleared_monotonic_s=10.765))
        return dict(frames=[row]),[target,first,close]
    def test_supplement_retains_raw_report_and_overall_failure(self):
        report,events=self.fixture();before=copy.deepcopy(report);r=audit(report,events)
        self.assertEqual(r['status'],'PASS_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION')
        self.assertEqual(report,before);self.assertFalse(r['overall_experiment_qualification_changed'])
        self.assertFalse(r['rows'][0]['saved_clock_is_first_actual_target_clear'])
    def test_actual_late_first_clear_stays_failed(self):
        report,events=self.fixture();events[1]['dispatch_monotonic_s']=10.55
        self.assertEqual(audit(report,events)['status'],'FAIL_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION')
    def test_following_target_cannot_supply_a_zero_for_previous_lease(self):
        report,events=self.fixture();events.insert(1,dict(events[0],command_id='different'))
        self.assertEqual(audit(report,events)['status'],'FAIL_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION')
    def test_command_payload_tamper_is_rejected(self):
        report,events=self.fixture();events[0]['dispatch_monotonic_s']=11.
        self.assertEqual(audit(report,events)['status'],'FAIL_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION')


if __name__=='__main__':unittest.main()
