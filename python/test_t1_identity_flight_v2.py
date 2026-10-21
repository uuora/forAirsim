"""Command reply observability and declared stall protocol checks."""
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import airsim
from t1_identity_flight_v2 import Flight
from prepare_t1_identity_flight_v2 import prepare_async

class Future:
    error=None
    result=None
    def attach_callback(self,callback):self.callback=callback
    def finish(self,result=None,error=None):
        self.result=result;self.error=error;self.callback(self)

class Client:
    def __init__(self):self.future=Future();self.calls=[]
    def moveByVelocityZAsync(self,*args,**kwargs):
        self.calls.append((args,kwargs));return self.future
    def getMultirotorState(self,**kwargs):return SimpleNamespace()

class ReplyTests(unittest.TestCase):
    def setUp(self):
        self.client=Client();self.logs=[]
        self.flight=Flight(self.client,[0.,-3.,-1.],-3.,[-.3,-3.3,-3.15],[.3,-2.7,-2.85],self.logs.append)
    def issue(self):
        now=time.monotonic()
        return self.flight.issue([.01,0.],.12,'CURRENT_TARGET_TRAJECTORY',gate={
            'latest_dispatch_monotonic_s':now+.12,'command_end_deadline_monotonic_s':now+.4,
            'current_binding':{'frame_id':'current'}})
    def test_rpc_reply_is_distinct_from_dispatch_and_bound_to_command(self):
        command=self.issue();self.assertEqual(len(self.logs),1)
        self.client.future.finish(True)
        self.assertEqual(self.logs[-1]['kind'],'COMMAND_RPC_RESULT')
        self.assertEqual(self.logs[-1]['command_id'],command['command_id'])
        self.assertEqual(self.logs[-1]['status'],'RETURNED_TRUE')
    def test_task_replacement_false_is_recorded_without_inventing_success(self):
        self.issue();self.client.future.finish(False)
        self.assertEqual(self.logs[-1]['status'],'RETURNED_FALSE')
        self.assertFalse(self.flight.rpc_fault)
    def test_rpc_error_category_triggers_failure_and_does_not_expose_body(self):
        self.issue();self.client.future.finish(error=RuntimeError('private endpoint or key'))
        self.assertEqual(self.logs[-1]['status'],'ERROR')
        self.assertNotIn('private',str(self.logs[-1]))
        with self.assertRaisesRegex(RuntimeError,'OWNED_COMMAND_RPC_ERROR'):self.flight.state()
    def test_expired_gate_dispatch_has_no_rpc_call_or_callback(self):
        with self.assertRaises(RuntimeError):
            self.flight.issue([.01,0.],.12,'CURRENT_TARGET_TRAJECTORY',gate={
                'latest_dispatch_monotonic_s':time.monotonic()-.01})
        self.assertEqual(self.client.calls,[])
    def test_receipt_failure_does_not_escape_rpc_future_callback(self):
        self.issue()
        def failing_log(row):raise RuntimeError('CONTROL_EVENT_BUFFER_OVERFLOW')
        self.flight.log=failing_log
        self.client.future.finish(True)
        self.assertTrue(self.flight.rpc_fault)
    def test_recovery_can_read_state_after_historical_command_rpc_fault(self):
        self.flight.rpc_fault=True
        zero=airsim.Vector3r()
        raw=SimpleNamespace(timestamp=1,landed_state=airsim.LandedState.Flying,
            kinematics_estimated=SimpleNamespace(position=zero,linear_velocity=zero,
                linear_acceleration=zero,orientation=airsim.Quaternionr()))
        self.client.getMultirotorState=lambda **kwargs:raw
        with self.assertRaises(RuntimeError):self.flight.state()
        state=self.flight.state(recovery=True)
        self.assertEqual(state['position_world_ned_m'],[0.,-3.,-1.])

class ProtocolTests(unittest.TestCase):
    def test_undeclared_stall_rejected_before_files_or_model_copy(self):
        with patch('prepare_t1_identity_flight_v2.prepare') as base:
            with self.assertRaises(ValueError):prepare_async(Path('unused'),'short','id',delay_s=.6)
            base.assert_not_called()
    def test_occlusion_stall_cannot_be_injected_in_zero_gate_run(self):
        with patch('prepare_t1_identity_flight_v2.prepare') as base:
            with self.assertRaises(ValueError):prepare_async(Path('unused'),'zero','id',delay_s=.65)
            base.assert_not_called()

if __name__=='__main__':unittest.main()
