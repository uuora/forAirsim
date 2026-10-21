import unittest
from types import SimpleNamespace
from t1_lifecycle_client_v1 import VerifiedLifecycleClient

class LifecycleTests(unittest.TestCase):
    def test_get_consumes_true_but_join_alone_is_not_used(self):
        events=[];calls=[];future=SimpleNamespace(get=lambda:calls.append('get') or True,join=lambda:calls.append('join'))
        client=VerifiedLifecycleClient(SimpleNamespace(moveToZAsync=lambda *a,**k:future),events.append)
        self.assertTrue(client.moveToZAsync(-3,.3).join());self.assertEqual(calls,['get'])
        self.assertEqual(events[-1]['status'],'RETURNED_TRUE')
    def test_false_task_result_fails_before_settling(self):
        events=[];client=VerifiedLifecycleClient(SimpleNamespace(takeoffAsync=lambda **k:SimpleNamespace(get=lambda:False)),events.append)
        with self.assertRaisesRegex(RuntimeError,'NOT_ACKNOWLEDGED'):client.takeoffAsync().join()
        self.assertEqual(events[-1]['status'],'REJECTED_OR_CANCELLED')
    def test_stored_future_error_is_raised_without_body_in_trace(self):
        def error():raise RuntimeError('private transport body')
        events=[];client=VerifiedLifecycleClient(SimpleNamespace(landAsync=lambda **k:SimpleNamespace(get=error)),events.append)
        with self.assertRaises(RuntimeError):client.landAsync().join()
        self.assertEqual(events[-1]['status'],'ERROR');self.assertNotIn('private',str(events))
    def test_short_velocity_and_state_calls_pass_through(self):
        method=lambda *a,**k:'short';client=VerifiedLifecycleClient(SimpleNamespace(moveByVelocityZAsync=method),lambda r:None)
        self.assertIs(client.moveByVelocityZAsync,method)

if __name__=='__main__':unittest.main()
