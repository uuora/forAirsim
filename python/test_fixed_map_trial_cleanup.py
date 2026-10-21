"""Terminal persistence and sensor-only cleanup fault tests; no live flight."""
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fixed_map_trial_cleanup import finalize_trial,wait_for_ground_release
from fixed_map_perception_process import PerceptionProcess


def ground(passed=True):
    return dict(passed=passed,vehicles={n:dict(landed=passed,api_control=not passed,fresh=True) for n in ['DroneA','DroneB']},errors=[])


class TerminalCleanupTests(unittest.TestCase):
    def setup_case(self,root):
        (root/'session_events.jsonl').write_text(json.dumps(dict(kind='LANGUAGE_SENSOR_APPROACH_HOVER_PASSED'))+'\n')
        executor=SimpleNamespace(state={},is_alive=MagicMock(return_value=False),join=MagicMock(),exit_requested=False)
        coordinator=SimpleNamespace(phase='LANDED_RELEASED',status=lambda:dict(outcome='APPROACH_HOVER_PASSED'))
        service=SimpleNamespace(process=object(),stop=MagicMock(return_value=dict(status='PASS')),reap_after_ground_release=MagicMock(return_value=dict(actions=[],exit_code=0)))
        return executor,coordinator,service
    def test_normal_exit_persists_pending_before_wait_and_final_after(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);e,c,s=self.setup_case(root)
            def stop(**kwargs):
                pending=json.loads((root/'result.json').read_text())
                self.assertEqual(pending['terminal_phase'],'POSTFLIGHT_SENSOR_CLEANUP_PENDING');self.assertFalse(pending['perception_guidance_hover_passed'])
                return dict(status='PASS')
            s.stop.side_effect=stop
            result=finalize_trial(root,e,c,s,None,True,lambda:ground())
            self.assertTrue(result['perception_guidance_hover_passed']);self.assertEqual(result['terminal_phase'],'FINAL');s.reap_after_ground_release.assert_not_called()
    def test_slow_exit_timeout_is_a_failed_original_terminal(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);e,c,s=self.setup_case(root);s.stop.side_effect=RuntimeError('PERCEPTION_PROCESS_STOP_TIMEOUT')
            result=finalize_trial(root,e,c,s,None,True,lambda:ground())
            self.assertEqual(result['issue'],'PERCEPTION_PROCESS_STOP_TIMEOUT');self.assertFalse(result['perception_guidance_hover_passed']);self.assertEqual(result['result_record_provenance'],'ORIGINAL_DRIVER_OUTPUT_V18');s.reap_after_ground_release.assert_called_once()
            self.assertTrue((root/'final_ground_receipt.json').exists());self.assertEqual(json.loads((root/'result.json').read_text())['issue'],result['issue'])
    def test_nonzero_exit_does_not_skip_ground_or_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);e,c,s=self.setup_case(root);s.stop.side_effect=RuntimeError('PERCEPTION_PROCESS_EXIT_NONZERO_7')
            probe=MagicMock(return_value=ground());result=finalize_trial(root,e,c,s,None,True,probe)
            self.assertFalse(result['perception_guidance_hover_passed']);self.assertIn('NONZERO',result['issue']);self.assertGreaterEqual(probe.call_count,2)
    def test_rpc_exception_is_unknown_not_grounded_and_owner_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);e,c,s=self.setup_case(root);e.is_alive.return_value=True
            result=finalize_trial(root,e,c,s,None,True,MagicMock(side_effect=RuntimeError('RPC_DOWN')))
            self.assertFalse(result['ground_verification_passed']);self.assertFalse(result['perception_guidance_hover_passed']);self.assertEqual(result['terminal_phase'],'FINAL_UNVERIFIED_GROUND');self.assertEqual(result['final_ground'],{});self.assertFalse(e.exit_requested);e.join.assert_not_called()
    def test_requested_flight_guard_keeps_waiting_until_ground_release(self):
        with tempfile.TemporaryDirectory() as temp:
            e=SimpleNamespace(stage='FAILED_HOVER',is_alive=lambda:True,command=MagicMock())
            probe=MagicMock(side_effect=[ground(False),ground(False),ground()]);sleep=MagicMock()
            receipt=wait_for_ground_release(temp,e,'RECOVERY',probe,sleep)
            self.assertTrue(receipt['passed']);self.assertEqual(probe.call_count,3);self.assertEqual(sleep.call_count,2)
            self.assertEqual([c.args[0] for c in e.command.call_args_list],['stop','return','stop','return'])
            self.assertTrue(json.loads((Path(temp)/'terminal_recovery_pending.json').read_text())['driver_process_retained'])
    def process_case(self,root):
        p=object.__new__(PerceptionProcess);p.output=root/'online';p.output.mkdir();p.publication_metrics=[dict(frame_id='sample')];p.done=threading.Event();p.threads=[];p.process=MagicMock();p.process.pid=123;p.process.poll.return_value=None;p.stream=MagicMock()
        return p
    def test_process_timeout_flushes_publication_metrics_and_error_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=self.process_case(root);p.process.wait.side_effect=subprocess.TimeoutExpired('sensor',15)
            with self.assertRaisesRegex(RuntimeError,'STOP_TIMEOUT'):p.stop()
            self.assertEqual(json.loads((root/'online_stop_receipt.json').read_text())['status'],'TIMEOUT');self.assertEqual(json.loads((root/'online_publication_metrics.json').read_text()),p.publication_metrics);self.assertTrue(p.done.is_set());p.process.terminate.assert_not_called()
    def test_process_nonzero_exit_flushes_failure_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);p=self.process_case(root);p.process.wait.return_value=7;p.process.poll.return_value=7
            with self.assertRaisesRegex(RuntimeError,'EXIT_NONZERO_7'):p.stop()
            self.assertEqual(json.loads((root/'online_stop_receipt.json').read_text())['status'],'ERROR');p.stream.close.assert_called_once()
    def test_reap_refuses_airborne_or_incomplete_ground_evidence(self):
        for receipt in [ground(False),dict(passed=True,vehicles={}),dict(passed=True,vehicles={'DroneA':dict(landed=True,api_control=False,fresh=True)})]:
            with tempfile.TemporaryDirectory() as temp:
                p=self.process_case(Path(temp))
                with self.assertRaisesRegex(ValueError,'FRESH_RELEASED_GROUND'):p.reap_after_ground_release(lambda:receipt)
                p.process.terminate.assert_not_called();p.process.kill.assert_not_called()
    def test_reap_uses_bounded_sensor_terminate_then_kill(self):
        with tempfile.TemporaryDirectory() as temp:
            p=self.process_case(Path(temp));p.process.poll.side_effect=[None,9];p.process.wait.side_effect=[subprocess.TimeoutExpired('sensor',3),9]
            receipt=p.reap_after_ground_release(lambda:ground())
            self.assertEqual(receipt['actions'],['terminate_sensor_child','kill_sensor_child']);self.assertEqual(p.process.wait.call_args_list[0].kwargs['timeout'],3);p.stream.close.assert_called_once()


if __name__=='__main__':unittest.main()
