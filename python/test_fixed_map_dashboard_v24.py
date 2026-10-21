"""No-flight checks for the desktop owner boundary and prelaunch cancellation."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QApplication

from fixed_map_dashboard_v24 import Dashboard
from fixed_map_cpu_runtime import configure_math_runtime
configure_math_runtime(dict(math_thread_budget_v20=True, torch_threads=2))

ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    def __init__(self):
        self.output = ROOT / 'reports/test_interface_v24'
        self.snapshot = dict(running=False, bootstrap=dict(status='READY', ground=dict(passed=True)),
                             engine_info=dict(pid=12), stage='READY')
        self.stops = 0
        self.closes = 0
    def poll(self): return self.snapshot
    def request_stop(self): self.stops += 1
    def close(self): self.closes += 1
    def bootstrap(self): raise AssertionError('Unexpected bootstrap')
    def refresh_scene(self): raise AssertionError('Unexpected scene mutation')
    def start_trial(self, text, cancel_event=None): raise AssertionError('Unexpected flight launch')


class WindowBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app = QApplication.instance() or QApplication([])
    def setUp(self):
        self.backend = FakeSession()
        self.window = Dashboard(self.backend, bootstrap_on_open=False)
    def tearDown(self):
        self.window.timer.stop()
        self.window.deleteLater()
        self.app.processEvents()
    def test_opening_and_polling_does_not_start_owner(self):
        self.window.refresh()
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertEqual(self.backend.stops, 0)
    def test_preparing_owner_cannot_be_duplicated_or_change_instruction(self):
        self.backend.snapshot['running'] = True
        self.window.refresh()
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertTrue(self.window.instruction.isReadOnly())
        self.assertFalse(self.window.sync_button.isEnabled())
    def test_close_during_flight_requests_return_and_keeps_window_alive(self):
        self.backend.snapshot['running'] = True
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertEqual(self.backend.stops, 1)
        self.assertEqual(self.backend.closes, 0)
    def test_close_waits_for_preparation_then_requests_cancellation(self):
        self.window.busy = 'start'
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(self.window.stop_pending)
        self.assertTrue(self.window.cancel_start.is_set())
        self.assertEqual(self.backend.stops, 1)
        self.assertEqual(self.backend.closes, 0)
    def test_old_scene_receipt_does_not_enable_flight(self):
        self.backend.snapshot['bootstrap']['status'] = 'READY_SCENE_SYNC_REQUIRED'
        self.window.refresh()
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.sync_button.isEnabled())


class DriverCancellationTests(unittest.TestCase):
    def test_interface_scene_change_refuses_flight_binding(self):
        from run_yolo_language_mission_v8 import check_interface_binding
        expected = dict(scene_version='one', engine_identity=dict(pid=4, created=1))
        check_interface_binding(expected, dict(runtime_scene_version='one', runtime_engine_identity=dict(pid=4, created=1)))
        for changed in [dict(runtime_scene_version='two', runtime_engine_identity=dict(pid=4, created=1)),
                        dict(runtime_scene_version='one', runtime_engine_identity=dict(pid=4, created=2))]:
            with self.assertRaisesRegex(ValueError, 'PREDECLARED_SCENE_CHANGED'):
                check_interface_binding(expected, changed)
    def execute_cancelled(self, cancel_after_parse):
        import run_yolo_language_mission_v8 as driver
        import fixed_map_mission as mission
        import fixed_map_perception_execution as execution
        import fixed_map_sensor_language as language
        import fixed_map_trial_cleanup as cleanup
        import fixed_map_retention as retention
        class Owner:
            stage = 'CHECKING'
            safety = {}
            state = {}
            def __init__(self, declaration): pass
            def is_alive(self): return False
        parent = ROOT / 'reports/interface_v24_checks'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            trial = Path(folder)
            out = trial / 'online_trial'
            stop = trial / 'stop.request'
            if not cancel_after_parse: stop.write_text('cancel', encoding='utf-8')
            def configure(path): Path(path).mkdir()
            def parse(*args):
                if not cancel_after_parse: raise AssertionError('Model called after prelaunch stop')
                stop.write_text('cancel', encoding='utf-8')
                return dict(validated=True, proposal=dict(action='start'))
            def finalize(output, owner, coordinator, service, issue, flight_started):
                terminal = dict(issue=issue, flight_start_requested=flight_started,
                                terminal_phase='FINAL', ground_verification_passed=True)
                driver.save(output/'result.json', terminal)
                return terminal
            config = dict(schema='test_cancel_no_model', resource_trace_v20=False)
            with patch.object(driver, 'configure_session', configure), \
                 patch.object(driver, 'check_output_storage', return_value=dict(status='PASS')), \
                 patch.object(driver, 'read', return_value=config), \
                 patch.object(mission, 'ensure_scene'), \
                 patch.object(execution, 'PerceptionExecutor', Owner), \
                 patch.object(language, 'interpret', side_effect=parse) as model, \
                 patch.object(cleanup, 'finalize_trial', side_effect=finalize), \
                 patch.object(retention, 'apply_postflight_raw_retention', return_value=dict(status='PASS')):
                driver.run(out, 'test', stop_request_path=stop)
            result = json.loads((out/'result.json').read_text(encoding='utf-8'))
            self.assertFalse(result['flight_start_requested'])
            self.assertNotIn('online', [p.name for p in out.iterdir()])
            return result, model.call_count
    def test_stop_before_owner_initialization_does_not_load_model_or_takeoff(self):
        result, calls = self.execute_cancelled(False)
        self.assertEqual(calls, 0)
        self.assertEqual(result['issue'], 'OPERATOR_STOP_BEFORE_MODEL_LOAD_NO_TAKEOFF')
    def test_stop_during_language_parse_prevents_model_and_takeoff_stage(self):
        result, calls = self.execute_cancelled(True)
        self.assertEqual(calls, 1)
        self.assertEqual(result['issue'], 'OPERATOR_STOP_BEFORE_TAKEOFF')


if __name__ == '__main__': unittest.main()
