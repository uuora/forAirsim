"""New panel checks: no RPC on opening, nonblocking observation and safe close."""
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QApplication
import numpy as np

from .qt_panel import GeometricControlWindow, LatestTelemetry, MapAdapter


CONFIG = {"start": [0, 0, -5], "cases": {
    "hover": {"offsets": [[0, 0, 0]], "segment_durations": [3]},
    "line": {"offsets": [[3, 0, 0]], "segment_durations": [4]},
    "turn": {"offsets": [[3, 0, 0], [3, 3, -1]], "segment_durations": [4, 4]},
    "waypoints": {"offsets": [[2, 0, -1], [2, 2, 0], [0, 2, 1]], "segment_durations": [4, 4, 4]},
}}


class AdapterTests(unittest.TestCase):
    def test_three_planner_preview_selection_never_runs_or_connects(self):
        from .qt_panel import DEFAULT_CONFIG
        app=QApplication.instance() or QApplication([])
        runner=Mock();viewport=Mock()
        window=GeometricControlWindow(DEFAULT_CONFIG,runner=runner,viewport_factory=viewport)
        self.assertEqual(window.case_combo.count(),3)
        for i in range(3):
            window.case_combo.setCurrentIndex(i);app.processEvents()
            self.assertEqual(len(window.view.snapshot['planned_path']),160)
            self.assertFalse(window.view.snapshot['connected'])
        runner.assert_not_called();viewport.assert_not_called()
        window.close()

    def test_all_environment_and_method_switches_only_load_frozen_previews(self):
        from .qt_panel import DEFAULT_CONFIG
        app=QApplication.instance() or QApplication([])
        runner=Mock();factory=Mock()
        with patch('socket.create_connection',side_effect=AssertionError('RPC on selection')), \
             patch('subprocess.Popen',side_effect=AssertionError('Launch on selection')):
            window=GeometricControlWindow(DEFAULT_CONFIG,runner=runner,viewport_factory=factory)
            self.assertEqual(window.environment_combo.count(),5)
            for i in range(5):
                window.environment_combo.setCurrentIndex(i)
                self.assertEqual(window.case_combo.count(),3)
                for j in range(3):
                    window.case_combo.setCurrentIndex(j);app.processEvents()
                    self.assertFalse(window.active)
                    self.assertEqual(len(window.view.snapshot['planned_path']),160)
                    self.assertFalse(window.view.snapshot['connected'])
                    self.assertIn('航点折线',window.environment_info.text())
                    self.assertIn('位置≤0.5m',window.threshold_label.text())
                    self.assertIn('姿态≤10°',window.threshold_label.text())
            runner.assert_not_called();factory.assert_not_called()
            window.close()

    def test_environment_change_preserves_method_and_clears_previous_run_link(self):
        from .qt_panel import DEFAULT_CONFIG
        app=QApplication.instance() or QApplication([])
        window=GeometricControlWindow(DEFAULT_CONFIG)
        window.case_combo.setCurrentIndex(window.case_combo.findData('bspline_pass'))
        window.run_dir='synthetic_previous_run';window.open_button.setEnabled(True)
        window.environment_combo.setCurrentIndex(window.environment_combo.findData('climb_descent'))
        app.processEvents()
        self.assertEqual(window.case_combo.currentData(),'bspline_pass')
        self.assertIsNone(window.run_dir)
        self.assertFalse(window.open_button.isEnabled())
        self.assertEqual(window.config['comparison']['environment']['id'],'climb_descent')
        window.close()

    def test_body_axes_convert_roll_pitch_and_yaw_to_height_coordinates(self):
        h = math.sqrt(.5)
        frame = dict(position=[1, 2, -6], quaternion_wxyz=[1, 0, 0, 0])
        cases = [([h, h, 0, 0], [1, 0, 0], [0, 0, -1]),
                 ([h, 0, h, 0], [0, 0, 1], [0, 1, 0]),
                 ([h, 0, 0, h], [0, 1, 0], [-1, 0, 0])]
        for quaternion, nose, right in cases:
            frame['quaternion_wxyz'] = quaternion
            snapshot = MapAdapter(CONFIG, 'turn').adapt(frame)
            rotation = np.asarray(snapshot['body_rotation_height'])
            np.testing.assert_allclose(rotation[:, 0], nose, atol=1e-14)
            np.testing.assert_allclose(rotation[:, 1], right, atol=1e-14)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.)
            frame['quaternion_wxyz'] = [-2*v for v in quaternion]
            other = MapAdapter(CONFIG, 'turn').adapt(frame)
            np.testing.assert_allclose(other['body_rotation_height'], rotation, atol=1e-14)

    def test_ned_conversion_and_measured_quaternion_yaw(self):
        adapter = MapAdapter(CONFIG, "turn")
        frame = dict(position=[1, 2, -6], velocity=[1, 0, -.5],
                     quaternion_wxyz=[math.sqrt(.5), 0, 0, math.sqrt(.5)],
                     reference_position=[1.1, 2, -6])
        snapshot = adapter.adapt(frame)
        self.assertEqual(snapshot["position"], [1, 2, 6])
        self.assertEqual(snapshot["velocity"], [1, 0, .5])
        self.assertAlmostEqual(snapshot["yaw"], math.pi / 2)
        self.assertEqual(frame["position"], [1, 2, -6])
        self.assertEqual(adapter.view_config["preparation"]["task"]["points"], [[3, 0, 5], [3, 3, 6]])

    def test_display_does_not_infer_visit_from_elapsed_time(self):
        adapter = MapAdapter(CONFIG, "turn")
        frame = dict(position=[0, 0, -5], velocity=[0, 0, 0],
                     quaternion_wxyz=[1, 0, 0, 0], t=1000, route_index=1000)
        snapshot = adapter.adapt(frame)
        self.assertEqual(snapshot["route_markers"][1]["state"], "pending")
        frame["position"] = [3, 0, -5]
        adapter.adapt(frame)
        frame["position"] = [3, 1, -5]
        self.assertEqual(adapter.adapt(frame)["route_markers"][1]["state"], "visited")

    def test_mailbox_never_waits_for_display_and_retains_one_time_path(self):
        mailbox = LatestTelemetry()
        path = [[0, 0, -5], [3, 0, -5]]
        mailbox.publish(dict(planned_path_ned=path))
        mailbox.publish(dict(position=[1, 0, -5]))
        self.assertIs(mailbox.take()["planned_path_ned"], path)
        mailbox.lock.acquire()
        try:
            # If this blocked it would deadlock this test on the same lock.
            mailbox.publish(dict(position=[2, 0, -5]))
        finally:
            mailbox.lock.release()
        self.assertEqual(mailbox.dropped, 1)


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "config.json"
        self.path.write_text(json.dumps(CONFIG), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_open_change_theme_plane_and_case_never_calls_runner(self):
        calls = []
        window = GeometricControlWindow(self.path, runner=lambda *a, **k: calls.append((a, k)))
        window.show()
        for i in range(window.case_combo.count()):
            window.case_combo.setCurrentIndex(i)
            for plane in ("3d", "xy", "xz"):
                window.view.set_plane(plane)
                self.app.processEvents()
        window.view._camera()
        matrix = window.view.projection_matrix.copy()
        window.view.scene.set_theme("light")
        self.app.processEvents()
        np.testing.assert_array_equal(matrix, window.view.projection_matrix)
        self.assertFalse(window.active)
        self.assertEqual(calls, [])
        self.assertFalse(window.view.snapshot["connected"])
        self.assertFalse(window.launch_checkbox.isChecked())
        self.assertEqual(len(window.view.snapshot["planned_path"]), 160)
        window.close()

    def test_record_selection_entry_blocks_start_and_only_opens_the_data_dialog(self):
        runner=Mock();window=GeometricControlWindow(self.path,runner=runner)
        with patch('geometric_validation.record_comparison_ui.RecordComparisonDialog') as factory:
            dialog=factory.return_value;dialog.busy=False;dialog.isVisible.return_value=True
            def attempt_start():
                self.assertFalse(window.start_button.isEnabled())
                window.start()
            dialog.exec_.side_effect=attempt_start
            window.select_records_button.click()
            factory.assert_called_once_with(window,'custom');dialog.exec_.assert_called_once()
            runner.assert_not_called();self.assertTrue(window.start_button.isEnabled())
        window._record_dialog=None;window.close()

    def test_launch_requires_start_and_result_verdicts_remain_separate(self):
        calls = []

        def fake_run(*args, **kwargs):
            calls.append(kwargs["launch"])
            return {"run_dir": self.temp.name, "report": {
                "outcome": "COMPLETED", "normal_completion": True,
                "control_constraints_passed": False, "data_audit_passed": True,
                "cleanup": {"landed": True}}}

        window = GeometricControlWindow(self.path, runner=fake_run)
        window.launch_checkbox.setChecked(True)
        self.app.processEvents()
        self.assertEqual(calls, [])
        window.start()
        window._thread.join(2)
        window.refresh()
        self.assertEqual(calls, [True])
        self.assertIn("正常完成：通过", window.status.text())
        self.assertIn("控制约束：未通过", window.status.text())
        self.assertIn("数据审计：通过", window.status.text())
        window.close()

    def test_passage_verdict_is_visible_separately(self):
        window=GeometricControlWindow(self.path,runner=lambda *a,**k: {'report': {
            'normal_completion':True,'control_constraints_passed':True,'data_audit_passed':True,
            'waypoint_passage_passed':False,'cleanup':{'landed':True}}})
        window.start();window._thread.join(2);window.refresh()
        self.assertIn('连续过点：未通过',window.status.text())
        self.assertIn('控制约束：通过',window.status.text())
        window.close()

    def test_close_waits_for_runtime_cleanup_after_stop(self):
        started = threading.Event()
        cleanup = threading.Event()
        worker_thread = []

        def fake_run(config_path, case, **kwargs):
            worker_thread.append(threading.get_ident())
            started.set()
            kwargs["stop_event"].wait(3)
            cleanup.wait(3)
            return {"run_dir": self.temp.name, "report": {"outcome": "STOPPED"}}

        window = GeometricControlWindow(self.path, runner=fake_run)
        window.show()
        window.start()
        self.assertTrue(started.wait(1))
        self.assertNotEqual(worker_thread[0], threading.get_ident())
        window.close()
        self.assertTrue(window.isVisible())
        self.assertTrue(window._stop_event.is_set())
        self.assertTrue(window.active)
        cleanup.set()
        deadline = time.monotonic() + 3
        while window.active and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.01)
        window.refresh()
        self.app.processEvents()
        self.assertFalse(window.isVisible())
        self.assertEqual(window.run_dir, self.temp.name)

    def test_environment_is_locked_while_running_and_runtime_receives_selected_path(self):
        from .qt_panel import DEFAULT_CONFIG
        from .environments import config_for
        started=threading.Event();finish=threading.Event();calls=[]
        def fake(config_path,case,**kwargs):
            calls.append((config_path,case));started.set();finish.wait(3)
            return {'report':{'outcome':'SYNTHETIC'}}
        window=GeometricControlWindow(DEFAULT_CONFIG,runner=fake)
        window.environment_combo.setCurrentIndex(window.environment_combo.findData('straight'))
        window.start();self.assertTrue(started.wait(1))
        self.assertFalse(window.environment_combo.isEnabled())
        window.environment_combo.setCurrentIndex(window.environment_combo.findData('planar_s'))
        self.assertEqual(window.environment_combo.currentData(),'straight')
        self.assertEqual(window.config_path,config_for('straight'))
        finish.set();window._thread.join(2);window.refresh()
        self.assertTrue(window.environment_combo.isEnabled())
        self.assertEqual(calls,[(config_for('straight'),'bezier_pass')])
        window.close()

    def test_failed_run_displays_runtime_error_and_preserves_record_link(self):
        message = "Missing runtime dependencies (no flight started):\nD:/missing/AirSim.dll"
        window = GeometricControlWindow(self.path, runner=lambda *a, **k: {
            "run_dir": self.temp.name, "report": {
                "outcome": "FAILED", "error": message,
                "normal_completion": False, "control_constraints_passed": False,
                "data_audit_passed": False, "cleanup": {"landed": False}}})
        window.start()
        window._thread.join(2)
        window.refresh()
        self.assertIn("Missing runtime dependencies", window.status.text())
        self.assertEqual(window.error_details.toPlainText(), message)
        self.assertFalse(window.error_details.isHidden())
        self.assertTrue(window.open_button.isEnabled())
        self.assertTrue(window.start_button.isEnabled())
        window.close()

    def test_worker_exception_is_visible_without_a_result(self):
        def fail(*args, **kwargs):
            raise RuntimeError("test startup failure")

        window = GeometricControlWindow(self.path, runner=fail)
        window.start()
        window._thread.join(2)
        window.refresh()
        self.assertIn("RuntimeError: test startup failure", window.status.text())
        self.assertEqual(window.error_details.toPlainText(), "RuntimeError: test startup failure")
        self.assertFalse(window.open_button.isEnabled())
        window.close()

    def test_roll_and_pitch_change_drone_pixels_without_changing_camera(self):
        from dual_target_control.ui import FlightView
        window = GeometricControlWindow(self.path)
        view = window.view
        view.resize(900, 580)
        view._camera()
        camera = view.projection_matrix.copy()
        images = []
        for quaternion in ([1, 0, 0, 0],
                           [math.cos(.25), math.sin(.25), 0, 0],
                           [math.cos(.25), 0, math.sin(.25), 0]):
            view.set_snapshot(window.adapter.adapt(dict(
                position=[0, 0, -5], quaternion_wxyz=quaternion)))
            view._camera()
            image = QImage(view.size(), QImage.Format_ARGB32)
            image.fill(0)
            painter = QPainter(image)
            painter.setRenderHint(QPainter.Antialiasing)
            FlightView._drone(view, painter, arm_pixels=12., show_label=False,
                              body_rotation=view.snapshot['body_rotation_height'])
            painter.end()
            images.append(image.bits().asstring(image.byteCount()))
            np.testing.assert_array_equal(view.projection_matrix, camera)
        self.assertNotEqual(images[0], images[1])
        self.assertNotEqual(images[0], images[2])
        self.assertNotEqual(images[1], images[2])
        window.close()

    def test_telemetry_updates_full_attitude_and_numeric_readout(self):
        window = GeometricControlWindow(self.path)
        frame = dict(position=[0, 0, -5], quaternion_wxyz=[math.cos(.1), math.sin(.1), 0, 0])
        window.mailbox.publish(frame)
        window.refresh()
        self.assertIn('+11.46', window.attitude_label.text())
        self.assertIn('body_rotation_height', window.view.snapshot)
        window.close()

    def test_viewport_toggle_is_explicit_and_never_starts_flight(self):
        runner = Mock()
        viewport = Mock()
        viewport.take_update.return_value = {}
        viewport.close.return_value = True
        factory = Mock(return_value=viewport)
        window = GeometricControlWindow(self.path, runner=runner, viewport_factory=factory)
        window.show()
        self.app.processEvents()
        factory.assert_not_called()
        window.toggle_scene()
        viewport.request.assert_called_once_with(True)
        self.assertFalse(window.scene_button.isEnabled())
        viewport.take_update.return_value = dict(scene_visible=True, scene_status='VISIBLE', scene_message='UE visible')
        window.refresh()
        self.assertTrue(window.scene_button.isEnabled())
        self.assertIn('隐藏', window.scene_button.text())
        window.toggle_scene()
        self.assertEqual(viewport.request.call_args.args, (False,))
        runner.assert_not_called()
        self.assertFalse(window.active)
        window.close()

    def test_viewport_error_is_visible_and_retry_is_enabled(self):
        viewport = Mock()
        viewport.close.return_value = True
        viewport.take_update.return_value = dict(scene_visible=False, scene_status='ERROR', scene_message='Port not running')
        window = GeometricControlWindow(self.path, viewport_factory=lambda *a: viewport)
        window.toggle_scene()
        window.refresh()
        self.assertIn('Port not running', window.scene_status.text())
        self.assertTrue(window.scene_button.isEnabled())
        window.close()

    def test_close_waits_for_viewport_cleanup(self):
        viewport = Mock()
        viewport.close.return_value = False
        viewport.take_update.return_value = {}
        window = GeometricControlWindow(self.path, viewport_factory=lambda *a: viewport)
        window.show()
        window.toggle_scene()
        window.close()
        self.assertTrue(window.isVisible())
        self.assertTrue(window._closing)
        viewport.close.return_value = True
        window.refresh()
        self.assertFalse(window.isVisible())


if __name__ == "__main__":
    unittest.main()
