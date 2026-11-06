"""New panel checks: no RPC on opening, nonblocking observation and safe close."""
import json
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


if __name__ == "__main__":
    unittest.main()
