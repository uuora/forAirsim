"""Headless UI checks for flight authority, close lifecycle, and 3D geometry."""
import copy
import io
import os
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication
from .ui import ControlWindow, _PreviewRuntime, main
from .overview_ui import OverviewCanvas


class FakeRuntime(_PreviewRuntime):
    def __init__(self, config):
        super().__init__(config)
        self.calls = []
        self.closed = False
        self.data.update(mode="airsim", status="IDLE", connected=False, running=False)

    def configure(self, config):
        self.calls.append("configure")
        super().configure(config)

    def start(self): self.calls.append("start")
    def pause(self): self.calls.append("pause")
    def resume(self): self.calls.append("resume")
    def stop(self): self.calls.append("stop")
    def shutdown(self): self.calls.append("shutdown")
    def show_scene(self):
        self.calls.append("show_scene")
        super().show_scene()
    def hide_scene(self):
        self.calls.append("hide_scene")
        super().hide_scene()
    def wait_closed(self, timeout): return self.closed


class FakeOverviewStream:
    def __init__(self, config, runtime):
        self.config = config
        self.runtime = runtime
        self.started = False
        self.stopped = threading.Event()
        self.finish_stop = threading.Event()
        self.finish_stop.set()
        self.data = dict(status="running", message="总览相机", frames=1,
                         frame=np.full((24, 40, 3), [230, 50, 90], dtype=np.uint8),
                         frame_ms=12.5, rpc_ms=10.0)

    def start(self): self.started = True

    def stop(self):
        self.stopped.set()
        self.finish_stop.wait(2)

    def snapshot(self): return self.data.copy()


class UiTest(unittest.TestCase):
    def test_waypoint_spacing_is_editable_and_uses_control_validation(self):
        self.window.inputs['target_distance_min'].setValue(6.)
        self.window.inputs['target_distance_max'].setValue(12.)
        self.assertTrue(self.window._apply())
        self.assertEqual(self.window.config['control']['target_distance_min'], 6.)
        self.assertEqual(self.window.config['control']['target_distance_max'], 12.)
        self.runtime.calls.clear()
        self.window.inputs['target_distance_max'].setValue(5.)
        self.assertFalse(self.window._apply())
        self.assertEqual(self.runtime.calls, [])

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.config = {"control": dict(x_min=-20, x_max=20, y_min=-20, y_max=20,
                                       height_min=2, height_max=10, margin=1,
                                       arrival_radius=.5, max_speed=4)}
        self.runtime = FakeRuntime(copy.deepcopy(self.config))
        self.streams = []
        def stream_factory(config, runtime):
            stream = FakeOverviewStream(config, runtime)
            self.streams.append(stream)
            return stream
        self.window = ControlWindow(self.runtime, self.config,
                                    overview_stream_factory=stream_factory)
        self.window.resize(1360, 885)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        for stream in self.streams:
            stream.finish_stop.set()
        self.runtime.closed = True
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_open_has_no_flight_authority(self):
        self.assertEqual(self.runtime.calls, [])
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertFalse(self.window.scene_button.isEnabled())
        self.assertFalse(self.window.overview_button.isEnabled())
        self.assertEqual(self.streams, [])

    def test_scene_request_waits_for_connection_and_shows_only_once(self):
        self.window.request_scene()
        self.assertEqual(self.runtime.calls, [])
        self.runtime.data.update(status="READY", connected=True)
        self.window.refresh()
        self.window.refresh()
        self.assertEqual(self.runtime.calls, ["show_scene"])
        self.assertEqual(self.window.scene_button.text(), "隐藏 UE4 场景")
        self.assertFalse(self.runtime.data["running"])

    def test_scene_button_only_observes_same_running_simulator(self):
        self.runtime.data.update(status="RUNNING", running=True, connected=True)
        self.window.refresh()
        self.window.scene_button.click()
        self.window.refresh()
        self.assertEqual(self.window.scene_button.text(), "隐藏 UE4 场景")
        self.window.scene_button.click()
        self.window.refresh()
        self.assertEqual(self.runtime.calls, ["show_scene", "hide_scene"])
        self.assertEqual(self.window.scene_button.text(), "查看 UE4 场景")
        self.assertEqual(self.runtime.data["status"], "RUNNING")
        self.assertTrue(self.runtime.data["running"])
        self.assertEqual(self.streams, [])

    def test_scene_is_unavailable_without_airsim_connection(self):
        self.window.scene_button.click()
        self.assertEqual(self.runtime.calls, [])
        self.runtime.data.update(mode="demo", connected=True)
        self.window.refresh()
        self.assertFalse(self.window.scene_button.isEnabled())
        self.window._toggle_scene()
        self.assertEqual(self.runtime.calls, [])

    def test_scene_message_does_not_replace_flight_failure(self):
        self.runtime.data.update(status="ERROR", running=True, connected=True,
                                 message="降落失败，请重试。", scene_message="场景显示失败。")
        self.window.refresh()
        self.assertEqual(self.window.message.text(), "降落失败，请重试。")
        self.assertEqual(self.window.scene_status.text(), "场景显示失败。")
        self.assertTrue(self.window.scene_status.isVisible())
        self.assertTrue(self.window.stop_button.isEnabled())

    def test_mode_change_hides_scene_and_clears_pending_request(self):
        self.runtime.data.update(status="READY", connected=True, scene_visible=True)
        self.window.runtime_factory = lambda cfg, mode: FakeRuntime(cfg)
        self.window._scene_pending = True
        self.window.mode_combo.setCurrentIndex(1)
        self.assertEqual(self.runtime.calls, ["hide_scene", "shutdown"])
        self.assertFalse(self.runtime.data["scene_visible"])
        self.assertFalse(self.window._scene_pending)

    def test_scene_cli_requires_real_airsim_mode_and_no_preview(self):
        for args in (["--scene", "--mode", "demo"], ["--scene", "--smoke-test"]):
            with self.subTest(args=args), patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)
        self.assertEqual(self.runtime.calls, [])

    def test_overview_is_explicitly_optional(self):
        self.assertEqual(self.window.overview_button.text(), "俯视图（可选）")

    def test_overview_opens_same_runtime_only_after_connection(self):
        self.window.overview_button.click()
        self.assertEqual(self.streams, [])
        self.runtime.data.update(status="connected", connected=True)
        self.window.refresh()
        self.assertTrue(self.window.overview_button.isEnabled())
        self.window.overview_button.click()
        self.assertEqual(len(self.streams), 1)
        self.assertIs(self.streams[0].runtime, self.runtime)
        self.assertEqual(self.streams[0].config, self.config)
        self.assertTrue(self.streams[0].started)
        self.assertTrue(self.window.overview_window.isVisible())
        self.assertEqual(self.window.overview_window.timer.interval(), 250)
        self.assertEqual(self.runtime.calls, [])
        self.window.overview_button.click()
        self.assertEqual(len(self.streams), 1)

    def test_overview_close_never_stops_flight_and_does_not_wait_for_image_rpc(self):
        self.runtime.data.update(status="running", connected=True, running=True)
        self.window.refresh(); self.window.overview_button.click()
        stream = self.streams[0]
        stream.finish_stop.clear()
        started = time.monotonic()
        self.window.overview_window.close()
        self.assertLess(time.monotonic() - started, .2)
        self.assertTrue(stream.stopped.wait(.5))
        self.assertIsNone(self.window.overview_window)
        self.assertEqual(self.runtime.calls, [])
        self.assertTrue(self.window.isVisible())
        stream.finish_stop.set()

    def test_overview_image_is_rgb_and_detached_from_stream_buffer(self):
        self.runtime.data.update(connected=True)
        self.window.refresh(); self.window.overview_button.click()
        viewer = self.window.overview_window
        viewer.refresh()
        self.assertEqual(viewer.canvas.image.size().width(), 40)
        self.assertEqual(viewer.canvas.image.pixelColor(0, 0).getRgb()[:3], (230, 50, 90))
        self.streams[0].data["frame"][:] = 0
        self.assertEqual(viewer.canvas.image.pixelColor(0, 0).getRgb()[:3], (230, 50, 90))
        self.assertIn("画面 RPC  10.0 ms", viewer.metrics.text())
        self.assertIn("每帧耗时  12.5 ms", viewer.metrics.text())

    def test_overview_is_closed_when_connection_is_lost_or_mode_is_demo(self):
        self.runtime.data.update(connected=True)
        self.window.refresh(); self.window.overview_button.click()
        self.runtime.data.update(connected=False)
        self.window.refresh()
        self.assertIsNone(self.window.overview_window)
        self.assertTrue(self.streams[0].stopped.wait(.5))
        self.runtime.data.update(mode="demo", connected=True)
        self.window.refresh()
        self.assertFalse(self.window.overview_button.isEnabled())
        self.assertEqual(self.runtime.calls, [])

    def test_overview_request_waits_for_connection_without_connecting_or_flying(self):
        self.window.request_overview()
        self.assertIsNone(self.window.overview_window)
        self.assertEqual(self.runtime.calls, [])
        self.runtime.data.update(connected=True)
        self.window.refresh()
        self.assertIsNotNone(self.window.overview_window)
        self.assertEqual(self.runtime.calls, [])

    def test_overview_closes_when_main_window_starts_shutdown(self):
        self.runtime.data.update(connected=True)
        self.window.refresh(); self.window.overview_button.click()
        self.window.close()
        self.assertIsNone(self.window.overview_window)
        self.assertTrue(self.streams[0].stopped.wait(.5))
        self.assertEqual(self.runtime.calls, ["hide_scene", "shutdown"])

    def test_canvas_rejects_invalid_frames(self):
        canvas = OverviewCanvas()
        with self.assertRaises(ValueError):
            canvas.set_frame(np.zeros((4, 5, 4), dtype=np.uint8))
        canvas.deleteLater()

    def test_overview_telemetry_annotations_update_without_changing_rgb_image(self):
        self.runtime.data.update(connected=True)
        self.window.refresh(); self.window.overview_button.click()
        viewer = self.window.overview_window
        stream = self.streams[0]
        stream.data["overlays"] = dict(targets=[[10, 10], [20, 15]],
                                       planned=[[10, 10], [20, 15]],
                                       trail=[[5, 3], [10, 10]], drone=[10, 10], nose=[12, 10])
        viewer.refresh()
        np.testing.assert_array_equal(viewer.canvas.overlays["drone"], [10, 10])
        stream.data["overlays"]["drone"] = [11, 10]
        viewer.refresh()  # Same frame counter: annotations still follow telemetry.
        np.testing.assert_array_equal(viewer.canvas.overlays["drone"], [11, 10])
        viewer.canvas.grab()  # Exercise painting; annotations never enter the RGB buffer.
        self.assertEqual(viewer.canvas.image.pixelColor(10, 10).getRgb()[:3], (230, 50, 90))

    def test_overview_annotations_follow_aspect_fit_and_resize(self):
        canvas = OverviewCanvas()
        canvas.resize(800, 600)
        canvas.set_frame(np.zeros((200, 400, 3), dtype=np.uint8))
        np.testing.assert_allclose(canvas.project_overlay_points([[0, 0], [400, 200]]),
                                   [[0, 100], [800, 500]])
        canvas.resize(600, 800)
        np.testing.assert_allclose(canvas.project_overlay_points([[0, 0], [400, 200]]),
                                   [[0, 250], [600, 550]])
        canvas.set_overlays(dict(drone=[float("nan"), 2], targets=[[3, 4]],
                                 trail=[[1, 2, 3]]))
        self.assertEqual(set(canvas.overlays), {"targets"})
        canvas.deleteLater()

    def test_overview_backend_statuses_are_displayed_in_chinese(self):
        self.runtime.data.update(connected=True)
        self.window.refresh(); self.window.overview_button.click()
        viewer = self.window.overview_window
        for status, label in (("STREAMING", "实时画面"), ("waiting", "等待连接"),
                              ("unavailable", "未提供画面"), ("stopping", "正在关闭")):
            with self.subTest(status=status):
                self.streams[0].data.update(status=status, message="")
                viewer.refresh()
                self.assertEqual(viewer.message.text(), label)

    def test_uppercase_runtime_status_resumes(self):
        self.runtime.data.update(status="PAUSED", running=True, connected=True)
        self.window.refresh()
        self.assertEqual(self.window.badge.text(), "已暂停")
        self.assertEqual(self.window.pause_button.text(), "继续飞行")
        self.window.pause_button.click()
        self.assertEqual(self.runtime.calls, ["resume"])

    def test_invalid_small_field_does_not_enqueue_start(self):
        self.window.inputs["height_max"].setValue(4)
        self.window._start()
        self.assertEqual(self.runtime.calls, [])

    def test_close_waits_for_worker_without_blocking_gui(self):
        self.window.close()
        self.assertTrue(self.window.isVisible())
        self.assertEqual(self.runtime.calls, ["hide_scene", "shutdown"])
        self.runtime.closed = True
        self.window._finish_lifecycle()
        self.assertFalse(self.window.isVisible())
        self.assertEqual(self.runtime.calls, ["hide_scene", "shutdown"])

    def test_close_failure_shows_error_and_allows_landing_retry(self):
        self.window.close()
        self.runtime.data.update(status="ERROR", running=True,
                                 message="降落失败：无人机仍在空中。")
        self.window.refresh()
        self.assertTrue(self.window.isVisible())
        self.assertEqual(self.window.badge.text(), "运行异常")
        self.assertEqual(self.window.message.text(), "降落失败：无人机仍在空中。")
        self.assertEqual(self.window.stop_button.text(), "重试降落")
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertFalse(self.window.pause_button.isEnabled())
        self.window.stop_button.click()
        self.assertEqual(self.runtime.calls, ["hide_scene", "shutdown", "stop"])
        self.runtime.closed = True
        self.window._finish_lifecycle()
        self.assertFalse(self.window.isVisible())

    def test_pending_mode_failure_keeps_landing_retry_available(self):
        self.window._pending_mode = "demo"
        self.runtime.data.update(status="ERROR", running=True,
                                 message="切换模式前降落失败，请重试。")
        self.window.refresh()
        self.assertEqual(self.window.badge.text(), "运行异常")
        self.assertEqual(self.window.message.text(), "切换模式前降落失败，请重试。")
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertFalse(self.window.start_button.isEnabled())
        self.window.stop_button.click()
        self.assertEqual(self.runtime.calls, ["stop"])

    def test_metric_values_fit_at_requested_window_size(self):
        for tile in self.window.metrics.values():
            self.assertGreaterEqual(tile.height(), 66)
            self.assertGreaterEqual(tile.value.height(), tile.value.fontMetrics().height())

    def test_height_changes_real_world_projection(self):
        self.window.view._camera()
        p1 = self.window.view.project([0, 0, 3])
        p2 = self.window.view.project([0, 0, 8])
        self.assertLess(p2.y(), p1.y())
        self.assertAlmostEqual(p2.x(), p1.x())
        self.window.view.azimuth += .4
        self.window.view._camera()
        p3 = self.window.view.project([8, 3, 4])
        self.window.view.azimuth += .4
        self.window.view._camera()
        p4 = self.window.view.project([8, 3, 4])
        self.assertNotEqual(p3.x(), p4.x())

    def test_display_trail_is_bounded_preserves_endpoints_and_telemetry(self):
        points = [[float(index), index * .1, 3.0] for index in range(3000)]
        sampled = self.window.view.display_points(points, self.window.view.MAX_DISPLAY_TRAIL)
        self.assertEqual(len(sampled), 600)
        np.testing.assert_array_equal(sampled[0], points[0])
        np.testing.assert_array_equal(sampled[-1], points[-1])
        self.assertEqual(len(points), 3000)
        self.assertTrue(np.all(np.diff(sampled[:, 0]) > 0))

    def test_batch_projection_tracks_zoom_and_pan(self):
        view = self.window.view
        view._camera()
        points = np.array([[0, 0, 5], [5, 3, 7], [-9, 11, 4]], dtype=float)
        initial = view.project_points(points)
        for point, projected in zip(points, initial):
            single = view.project(point)
            np.testing.assert_allclose(projected, [single.x(), single.y()])
        view.zoom = 2.0
        view.pan = np.array([13.0, -7.0])
        view._camera()
        updated = view.project_points(points)
        np.testing.assert_allclose(updated[0], initial[0] + view.pan)
        np.testing.assert_allclose(updated[1:] - updated[0],
                                   2.0 * (initial[1:] - initial[0]))


if __name__ == "__main__":
    unittest.main()
