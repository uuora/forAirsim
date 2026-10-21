"""Overview isolation, image fidelity and bounded visualization RPC regressions."""
from __future__ import annotations

from copy import deepcopy
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import airsim
import numpy as np

from dual_target_control.core import ControlConfig
from dual_target_control.overview import (
    OverviewStream, decode_scene_frame, field_edges, marker_geometry, project_overlays)


RGB = np.array([[[255, 0, 0], [0, 0, 255]],
                [[0, 255, 0], [25, 50, 75]]], dtype=np.uint8)


def response(**changes):
    values = dict(width=2, height=2, message='',
                  image_data_uint8=RGB[..., ::-1].tobytes())
    values.update(changes)
    return SimpleNamespace(**values)


class FakeRuntime:
    mode = 'airsim'

    def __init__(self):
        self.state = dict(mode='airsim', connected=True,
                          config=dict(control={}), targets=[[2., 3., 4.], [8., 7., 6.]],
                          position=[0., 0., 5.], yaw=0.,
                          planned_path=[[0., 0., 5.], [2., 3., 4.], [8., 7., 6.]],
                          trail=[[0., 0., 5.], [1., 1., 4.8]])

    def snapshot(self):
        return deepcopy(self.state)


class FakeClient:
    """Intentionally has no arm, API-control, vehicle-pose or motion methods."""
    def __init__(self):
        self.calls = []
        self.client = SimpleNamespace(close=self.close)
        self.capture_entered = threading.Event()
        self.capture_release = None
        self.capture_error = None
        self.pose_accepted = True
        self.camera_error = None

    def record(self, method, *args, **kwargs):
        self.calls.append((method, args, kwargs, threading.get_ident()))

    def ping(self):
        self.record('ping')
        return True

    def close(self):
        self.record('close')

    def simSetObjectPose(self, *args, **kwargs):
        self.record('simSetObjectPose', *args, **kwargs)
        return self.pose_accepted

    def simPlotPoints(self, *args, **kwargs):
        self.record('simPlotPoints', *args, **kwargs)

    def simPlotLineList(self, *args, **kwargs):
        self.record('simPlotLineList', *args, **kwargs)

    def simPlotLineStrip(self, *args, **kwargs):
        self.record('simPlotLineStrip', *args, **kwargs)

    def simGetImages(self, *args, **kwargs):
        self.record('simGetImages', *args, **kwargs)
        self.capture_entered.set()
        if self.capture_release is not None:
            self.capture_release.wait(1.)
        if self.capture_error:
            raise self.capture_error
        return [response()]

    def simGetCameraInfo(self, *args, **kwargs):
        self.record('simGetCameraInfo', *args, **kwargs)
        if self.camera_error:
            raise self.camera_error
        return SimpleNamespace(pose=airsim.Pose(airsim.Vector3r(0, 0, -90),
            airsim.to_quaternion(-np.pi / 2, 0, np.pi / 2)))


class OverviewTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=2.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.005)
        self.fail('map worker did not reach expected state')

    def make_stream(self, config=None):
        runtime = FakeRuntime()
        values = dict(airsim=dict(host='127.0.0.7', port=41460,
                                  vehicle='DroneControl', rpc_timeout=3.))
        if config:
            values.update(config)
        stream = OverviewStream(values, runtime)
        self.addCleanup(stream.stop)
        return stream, runtime

    def test_raw_bgr_converted_without_vertical_flip(self):
        decoded = decode_scene_frame(response())
        np.testing.assert_array_equal(decoded, RGB)
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertTrue(decoded.flags.c_contiguous)
        for changes in (dict(width=0), dict(height=9000),
                        dict(image_data_uint8=b'\x00'), dict(message='camera missing')):
            with self.subTest(changes=changes), self.assertRaises((ValueError, RuntimeError)):
                decode_scene_frame(response(**changes))

    def test_demo_or_disconnected_does_not_construct_client(self):
        stream, runtime = self.make_stream()
        with patch('airsim.MultirotorClient') as constructor:
            runtime.state.update(mode='demo')
            self.assertFalse(stream.start())
            self.assertEqual(stream.snapshot()['status'], 'UNAVAILABLE')
            runtime.state.update(mode='airsim', connected=False)
            self.assertFalse(stream.start())
            self.assertEqual(stream.snapshot()['status'], 'WAITING')
            constructor.assert_not_called()
        self.assertIsNone(stream._thread)

    def test_separate_thread_owns_all_rpcs_and_returns_rgb(self):
        stream, runtime = self.make_stream()
        original_state = runtime.snapshot()
        client = FakeClient()
        creation_threads = []

        def construct(**kwargs):
            creation_threads.append(threading.get_ident())
            self.assertEqual(kwargs, dict(ip='127.0.0.7', port=41460, timeout_value=1.))
            return client

        with patch('airsim.MultirotorClient', side_effect=construct) as factory:
            self.assertTrue(stream.start())
            self.wait_for(lambda: stream.snapshot()['frames'] >= 1)
            self.assertTrue(stream.start())  # A second viewer call cannot start another socket.
            image = stream.snapshot()
            np.testing.assert_array_equal(image['frame'], RGB)
            self.assertGreater(image['rpc_ms'], 0.)
            self.assertGreaterEqual(image['frame_ms'], image['rpc_ms'])
            self.assertEqual(set(image['overlays']), {'targets', 'planned', 'trail', 'drone', 'nose'})
            image['overlays']['drone'][0] = -9999.
            self.assertNotEqual(stream.snapshot()['overlays']['drone'][0], -9999.)
            image['frame'][:] = 0
            np.testing.assert_array_equal(stream.snapshot()['frame'], RGB)
            stream.stop()
            factory.assert_called_once()
        self.assertTrue(stream._thread.daemon)
        self.assertFalse(stream._thread.is_alive())
        self.assertEqual(stream.snapshot()['status'], 'STOPPED')
        self.assertEqual(runtime.snapshot(), original_state)
        self.assertEqual(set(call[3] for call in client.calls), set(creation_threads))
        self.assertNotEqual(creation_threads[0], threading.get_ident())
        captures = [row for row in client.calls if row[0] == 'simGetImages']
        request = captures[0][1][0][0]
        self.assertEqual(request.camera_name, 'control_overview')
        self.assertEqual(request.image_type, airsim.ImageType.Scene)
        self.assertFalse(request.compress)
        self.assertFalse(request.pixels_as_float)
        self.assertEqual(captures[0][2], dict(vehicle_name='DroneControl', external=True))
        camera_info = [row for row in client.calls if row[0] == 'simGetCameraInfo']
        self.assertEqual(len(camera_info), 1)
        self.assertEqual(camera_info[0][1], ('control_overview',))
        self.assertEqual(camera_info[0][2], dict(vehicle_name='DroneControl', external=True))

    def test_targets_paths_and_bounds_share_height_up_conversion(self):
        stream, runtime = self.make_stream(dict(overview=dict(ue_debug_markers=True)))
        client = FakeClient()
        stream._draw(client, airsim, runtime.snapshot())
        moves = [row for row in client.calls if row[0] == 'simSetObjectPose']
        self.assertEqual([row[1][0] for row in moves], ['DC_TargetRed', 'DC_TargetBlue'])
        self.assertEqual(moves[0][1][1].position.to_numpy_array().tolist(), [2., 3., -4.])
        self.assertTrue(moves[0][2]['teleport'])
        plots = [row for row in client.calls if row[0].startswith('simPlot')]
        self.assertTrue(all(row[2]['is_persistent'] is False for row in plots))
        self.assertTrue(all(.6 <= row[2]['duration'] <= 1. for row in plots))
        points = [row for row in plots if row[0] == 'simPlotPoints']
        self.assertEqual(points[0][2]['color_rgba'], [1., .05, .05, 1.])
        self.assertEqual(points[1][2]['color_rgba'], [.1, .3, 1., 1.])
        paths = [row for row in plots if row[0] == 'simPlotLineStrip']
        self.assertEqual(paths[0][2]['color_rgba'], [.12, .9, .3, 1.])
        self.assertEqual(paths[1][2]['color_rgba'], [.65, .65, .65, 1.])
        self.assertEqual(len(field_edges(ControlConfig())), 24)
        self.assertNotIn('simFlushPersistentMarkers', [row[0] for row in client.calls])

    def test_default_skips_debug_rpcs_but_syncs_real_targets(self):
        stream, runtime = self.make_stream()
        client = FakeClient()
        stream._draw(client, airsim, runtime.snapshot())
        self.assertEqual([row[0] for row in client.calls],
                         ['simSetObjectPose', 'simSetObjectPose'])
        self.assertEqual([row[1][0] for row in client.calls],
                         ['DC_TargetRed', 'DC_TargetBlue'])
        self.assertEqual(client.calls[0][1][1].position.to_numpy_array().tolist(), [2., 3., -4.])
        runtime.state['planned_path'] = [[float('nan'), 0, 5]]
        with self.assertRaises(ValueError):
            stream._draw(client, airsim, runtime.snapshot())

    def test_geometry_decimated_and_invalid_geometry_refused(self):
        cfg = ControlConfig()
        runtime = FakeRuntime()
        runtime.state['trail'] = np.column_stack((np.linspace(0, 10, 3000),
                                                np.zeros(3000), np.full(3000, 5.)))
        geometry = marker_geometry(runtime.snapshot(), cfg)
        self.assertEqual(len(geometry['trail']), 300)
        self.assertEqual(geometry['trail'][0], [0., 0., 5.])
        self.assertEqual(geometry['trail'][-1], [10., 0., 5.])
        for changes in (dict(targets=[[float('nan'), 0, 5], [1, 2, 3]]),
                        dict(targets=[[1000, 0, 5], [1, 2, 3]]),
                        dict(targets=[[1, 2, 3]]),
                        dict(planned_path=[[1, 2]])):
            bad = runtime.snapshot()
            bad.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                marker_geometry(bad, cfg)

    def test_bad_markers_keep_actual_image_and_report_warning(self):
        stream, runtime = self.make_stream()
        runtime.state['targets'] = [[float('nan'), 0, 5], [1, 2, 3]]
        client = FakeClient()
        with patch('airsim.MultirotorClient', return_value=client):
            stream.start()
            self.wait_for(lambda: stream.snapshot()['frames'] == 1)
            snapshot = stream.snapshot()
            stream.stop()
        self.assertEqual(snapshot['status'], 'STREAMING')
        self.assertIn('地图标记未更新', snapshot['message'])
        np.testing.assert_array_equal(snapshot['frame'], RGB)
        self.assertFalse(any(row[0].startswith('simPlot') for row in client.calls))

    def test_missing_actor_is_reported_without_fake_pixels(self):
        stream, runtime = self.make_stream()
        client = FakeClient()
        client.pose_accepted = False
        with patch('airsim.MultirotorClient', return_value=client):
            stream.start()
            self.wait_for(lambda: stream.snapshot()['frames'] >= 1)
            snapshot = stream.snapshot()
            stream.stop()
        self.assertIn('DC_TargetRed 实体未同步', snapshot['message'])
        np.testing.assert_array_equal(snapshot['frame'], RGB)

    def test_capture_error_does_not_touch_flight_or_runtime(self):
        stream, runtime = self.make_stream()
        original = runtime.snapshot()
        client = FakeClient()
        client.capture_error = TimeoutError('injected map RPC timeout')
        with patch('airsim.MultirotorClient', return_value=client):
            stream.start()
            self.wait_for(lambda: stream.snapshot()['status'] == 'ERROR')
            stream._thread.join(1.)
        self.assertIn('injected map RPC timeout', stream.snapshot()['message'])
        self.assertIsNone(stream.snapshot()['frame'])
        self.assertEqual(runtime.snapshot(), original)
        self.assertFalse(stream._thread.is_alive())

    def test_stop_interrupts_cadence_without_new_rpc(self):
        stream, runtime = self.make_stream(dict(overview=dict(fps=.1)))
        client = FakeClient()
        with patch('airsim.MultirotorClient', return_value=client):
            stream.start()
            self.wait_for(lambda: stream.snapshot()['frames'] >= 1)
            before = len([row for row in client.calls if row[0] == 'simGetImages'])
            begin = time.monotonic()
            stream.stop()
            self.assertLess(time.monotonic() - begin, .5)
        after = len([row for row in client.calls if row[0] == 'simGetImages'])
        self.assertEqual(before, after)
        self.assertEqual(stream.snapshot()['status'], 'STOPPED')

    def test_requested_rate_cannot_exceed_four_hz(self):
        stream, runtime = self.make_stream(dict(overview=dict(fps=1000)))
        self.assertEqual(stream._interval, .25)
        with self.assertRaises(ValueError):
            OverviewStream(dict(overview=dict(fps=float('nan'))), runtime)

    def test_orthographic_metadata_matches_known_top_down_axes(self):
        state = FakeRuntime().snapshot()
        state.update(position=[10., 0., 5.], yaw=np.pi / 2,
                     targets=[[0., 10., 4.], [0., 10., 9.]])
        pose = airsim.Pose(airsim.Vector3r(0, 0, -90),
                          airsim.to_quaternion(-np.pi / 2, 0, np.pi / 2))
        overlay = project_overlays(state, pose, 1000, 800, 125.)
        # Yaw 90 downward camera: image-right is -X, image-down is -Y.
        np.testing.assert_allclose(overlay['drone'], [420., 400.], atol=1e-8)
        np.testing.assert_allclose(overlay['nose'], [420., 376.], atol=1e-8)
        np.testing.assert_allclose(overlay['targets'], [[500., 320.], [500., 320.]], atol=1e-8)

    def test_projection_uses_inverse_arbitrary_camera_quaternion(self):
        origin = airsim.Vector3r(3., -7., -80.)
        quaternion = airsim.to_quaternion(-.9, .3, 1.2)
        camera_local = airsim.Quaternionr(50., 10., 5., 0.)
        world_offset = quaternion * camera_local * quaternion.inverse()
        world = [origin.x_val + world_offset.x_val,
                 origin.y_val + world_offset.y_val,
                 -(origin.z_val + world_offset.z_val)]
        state = FakeRuntime().snapshot()
        state.update(position=world, targets=[], planned_path=[], trail=[])
        state['config']['control'] = dict(x_min=-200, x_max=200, y_min=-200,
                                          y_max=200, height_min=2, height_max=200)
        overlay = project_overlays(state, airsim.Pose(origin, quaternion), 1024, 512, 128.)
        np.testing.assert_allclose(overlay['drone'], [512. + 80., 256. + 40.], atol=1e-8)
        self.assertEqual(overlay['targets'], [])

    def test_unavailable_projection_preserves_actual_pixels_with_warning(self):
        stream, runtime = self.make_stream()
        client = FakeClient()
        client.camera_error = RuntimeError('injected camera metadata unavailable')
        with patch('airsim.MultirotorClient', return_value=client):
            stream.start()
            self.wait_for(lambda: stream.snapshot()['frames'] >= 1)
            snapshot = stream.snapshot()
            stream.stop()
        self.assertEqual(snapshot['status'], 'STREAMING')
        self.assertIn('地图投影信息不可用', snapshot['message'])
        self.assertEqual(snapshot['overlays'], {})
        np.testing.assert_array_equal(snapshot['frame'], RGB)

    def test_invalid_camera_pose_or_telemetry_never_emits_projection(self):
        state = FakeRuntime().snapshot()
        valid_pose = airsim.Pose(airsim.Vector3r(0, 0, -90), airsim.Quaternionr())
        for pose in (airsim.Pose(airsim.Vector3r(float('nan'), 0, -90), airsim.Quaternionr()),
                     airsim.Pose(airsim.Vector3r(0, 0, -90), airsim.Quaternionr(0, 0, 0, 0))):
            with self.assertRaises(ValueError):
                project_overlays(state, pose, 1024, 1024, 125.)
        state['yaw'] = float('nan')
        with self.assertRaises(ValueError):
            project_overlays(state, valid_pose, 1024, 1024, 125.)


if __name__ == '__main__':
    unittest.main()
