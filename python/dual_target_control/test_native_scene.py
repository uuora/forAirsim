"""Receipts and isolation: viewing failures must never stop the control owner."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from dual_target_control.native_scene import NativeScene, reflected_command
from dual_target_control.runtime import ControlRuntime


class NativeSceneTests(unittest.TestCase):
    def test_rpc_true_without_actual_execution_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'ue.log'
            log.write_text("Called ' setMode 3' on 1 instance(s) (1 succeeded)\n")
            client = Mock()
            client.simRunConsoleCommand.return_value = True
            # An old success receipt must not validate a new failed/no-op call.
            with self.assertRaisesRegex(RuntimeError, '回执'):
                reflected_command(client, 'ke CameraDirector setMode 3', log, timeout=0)

    def test_reflection_failure_despite_rpc_true_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'ue.log'; log.write_text('')
            client = Mock()
            def write_receipt(command):
                with log.open('a') as stream:
                    stream.write("Called ' setMode 3' on 1 instance(s) (0 succeeded)\n")
                return True
            client.simRunConsoleCommand.side_effect = write_receipt
            with self.assertRaisesRegex(RuntimeError, '执行失败'):
                reflected_command(client, 'ke CameraDirector setMode 3', log)

    def test_view_failure_leaves_flight_status_unchanged(self):
        runtime = Mock()
        state = dict(status='RUNNING', running=True, connected=True, mode='airsim', hits=5)
        runtime.snapshot.side_effect = lambda: dict(state)
        runtime._publish.side_effect = lambda **values: state.update(values)
        scene = NativeScene({}, runtime)
        with patch.object(scene, '_switch', side_effect=RuntimeError('view unavailable')):
            scene.request(True)
            deadline = time.monotonic() + 1
            while state.get('scene_status') != 'ERROR' and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(state['scene_status'], 'ERROR')
            self.assertEqual(state['status'], 'RUNNING')
            self.assertTrue(state['running'])
            self.assertEqual(state['hits'], 5)
            runtime.stop.assert_not_called()
            runtime.pause.assert_not_called()
            runtime.start.assert_not_called()
            self.assertTrue(scene.close(timeout=1))

    def test_offline_view_request_never_connects_or_launches_engine(self):
        runtime = ControlRuntime({'control': {'seed': 13}}, mode='demo')
        try:
            with patch.object(runtime, '_connect') as connect, \
                    patch('dual_target_control.runtime.launch_simulator') as launch:
                runtime.show_scene()
                self.assertIsNone(runtime._scene)
                self.assertFalse(runtime.snapshot()['scene_visible'])
                connect.assert_not_called(); launch.assert_not_called()
        finally:
            runtime.shutdown()
            self.assertTrue(runtime.wait_closed(2))

    def test_failed_rpc_rollback_still_hides_the_native_window(self):
        runtime = Mock()
        runtime.snapshot.return_value = dict(mode='airsim', connected=True)
        scene = NativeScene({'airsim': {'map': '/Game/Test/Test'}}, runtime)
        scene._client = Mock(); scene._window = Mock()
        scene._window.show.return_value = dict(pid=7, hwnd=9)
        scene._window.hide.return_value = dict(pid=7, hwnd=9)
        with patch.object(scene, '_initialize'), \
                patch('dual_target_control.native_scene.reflected_command',
                      side_effect=RuntimeError('RPC unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'RPC unavailable'):
                scene._switch(True)
        scene._window.hide.assert_called_once()
        self.assertFalse(scene._visible)
        self.assertTrue(scene._needs_restore)
        self.assertTrue(runtime._publish.call_args.kwargs['scene_render_unknown'])

    def test_close_preempts_queued_view_changes(self):
        import threading
        runtime = Mock()
        scene = NativeScene({}, runtime)
        entered = threading.Event(); release = threading.Event()
        def slow_switch(visible):
            entered.set(); release.wait(1)
        with patch.object(scene, '_switch', side_effect=slow_switch) as switch:
            scene.request(True)
            self.assertTrue(entered.wait(1))
            scene.request(False); scene.request(True)
            scene.close(timeout=0)
            release.set()
            self.assertTrue(scene.close(timeout=1))
            switch.assert_called_once_with(True)

    def test_config_replacement_waits_for_observer_to_finish(self):
        runtime = ControlRuntime({'control': {'seed': 13}}, mode='demo')
        old_scene = Mock(); old_scene.close.return_value = False
        runtime._scene = old_scene
        try:
            runtime.configure({'control': {'seed': 13, 'max_speed': 9}})
            deadline = time.monotonic() + 1
            while runtime.snapshot()['status'] != 'ERROR' and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(runtime.snapshot()['status'], 'ERROR')
            self.assertIs(runtime._scene, old_scene)
            self.assertEqual(runtime.cfg.max_speed, 4)
        finally:
            runtime._scene = None
            runtime.shutdown()
            self.assertTrue(runtime.wait_closed(2))


if __name__ == '__main__': unittest.main()
