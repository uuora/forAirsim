"""Observer wiring only; no simulator, images, ownership or flight commands."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from .viewport import GeometricViewport, _GeometricScene


class ViewportTests(unittest.TestCase):
    def test_construction_and_task_changes_do_not_start_observer(self):
        factory = Mock()
        viewport = GeometricViewport({}, {'preparation': {'task': {'start': [0, 0, 5], 'points': [[3, 0, 5]]}}}, factory)
        factory.return_value.request.assert_not_called()
        viewport.update_task({'preparation': {'task': {'start': [0, 0, 5], 'points': [[4, 0, 5]]}}})
        factory.return_value.request.assert_not_called()
        viewport.update_snapshot(dict(position=[1, 0, 5], connected=True))
        self.assertEqual(viewport.snapshot()['position'], [1, 0, 5])
        viewport.request(True)
        factory.return_value.request.assert_called_once_with(True)
        viewport._publish(scene_status='VISIBLE')
        viewport._publish(scene_message='ok')
        self.assertEqual(viewport.take_update(), dict(scene_status='VISIBLE', scene_message='ok'))
        self.assertEqual(viewport.take_update(), {})

    def test_current_process_log_is_used_instead_of_legacy_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'ue_startup.log'
            log.write_text('test startup receipt', encoding='utf-8')
            runtime = Mock()
            scene = _GeometricScene({}, runtime)
            scene._window = Mock()
            scene._window._process_pid.return_value = 42
            process = Mock()
            process.cmdline.return_value = ['UE4Editor.exe', '-abslog=' + str(log)]
            with patch('dual_target_control.native_scene.NativeScene._initialize'), \
                    patch('geometric_validation.viewport.psutil.Process', return_value=process) as lookup:
                scene._initialize()
            lookup.assert_called_once_with(42)
            self.assertEqual(scene._log_path, log)
            runtime.set_connected.assert_called_once_with()

    def test_missing_or_ambiguous_startup_log_fails_explicitly(self):
        scene = _GeometricScene({}, Mock())
        scene._window = Mock()
        for args in ([], ['-abslog=relative.log'], ['-abslog=C:/a', '-abslog=C:/b']):
            with self.subTest(args=args), patch('dual_target_control.native_scene.NativeScene._initialize'), \
                    patch('geometric_validation.viewport.psutil.Process') as process:
                process.return_value.cmdline.return_value = args
                with self.assertRaises(RuntimeError):
                    scene._initialize()
