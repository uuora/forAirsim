"""Non-flight adapter checks: all RPC/runtime/process boundaries are fake.

Fixtures are created only in D:\\forAirsim\\reports\\interface_v24_checks.
No detector, real UE, subprocess, RPC, or model is started by this suite.
"""
import hashlib
import json
import os
import sys
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import fixed_map_dashboard_session_v24 as adapter


class FakeProcess:
    def __init__(self, pid=880001):
        self.pid = pid
        self.exit_code = None

    def poll(self):
        return self.exit_code

    def terminate(self):
        raise AssertionError("Adapter must never terminate a control owner")

    kill = terminate


class SessionChecks(unittest.TestCase):
    def setUp(self):
        self.root = adapter.ROOT / "reports/interface_v24_checks" / ("run_" + uuid.uuid4().hex)
        (self.root / "reports").mkdir(parents=True)
        (self.root / "configs").mkdir()
        self.config = self.root / "configs/mission.json"
        self.detector = self.root / "configs/detector.json"
        self.weights = self.root / "weights.bin"
        self.receipt = self.root / "training.json"
        self.weights.write_bytes(b"frozen weights fixture")
        self.receipt.write_text('{"training":"fixed"}', encoding="utf-8")
        self.detector.write_text(json.dumps(dict(weights=str(self.weights),
            weights_sha256=adapter._sha(self.weights), training_receipt=str(self.receipt),
            training_receipt_sha256=adapter._sha(self.receipt), confidence=.25)), encoding="utf-8")
        self.config.write_text(json.dumps(dict(schema="fixture_v24", detector_config="configs/detector.json")), encoding="utf-8")
        self.engine = dict(pid=99111, created=100.)
        self.ground = dict(passed=True, vehicles={name: dict(landed=True, api_control=False, fresh=True)
                                                for name in ["DroneA", "DroneB"]})
        self.claim = self.root / "reports/owner.json"
        self.stack = []
        self._patch("ROOT", self.root)
        self._patch("CONFIG_PATH", self.config)
        self._patch("OWNER_CLAIM", self.claim)
        self._patch("CONFIG_SHA256", adapter._sha(self.config))
        self._patch("DETECTOR_CONFIG_SHA256", adapter._sha(self.detector))
        self._patch("WEIGHTS_SHA256", adapter._sha(self.weights))
        self._patch("TRAINING_RECEIPT_SHA256", adapter._sha(self.receipt))
        self._patch("_identity", lambda pid: dict(pid=pid, created=100.))
        self._patch("_alive", lambda identity: identity["pid"] in {os.getpid(), 880001})
        self.fake_processes = []
        psutil = types.ModuleType("psutil")
        psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        psutil.process_iter = lambda attrs: self.fake_processes
        runtime = types.ModuleType("fixed_map_runtime")
        runtime.configure_session = self._configure_session
        cpu = types.ModuleType("fixed_map_cpu_runtime")
        cpu.configure_math_runtime = Mock(return_value={})
        mission = types.ModuleType("fixed_map_mission")
        mission.ensure_scene = Mock(side_effect=self._ensure_scene)
        cleanup = types.ModuleType("fixed_map_trial_cleanup")
        cleanup.probe_ground = Mock(side_effect=lambda: dict(self.ground))
        binding = types.ModuleType("fixed_map_scene_binding")
        binding.current_engine_identity = Mock(side_effect=lambda: dict(self.engine))
        binding.bind_scene_config = Mock(side_effect=lambda config, receipt: dict(config, runtime_scene_version="epoch_fixture"))
        self.modules = dict(psutil=psutil, fixed_map_runtime=runtime, fixed_map_cpu_runtime=cpu,
            fixed_map_mission=mission, fixed_map_trial_cleanup=cleanup, fixed_map_scene_binding=binding)
        module_patch = patch.dict(sys.modules, self.modules)
        module_patch.start()
        self.stack.append(module_patch)
        env_patch = patch.object(adapter.V24Session, "_environment", return_value={})
        env_patch.start()
        self.stack.append(env_patch)
        self.popen = patch.object(adapter.subprocess, "Popen", return_value=FakeProcess())
        self.popen_mock = self.popen.start()
        self.stack.append(self.popen)
        self.current_bootstrap = None
        self.sessions = []

    def _patch(self, name, value):
        item = patch.object(adapter, name, value)
        item.start()
        self.stack.append(item)

    def _configure_session(self, output):
        output.mkdir(exist_ok=True)
        self.current_bootstrap = output

    def _ensure_scene(self):
        adapter._write(self.current_bootstrap / "active_engine.json", self.engine)

    def session(self, ready=False):
        result = adapter.V24Session(self.root / "reports" / ("session_" + uuid.uuid4().hex))
        self.sessions.append(result)
        if ready:
            self.assertEqual(result.bootstrap()["status"], "READY")
        return result

    @staticmethod
    def prepare(arguments, log):
        Path(arguments[1]).mkdir(exist_ok=False)
        log.write_text("fake ground preparation", encoding="utf-8")

    def tearDown(self):
        for session in self.sessions:
            if session._process is not None:
                session._process.exit_code = 0
            session.close()
        for item in reversed(self.stack):
            item.stop()
        # Retain these tiny fixtures and receipts as check evidence; no cleanup
        # ever targets C or user experiment data.

    def test_config_change_is_rejected(self):
        session = self.session()
        self.config.write_text(self.config.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "FROZEN_CONFIG_HASH"):
            session._verify_frozen()
        self.popen_mock.assert_not_called()

    def test_detector_parameter_change_is_rejected(self):
        session = self.session()
        detector = json.loads(self.detector.read_text())
        detector["confidence"] = .01
        self.detector.write_text(json.dumps(detector), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "DETECTOR_CONFIG_HASH"):
            session._verify_frozen()
        self.popen_mock.assert_not_called()

    def test_c_drive_and_outside_workspace_are_rejected(self):
        for path in [Path("C:/v24_forbidden"), self.root.parent / "outside"]:
            with self.assertRaisesRegex(ValueError, "D_WORKSPACE"):
                adapter.V24Session(path)
        self.assertFalse(Path("C:/v24_forbidden").exists())

    def test_request_stop_does_not_create_driver_output(self):
        session = self.session()
        trial = session.output / "trial_01"
        trial.mkdir()
        session._trial_root = trial
        session._trial_output = trial / "online_trial"
        session._process = FakeProcess()
        receipt = session.request_stop()
        self.assertTrue(receipt["requested"])
        self.assertTrue((trial / "stop.request").is_file())
        self.assertFalse(session._trial_output.exists())
        self.popen_mock.assert_not_called()

    def test_bootstrap_and_poll_never_launch_trial_owner(self):
        session = self.session()
        self.assertFalse(session.poll()["running"])
        receipt = session.bootstrap()
        self.assertTrue(receipt["ground"]["passed"])
        self.assertEqual(receipt["status"], "READY")
        self.assertFalse(session.poll()["running"])
        self.assertEqual(session.poll()["engine_info"]["pid"], self.engine["pid"])
        self.popen_mock.assert_not_called()

    def test_stale_scene_bootstrap_offers_sync_without_owner(self):
        session = self.session()
        self.modules["fixed_map_scene_binding"].bind_scene_config.side_effect = ValueError("SCENE_ENGINE_LIFETIME_CHANGED")
        self.assertEqual(session.bootstrap()["status"], "READY_SCENE_SYNC_REQUIRED")
        self.assertFalse(session.poll()["scene_bound"])
        self.popen_mock.assert_not_called()

    def test_pre_set_cancel_never_prepares_or_spawns(self):
        session = self.session(ready=True)
        event = threading.Event()
        event.set()
        with patch.object(session, "_command") as command:
            with self.assertRaisesRegex(RuntimeError, "CANCELLED"):
                session.start_trial("识别接近气球后返航", event)
            command.assert_not_called()
        self.popen_mock.assert_not_called()

    def test_cancel_during_prepare_never_spawns(self):
        session = self.session(ready=True)
        event = threading.Event()
        def prepare_and_cancel(arguments, log):
            self.prepare(arguments, log)
            event.set()
        with patch.object(session, "_command", side_effect=prepare_and_cancel):
            with self.assertRaisesRegex(RuntimeError, "CANCELLED"):
                session.start_trial("识别接近气球后返航", event)
        self.assertFalse((session.output / "trial_01/online_trial").exists())
        self.assertFalse(self.claim.exists())
        self.popen_mock.assert_not_called()

    def test_backend_stop_during_prepare_never_spawns(self):
        session = self.session(ready=True)
        def prepare_and_stop(arguments, log):
            self.prepare(arguments, log)
            session.request_stop()
        with patch.object(session, "_command", side_effect=prepare_and_stop):
            with self.assertRaisesRegex(RuntimeError, "CANCELLED"):
                session.start_trial("识别接近气球后返航")
        self.popen_mock.assert_not_called()
        self.assertTrue((session.output / "trial_01/stop.request").is_file())
        self.assertFalse((session.output / "trial_01/online_trial").exists())

    def test_cancel_immediately_before_popen_never_spawns(self):
        session = self.session(ready=True)
        event = threading.Event()
        calls = 0
        def environment():
            nonlocal calls
            calls += 1
            event.set()
            return {}
        with patch.object(session, "_command", side_effect=self.prepare), patch.object(session, "_environment", side_effect=environment):
            with self.assertRaisesRegex(RuntimeError, "CANCELLED"):
                session.start_trial("识别接近气球后返航", event)
        self.assertEqual(calls, 1)
        self.popen_mock.assert_not_called()

    def test_live_owner_claim_cannot_be_replaced_by_another_window(self):
        first, second = self.session(), self.session()
        first._claim_owner()
        original = self.claim.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "OWNER_RUNNING"):
            second._claim_owner()
        self.assertEqual(self.claim.read_bytes(), original)
        self.assertIsNone(second._claim_token)

    def test_leftover_sensor_worker_blocks_another_owner(self):
        session = self.session()
        self.fake_processes.append(types.SimpleNamespace(pid=55123, info={"cmdline": ["python.exe",
            str(self.root / "python/fixed_map_perception_process.py"), "--worker", "output"]}))
        with self.assertRaisesRegex(RuntimeError, "SENSOR_WORKER_RUNNING"):
            session._claim_owner()
        self.assertFalse(self.claim.exists())

    def test_successful_mock_launch_is_detached_and_request_pinned(self):
        session = self.session(ready=True)
        with patch.object(session, "_command", side_effect=self.prepare):
            receipt = session.start_trial("识别接近气球后返航", threading.Event())
        self.assertTrue(receipt["owner_running"])
        arguments, options = self.popen_mock.call_args
        command = arguments[0]
        path = Path(command[command.index("--interface-request") + 1])
        self.assertEqual(command[command.index("--interface-request-sha256") + 1], adapter._sha(path))
        self.assertEqual(command[command.index("--stop-request") + 1], str(session._trial_root / "stop.request"))
        self.assertFalse(session._trial_output.exists())
        if os.name == "nt":
            self.assertTrue(options["creationflags"] & adapter.subprocess.DETACHED_PROCESS)
        session.close()
        self.assertIsNone(session._process.poll())
        self.assertTrue(session._log is not None and not session._log.closed)
        self.assertTrue(session._trial_root.joinpath("stop.request").is_file())

    def test_metadata_write_failure_after_launch_requests_stop(self):
        session = self.session(ready=True)
        original_write = adapter._write
        def fail_owner_record(path, value):
            if Path(path).name == "OWNER_PROCESS.json":
                raise OSError("fake metadata write failure")
            return original_write(path, value)
        with patch.object(session, "_command", side_effect=self.prepare), patch.object(adapter, "_write", side_effect=fail_owner_record):
            with self.assertRaisesRegex(OSError, "metadata"):
                session.start_trial("识别接近气球后返航")
        self.assertIsNone(session._process.poll())
        self.assertTrue((session._trial_root / "stop.request").is_file())
        self.assertTrue(self.claim.exists())

    def test_incomplete_previous_final_blocks_new_start(self):
        session = self.session(ready=True)
        session._trial_output = session.output / "previous"
        session._trial_output.mkdir()
        session._previous_cleanup_required = True
        adapter._write(session._trial_output / "result.json", dict(terminal_phase="FINAL_UNVERIFIED_GROUND",
            ground_verification_passed=False, cleanup_errors=[]))
        with self.assertRaisesRegex(RuntimeError, "PREVIOUS_TRIAL_FINAL_CLEANUP"):
            session.start_trial("识别接近气球后返航")
        self.popen_mock.assert_not_called()

    def test_partial_publication_preserves_image_but_marks_not_current(self):
        session = self.session()
        output = session.output / "online_trial"
        (output / "online").mkdir(parents=True)
        session._trial_output = output
        session._process = FakeProcess()
        path = output / "online/publication.json"
        adapter._write(path, dict(latest=dict(valid=True, frame_id="frame-1",
            capture=dict(rgb_path="D:/frame.png", received_monotonic_s=time.monotonic())), closed=False))
        first = session.poll()
        self.assertTrue(first["latest"]["valid"])
        path.write_text('{"latest":', encoding="utf-8")
        second = session.poll()
        self.assertEqual(second["latest"]["frame_id"], "frame-1")
        self.assertEqual(second["latest"]["capture"]["rgb_path"], "D:/frame.png")
        self.assertFalse(second["latest"]["valid"])
        self.assertFalse(second["latest_current"])
        self.assertFalse(second["publication_current_read"])
        self.assertIn("publication", second["records_cached"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
