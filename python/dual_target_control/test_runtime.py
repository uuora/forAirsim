"""Runtime lifecycle and telemetry regressions, without AirSim or flight."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dual_target_control.runtime import ControlRuntime, from_ned, json_value, to_ned


class RuntimeTests(unittest.TestCase):
    def make_runtime(self):
        # Demo mode cannot import a flight client or issue simulator commands.
        runtime = ControlRuntime({"control": {"seed": 13}}, mode="demo")
        self.addCleanup(self.close_runtime, runtime)
        return runtime

    @staticmethod
    def close_runtime(runtime):
        # Fault-injection tests must never leave their non-daemon worker alive,
        # including when an assertion fails before the recovery action.
        runtime._active = False
        runtime._flight_owned = False
        runtime.mode = "demo"
        runtime._land = lambda: None
        runtime.shutdown()
        if not runtime.wait_closed(2):
            runtime._close_requested = True
            runtime._commands.put(("shutdown", None))
        runtime._thread.join(2)
        if runtime._thread.is_alive():
            raise AssertionError("test left its control worker running")

    def wait_for(self, predicate, timeout=2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("runtime did not reach the expected state")

    def test_frame_conversion_flips_only_vertical_axis(self):
        original = np.array([3.25, -7.5, 4.75])
        ned = to_ned(original)
        self.assertEqual(ned, [3.25, -7.5, -4.75])
        recovered = from_ned(SimpleNamespace(x_val=ned[0], y_val=ned[1], z_val=ned[2]))
        np.testing.assert_array_equal(recovered, original)

    def test_nested_numpy_event_is_json_serializable(self):
        event = {"target": np.array([1., -2., 6.]), "hit": np.bool_(True),
                 "details": (np.int64(3), {"error": np.float32(.125)})}
        native = json.loads(json.dumps(json_value(event)))
        self.assertEqual(native, {"target": [1., -2., 6.], "hit": True,
                                  "details": [3, {"error": .125}]})

    def test_constructor_starts_worker_but_never_starts_flight(self):
        with patch.object(ControlRuntime, "_start") as start, \
             patch.object(ControlRuntime, "_connect") as connect, \
             patch("dual_target_control.runtime.launch_simulator") as launch:
            runtime = self.make_runtime()
            time.sleep(.03)  # Give the command consumer multiple idle turns.
            self.assertTrue(runtime._thread.is_alive())
            self.assertEqual(runtime.snapshot()["status"], "IDLE")
            self.assertFalse(runtime._active)
            self.assertFalse(runtime._flight_owned)
            self.assertIsNone(runtime._client)
            self.assertIsNone(runtime._run_dir)
            start.assert_not_called()
            connect.assert_not_called()
            launch.assert_not_called()

    def test_failed_shutdown_stays_open_and_stop_retry_closes_after_landing(self):
        runtime = self.make_runtime()
        runtime._flight_owned = True
        allow_landing = threading.Event()
        attempts = []

        def controlled_landing():
            attempts.append("landing")
            if not allow_landing.is_set():
                raise RuntimeError("injected stationary-contact failure")
            runtime._flight_owned = False

        runtime._land = controlled_landing
        runtime.shutdown()
        self.wait_for(lambda: runtime.snapshot()["status"] == "ERROR")
        self.assertFalse(runtime.wait_closed(.03))
        self.assertTrue(runtime._thread.is_alive())
        self.assertTrue(runtime._flight_owned)
        self.assertTrue(runtime.snapshot()["running"])
        self.assertIn("stationary-contact failure", runtime.snapshot()["message"])
        self.assertEqual(len(attempts), 1)

        allow_landing.set()
        runtime.stop()
        self.assertTrue(runtime.wait_closed(2))
        runtime._thread.join(2)
        self.assertFalse(runtime._thread.is_alive())
        self.assertFalse(runtime._flight_owned)
        self.assertFalse(runtime.snapshot()["running"])
        self.assertEqual(runtime.snapshot()["status"], "ERROR")
        self.assertIn("stationary-contact failure", runtime.snapshot()["message"])
        self.assertEqual(len(attempts), 2)

    def test_stop_and_shutdown_preserve_original_error(self):
        runtime = self.make_runtime()
        runtime._failure = "injected RPC timeout"
        runtime._publish("ERROR", runtime._failure)
        saved = Mock()
        runtime._save = saved
        runtime.stop()
        runtime.shutdown()
        self.assertTrue(runtime.wait_closed(2))
        self.assertEqual(runtime.snapshot()["status"], "ERROR")
        self.assertEqual(runtime.snapshot()["message"], "injected RPC timeout")
        self.assertFalse(runtime.snapshot()["running"])
        self.assertGreaterEqual(saved.call_count, 2)
        self.assertEqual(saved.call_args_list[0].args, ("FAILED",))
        self.assertEqual(saved.call_args_list[1].args, ("FAILED",))

    def test_raw_interval_preserved_while_controller_dt_is_bounded(self):
        runtime = self.make_runtime()
        runtime._start_time = time.perf_counter()
        runtime.controller.reset(runtime._demo.position, runtime._demo.velocity, runtime._demo.yaw)
        runtime.controller.update = Mock(wraps=runtime.controller.update)
        runtime._demo.step = Mock(wraps=runtime._demo.step)
        runtime._tick(.25)
        self.assertEqual(runtime._samples[-1]["interval_ms"], 250.)
        self.assertEqual(runtime._timings[-1][2], 250.)
        self.assertEqual(runtime.controller.update.call_args.args[-1], .1)
        self.assertEqual(runtime._demo.step.call_args.args[-1], .1)
        json.dumps(runtime._samples[-1])

    def test_failed_summary_keeps_error_and_ownership_evidence(self):
        runtime = self.make_runtime()
        with tempfile.TemporaryDirectory(prefix="dual_target_runtime_test_") as folder:
            runtime._run_dir = Path(folder)
            runtime._failure = "injected landing failure"
            runtime._flight_owned = True
            runtime._events = [{"point": np.array([2., 3., 4.]), "index": np.int64(1)}]
            runtime._save("FAILED")
            summary = json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["outcome"], "FAILED")
            self.assertEqual(summary["error"], "injected landing failure")
            self.assertTrue(summary["flight_owned"])
            self.assertEqual(summary["events"], [{"point": [2., 3., 4.], "index": 1}])
            runtime._run_dir = None

    def test_camera_compensates_body_attitude_and_actual_camera_origin(self):
        import airsim  # Quaternion math only; no real client is constructed.

        runtime = self.make_runtime()
        runtime.mode = "airsim"
        runtime.config["airsim"] = {"aim_camera": True, "camera": "front_center"}
        runtime._airsim = airsim
        runtime._client = Mock()
        position = np.array([-3., 2., 5.])
        cases = [(.31, -.27, 1.18, np.array([2., -4., 1.8])),
                 (-.42, .36, -2.6, np.array([-.7, 1.2, -1.5]))]
        for pitch, roll, yaw, target_offset in cases:
            with self.subTest(body_pitch=pitch, body_roll=roll, body_yaw=yaw):
                body = airsim.to_quaternion(pitch, roll, yaw)
                runtime._last_state = SimpleNamespace(
                    kinematics_estimated=SimpleNamespace(orientation=body))
                runtime._last_camera = 0.
                runtime._client.reset_mock()
                target = position + target_offset
                runtime._aim_camera(position, target, 1.)
                runtime._client.simSetCameraPose.assert_called_once()
                call = runtime._client.simSetCameraPose.call_args
                self.assertEqual(call.args[0], "front_center")
                self.assertEqual(call.kwargs["vehicle_name"], "DroneControl")
                pose = call.args[1]
                np.testing.assert_allclose(
                    [pose.position.x_val, pose.position.y_val, pose.position.z_val],
                    [.1, 0., -.03], atol=1e-12)

                camera_offset_world = body * pose.position.to_Quaternionr() * body.inverse()
                camera_origin = position + from_ned(camera_offset_world)
                expected_direction = target - camera_origin
                expected_direction /= np.linalg.norm(expected_direction)

                world_orientation = body * pose.orientation
                forward_ned = (world_orientation * airsim.Quaternionr(1., 0., 0., 0.)
                               * world_orientation.inverse())
                actual_direction = from_ned(forward_ned)
                actual_direction /= np.linalg.norm(actual_direction)
                np.testing.assert_allclose(actual_direction, expected_direction, atol=1e-10)

    def test_camera_preserves_last_valid_pose_inside_arrival_radius(self):
        import airsim

        runtime = self.make_runtime()
        runtime.mode = "airsim"
        runtime.config["airsim"] = {"aim_camera": True}
        runtime._airsim = airsim
        runtime._client = Mock()
        body = airsim.to_quaternion(.2, -.3, .8)
        runtime._last_state = SimpleNamespace(
            kinematics_estimated=SimpleNamespace(orientation=body))
        position = np.array([1., 2., 5.])
        runtime._aim_camera(position, position + [2., 0., 1.], 1.)
        original_pose = runtime._client.simSetCameraPose.call_args.args[1]
        original_rotation = original_pose.orientation.to_numpy_array().copy()

        # The target changes from ahead to behind while the vehicle passes it.
        # Inside the hit sphere there is no well-defined direction at coincidence.
        runtime._aim_camera(position, position + [.2, 0., 0.], 1.1)
        runtime._aim_camera(position, position + [-.2, 0., 0.], 1.2)
        runtime._client.simSetCameraPose.assert_called_once()
        np.testing.assert_array_equal(original_pose.orientation.to_numpy_array(), original_rotation)
        self.assertEqual(runtime._last_camera, 1.)

    def make_landing_runtime(self, already_disarmed=False, landed_flags=(1, 1, 0)):
        runtime = self.make_runtime()
        runtime.mode = "airsim"
        runtime._flight_owned = True
        runtime._motors_disarmed = already_disarmed
        runtime._airsim = SimpleNamespace(LandedState=SimpleNamespace(Flying=1, Landed=0))
        client = runtime._client = Mock()
        position = np.array([2., -1., -.8])
        runtime._read_state = Mock(return_value=(position, np.zeros(3), 0., 0.))
        # Retained floor contact is valid even after has_collided is cleared.
        client.simGetCollisionInfo.return_value = SimpleNamespace(
            has_collided=False, object_name="DC_CollisionFloor",
            normal=SimpleNamespace(z_val=-1.),
            position=SimpleNamespace(x_val=2., y_val=-1., z_val=.8))
        records = []
        state = {"disarmed": already_disarmed, "landed": False}
        remaining = iter(landed_flags)
        runtime._send = Mock(side_effect=lambda *args: records.append("descent"))
        client.cancelLastTask.side_effect = lambda **kwargs: records.append("cancel")

        def disarm(armed, **kwargs):
            self.assertFalse(armed)
            self.assertEqual(kwargs["vehicle_name"], "DroneControl")
            records.append("disarm")
            state["disarmed"] = True
            return True

        def read_landed(**kwargs):
            self.assertTrue(state["disarmed"])
            flag = next(remaining)
            records.append("landed" if flag == 0 else "flying")
            state["landed"] = flag == 0
            return SimpleNamespace(landed_state=flag)

        def release(enabled, **kwargs):
            self.assertFalse(enabled)
            self.assertTrue(state["landed"], "API control released before Landed confirmation")
            records.append("release")

        client.armDisarm.side_effect = disarm
        client.getMultirotorState.side_effect = read_landed
        client.enableApiControl.side_effect = release
        return runtime, client, records

    def test_landing_waits_for_asynchronous_landed_flag_after_disarm(self):
        runtime, client, records = self.make_landing_runtime()
        runtime._land()
        client.armDisarm.assert_called_once_with(False, vehicle_name="DroneControl")
        self.assertEqual(client.getMultirotorState.call_count, 3)
        self.assertEqual(records[-4:], ["flying", "flying", "landed", "release"])
        self.assertEqual(runtime._send.call_count, 8)
        self.assertLess(max(i for i, value in enumerate(records) if value == "descent"),
                        records.index("disarm"))
        client.enableApiControl.assert_called_once_with(False, vehicle_name="DroneControl")
        self.assertTrue(runtime._motors_disarmed)
        self.assertFalse(runtime._flight_owned)

    def test_landing_retry_after_disarm_sends_no_descent_or_second_disarm(self):
        runtime, client, records = self.make_landing_runtime(
            already_disarmed=True, landed_flags=(1, 0))
        runtime._land()
        runtime._send.assert_not_called()
        client.armDisarm.assert_not_called()
        self.assertEqual(client.cancelLastTask.call_count, 1)
        self.assertEqual(client.getMultirotorState.call_count, 2)
        self.assertEqual(records[-3:], ["flying", "landed", "release"])
        client.enableApiControl.assert_called_once_with(False, vehicle_name="DroneControl")
        self.assertTrue(runtime._motors_disarmed)
        self.assertFalse(runtime._flight_owned)


if __name__ == "__main__":
    unittest.main()
