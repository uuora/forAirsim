"""No-RPC fake-client lifecycle/sign tests. Not simulated flight evidence."""
from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from geometric_validation import runtime


class Done:
    _set_flag = True

    def get(self):
        return None


class FakeClient:
    """A deterministic API stub; it intentionally contains no vehicle dynamics."""
    def __init__(self, start, yaw):
        self.start = start
        self.yaw = yaw
        self.stamp = 1_000_000_000
        self.armed = False
        self.owner = False
        self.attitudes = []
        self.velocity_calls = []

    @staticmethod
    def vec(value):
        return SimpleNamespace(x_val=float(value[0]), y_val=float(value[1]), z_val=float(value[2]))

    def getMultirotorState(self, **_kwargs):
        self.stamp += 20_000_000
        zeros = self.vec([0, 0, 0])
        return SimpleNamespace(timestamp=self.stamp, landed_state=int(self.armed),
            kinematics_estimated=SimpleNamespace(position=self.vec(self.start), linear_velocity=zeros,
                linear_acceleration=zeros, angular_velocity=zeros,
                orientation=SimpleNamespace(w_val=math.cos(self.yaw/2), x_val=0., y_val=0., z_val=math.sin(self.yaw/2))),
            collision=SimpleNamespace(has_collided=False, time_stamp=0))

    def enableApiControl(self, enabled, **_kwargs):
        self.owner = enabled

    def armDisarm(self, armed, **_kwargs):
        self.armed = armed
        return True

    def moveByVelocityAsync(self, *args, **kwargs):
        self.velocity_calls.append((args, kwargs))
        return Done()

    def moveByRollPitchYawThrottleAsync(self, *args, **kwargs):
        self.attitudes.append((args, kwargs))
        return Done()

    def cancelLastTask(self, **_kwargs):
        pass

    def simGetCollisionInfo(self, **_kwargs):
        # Synthetic cleanup receipt only. This is not a model of landing.
        return SimpleNamespace(position=self.vec(self.start), object_name="DC_CollisionFloor", normal=self.vec([0, 0, -1]))


class RuntimeContractTests(unittest.TestCase):
    def test_snapshot_paths_longer_than_windows_legacy_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/('nested_'*18)/('source_'*18)/'receipt.json'
            extended=runtime.filesystem_path(path)
            extended.parent.mkdir(parents=True)
            try:
                runtime.write_json(path,{'snapshot':True})
                self.assertEqual(json.loads(extended.read_text(encoding='utf-8')),{'snapshot':True})
                self.assertEqual(len(runtime.digest(path)),64)
            finally:
                extended.unlink(missing_ok=True)
                extended.parent.rmdir()
                extended.parent.parent.rmdir()

    def test_loading_default_config_does_not_open_rpc_or_launch(self):
        with patch("socket.create_connection", side_effect=AssertionError("Unexpected socket")), \
             patch("subprocess.Popen", side_effect=AssertionError("Unexpected launch")):
            cfg = runtime.load_config()
        self.assertEqual(cfg["protocol"], "geometric_hybrid_v1")
        self.assertIs(cfg["visualization"]["capture_overview"], False)

    def synthetic_run(self, stop=False):
        cfg = json.loads(runtime.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        cfg["yaw"] = .3
        cfg["execution"].update(start_hold_s=.02, terminal_hold_s=.1,
                                start_timeout_s=2., settle_timeout_s=2.)
        cfg["cases"] = {"line": {"offsets": [[.001, 0, 0]], "segment_durations": [.2]}}
        client = FakeClient(cfg["start"], cfg["yaw"])
        stop_event = threading.Event()
        if stop:
            stop_event.set()
        temporary = tempfile.TemporaryDirectory(prefix="geometric_fake_runtime_")
        self.addCleanup(temporary.cleanup)
        with patch.object(runtime, "load_config", return_value=cfg), \
             patch.object(runtime, "local_preflight", return_value={"files": {}}), \
             patch.object(runtime, "connect", return_value=(client, None)), \
             patch.object(runtime, "inspect_live", return_value={"initial_state": {"collision_timestamp_ns": 0}, "loaded_settings": {}}), \
             patch.object(runtime.time, "sleep", return_value=None):
            result = runtime.run(case="line", output_root=temporary.name, stop_event=stop_event)
        directory = Path(result["run_dir"])
        rows = [json.loads(line) for line in (directory/"samples.jsonl").read_text(encoding="utf-8").splitlines()]
        return result, client, rows

    def test_physical_euler_to_airsim_wrapper_signs_and_cleanup(self):
        result, client, rows = self.synthetic_run()
        command_rows = [r for r in rows if "command" in r and r.get("command_sent")]
        self.assertEqual(len(command_rows), len(client.attitudes))
        self.assertGreater(len(command_rows), 2)
        self.assertTrue(any(abs(r["command"]["pitch"]) > 1e-5 for r in command_rows), "Fixture must exercise pitch")
        for row, (args, kwargs) in zip(command_rows, client.attitudes):
            cmd = row["command"]
            self.assertAlmostEqual(args[0], cmd["roll"])
            self.assertAlmostEqual(args[1], -cmd["pitch"])
            self.assertAlmostEqual(args[2], -cmd["yaw"])
            self.assertAlmostEqual(args[3], cmd["throttle"])
            self.assertEqual(kwargs["vehicle_name"], "DroneControl")
        self.assertTrue(result["report"]["normal_completion"], result["report"])
        self.assertFalse(client.owner)
        self.assertFalse(client.armed)
        self.assertTrue(result["report"]["cleanup"]["landed"])

    def test_preparation_stop_preserves_failure_and_performs_cleanup(self):
        result, client, rows = self.synthetic_run(stop=True)
        self.assertFalse(result["report"]["normal_completion"])
        self.assertFalse(result["report"]["accepted"])
        self.assertEqual(client.attitudes, [])
        self.assertFalse(client.owner)
        self.assertFalse(client.armed)
        self.assertTrue(any(r["phase"] == "LANDING" for r in rows))


if __name__ == "__main__":
    unittest.main()
