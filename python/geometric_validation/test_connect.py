"""No-RPC launcher regressions; all sockets, clients, clocks and processes fake."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from . import runtime


CFG = {"airsim": {"host": "127.0.0.1", "port": 41460, "rpc_timeout": 2.,
                  "editor": "fake/UE4Editor.exe", "project": "fake/Blocks.uproject",
                  "map": "/Game/FakeNoFlightMap", "settings": "fake/settings.json"}}


class ConnectTests(unittest.TestCase):
    def test_launch_waits_for_open_port_before_constructing_rpc_client(self):
        calls = []
        probe_count = 0
        port_open = False
        client = SimpleNamespace(ping=Mock(return_value=True), armDisarm=Mock(), enableApiControl=Mock())
        process = SimpleNamespace(pid=12345, poll=Mock(return_value=None))

        def probe(*args, **kwargs):
            nonlocal probe_count, port_open
            probe_count += 1
            calls.append("socket_probe")
            # Initial existing-port check and first post-launch check are closed.
            if probe_count <= 2:
                raise ConnectionRefusedError("fake port not yet ready")
            port_open = True
            return MagicMock()

        def create_client(**kwargs):
            self.assertTrue(port_open, "RPC client must not be created while launch port is closed")
            calls.append("client_created")
            return client

        def popen(*args, **kwargs):
            calls.append("launch")
            return process

        fake_airsim = SimpleNamespace(MultirotorClient=Mock(side_effect=create_client))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"airsim": fake_airsim}), \
                patch.object(runtime.socket, "create_connection", side_effect=probe), \
                patch.object(runtime.subprocess, "Popen", side_effect=popen) as launch_mock, \
                patch("dual_target_control.simulator.hidden_process_options", return_value={}), \
                patch.object(runtime.time, "perf_counter", return_value=0.), \
                patch.object(runtime.time, "sleep") as sleep_mock:
            connected, owned_process = runtime.connect(CFG, Path(directory), launch=True)
        self.assertIs(connected, client)
        self.assertIs(owned_process, process)
        self.assertEqual(calls, ["socket_probe", "launch", "socket_probe", "socket_probe", "client_created"])
        launch_mock.assert_called_once()
        fake_airsim.MultirotorClient.assert_called_once_with(ip="127.0.0.1", port=41460, timeout_value=2.)
        sleep_mock.assert_called_once_with(.2)
        client.ping.assert_called_once_with()
        client.armDisarm.assert_not_called()
        client.enableApiControl.assert_not_called()

    def test_ping_false_expires_once_without_recreating_client_or_launching(self):
        client = SimpleNamespace(ping=Mock(return_value=False), armDisarm=Mock(), enableApiControl=Mock())
        fake_airsim = SimpleNamespace(MultirotorClient=Mock(return_value=client))
        # Deadline established at 0; one retry at 0; second False reaches 61.
        # Exhausting this finite clock also makes an accidental infinite retry
        # fail promptly, without a real wait or socket connection.
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"airsim": fake_airsim}), \
                patch.object(runtime.socket, "create_connection", return_value=MagicMock()) as socket_mock, \
                patch.object(runtime.subprocess, "Popen") as launch_mock, \
                patch.object(runtime.time, "perf_counter", side_effect=[0., 0., 61.]), \
                patch.object(runtime.time, "sleep") as sleep_mock:
            with self.assertRaisesRegex(TimeoutError, "AirSim ping deadline exceeded"):
                runtime.connect(CFG, Path(directory), launch=False)
        socket_mock.assert_called_once()
        launch_mock.assert_not_called()
        fake_airsim.MultirotorClient.assert_called_once()
        self.assertEqual(client.ping.call_count, 2)
        sleep_mock.assert_called_once_with(.2)
        client.armDisarm.assert_not_called()
        client.enableApiControl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
