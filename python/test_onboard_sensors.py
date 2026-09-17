import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from onboard_sensors import OnboardSensorMonitor


def vector(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x_val=x, y_val=y, z_val=z)


def quaternion(w=1.0, x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(w_val=w, x_val=x, y_val=y, z_val=z)


def sensor_client(timestamp=100):
    client = Mock()
    client.getImuData.return_value = SimpleNamespace(
        time_stamp=timestamp, orientation=quaternion(), angular_velocity=vector(),
        linear_acceleration=vector(z=-9.81))
    client.getBarometerData.return_value = SimpleNamespace(
        time_stamp=timestamp, altitude=121.0, pressure=99800.0, qnh=1013.25)
    client.getGpsData.return_value = SimpleNamespace(
        time_stamp=timestamp, is_valid=True,
        gnss=SimpleNamespace(geo_point=SimpleNamespace(latitude=47.0, longitude=-122.0,
                                                       altitude=121.0),
                             velocity=vector(), fix_type=3, eph=0.1, epv=0.1))
    client.getMagnetometerData.return_value = SimpleNamespace(
        time_stamp=timestamp, magnetic_field_body=vector(0.2, 0.0, 0.4))
    client.getDistanceSensorData.return_value = SimpleNamespace(
        time_stamp=timestamp, distance=3.0, min_distance=0.2, max_distance=20.0)
    return client


class OnboardSensorMonitorTests(unittest.TestCase):
    def test_settings_explicitly_define_both_sensor_suites(self):
        settings = json.loads((Path(__file__).parents[1] / "configs" / "settings.json")
                              .read_text(encoding="utf-8"))
        expected = {"Imu": 2, "Barometer": 1, "Gps": 3, "Magnetometer": 4,
                    "DistanceFront": 5, "DistanceDown": 5}
        for vehicle in ("DroneA", "DroneB"):
            sensors = settings["Vehicles"][vehicle]["Sensors"]
            self.assertEqual({name: item["SensorType"] for name, item in sensors.items()},
                             expected)
            self.assertTrue(all(item["Enabled"] for item in sensors.values()))

    def test_reads_named_sensors_and_marks_first_snapshot_fresh(self):
        client = sensor_client()
        monitor = OnboardSensorMonitor(client, clock=lambda: 1.0)
        snapshot = monitor.read_vehicle("DroneA")
        self.assertEqual(set(snapshot["sensors"]), {
            "imu", "barometer", "gps", "magnetometer",
            "distance_front", "distance_down"})
        self.assertTrue(all(item["valid"] and item["fresh"]
                            for item in snapshot["sensors"].values()))
        client.getImuData.assert_called_once_with("Imu", vehicle_name="DroneA")
        self.assertEqual(client.getDistanceSensorData.call_count, 2)

    def test_unchanged_timestamp_becomes_stale(self):
        now = [1.0]
        monitor = OnboardSensorMonitor(sensor_client(), max_stale_s=2.5,
                                       clock=lambda: now[0])
        monitor.read_vehicle("DroneA")
        now[0] = 4.0
        snapshot = monitor.read_vehicle("DroneA")
        self.assertTrue(all(not item["fresh"]
                            for item in snapshot["sensors"].values()))

    def test_bad_sensor_is_recorded_without_aborting_snapshot(self):
        client = sensor_client()
        client.getBarometerData.side_effect = RuntimeError("offline")
        monitor = OnboardSensorMonitor(client, clock=lambda: 1.0)
        snapshot = monitor.read_vehicle("DroneB")
        self.assertFalse(snapshot["sensors"]["barometer"]["valid"])
        self.assertIn("RuntimeError: offline", snapshot["sensors"]["barometer"]["error"])
        self.assertTrue(snapshot["sensors"]["imu"]["valid"])
        self.assertEqual(monitor.summary()["DroneB"]["barometer"]["errors"], 1)

    def test_nonfinite_imu_is_invalid(self):
        client = sensor_client()
        client.getImuData.return_value.linear_acceleration = vector(float("nan"), 0, 0)
        snapshot = OnboardSensorMonitor(client, clock=lambda: 1.0).read_vehicle("DroneA")
        self.assertFalse(snapshot["sensors"]["imu"]["valid"])

    def test_distance_endpoint_tolerance_is_explicitly_saturated(self):
        client = sensor_client()
        client.getDistanceSensorData.return_value.distance = 20.25
        snapshot = OnboardSensorMonitor(client, clock=lambda: 1.0).read_vehicle("DroneA")
        front = snapshot["sensors"]["distance_front"]
        self.assertTrue(front["valid"])
        self.assertTrue(front["data"]["saturated"])
        self.assertFalse(front["data"]["in_range"])

    def test_implausible_distance_remains_invalid(self):
        client = sensor_client()
        client.getDistanceSensorData.return_value.distance = 25.0
        snapshot = OnboardSensorMonitor(client, clock=lambda: 1.0).read_vehicle("DroneA")
        self.assertFalse(snapshot["sensors"]["distance_front"]["valid"])

    def test_below_minimum_distance_is_valid_but_flagged(self):
        client = sensor_client()
        client.getDistanceSensorData.return_value.distance = 0.1
        snapshot = OnboardSensorMonitor(client, clock=lambda: 1.0).read_vehicle("DroneA")
        front = snapshot["sensors"]["distance_front"]
        self.assertTrue(front["valid"])
        self.assertTrue(front["data"]["below_minimum"])
        self.assertFalse(front["data"]["in_range"])

    def test_timestamp_regression_cannot_refresh_watchdog(self):
        client = sensor_client(100)
        now = [1.0]
        monitor = OnboardSensorMonitor(client, clock=lambda: now[0])
        monitor.read_vehicle("DroneA")
        now[0] = 2.0
        client.getImuData.return_value.time_stamp = 99
        item = monitor.read_vehicle("DroneA")["sensors"]["imu"]
        self.assertFalse(item["fresh"])
        self.assertEqual(item["timestamp_status"], "regressed")
        now[0] = 4.0
        client.getImuData.return_value.time_stamp = 100
        self.assertFalse(monitor.read_vehicle("DroneA")["sensors"]["imu"]["fresh"])

    def test_no_hit_noise_below_maximum_is_not_usable_range(self):
        client = sensor_client()
        client.getDistanceSensorData.return_value.distance = 19.8
        item = OnboardSensorMonitor(client).read_vehicle("DroneA")["sensors"]["distance_front"]
        self.assertTrue(item["valid"])
        self.assertFalse(item["data"]["in_range"])
        self.assertEqual(item["data"]["range_status"], "no_hit_or_near_limit")

    def test_negative_range_is_preserved_and_invalid(self):
        client = sensor_client()
        client.getDistanceSensorData.return_value.distance = -.1
        item = OnboardSensorMonitor(client).read_vehicle("DroneA")["sensors"]["distance_down"]
        self.assertFalse(item["valid"])
        self.assertEqual(item["data"]["distance_m"], -.1)
        self.assertFalse(item["data"]["in_range"])

    def test_acceptance_rejects_a_frozen_clock(self):
        from check_onboard_sensors import assess_sequence
        now = [0.0]
        monitor = OnboardSensorMonitor(sensor_client(), clock=lambda: now[0])
        samples = [monitor.read_all(["DroneA"])]
        now[0] = 4.0
        samples.append(monitor.read_all(["DroneA"]))
        failed, _ = assess_sequence(samples, ["DroneA"], 2.5)
        self.assertEqual(len(failed), 6)
        self.assertIn("no_timestamp_progress_observed", failed[0]["reasons"])

    def test_acceptance_marks_negative_measurements_limited_not_valid(self):
        from check_onboard_sensors import assess_sequence
        now = [0.0]
        client = sensor_client(100)
        client.getDistanceSensorData.return_value.distance = -.1
        monitor = OnboardSensorMonitor(client, clock=lambda: now[0])
        samples = [monitor.read_all(["DroneA"])]
        now[0] = 4.0
        for method in (client.getImuData, client.getBarometerData, client.getGpsData,
                       client.getMagnetometerData, client.getDistanceSensorData):
            method.return_value.time_stamp = 200
        samples.append(monitor.read_all(["DroneA"]))
        failed, summary = assess_sequence(samples, ["DroneA"], 2.5)
        self.assertFalse(failed)
        self.assertEqual(summary["DroneA"]["distance_down"]["status"], "LIMITED")
        self.assertEqual(summary["DroneA"]["distance_down"]["valid"], 0)


if __name__ == "__main__":
    unittest.main()
