"""Typed, advisory snapshots of AirSim onboard sensors."""
import math
import time


SENSORS = (
    ("imu", "getImuData", "Imu"),
    ("barometer", "getBarometerData", "Barometer"),
    ("gps", "getGpsData", "Gps"),
    ("magnetometer", "getMagnetometerData", "Magnetometer"),
    ("distance_front", "getDistanceSensorData", "DistanceFront"),
    ("distance_down", "getDistanceSensorData", "DistanceDown"),
)


def _vector(value):
    return [float(value.x_val), float(value.y_val), float(value.z_val)]


def _quaternion(value):
    return [float(value.w_val), float(value.x_val),
            float(value.y_val), float(value.z_val)]


def _finite(values):
    return all(math.isfinite(float(value)) for value in values)


def _payload(kind, value):
    if kind == "imu":
        orientation = _quaternion(value.orientation)
        angular_velocity = _vector(value.angular_velocity)
        linear_acceleration = _vector(value.linear_acceleration)
        data = {"orientation_wxyz": orientation,
                "angular_velocity_rad_s": angular_velocity,
                "linear_acceleration_m_s2": linear_acceleration}
        return data, _finite(orientation + angular_velocity + linear_acceleration)
    if kind == "barometer":
        data = {"altitude_m": float(value.altitude),
                "pressure_pa": float(value.pressure), "qnh_hpa": float(value.qnh)}
        return data, _finite(data.values()) and data["pressure_pa"] > 0
    if kind == "gps":
        gnss = value.gnss
        position = [float(gnss.geo_point.latitude), float(gnss.geo_point.longitude),
                    float(gnss.geo_point.altitude)]
        velocity = _vector(gnss.velocity)
        data = {"latitude_deg": position[0], "longitude_deg": position[1],
                "altitude_m": position[2], "velocity_ned_m_s": velocity,
                "fix_type": int(gnss.fix_type), "eph_m": float(gnss.eph),
                "epv_m": float(gnss.epv), "is_valid": bool(value.is_valid)}
        valid = (data["is_valid"] and data["fix_type"] >= 2
                 and _finite(position + velocity + [data["eph_m"], data["epv_m"]]))
        return data, valid
    if kind == "magnetometer":
        field = _vector(value.magnetic_field_body)
        return {"magnetic_field_body": field}, _finite(field)
    if kind.startswith("distance_"):
        data = {"distance_m": float(value.distance),
                "min_distance_m": float(value.min_distance),
                "max_distance_m": float(value.max_distance)}
        data["saturated"] = data["distance_m"] >= data["max_distance_m"]
        data["below_minimum"] = data["distance_m"] < data["min_distance_m"]
        # This source adds Gaussian noise (sigma 0.2 m) even to no-hit rays.
        # A conservative 3-sigma margin avoids treating most noisy no-hit
        # readings just below MaxDistance as an obstacle. It is not proof of a hit.
        data["near_maximum_margin_m"] = 0.6
        data["near_maximum"] = data["distance_m"] >= data["max_distance_m"] - 0.6
        data["in_range"] = (_finite([data["distance_m"]]) and
                            not data["near_maximum"] and not data["below_minimum"])
        data["range_status"] = ("invalid_nonfinite" if not _finite([data["distance_m"]]) else
                                "negative_invalid" if data["distance_m"] < 0 else
                                "below_minimum" if data["below_minimum"] else
                                "no_hit_or_near_limit" if data["near_maximum"] else "in_range")
        # Preserve physically invalid negative readings, never clamp to zero.
        valid = (_finite([data["distance_m"], data["min_distance_m"], data["max_distance_m"]])
                 and data["min_distance_m"] >= 0
                 and data["max_distance_m"] > data["min_distance_m"]
                 and 0 <= data["distance_m"]
                 <= data["max_distance_m"] + 1.0)
        return data, valid
    raise ValueError(f"Unknown sensor kind: {kind}")


class OnboardSensorMonitor:
    """Read sensors without making advisory health terminal to a mission."""

    def __init__(self, client, max_stale_s=2.5, clock=time.monotonic):
        if not math.isfinite(max_stale_s) or max_stale_s <= 0:
            raise ValueError("max_stale_s must be positive")
        self.client = client
        self.max_stale_s = float(max_stale_s)
        self.clock = clock
        self._timestamps = {}
        self._changed_at = {}
        self._summary = {}

    def _fresh(self, vehicle, kind, timestamp_ns, now):
        key = (vehicle, kind)
        if timestamp_ns <= 0:
            return False
        previous = self._timestamps.get(key)
        if previous is not None and timestamp_ns < previous:
            # Do not refresh the watchdog when a clock jumps backwards.
            return False
        if previous != timestamp_ns:
            self._timestamps[key] = timestamp_ns
            self._changed_at[key] = now
        return now - self._changed_at.get(key, now) <= self.max_stale_s

    def _count(self, vehicle, kind, item):
        counters = self._summary.setdefault(vehicle, {}).setdefault(
            kind, {"samples": 0, "valid": 0, "fresh": 0, "errors": 0})
        counters["samples"] += 1
        counters["valid"] += int(item["valid"])
        counters["fresh"] += int(item["fresh"])
        counters["errors"] += int("error" in item)

    def read_vehicle(self, vehicle):
        snapshot = {"sampled_monotonic_s": self.clock(), "role": "advisory_only",
                    "sensors": {}}
        for kind, method_name, sensor_name in SENSORS:
            started = self.clock()
            try:
                method = getattr(self.client, method_name)
                value = method(sensor_name, vehicle_name=vehicle)
                ended = self.clock()
                timestamp_ns = int(value.time_stamp)
                data, valid = _payload(kind, value)
                previous = self._timestamps.get((vehicle, kind))
                timestamp_status = ("invalid" if timestamp_ns <= 0 else
                                    "first_observation" if previous is None else
                                    "advanced" if timestamp_ns > previous else
                                    "regressed" if timestamp_ns < previous else "unchanged")
                item = {"timestamp_ns": timestamp_ns,
                        "timestamp_status": timestamp_status,
                        "fresh": self._fresh(vehicle, kind, timestamp_ns, ended),
                        "freshness_scope": "timestamp_progress_watchdog_not_absolute_age",
                        "host_request_monotonic_s": started,
                        "host_response_monotonic_s": ended,
                        "rpc_latency_ms": (ended - started) * 1000,
                        "valid": bool(valid), "data": data}
            except Exception as exc:
                item = {"timestamp_ns": None, "fresh": False, "valid": False,
                        "error": f"{type(exc).__name__}: {exc}"}
            snapshot["sensors"][kind] = item
            self._count(vehicle, kind, item)
        return snapshot

    def read_all(self, vehicles):
        return {vehicle: self.read_vehicle(vehicle) for vehicle in vehicles}

    def summary(self):
        return {vehicle: {kind: dict(values) for kind, values in sensors.items()}
                for vehicle, sensors in self._summary.items()}
