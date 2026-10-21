"""Single-aircraft balloon stand-off experiment in the inspected Blocks scene.

Only DroneA receives API control. DroneB is a stationary bystander, not a teammate.
Both policies start at the same observation point and use the same terminal camera
criterion. The coordinate policy is a fixed-waypoint baseline; the visual policy
uses RGB candidate error and advisory depth to guide a bounded approach.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime

import airsim
import cv2
import numpy as np

from balloon_detector import annotate
from perception_state import PerceptionTracker, TrackingState
from setup_scene import ROOT, vector
from target_provider import CameraTargetProvider
from visual_servo import command, stop_forward_at_range


CONFIG_PATH = ROOT / "configs" / "single_drone_experiment.json"


def load_config(path=CONFIG_PATH):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["vehicle"] != "DroneA" or cfg["target"] != "BalloonTarget":
        raise ValueError("This inspected experiment requires DroneA and BalloonTarget")
    for key in ("target_world_ned_m", "observation_world_ned_m",
                "coordinate_endpoint_world_ned_m", "return_world_ned_m",
                "workspace_min_ned_m", "workspace_max_ned_m"):
        value = np.asarray(cfg[key], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
    low = np.asarray(cfg["workspace_min_ned_m"])
    high = np.asarray(cfg["workspace_max_ned_m"])
    if not np.all(low < high):
        raise ValueError("Invalid workspace")
    for key in ("observation_world_ned_m", "coordinate_endpoint_world_ned_m"):
        p = np.asarray(cfg[key])
        if not np.all((low <= p) & (p <= high)):
            raise ValueError(f"{key} outside workspace")
    v, s = cfg["visual"], cfg["safety"]
    if not (1 <= v["confirm_frames"] <= 10 and 1 <= v["lost_frames"] <= 20
            and 1 <= v["stable_frames"] <= 10 and 0 < v["frame_max_age_s"] <= 1
            and 1.5 <= v["stop_surface_range_m"] <= 4
            and 0 < v["alignment_error_max"] <= 0.2
            and 10 <= v["max_duration_s"] <= 180
            and 0 < v["max_altitude_setpoint_change_m"] <= 0.5):
        raise ValueError("Unsafe visual settings")
    if not (0 < s["coordinate_speed_m_s"] <= 0.8
            and 0 < s["visual_command_duration_s"] <= 0.5
            and 0 < s["bystander_max_drift_m"] <= 0.5
            and 0 < s["bystander_max_speed_m_s"] <= 0.2):
        raise ValueError("Unsafe motion settings")
    cap = s.get("approach_speed_cap_m_s")
    if cap is not None and (not isinstance(cap, (int, float)) or not 0.05 <= cap <= 0.8
                            or s["coordinate_speed_m_s"] > cap):
        raise ValueError("Invalid common approach speed cap")
    return cfg


def limit_velocity(velocity, cap):
    values = np.asarray(velocity, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Velocity must be three finite values")
    if cap is None:
        return values
    speed = float(np.linalg.norm(values))
    return values * min(1.0, cap / max(speed, 1e-12))


def landing_ready(position, speed, ground_z, pad, ground_actor):
    """Ground actor plus stable return-pad geometry; landed flag may lag disarm."""
    p, goal = np.asarray(position), np.asarray(pad)
    return (np.isfinite(p).all() and math.isfinite(speed) and math.isfinite(ground_z)
            and ground_actor.startswith("Ground") and speed < 0.08
            and np.linalg.norm(p[:2] - goal[:2]) < 0.4
            and 0.65 < p[2] < 1.05 and abs(p[2] - ground_z) < 0.08)


def terminal_success(rows, visual):
    """Common stand-off criterion, independent of which policy moved the drone."""
    recent = rows[-visual["stable_frames"]:]
    return len(recent) == visual["stable_frames"] and all(
        row["tracking_state"] == "DETECTED" and row["detected"]
        and not row["truncated"] and row["frame_age_s"] <= visual["frame_max_age_s"]
        and row["depth_m"] is not None
        and 0 < row["depth_m"] <= visual["stop_surface_range_m"]
        and row["error_xy"] is not None
        and max(abs(x) for x in row["error_xy"]) <= visual["alignment_error_max"]
        for row in recent)


def summarize(rows, path_m, duration_s, visual):
    valid = [r for r in rows if r["detected"] and not r["truncated"]]
    errors = [max(abs(x) for x in r["error_xy"]) for r in valid if r["error_xy"]]
    terminal_rows = rows[-visual["stable_frames"]:]
    terminal_errors = [max(abs(x) for x in r["error_xy"])
                       for r in terminal_rows if r["error_xy"] is not None]
    ranges = [r["depth_m"] for r in rows if r["depth_m"] is not None]
    return {"task_success": terminal_success(rows, visual),
            "observation_count": len(rows),
            "valid_detection_fraction": len(valid) / len(rows) if rows else None,
            "mean_max_axis_error": float(np.mean(errors)) if errors else None,
            "terminal_mean_max_axis_error": float(np.mean(terminal_errors))
                if len(terminal_errors) == visual["stable_frames"] else None,
            "last_surface_range_m": ranges[-1] if ranges else None,
            "approach_path_m": path_m,
            "approach_duration_s": duration_s}


class SingleDroneExperiment:
    def __init__(self, policy, client=None, config_path=CONFIG_PATH, record_prefix=None):
        if policy not in ("coordinate", "visual"):
            raise ValueError("Policy must be coordinate or visual")
        self.cfg = load_config(config_path)
        self.policy = policy
        self.client = client or airsim.MultirotorClient(timeout_value=10)
        self.camera = CameraTargetProvider(self.client)
        self.vehicle = self.cfg["vehicle"]
        prefix = record_prefix or f"single_{policy}"
        self.folder = ROOT / "logs" / datetime.now().strftime(
            f"{prefix}_%Y%m%d_%H%M%S_%f")
        self.folder.mkdir(parents=True)
        self.report = {
            "status": "RUNNING", "policy": policy, "vehicle_commanded": self.vehicle,
            "bystander": "DroneB", "target": self.cfg["target"],
            "configuration": self.cfg,
            "truth_access": {
                "target_pose": "preflight_only",
                "vehicle_pose": "staging_return_and_safety; coordinate_policy_approach",
                "rgb_camera": "observation_and_visual_policy_guidance",
                "depth_camera": "sequential_rgb_depth_advisory_range",
                "collision": "safety_abort_only"
            },
            "source_sha256": {name: hashlib.sha256((ROOT / "python" / name).read_bytes()).hexdigest()
                              for name in ("single_drone_experiment.py", "visual_servo.py",
                                           "target_provider.py", "balloon_detector.py")},
            "observations": []}
        self.phase = "PREFLIGHT"
        self.bystander_start = None
        self.max_bystander_drift_m = 0.0
        self.collision_stamp = None
        self.last_position = None
        self.path_m = 0.0
        self.armed = False
        self.ground_contact = None
        self.tracker = PerceptionTracker(self.cfg["visual"]["confirm_frames"],
                                         self.cfg["visual"]["lost_frames"])

    def state(self, check_workspace=False):
        p = np.asarray(vector(self.client.simGetObjectPose(self.vehicle).position), dtype=float)
        s = self.client.getMultirotorState(vehicle_name=self.vehicle)
        speed = float(np.linalg.norm(vector(s.kinematics_estimated.linear_velocity)))
        b = np.asarray(vector(self.client.simGetObjectPose("DroneB").position), dtype=float)
        bs = self.client.getMultirotorState(vehicle_name="DroneB")
        b_speed = float(np.linalg.norm(vector(bs.kinematics_estimated.linear_velocity)))
        if not np.isfinite(p).all() or not np.isfinite(b).all():
            raise RuntimeError("Non-finite vehicle pose")
        if self.bystander_start is not None:
            drift = float(np.linalg.norm(b - self.bystander_start))
            self.max_bystander_drift_m = max(self.max_bystander_drift_m, drift)
            if drift > self.cfg["safety"]["bystander_max_drift_m"]:
                raise RuntimeError("DroneB moved during single-drone experiment")
            if b_speed > self.cfg["safety"]["bystander_max_speed_m_s"]:
                raise RuntimeError("DroneB is moving")
        if check_workspace:
            low = np.asarray(self.cfg["workspace_min_ned_m"])
            high = np.asarray(self.cfg["workspace_max_ned_m"])
            if not np.all((low <= p) & (p <= high)):
                raise RuntimeError(f"DroneA left inspected workspace: {p.tolist()}")
        collision = self.client.simGetCollisionInfo(vehicle_name=self.vehicle)
        if self.collision_stamp is not None and collision.time_stamp > self.collision_stamp:
            self.collision_stamp = collision.time_stamp
            if self.phase == "LANDING" and collision.object_name.startswith("Ground"):
                self.ground_contact = collision.object_name
            if not (self.phase in ("TAKEOFF", "LANDING")
                    and collision.object_name.startswith("Ground")):
                raise RuntimeError(f"Unexpected collision with {collision.object_name}")
        return p, speed, s

    def move_to(self, goal, speed=0.7, timeout=50):
        goal = np.asarray(goal, dtype=float)
        deadline = time.monotonic() + timeout
        stable = None
        while time.monotonic() < deadline:
            p, actual_speed, _ = self.state()
            if self.phase == "APPROACH" and self.last_position is not None:
                self.path_m += float(np.linalg.norm(p - self.last_position))
                self.last_position = p
            delta = goal - p
            distance = float(np.linalg.norm(delta))
            velocity = delta * min(0.8, speed / max(distance, 0.001))
            if self.phase == "APPROACH":
                velocity = limit_velocity(velocity, self.cfg["safety"].get("approach_speed_cap_m_s"))
            self.client.moveByVelocityAsync(*velocity, duration=0.7,
                yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle).join()
            if self.phase == "APPROACH" and self.policy == "coordinate":
                self.sample()
            ready = distance < 0.2 and actual_speed < 0.2
            stable = (stable or time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable >= 0.7:
                self.client.hoverAsync(vehicle_name=self.vehicle).join()
                return
        raise TimeoutError(f"Could not settle in {self.phase}")

    def sample(self):
        started = time.monotonic()
        observation, image, detection = self.camera.observe_with_depth(self.vehicle)
        age = time.monotonic() - started
        track = self.tracker.update(observation.detected and not observation.truncated)
        p, _, _ = self.state(check_workspace=True)
        if self.last_position is not None:
            self.path_m += float(np.linalg.norm(p - self.last_position))
        self.last_position = p
        row = {"time": datetime.now().astimezone().isoformat(),
               "world_ned_m": p.tolist(), "detected": observation.detected,
               "truncated": observation.truncated, "tracking_state": track.state.value,
               "candidate_count": observation.candidate_count,
               "shape_score_not_probability": observation.confidence,
               "error_xy": observation.normalised_error_xy,
               "depth_m": observation.depth_m, "frame_age_s": age}
        self.report["observations"].append(row)
        if image is not None:
            raw_path = self.folder / f"raw_frame_{len(self.report['observations']):04d}.png"
            if not cv2.imwrite(str(raw_path), image):
                raise RuntimeError("Could not save unannotated RGB observation")
            row["raw_image"] = str(raw_path)
            row["raw_image_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            cv2.imwrite(str(self.folder / f"frame_{len(self.report['observations']):04d}.png"),
                        annotate(image, detection))
        return row

    def approach(self):
        self.phase = "APPROACH"
        rows = self.report["observations"]
        self.last_position = self.state(check_workspace=True)[0]
        self.path_m = 0.0
        start = time.monotonic()
        if self.policy == "coordinate":
            self.move_to(self.cfg["coordinate_endpoint_world_ned_m"],
                         speed=self.cfg["safety"]["coordinate_speed_m_s"], timeout=45)
            for _ in range(self.cfg["visual"]["stable_frames"] + self.cfg["visual"]["confirm_frames"]):
                self.sample()
                time.sleep(0.2)
        else:
            z_initial = self.state()[0][2]
            z_setpoint = z_initial
            missed = 0
            deadline = start + self.cfg["visual"]["max_duration_s"]
            while time.monotonic() < deadline:
                row = self.sample()
                valid = (row["tracking_state"] == TrackingState.DETECTED.value
                         and row["detected"] and not row["truncated"]
                         and row["frame_age_s"] <= self.cfg["visual"]["frame_max_age_s"]
                         and row["depth_m"] is not None)
                missed = 0 if valid else missed + 1
                if missed >= self.cfg["visual"]["lost_frames"]:
                    raise RuntimeError("Target unavailable for too many observations")
                if terminal_success(rows, self.cfg["visual"]):
                    break
                cmd = command(valid, row["error_xy"], row["depth_m"], row["truncated"],
                              stop_depth=0.35)
                velocity = cmd.velocity
                reached = valid and row["depth_m"] <= self.cfg["visual"]["stop_surface_range_m"]
                velocity = stop_forward_at_range(velocity, reached)
                z_setpoint += velocity[2] * 0.3
                bound = self.cfg["visual"]["max_altitude_setpoint_change_m"]
                z_setpoint = max(z_initial - bound, min(z_initial + bound, z_setpoint))
                row["control_reason"] = cmd.reason
                current_z = row["world_ned_m"][2]
                vz = max(-0.15, min(0.15, 0.8 * (z_setpoint - current_z)))
                applied = limit_velocity([velocity[0], velocity[1], vz],
                                         self.cfg["safety"].get("approach_speed_cap_m_s"))
                row["command_velocity_ned_m_s"] = applied.tolist()
                row["altitude_setpoint_ned_m"] = z_setpoint
                # moveByVelocityZAsync interprets Z in the vehicle's local frame;
                # the workspace and setpoint above use world NED. Use measured
                # world-frame vertical velocity feedback, as the route driver does.
                self.client.moveByVelocityAsync(*applied,
                    duration=self.cfg["safety"]["visual_command_duration_s"],
                    yaw_mode=airsim.YawMode(False, 0), vehicle_name=self.vehicle).join()
            else:
                raise TimeoutError("Visual approach timeout")
        self.client.hoverAsync(vehicle_name=self.vehicle).join()
        self.report["metrics"] = summarize(rows, self.path_m, time.monotonic() - start,
                                            self.cfg["visual"])
        self.report["task_success"] = self.report["metrics"]["task_success"]

    def land(self):
        self.phase = "RETURN"
        self.move_to(self.cfg["return_world_ned_m"], timeout=50)
        self.phase = "LANDING"
        self.client.landAsync(timeout_sec=45, vehicle_name=self.vehicle)
        deadline = time.monotonic() + 45
        stable = None
        self.report["landing_trace"] = []
        while time.monotonic() < deadline:
            p, speed, s = self.state()
            pad = np.asarray(self.cfg["return_world_ned_m"])
            contact = self.client.simGetCollisionInfo(vehicle_name=self.vehicle)
            ready = landing_ready(p, speed, self.bystander_start[2], pad, contact.object_name)
            self.report["landing_trace"].append({"time": datetime.now().astimezone().isoformat(),
                "world_ned_m": p.tolist(), "speed_m_s": speed,
                "ground_actor": contact.object_name, "collision_timestamp": contact.time_stamp,
                "landed_flag": s.landed_state == airsim.LandedState.Landed,
                "physically_grounded": bool(ready)})
            stable = (stable or time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable >= 2:
                self.ground_contact = contact.object_name
                self.report["landing_evidence"] = {"world_ned_m": p.tolist(),
                    "speed_m_s": speed, "ground_actor": self.ground_contact,
                    "landed_flag_before_disarm": s.landed_state == airsim.LandedState.Landed,
                    "stable_ground_seconds": time.monotonic() - stable,
                    "ground_reference_z_m": float(self.bystander_start[2])}
                self.client.cancelLastTask(vehicle_name=self.vehicle)
                if not self.client.armDisarm(False, vehicle_name=self.vehicle):
                    raise RuntimeError("Could not disarm DroneA")
                self.armed = False
                self.client.enableApiControl(False, vehicle_name=self.vehicle)
                if self.client.isApiControlEnabled(vehicle_name=self.vehicle):
                    raise RuntimeError("Could not release DroneA API control")
                settled_deadline = time.monotonic() + 5
                while time.monotonic() < settled_deadline:
                    _, settled_speed, settled_state = self.state()
                    if settled_state.landed_state == airsim.LandedState.Landed and settled_speed < 0.08:
                        self.report["landing_evidence"]["landed_flag"] = True
                        self.report["landing_evidence"]["api_control_released"] = True
                        return
                    time.sleep(0.2)
                raise TimeoutError("DroneA disarmed and control released, but landed state unconfirmed")
            time.sleep(0.25)
        raise TimeoutError("Landing not verified; simulation will pause with DroneA armed")

    def run(self):
        try:
            if not {"DroneA", "DroneB"}.issubset(self.client.listVehicles()):
                raise RuntimeError("Configured Blocks scene requires both vehicles")
            target = np.asarray(vector(self.client.simGetObjectPose(self.cfg["target"]).position))
            if not np.allclose(target, self.cfg["target_world_ned_m"], atol=0.02):
                raise RuntimeError("Run setup_scene.py first; balloon pose mismatch")
            if self.client.isApiControlEnabled(vehicle_name="DroneB"):
                raise RuntimeError("DroneB API control must be released")
            p, speed, _ = self.state()
            b, b_speed, _ = self._bystander_preflight()
            if not (0.65 < p[2] < 1.1 and speed < 0.08
                    and 0.65 < b[2] < 1.1 and b_speed < 0.08):
                raise RuntimeError("Both vehicles must be stationary on the ground")
            if self.client.isApiControlEnabled(vehicle_name=self.vehicle):
                raise RuntimeError("DroneA API control must be released")
            self.bystander_start = b
            self.report["bystander_start_world_ned_m"] = b.tolist()
            self.collision_stamp = self.client.simGetCollisionInfo(vehicle_name=self.vehicle).time_stamp
            self.client.simPause(False)
            self.client.enableApiControl(True, vehicle_name=self.vehicle)
            if not self.client.armDisarm(True, vehicle_name=self.vehicle):
                raise RuntimeError("Could not arm DroneA")
            self.armed = True
            self.phase = "TAKEOFF"
            self.move_to([p[0], p[1], -3.0], timeout=35)
            self.phase = "STAGING"
            self.move_to(self.cfg["observation_world_ned_m"], timeout=40)
            self.approach()
            self.land()
            self.report["status"] = "PASS"
        except BaseException as exc:
            self.report.update(status="FAIL", error=str(exc), error_type=type(exc).__name__,
                               failed_phase=self.phase, manual_recovery_required=self.armed)
            raise
        finally:
            try:
                self.client.simPause(True)
                self.report["paused"] = True
            except Exception as exc:
                self.report.update(status="FAIL", paused=False, pause_error=str(exc),
                                   manual_recovery_required=self.armed)
            self.report["finished_at"] = datetime.now().astimezone().isoformat()
            self.report["bystander_max_drift_m"] = self.max_bystander_drift_m
            (self.folder / "report.json").write_text(
                json.dumps(self.report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(self.folder)

    def _bystander_preflight(self):
        p = np.asarray(vector(self.client.simGetObjectPose("DroneB").position), dtype=float)
        s = self.client.getMultirotorState(vehicle_name="DroneB")
        return p, float(np.linalg.norm(vector(s.kinematics_estimated.linear_velocity))), s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("coordinate", "visual"), required=True)
    args = parser.parse_args()
    SingleDroneExperiment(args.policy).run()
