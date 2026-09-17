"""Local Blocks balloon contact experiment, with first-attempt/fallback demos."""
import argparse
import hashlib
import json
import logging
import math
import time
from datetime import datetime

import airsim
import cv2
import numpy as np

from setup_scene import ROOT, vector, capture_overview
from hit_judge import classify_collision
from coordinator import MissionCoordinator
from balloon_detector import annotate
from mission_config import MissionConfig
from target_provider import CameraTargetProvider
from perception_state import PerceptionTracker
from guidance_advisor import advise
from onboard_sensors import OnboardSensorMonitor


class Experiment:
    def __init__(self, scenario, miss_offset_y=None, target_offset=None):
        self.scenario = scenario
        self.config = MissionConfig.load(ROOT / "configs" / "mission.json")
        self.miss_offset_y = (self.config.deliberate_miss_offset_y_m
                              if miss_offset_y is None else float(miss_offset_y))
        if abs(self.miss_offset_y) < 1.5:
            raise ValueError("Runtime miss offset must remain outside the fixed balloon body")
        self.target_offset = np.array(target_offset if target_offset is not None else [0.0, 0.0, 0.0], dtype=float)
        if self.target_offset.shape != (3,) or not np.isfinite(self.target_offset).all():
            raise ValueError("Target offset must be three finite values")
        if abs(self.target_offset[0]) > 1.0 or abs(self.target_offset[1]) > 1.0 or abs(self.target_offset[2]) > 0.5:
            raise ValueError("Target offset exceeds the inspected Blocks test region")
        self.target_position = self.config.target_world_ned_m + self.target_offset
        self.approach_position = self.config.approach_world_ned_m + self.target_offset
        self.folder = ROOT / "logs" / datetime.now().strftime(f"mission_{scenario}_%Y%m%d_%H%M%S_%f")
        self.folder.mkdir(parents=True)
        self.client = airsim.MultirotorClient(timeout_value=10)
        self.camera_provider = CameraTargetProvider(self.client)
        self.onboard_sensors = OnboardSensorMonitor(self.client)
        self.names = tuple(self.config.vehicles)
        self.perception_trackers = {name: PerceptionTracker(
                                        confirm_frames=self.config.perception_confirm_frames,
                                        lost_frames=self.config.perception_lost_frames)
                                    for name in self.names}
        self.coordinator = MissionCoordinator(*self.names)
        self.phase = "PREFLIGHT"
        self.active = None
        self.target = self.config.target_name
        self.hit = None
        self.timestamps = {}
        evaluation_mode = {
            "a_hit": "coordinate_first_attempt",
            "fallback": "coordinate_fault_injection",
            "visual_fallback": "visual_fault_injection",
        }.get(scenario, "unknown")
        self.report = {"scenario": scenario, "status": "RUNNING", "events": [], "finish": "land",
                       "evaluation": {
                           "mode": evaluation_mode,
                           "policy_id": "a_first_fallback_v1",
                           "decision_source": "scenario_route",
                           "completion_authority": "airsim_new_target_collision",
                       },
                       "truth_access": {
                           "target_pose": {"used_for": ["scene_setup_check", "coordinate_route", "evaluation"]},
                           "vehicle_pose": {"used_for": ["position_control", "separation_safety", "evaluation"]},
                           "rgb_camera": {"used_for": ["target_detection", "visual_guidance"]},
                           "depth_camera": {"used_for": ["relative_distance_advisory"]},
                           "collision_feedback": {"used_for": ["completion_authority"]},
                       },
                       "perception_transitions": [],
                       "perception_policy": {"confirm_frames": self.config.perception_confirm_frames,
                                             "lost_frames": self.config.perception_lost_frames,
                                             "alignment_deadband": self.config.perception_alignment_deadband,
                                             "stop_distance_m": self.config.perception_stop_distance_m,
                                             "role": "advisory_only"},
                       "onboard_sensor_policy": {"sample_period_s": 1.0,
                                                 "max_stale_s": 2.5,
                                                 "role": "advisory_only"}}
        self.report["configuration"] = self.config.to_dict()
        self.report["runtime_overrides"] = {"deliberate_miss_offset_y_m": self.miss_offset_y,
                                            "target_offset_ned_m": self.target_offset.tolist()}
        self.report["source_sha256"] = {name: hashlib.sha256((ROOT / "python" / name).read_bytes()).hexdigest()
            for name in ("run_balloon_mission.py", "mission_config.py", "coordinator.py", "hit_judge.py",
                         "mission_state.py", "perception_state.py", "guidance_advisor.py",
                         "target_provider.py", "balloon_detector.py", "onboard_sensors.py")}
        self.report["collision_position_frame"] = "AirSim vehicle-local NED metres; telemetry positions are world NED"
        self.stream = (self.folder / "telemetry.jsonl").open("w", encoding="utf-8")
        self.log = logging.getLogger(str(self.folder))
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        for handler in (logging.StreamHandler(), logging.FileHandler(self.folder / "mission.log", encoding="utf-8")):
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self.log.addHandler(handler)
        self.min_separation = float("inf")
        self.max_waiting_drift = 0.0
        self.min_miss_distance = float("inf")
        self.holds = {}
        self.ground_contacts = {}
        self.last_vision_s = 0.0
        self.vision_records = []
        self.last_onboard_sensor_s = float("-inf")

    def event(self, name, **data):
        item = dict(event=name, wall_time=datetime.now().astimezone().isoformat(),
                    monotonic_s=time.monotonic(), state=self.state.value, **data)
        self.report["events"].append(item)
        self.log.info("%s %s", name, data)
        return item

    @property
    def state(self):
        return self.coordinator.state

    def change(self, event, **metadata):
        previous = self.state.value
        if event == "start":
            dispatch = self.coordinator.start()
            self.active = dispatch.vehicle
        elif event == "hit":
            self.coordinator.confirm_hit(self.active)
        elif event == "miss":
            self.coordinator.confirm_miss(self.active)
        elif event == "fallback_ready":
            dispatch = self.coordinator.mark_corridor_clear()
            self.active = dispatch.vehicle
        elif event == "timeout":
            self.coordinator.timeout()
        else:
            raise ValueError(f"Unknown coordinator event: {event}")
        self.event(event, previous_state=previous,
                   coordinator_active_vehicle=self.coordinator.active_vehicle,
                   **metadata)

    def observe(self):
        positions, speeds, landed = {}, {}, {}
        for name in self.names:
            positions[name] = np.array(vector(self.client.simGetObjectPose(name).position))
            state = self.client.getMultirotorState(vehicle_name=name)
            speeds[name] = float(np.linalg.norm(vector(state.kinematics_estimated.linear_velocity)))
            landed[name] = state.landed_state == airsim.LandedState.Landed
            if not np.isfinite(positions[name]).all():
                raise RuntimeError(f"Invalid position: {name}")
            collision = self.client.simGetCollisionInfo(vehicle_name=name)
            kind = classify_collision(collision, self.timestamps[name], self.target)
            if kind != "NONE":
                self.timestamps[name] = collision.time_stamp
                self.event("collision", vehicle=name, object=collision.object_name,
                           timestamp=collision.time_stamp, position=vector(collision.position))
                if kind == "TARGET" and name == self.active and self.phase == "CONTACT":
                    self.hit = {"vehicle": name, "object": collision.object_name,
                                "timestamp": collision.time_stamp, "position": vector(collision.position)}
                elif kind == "TARGET" and self.hit is not None and self.state.value == "HIT_CONFIRMED":
                    # Unreal may publish one final contact timestamp while the
                    # already-hit actor is being destroyed. The first collision
                    # remains the authoritative evidence; this cannot score again.
                    self.event("post_hit_target_contact_ignored", vehicle=name,
                               timestamp=collision.time_stamp)
                elif self.phase == "LANDING" and collision.object_name.startswith("Ground"):
                    self.ground_contacts[name] = collision.object_name
                else:
                    raise RuntimeError(f"Unexpected collision: {name} / {collision.object_name}")
        now = time.monotonic()
        onboard_snapshot = None
        if now - self.last_onboard_sensor_s >= 1.0:
            self.last_onboard_sensor_s = now
            onboard_snapshot = self.onboard_sensors.read_all(self.names)
            self.report["onboard_sensor_summary"] = self.onboard_sensors.summary()
        if now - self.last_vision_s >= 1.0:
            self.last_vision_s = now
            self.capture_vision(now)
        separation = float(np.linalg.norm(positions["DroneA"] - positions["DroneB"]))
        self.min_separation = min(self.min_separation, separation)
        if self.phase == "DELIBERATE_MISS":
            self.min_miss_distance = min(self.min_miss_distance,
                float(np.linalg.norm(positions["DroneA"] - self.target_position)))
        if separation < self.config.minimum_drone_separation_m:
            raise RuntimeError(f"Separation below {self.config.minimum_drone_separation_m} m: {separation}")
        for name, target in self.holds.items():
            drift = float(np.linalg.norm(positions[name] - target))
            self.max_waiting_drift = max(self.max_waiting_drift, drift)
            if drift > self.config.maximum_waiting_drift_m:
                raise RuntimeError(f"Waiting vehicle drift: {name}, {drift:.2f} m")
        self.stream.write(json.dumps({"time": time.time(), "phase": self.phase,
                                      "state": self.state.value,
                                      "world_ned_m": {n: p.tolist() for n, p in positions.items()},
                                      "speed_m_s": speeds, "separation_m": separation,
                                      "onboard_sensors": onboard_snapshot}) + "\n")
        self.stream.flush()
        for name in self.names:
            self.client.simPlotStrings([name + " - " + (self.phase if name == self.active else "STANDBY")],
                                       [airsim.Vector3r(*(positions[name] + [0, 0, -0.6]))], 1.2,
                                       [0, 0.8, 1, 1] if name == "DroneA" else [1, 0.8, 0, 1], 0.22)
        return positions, speeds, landed

    def capture_vision(self, now):
        """Record camera evidence as advisory telemetry; collision remains authoritative."""
        frame_record = {"time": datetime.now().astimezone().isoformat(), "phase": self.phase, "vehicles": {}}
        for name in self.names:
            observation, image, result = self.camera_provider.observe_with_depth(name)
            tracking = self.perception_trackers[name].update(observation.detected)
            tracking_fields = {"tracking_state": tracking.state.value,
                               "detection_streak": tracking.detection_streak,
                               "miss_streak": tracking.miss_streak}
            if tracking.changed:
                self.report["perception_transitions"].append({
                    "time": frame_record["time"], "phase": self.phase, "vehicle": name,
                    "from": tracking.previous_state.value, "to": tracking.state.value,
                    "raw_detected": observation.detected})
            if image is None:
                frame_record["vehicles"][name] = dict(error="empty_image", detected=False,
                                                       source=observation.source,
                                                       guidance_advice="HOLD",
                                                       guidance_role="advisory_only",
                                                       **tracking_fields)
                continue
            entry = {"detected": observation.detected,
                     "source": observation.source,
                     "candidate_count": observation.candidate_count,
                     "centre_px": observation.centre_px,
                     "bbox_xywh": observation.bbox_xywh,
                     "confidence_shape_score": observation.confidence,
                     "normalised_error_xy": observation.normalised_error_xy,
                     "truncated": observation.truncated,
                     "depth_m": observation.depth_m,
                     "guidance_advice": advise(tracking.state,
                                                observation.normalised_error_xy,
                                                observation.depth_m,
                                                self.config.perception_alignment_deadband,
                                                self.config.perception_stop_distance_m).value,
                     "guidance_role": "advisory_only",
                     **tracking_fields}
            frame_record["vehicles"][name] = entry
            if self.phase in ("APPROACH", "CONTACT", "DELIBERATE_MISS"):
                path = self.folder / f"vision_{len(self.vision_records):04d}_{name}.png"
                cv2.imwrite(str(path), annotate(image, result))
        self.vision_records.append(frame_record)
        self.report["vision_last"] = frame_record
        self.report["vision_frames"] = len(self.vision_records)
        if self.active and frame_record["vehicles"].get(self.active, {}).get("detected"):
            self.report["vision_active_detected_frames"] = self.report.get("vision_active_detected_frames", 0) + 1

    def drive(self, targets, speed=0.8, timeout=50, hold=1.0, contact=False):
        deadline = time.monotonic() + timeout
        stable = None
        all_targets = dict(self.holds, **targets)
        while time.monotonic() < deadline:
            positions, speeds, _ = self.observe()
            if self.hit is not None and contact:
                for name in self.names:
                    self.client.hoverAsync(vehicle_name=name)
                return True
            for name, target in all_targets.items():
                delta = np.array(target) - positions[name]
                distance = np.linalg.norm(delta)
                velocity = delta * min(0.8, speed / max(distance, 0.001))
                self.client.moveByVelocityAsync(*velocity, duration=1.0,
                                               yaw_mode=airsim.YawMode(False, 0), vehicle_name=name)
            ready = all(np.linalg.norm(positions[n] - t) < 0.2 and speeds[n] < 0.2
                        for n, t in all_targets.items())
            stable = (stable or time.monotonic()) if ready else None
            if not contact and stable is not None and time.monotonic() - stable >= hold:
                return False
            time.sleep(0.2)
        if contact:
            return False
        raise TimeoutError(f"Could not settle in phase {self.phase}")

    def contact(self, name):
        if name != self.coordinator.active_vehicle:
            raise RuntimeError(f"Vehicle has no coordinator dispatch: {name}")
        self.active = name
        self.phase = "APPROACH"
        self.drive({name: self.approach_position}, speed=0.7)
        capture_overview(self.client, self.folder / f"before_{name}.png")
        self.phase = "CONTACT"
        self.event("contact_attempt", vehicle=name)
        result = self.drive({name: self.target_position}, speed=0.25,
                            timeout=self.config.contact_timeout_s, contact=True)
        if result:
            self.change("hit", source="collision_feedback",
                        reason_code="TARGET_COLLISION", evaluation_role="completion")
            self.report["hit_evidence"] = self.hit
            capture_overview(self.client, self.folder / "contact.png")
            # The disappearing balloon is a visual effect AFTER collision evidence.
            for object_name in (self.target, self.target + "_String"):
                if not self.client.simDestroyObject(object_name):
                    raise RuntimeError(f"Could not remove hit target: {object_name}")
            self.event("balloon_removed", cause="confirmed_collision")
            capture_overview(self.client, self.folder / "after_hit.png")
        return result

    def land(self):
        self.holds = {}
        self.active = None
        self.phase = "RETURN"
        self.event("return_to_pads")
        self.drive(self.config.pads_world_ned_m)
        self.phase = "LANDING"
        self.event("landing")
        for name in self.names:
            self.client.landAsync(timeout_sec=45, vehicle_name=name)
        deadline = time.monotonic() + 50
        stable = None
        while time.monotonic() < deadline:
            positions, speeds, landed = self.observe()
            # SimpleFlight's landed flag depends on RC throttle and may remain
            # Flying after landAsync reaches the floor. Require fresh ground
            # contact, the known flat Blocks pad height and stable motion BEFORE
            # disarming. This is specific to the inspected Blocks landing pads.
            ready = all(n in self.ground_contacts and speeds[n] < 0.08
                        and np.linalg.norm(positions[n][:2] - self.config.pads_world_ned_m[n][:2]) < 0.4
                        and 0.65 < positions[n][2] < 1.05 for n in self.names)
            stable = (stable or time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable > 2:
                self.report["landing_evidence"] = {n: {"ground_actor": self.ground_contacts[n],
                    "world_ned_m": positions[n].tolist(), "speed_m_s": speeds[n],
                    "landed_flag_before_disarm": landed[n]} for n in self.names}
                for name in self.names:
                    if not self.client.armDisarm(False, vehicle_name=name):
                        raise RuntimeError(f"Could not disarm landed {name}")
                    self.client.enableApiControl(False, vehicle_name=name)
                self.event("landed_and_disarmed", vehicles=list(self.names))
                self.report["landing_verified"] = True
                self.report["landing_verification_method"] = "fresh_ground_contact_at_known_pad_and_stable_for_2s"
                self.report["api_control_released"] = {n: not self.client.isApiControlEnabled(vehicle_name=n) for n in self.names}
                if not all(self.report["api_control_released"].values()):
                    raise RuntimeError("Could not release API control after landing")
                return
            time.sleep(0.25)
        raise TimeoutError("Landing not confirmed; will pause without disarming")

    def run(self):
        try:
            if not set(self.names).issubset(self.client.listVehicles()):
                raise RuntimeError("Both vehicles required")
            balloon = vector(self.client.simGetObjectPose(self.target).position)
            if not np.allclose(balloon, self.target_position, atol=0.02):
                raise RuntimeError("Run setup_scene.py first; route requires target at configured position")
            positions = {}
            for name in self.names:
                p = np.array(vector(self.client.simGetObjectPose(name).position))
                if not np.isfinite(p).all() or not (-7 <= p[0] <= 0.5 and abs(p[1]) <= 9 and -4 <= p[2] <= 1.5):
                    raise RuntimeError(f"Outside inspected Blocks flight area: {name}, {p}")
                positions[name] = p
                self.timestamps[name] = self.client.simGetCollisionInfo(vehicle_name=name).time_stamp
            self.client.simRunConsoleCommand("ke SimHUD_0 RemoveAllDebugStrings")
            self.client.simSetObjectPose("ExternalCamera", airsim.Pose(airsim.Vector3r(-12,-14,-8),
                 airsim.to_quaternion(math.atan2(-7, math.hypot(10,14)),0,math.atan2(14,10))), True)
            self.client.simPause(False)
            for name in self.names:
                if not self.client.isApiControlEnabled(vehicle_name=name):
                    self.client.enableApiControl(True, vehicle_name=name)
                if not self.client.armDisarm(True, vehicle_name=name):
                    raise RuntimeError(f"Cannot arm {name}")
            self.phase = "TAKEOFF"
            self.event("takeoff")
            self.drive({n: [p[0],p[1],-3] for n,p in positions.items()})
            self.phase = "STAGING"
            self.drive(self.config.staging_world_ned_m)
            self.holds = {"DroneB": self.config.staging_world_ned_m["DroneB"]}
            self.change("start")
            if self.scenario == "a_hit":
                success = self.contact(self.coordinator.active_vehicle)
                if not success:
                    self.change("miss")
            else:
                self.active = "DroneA"
                self.phase = "DELIBERATE_MISS"
                self.event("planned_miss", offset_y_m=self.miss_offset_y,
                           description="A follows the configured offset route outside the balloon",
                           source="test_injection", reason_code="INJECTED_MISS",
                           evaluation_role="fault_injection")
                miss = self.target_position.copy()
                miss[1] += self.miss_offset_y
                self.drive({"DroneA": miss}, speed=0.7)
                self.change("miss", source="test_injection",
                            reason_code="INJECTED_MISS", evaluation_role="fault_injection")
                success = False
            if not success:
                miss_time = time.monotonic()
                self.phase = "CLEAR_CORRIDOR"
                self.drive({"DroneA": self.config.staging_world_ned_m["DroneA"]})
                clear = self.event("corridor_clear")
                self.holds = {"DroneA": self.config.staging_world_ned_m["DroneA"]}
                self.change("fallback_ready")
                dispatched = self.event("fallback_dispatched", vehicle="DroneB")
                self.report["clear_to_dispatch_s"] = dispatched["monotonic_s"] - clear["monotonic_s"]
                self.report["miss_to_dispatch_s"] = dispatched["monotonic_s"] - miss_time
                if not self.contact(self.coordinator.active_vehicle):
                    self.change("timeout")
                    raise RuntimeError("B timed out without target collision evidence")
            self.holds = {}
            self.phase = "RETREAT"
            self.drive({self.active: self.approach_position})
            self.land()
            expected = "DroneA" if self.scenario == "a_hit" else "DroneB"
            if self.hit["vehicle"] != expected:
                raise RuntimeError(f"Scenario outcome mismatch; expected {expected}")
            self.report["status"] = "PASS"
            self.report["vision_records"] = self.vision_records
            self.event("experiment_pass")
        except BaseException as exc:
            self.report.update(status="FAIL", error=str(exc), failed_phase=self.phase,
                               error_type=type(exc).__name__, state_before_abort=self.state.value)
            self.report["failure_reason"] = str(exc)
            self.coordinator.abort()
            self.event("mission_aborted", reason=str(exc), phase=self.phase)
            self.log.exception("Experiment failed; freezing simulation for diagnosis")
            raise
        finally:
            try:
                self.client.simPause(True)
                self.report["paused"] = True
            except Exception as exc:
                self.report.update(status="FAIL", paused=False, pause_error=str(exc),
                                   manual_stop_required=True)
            self.report["vision_records"] = self.vision_records
            self.report["onboard_sensor_summary"] = self.onboard_sensors.summary()
            self.report.update(final_state=self.state.value, min_separation_m=self.min_separation,
                               max_waiting_drift_m=self.max_waiting_drift)
            if math.isfinite(self.min_miss_distance):
                self.report["min_deliberate_miss_centre_distance_m"] = self.min_miss_distance
            if not math.isfinite(self.min_separation):
                self.report["min_separation_m"] = None
            (self.folder / "report.json").write_text(json.dumps(self.report, indent=2), encoding="utf-8")
            self.stream.close()
            self.log.info("Results: %s", self.folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["a_hit", "fallback"], default="fallback")
    parser.add_argument("--miss-offset-y", type=float,
                        help="Override the deliberate fallback-test miss offset in metres")
    parser.add_argument("--target-offset", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="Shift target and approach positions within the inspected test region")
    args = parser.parse_args()
    Experiment(args.scenario, args.miss_offset_y, args.target_offset).run()
