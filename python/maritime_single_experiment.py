"""Single-drone hover perception benchmark on the saved MaritimeHarborV2 map.

The color detector sees only RGB. AirSim projected boxes are evaluation-only,
obtained by a separate RPC after each frame. No target pose or box guides flight.
"""
import argparse
import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import airsim
import cv2
import numpy as np

from balloon_detector import annotate, detect_balloon
from maritime_experiments import ROOT, apply, balloon_locations


CONFIG = ROOT / "configs" / "maritime_single_experiment.json"


def vec(v):
    return np.array([v.x_val, v.y_val, v.z_val], dtype=float)


def box_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    b = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return overlap / (a + b - overlap) if a + b > overlap else 0.0


def associate_selected(selected, boxes, target_id):
    """Associate a selected RGB box with simulator boxes for later evaluation."""
    if selected is None:
        return "NO_CANDIDATE", None, 0.0
    x, y, w, h = selected["bbox_xywh"]
    candidate = (x, y, x + w, y + h)
    scores = [(box_iou(candidate, box), key) for key, box in boxes.items()]
    score, key = max(scores, default=(0.0, None))
    if score < 0.1:
        return "UNMATCHED_CANDIDATE", None, score
    return ("TARGET" if key == target_id else "OTHER_INSTANCE"), key, score


def load_config(path=CONFIG):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if (cfg["scene"] != "MaritimeHarborV2" or cfg["rpc_port"] != 41452
            or cfg["vehicle"] != "DroneA" or cfg["bystander"] != "DroneB"
            or cfg["target_id"] != "balloon_001"):
        raise ValueError("Maritime single-drone scene contract mismatch")
    if cfg["observation_local_ned_m"] != [0.0, 0.0, -6.0]:
        raise ValueError("Only the validated vertical observation point is allowed")
    profiles = json.loads((ROOT / "configs" / "maritime_experiments.json").read_text(
        encoding="utf-8"))["profiles"]
    if not cfg["profiles"] or len(cfg["profiles"]) != len(set(cfg["profiles"])):
        raise ValueError("Profile list must be nonempty and unique")
    for name in cfg["profiles"]:
        if name not in profiles:
            raise ValueError(f"Unknown profile: {name}")
        balloon_locations(profiles[name])
    tasks = cfg.get("tasks", {})
    if tasks and set(tasks) != set(cfg["profiles"]):
        raise ValueError("Every configured profile needs exactly one task")
    for name, task in tasks.items():
        side = task.get("expected_target_side")
        description = task.get("description", "")
        if side not in ("left", "right") or not isinstance(description, str):
            raise ValueError(f"Invalid target relation task for {name}")
        if not description.strip() or f"{side}most red balloon" not in description.lower():
            raise ValueError(f"Task description disagrees with expected target side: {name}")
    if not 1 <= cfg["frames_per_profile"] <= 30 or not 0 <= cfg["frame_interval_s"] <= 2:
        raise ValueError("Invalid sampling bounds")
    safety = cfg["safety"]
    if not (0 < safety["hover_position_tolerance_m"] <= 0.5
            and 0 < safety["hover_speed_max_m_s"] <= 0.3
            and 0 < safety["bystander_drift_max_m"] <= 0.15
            and 0 < safety["bystander_speed_max_m_s"] <= 0.1):
        raise ValueError("Unsafe motion threshold")
    return cfg


class MaritimeSingleExperiment:
    def __init__(self, config_path=CONFIG, client=None, record_prefix="maritime_single"):
        self.cfg = load_config(config_path)
        self.client = client or airsim.MultirotorClient(port=41452, timeout_value=15)
        self.folder = ROOT / "logs" / datetime.now().strftime(
            f"{record_prefix}_%Y%m%d_%H%M%S_%f")
        self.folder.mkdir(parents=True)
        self.report = {"status": "RUNNING", "config": self.cfg,
            "detector": "RGB red/ellipse candidate, largest-area selection; no target identity input",
            "evaluation_truth": "AirSim projected actor boxes via separate RPC after RGB capture",
            "flight": "DroneA vertical takeoff/hover/land at validated observation point only",
            "target_identity": self.cfg["target_id"], "conditions": [],
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source_sha256": {name: hashlib.sha256((ROOT / "python" / name).read_bytes()).hexdigest()
                for name in ("maritime_single_experiment.py", "maritime_experiments.py",
                             "balloon_detector.py")}}
        self.active = False
        self.baseline_ground_z = None
        self.bystander_start = None
        self.collision_stamp = None
        self.profile_applied = False

    def snapshot(self, expect_hover=False):
        c = self.client
        a = c.getMultirotorState(vehicle_name="DroneA")
        b = c.getMultirotorState(vehicle_name="DroneB")
        ap, bp = vec(a.kinematics_estimated.position), vec(b.kinematics_estimated.position)
        av, bv = vec(a.kinematics_estimated.linear_velocity), vec(b.kinematics_estimated.linear_velocity)
        if not np.isfinite(np.r_[ap, bp, av, bv]).all():
            raise RuntimeError("Invalid vehicle telemetry")
        if self.bystander_start is not None:
            if np.linalg.norm(bp - self.bystander_start) > self.cfg["safety"]["bystander_drift_max_m"]:
                raise RuntimeError("DroneB moved during the single-drone experiment")
            if np.linalg.norm(bv) > self.cfg["safety"]["bystander_speed_max_m_s"]:
                raise RuntimeError("DroneB is moving")
        collision = c.simGetCollisionInfo(vehicle_name="DroneA")
        if self.collision_stamp is not None and collision.time_stamp != self.collision_stamp:
            raise RuntimeError(f"DroneA collision changed: {collision.object_name}")
        if expect_hover:
            goal = np.asarray(self.cfg["observation_local_ned_m"])
            if (np.linalg.norm(ap - goal) > self.cfg["safety"]["hover_position_tolerance_m"]
                    or np.linalg.norm(av) > self.cfg["safety"]["hover_speed_max_m_s"]):
                raise RuntimeError(f"DroneA left stable observation state: {ap.tolist()}, {av.tolist()}")
        return ap, av, bp, bv

    def wait_hover(self):
        deadline, stable = time.monotonic() + 20, None
        goal = np.asarray(self.cfg["observation_local_ned_m"])
        while time.monotonic() < deadline:
            ap, av, _, _ = self.snapshot()
            ready = (np.linalg.norm(ap - goal) <= self.cfg["safety"]["hover_position_tolerance_m"]
                     and np.linalg.norm(av) <= self.cfg["safety"]["hover_speed_max_m_s"])
            stable = (stable or time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable >= 2:
                return
            time.sleep(0.2)
        raise TimeoutError("DroneA observation hover did not settle")

    def setup_truth(self):
        c = self.client
        registry = json.loads((ROOT / "configs" / "maritime_targets.json").read_text(
            encoding="utf-8"))
        names = c.simListSceneObjects("MH2_.*")
        resolved = {}
        for target in registry["targets"]:
            found = [name for name in names if re.fullmatch(target["actor_pattern"], name)]
            if len(found) != target["expected_components"]:
                raise RuntimeError(f"Target registry mismatch: {target['id']}: {found}")
            resolved[target["id"]] = found
        c.simClearDetectionMeshNames("front_center", airsim.ImageType.Scene,
                                     vehicle_name="DroneA")
        c.simSetDetectionFilterRadius("front_center", airsim.ImageType.Scene, 20000,
                                       vehicle_name="DroneA")
        for found in resolved.values():
            for name in found:
                c.simAddDetectionFilterMeshName("front_center", airsim.ImageType.Scene,
                                                name, vehicle_name="DroneA")
        self.report["evaluation_registry"] = resolved
        return resolved

    def capture(self, profile, index, resolved):
        c = self.client
        self.snapshot(expect_hover=True)
        image_rpc_start_ns = time.time_ns()
        response = c.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene,
                                                       False, True)], vehicle_name="DroneA")[0]
        image_rpc_end_ns = time.time_ns()
        data = bytes(response.image_data_uint8)
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if frame is None or frame.shape[:2] != (720, 1280):
            raise RuntimeError("Invalid maritime RGB frame")
        prediction = detect_balloon(frame)
        truth_rpc_start_ns = time.time_ns()
        detections = c.simGetDetections("front_center", airsim.ImageType.Scene,
                                        vehicle_name="DroneA")
        truth_rpc_end_ns = time.time_ns()
        by_name = {d.name: [d.box2D.min.x_val, d.box2D.min.y_val,
                            d.box2D.max.x_val, d.box2D.max.y_val] for d in detections}
        boxes = {}
        for target_id, names in resolved.items():
            parts = [by_name[n] for n in names if n in by_name]
            if parts:
                boxes[target_id] = [min(x[0] for x in parts), min(x[1] for x in parts),
                                    max(x[2] for x in parts), max(x[3] for x in parts)]
        task = self.cfg.get("tasks", {}).get(profile)
        observed_side = None
        relation_verified = None
        if task:
            target_box, distractor_box = boxes.get(self.cfg["target_id"]), boxes.get("balloon_003")
            if target_box and distractor_box:
                target_cx = (target_box[0] + target_box[2]) / 2
                distractor_cx = (distractor_box[0] + distractor_box[2]) / 2
                if abs(target_cx - distractor_cx) > 10:
                    observed_side = "left" if target_cx < distractor_cx else "right"
            relation_verified = observed_side == task["expected_target_side"]
        outcome, matched_id, overlap = associate_selected(
            prediction["selected"], boxes, self.cfg["target_id"])
        stem = f"{profile}_{index:03d}"
        (self.folder / f"{stem}.png").write_bytes(data)
        cv2.imwrite(str(self.folder / f"{stem}_detected.png"), annotate(frame, prediction))
        ap, av, bp, bv = self.snapshot(expect_hover=True)
        return {"frame": f"{stem}.png", "frame_timestamp_ns": int(response.time_stamp),
                "image_rpc_host_ns": [image_rpc_start_ns, image_rpc_end_ns],
                "truth_rpc_host_ns": [truth_rpc_start_ns, truth_rpc_end_ns],
                "vehicle_local_ned_m": ap.tolist(), "vehicle_speed_m_s": float(np.linalg.norm(av)),
                "bystander_local_ned_m": bp.tolist(),
                "detector_candidate_count": prediction["candidate_count"],
                "detector_selected_bbox_xywh": prediction["selected"]["bbox_xywh"]
                    if prediction["selected"] else None,
                "simulator_boxes_xyxy": boxes,
                "target_box_returned": self.cfg["target_id"] in boxes,
                "association": outcome, "associated_target_id": matched_id,
                "association_iou": overlap,
                "task_description": task["description"] if task else None,
                "expected_target_side": task["expected_target_side"] if task else None,
                "observed_target_side": observed_side,
                "task_relation_verified": relation_verified,
                "association_temporal_limit": "RGB and simulator boxes obtained via separate RPCs"}

    def land(self):
        c = self.client
        c.moveToZAsync(-1.0, 1.0, timeout_sec=20, vehicle_name="DroneA").join()
        c.landAsync(timeout_sec=45, vehicle_name="DroneA").join()
        deadline, stable = time.monotonic() + 20, None
        while time.monotonic() < deadline:
            a = c.getMultirotorState(vehicle_name="DroneA")
            p, speed = vec(a.kinematics_estimated.position), float(np.linalg.norm(
                vec(a.kinematics_estimated.linear_velocity)))
            ready = abs(p[2] - self.baseline_ground_z) < 0.15 and speed < 0.1
            stable = (stable or time.monotonic()) if ready else None
            if stable is not None and time.monotonic() - stable >= 1:
                self.report["landing_evidence"] = {"local_ned_m": p.tolist(),
                    "ground_reference_z_m": self.baseline_ground_z, "speed_m_s": speed,
                    "landed_flag_before_disarm": a.landed_state == airsim.LandedState.Landed}
                if not c.armDisarm(False, vehicle_name="DroneA"):
                    raise RuntimeError("DroneA disarm failed")
                c.enableApiControl(False, vehicle_name="DroneA")
                if c.isApiControlEnabled(vehicle_name="DroneA"):
                    raise RuntimeError("DroneA API release failed")
                self.active = False
                # SimpleFlight may update Landed shortly after landAsync and disarm.
                settle_deadline = time.monotonic() + 5
                settled = False
                while time.monotonic() < settle_deadline:
                    state = c.getMultirotorState(vehicle_name="DroneA")
                    settled = state.landed_state == airsim.LandedState.Landed
                    if settled:
                        break
                    time.sleep(0.2)
                self.report["landing_evidence"]["landed_flag_after_settle"] = settled
                return
            time.sleep(0.25)
        raise TimeoutError("DroneA landing not verified; keep armed and pause")

    def run(self):
        c = self.client
        try:
            if not {"DroneA", "DroneB"}.issubset(c.listVehicles()):
                raise RuntimeError("Both configured vehicles must exist")
            expected = json.loads((ROOT / "configs" / "maritime_harbor_settings.json").read_text(
                encoding="utf-8"))
            live = json.loads(c.getSettingsString())
            if live["Vehicles"] != expected["Vehicles"]:
                raise RuntimeError("Restart harbor with maritime_harbor_settings.json")
            if any(c.isApiControlEnabled(vehicle_name=n) for n in ("DroneA", "DroneB")):
                raise RuntimeError("Both vehicles must have API control released")
            ap, av, bp, bv = self.snapshot()
            if not (0.5 < ap[2] < 1.1 and np.linalg.norm(av) < 0.1
                    and 0.5 < bp[2] < 1.1 and np.linalg.norm(bv) < 0.1):
                raise RuntimeError("Both drones must be stationary on the observation pad")
            self.baseline_ground_z = float(ap[2])
            self.bystander_start = bp
            self.report["ground_reference"] = {"DroneA_local_ned_m": ap.tolist(),
                                                "DroneB_local_ned_m": bp.tolist()}
            self.collision_stamp = c.simGetCollisionInfo(vehicle_name="DroneA").time_stamp
            resolved = self.setup_truth()
            c.simPause(False)
            c.enableApiControl(True, vehicle_name="DroneA")
            if not c.armDisarm(True, vehicle_name="DroneA"):
                raise RuntimeError("DroneA arm failed")
            self.active = True
            c.moveToZAsync(-6.0, 1.0, timeout_sec=35, vehicle_name="DroneA").join()
            self.wait_hover()
            for profile in self.cfg["profiles"]:
                self.snapshot(expect_hover=True)
                applied = apply(c, profile)
                self.profile_applied = True
                entry = {"profile": profile, "applied": applied,
                         "task": self.cfg.get("tasks", {}).get(profile), "frames": []}
                self.report["conditions"].append(entry)
                for index in range(self.cfg["frames_per_profile"]):
                    entry["frames"].append(self.capture(profile, index, resolved))
                    if index + 1 < self.cfg["frames_per_profile"]:
                        time.sleep(self.cfg["frame_interval_s"])
                visible = [f for f in entry["frames"] if f["target_box_returned"]]
                entry["summary"] = {
                    "frames": len(entry["frames"]), "target_box_returned_frames": len(visible),
                    "target_selected_when_box_returned": sum(f["association"] == "TARGET"
                        for f in visible),
                    "other_instance_selected_frames": sum(f["association"] == "OTHER_INSTANCE"
                        for f in entry["frames"]),
                    "no_candidate_frames": sum(f["association"] == "NO_CANDIDATE"
                        for f in entry["frames"]),
                    "task_relation_verified_frames": sum(f["task_relation_verified"] is True
                        for f in entry["frames"])}
                (self.folder / "report.json").write_text(json.dumps(self.report,
                    ensure_ascii=False, indent=2), encoding="utf-8")
            apply(c, "baseline")
            self.report["scene_restored_to_baseline"] = True
            self.land()
            mismatches = [f["frame"] for condition in self.report["conditions"]
                for f in condition["frames"] if f["task_relation_verified"] is False]
            self.report["scene_relation_mismatches"] = mismatches
            self.report["status"] = ("LANDING_STATE_UNCONFIRMED" if not
                self.report["landing_evidence"]["landed_flag_after_settle"] else
                "SCENE_VALIDATION_FAIL" if mismatches else "PASS")
        except BaseException as exc:
            self.report.update(status="FAIL", error=str(exc), error_type=type(exc).__name__,
                               manual_recovery_required=self.active)
            raise
        finally:
            if self.profile_applied and not self.report.get("scene_restored_to_baseline"):
                try:
                    apply(c, "baseline")
                    self.report["scene_restored_to_baseline"] = True
                except Exception as exc:
                    self.report.update(status="FAIL", scene_restore_error=str(exc))
            try:
                c.simClearDetectionMeshNames("front_center", airsim.ImageType.Scene,
                                             vehicle_name="DroneA")
            except Exception as exc:
                self.report["detection_cleanup_error"] = str(exc)
            try:
                c.simPause(True)
                self.report["paused"] = True
            except Exception as exc:
                self.report.update(status="FAIL", paused=False, pause_error=str(exc))
            self.report["finished_at"] = datetime.now().astimezone().isoformat()
            (self.folder / "report.json").write_text(json.dumps(self.report,
                ensure_ascii=False, indent=2), encoding="utf-8")
            print(self.folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG, type=Path)
    experiment = MaritimeSingleExperiment(parser.parse_args().config)
    experiment.run()
    if experiment.report["status"] != "PASS":
        raise SystemExit(1)
