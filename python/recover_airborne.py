"""Recover both vehicles after an airborne mission abort."""
import json
import logging
import time
from datetime import datetime

import airsim
import numpy as np

from mission_config import MissionConfig
from setup_scene import ROOT, vector


def grounded(position, speed, landed_flag=False):
    return speed < 0.08 and (landed_flag or 0.65 <= position[2] <= 1.90)


def snapshot(client, vehicles):
    evidence = {}
    for name in vehicles:
        state = client.getMultirotorState(vehicle_name=name)
        position = vector(client.simGetObjectPose(name).position)
        speed = float(np.linalg.norm(vector(state.kinematics_estimated.linear_velocity)))
        landed_flag = state.landed_state == airsim.LandedState.Landed
        evidence[name] = {"world_ned_m": position, "speed_m_s": speed,
                          "landed_flag": landed_flag,
                          "grounded_by_flag_or_pose_and_speed": grounded(position, speed, landed_flag)}
    return evidence


def drive_to_pads(client, cfg):
    deadline = time.monotonic() + 50
    stable_since = None
    while time.monotonic() < deadline:
        positions = {name: np.asarray(vector(client.simGetObjectPose(name).position))
                     for name in cfg.vehicles}
        states = {name: client.getMultirotorState(vehicle_name=name) for name in cfg.vehicles}
        ready = True
        for name in cfg.vehicles:
            delta = cfg.pads_world_ned_m[name] - positions[name]
            distance = float(np.linalg.norm(delta))
            speed = float(np.linalg.norm(vector(states[name].kinematics_estimated.linear_velocity)))
            ready &= distance < 0.25 and speed < 0.2
            velocity = delta * min(0.8, 1.0 / max(distance, 0.001))
            client.moveByVelocityAsync(*velocity, duration=1.0, vehicle_name=name)
        stable_since = (stable_since or time.monotonic()) if ready else None
        if stable_since is not None and time.monotonic() - stable_since >= 1:
            return
        time.sleep(0.2)
    raise TimeoutError("Could not reach recovery pads")


def main():
    folder = ROOT / "logs" / datetime.now().strftime("recovery_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    log = logging.getLogger(str(folder))
    log.setLevel(logging.INFO)
    log.propagate = False
    for handler in (logging.StreamHandler(), logging.FileHandler(folder / "recovery.log", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        log.addHandler(handler)
    report = {"status": "RUNNING", "started_at": datetime.now().astimezone().isoformat()}
    client = airsim.MultirotorClient(timeout_value=10)
    cfg = MissionConfig.load(ROOT / "configs" / "mission.json")
    try:
        client.confirmConnection()
        if not set(cfg.vehicles).issubset(client.listVehicles()):
            raise RuntimeError("Both vehicles are required for coordinated recovery")
        before = {name: vector(client.simGetObjectPose(name).position) for name in cfg.vehicles}
        if any(not np.isfinite(position).all() for position in map(np.asarray, before.values())):
            raise RuntimeError("Invalid vehicle position; refusing blind recovery")
        report["before_world_ned_m"] = before
        evidence = snapshot(client, cfg.vehicles)
        if not all(item["grounded_by_flag_or_pose_and_speed"] for item in evidence.values()):
            for name in cfg.vehicles:
                client.cancelLastTask(vehicle_name=name)
                client.enableApiControl(True, vehicle_name=name)
                if not client.armDisarm(True, vehicle_name=name):
                    raise RuntimeError(f"Cannot arm {name} for recovery")
            client.simPause(False)
            drive_to_pads(client, cfg)
            lands = [client.landAsync(timeout_sec=45, vehicle_name=name) for name in cfg.vehicles]
            for land in lands:
                land.join()
            deadline = time.monotonic() + 20
            stable_since = None
            while time.monotonic() < deadline:
                evidence = snapshot(client, cfg.vehicles)
                ready = all(item["grounded_by_flag_or_pose_and_speed"] for item in evidence.values())
                stable_since = (stable_since or time.monotonic()) if ready else None
                if stable_since is not None and time.monotonic() - stable_since >= 2:
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError("Recovery landing did not become stable")
        for name in cfg.vehicles:
            if not client.armDisarm(False, vehicle_name=name):
                raise RuntimeError(f"Cannot disarm recovered {name}")
            client.enableApiControl(False, vehicle_name=name)
        report.update(status="PASS", landing_evidence=evidence,
                      api_control_released={name: not client.isApiControlEnabled(vehicle_name=name)
                                            for name in cfg.vehicles})
        log.info("RECOVERY_PASS: both vehicles landed, disarmed and released")
    except BaseException as exc:
        report.update(status="FAIL", error=str(exc), error_type=type(exc).__name__,
                      manual_stop_required=True)
        log.exception("Recovery failed; simulation will remain paused")
        raise
    finally:
        try:
            client.simPause(True)
            report["paused"] = True
        except Exception as exc:
            report.update(status="FAIL", paused=False, pause_error=str(exc), manual_stop_required=True)
        report["finished_at"] = datetime.now().astimezone().isoformat()
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("Results: %s", folder)


if __name__ == "__main__":
    main()
