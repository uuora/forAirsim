"""Steps 1/2: two-drone staging, A approaches, B holds; no balloon contact.

Ends with the simulator paused for inspection. Errors also pause physics,
without disarming an airborne vehicle. Rerunning resumes the simulation.
"""
import json
import logging
import math
import sys
import time
from datetime import datetime

import airsim
import numpy as np

from setup_scene import ROOT, vector, capture_overview

NAMES = ("DroneA", "DroneB")


def main():
    folder = ROOT / "logs" / datetime.now().strftime("approach_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(folder / "mission.log", encoding="utf-8")])
    client = airsim.MultirotorClient(timeout_value=10)
    report = {"status": "RUNNING", "phases": [], "finish": "pause_for_inspection"}
    phase = "PREFLIGHT"
    offsets = {}
    collision_times = {}
    min_separation = float("inf")
    max_b_drift = 0.0
    stream = (folder / "telemetry.jsonl").open("w", encoding="utf-8")

    def sample(labels=True):
        nonlocal min_separation, max_b_drift
        positions, velocities = {}, {}
        for name in NAMES:
            positions[name] = np.array(vector(client.simGetObjectPose(name).position))
            state = client.getMultirotorState(vehicle_name=name)
            velocities[name] = np.linalg.norm(vector(state.kinematics_estimated.linear_velocity))
            if not np.isfinite(positions[name]).all():
                raise RuntimeError(f"Invalid pose: {name}")
            collision = client.simGetCollisionInfo(vehicle_name=name)
            if collision.has_collided and collision.time_stamp > collision_times[name]:
                raise RuntimeError(f"New collision: {name} with {collision.object_name}")
        separation = float(np.linalg.norm(positions[NAMES[0]] - positions[NAMES[1]]))
        min_separation = min(min_separation, separation)
        if separation < 2.0:
            raise RuntimeError(f"Drone separation too small: {separation:.2f} m")
        if phase in ("A_APPROACH", "HOLD"):
            drift = float(np.linalg.norm(positions["DroneB"] - staging["DroneB"]))
            max_b_drift = max(max_b_drift, drift)
            if drift > 0.5:
                raise RuntimeError(f"DroneB drifted {drift:.2f} m from standby")
        row = {"time": datetime.now().astimezone().isoformat(), "phase": phase,
               "world_ned_m": {k: v.tolist() for k, v in positions.items()},
               "speed_m_s": velocities, "separation_m": separation}
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        for name, color in (("DroneA", [0, 0.8, 1, 1]), ("DroneB", [1, 0.8, 0, 1])) if labels else ():
            p = positions[name] + [0, 0, -0.6]
            client.simPlotStrings([name + (" - APPROACH" if name == "DroneA" and phase == "A_APPROACH" else " - " + phase)],
                                 [airsim.Vector3r(*p)], scale=1.5, color_rgba=color, duration=0.6)
        return positions, velocities

    def wait_positions(targets, timeout=35, hold=1.5):
        deadline = time.monotonic() + timeout
        stable = None
        started = time.monotonic()
        refining = set()
        while time.monotonic() < deadline:
            positions, speeds = sample()
            # AirSim's path follower can stop about 0.4 m short. Near the
            # endpoint use continuously refreshed, capped P-control velocities.
            # Discrete pulses allow SimpleFlight to lose height between commands.
            for name, target in targets.items():
                delta = np.array(target) - positions[name]
                distance = np.linalg.norm(delta)
                if time.monotonic() - started > 2 and (distance < 1.0 or speeds[name] < 0.1):
                    refining.add(name)
                if name in refining:
                    velocity = delta * min(0.8, 0.6 / max(distance, 0.001))
                    client.moveByVelocityAsync(*velocity, duration=1.0,
                                               yaw_mode=airsim.YawMode(False, 0), vehicle_name=name)
            ok = all(np.linalg.norm(positions[n] - p) < 0.3 and speeds[n] < 0.25
                     for n, p in targets.items())
            stable = (stable or time.monotonic()) if ok else None
            if stable is not None and time.monotonic() - stable >= hold:
                report["phases"].append({"phase": phase, "actual_world_ned_m":
                                          {n: positions[n].tolist() for n in NAMES}})
                return
            time.sleep(0.25)
        raise TimeoutError(f"{phase}: failed to reach and hold target positions")

    def move(name, world, speed=1.0):
        local = np.array(world) - offsets[name]
        client.moveToPositionAsync(*local, speed, timeout_sec=35,
                                   yaw_mode=airsim.YawMode(False, 0), vehicle_name=name)

    try:
        if not set(NAMES).issubset(client.listVehicles()):
            raise RuntimeError("DroneA / DroneB missing; run start-scene.ps1 first")
        cfg = json.loads((ROOT / "configs/balloon.json").read_text())
        balloon = np.array(vector(client.simGetObjectPose(cfg["name"]).position))
        if not np.isfinite(balloon).all() or not np.allclose(balloon, [0, 0, -3], atol=0.02):
            raise RuntimeError("This verified Blocks route requires balloon world NED (0,0,-3)")
        staging = {"DroneA": np.array([-5., -4., -3.]), "DroneB": np.array([-5., 4., -3.])}
        approach = np.array([-2., 0., -3.])
        report["staging_world_ned_m"] = {n: p.tolist() for n, p in staging.items()}
        report["approach_world_ned_m"] = approach.tolist()
        for name in NAMES:
            world = np.array(vector(client.simGetObjectPose(name).position))
            local = np.array(vector(client.getMultirotorState(vehicle_name=name).kinematics_estimated.position))
            offsets[name] = world - local
            # This is a bounded demo for the inspected clear area in Blocks.
            if not (-7 <= world[0] <= 0.5 and -9 <= world[1] <= 9 and -4 <= world[2] <= 1.5):
                raise RuntimeError(f"{name} outside the inspected demo area: {world}")
            collision_times[name] = client.simGetCollisionInfo(vehicle_name=name).time_stamp
        report["world_minus_local_offsets"] = {n: p.tolist() for n, p in offsets.items()}
        # UE editor-game runtime: clear labels left by a previous paused replay.
        client.simRunConsoleCommand("ke SimHUD_0 RemoveAllDebugStrings")
        client.simPause(False)
        for name in NAMES:
            # Re-requesting active API control resets the SimpleFlight goal to
            # RC input. Preserve control when resuming an already armed drone.
            if not client.isApiControlEnabled(vehicle_name=name):
                client.enableApiControl(True, vehicle_name=name)
            if not client.armDisarm(True, vehicle_name=name):
                raise RuntimeError(f"Cannot arm {name}")
        camera = airsim.Pose(airsim.Vector3r(-12, -14, -8), airsim.to_quaternion(
            math.atan2(-7, math.hypot(10, 14)), 0, math.atan2(14, 10)))
        client.simSetObjectPose("ExternalCamera", camera, teleport=True)
        phase = "TAKEOFF"
        logging.info("TAKEOFF: raising both drones to world Z=-3 m")
        climb = {}
        for name in NAMES:
            climb[name] = np.array(vector(client.simGetObjectPose(name).position))
            climb[name][2] = -3
            move(name, climb[name])
        wait_positions(climb)
        phase = "STAGING"
        logging.info("STAGING: A=(-5,-4,-3), B=(-5,4,-3), world metres")
        for name in NAMES:
            move(name, staging[name])
        wait_positions(staging)
        client.hoverAsync(vehicle_name="DroneB")
        capture_overview(client, folder / "01_staging.png")
        phase = "A_APPROACH"
        logging.info("A_APPROACH: A to (-2,0,-3); B holds; stop 2 m from balloon centre")
        move("DroneA", approach, speed=0.7)
        wait_positions({"DroneA": approach, "DroneB": staging["DroneB"]})
        for name in NAMES:
            client.hoverAsync(vehicle_name=name)
        phase = "HOLD"
        logging.info("HOLD: observing both drones for 10 seconds")
        wait_positions({"DroneA": approach, "DroneB": staging["DroneB"]}, timeout=20, hold=10)
        capture_overview(client, folder / "02_approach.png")
        for name in NAMES:
            client.hoverAsync(vehicle_name=name)
        # Let the short-lived moving labels expire before adding final labels.
        time.sleep(0.7)
        positions, _ = sample(labels=False)
        report.update(status="PASS", min_separation_m=min_separation,
                      max_b_standby_drift_m=max_b_drift,
                      a_balloon_centre_distance_m=float(np.linalg.norm(positions["DroneA"] - balloon)))
        for name in NAMES:
            client.simPlotStrings([name + (" - READY" if name == "DroneA" else " - STANDBY")],
                                 [airsim.Vector3r(*(positions[name] + [0, 0, -0.6]))],
                                 scale=1.5, color_rgba=[0, 0.8, 1, 1] if name == "DroneA" else [1, 0.8, 0, 1], duration=0.6)
        logging.info("PASS: demo complete; pausing simulation for inspection. Logs: %s", folder)
    except BaseException as exc:
        report.update(status="FAIL", failed_phase=phase, error=str(exc))
        logging.exception("Demo failed; pausing physics, keeping airborne motors armed")
        raise
    finally:
        try:
            client.simPause(True)
            report["simulator_paused"] = True
        except Exception as exc:
            report["pause_error"] = str(exc)
            report["status"] = "FAIL"
            logging.error("Could not pause simulator: %s", exc)
        stream.close()
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
