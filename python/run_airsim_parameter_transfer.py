"""Replay one fixed scenario through the 03 controller with AirSim as the plant.

The offline objective (``trajectory_objective.evaluate_parameters``) runs the
production ``RollingController`` against a first-order lagged point mass. This
script runs the *same* controller, the *same* scenario and the *same* fixed
parameter vector, and swaps only the plant for AirSim's SimpleFlight. It exists
to test whether an offline ranking transfers, not to claim flight performance.

State injection is used to reach the scenario start pose, so this is a
simulation-only start; no claim is made about real takeoff behaviour.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from dual_target_control.core import BezierParameters, ControlConfig, RollingController
from dual_target_control.runtime import to_ned, wrap_angle
from dual_target_control.simulator import launch_simulator
from validate_dual_target_turning import SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ("Fixed-scenario AirSim plant swap for parameter-ranking transfer. Simulation-only state "
         "injection at the start pose; not real flight and not a collision or agility claim.")


def scenario_named(name):
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise ValueError("unknown scenario: " + name + " | available: " + ", ".join(s.name for s in SCENARIOS))


def parse_parameters(values):
    if values is None:
        return BezierParameters()
    numbers = [float(value) for value in values]
    return BezierParameters.from_vector(numbers)


def read_state(client, vehicle):
    state = client.getMultirotorState(vehicle_name=vehicle).kinematics_estimated
    position = np.array([state.position.x_val, state.position.y_val, -state.position.z_val], dtype=float)
    velocity = np.array([state.linear_velocity.x_val, state.linear_velocity.y_val,
                         -state.linear_velocity.z_val], dtype=float)
    orientation = state.orientation
    yaw = math.atan2(2.0 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
                     1.0 - 2.0 * (orientation.y_val ** 2 + orientation.z_val ** 2))
    return position, velocity, wrap_angle(yaw), state


def inject_start(client, airsim, vehicle, position, yaw):
    """Park the vehicle on the scenario start pose with zero velocity.

    The simulation is paused across the injection so the first control tick
    begins from the declared state instead of from a free-falling vehicle.
    """
    client.simPause(True)
    kinematics = airsim.KinematicsState()
    kinematics.position = airsim.Vector3r(float(position[0]), float(position[1]), float(-position[2]))
    kinematics.orientation = airsim.to_quaternion(0.0, 0.0, float(yaw))
    kinematics.linear_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    kinematics.angular_velocity = airsim.Vector3r(0.0, 0.0, 0.0)
    client.simSetKinematics(kinematics, True, vehicle_name=vehicle)
    measured, velocity, measured_yaw, _ = read_state(client, vehicle)
    return measured, velocity, measured_yaw


def run(options):
    config = json.loads(Path(options.config).read_text(encoding="utf-8-sig"))
    scenario = scenario_named(options.scenario)
    parameters = parse_parameters(options.parameters)
    control = dict(config["control"])
    control.update(scenario.config)
    cfg = ControlConfig(**control).validate()
    controller = RollingController(cfg, parameters=parameters)

    import airsim
    endpoint = config["airsim"]
    process = None
    if options.launch:
        process = launch_simulator(config, lambda status, message: print(f"[{status}] {message}", flush=True))
    client = airsim.MultirotorClient(ip=endpoint["host"], port=endpoint["port"],
                                     timeout_value=endpoint.get("rpc_timeout", 5.0))
    client.confirmConnection()
    vehicle = endpoint["vehicle"]
    client.enableApiControl(True, vehicle_name=vehicle)
    if not client.armDisarm(True, vehicle_name=vehicle):
        raise RuntimeError("AirSim refused to arm")

    legs, samples = [], []
    receipt = dict(schema_version=1, scope=SCOPE, scenario=scenario.name,
                   parameters=parameters.to_dict(), parameter_vector=parameters.to_vector().tolist(),
                   config=dict(arrival_radius=cfg.arrival_radius, max_speed=cfg.max_speed,
                               max_acceleration=cfg.max_acceleration, seed=cfg.seed),
                   targets=[list(map(float, t)) for t in scenario.targets], launched=bool(process))
    try:
        start = np.array(scenario.position, dtype=float)
        position, velocity, yaw = inject_start(client, airsim, vehicle, start, float(scenario.yaw))
        receipt["start_measured"] = dict(position=position.tolist(), velocity=velocity.tolist(), yaw_rad=yaw)
        receipt["start_error_m"] = float(np.linalg.norm(position - start))
        receipt["start_speed_mps"] = float(np.linalg.norm(velocity))
        if receipt["start_error_m"] > 0.2 or receipt["start_speed_mps"] > 0.2:
            raise RuntimeError(f"start injection rejected: error {receipt['start_error_m']:.3f} m, "
                               f"speed {receipt['start_speed_mps']:.3f} m/s")
        controller.reset(position, velocity, yaw)
        controller.set_targets(scenario.targets)
        client.simPause(False)

        duration = float(endpoint.get("command_duration", 0.15))
        hits_at = []
        started = time.perf_counter()
        previous_hits = 0
        previous_time = None
        dt_target = 1.0 / float(cfg.control_hz)
        while True:
            tick = time.perf_counter()
            elapsed = tick - started
            if elapsed > options.max_seconds:
                receipt["stopped_reason"] = "deadline"
                break
            position, velocity, yaw, raw = read_state(client, vehicle)
            dt = dt_target if previous_time is None else min(max(tick - previous_time, 0.005), 0.1)
            previous_time = tick
            out = controller.update(position, velocity, yaw, dt)
            command = np.asarray(out["velocity_command"], dtype=float)
            yaw_deg = float(out["yaw_command_deg"])
            client.moveByVelocityAsync(*to_ned(command), duration,
                                       drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                       yaw_mode=airsim.YawMode(False, yaw_deg), vehicle_name=vehicle)
            samples.append(dict(t=elapsed, position=position.tolist(), velocity=velocity.tolist(),
                                command=command.tolist(), yaw_deg=yaw_deg,
                                target_error_m=float(np.linalg.norm(position - np.asarray(controller.targets[0])))))
            hits = int(out["hits"])
            if hits > previous_hits:
                for _ in range(hits - previous_hits):
                    hits_at.append(dict(leg=len(hits_at) + 1, elapsed_s=elapsed,
                                        position=position.tolist()))
                previous_hits = hits
            if hits >= len(scenario.targets):
                receipt["stopped_reason"] = "targets_reached"
                break
            remaining = dt_target - (time.perf_counter() - tick)
            if remaining > 0:
                time.sleep(remaining)
        receipt["legs"] = hits_at
        receipt["elapsed_s"] = samples[-1]["t"] if samples else 0.0
        receipt["hits"] = len(hits_at)
        receipt["targets_total"] = len(scenario.targets)
        receipt["samples"] = len(samples)
        receipt["peak_command_speed_mps"] = max((float(np.linalg.norm(s["command"])) for s in samples), default=None)
        receipt["peak_measured_speed_mps"] = max((float(np.linalg.norm(s["velocity"])) for s in samples), default=None)
        receipt["max_target_error_m"] = max((s["target_error_m"] for s in samples), default=None)
        receipt["final_position"] = samples[-1]["position"] if samples else None
        receipt["status"] = "COMPLETED" if len(hits_at) == len(scenario.targets) else "INCOMPLETE"
        receipt["_samples"] = samples
    except Exception as error:
        receipt.update(status="FAILED", failure_type=type(error).__name__, failure_reason=str(error))
    finally:
        try:
            client.simPause(True)
            client.enableApiControl(False, vehicle_name=vehicle)
        except Exception as error:  # keep the receipt even when cleanup fails
            receipt["cleanup_error"] = f"{type(error).__name__}: {error}"
        if process is not None and not options.keep_alive:
            process.terminate()

    output = Path(options.output)
    output.mkdir(parents=True, exist_ok=False)
    samples = receipt.pop("_samples", None)
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
    if options.dump_samples and samples:
        (output / "samples.json").write_text(
            json.dumps(samples, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=receipt["status"], scenario=scenario.name, hits=receipt.get("hits"),
                          elapsed_s=receipt.get("elapsed_s"),
                          peak_measured_speed_mps=receipt.get("peak_measured_speed_mps"),
                          start_error_m=receipt.get("start_error_m"),
                          failure_reason=receipt.get("failure_reason"), output=str(output)), ensure_ascii=False))
    return 0 if receipt["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/dual_target_control.json"))
    parser.add_argument("--scenario", default="right_angle")
    parser.add_argument("--parameters", nargs="+", type=float, default=None,
                        help="ordered BezierParameters vector; omitted uses defaults")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--launch", action="store_true", help="start the 03 AirSim instance if the port is closed")
    parser.add_argument("--keep-alive", action="store_true",
                        help="leave the simulator running so a sweep reuses one instance")
    parser.add_argument("--dump-samples", action="store_true", help="write the per-tick trace to samples.json")
    raise SystemExit(run(parser.parse_args()))
