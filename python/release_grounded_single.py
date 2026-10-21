"""Release a physically grounded DroneA after a paused landing verification failure.

Requires a matching failure report, ground actor, return-pad position, stationary
DroneB and two seconds of fresh unpaused telemetry before disarming DroneA.
Does not issue a flight command or resume the failed experiment.
"""
import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import airsim

ROOT = Path(__file__).resolve().parents[1]


def length(values):
    return math.sqrt(sum(value * value for value in values))


def grounded(snapshot, pad):
    a, b = snapshot["DroneA"], snapshot["DroneB"]
    ap, bp = a["position"], b["position"]
    values = ap + bp + a["velocity"] + b["velocity"]
    return (all(math.isfinite(value) for value in values)
            and a["api_control"] and not b["api_control"]
            and b["landed_state"] == airsim.LandedState.Landed
            and a["collision_object"].startswith("Ground")
            and a["collision_timestamp"] > 0
            and 0.65 < ap[2] < 1.1 and 0.65 < bp[2] < 1.1
            and abs(ap[2] - bp[2]) < 0.08
            and length([ap[0] - pad[0], ap[1] - pad[1]]) < 0.4
            and length(a["velocity"]) < 0.08 and length(b["velocity"]) < 0.08)


def snapshot(client):
    result = {}
    for name in ("DroneA", "DroneB"):
        state = client.getMultirotorState(vehicle_name=name)
        p = client.simGetObjectPose(name).position
        v = state.kinematics_estimated.linear_velocity
        contact = client.simGetCollisionInfo(vehicle_name=name)
        result[name] = {"position": [p.x_val, p.y_val, p.z_val],
                        "velocity": [v.x_val, v.y_val, v.z_val],
                        "api_control": client.isApiControlEnabled(vehicle_name=name),
                        "landed_state": state.landed_state,
                        "collision_object": contact.object_name,
                        "collision_timestamp": contact.time_stamp}
    return result


def run(failure_path):
    failure_path = Path(failure_path).resolve()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if (failure.get("status") != "FAIL" or failure.get("failed_phase") != "LANDING"
            or failure.get("vehicle_commanded") != "DroneA"):
        raise ValueError("Requires a failed DroneA landing report")
    pad = failure["configuration"]["return_world_ned_m"]
    output = ROOT / "logs" / datetime.now().strftime("grounded_release_%Y%m%d_%H%M%S_%f")
    output.mkdir()
    report = {"status": "RUNNING", "source_failure": str(failure_path),
              "source_failure_sha256": hashlib.sha256(failure_path.read_bytes()).hexdigest(),
              "samples": [], "flight_commands_sent": False}
    client = airsim.MultirotorClient(timeout_value=10)
    try:
        if not client.simIsPause():
            raise RuntimeError("Simulation must be paused for the initial inspection")
        initial = snapshot(client)
        report["initial"] = initial
        if not grounded(initial, pad):
            raise RuntimeError("Grounded return-pad evidence is insufficient; API control was preserved")
        bystander = initial["DroneB"]["position"]
        client.simPause(False)
        started = time.monotonic()
        while time.monotonic() - started < 2.0:
            sample = snapshot(client)
            report["samples"].append(sample)
            if not grounded(sample, pad) or length([a - b for a, b in zip(
                    sample["DroneB"]["position"], bystander)]) > 0.1:
                raise RuntimeError("Ground telemetry changed; controls preserved and simulation paused")
            time.sleep(0.2)
        if not client.armDisarm(False, vehicle_name="DroneA"):
            raise RuntimeError("Disarm was refused")
        report["disarmed"] = True
        client.enableApiControl(False, vehicle_name="DroneA")
        if client.isApiControlEnabled(vehicle_name="DroneA"):
            raise RuntimeError("API release failed")
        report["api_released"] = True
        time.sleep(2.0)
        final = snapshot(client)
        report["final"] = final
        if (final["DroneA"]["landed_state"] != airsim.LandedState.Landed
                or length(final["DroneA"]["velocity"]) >= 0.08):
            raise RuntimeError("Disarmed and API released, but landed flag is still unconfirmed")
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=str(exc))
        raise
    finally:
        try:
            client.simPause(True)
            report["paused"] = True
        except Exception as exc:
            report.update(status="FAIL", paused=False, pause_error=str(exc))
        report["finished"] = datetime.now().astimezone().isoformat()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-report", type=Path, required=True)
    run(parser.parse_args().failure_report)
