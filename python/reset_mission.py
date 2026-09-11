"""Restore the inspected Blocks scene before a new experiment.

The reset is deliberately conservative: it refuses to disarm an airborne
vehicle. Land the vehicles first, then this command recreates the balloon and
writes a reset report under logs/.
"""
import json
import logging
import subprocess
import sys
from datetime import datetime

import airsim

from setup_scene import ROOT


def main():
    run_dir = ROOT / "logs" / datetime.now().strftime("reset_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True)
    report = {"status": "RUNNING", "timestamp": datetime.now().astimezone().isoformat()}
    log_path = run_dir / "reset.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")])
    try:
        client = airsim.MultirotorClient(timeout_value=20)
        vehicles = client.listVehicles()
        if not {"DroneA", "DroneB"}.issubset(vehicles):
            raise RuntimeError(f"Expected DroneA and DroneB, got {vehicles}")
        report["vehicles"] = {}
        for name in ("DroneA", "DroneB"):
            state = client.getMultirotorState(vehicle_name=name)
            position = state.kinematics_estimated.position
            velocity = state.kinematics_estimated.linear_velocity
            z = float(position.z_val)
            speed = (float(velocity.x_val) ** 2 + float(velocity.y_val) ** 2 + float(velocity.z_val) ** 2) ** 0.5
            report["vehicles"][name] = {"landed_state": state.landed_state,
                                        "api_control": client.isApiControlEnabled(vehicle_name=name),
                                        "local_ned_z": z, "speed_m_s": speed}
            # Blocks object poses report the pad at about z=1.68 in world NED;
            # this is different from the local flight-controller height.
            grounded_by_telemetry = 1.50 < z < 1.90 and speed < 0.08
            if state.landed_state != airsim.LandedState.Landed and not grounded_by_telemetry:
                raise RuntimeError(f"{name} is not landed; refusing airborne reset")
        client.simPause(True)
        for name in ("DroneA", "DroneB"):
            client.cancelLastTask(vehicle_name=name)
            if client.isApiControlEnabled(vehicle_name=name):
                client.armDisarm(False, vehicle_name=name)
                client.enableApiControl(False, vehicle_name=name)
        subprocess.run([sys.executable, str(ROOT / "python" / "setup_scene.py")], cwd=ROOT, check=True)
        report["status"] = "PASS"
        logging.info("MISSION_RESET_PASS: %s", run_dir)
    except Exception as exc:
        report.update(status="FAIL", error=str(exc), error_type=type(exc).__name__)
        logging.exception("Mission reset failed")
        raise
    finally:
        (run_dir / "reset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
