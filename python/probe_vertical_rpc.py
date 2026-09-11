"""Calibrate SimpleFlight vertical velocity sign and drift in a safe single-drone run."""
import json
import time
from datetime import datetime
from pathlib import Path

import airsim

from setup_scene import ROOT, vector


def main():
    c = airsim.MultirotorClient(timeout_value=5)
    c.confirmConnection()
    if c.isApiControlEnabled(vehicle_name="DroneA"):
        raise RuntimeError("DroneA must start with API control released")
    folder = ROOT / "logs" / datetime.now().strftime("vertical_rpc_probe_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)
    rows = []
    try:
        c.simPause(False)
        c.enableApiControl(True, vehicle_name="DroneA")
        c.armDisarm(True, vehicle_name="DroneA")
        c.takeoffAsync(vehicle_name="DroneA").join()
        c.moveToZAsync(-3.0, 0.6, vehicle_name="DroneA").join()
        c.hoverAsync(vehicle_name="DroneA").join()
        base = c.simGetVehiclePose(vehicle_name="DroneA").position.z_val
        for label, vz in (("hold", 0.0), ("positive", 0.15), ("negative", -0.15)):
            for _ in range(12):
                p = c.simGetVehiclePose(vehicle_name="DroneA").position
                rows.append({"phase": label, "command_vz": vz, "z": p.z_val,
                             "world_ned_m": vector(c.simGetObjectPose("DroneA").position),
                             "time": time.time()})
                c.moveByVelocityAsync(0, 0, vz, 0.25,
                                      vehicle_name="DroneA").join()
                time.sleep(0.1)
        c.hoverAsync(vehicle_name="DroneA").join()
        c.moveToZAsync(-3.0, 0.6, vehicle_name="DroneA").join()
        c.landAsync(vehicle_name="DroneA").join()
        c.armDisarm(False, vehicle_name="DroneA")
        c.enableApiControl(False, vehicle_name="DroneA")
        status = "PASS"
    except Exception as exc:
        status = "FAIL"
        rows.append({"error": str(exc), "type": type(exc).__name__})
        raise
    finally:
        c.simPause(True)
        (folder / "report.json").write_text(json.dumps({"status": status, "base_z": locals().get("base"), "rows": rows}, indent=2), encoding="utf-8")
        print(folder)
        for label in ("hold", "positive", "negative"):
            values = [r["z"] for r in rows if r.get("phase") == label]
            if values:
                print(label, values[0], values[-1], "delta", values[-1] - values[0])


if __name__ == "__main__":
    main()
