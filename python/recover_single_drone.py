"""Recover DroneA from a paused single-drone experiment without commanding DroneB."""
import json
from datetime import datetime

import airsim
import numpy as np

from setup_scene import ROOT, vector
from single_drone_experiment import SingleDroneExperiment


def main():
    run = SingleDroneExperiment("visual", record_prefix="single_recovery")
    run.report["mode"] = "single_drone_recovery"
    run.report["source_failure"] = "operator_inspected_paused_state"
    c = run.client
    try:
        if not c.simIsPause():
            raise RuntimeError("Recovery requires a paused simulation")
        if not c.isApiControlEnabled(vehicle_name="DroneA"):
            raise RuntimeError("DroneA API control is not held; inspect state manually")
        if c.isApiControlEnabled(vehicle_name="DroneB"):
            raise RuntimeError("DroneB API control is held; use coordinated recovery")
        b, bs, _ = run._bystander_preflight()
        if not (np.isfinite(b).all() and 0.65 < b[2] < 1.1 and bs < 0.08):
            raise RuntimeError("DroneB is not stationary on the ground")
        p = np.asarray(vector(c.simGetObjectPose("DroneA").position))
        if not np.isfinite(p).all() or not (-6 < p[0] < 0 and -5 < p[1] < 1
                                             and -4.2 < p[2] < 1.1):
            raise RuntimeError("DroneA outside bounded recovery region")
        run.bystander_start = b
        run.collision_stamp = c.simGetCollisionInfo(vehicle_name="DroneA").time_stamp
        run.armed = True
        run.report["recovery_start_world_ned_m"] = p.tolist()
        c.simPause(False)
        if not c.armDisarm(True, vehicle_name="DroneA"):
            raise RuntimeError("Could not arm DroneA for recovery")
        run.land()
        run.report["status"] = "PASS"
    except BaseException as exc:
        run.report.update(status="FAIL", error=str(exc), manual_recovery_required=True)
        raise
    finally:
        try:
            c.simPause(True)
            run.report["paused"] = True
        except Exception as exc:
            run.report.update(status="FAIL", paused=False, pause_error=str(exc))
        run.report["finished_at"] = datetime.now().astimezone().isoformat()
        (run.folder / "report.json").write_text(
            json.dumps(run.report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(run.folder)


if __name__ == "__main__":
    main()
