"""Recover only DroneA from a paused maritime single-drone observation run."""
import json
from datetime import datetime

import numpy as np

from maritime_single_experiment import MaritimeSingleExperiment, vec


def main():
    run = MaritimeSingleExperiment(record_prefix="maritime_single_recovery")
    c = run.client
    run.report["mode"] = "maritime_single_recovery"
    try:
        if not c.simIsPause():
            raise RuntimeError("Recovery requires a paused scene")
        if not c.isApiControlEnabled(vehicle_name="DroneA"):
            raise RuntimeError("DroneA API control is not held")
        if c.isApiControlEnabled(vehicle_name="DroneB"):
            raise RuntimeError("DroneB has API control; use coordinated recovery")
        a = c.getMultirotorState(vehicle_name="DroneA")
        b = c.getMultirotorState(vehicle_name="DroneB")
        ap, av = vec(a.kinematics_estimated.position), vec(a.kinematics_estimated.linear_velocity)
        bp, bv = vec(b.kinematics_estimated.position), vec(b.kinematics_estimated.linear_velocity)
        if (not np.isfinite(np.r_[ap, av, bp, bv]).all() or
                np.linalg.norm(ap[:2]) > 0.5 or not -7 < ap[2] < 0 or
                not 0.5 < bp[2] < 1.1 or np.linalg.norm(bv) >= 0.1):
            raise RuntimeError("Vehicles outside inspected recovery state")
        run.baseline_ground_z = float(bp[2])
        run.active = True
        run.report["recovery_start"] = {"DroneA_local_ned_m": ap.tolist(),
                                         "DroneB_local_ned_m": bp.tolist()}
        c.simPause(False)
        if not c.armDisarm(True, vehicle_name="DroneA"):
            raise RuntimeError("DroneA arm failed")
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
        (run.folder / "report.json").write_text(json.dumps(run.report,
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(run.folder)


if __name__ == "__main__":
    main()
