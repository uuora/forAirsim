"""Diagnostic continuation only: paused controlled DroneB near the balloon."""
import json
import numpy as np
from run_balloon_mission import Experiment
from run_visual_fallback import visual_final_contact
from setup_scene import vector


def main():
    run = Experiment('visual_contact_resume_probe')
    c = run.client
    try:
        if not c.simIsPause() or not all(c.isApiControlEnabled(vehicle_name=n) for n in run.names):
            raise RuntimeError('Requires paused controlled two-drone diagnostic scene')
        for n in run.names:
            run.timestamps[n] = c.simGetCollisionInfo(vehicle_name=n).time_stamp
        run.report['diagnostic_resume'] = True
        run.change('start')
        run.active = 'DroneA'
        run.change('miss')
        run.change('fallback_ready')
        run.active = 'DroneB'
        run.holds = {'DroneA': np.array(vector(c.simGetObjectPose('DroneA').position))}
        c.simPause(False)
        visual_final_contact(run, 'DroneB', timeout_s=60)
        run.change('hit')
        run.report['hit_evidence'] = run.hit
        for n in (run.target, run.target + '_String'):
            if not c.simDestroyObject(n):
                raise RuntimeError('Failed to remove confirmed target')
        run.land()
        run.report['status'] = 'PASS'
    except BaseException as exc:
        run.report.update(status='FAIL', error=str(exc))
        raise
    finally:
        c.simPause(True)
        run.stream.close()
        (run.folder/'report.json').write_text(json.dumps(run.report, indent=2), encoding='utf-8')
        print(run.folder)


if __name__ == '__main__':
    main()
