"""Ground-only real viewport framing and Qt asynchronous close verification."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import json
from pathlib import Path
import time
from PyQt5.QtWidgets import QApplication
from dual_target_control.runtime import ControlRuntime
from dual_target_control.simulator import ROOT, port_open
from dual_target_control.ui import ControlWindow


def main():
    app = QApplication([])
    path = ROOT / 'configs/dual_target_control.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    runtime = ControlRuntime(config, 'airsim')
    window = ControlWindow(runtime, config, path)
    shot = ROOT / 'reports/dual_target_control/native_full_field.png'
    previous_shot_time = shot.stat().st_mtime_ns if shot.exists() else 0
    # No start() call: this check cannot arm or take off.
    window.show(); window.request_scene(); runtime.start_simulator()
    requested = False
    result = {}
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            app.processEvents()
            snap = runtime.snapshot()
            if snap['status'] == 'ERROR' or snap.get('scene_status') == 'ERROR':
                raise RuntimeError(snap.get('scene_message') or snap['message'])
            if snap.get('scene_visible') and not requested and abs(snap['velocity'][2]) < .02:
                runtime._scene._queue.put(('screenshot', str(shot)))
                requested = True
            if requested and shot.exists() and shot.stat().st_mtime_ns > previous_shot_time:
                result.update(shot=str(shot), pid=snap.get('scene_pid'),
                    idle_speed=sum(v*v for v in snap['velocity'])**.5,
                    idle_position=snap['position'], running=snap['running'],
                    receipts=snap.get('scene_receipts'))
                break
            time.sleep(.025)
        else: raise TimeoutError('全景地面检查超时')
    finally:
        # Exercise the actual closeEvent -> owner shutdown -> lifecycle timer.
        window.close()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and (not runtime.wait_closed(0) or window.isVisible()):
            app.processEvents(); time.sleep(.025)
        result['closed'] = runtime.wait_closed(0) and not window.isVisible()
        result['engine_stopped'] = not port_open(config['airsim']['host'], config['airsim']['port'])
        result['scene_hidden'] = not runtime.snapshot().get('scene_visible')
        result['scene_message'] = runtime.snapshot().get('scene_message')
        result['scene_render_unknown'] = runtime.snapshot().get('scene_render_unknown')
        result['observer_alive'] = runtime._scene._thread.is_alive()
    result['pass'] = (result['closed'] and result['engine_stopped'] and result['scene_hidden']
                      and not result.get('running') and result.get('idle_speed', 1) < .02)
    output = ROOT / 'reports/dual_target_control/native_ground_validation.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__': raise SystemExit(main())
