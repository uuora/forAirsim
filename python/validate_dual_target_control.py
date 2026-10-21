"""Real desktop/runtime acceptance including pause/resume and actual screenshot."""
from __future__ import annotations
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from dual_target_control.runtime import ControlRuntime
from dual_target_control.simulator import ROOT
from dual_target_control.ui import ControlWindow
from dual_target_control.native_scene import reflected_command


def queue_transition_valid(before, after):
    """Viewing may span a legitimate hit; preserve queue order, not its age."""
    advanced = int(after['hits']) - int(before['hits'])
    if advanced == 0:
        return bool(np.allclose(before['targets'], after['targets']))
    if advanced == 1:
        return bool(np.allclose(before['targets'][1], after['targets'][0]))
    return False


def observer_state(config):
    """Read this native observer using an independent diagnostic connection."""
    import airsim
    options = config['airsim']
    client = airsim.MultirotorClient(ip=options['host'], port=options['port'], timeout_value=2.)
    log = ROOT / 'logs/dual_target_control_ue.log'
    try:
        offset = log.stat().st_size
        reflected_command(client, 'ke /Script/Blocks.DroneControlObserver DumpState', log)
        with log.open('rb') as stream:
            stream.seek(offset)
            rows = stream.read().decode('utf8', errors='replace').splitlines()
        return json.loads(next(row.split('DCO_STATE ', 1)[1] for row in rows if 'DCO_STATE ' in row))
    finally:
        client.client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('airsim', 'demo'), default='airsim')
    parser.add_argument('--seconds', type=float, default=90)
    parser.add_argument('--screenshot', type=Path, default=ROOT / 'reports/dual_target_control/airsim_running.png')
    parser.add_argument('--overview', action='store_true')
    parser.add_argument('--scene', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    config_path = ROOT / 'configs/dual_target_control.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    app = QApplication([])
    runtime = ControlRuntime(config, args.mode)
    window = ControlWindow(runtime, config, config_path)
    window.show()
    if args.overview: window.request_overview()
    result = dict(mode=args.mode, pause_queue_preserved=False, pause_speed=None, screenshot=None)
    if args.launch: runtime.start_simulator()
    runtime.start()
    begin = None
    pause_at = None
    pause_snapshot = None
    resumed = False
    last_print = 0
    overview_closed_at = None
    overview_reopened = False
    scene_shown = scene_hidden = scene_reopened = False
    scene_before_hide = None
    try:
        deadline = time.monotonic() + args.seconds + 70
        while time.monotonic() < deadline:
            app.processEvents()
            snap = runtime.snapshot()
            if snap['status'] == 'ERROR': raise RuntimeError(snap['message'])
            if snap['status'] == 'RUNNING' and begin is None:
                begin = time.monotonic()
            if begin is not None:
                elapsed = time.monotonic() - begin
                if args.scene:
                    if snap.get('scene_status') == 'ERROR':
                        raise RuntimeError(snap['scene_message'])
                    if elapsed >= 3 and not scene_shown:
                        runtime.show_scene(); scene_shown = True
                    if snap.get('scene_visible'):
                        result.setdefault('scene_pid', snap.get('scene_pid'))
                        result['scene_receipts'] = snap.get('scene_receipts')
                        stage = ('observer_after_reopen' if scene_reopened and elapsed >= 20
                                 else 'observer_during_flight' if elapsed >= 6 and not scene_hidden else None)
                        if stage and stage not in result:
                            result[stage] = observer_state(config)
                    if elapsed >= 13 and not scene_hidden and snap.get('scene_visible'):
                        scene_before_hide = snap
                        runtime.hide_scene(); scene_hidden = True
                    if scene_hidden and not snap.get('scene_visible') and not scene_reopened:
                        if 'scene_hide_keeps_control' not in result:
                            result['scene_hide_keeps_control'] = snap['status'] == 'RUNNING'
                            result['scene_hide_keeps_queue'] = queue_transition_valid(scene_before_hide, snap)
                            result['scene_hide_hit_advance'] = snap['hits'] - scene_before_hide['hits']
                        if elapsed >= 18:
                            runtime.show_scene(); scene_reopened = True
                    if scene_reopened and snap.get('scene_visible'):
                        result['scene_reopen_keeps_control'] = snap['status'] == 'RUNNING'
                        result['scene_same_process'] = snap.get('scene_pid') == result.get('scene_pid')
                if args.overview and window.overview_window is not None:
                    video = window.overview_window.stream.snapshot()
                    result['overview_frames'] = video['frames']
                    result['overview_status'] = video['status']
                    result['overview_rpc_ms'] = video.get('rpc_ms')
                    if video['frames'] >= 4 and not result.get('overview_early'):
                        shot = args.screenshot.with_name(args.screenshot.stem + '_map_early.png')
                        shot.parent.mkdir(parents=True, exist_ok=True)
                        window.overview_window.refresh()
                        window.overview_window.grab().save(str(shot))
                        result['overview_early'] = str(shot)
                    if elapsed >= 40 and overview_closed_at is None:
                        window.overview_window.close()
                        overview_closed_at = time.monotonic()
                if overview_closed_at and not overview_reopened and time.monotonic() - overview_closed_at > 1:
                    result['overview_close_keeps_control'] = runtime.snapshot()['status'] == 'RUNNING'
                    window._open_overview()
                    overview_reopened = True
                if elapsed - last_print >= 10:
                    print(json.dumps(dict(elapsed=round(elapsed, 1), status=snap['status'], hits=snap['hits']), ensure_ascii=False), flush=True)
                    last_print = elapsed
                if elapsed >= 8 and pause_at is None:
                    pause_snapshot = snap
                    runtime.pause()
                    pause_at = time.monotonic()
                if pause_at and not resumed and time.monotonic() - pause_at >= 2:
                    result['pause_queue_preserved'] = (snap['hits'] == pause_snapshot['hits']
                            and np.allclose(snap['targets'], pause_snapshot['targets']))
                    result['pause_speed'] = float(np.linalg.norm(snap['velocity']))
                    if result['pause_speed'] < .3:
                        result['pause_settle_s'] = time.monotonic() - pause_at
                        runtime.resume()
                        resumed = True
                    elif time.monotonic() - pause_at >= 5:
                        raise RuntimeError('暂停 5 秒内未减速至 0.3 m/s。')
                # Short legs can hit the third target while the view is being
                # hidden. Export only after reopen, when rendering is stable.
                if (snap['hits'] >= 3 and not result['screenshot']
                        and (not args.scene or (scene_reopened and snap.get('scene_visible')))):
                    window.refresh()
                    app.processEvents()
                    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                    if not window.grab().save(str(args.screenshot)): raise RuntimeError('截图保存失败')
                    result['screenshot'] = str(args.screenshot)
                    if args.scene and snap.get('scene_visible'):
                        shot = args.screenshot.with_name(args.screenshot.stem + '_ue.png')
                        # Export from the actual main UE viewport, not an external camera.
                        runtime._scene._queue.put(('screenshot', str(shot)))
                        result['scene_screenshot'] = str(shot)
                    if args.overview and window.overview_window is not None:
                        shot = args.screenshot.with_name(args.screenshot.stem + '_map.png')
                        window.overview_window.refresh()
                        window.overview_window.grab().save(str(shot))
                        result['overview_screenshot'] = str(shot)
                if elapsed >= args.seconds:
                    break
            time.sleep(.01)
        else: raise TimeoutError('验收超过时间限制')
    finally:
        runtime.stop()
        runtime.shutdown()
        deadline = time.monotonic() + 65
        while time.monotonic() < deadline and not runtime.wait_closed(0):
            app.processEvents()
            time.sleep(.03)
        if not runtime.wait_closed(0):
            raise RuntimeError('落地未完成，控制线程保留；不要强行关闭仿真。')
        window.close()
        app.processEvents()
    snap = runtime.snapshot()
    run = Path(snap['run_dir'])
    rows = [json.loads(line) for line in (run / 'telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    positions = np.asarray([r['position'] for r in rows])
    lo = np.array([config['control'][key] for key in ('x_min', 'y_min', 'height_min')])
    hi = np.array([config['control'][key] for key in ('x_max', 'y_max', 'height_max')])
    result.update(hits=snap['hits'], samples=len(rows), state_min=positions.min(0).tolist(),
                  state_max=positions.max(0).tolist(), run_dir=str(run),
                  within_bounds=bool(np.all(positions >= lo) and np.all(positions <= hi)),
                  stopped=snap['status'] == 'STOPPED',
                  cleanup=json.loads((run / 'cleanup.json').read_text(encoding='utf-8')) if args.mode == 'airsim' else {})
    result['pass'] = (result['hits'] >= 3 and result['within_bounds'] and result['stopped']
                      and result['pause_queue_preserved'] and result['pause_speed'] < .3)
    if args.overview:
        result['pass'] = result['pass'] and result.get('overview_frames', 0) > 10 and result.get('overview_close_keeps_control', False)
    if args.scene:
        result['scene_final_hidden'] = not snap.get('scene_visible')
        observations = [result.get(key, {}) for key in
                        ('observer_during_flight', 'observer_after_reopen')]
        result['observer_follows_actual_drone'] = all(
            state.get('enabled') and state.get('follow')
            and state.get('viewtarget') == 'DC_MouseObserver'
            and np.linalg.norm(np.asarray(state.get('pivot', [1e9]*3))
                               - np.asarray(state.get('drone_world', [0]*3))) < 50.
            # The camera manager caches the previous render tick; at 4 m/s,
            # one 120 Hz tick already exceeds 1 cm. Check visible centering as
            # well as a bounded cache lag rather than requiring simultaneity.
            and np.linalg.norm(np.asarray(state.get('camera', [1e9]*3))
                               - np.asarray(state.get('manager_location', [0]*3))) < 50.
            and state.get('drone_projected')
            and np.linalg.norm(np.asarray(state.get('drone_screen', [1e9]*2))
                               - np.asarray(state.get('viewport', [0]*2)) / 2.) < 5.
            for state in observations)
        result['pass'] = (result['pass'] and result.get('scene_hide_keeps_control', False)
                          and result.get('scene_hide_keeps_queue', False)
                          and result.get('scene_reopen_keeps_control', False)
                          and result.get('scene_same_process', False)
                          and result['observer_follows_actual_drone']
                          and result['scene_final_hidden']
                          and Path(result.get('scene_screenshot', '')).is_file())
    (run / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result['pass'] else 1


if __name__ == '__main__': raise SystemExit(main())
