"""Explicit GUI/AirSim display check; production entry still opens idle."""
import argparse,json,re,time
from pathlib import Path
from copy import deepcopy
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from prepared_route_panel import PreparedRouteWindow,prepared_config,ROOT
from dual_target_control.prepared_route_live import PreparedRouteRuntime
from dual_target_control.native_scene import reflected_command
from dual_target_control.curve_audit import audit_run


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    app=QApplication([]);cfg=prepared_config();runtime=PreparedRouteRuntime(cfg);window=PreparedRouteWindow(runtime,cfg)
    window.setWindowTitle('显示核验 · 真实AirSim · 两次明确选择的方法');window.show()
    for spin,value in zip(window.preparation_spins,[-3,1,5]):spin.setValue(value)
    window.policy.setCurrentIndex(window.policy.findData('pass'))
    state=dict(phase='CONNECT',runs=[],audits=[],deadline=time.monotonic()+180,last_status=None)
    camera_client=None

    def save():
        (output/'verification.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')

    def camera():
        nonlocal camera_client
        if camera_client is None:
            import airsim
            camera_client=airsim.MultirotorClient(ip=cfg['airsim']['host'],port=cfg['airsim']['port'],timeout_value=2.)
        log=ROOT/'logs/dual_target_control_ue.log';offset=log.stat().st_size
        reflected_command(camera_client,'ke /Script/Blocks.DroneControlObserver DumpState',log)
        with log.open('rb') as stream:stream.seek(offset);text=stream.read().decode('utf-8',errors='replace')
        return json.loads(re.findall(r'DCO_STATE (\{[^\r\n]+\})',text)[-1])

    def launch(method,phase):
        if window.locked():return
        window.method.setCurrentIndex(window.method.findData(method));window._start()
        state.update(phase=phase,deadline=time.monotonic()+180);save()

    def tick():
        try:
            snap=runtime.snapshot();status=snap['status']
            if status!=state['last_status']:print(state['phase'],status,flush=True);state['last_status']=status
            if status=='ERROR':
                state['failure_run_dir']=snap.get('run_dir');raise RuntimeError(snap['message'])
            if time.monotonic()>state['deadline']:raise TimeoutError('核验阶段超时：'+state['phase'])
            phase=state['phase']
            if phase=='CONNECT' and status=='READY' and snap.get('scene_visible'):
                state['initial_camera']=camera();state['scene_identity']=id(runtime._scene)
                launch('hermite_ordinary','FIRST')
            elif phase in ('FIRST','SECOND') and status=='STOPPED' and snap.get('run_dir') and snap['run_dir'] not in state['runs']:
                folder=Path(snap['run_dir']);state['runs'].append(str(folder));state['audits'].append(audit_run(folder))
                display=json.loads((folder/'route_display.json').read_text(encoding='utf-8'))
                assert [m['state'] for m in display['markers'][1:-1]]==['visited','visited']
                assert state['audits'][-1]['control_passed']
                window.grab().save(str(output/(phase.lower()+'_qt_actual.png')))
                if phase=='FIRST':
                    state.update(phase='INSPECT',deadline=time.monotonic()+70)
                    state['camera_before_manual']=camera();save();print('READY_FOR_CAMERA_INSPECTION',flush=True)
                else:
                    state['camera_after_second']=camera()
                    baseline=state['camera_before_second'];after=state['camera_after_second']
                    state['camera_preserved']=all(baseline[k]==after[k] for k in ('pivot','camera','yaw','elevation','distance','follow'))
                    state['observer_reused']=id(runtime._scene)==state['scene_identity']
                    assert state['camera_preserved'] and state['observer_reused']
                    state.update(phase='PASS',deadline=time.monotonic()+90);save();print('PASS',flush=True);timer.stop();window.close()
            elif phase=='INSPECT' and ((output/'continue_second').exists() or time.monotonic()>state['deadline']-10):
                if window.locked():return
                state['camera_before_second']=camera()
                state['manual_wheel_observed']=state['camera_before_second']['wheel']>state['camera_before_manual']['wheel']
                launch('bezier_ordinary','SECOND')
        except Exception as exc:
            state.update(phase='FAILED',error=str(exc));save();print('FAILED',exc,flush=True);timer.stop();runtime.stop();window.close()

    timer=QTimer();timer.setInterval(100);timer.timeout.connect(tick);timer.start();runtime.start_simulator()
    try:app.exec_()
    finally:
        runtime.stop();runtime.shutdown()
        if not runtime.wait_closed(90):raise RuntimeError('落地或关闭未完成')
        if camera_client is not None:camera_client.client.close()
    if state['phase']!='PASS':raise RuntimeError(state.get('error','核验未完成'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.output)
