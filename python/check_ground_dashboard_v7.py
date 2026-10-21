"""Offline Qt isolation checks, plus optional native read-only ground screenshot."""
import os,sys,json,time
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]).resolve();native=len(sys.argv)>2 and sys.argv[2]=='--native'
if not native:os.environ['QT_QPA_PLATFORM']='offscreen'
from fixed_map_runtime import configure_session
configure_session(root/('qt_native_ground' if native else 'qt_offline_ground'))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer,QSharedMemory
from fixed_map_dashboard import Dashboard,CameraWorker
from fixed_map_perception_execution import PerceptionExecutor
from fixed_map_perception import save,utc
app=QApplication([]);app.setFont(QFont('Microsoft YaHei',10))
if native:
    lock=QSharedMemory('FixedMapMissionDroneA20261009')
    if not lock.create(1):raise ValueError('ANOTHER_DASHBOARD_IS_RUNNING')
    w=Dashboard();w.show();deadline=time.monotonic()+20
    def inspect():
        if w.e.stage=='CHECKING' and time.monotonic()<deadline:QTimer.singleShot(300,inspect);return
        if w.e.stage!='READY' or not w.e.state.get('landed'):raise ValueError('NATIVE_GROUND_STATE_NOT_READY')
        w.show_reference_replay();app.processEvents()
        if w.replay_preview is None:raise ValueError('NATIVE_REPLAY_NOT_LOADED')
        w.grab().save(str(root/'dashboard_native_ground_REPLAY.png'))
        save(root/'dashboard_native_ground_receipt.json',dict(utc=utc(),status='PASS',scope='actual ground state plus explicit historic replay display; no model or motion',state=w.e.state,api_control=w.e.c.isApiControlEnabled(vehicle_name='DroneA'),active_execution_plan=w.e.active_plan,replay_label=w.onboard.title(),flight_start_disabled=not w.observe_button.isEnabled() and not w.buttons['start'].isEnabled(),perception_execute_disabled=not w.perception_execute_button.isEnabled()))
        w.close();app.quit()
    QTimer.singleShot(3000,inspect);sys.exit(app.exec_())
with patch.object(PerceptionExecutor,'start'),patch.object(CameraWorker,'start'):
    w=Dashboard();w.timer.stop();w.e.stage='READY';w.refresh()
    assert w.perception_config['ground_development_only'] and not w.observe_button.isEnabled() and not w.buttons['start'].isEnabled()
    with patch.object(w.e,'command') as command:
        w.action('start');w.prepare_observation();command.assert_not_called()
    w.show_reference_replay();assert w.replay_preview is not None
    assert w.e.active_plan is None and w.e.service is None and w.map.preview_path
    assert '回放' in w.onboard.title() and '不是新v7结果' in w.perception_status.toPlainText()
    w.refresh();assert not w.preview_button.isEnabled() and not w.perception_execute_button.isEnabled()
    assert not w.camera_worker.enabled
    with patch('fixed_map_perception_process.PerceptionProcess.start',side_effect=ValueError('MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB')),patch('fixed_map_scene_binding.bind_scene_config',return_value=w.perception_config):
        w.start_perception();w.refresh();assert w.perception is None and w.e.service is None
        assert 'MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB' in w.perception_status.toPlainText()
    w.e.exit_requested=True;w.e.planner.shutdown(wait=False);w.close()
save(root/'dashboard_offline_receipt.json',dict(utc=utc(),status='PASS',checks=8,mode='OFFLINE_QT_NO_RPC_NO_FLIGHT',scope=['ground profile','start actions guarded','historic replay loads','no active plan/service','explicit old-v4 labels','live controls disabled','camera cannot overwrite replay','memory rejection keeps GUI usable and service detached']))
print('PASS 8 offline Qt ground/replay checks')
