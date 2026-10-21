"""Offline UI guards or native ground-only presentation. Never starts a mission."""
import sys,os,time
from pathlib import Path
from unittest.mock import patch
root=Path(sys.argv[1]).resolve();native='--native' in sys.argv
if not native:os.environ['QT_QPA_PLATFORM']='offscreen'
from fixed_map_runtime import configure_session
configure_session(root/('native_interface_check' if native else 'offline_interface_check'))
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer,QSharedMemory
from PyQt5.QtGui import QFont
from fixed_map_dashboard import Dashboard,CameraWorker
from fixed_map_perception_execution import PerceptionExecutor
from fixed_map_perception import save,utc
app=QApplication([]);app.setFont(QFont('Microsoft YaHei',10))

if not native:
    with patch.object(PerceptionExecutor,'start'),patch.object(CameraWorker,'start'):
        w=Dashboard();w.timer.stop();w.mission_timer.stop();w.e.stage='READY';w.refresh()
        assert w.perception_config['schema']=='yolo_language_cpu_v11' and w.perception_config['ground_development_only']
        assert not w.buttons['start'].isEnabled() and not w.observe_button.isEnabled()
        assert w.instruction.isVisibleTo(w) and w.mission_timer.interval()==50
        w.show_reference_replay();assert w.replay_preview is not None and 'YOLO' in w.replay_preview['preview']['task']['source_kind']
        assert w.e.active_plan is None and w.e.service is None and w.replay_preview['flight_authorized'] is False
        assert '不是当前在线结果' in w.perception_status.toPlainText()
        with patch('fixed_map_perception_process.PerceptionProcess.start',side_effect=ValueError('MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB')):
            w.start_perception();assert w.e.service is None and w.perception is None
        w.e.planner.shutdown(wait=False);w.close()
    save(root/'qt_offline_v9.json',dict(status='PASS',checks=7,mode='OFFLINE_NO_RPC_NO_FLIGHT'));print('PASS 7 UI guards');sys.exit(0)

lock=QSharedMemory('FixedMapMissionDroneA20261009')
if not lock.create(1):raise ValueError('ANOTHER_INTERFACE_RUNNING')
w=Dashboard();w.show();deadline=time.monotonic()+20
def inspect():
    if w.e.stage=='CHECKING' and time.monotonic()<deadline:QTimer.singleShot(250,inspect);return
    if w.e.stage!='READY' or not w.e.state.get('landed'):raise ValueError('NATIVE_GROUND_NOT_READY')
    w.show_reference_replay();app.processEvents();w.grab().save(str(root/'dashboard_YOLO_saved_preview_REPLAY.png'))
    # RPC clients own thread-local transport; never call the executor's client
    # from a Qt callback in the main thread.
    import airsim
    probe=airsim.MultirotorClient(port=41454,timeout_value=3)
    save(root/'qt_native_v9.json',dict(status='PASS_PRESENTATION_ONLY',mode='REAL_GROUND_PLUS_SAVED_YOLO_PREVIEW_NOT_LIVE_DETECTION',state=w.e.state,DroneA_api_control=probe.isApiControlEnabled(vehicle_name='DroneA'),active_plan=w.e.active_plan,preview_source=w.replay_preview['source'],flight_buttons_disabled=not w.buttons['start'].isEnabled() and not w.observe_button.isEnabled()))
    w.timer.stop();w.mission_timer.stop();w.camera_worker.enabled=False;w.camera_worker.overview_enabled=False
    w.camera_worker.requestInterruption();w.e.exit_requested=True;w.e.join(timeout=3)
    if not w.camera_worker.wait(5000):raise ValueError('NATIVE_CAMERA_CLEANUP_TIMEOUT')
    w.close();app.exit(0)
QTimer.singleShot(4000,inspect);sys.exit(app.exec_())
