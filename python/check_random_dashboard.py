"""Offline Qt wiring/replay check. No RPC, scenario change, or flight."""
import os,sys,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
os.environ['QT_QPA_PLATFORM']='offscreen'
from fixed_map_runtime import configure_session
root=Path(sys.argv[1]).resolve();configure_session(root/'ui_offline_check')
from PyQt5.QtWidgets import QApplication,QSplitter,QGroupBox
from fixed_map_perception_execution import PerceptionExecutor
from fixed_map_dashboard import Dashboard,CameraWorker
from fixed_map_perception import read,save
app=QApplication([])
from PyQt5.QtGui import QFont
app.setFont(QFont('Microsoft YaHei',10));checks=[]
with patch.object(PerceptionExecutor,'start'),patch.object(CameraWorker,'start'):
    w=Dashboard();w.timer.stop();w.e.stage='READY';w.refresh()
    assert not w.scene_baseline_available and not w.buttons['start'].isEnabled();checks.append('random scene disables fixed baseline')
    assert w.scene_random.isEnabled() and w.scene_restore.isEnabled();checks.append('ground scene controls enabled')
    assert w.perception_config['capture_after_model_load'];checks.append('v6 model-first initialization')
    with patch.object(w.e,'command') as command:
        w.action('start');command.assert_not_called();checks.append('baseline action guarded')
    assert not w.preview_button.isEnabled() and not w.perception_execute_button.isEnabled();checks.append('missing estimate rejected')
    row=read(root/'online_base_v6/frames/frame-000266/perception.json');row.pop('overlay_path',None)
    w.perception=SimpleNamespace(latest=row,cfg=w.perception_config,process=SimpleNamespace(poll=lambda:None));w.e.service=w.perception
    w.refresh();assert 'CURRENT_FRAME_EXPIRED' in w.perception_status.toPlainText();checks.append('recorded frame stays expired')
    assert w.onboard.receipt is not None and not w.perception_execute_button.isEnabled();checks.append('raw RGB fallback and no false valid target')
    assert not w.scene_random.isEnabled() and not w.scene_restore.isEnabled();checks.append('active sensing prevents scene mutation')
    w.e.service=None;w.perception=None;w.e.stage='APPROACH';w.refresh()
    assert not w.scene_random.isEnabled() and not w.scene_restore.isEnabled();checks.append('airborne scene mutation disabled')
    w.e.stage='READY';w.e.exit_requested=True;w.e.planner.shutdown(wait=False)
    w.perception_status.setPlainText('回放显示检查：本图为已保存机载RGB，不代表当前在线数据。白条红球未形成有效指定目标确认。')
    w.show();app.processEvents();w.grab().save(str(root/'realtime_planning_audit/dashboard_REPLAY_check.png'))
    sizes=[s.sizes() for s in w.findChildren(QSplitter)]
    w.close();checks.append('Qt renders resized panels')
save(root/'realtime_planning_audit/dashboard_offline_check.json',dict(status='PASS',mode='OFFLINE_UI_REPLAY_NO_RPC_NO_FLIGHT',checks=checks,splitter_sizes=sizes))
print(json.dumps(dict(status='PASS',checks=len(checks),splitter_sizes=sizes)))
