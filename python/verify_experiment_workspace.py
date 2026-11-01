"""Offline Qt workspace acceptance: existing recorded flights, no simulator I/O."""
import argparse,json,os,time,hashlib
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from unittest.mock import Mock
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from route_experiment_panel import RouteExperimentWindow,route_live_config,ROOT
from dual_target_control.route_live import RouteLiveRuntime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/qt_workspace_v2');parser.add_argument('--render',action='store_true');args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True);app=QApplication([]);app.setFont(QFont('Microsoft YaHei',9))
    cfg=route_live_config();runtime=RouteLiveRuntime(cfg,mode='demo');runtime.start=Mock();runtime.start_simulator=Mock();runtime.connect=Mock()
    window=RouteExperimentWindow(runtime,cfg);window.resize(1360,900);window.show();receipt={}
    def settle(timeout=180):
        deadline=time.monotonic()+timeout;quiet=0
        while quiet<5:
            app.processEvents();time.sleep(.025)
            quiet=0 if window.workspace.library.busy else quiet+1
            if time.monotonic()>deadline:raise TimeoutError('离线数据/绘图任务超时')
    try:
        settle()
        if args.render:
            campaign=json.loads((ROOT/'reports/route_protocol14_all12_v1/campaign.json').read_text(encoding='utf-8'));paths={str(Path(s['run_dir']).resolve()) for s in campaign['slots']}
            lib=window.workspace.library;lib.model.checked={r['id'] for r in lib.model.rows if r['path'] in paths}
            assert len(lib.model.checked)==12;lib.compare();settle();assert window.last_results and window.last_results.is_file(),lib.message.text()
            receipt['selected_flights']=12;receipt['result_page']=str(window.last_results)
        for i,name in enumerate(['control','library','plots']):
            window.workspace.tabs.setCurrentIndex(i)
            if name=='library':window.workspace.library.filters['kind'].setCurrentIndex(window.workspace.library.filters['kind'].findData('flight'))
            if name=='plots' and window.workspace.gallery.figures:
                window.workspace.gallery.choice.setCurrentIndex(len(window.workspace.gallery.figures)-1)
            settle();window.grab().save(str(output/(name+'.png')))
        runtime.start.assert_not_called();runtime.start_simulator.assert_not_called();runtime.connect.assert_not_called();assert runtime.snapshot()['status']=='IDLE' and runtime._run_dir is None
        receipt.update(status='PASS',ui_geometry=[window.width(),window.height()],catalog_records=len(window.workspace.library.model.rows),flight_calls=0,simulator_launch_calls=0,
            scope='Offscreen native Qt UI + offline rendering of saved AirSim records, not a new flight or simulator screenshot')
        (output/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt,ensure_ascii=False))
    finally:
        settle();window.timer.stop();runtime.shutdown();assert runtime.wait_closed(5);window.close();app.processEvents()

if __name__=='__main__':main()
