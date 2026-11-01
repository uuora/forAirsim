"""Default interactive multi-point experiment entry; opens idle."""
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from prepared_route_panel import PreparedRouteWindow as RouteExperimentWindow,prepared_config as route_live_config
from dual_target_control.prepared_route_live import PreparedRouteRuntime as RouteLiveRuntime


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--tab',choices=['control','data','plots'],default='control');parser.add_argument('--online',action='store_true');args=parser.parse_args()
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    app=QApplication(sys.argv);app.setFont(QFont('Microsoft YaHei',9));config=route_live_config()
    if args.online:
        from online_curve_panel import online_config,OnlineWindow
        from dual_target_control.online_live import OnlineRuntime
        config=online_config();runtime=OnlineRuntime(config);window=OnlineWindow(runtime,config)
    else:
        runtime=RouteLiveRuntime(config);window=RouteExperimentWindow(runtime,config);window.workspace.tabs.setCurrentIndex(['control','data','plots'].index(args.tab))
    window.show();raise SystemExit(app.exec_())
