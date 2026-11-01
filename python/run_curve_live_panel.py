"""Six selectable references in the original03 Qt/runtime/observer framework."""
import copy,hashlib,json,os,queue,sys,threading
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import Qt,QUrl
from PyQt5.QtGui import QDesktopServices,QFont
from PyQt5.QtWidgets import (QApplication,QComboBox,QDialog,QFileDialog,QHBoxLayout,QLabel,QLineEdit,
    QListWidget,QListWidgetItem,QAbstractItemView,QPushButton,QSpinBox,QDoubleSpinBox,QVBoxLayout)
from dual_target_control.ui import ControlWindow,DEFAULT_CONFIG
from dual_target_control.curve_live import LiveCurveRuntime
from dual_target_control.curve_results import render_results
from curve_airsim_experiment_panel import TITLES,DEFAULT_PLANS,catalogue,curve_order_note

ROOT=Path(__file__).resolve().parents[1]
LIVE_DEFAULT_PLANS=ROOT/'reports/live_control_turn_case_1201/airsim_replay_plan.json'


def live_config(path=LIVE_DEFAULT_PLANS):
    preset=json.loads((ROOT/'configs/curve_live_protocol13.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/preset['base_config']).read_text(encoding='utf-8-sig'))
    if path==LIVE_DEFAULT_PLANS:path=ROOT/preset['plans']
    plans=catalogue(path);plan=next(p for p in plans if p['group_id']=='hermite_ordinary')
    cfg['airsim']['initial_height']=plan['boundary']['p0'][2]
    cfg['control'].update(max_speed=preset['max_speed_mps'],max_acceleration=preset['max_acceleration_mps2'])
    cfg['curve_experiment']=dict(plan=plan,plans_file=str(Path(path).resolve()),
        plans_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),time_scale=preset['reference_time_scale'],auto_show_scene=preset['auto_show_scene'])
    return cfg


class CurveLiveWindow(ControlWindow):
    def __init__(self,runtime,config):
        self.events=queue.Queue();self.planning_busy=False;self.planner=None;self.last_results=None
        super().__init__(runtime,config,config_path=ROOT/'configs/curve_live_protocol13.json')
        self.setWindowTitle('六组曲线控制实验 · 原03实时框架 · 协议13')
        for label in self.findChildren(QLabel):
            if label.objectName()=='heading':label.setText('六组曲线控制实验')
            if label.objectName()=='subtitle':label.setText('原03运行环境 / 实时控制 / 机头朝向 / 原生鼠标跟随')
        self.plans_path=Path(config['curve_experiment']['plans_file']);self.plans=catalogue(self.plans_path)
        bar=QHBoxLayout();self.method=QComboBox()
        for key,title in TITLES.items():self.method.addItem(title,key)
        self.method.setCurrentIndex(list(TITLES).index('hermite_ordinary'))
        self.method.currentIndexChanged.connect(self.choose_method)
        bar.addWidget(QLabel('方法'));bar.addWidget(self.method,1)
        self.time_scale=QDoubleSpinBox();self.time_scale.setRange(.5,2);self.time_scale.setSingleStep(.1);self.time_scale.setValue(config['curve_experiment']['time_scale'])
        self.time_scale.setToolTip('1.25表示参考时长除以1.25。加快后可能增加误差；六组比较须保持同一倍率。')
        self.acceleration=QDoubleSpinBox();self.acceleration.setRange(.5,15);self.acceleration.setValue(config['control']['max_acceleration'])
        bar.addWidget(QLabel('参考时间倍率'));bar.addWidget(self.time_scale)
        bar.addWidget(QLabel('加速度上限 m/s²'));bar.addWidget(self.acceleration)
        self.data_button=QPushButton('打开本次数据');self.data_button.clicked.connect(self.open_data);bar.addWidget(self.data_button)
        self.figure_button=QPushButton('查看曲线图');self.figure_button.clicked.connect(self.show_current_results);bar.addWidget(self.figure_button)
        compare=QPushButton('选择历史对比');compare.clicked.connect(self.compare_history);bar.addWidget(compare)
        outer=self.centralWidget().layout();outer.insertLayout(1,bar)
        generators=QHBoxLayout();self.goal=QLineEdit(', '.join(str(x) for x in config['curve_experiment']['plan']['boundary']['p1']));self.goal.setMaximumWidth(170)
        self.seed=QSpinBox();self.seed.setRange(0,2147483647);self.seed.setValue(config['curve_experiment']['plan']['seed'])
        generators.addWidget(QLabel('目标 x,y,高度'));generators.addWidget(self.goal)
        generators.addWidget(QLabel('种子'));generators.addWidget(self.seed)
        fixed=QPushButton('固定目标生成');fixed.clicked.connect(lambda:self.generate(False));generators.addWidget(fixed)
        random=QPushButton('随机目标生成');random.clicked.connect(lambda:self.generate(True));generators.addWidget(random)
        pick=QPushButton('加载已有方案');pick.clicked.connect(self.pick_plans);generators.addWidget(pick)
        self.path_label=QLabel();self.path_label.setWordWrap(True);generators.addWidget(self.path_label,1)
        outer.insertLayout(2,generators)
        self.footer.setText('青/绿线：参考；灰线：实际。UE4内左拖旋转、右拖平移、滚轮缩放；双击无人机聚焦，可切换跟随。')
        for button in self.findChildren(QPushButton):
            if button.text()=='六组控制实验：固定 / 随机目标':button.hide()
            if button.text()=='打开日志':button.setText('打开本次数据')
        self.logs_button.setText('打开本次数据')
        for label in self.metrics['planning_ms'].findChildren(QLabel):
            if label.objectName()=='caption':label.setText('方案生成（历史）')
        self.choose_method()
        screen=QApplication.primaryScreen()
        if screen:
            available=screen.availableGeometry();height=max(650,available.height()-28);width=max(980,available.width()-28)
            self.setMinimumSize(min(1100,width),min(800,height));self.resize(min(1360,width),min(970,height))

    def choose_method(self):
        plan=next(p for p in self.plans if p['group_id']==self.method.currentData())
        self.config['curve_experiment']['plan']=copy.deepcopy(plan)
        self.config['airsim']['initial_height']=plan['boundary']['p0'][2]
        self.path_label.setText(curve_order_note(plan)+f" · T={plan['parameters'][-1]:.3f}s · 目标{plan['boundary']['p1']}")

    def _apply(self):
        if self.planning_busy:self._set_message('请等待方案生成完成');return False
        self.choose_method()
        p0=self.config['curve_experiment']['plan']['boundary']['p0']
        centre=[(self.inputs[a].value()+self.inputs[b].value())/2 for a,b in [('x_min','x_max'),('y_min','y_max')]]
        if any(abs(a-b)>1e-6 for a,b in zip(p0[:2],centre)):
            self._set_message('参考起点须与场地水平中心一致；当前固定方案起点为'+str(p0));return False
        self.config['curve_experiment']['time_scale']=self.time_scale.value()
        self.config['control']['max_acceleration']=self.acceleration.value()
        return super()._apply()

    def pick_plans(self):
        if self.runtime.snapshot().get('running') or self.planning_busy:return
        path,_=QFileDialog.getOpenFileName(self,'加载完整六组方案',str(ROOT/'reports'),'JSON (*.json)')
        if path:self.accept_plans(path)

    def accept_plans(self,path):
        try:
            plans=catalogue(path)
            for plan in plans:
                if 'boundary' not in plan:raise ValueError('实时协议需要完整起终边界；请重新生成纯控制方案')
            self.plans_path=Path(path).resolve();self.plans=plans
            self.config['curve_experiment'].update(plans_file=str(self.plans_path),plans_sha256=hashlib.sha256(self.plans_path.read_bytes()).hexdigest())
            self.choose_method()
            plan=self.config['curve_experiment']['plan'];self.goal.setText(', '.join(str(x) for x in plan['boundary']['p1']));self.seed.setValue(plan['seed'])
            self._set_message('方案已加载，选择方法后点击开始飞行。')
        except Exception as exc:self._set_message(str(exc))

    def generate(self,random_target):
        if self.runtime.snapshot().get('running') or self.planning_busy:return
        try:
            from control_target_experiments import make_case
            seed=self.seed.value();goal=None if random_target else [float(v.strip()) for v in self.goal.text().replace('，',',').split(',')]
            make_case(seed,goal)
        except Exception as exc:self._set_message(str(exc));return
        self.planning_busy=True;self._set_message('后台生成六组参考，不启动飞行。')
        output=ROOT/'reports'/f'control_target_plans_live_{datetime.now():%Y%m%d_%H%M%S_%f}'
        def work():
            try:
                from control_target_planner_client import PlannerClient
                if self.planner is None:self.planner=PlannerClient()
                receipt=self.planner.generate(output,seed,goal);self.events.put(('plans',receipt['plan']))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def open_data(self):
        folder=self.runtime.snapshot().get('run_dir')
        path=Path(folder) if folder else ROOT/'logs'
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _open_logs(self):self.open_data()

    def show_current_results(self):
        folder=self.runtime.snapshot().get('run_dir')
        if not folder or not (Path(folder)/'summary.json').is_file():
            self._set_message('飞行结束并保存后可查看曲线；当前数据目录始终可打开。');return
        self.plot([Path(folder)])

    def plot(self,folders):
        self.figure_button.setEnabled(False);self._set_message('正在读取原始数据生成轨迹/误差/速度/朝向图。')
        output=ROOT/'reports'/f'live_curve_view_{datetime.now():%Y%m%d_%H%M%S_%f}'
        def work():
            try:self.events.put(('figure',str(render_results(folders,output))))
            except Exception as exc:self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def compare_history(self):
        dialog=QDialog(self);dialog.setWindowTitle('选择同条件的历史记录');dialog.resize(780,440)
        layout=QVBoxLayout(dialog);layout.addWidget(QLabel('勾选两条或更多记录；不同规则、航点、速度档、倍率、源码/协议将显示差异并拒绝混比。'))
        box=QListWidget();box.setSelectionMode(QAbstractItemView.ExtendedSelection);folders=[]
        for path in sorted((ROOT/'logs').glob('dual_target_*/live_manifest.json'),reverse=True):
            if not (path.parent/'summary.json').is_file():continue
            manifest=json.loads(path.read_text(encoding='utf-8'));folders.append(path.parent)
            route=manifest['plan'].get('route',{})
            item=QListWidgetItem(path.parent.name+' · '+route.get('policy_name','单目标')+' · '+TITLES.get(manifest['method'],manifest['method']))
            item.setFlags(item.flags()|Qt.ItemIsUserCheckable);item.setCheckState(Qt.Unchecked);box.addItem(item)
        layout.addWidget(box);button=QPushButton('比较选中记录');layout.addWidget(button)
        def selected():
            choices=[folder for i,folder in enumerate(folders) if box.item(i).checkState()==Qt.Checked]
            if len(choices)<2:return
            dialog.accept();self.plot(choices)
        button.clicked.connect(selected);dialog.exec_()

    def refresh(self):
        super().refresh()
        if not hasattr(self,'method'):return
        state=self._last_snapshot;idle=not state.get('running') and not self.planning_busy
        for widget in (self.method,self.time_scale,self.acceleration,self.goal,self.seed):widget.setEnabled(idle)
        if state.get('run_dir'):
            metrics=state.get('control_metrics') or {}
            outcome=(' 控制通过' if metrics.get('control_error_gates_passed') else ' 控制未通过') if metrics else ''
            self.footer.setText('本次数据：'+state['run_dir']+outcome+'  |  点击“查看曲线图”。')
        else:self.footer.setText('先启动仿真 → 选择方法 → 开始飞行；右上“打开本次数据”和“查看曲线图”直接查看日志及图表。')
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='plans':self.planning_busy=False;self.accept_plans(value)
                elif kind=='figure':
                    self.last_results=Path(value);self.figure_button.setEnabled(True)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_results)));self._set_message('曲线已生成：'+str(self.last_results.parent))
                else:self.planning_busy=False;self.figure_button.setEnabled(True);self._set_message(value)
        except queue.Empty:pass

    def closeEvent(self,event):
        if self.planner is not None:self.planner.close()
        super().closeEvent(event)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling,True)
    app=QApplication.instance() or QApplication(sys.argv);app.setFont(QFont('Microsoft YaHei',9));config=live_config()
    runtime=LiveCurveRuntime(config);window=CurveLiveWindow(runtime,config);window.show()
    return app.exec_()


if __name__=='__main__':raise SystemExit(main())
