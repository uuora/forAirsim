"""Two waypoint policies, explicit selected-method batches, original03 UI."""
import copy,hashlib,json,threading
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (QApplication,QCheckBox,QDialog,QDialogButtonBox,QDoubleSpinBox,QHBoxLayout,
    QLabel,QPushButton,QSpinBox,QTableWidget,QTableWidgetItem,QVBoxLayout)
from run_curve_live_panel import CurveLiveWindow,live_config,ROOT
from curve_airsim_experiment_panel import TITLES,catalogue
from dual_target_control.route_curve import POLICIES,route_spec
from dual_target_control.curve_campaign import CurveCampaign

PRESET=ROOT/'configs/curve_live_protocol14.json'
PHASE_NAMES={'READY':'待执行','EXECUTING':'执行中','STOPPED':'已停止','CLOSED':'已关闭','NEEDS_LANDING':'需要处理落地','COMPLETED':'全部结束'}


def route_live_config(policy='stop'):
    preset=json.loads(PRESET.read_text(encoding='utf-8'));cfg=live_config(ROOT/preset['packages'][policy])
    plan=cfg['curve_experiment']['plan'];cfg['curve_experiment']['time_scale']=plan['route']['time_scale']
    cfg['control'].update(max_speed=plan['route']['command_limits']['max_speed_mps'],max_acceleration=plan['route']['command_limits']['max_slew_mps2'])
    return cfg


class RouteExperimentWindow(CurveLiveWindow):
    def __init__(self,runtime,config):
        self.campaign=None;self.pending_batch=None;self.batch_reported=False;self.packages={};self.prepared_config=None;self.points=config['curve_experiment']['plan']['route']['waypoints']
        super().__init__(runtime,config)
        self.config_path=PRESET;self.setWindowTitle('六种方法 × 两组多点实验 · 原03实时控制 · 协议14')
        preset=json.loads(PRESET.read_text(encoding='utf-8'));self.packages={k:ROOT/v for k,v in preset['packages'].items()}
        from PyQt5.QtWidgets import QComboBox
        row=QHBoxLayout();self.policy=QComboBox()
        for key,title in POLICIES.items():self.policy.addItem(title,key)
        self.policy.currentIndexChanged.connect(self.switch_policy)
        row.addWidget(QLabel('实验组'));row.addWidget(self.policy)
        edit=QPushButton('编辑多点路线');edit.clicked.connect(self.edit_points);row.addWidget(edit)
        self.pass_speed=QDoubleSpinBox();self.pass_speed.setRange(.5,3);self.pass_speed.setValue(1.2)
        row.addWidget(QLabel('名义过点速度 m/s'));row.addWidget(self.pass_speed)
        self.dwell=QDoubleSpinBox();self.dwell.setRange(.5,5);self.dwell.setValue(1.)
        row.addWidget(QLabel('停稳组停留秒'));row.addWidget(self.dwell)
        batch=QPushButton('勾选批量运行');batch.clicked.connect(self.batch_dialog);row.addWidget(batch)
        stop=QPushButton('本次后停止批量');stop.clicked.connect(self.stop_batch);row.addWidget(stop)
        definitions=QPushButton('数据怎么读');definitions.clicked.connect(self.open_definitions);row.addWidget(definitions)
        self.centralWidget().layout().insertLayout(3,row)
        self.batch_label=QLabel('批量需明确勾选并点击启动；打开窗口不飞行。');self.batch_label.setWordWrap(True)
        self.centralWidget().layout().insertWidget(4,self.batch_label)
        self.batch_results=QPushButton('批量数据与对比');self.batch_results.clicked.connect(self.show_batch_results)
        self.centralWidget().layout().itemAt(2).layout().addWidget(self.batch_results)
        self.acceleration.setToolTip('发送的速度指令变化率上限；不是电机饱和或实测加速度硬限制。')
        for label in self.findChildren(QLabel):
            if label.text()=='加速度上限 m/s²':label.setText('指令变化率 m/s²')
        self.goal.setReadOnly(True)
        for label in self.findChildren(QLabel):
            if label.text()=='目标 x,y,高度':label.setText('多点路线')
        for button in self.findChildren(QPushButton):
            if button.text()=='固定目标生成':button.setText('生成当前组路线')
            if button.text()=='生成随机目标' or button.text()=='随机目标生成':button.setText('随机多点路线')
        self.stop_button.clicked.connect(self.stop_batch)
        self.update_points_text();self.choose_method()
        from dual_target_control.route_workspace import RouteWorkspace
        self.workspace=RouteWorkspace(self,ROOT)

    def compare_history(self):
        if not hasattr(self,'workspace'):return super().compare_history()
        library=self.workspace.library;library.filters['kind'].setCurrentIndex(library.filters['kind'].findData('flight'))
        self.workspace.tabs.setCurrentWidget(library)
        library.message.setText('勾选要比较的飞行记录，点击“生成所选对比”；不同规则和条件会分开展示。')

    def locked(self):return self.planning_busy or bool(self.campaign and self.campaign.active) or self.runtime.snapshot().get('running') or bool(hasattr(self,'workspace') and self.workspace.library.busy)

    def choose_method(self):
        super().choose_method()
        plan=self.config['curve_experiment']['plan']
        if 'route' in plan:
            spec=plan['route'];n=len(spec['waypoints']);status='简化模型通过' if plan.get('selection_provenance',{}).get('model_feasible') else '简化模型未通过'
            self.path_label.setText(spec['policy_name']+f" · {n}个航点 · {4*n}维搜索 · 五次基线/六次扰动族 · {status} · 种子{spec['seed']}")

    def update_points_text(self):
        self.goal.setText(str(len(self.points))+'个航点（点击编辑）')
        self.goal.setToolTip(' → '.join(str(p) for p in self.points))

    def current_spec(self,policy=None):
        cfg=self.edited_config();bounds=cfg['control'];all_points=[[0,0,5],*self.points]
        if abs(bounds['x_min']+bounds['x_max'])>1e-9 or abs(bounds['y_min']+bounds['y_max'])>1e-9:raise ValueError('多点实验起点固定(0,0,5)，场地水平中心须为(0,0)')
        for point in all_points:
            for value,low,high in zip(point,['x_min','y_min','height_min'],['x_max','y_max','height_max']):
                if not bounds[low]+bounds['margin']<=value<=bounds[high]-bounds['margin']:raise ValueError('起点或航点超出当前场地安全边界')
        return route_spec(self.points,policy or self.policy.currentData(),self.seed.value(),pass_speed=self.pass_speed.value(),
            dwell_s=self.dwell.value(),time_scale=self.time_scale.value(),max_speed=self.inputs['max_speed'].value(),max_slew=self.acceleration.value())

    def edited_config(self):
        cfg=copy.deepcopy(self.config);cfg['control'].update({key:spin.value() for key,spin in self.inputs.items()})
        cfg['control']['max_acceleration']=self.acceleration.value();cfg['curve_experiment']['time_scale']=self.time_scale.value()
        from dual_target_control.core import ControlConfig
        ControlConfig.from_dict(cfg['control'])
        return cfg

    def pick_plans(self):
        if not self.locked():super().pick_plans()

    def switch_policy(self):
        if not hasattr(self,'policy') or self.locked():return
        key=self.policy.currentData();path=self.packages.get(key)
        if path and path.is_file():
            plan=catalogue(path)[0]
            try:
                if plan['route']==self.current_spec(key):self.accept_plans(str(path));return
            except Exception as exc:self._set_message(str(exc));return
        self._set_message('当前组尚未生成匹配路线；点击“生成当前组路线”。')

    def edit_points(self):
        if self.locked():return
        dialog=QDialog(self);dialog.setWindowTitle('共同航点：两组及六种方法使用相同顺序');dialog.resize(650,470)
        layout=QVBoxLayout(dialog);layout.addWidget(QLabel('起点固定(0,0,5)，下表是依次到达的航点。最后一点始终停稳。'))
        table=QTableWidget(len(self.points),3);table.setHorizontalHeaderLabels(['X /米','Y /米','高度 /米'])
        for i,p in enumerate(self.points):
            for j,v in enumerate(p):table.setItem(i,j,QTableWidgetItem(str(v)))
        layout.addWidget(table);buttons=QHBoxLayout();add=QPushButton('增加航点');delete=QPushButton('删除选中航点')
        add.clicked.connect(lambda:table.insertRow(table.rowCount()));delete.clicked.connect(lambda:table.removeRow(table.currentRow()) if table.currentRow()>=0 else None)
        buttons.addWidget(add);buttons.addWidget(delete);layout.addLayout(buttons)
        error=QLabel();error.setWordWrap(True);layout.addWidget(error)
        ok=QPushButton('保存共同路线');layout.addWidget(ok)
        def accept():
            try:
                points=[[float(table.item(i,j).text()) for j in range(3)] for i in range(table.rowCount())]
                route_spec(points,'stop',self.seed.value());self.points=points;self.update_points_text();dialog.accept()
                self._set_message('航点已编辑。请生成当前组路线；批量入口会统一准备所选两组。')
            except Exception as exc:error.setText(str(exc))
        ok.clicked.connect(accept);dialog.exec_()

    def accept_plans(self,path):
        try:
            plans=catalogue(path)
            if any('route' not in p for p in plans):raise ValueError('这是旧单点方案；请在多点界面重新生成')
            super().accept_plans(path);spec=plans[0]['route'];self.points=spec['waypoints'];self.update_points_text()
            if hasattr(self,'policy'):
                self.policy.blockSignals(True);self.policy.setCurrentIndex(list(POLICIES).index(spec['policy']));self.policy.blockSignals(False)
                self.pass_speed.setValue(spec['pass_speed_mps']);self.dwell.setValue(spec['dwell_s']);self.time_scale.setValue(spec['time_scale'])
                limits=spec.get('command_limits',dict(max_speed_mps=6.,max_slew_mps2=6.))
                self.inputs['max_speed'].setValue(limits['max_speed_mps']);self.acceleration.setValue(limits['max_slew_mps2'])
                self.config['control'].update(max_speed=limits['max_speed_mps'],max_acceleration=limits['max_slew_mps2'])
                self.config['curve_experiment']['time_scale']=spec['time_scale']
            self.packages[spec['policy']]=Path(path)
            if self.runtime.snapshot().get('connected') and not self.runtime.snapshot().get('running'):self.runtime.configure(copy.deepcopy(self.config))
        except Exception as exc:self._set_message(str(exc))

    def _apply(self):
        if not hasattr(self,'policy'):return super()._apply()
        if hasattr(self,'workspace') and self.workspace.library.busy:self._set_message('数据任务完成后才能开始飞行或应用参数。');return False
        try:spec=self.current_spec()
        except Exception as exc:self._set_message(str(exc));return False
        if self.config['curve_experiment']['plan'].get('route')!=spec:
            self._set_message('航点、规则、速度或倍率已变化，请先生成匹配的多点方案。');return False
        return super()._apply()

    def generate(self,random_target):
        if self.locked():return
        if random_target:
            import numpy as np
            rng=np.random.default_rng(self.seed.value());points=[];angle=0.
            for i in range(max(2,len(self.points))):
                angle+=rng.uniform(.36,.5);points.append([15*np.sin(angle),15*(1-np.cos(angle)),5+float(rng.uniform(-.8,.8))])
            self.points=points;self.update_points_text()
        self.prepare([self.policy.currentData()])

    def prepare(self,policies):
        if self.planning_busy:return
        try:specs={p:self.current_spec(p) for p in policies};self.prepared_config=self.edited_config()
        except Exception as exc:self._set_message(str(exc));self.pending_batch=None;return
        cached={};missing={}
        for policy,spec in specs.items():
            path=self.packages.get(policy)
            if path and path.is_file() and catalogue(path)[0]['route']==spec:cached[policy]=str(path)
            else:missing[policy]=spec
        self.planning_busy=True
        if not missing:self.events.put(('routes_ready',cached));return
        self.batch_label.setText('后台生成多点方案；不飞行。每个实验组共用一套航点和预算。')
        def work():
            try:
                from control_target_planner_client import PlannerClient
                if self.planner is None:self.planner=PlannerClient()
                packages=dict(cached)
                for policy,s in missing.items():
                    output=ROOT/'reports'/f'route_plans_{policy}_{datetime.now():%Y%m%d_%H%M%S_%f}'
                    receipt=self.planner.generate_route(output,s['seed'],s['waypoints'],policy,pass_speed=s['pass_speed_mps'],dwell_s=s['dwell_s'],time_scale=s['time_scale'],
                        max_speed=s['command_limits']['max_speed_mps'],max_slew=s['command_limits']['max_slew_mps2'])
                    packages[policy]=receipt['plan']
                self.events.put(('routes_ready',packages))
            except Exception as exc:self.events.put(('route_error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def batch_dialog(self):
        if self.locked():return
        if not self.runtime.snapshot().get('connected'):self._set_message('请先启动并连接仿真，再选择批量。');return
        dialog=QDialog(self);dialog.setWindowTitle('明确选择批量：方法 × 两组规则 × 重复次数');layout=QVBoxLayout(dialog)
        layout.addWidget(QLabel('不会默认运行全部。请选择项目，再点击“启动所选实验”；每次到末点后保存并降落。'))
        method_boxes={};policy_boxes={}
        for key,title in TITLES.items():box=QCheckBox(title);layout.addWidget(box);method_boxes[key]=box
        all_button=QPushButton('全选六种方法');all_button.clicked.connect(lambda:[b.setChecked(True) for b in method_boxes.values()]);layout.addWidget(all_button)
        for key,title in POLICIES.items():box=QCheckBox(title);box.setChecked(key==self.policy.currentData());layout.addWidget(box);policy_boxes[key]=box
        count=QSpinBox();count.setRange(1,10);layout.addWidget(QLabel('每项到准备点后重新规划；重复同种子不等于独立种子' if 'preparation' in self.config else '相同冻结方案的执行重复次数（不是重新训练次数）'));layout.addWidget(count)
        description=QLabel('两组规则分别生成和比较；失败记录保留，落地成功后继续其余所选项目。');description.setWordWrap(True);layout.addWidget(description)
        launch=QPushButton('启动所选实验');layout.addWidget(launch)
        def start():
            methods=[k for k,b in method_boxes.items() if b.isChecked()];policies=[k for k,b in policy_boxes.items() if b.isChecked()]
            if not methods or not policies:description.setText('至少选择一种方法和一个实验组。');return
            self.pending_batch=dict(selections=[(p,m) for p in policies for m in methods],repeats=count.value())
            dialog.accept();self.prepare(policies)
        launch.clicked.connect(start);dialog.exec_()

    def stop_batch(self):
        self.pending_batch=None
        if self.campaign and self.campaign.active:self.campaign.stop_after_current()
        self.batch_label.setText('后续批量已取消；当前已开始的实验保留结果。')

    def open_definitions(self):
        from dual_target_control.metric_explanations import write_glossary
        from PyQt5.QtGui import QDesktopServices
        from PyQt5.QtCore import QUrl
        path=write_glossary(ROOT/'reports'/'control_data_help');QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def show_batch_results(self):
        if hasattr(self,'workspace') and self.workspace.library.busy:self._set_message('请等待当前数据任务完成。');return
        if self.campaign and self.campaign.active:self._set_message('批量结束或停止后可生成整批对比；当前原始日志持续保存。');return
        from PyQt5.QtWidgets import QFileDialog
        path=str(self.campaign.folder/'campaign.json') if self.campaign else None
        if not path:path,_=QFileDialog.getOpenFileName(self,'选择批量记录',str(ROOT/'reports'),'批量记录 (campaign.json)')
        if not path:return
        self.figure_button.setEnabled(False);self.batch_results.setEnabled(False)
        self._set_message('正在生成两组分别比较的数据页，包括失败和未运行项目。')
        def work():
            try:
                from dual_target_control.curve_results import render_campaign
                self.events.put(('batch_figure',str(render_campaign(path))))
            except Exception as exc:self.events.put(('batch_figure_error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def refresh(self):
        # Consume route-specific events before the legacy handler sees them.
        if hasattr(self,'events'):
            pending=[]
            while not self.events.empty():
                kind,value=self.events.get_nowait()
                if kind=='routes_ready':
                    self.planning_busy=False;self.packages.update({k:Path(v) for k,v in value.items()})
                    if self.pending_batch:
                        options=self.pending_batch;self.pending_batch=None
                        folder=ROOT/'reports'/f'route_campaign_{datetime.now():%Y%m%d_%H%M%S_%f}'
                        self.campaign=CurveCampaign(self.prepared_config,value,options['selections'],options['repeats'],folder);self.batch_reported=False
                    else:
                        key=self.policy.currentData();self.accept_plans(value.get(key,next(iter(value.values()))))
                        self.batch_label.setText('多点方案已生成；连接后可预览红/蓝实体，选择一种方法开始。')
                elif kind=='route_error':self.planning_busy=False;self.pending_batch=None;self._set_message(value);self.batch_label.setText(value)
                elif kind in ('batch_figure','batch_figure_error'):
                    self.batch_results.setEnabled(True)
                    if kind=='batch_figure' and hasattr(self,'workspace'):self.figure_button.setEnabled(True);self.workspace.show_results(Path(value))
                    else:pending.append(('figure' if kind=='batch_figure' else 'error',value))
                elif kind=='figure' and hasattr(self,'workspace'):
                    self.figure_button.setEnabled(True);self.workspace.show_results(Path(value))
                else:pending.append((kind,value))
            for item in pending:self.events.put(item)
        super().refresh()
        if not hasattr(self,'policy'):return
        if self.campaign and self.campaign.active:self.campaign.tick(self.runtime)
        active=bool(self.campaign and self.campaign.active);locked=active or self.planning_busy or self.runtime.snapshot().get('running')
        for widget in (self.policy,self.pass_speed,self.dwell,self.method,self.time_scale,self.acceleration,self.seed):widget.setEnabled(not locked)
        for widget in self.inputs.values():widget.setEnabled(not locked)
        if active:
            self.start_button.setEnabled(False);slot=self.campaign.slots[min(self.campaign.index,len(self.campaign.slots)-1)]
            self.batch_label.setText(f"批量 {self.campaign.index}/{len(self.campaign.slots)} · {POLICIES[slot['policy']]} · {TITLES[slot['method']]} · {PHASE_NAMES.get(self.campaign.phase,self.campaign.phase)}")
        elif self.campaign and not self.batch_reported:
            self.batch_reported=True;self.batch_label.setText('批量状态：'+PHASE_NAMES.get(self.campaign.phase,self.campaign.phase)+'；点击“批量数据与对比”。明细：'+str(self.campaign.folder/'campaign.json'))
        state=self.runtime.snapshot()
        if state.get('marker_status') and not self.campaign and not self.planning_busy:self.batch_label.setText(state['marker_status']+f" · 当前航点{min(state.get('route_index',0)+1,state.get('route_total',1))}/{state.get('route_total',1)}")
        if hasattr(self,'workspace'):self.workspace.refresh()

    def closeEvent(self,event):
        if hasattr(self,'workspace') and self.workspace.library.busy:
            from PyQt5.QtCore import QTimer
            self._set_message('正在完成数据写入或归档，完成后关闭。');event.ignore();QTimer.singleShot(250,self.close);return
        if hasattr(self,'workspace'):self.workspace.save()
        self.pending_batch=None
        if self.campaign and self.campaign.active:self.campaign.close()
        super().closeEvent(event)
