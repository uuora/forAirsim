"""Selected-method interface for protocol15; task editing never runs a planner."""
import copy,json,uuid
from datetime import datetime
from PyQt5.QtWidgets import QDialog,QVBoxLayout,QLabel,QPushButton,QDoubleSpinBox,QFormLayout,QGroupBox,QTableWidget,QTableWidgetItem,QHBoxLayout,QScrollArea
from route_experiment_panel import RouteExperimentWindow,route_live_config,ROOT
from prepared_route_planner import prepared_spec
from route_control_experiments import plan_from_vector
from control_target_experiments import GROUPS
from dual_target_control.curve_campaign import CurveCampaign


def task_from_spec(spec):
    return dict(points=spec['waypoints'],start=spec['start'],policy=spec['policy'],seed=spec['seed'],
        pass_speed=spec['pass_speed_mps'],dwell_s=spec['dwell_s'],time_scale=spec['time_scale'],
        max_speed=spec['command_limits']['max_speed_mps'],max_slew=spec['command_limits']['max_slew_mps2'])


def task_template(spec,method):
    # Placeholder for legacy UI/campaign metadata only; never decoded or flown.
    plan=plan_from_vector([0.,0.,0.,6.]*len(spec['waypoints']),spec,method.split('_')[0])
    plan.update(group_id=method,planning_pending=True)
    return plan


def prepared_config():
    cfg=route_live_config();spec=cfg['curve_experiment']['plan']['route']
    cfg['preparation']=dict(task=task_from_spec(spec),method='hermite_ordinary',protocol_revision=15)
    return cfg


class PreparedRouteWindow(RouteExperimentWindow):
    def __init__(self,runtime,config):
        self.prepared_ready=False
        super().__init__(runtime,config)
        self.setWindowTitle('无人机控制实验 · 到准备点后规划 · 协议15')
        from dual_target_control.route_view import RouteFlightView
        old=self.view;frame=old.parentWidget();self.view=RouteFlightView(config,frame)
        self.view.setMinimumSize(350,200);frame.layout().replaceWidget(old,self.view);old.hide();old.deleteLater()
        self.preparation_spins=[]
        group=QGroupBox('起飞后先到准备点（米，高度向上）');form=QFormLayout(group)
        for label,value in zip(('准备点 X','准备点 Y','准备点高度'),config['preparation']['task']['start']):
            spin=QDoubleSpinBox();spin.setDecimals(2);spin.setRange(3,23) if label.endswith('高度') else spin.setRange(-45,45)
            spin.setValue(value);self.preparation_spins.append(spin);form.addRow(label,spin)
        note=QLabel('到达并停稳 → 仅规划所选方法 → 执行参考。准备和悬停不计入跟踪误差，规划耗时单独保存。');note.setWordWrap(True);form.addRow(note)
        settings=self.workspace.findChildren(QGroupBox)
        advanced=next(w for w in settings if w.title().startswith('场地边界'))
        advanced.parentWidget().layout().insertWidget(1,group)
        help_button=QPushButton('六种方法改了什么 / 通过标准');help_button.clicked.connect(self.method_help);form.addRow(help_button)
        self.online_button=QPushButton('连续飞行 · 后台规划（协议16开发版）');self.online_button.clicked.connect(self.open_online);form.addRow(self.online_button)
        for old,new in [('生成当前组路线','应用任务设置'),('随机多点路线','随机航点（不规划）'),('加载已有方案','载入旧任务坐标')]:
            self.workspace.buttons[old].setText(new)
        self.start_button.setText('到准备点 → 规划 → 飞行')
        self.source.hide();self.scene_status.hide()
        self.dynamics_status.setMaximumHeight(19)
        for key in ('speed','altitude','distance','hits','yaw_error_deg'):
            self.metrics[key].setMinimumHeight(54);self.metrics[key].setMaximumHeight(58)
        self.footer.hide();self.message.setMaximumHeight(22)
        self.batch_label.setWordWrap(False);self.batch_label.setMaximumHeight(18)
        from PyQt5.QtWidgets import QSizePolicy
        self.batch_label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        self.prepared_ready=True;self.choose_method();self.view.set_plane('3d')
        self.footer.setText('绿线：规划后参考；亮紫线：实际飞行；橙色方框：准备点。等比例视角中同样的米数对应相同像素。')
        self.batch_label.setText('协议15：准备点可修改。打开窗口、切换方法和应用任务均不起飞、不规划。')

    def current_spec(self,policy=None):
        if not self.prepared_ready:return super().current_spec(policy)
        cfg=self.edited_config();start=[s.value() for s in self.preparation_spins]
        spec=prepared_spec(self.points,policy or self.policy.currentData(),self.seed.value(),start=start,
            pass_speed=self.pass_speed.value(),dwell_s=self.dwell.value(),time_scale=self.time_scale.value(),
            max_speed=self.inputs['max_speed'].value(),max_slew=self.acceleration.value())
        bounds=cfg['control']
        for point in [start,*self.points]:
            for value,low,high in zip(point,('x_min','y_min','height_min'),('x_max','y_max','height_max')):
                if not bounds[low]+bounds['margin']<=value<=bounds[high]-bounds['margin']:raise ValueError('准备点或航点超出当前场地安全边界')
        centre=[(bounds['x_min']+bounds['x_max'])/2,(bounds['y_min']+bounds['y_max'])/2,start[2]]
        # Real takeoff remains at field centre before moving to the selected point.
        if any(abs(v)>45 for v in centre[:2]):raise ValueError('起飞中心超出已验证区域')
        return spec

    def open_online(self):
        if self.locked():self._set_message('先停止并降落，再打开连续在线实验。');return
        import subprocess,sys
        subprocess.Popen([sys.executable,str(ROOT/'python/run_route_experiment_panel.py'),'--online'],cwd=ROOT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.close()  # switch idle workbenches; do not leave two flight UIs active

    def choose_method(self):
        if not getattr(self,'prepared_ready',False):return super().choose_method()
        if not self.runtime.snapshot().get('running'):
            # Display reset only; raw flight evidence remains in its own directory.
            self.runtime._publish(planned_path=[],trail=[],route_markers=[])
        kind='普通五次基线' if self.method.currentData().endswith('ordinary') else 'ABOM + 配对扰动候选' if self.method.currentData().endswith('candidate') else '基础 ABOM 网络适配'
        self.path_label.setText(f'{len(self.points)}个航点 · 每段4维（形状3+时长1） · {kind}\n尚未规划；到准备点后重新计算。')

    def switch_policy(self):
        if not getattr(self,'prepared_ready',False):return
        if not self.locked():self.choose_method();self._set_message('已选择任务组；到准备点后按所选方法规划。')

    def _configured(self,spec,method):
        cfg=self.edited_config();cfg['preparation']=dict(task=task_from_spec(spec),method=method,protocol_revision=15)
        cfg['airsim']['initial_height']=spec['start'][2]
        cfg['curve_experiment'].update(plan=task_template(spec,method),time_scale=spec['time_scale'],execution_id=uuid.uuid4().hex)
        return cfg

    def _apply(self):
        if not getattr(self,'prepared_ready',False):return super()._apply()
        if self.locked():self._set_message('当前正在飞行、准备或处理数据，请完成后再修改。');return False
        try:
            spec=self.current_spec();cfg=self._configured(spec,self.method.currentData())
            self.runtime.configure(copy.deepcopy(cfg));self.config=cfg;self.choose_method()
            self._set_message('任务已应用；点击开始后真实起飞，到准备点再规划。');return True
        except Exception as exc:self._set_message(str(exc));return False

    def generate(self,random_target):
        if not self.prepared_ready:return super().generate(random_target)
        if self.locked():return
        if random_target:
            import numpy as np
            rng=np.random.default_rng(self.seed.value());s=np.array([w.value() for w in self.preparation_spins]);points=[]
            for i in range(max(2,len(self.points))):
                angle=.42*(i+1);points.append((s+[12*np.sin(angle),12*(1-np.cos(angle)),rng.uniform(-.7,.7)]).tolist())
            self.points=points;self.update_points_text()
        self._apply()

    def prepare(self,policies):
        if not self.prepared_ready:return super().prepare(policies)
        try:
            specs={p:self.current_spec(p) for p in policies}
            if not self.pending_batch:self._apply();return
            options=self.pending_batch;self.pending_batch=None
            folder=ROOT/'reports'/f'route_campaign_{datetime.now():%Y%m%d_%H%M%S_%f}'
            templates=ROOT/'experiment_data'/'task_templates'/folder.name;templates.mkdir(parents=True,exist_ok=False)
            packages={}
            for policy,spec in specs.items():
                path=templates/(policy+'.json');path.write_text(json.dumps(dict(plans=[task_template(spec,m) for m in GROUPS]),ensure_ascii=False,indent=2),encoding='utf-8');packages[policy]=path
            cfg=self._configured(specs[policies[0]],self.method.currentData())
            self.campaign=CurveCampaign(cfg,packages,options['selections'],options['repeats'],folder);self.batch_reported=False
            self.batch_label.setText('已排入明确选择的实验；每项重新到准备点、重新规划、独立保存。')
        except Exception as exc:self.pending_batch=None;self._set_message(str(exc))

    def batch_dialog(self):
        # The inherited dialog is an explicit chooser; only its task description changes.
        return super().batch_dialog()

    def edit_points(self):
        if not self.prepared_ready:return super().edit_points()
        if self.locked():return
        dialog=QDialog(self);dialog.setWindowTitle('共同航点：六种方法使用相同任务');dialog.resize(600,430);layout=QVBoxLayout(dialog)
        layout.addWidget(QLabel('起点由准备点设置决定；最后一个航点统一停稳。'))
        table=QTableWidget(len(self.points),3);table.setHorizontalHeaderLabels(['X /米','Y /米','高度 /米']);layout.addWidget(table)
        for i,p in enumerate(self.points):
            for j,v in enumerate(p):table.setItem(i,j,QTableWidgetItem(str(v)))
        row=QHBoxLayout();add=QPushButton('增加航点');remove=QPushButton('删除选中');row.addWidget(add);row.addWidget(remove);layout.addLayout(row)
        add.clicked.connect(lambda:table.insertRow(table.rowCount()));remove.clicked.connect(lambda:table.removeRow(table.currentRow()) if table.currentRow()>=0 else None)
        error=QLabel();error.setWordWrap(True);layout.addWidget(error);save=QPushButton('保存坐标（不规划）');layout.addWidget(save)
        def accept():
            old=self.points
            try:
                self.points=[[float(table.item(i,j).text()) for j in range(3)] for i in range(table.rowCount())]
                self.current_spec();self.update_points_text();self.choose_method();dialog.accept()
            except Exception as exc:self.points=old;error.setText(str(exc))
        save.clicked.connect(accept);dialog.exec_()

    def accept_plans(self,path):
        if not getattr(self,'prepared_ready',False):return super().accept_plans(path)
        try:
            document=json.loads(__import__('pathlib').Path(path).read_text(encoding='utf-8'));spec=document['plans'][0]['route']
            self.points=spec['waypoints']
            for spin,value in zip(self.preparation_spins,spec['start']):spin.setValue(value)
            self.policy.setCurrentIndex(self.policy.findData(spec['policy']));self.seed.setValue(spec['seed'])
            self.pass_speed.setValue(spec['pass_speed_mps']);self.dwell.setValue(spec['dwell_s']);self.time_scale.setValue(spec['time_scale'])
            self.inputs['max_speed'].setValue(spec['command_limits']['max_speed_mps']);self.acceleration.setValue(spec['command_limits']['max_slew_mps2'])
            self.update_points_text();self.choose_method();self._set_message('仅载入任务坐标及条件；旧轨迹不执行，到准备点后重新规划。')
        except Exception as exc:self._set_message(str(exc))

    def method_help(self):
        from dual_target_control.metric_explanations import METHOD_SUMMARY
        dialog=QDialog(self);dialog.setWindowTitle('方法差异与评判标准');dialog.resize(740,590);layout=QVBoxLayout(dialog)
        label=QLabel(METHOD_SUMMARY);label.setWordWrap(True);label.setOpenExternalLinks(False);label.setMinimumWidth(500)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(label);layout.addWidget(scroll)
        close=QPushButton('知道了');close.clicked.connect(dialog.accept);layout.addWidget(close);dialog.exec_()

    def refresh(self):
        super().refresh()
        if not getattr(self,'prepared_ready',False):return
        locked=self.locked()
        self.dynamics_status.setToolTip(self.source.text()+'\n'+self.dynamics_status.text()+'\n'+self.scene_status.text())
        self.source.hide();self.scene_status.hide()
        self.batch_label.setToolTip(self.batch_label.text())
        for spin in self.preparation_spins:spin.setEnabled(not locked)
        status=self.runtime.snapshot().get('status')
        if status in ('MOVING_TO_START','PLANNING','SETTLING'):
            self.pause_button.setEnabled(False)
            self.batch_label.setText(self.runtime.snapshot().get('message',''))
