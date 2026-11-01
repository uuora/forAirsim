"""Explicitly selected online protocol; opening it never starts a flight."""
from copy import deepcopy
import json,os
from PyQt5.QtWidgets import QComboBox,QLabel,QPushButton,QDoubleSpinBox,QSpinBox,QGroupBox,QFormLayout
from dual_target_control.ui import ControlWindow
from dual_target_control.route_view import RouteFlightView
from dual_target_control.online_curve import target_stream,stream_hash
from dual_target_control.curve_results import TITLES
from prepared_route_panel import prepared_config


def online_config(seed=1201,duration=25.,method='hermite_abom'):
    cfg=prepared_config();start=[0.,0.,5.];points=target_stream(start,seed)
    cfg['preparation']=dict(task=dict(start=start,points=points),protocol_revision=16)
    cfg['online']=dict(method=method,seed=seed,duration_s=duration,budget_s=.8,targets=points,target_stream_sha256=stream_hash(points),
        controller_revision='goal_bearing_lead_v3',yaw_lead_s=.18,
        protocol_revision=16,lookahead_targets=2,handoff_lead_s=1.3,population=8,max_candidates=32,
        curve_family='fifth_degree_endpoint_state_baseline_plus_sixth_degree_C2_bump',pass_speed_mps=1.2,
        target_generation='Precommitted bounded 6–9m steps, adjacent direction cosine >= -0.4; only current/next revealed to worker')
    auto=cfg['curve_experiment'].get('auto_show_scene',False)
    cfg['airsim']['initial_height']=5.;cfg['curve_experiment']=dict(time_scale=1.,auto_show_scene=auto)
    return cfg


class OnlineWindow(ControlWindow):
    def __init__(self,runtime,config):
        self.online_ready=False
        super().__init__(runtime,config)
        self.setWindowTitle('连续飞行 · 后台规划 · 协议16（开发验证）')
        old=self.view;parent=old.parentWidget();self.view=RouteFlightView(config,parent);parent.layout().replaceWidget(old,self.view);old.hide();old.deleteLater()
        self.method=QComboBox()
        for key,title in TITLES.items():self.method.addItem(title,key)
        self.method.setCurrentIndex(self.method.findData(config['online']['method']))
        self.duration=QSpinBox();self.duration.setRange(10,180);self.duration.setValue(int(config['online']['duration_s']));self.duration.setSuffix(' 秒')
        self.seed=QSpinBox();self.seed.setRange(0,1000000);self.seed.setValue(config['online']['seed'])
        self.budget=QDoubleSpinBox();self.budget.setRange(.2,1.);self.budget.setSingleStep(.1);self.budget.setValue(config['online']['budget_s']);self.budget.setSuffix(' 秒/次')
        self.start_spins=[];group=QGroupBox('连续在线任务（新协议，不与旧结果混排）');form=QFormLayout(group)
        form.addRow('选择方法',self.method);form.addRow('飞行时长',self.duration);form.addRow('固定目标种子',self.seed);form.addRow('后台墙钟预算',self.budget)
        for label,value in zip(('起点 X','起点 Y','起点高度'),config['preparation']['task']['start']):
            spin=QDoubleSpinBox();spin.setRange(-15,15) if label!='起点高度' else spin.setRange(4,10);spin.setValue(value);self.start_spins.append(spin);form.addRow(label,spin)
        label=QLabel('只给规划进程当前/下一点。起飞准备完成即执行后备参考，后台搜索不暂停飞行。目标标记没有物理碰撞含义。');label.setWordWrap(True);form.addRow(label)
        container=self.mode_combo.parentWidget();container.layout().insertWidget(0,group)
        self.online_group=group;self.pause_button.hide();self.overview_button.hide()
        self.start_button.setText('开始连续飞行');self.apply_button.setText('应用在线设置（不飞行）')
        self.result_button=QPushButton('本次轨迹与数据图');self.result_button.clicked.connect(self.open_result);container.layout().insertWidget(2,self.result_button)
        self.library_button=QPushButton('返回记录库 / 勾选对比');self.library_button.clicked.connect(self.open_library);container.layout().insertWidget(3,self.library_button)
        self._indexed=set();self.online_ready=True;self.refresh()

    def _apply(self):
        if not self.online_ready:return False
        if self.runtime.snapshot().get('running'):return False
        try:
            cfg=deepcopy(self.config);start=[s.value() for s in self.start_spins];seed=self.seed.value();points=target_stream(start,seed)
            cfg['preparation']['task']=dict(start=start,points=points);cfg['online'].update(method=self.method.currentData(),seed=seed,targets=points,
                target_stream_sha256=stream_hash(points),duration_s=float(self.duration.value()),budget_s=self.budget.value())
            cfg['airsim']['initial_height']=start[2]
            # Flight controls are frozen across methods in this development protocol.
            self.config=cfg;self.view.config=deepcopy(cfg);self.view._camera_signature=None
            self.runtime._publish(result_ready=False,result_page=None,online_metrics=None,planned_path=[],trail=[])
            self.runtime.configure(cfg);self._set_message('已提交在线任务设置；未开始飞行。');return True
        except Exception as ex:self._set_message(str(ex));return False

    def refresh(self):
        super().refresh()
        if not self.online_ready:return
        snap=self.runtime.snapshot();self.online_group.setEnabled(not snap.get('running') and not self._closing)
        self.result_button.setEnabled(bool(snap.get('result_page')))
        self.library_button.setEnabled(not snap.get('running') and not self._closing)
        folder=snap.get('run_dir')
        if folder and snap.get('result_ready') and not snap.get('running') and folder not in self._indexed:
            self._indexed.add(folder)
            import threading
            def index_completed():
                try:
                    from dual_target_control.experiment_store import ExperimentStore
                    from dual_target_control.simulator import ROOT
                    ExperimentStore(ROOT).index_paths([folder])
                except Exception as ex:self.runtime._publish(message='实验已保存；索引失败，请在数据页重新扫描：'+str(ex))
            threading.Thread(target=index_completed,name='online-result-index',daemon=True).start()
        self.mode_combo.setEnabled(False)
        for spin in self.inputs.values():spin.setEnabled(False)
        self.footer.setText('绿：已承诺参考 · 紫：实际飞行 · 当前/下一目标 · 后台不等于候选自动采用')

    def open_result(self):
        path=self.runtime.snapshot().get('result_page')
        if path and os.path.isfile(path):os.startfile(path)

    def open_library(self):
        if self.runtime.snapshot().get('running'):return
        import subprocess,sys
        from dual_target_control.simulator import ROOT
        from dual_target_control.experiment_store import ExperimentStore
        # Index only this completed run, not the full historical tree.
        folder=self.runtime.snapshot().get('run_dir')
        if folder:ExperimentStore(ROOT).index_paths([folder])
        subprocess.Popen([sys.executable,str(ROOT/'python/run_route_experiment_panel.py'),'--tab','data'],cwd=ROOT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));self.close()
