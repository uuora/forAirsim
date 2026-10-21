"""Manual scene/task/flight controls for the existing Qt research workbench."""
import json,sys
from pathlib import Path
from PyQt5.QtCore import Qt,QProcess,QTimer
from PyQt5.QtGui import QPainter,QPen,QColor,QPixmap
from PyQt5.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QLineEdit,QComboBox,QTextEdit,QFileDialog,QSplitter,QGroupBox,QScrollArea
from t1_obstacle_scene import ROOT,load_scene,route,compile_task,box
from t1_manual_runtime import stamp,save

class SceneMap(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.setMinimumSize(340,280);self.scene=None;self.position=None;self.path=[];self.trace=[]
    def set_scene(self,scene):
        self.scene=scene;self.path=route(scene)['waypoints_ned_m'];self.position=scene['start_hover_ned_m'];self.trace=[];self.update()
    def live(self,position):self.position=position;self.trace.append(position);self.update()
    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#f8fafc'));p.setRenderHint(QPainter.Antialiasing)
        if self.scene is None:p.drawText(self.rect(),Qt.AlignCenter,'请选择场景配置');return
        lo=self.scene['workspace']['low_ned_m'];hi=self.scene['workspace']['high_ned_m'];pad=40
        scale=min((self.width()-2*pad)/(hi[0]-lo[0]),(self.height()-2*pad)/(hi[1]-lo[1]))
        def xy(v):return (pad+(v[0]-lo[0])*scale,self.height()-pad-(v[1]-lo[1])*scale)
        def line(a,b):
            x,y=xy(a);u,v=xy(b);p.drawLine(int(x),int(y),int(u),int(v))
        p.setPen(QPen(QColor('#ccd3dd'),1));p.drawRect(pad,int(self.height()-pad-(hi[1]-lo[1])*scale),int((hi[0]-lo[0])*scale),int((hi[1]-lo[1])*scale))
        for b in self.scene['buildings']:
            a,z=box(b,self.scene['clearance_m']);x,y=xy([a[0],z[1]]);w=(z[0]-a[0])*scale;h=(z[1]-a[1])*scale
            p.setPen(QPen(QColor('#c2b39f'),1,Qt.DashLine));p.setBrush(QColor('#f5ede1'));p.drawRect(int(x),int(y),int(w),int(h))
            a,z=box(b);x,y=xy([a[0],z[1]]);p.setPen(QColor('#657386'));p.setBrush(QColor('#a8b3c1'));p.drawRect(int(x),int(y),int((z[0]-a[0])*scale),int((z[1]-a[1])*scale));p.drawText(int(x),int(y)-4,b['name'].replace('T1Building_','建筑 '))
        p.setPen(QPen(QColor('#285f9e'),2,Qt.DashLine))
        for a,b in zip(self.path,self.path[1:]):line(a,b)
        p.setPen(QPen(QColor('#20936d'),2))
        for a,b in zip(self.trace,self.trace[1:]):line(a,b)
        for label,point,color in [('目标',self.scene['target']['position_ned_m'],'#bd4646'),('观察点',self.scene['observation_goal_ned_m'],'#285f9e'),('无人机',self.position,'#20866a')]:
            x,y=xy(point);p.setPen(QColor(color));p.setBrush(QColor(color));p.drawEllipse(int(x)-4,int(y)-4,8,8);p.drawText(int(x)+7,int(y)+4,label)
        p.setPen(QColor('#344054'));p.drawText(12,20,'俯视图 · NED米 · 虚线为规划，绿线为实际采样轨迹')
        p.drawText(12,self.height()-9,'建筑外虚框：配置净空；当前采用已知静态地图')
        p.end()

class ManualFlightWorkbench(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.scene_path=ROOT/'configs/t1_building_scene_v2.json';self.scene=None
        self.session_path=None;self.task_path=None;self.plan_path=None;self.run_folder=None;self.busy=False;self.buffer=b'';self.action=None;self.runner_buffer=b''
        self.process=QProcess(self);self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output);self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        outer=QVBoxLayout(self);top=QLabel('控制闭环 · 人工操作台');top.setStyleSheet('font-size:20px;font-weight:600');outer.addWidget(top)
        self.status=QLabel('未启动飞行。先查看任务与场景，再单独启动。');outer.addWidget(self.status)
        split=QSplitter(Qt.Horizontal);outer.addWidget(split,1);left=QWidget();ll=QVBoxLayout(left)
        self.map=SceneMap();ll.addWidget(self.map,1);self.camera=QLabel('飞行启动后显示实时保存的机载RGB');self.camera.setAlignment(Qt.AlignCenter);self.camera.setMinimumSize(300,200);ll.addWidget(self.camera,1);split.addWidget(left)
        right=QWidget();rl=QVBoxLayout(right)
        scene_box=QGroupBox('1  场景');s=QVBoxLayout(scene_box);self.path_label=QLabel();self.path_label.setWordWrap(True);s.addWidget(self.path_label)
        row=QHBoxLayout();self.choose=QPushButton('选择场景JSON');self.preview=QPushButton('刷新地图');row.addWidget(self.choose);row.addWidget(self.preview);s.addLayout(row)
        self.scene_start=QPushButton('启动建筑场景（地面）');self.resume_scene=QPushButton('载入已有场景会话');self.scene_close=QPushButton('安全关闭场景');s.addWidget(self.scene_start);s.addWidget(self.resume_scene);s.addWidget(self.scene_close);rl.addWidget(scene_box)
        task_box=QGroupBox('2  任务输入');t=QVBoxLayout(task_box)
        self.instruction=QLineEdit('靠近带白色横条的红色气球，保持3米高度，避开建筑物');t.addWidget(self.instruction)
        self.compile=QPushButton('生成任务预览');t.addWidget(self.compile);self.task_text=QTextEdit();self.task_text.setReadOnly(True);self.task_text.setMaximumHeight(155);t.addWidget(self.task_text)
        t.addWidget(QLabel('当前为本地受约束解析；后续LLM输出使用同一任务格式。'));rl.addWidget(task_box)
        flight_box=QGroupBox('3  闭环验证');f=QVBoxLayout(flight_box)
        self.mode=QComboBox();self.mode.addItem('基础闭环 · 独立UE会话','baseline');self.mode.addItem('新建筑场景 · 低速短验证','building');f.addWidget(self.mode)
        self.condition=QComboBox();self.condition.addItem('零门控 · 8秒','zero');self.condition.addItem('运动先导 / 独立重复 · 160秒','scale50');f.addWidget(self.condition)
        self.prepare=QPushButton('生成并检查本次计划');self.fly=QPushButton('启动本次闭环飞行');self.stop=QPushButton('停止目标动作并落地');self.stop.setStyleSheet('QPushButton:enabled{background:#b84646;color:white;} QPushButton:disabled{background:#edf1f5;color:#9aa8b6;}');f.addWidget(self.prepare);f.addWidget(self.fly);f.addWidget(self.stop)
        self.run_label=QLabel('每次运行使用新目录，停止请求和失败记录保留。');self.run_label.setWordWrap(True);f.addWidget(self.run_label);rl.addWidget(flight_box)
        self.console=QTextEdit();self.console.setReadOnly(True);self.console.document().setMaximumBlockCount(500);self.console.setMinimumHeight(100);rl.addWidget(self.console,1)
        for button in right.findChildren(QPushButton):button.setMinimumHeight(34)
        for label in right.findChildren(QLabel):label.setWordWrap(True)
        self.instruction.setMinimumHeight(36);self.task_text.setMinimumHeight(100)
        right.setMinimumHeight(840)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(right);scroll.setMinimumWidth(390)
        split.addWidget(scroll);split.setSizes([740,420])
        f.removeWidget(self.stop)
        footer=QHBoxLayout();footer.addWidget(QLabel('飞行期间停止入口始终可见；请求后等待清零、停稳与落地。'),1);footer.addWidget(self.stop)
        outer.addLayout(footer)
        self.choose.clicked.connect(self.choose_scene);self.preview.clicked.connect(self.refresh);self.compile.clicked.connect(self.compile_task)
        self.scene_start.clicked.connect(lambda:self.start('scene-start',['--scene',str(self.scene_path)]))
        self.scene_close.clicked.connect(self.close_scene);self.prepare.clicked.connect(self.prepare_plan);self.fly.clicked.connect(self.start_flight);self.stop.clicked.connect(self.request_stop)
        self.resume_scene.clicked.connect(self.resume_session)
        self.mode.currentIndexChanged.connect(self.invalidate_plan);self.condition.currentIndexChanged.connect(self.invalidate_plan);self.instruction.textChanged.connect(self.invalidate_task)
        self.refresh();self.buttons()
        self.timer=QTimer(self);self.timer.setInterval(250);self.timer.timeout.connect(self.poll_runner);self.timer.start();self._stdout_offset=0
    def buttons(self):
        for b in (self.choose,self.preview,self.compile,self.prepare,self.resume_scene):b.setEnabled(not self.busy)
        self.scene_start.setEnabled(not self.busy and not self.session_path)
        self.scene_close.setEnabled(not self.busy and self.session_path is not None)
        self.fly.setEnabled(not self.busy and self.plan_path is not None)
        self.stop.setEnabled(self.busy and self.action=='run')
        self.mode.setEnabled(not self.busy);self.condition.setEnabled(not self.busy);self.instruction.setEnabled(not self.busy)
    def has_running_job(self):return self.busy
    def invalidate_plan(self,*_):self.plan_path=None;self.buttons()
    def invalidate_task(self,*_):self.task_path=None;self.invalidate_plan()
    def refresh(self):
        try:
            self.scene=load_scene(self.scene_path);self.map.set_scene(self.scene);self.path_label.setText(str(self.scene_path));self.status.setText('场景地图已检查；尚未启动UE或飞行。');self.invalidate_task()
        except Exception as e:self.status.setText(str(e));self.scene=None
    def choose_scene(self):
        p,_=QFileDialog.getOpenFileName(self,'选择静态建筑场景',str(ROOT/'configs'),'JSON (*.json)')
        if p:self.scene_path=Path(p);self.refresh()
    def resume_session(self):
        path,_=QFileDialog.getOpenFileName(self,'载入本工具的场景会话',str(ROOT/'logs'),'场景会话 (scene_session.json)')
        if not path:return
        try:
            from t1_manual_runtime import assert_owner
            value=json.loads(Path(path).read_text(encoding='utf-8'));assert_owner(value)
            if value['status'] not in ('SCENE_READY_GROUND_ONLY','FLIGHT_ENDED_SEE_REPORT','SCENE_PREPARATION_FAILED'):raise ValueError('该会话正在执行或已经关闭。')
            self.scene_path=Path(value['scene_snapshot']);self.refresh();self.session_path=Path(path)
            self.status.setText('已载入所属场景；飞行已结束的会话可安全关闭。');self.buttons()
        except Exception as e:self.status.setText('载入失败：'+str(e))
    def compile_task(self):
        try:
            task=compile_task(self.instruction.text(),self.scene)
            self.task_path=ROOT/'logs'/('t1_manual_task_'+stamp())/'task.json';save(self.task_path,task)
            self.task_text.setPlainText('目标：'+task['target_description']+'\n高度：3米\n速度上限：0.03米/秒\n避障：开启（静态地图）\n状态：仅预览，未授权起飞\n'+str(self.task_path))
            self.status.setText('任务预览已保存；飞行需单独启动。')
        except Exception as e:self.status.setText(str(e));self.task_path=None
    def start(self,action,args):
        if self.busy:return
        self.action=action;self.busy=True;self.buffer=b'';self.status.setText('正在执行：'+action);self.buttons()
        self.process.setWorkingDirectory(str(ROOT));self.process.start(str(ROOT/'.venv/python.exe'),['-u',str(ROOT/'python/t1_manual_runtime.py'),action]+args)
    def prepare_plan(self):
        args=['--condition',self.condition.currentData()]
        if self.mode.currentData()=='building':
            if not self.session_path or not self.task_path:self.status.setText('请先启动建筑场景并生成任务预览。');return
            args+=['--scene',str(self.scene_path),'--session',str(self.session_path),'--task',str(self.task_path)]
        self.start('prepare',args)
    def start_flight(self):
        if not self.plan_path:return
        self.run_folder=ROOT/'logs'/('t1_manual_run_'+stamp());self.run_folder.mkdir();self._stdout_offset=0;self.runner_buffer=b''
        args=['--plan',str(self.plan_path),'--output',str(self.run_folder/'flight')]
        if self.mode.currentData()=='building':args+=['--session',str(self.session_path)]
        self.run_label.setText(str(self.run_folder));self.start('run',args)
    def request_stop(self):
        if self.busy and self.action=='run' and self.run_folder:
            signal=self.run_folder/'manual_stop.request'
            if not signal.exists():signal.write_text('OPERATOR_STOP\n',encoding='utf-8')
            self.status.setText('停止请求已写入；等待控制子进程清零、核查停稳和落地。');self.stop.setEnabled(False)
    def close_scene(self):
        if self.session_path:self.start('scene-close',['--session',str(self.session_path)])
    def process_error(self,*_):
        if self.process.state()==QProcess.NotRunning:self.busy=False;self.status.setText('运行程序未能启动。');self.buttons()
    def read_output(self):
        self.buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.buffer:
            line,self.buffer=self.buffer.split(b'\n',1);self.handle_line(line.decode('utf-8','replace'))
    def handle_line(self,line):
        try:row=json.loads(line)
        except (ValueError,TypeError):
            if line.strip():self.console.append(line[:1500])
            return
        kind=row.get('kind')
        if kind=='PLAN_READY':self.plan_path=Path(row['plan']);self.status.setText('计划已冻结并核查；点击飞行按钮启动。')
        elif kind=='SCENE_SESSION':self.session_path=Path(row['path'])
        elif kind=='SCENE_READY':self.session_path=Path(row['path']);self.status.setText('建筑场景已就绪，尚未起飞。')
        elif kind=='SCENE_CLOSED':self.session_path=None;self.invalidate_plan()
        elif kind=='LIVE_FRAME':
            self.status.setText(('初始化' if row['acquisition'] else '主测量')+f"槽 {row['slot']} · {row['gate_status']}")
            self.map.live(row['position_ned_m']);pix=QPixmap(row['rgb_path'])
            if not pix.isNull():self.camera.setPixmap(pix.scaled(self.camera.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
            return
        elif kind=='MANUAL_ACTION_FAILED':self.status.setText(row.get('message','操作失败。'))
        self.console.append(json.dumps(row,ensure_ascii=False)[:1800]);self.buttons()
    def poll_runner(self):
        # The baseline PowerShell launcher redirects the runner's JSON stream.
        if not self.busy or self.action!='run' or not self.run_folder:return
        path=self.run_folder/'attempt/runner.stdout.log'
        if not path.is_file():return
        with path.open('rb') as f:
            f.seek(self._stdout_offset);data=f.read();self._stdout_offset+=len(data)
        if data:
            self.runner_buffer+=data
            while b'\n' in self.runner_buffer:
                line,self.runner_buffer=self.runner_buffer.split(b'\n',1);self.handle_line(line.decode('utf-8','replace'))
    def finished(self,code,*_):
        self.read_output();self.busy=False
        if self.action=='run':
            self.plan_path=None;self.status.setText('飞行已结束，请查看保存报告。' if code==0 else '本次未通过或人工停止，结果与收尾记录已保留。')
        elif code!=0:self.status.setText(self.status.text()+'（未通过）')
        self.buttons()
