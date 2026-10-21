"""Manual UI for the fixed map executor. No autonomous start on application launch."""
import sys,json,time,os
from pathlib import Path
from datetime import datetime
from PyQt5.QtCore import Qt,QTimer,QPointF,QThread,pyqtSignal
from PyQt5.QtGui import QPainter,QPen,QColor,QPolygonF,QPixmap
from PyQt5.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QTextEdit,QPlainTextEdit,QCheckBox,QComboBox,QGroupBox,QSplitter,QLineEdit
if __name__=='__main__':
    from fixed_map_runtime import configure_session
    configure_session(os.environ.get('FIXED_MAP_SESSION_OUTPUT') or str(Path(__file__).resolve().parents[1]/'reports'/('fixed_map_interface_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))))
from fixed_map_mission import Executor,OUT,read,save,utc,ensure_scene,sha
from fixed_map_sensor_language import interpret,ACTION_LABELS,ACTION_STAGES
from fixed_map_sensor_mission import SensorMission
from fixed_map_resources import memory_headroom

REVISION=OUT/'interface_revision_20261009'

class CameraWorker(QThread):
    frame=pyqtSignal(object)
    failure=pyqtSignal(str,str)
    def __init__(self,executor,folder):
        super().__init__();self.e=executor;self.folder=folder;self.onboard_hz=.5;self.enabled=True;self.overview_enabled=True
    def run(self):
        import airsim
        c=airsim.MultirotorClient(port=41454,timeout_value=3);due={'overview':0.,'onboard':0.}
        while not self.e.exit_requested and not self.isInterruptionRequested():
            if self.e.stage=='CHECKING':self.msleep(100);continue
            for key in ('overview','onboard'):
                if self.e.exit_requested or self.isInterruptionRequested():break
                if key=='onboard' and not self.enabled:continue
                if key=='overview' and not self.overview_enabled:continue
                if time.monotonic()<due[key]:continue
                budget=memory_headroom()
                if min(budget['physical_available_bytes'],budget['commit_available_bytes'])<2**30:
                    self.failure.emit(key,'内存/提交余量不足，暂缓采集');due[key]=time.monotonic()+2;continue
                # Rendering and IPC also use engine resources; yield to stale control state.
                state=self.e.state
                if self.e.stage in ('TAKEOFF','APPROACH','RETURN','LANDING') and state and (time.time_ns()-state['host_time_ns'])/1e9>.4:
                    due[key]=time.monotonic()+.3;continue
                started=utc();start=time.monotonic()
                try:
                    camera='lab_overview' if key=='overview' else 'front_center'
                    r=c.simGetImages([airsim.ImageRequest(camera,airsim.ImageType.Scene,False,True)],
                        vehicle_name='' if key=='overview' else 'DroneA',external=key=='overview')[0]
                    data=bytes(r.image_data_uint8)
                    if not data:raise ValueError('相机返回空图像')
                    latency=time.monotonic()-start
                    path=self.folder/('latest_'+key+'.png');path.write_bytes(data)
                    receipt=dict(stream=key,camera=camera,vehicle='external' if key=='overview' else 'DroneA',
                        path=str(path),width=r.width,height=r.height,image_timestamp_ns=int(r.time_stamp),
                        request_started_utc=started,received_utc=utc(),latency_host_s=latency,
                        configured_hz=.5 if key=='overview' else self.onboard_hz,
                        source='actual Unreal Scene PNG',stage=self.e.stage,run=str(self.e.run_dir))
                    receipt['memory_headroom']=budget
                    save(self.folder/('latest_'+key+'.json'),receipt)
                    with (self.folder/'camera_timestamps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(receipt)+'\n')
                    self.frame.emit(dict(receipt=receipt,data=data))
                    if key=='onboard' and latency>.25:
                        due[key]=time.monotonic()+max(2.,1/self.onboard_hz)
                    else:due[key]=start+(2. if key=='overview' else 1/self.onboard_hz)
                except Exception as error:
                    self.failure.emit(key,str(error));due[key]=time.monotonic()+2
            self.msleep(30)

class ImageCanvas(QWidget):
    def __init__(self):
        super().__init__();self.pixmap=QPixmap();self.setMinimumSize(240,150)
    def paintEvent(self,event):
        painter=QPainter(self);painter.fillRect(self.rect(),QColor('#10161c'))
        if self.pixmap.isNull():
            painter.setPen(QColor('#c4cbd2'));painter.drawText(self.rect(),Qt.AlignCenter,'等待实际相机图像…')
        else:
            scaled=self.pixmap.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            painter.drawPixmap((self.width()-scaled.width())//2,(self.height()-scaled.height())//2,scaled)

class ImagePanel(QGroupBox):
    def __init__(self,title):
        super().__init__(title);layout=QVBoxLayout(self);self.canvas=ImageCanvas();layout.addWidget(self.canvas,1)
        self.info=QLabel('实际 Unreal 图像 · 尚未收到');layout.addWidget(self.info)
        self.receipt=None
    def receive(self,record):
        self.receipt=record['receipt'];self.canvas.pixmap.loadFromData(record['data']);self.canvas.update()
        r=self.receipt
        self.info.setText(f"{r['width']}×{r['height']} · 请求耗时 {r['latency_host_s']*1000:.0f} ms · UTC {r['received_utc'][11:23]}")

class LanguageWorker(QThread):
    result=pyqtSignal(object)
    def __init__(self,text,task,cfg):super().__init__();self.text=text;self.task=task;self.cfg=cfg
    def run(self):
        try:self.result.emit(interpret(self.text,self.task,self.cfg))
        except Exception as error:self.result.emit(dict(instruction=self.text,validated=False,validation_error=str(error),flight_executed=False))

class SceneWorker(QThread):
    result=pyqtSignal(object)
    def __init__(self,output,seed,restore):super().__init__();self.output=output;self.seed=seed;self.restore=restore
    def run(self):
        import subprocess
        args=[sys.executable,str(Path(__file__).with_name('fixed_map_random_region.py')),'--output',str(self.output),'--seed',str(self.seed)]
        if self.restore:args.append('--restore')
        try:
            result=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),timeout=35)
            save(self.output.parent/(self.output.name+'_process.json'),dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
            self.result.emit(read(self.output/'public_receipt.json') if result.returncode==0 else dict(status='FAILED',reason=result.stderr[-500:]))
        except Exception as ex:self.result.emit(dict(status='FAILED',reason=str(ex)))

class MapCanvas(QWidget):
    def __init__(self,executor):super().__init__();self.e=executor;self.setMinimumSize(410,225)
    def paintEvent(self,event):
        painter=QPainter(self);painter.setRenderHint(QPainter.Antialiasing);painter.fillRect(self.rect(),QColor('#f4f6f8'))
        scale=min((self.width()-60)/60,(self.height()-60)/36)
        def point(p):return QPointF(30+(p[0]+6)*scale,30+(p[1]+18)*scale)
        painter.setPen(QColor('#354152'));painter.drawText(12,19,'世界坐标 / m · 虚线：计划 · 绿色：实测')
        perception_active=getattr(self,'perception_active',False)
        for part in (getattr(self,'perception_parts',[]) if perception_active else self.e.g.parts):
            if part['group']=='ground':continue
            painter.setBrush(QColor('#d8dbe0'));painter.setPen(QColor('#939aa5'))
            c=part['center_xyz_m']
            if part['shape']=='sphere':painter.setBrush(QColor('#ba4b42'));painter.drawEllipse(point(c),part['radius_m']*scale,part['radius_m']*scale)
            else:
                a=part['orientation_axes'];h=[v/2 for v in part['local_size_xyz_m']]
                corners=[point([c[i]+s*a[0][i]*h[0]+t*a[1][i]*h[1] for i in range(3)]) for s,t in [(-1,-1),(1,-1),(1,1),(-1,1)]]
                painter.drawPolygon(QPolygonF(corners))
        painter.setPen(QPen(QColor('#2365a1'),2,Qt.DashLine))
        display_path=getattr(self,'preview_path',[]) if perception_active else self.e.path
        for a,b in zip(display_path,display_path[1:]):painter.drawLine(point(a),point(b))
        painter.setPen(QPen(QColor('#19845c'),2))
        for a,b in zip(self.e.trace,self.e.trace[1:]):painter.drawLine(point(a),point(b))
        if self.e.state:
            p=self.e.state['position_xyz_m'];painter.setBrush(QColor('#19845c'));painter.setPen(QColor('#19845c'));painter.drawEllipse(point(p),.75*scale,.75*scale)
            painter.drawText(point(p)+QPointF(8,-9),'DroneA')
            b=self.e.state['DroneB_position_xyz_m'];painter.setBrush(QColor('#656b72'));painter.drawEllipse(point(b),.75*scale,.75*scale);painter.drawText(point(b)+QPointF(8,15),'DroneB 地面')
        if not perception_active or display_path:
            g=display_path[-1] if perception_active else self.e.task['goal_xyz_m'];painter.setPen(QColor('#ae5121'));painter.setBrush(Qt.NoBrush);painter.drawEllipse(point(g),.2*scale,.2*scale);painter.drawText(point(g)+QPointF(-70,-12),'感知预览点' if perception_active else '固定基线悬停点')
        painter.end()

class Dashboard(QWidget):
    def __init__(self):
        super().__init__();self.setWindowTitle('固定地图单机绕障接近 · 人工任务界面');self.resize(1380,960)
        self.session=REVISION/('session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));self.session.mkdir(parents=True)
        self.perception=None;self.perception_serial=0;self.last_perception_frame=None;self.scene_worker=None;self.replay_preview=None
        scene_state=Path(__file__).resolve().parents[1]/'configs/fixed_map_scene_state.json'
        self.scene_baseline_available=read(scene_state).get('original_baseline_available',True) if scene_state.exists() else True
        self.proposal_record=None;self.pending_language_record=None;self.language_worker=None;self.frame_count={};self.last_frame={}
        save(self.session/'session.json',dict(utc=utc(),dashboard_sha256=sha(__file__),language_sha256=sha(__import__('fixed_map_sensor_language').__file__),
            executor_sha256=sha(__import__('fixed_map_mission').__file__),fixed_manifest_sha256=sha(OUT/'fixed_version/manifest.json'),
            camera_request_hz={'overview':.5,'onboard':.5},llm='local Ollama fixed-mission-qwen3:0.6b; CPU two threads; on demand; keep_alive=0'))
        from fixed_map_perception_execution import PerceptionExecutor
        declaration=None if self.scene_baseline_available else dict(observation_goal_xyz_m=[30.,-9.,3.1],stage='predeclared_observation_NOT_perception_guidance')
        self.e=PerceptionExecutor(declaration);self.sensor_mission=SensorMission(self.e);layout=QVBoxLayout(self)
        self.mission_timer=QTimer(self);self.mission_timer.timeout.connect(self.sensor_mission.tick);self.mission_timer.start(50)
        title=QLabel('DroneA 绕障接近 / 固定地图实验');title.setStyleSheet('font-size:22px;font-weight:600');layout.addWidget(title)
        t=self.e.baseline_task;info=QLabel(f"固定坐标基线（独立于感知，随机场景禁用）：目标 {t['target_xyz_m']} → 悬停点 {t['goal_xyz_m']}  |  世界高度 3.10 m  |  实测限速 0.80 m/s  |  净空停止阈值 0.75 m\n机体保守球 r=0.75 m；路径净空≥1.25 m；到达：误差≤0.20 m、速度≤0.12 m/s 持续3模拟秒。DroneB不获取控制。")
        info.setWordWrap(True);layout.addWidget(info);self.status=QLabel('读取固定版本、实际状态并核对路径…');self.status.setStyleSheet('font-size:16px');layout.addWidget(self.status)
        row=QHBoxLayout();layout.addLayout(row)
        self.buttons={}
        for label,action in [('固定坐标开始','start'),('暂停','pause'),('继续','resume'),('停止并悬停','stop'),('返航并落地','return')]:
            b=QPushButton(label);b.setMinimumHeight(42);b.clicked.connect(lambda _,a=action:self.action(a));self.buttons[action]=b;row.addWidget(b)
        self.buttons['stop'].setStyleSheet('background:#a93d3d;color:white;font-weight:600')
        visuals=QSplitter(Qt.Horizontal)
        self.overview=ImagePanel('Unreal 实际场景总览 · 后台引擎 / AirSim 实时取图')
        visuals.addWidget(self.overview)
        self.onboard=ImagePanel('DroneA 机载前视 · front_center')
        controls=QHBoxLayout();self.camera_enabled=QCheckBox('启用机载画面');self.camera_enabled.setChecked(True);controls.addWidget(self.camera_enabled)
        controls.addWidget(QLabel('采集频率'));self.camera_rate=QComboBox();self.camera_rate.addItems(['0.5 Hz','1 Hz','2 Hz']);controls.addWidget(self.camera_rate)
        self.onboard.layout().insertLayout(0,controls);visuals.addWidget(self.onboard);visuals.setSizes([500,850])
        layout.addWidget(visuals,5)
        lower=QSplitter(Qt.Horizontal);self.map=MapCanvas(self.e);lower.addWidget(self.map)
        self.map.perception_active=not self.scene_baseline_available;self.map.perception_parts=self.e.static_geometry.parts
        self.details=QTextEdit();self.details.setReadOnly(True);lower.addWidget(self.details);lower.setSizes([700,650]);layout.addWidget(lower,2)
        language=QGroupBox('自然语言 → 任务JSON → YOLO感知闭环');language.setCheckable(True);language.setChecked(True)
        language_layout=QHBoxLayout(language);entry=QVBoxLayout();language_layout.addLayout(entry,1)
        self.instruction=QPlainTextEdit();self.instruction.setPlaceholderText('例如：让单机识别并接近带白色横条的红色气球，然后返航落地\n或：预览红色气球的接近路径 / 停止并悬停 / 返航落地');self.instruction.setMaximumHeight(80);entry.addWidget(self.instruction)
        parse_row=QHBoxLayout();self.parse_button=QPushButton('LLM 解析');parse_row.addWidget(self.parse_button)
        self.execute_button=QPushButton('确认执行解析结果');self.execute_button.setEnabled(False);parse_row.addWidget(self.execute_button);entry.addLayout(parse_row)
        self.language_preview=QTextEdit();self.language_preview.setReadOnly(True);self.language_preview.setMaximumHeight(125)
        self.language_preview.setPlainText('本地文字模型解析任务JSON；YOLO与深度产生目标位置。支持：感知接近后返航、仅规划预览、停止、返航。\n多个相同外观目标会拒识。当前地面配置禁止起飞；紧急停止可直接点击上方按钮。')
        language_layout.addWidget(self.language_preview,1);layout.addWidget(language,0)
        language.toggled.connect(lambda active:[x.setVisible(active) for x in [self.instruction,self.parse_button,self.execute_button,self.language_preview]])
        for x in [self.instruction,self.parse_button,self.execute_button,self.language_preview]:x.setVisible(True)
        self.parse_button.clicked.connect(self.parse_instruction);self.execute_button.clicked.connect(self.execute_instruction)
        self.instruction.textChanged.connect(self.invalidate_proposal)
        self.footer=QLabel('第3步尚未通过：历史原场景有在线正例；本轮随机场景白条红球漏检，尚无有效感知规划执行证据。随机场景禁用固定坐标基线，当前不具备四方法先导条件。');self.footer.setWordWrap(True);layout.addWidget(self.footer)
        self.timer=QTimer();self.timer.timeout.connect(self.refresh);self.timer.start(200);self.last_snap='';self.e.start()
        self.camera_worker=CameraWorker(self.e,self.session);self.camera_worker.frame.connect(self.on_frame);self.camera_worker.failure.connect(self.camera_failure)
        self.camera_enabled.toggled.connect(self.set_camera_enabled);self.camera_rate.currentIndexChanged.connect(self.set_camera_rate);self.camera_worker.start()
        perception_box=QGroupBox('机载识别与目标定位 · 观察准备与感知接近分别记录')
        pl=QVBoxLayout(perception_box);pr=QHBoxLayout();pl.addLayout(pr)
        from fixed_map_perception import ROOT as perception_root,read as perception_read
        self.perception_config=perception_read(perception_root/'configs/fixed_map_perception_yolo_cpu_v11.json')
        if self.perception_config.get('ground_development_only'):
            self.setWindowTitle('地面感知与规划开发 · DroneA');title.setText('DroneA 地面感知与规划开发')
            self.footer.setText('YOLO CPU v11：v10已有在线规划和实际指令；表面变化后恢复被高度检查拒绝，已落地。表面像素与恢复定高修复仅地面/回放验证，当前禁止起飞；无VLX调用。')
        self.perception_serial=self.perception_config['target_generation']-1
        self.target_description=QLineEdit(self.perception_config['target_description']);pr.addWidget(self.target_description)
        self.perception_start=QPushButton('启动 / 应用新目标版本');pr.addWidget(self.perception_start);self.perception_start.clicked.connect(self.start_perception)
        self.perception_stop=QPushButton('停止感知');pr.addWidget(self.perception_stop);self.perception_stop.clicked.connect(self.stop_perception)
        self.show_detection=QCheckBox('显示检测框');self.show_detection.setChecked(True);pr.addWidget(self.show_detection)
        task_row=QHBoxLayout();pl.addLayout(task_row)
        scene_row=QHBoxLayout();pl.addLayout(scene_row);scene_row.addWidget(QLabel('无系绳随机区：X42–49 / Y−7–−1.5 / Z2.4–5m；地面设置'))
        self.scene_seed=QLineEdit('24101101');self.scene_seed.setMaximumWidth(110);scene_row.addWidget(self.scene_seed)
        self.scene_random=QPushButton('三种气球随机位置与高度');scene_row.addWidget(self.scene_random);self.scene_random.clicked.connect(lambda:self.change_scene(False))
        self.scene_restore=QPushButton('恢复副本初始位置');scene_row.addWidget(self.scene_restore);self.scene_restore.clicked.connect(lambda:self.change_scene(True))
        self.observe_button=QPushButton('飞到预声明观察点');task_row.addWidget(self.observe_button);self.observe_button.clicked.connect(self.prepare_observation)
        self.preview_button=QPushButton('预览感知任务');task_row.addWidget(self.preview_button);self.preview_button.clicked.connect(lambda:self.e.plan_perception(False))
        self.perception_execute_button=QPushButton('启动感知任务');task_row.addWidget(self.perception_execute_button);self.perception_execute_button.clicked.connect(lambda:self.e.plan_perception(True))
        self.replay_button=QPushButton('查看历史感知规划回放');task_row.addWidget(self.replay_button);self.replay_button.clicked.connect(self.show_reference_replay)
        self.perception_status=QPlainTextEdit();self.perception_status.setReadOnly(True);self.perception_status.setMinimumHeight(105);self.perception_status.setMaximumHeight(130)
        self.perception_status.setPlainText('尚未启动；固定坐标基线与感知版本分开。无有效感知时不生成目标位置。随机场景需在地面设置，位置不传给感知。')
        pl.addWidget(self.perception_status);layout.addWidget(perception_box,2)
        QTimer.singleShot(6000,lambda:self.grab().save(str(self.session/'dashboard_live.png')))
    def show_reference_replay(self):
        if self.e.stage!='READY' or (self.perception and self.perception.process.poll() is None):
            self.perception_status.setPlainText('仅地面且感知停止后查看独立回放。');return
        try:
            root=Path(__file__).resolve().parents[1];reference=read(root/'configs/ground_replay_preview_reference.json');path=Path(reference['path'])
            if sha(path)!=reference['sha256']:raise ValueError('REPLAY_FILE_HASH_CHANGED')
            row=read(path)
            if row.get('mode')!='HISTORICAL_VALID_EVIDENCE_REPLAY_PREVIEW_NOT_ONLINE' or row.get('flight_authorized') is not False:raise ValueError('REPLAY_MODE_INVALID')
            self.replay_preview=row;self.map.perception_active=True;self.map.preview_path=row['preview']['path']
            self.camera_worker.enabled=False;self.onboard.canvas.pixmap.load(row['overlay']);self.onboard.canvas.update()
            self.onboard.setTitle('历史机载感知回放 · 不可执行');self.onboard.info.setText('历史源帧 '+row['source_frame_id']+' · 原时间 '+row['source_capture_utc']+' · 非当前在线图像')
            self.perception_status.setPlainText('已保存真实图像的独立规划回放（不是当前在线结果）\n来源：'+row['source']+'\n原记录可见表面 XYZ/m：'+str(row['estimate']['surface_world_xyz_m'])+'（不是气球中心）\n原记录起点：'+str(row['recorded_start_xyz_m'])+'；回放接近点：'+str(row['preview']['task']['goal_xyz_m'])+'\n安全静态预览耗时：'+str(round(row['preview_wall_s'],3))+' s；当前无人机仍在地面。回放证据不会进入在线执行接口。')
        except (OSError,ValueError,KeyError) as ex:self.perception_status.setPlainText('回放拒绝：'+str(ex))
    def change_scene(self,restore):
        if self.e.stage!='READY' or (self.scene_worker and self.scene_worker.isRunning()):return
        if self.perception and self.perception.process.poll() is None:
            self.perception_status.setPlainText('请先停止感知并等待进程退出，再设置场景。');return
        try:seed=int(self.scene_seed.text())
        except ValueError:self.perception_status.setPlainText('随机种子需要整数');return
        self.scene_worker=SceneWorker(OUT/('scenario_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')),seed,restore)
        self.scene_worker.result.connect(self.scene_changed);self.scene_worker.start()
        self.scene_random.setEnabled(False);self.scene_restore.setEnabled(False)
        self.perception_status.setPlainText('正在设置独立运行场景；没有飞行命令…')
    def scene_changed(self,receipt):
        if receipt['status'].startswith('PASS_'):
            self.scene_baseline_available=receipt['original_baseline_available'];self.e.service=None;self.e.preview_result=None
            self.map.perception_active=not self.scene_baseline_available;self.map.preview_path=[]
            self.perception_status.setPlainText('场景版本：'+receipt['scene_version']+'；真值未传入感知。请重新启动感知。')
        else:self.perception_status.setPlainText('场景设置失败：'+receipt.get('reason','未知'))
    def start_perception(self):
        self.replay_preview=None;self.map.preview_path=[];self.onboard.setTitle('DroneA 机载前视 · front_center')
        self.onboard.canvas.pixmap=QPixmap();self.onboard.canvas.update();self.onboard.info.setText('等待新鲜实际图像；旧回放已清除')
        description=self.target_description.text().strip()
        if not description:self.perception_status.setPlainText('拒绝：指定目标描述不能为空');return
        if self.perception and self.perception.process.poll() is None:
            self.perception.stop(wait=False);self.e.service=None
            self.perception_status.setPlainText('旧感知进程正在退出，旧证据已失效；退出后再应用新目标版本。');return
        from fixed_map_perception_process import PerceptionProcess
        self.perception_serial+=1;cfg=dict(self.perception_config,target_description=description,target_generation=self.perception_serial)
        state_path=Path(__file__).resolve().parents[1]/'configs/fixed_map_scene_state.json'
        if state_path.exists():
            from fixed_map_scene_binding import bind_scene_config
            try:cfg=bind_scene_config(cfg,read(state_path))
            except ValueError as ex:self.perception_status.setPlainText('场景绑定拒绝：'+str(ex));return
        self.perception=PerceptionProcess(OUT/('perception_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')),cfg)
        from fixed_map_perception_execution import TrackingMarginGeometry
        self.e.static_geometry=TrackingMarginGeometry(self.e.static_geometry.parts,.75,cfg.get('tracking_clearance_margin_m',0.))
        if not self.e.baseline_mode:self.e.g=self.e.static_geometry
        self.map.perception_active=True;self.map.perception_parts=self.perception.geometry.parts;self.map.preview_path=[]
        self.last_perception_frame=None;self.camera_worker.enabled=False;self.camera_worker.overview_enabled=False
        try:self.perception.start()
        except (ValueError,OSError,RuntimeError) as ex:
            if getattr(self.perception,'process',None) is not None and self.perception.process.poll() is None:self.perception.stop(wait=False)
            save(self.session/('perception_start_rejected_'+datetime.now().strftime('%H%M%S_%f')+'.json'),dict(utc=utc(),reason=str(ex),flight_commands_issued=False))
            self.perception=None;self.e.service=None;self.camera_worker.enabled=self.camera_enabled.isChecked();self.camera_worker.overview_enabled=True
            self.perception_status.setPlainText('感知启动被拒绝：'+str(ex)+'；没有推理或定位结果。界面保持地面开发，可查看独立历史回放。');return
        self.e.service=self.perception
        self.perception_status.setPlainText('独立感知进程已启动；辅助相机取图暂停。感知任务需要新鲜指定实例证据，通过校验后才可启动。')
    def stop_perception(self):
        if self.perception:self.perception.stop(wait=False)
        self.e.service=None;self.camera_worker.enabled=self.camera_enabled.isChecked();self.camera_worker.overview_enabled=True
        self.map.perception_active=not self.scene_baseline_available;self.map.preview_path=[]
        self.perception_status.setPlainText('感知已停止，旧估计不再有效。')
    def prepare_observation(self):
        if self.perception_config.get('ground_development_only'):
            self.perception_status.setPlainText('本版本用于地面感知与规划开发，飞行观察入口已禁用。');return
        declaration=dict(created_utc=utc(),observation_goal_xyz_m=[30.,-9.,3.1],heading_yaw_deg=0.,selection_basis='predeclared static workspace side corridor; no target Actor or baseline goal input',stage='observation_preparation_NOT_perception_guidance',target_description=self.target_description.text(),static_geometry_version='v2')
        save(self.session/('observation_declaration_'+datetime.now().strftime('%H%M%S_%f')+'.json'),declaration)
        self.e.prepare_observation(declaration)
    def refresh_perception(self):
        if not self.perception or self.e.service is None:return
        row=self.perception.latest
        if not row:return
        estimate=row.get('position_estimate',{});track=row.get('temporal_identity',{});semantic=row.get('semantic_poll',{})
        age=time.monotonic()-row.get('capture',{}).get('received_monotonic_s',time.monotonic())
        fresh=age<=self.perception.cfg['current_max_age_s']
        position=estimate.get('surface_world_xyz_m') if fresh and estimate.get('valid') else None
        self.perception_status.setPlainText(f"目标版本 {self.perception.cfg['target_generation']}：{self.perception.cfg['target_description']}\n帧：{row.get('frame_id','尚无')}；候选：{len(row.get('semantic_frame',{}).get('candidates',[]))}；本地关联号：{track.get('track_id')}（不是实例真值）\n指定目标：{track.get('state','未确认')}；证据年龄：{age:.3f}s；当前有效：{bool(position is not None)}\n可见表面估计 XYZ/m：{position}（非气球中心）\n识别来源：YOLO＋当前RGB属性；外部复核已关闭\n规划预览：{row.get('planning_preview',{}).get('status')}；仅静态障碍；过期后不采用\n拒识：{', '.join(row.get('rejection_reasons',[]))}{'; CURRENT_FRAME_EXPIRED' if not fresh else ''}")
        stamp=(row.get('frame_id'),self.show_detection.isChecked(),Path(row.get('overlay_path','')).is_file())
        if row.get('frame_id') and stamp!=self.last_perception_frame:
            raw_path=row['capture']['rgb_path']
            path=row.get('overlay_path',raw_path) if self.show_detection.isChecked() else raw_path
            if not Path(path).is_file():path=raw_path
            if not Path(path).is_file():
                self.perception_status.appendPlainText('拒识：当前帧图像文件不可读');return
            receipt=dict(width=row['capture']['rgb_size_wh'][0],height=row['capture']['rgb_size_wh'][1],latency_host_s=row['capture']['capture_wall_s'],received_utc=row['capture']['received_utc'])
            self.onboard.receive(dict(receipt=receipt,data=Path(path).read_bytes()));self.last_perception_frame=stamp
        plan=self.e.preview_result or {};self.map.preview_path=plan.get('path',[]) if fresh and estimate.get('valid') else []
        self.perception_status.appendPlainText('执行接口：'+plan.get('status','尚未请求规划')+' / '+plan.get('reason',''))
        self.preview_button.setEnabled(fresh and estimate.get('valid',False) and self.e.stage in ['READY','ARRIVED','ABORTED','PAUSED'])
        self.perception_execute_button.setEnabled(not self.perception_config.get('ground_development_only') and fresh and estimate.get('valid',False) and not self.e.baseline_mode and self.e.stage in ['ARRIVED','ABORTED','PAUSED'])
    def set_camera_enabled(self,enabled):
        self.camera_worker.enabled=enabled
        if not enabled:self.onboard.info.setText('机载采集已关闭；保留的最后一帧不是实时画面')
        save(self.session/'camera_preferences.json',dict(utc=utc(),onboard_enabled=enabled,onboard_hz=self.camera_worker.onboard_hz))
    def set_camera_rate(self,index):
        self.camera_worker.onboard_hz=[.5,1.,2.][index]
        save(self.session/'camera_preferences.json',dict(utc=utc(),onboard_enabled=self.camera_worker.enabled,onboard_hz=self.camera_worker.onboard_hz))
    def on_frame(self,record):
        key=record['receipt']['stream']
        if key=='onboard' and self.replay_preview is not None:return
        if key=='onboard' and not self.camera_worker.enabled:return
        getattr(self,key).receive(record)
        self.frame_count[key]=self.frame_count.get(key,0)+1;self.last_frame[key]=time.monotonic()
    def camera_failure(self,key,error):getattr(self,key).info.setText('实际相机读取失败：'+error)
    def invalidate_proposal(self):self.proposal_record=None;self.pending_language_record=None;self.execute_button.setEnabled(False)
    def parse_instruction(self):
        text=self.instruction.toPlainText().strip();self.invalidate_proposal()
        if not text:self.language_preview.setPlainText('请输入一条任务指令');return
        self.parse_button.setEnabled(False);self.language_preview.setPlainText('正在调用本地文字 LLM；不会发出飞行命令…')
        self.language_worker=LanguageWorker(text,self.e.safety,self.perception_config);self.language_worker.result.connect(self.parsed);self.language_worker.start()
    def parsed(self,record):
        name=datetime.now().strftime('language_%H%M%S_%f.json');save(self.session/name,record);self.parse_button.setEnabled(True)
        if record['instruction']!=self.instruction.toPlainText().strip():
            self.language_preview.setPlainText('输入在解析期间已变更，请重新解析');return
        if not record.get('validated'):
            self.language_preview.setPlainText('不执行：'+record.get('validation_error','解析失败'));return
        self.proposal_record=record;p=record['proposal'];self.execute_button.setEnabled(self.e.stage in ACTION_STAGES[p['action']])
        self.language_preview.setPlainText(f"解析：{ACTION_LABELS[p['action']]} · {p['vehicle']}\n{p['reason']}\n本地 LLM 耗时 {record['latency_host_s']:.2f} s；尚未执行。\n目标：{p['target_description']}；飞行高度3.1m / 指令上限0.65m/s。输入不包含目标真值坐标。")
        QTimer.singleShot(300,lambda:self.grab().save(str(self.session/'dashboard_language_preview.png')))
    def execute_instruction(self):
        record=self.proposal_record
        if not record:return
        try:
            if record['instruction']!=self.instruction.toPlainText().strip():raise ValueError('输入已变更，请重新解析')
            p=record['proposal']
            if p['action']=='start' and self.perception_config.get('ground_development_only'):raise ValueError('在线闭环尚未通过；当前配置仅允许感知与规划地面开发')
            if p['action'] in ['start','preview']:
                if self.target_description.text()!=p['target_description'] or self.e.service is None:
                    self.target_description.setText(p['target_description']);self.start_perception()
                if self.e.service is None:raise ValueError('感知尚未启动，旧任务未执行')
                if not self.e.service.warm_ready.is_set():
                    self.pending_language_record=record;self.proposal_record=None;self.execute_button.setEnabled(False)
                    self.language_preview.append('等待感知模型完成预热，再校验并提交任务；尚未发出飞行命令。');return
            self.sensor_mission.submit(record,self.perception_config)
            save(self.session/datetime.now().strftime('language_dispatch_%H%M%S_%f.json'),dict(utc=utc(),record=record,stage=self.e.stage,actor='llm_preview_confirmed',queued=True))
            self.language_preview.append('已提交统一执行接口，实际状态请查看上方任务阶段。');self.invalidate_proposal()
        except ValueError as error:self.language_preview.append('拒绝执行：'+str(error));self.invalidate_proposal()
    def action(self,action):
        if action=='start' and self.perception_config.get('ground_development_only'):
            self.perception_status.setPlainText('本版本用于地面开发，固定坐标起飞入口已禁用。');return
        if action=='start' and not self.scene_baseline_available:
            self.perception_status.setPlainText('随机场景仅支持感知分支；恢复原场景后才可使用固定坐标基线。');return
        save(OUT/'last_gui_action.json',dict(action=action,utc=utc(),origin='native_button_clicked',stage=self.e.stage))
        if action=='start':self.e.start_baseline('native_fixed_baseline_button')
        else:self.e.command(action,'native_gui_button')
    def refresh(self):
        if self.pending_language_record:
            record=self.pending_language_record
            if time.monotonic()-record['created_monotonic_s']>120 or self.e.service is None:
                self.pending_language_record=None;self.language_preview.append('任务等待失败或已过期，请重新解析。')
            elif self.e.service.warm_ready.is_set():
                self.pending_language_record=None;self.proposal_record=record;self.execute_instruction()
        scene_busy=self.scene_worker is not None and self.scene_worker.isRunning()
        sensing=self.perception is not None and self.perception.process.poll() is None
        self.scene_random.setEnabled(self.e.stage=='READY' and not scene_busy and not sensing);self.scene_restore.setEnabled(self.e.stage=='READY' and not scene_busy and not sensing)
        self.observe_button.setEnabled(self.e.stage=='READY' and not self.perception_config.get('ground_development_only'))
        self.replay_button.setEnabled(self.e.stage=='READY' and not sensing and not scene_busy)
        if self.e.service is None:
            self.preview_button.setEnabled(False);self.perception_execute_button.setEnabled(False)
        self.refresh_perception()
        stage=self.e.stage;r=self.e.state;reason=self.e.reason
        labels={'CHECKING':'检查中','READY':'地面就绪','TAKEOFF':'起飞','APPROACH':'绕障接近','ARRIVAL_HOVER':'到达悬停验证','ARRIVED':'到达悬停已验证',
            'STOP_HOVER':'人工中止悬停验证','ABORTED':'中止悬停已验证','PAUSED':'暂停并悬停','RETURN':'返航','LANDING':'落地','FAILED_HOVER':'失败后悬停','FAILED':'检查失败'}
        self.status.setText('阶段：'+labels.get(stage,stage)+'  |  停止原因：'+(reason or '无'))
        self.buttons['start'].setEnabled(stage=='READY' and self.scene_baseline_available and not scene_busy and not self.perception_config.get('ground_development_only'))
        self.buttons['pause'].setEnabled(stage in ('TAKEOFF','APPROACH','RETURN'))
        self.buttons['resume'].setEnabled(stage=='PAUSED')
        self.buttons['stop'].setEnabled(stage in ('TAKEOFF','APPROACH','RETURN','PAUSED','ARRIVAL_HOVER','ARRIVED'))
        self.buttons['return'].setEnabled(stage in ('PAUSED','ARRIVED','ABORTED','STOP_HOVER','ARRIVAL_HOVER'))
        if self.proposal_record:self.execute_button.setEnabled(stage in ACTION_STAGES[self.proposal_record['proposal']['action']])
        if r:
            p=r['position_xyz_m'];q=r['orientation_xyzw'];v=r['velocity_xyz_m_s'];b=r['DroneB_position_xyz_m']
            self.details.setPlainText(f"实测位姿 XYZ / m： {p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}\n实测四元数 XYZW： {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}\n实测速度 XYZ / m/s： {v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}\n实测合速度： {r['speed_m_s']:.3f} m/s\n计算到达误差： {r['goal_error_m']:.3f} m\n计算机体保守包络净空： {r['envelope_clearance_calculated_m']:.3f} m\n最近实体： {r['nearest_obstacle']}\nDroneB：位置 {b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f} m\nDroneB地面：{r['DroneB_landed']}；API控制：{r['DroneB_api_control']}\n实际状态时间（UTC）：{r['utc']}\n负净空在地面表示保守球与起降平台相交，起降阶段有单独接触规则。")
        elif self.e.error:self.details.setPlainText(self.e.error)
        self.map.update()
        for key in ('overview','onboard'):
            if key=='onboard' and self.replay_preview is not None:continue
            panel=getattr(self,key)
            if panel.receipt and time.monotonic()-self.last_frame.get(key,0)>5 and (key!='onboard' or self.camera_worker.enabled):
                panel.info.setText('最后一帧已超过5秒，当前画面未更新')
        stamp=str(self.e.run_dir)+stage
        if stamp!=self.last_snap and stage in ('READY','APPROACH','ARRIVED','ABORTED','PAUSED','RETURN','FAILED_HOVER','FAILED'):
            self.last_snap=stamp
            destination=(self.e.run_dir/'ui') if self.e.run_dir else self.session;destination.mkdir(exist_ok=True)
            QTimer.singleShot(300,lambda p=destination/(datetime.now().strftime('%H%M%S')+'_'+stage+'.png'):self.grab().save(str(p)))
    def closeEvent(self,event):
        if self.e.stage not in ('READY','FAILED','CHECKING'):
            self.e.command('stop','window_close_request');self.footer.setText('已请求悬停；请返航落地后关闭界面。');event.ignore()
        else:
            if self.scene_worker and self.scene_worker.isRunning():event.ignore();QTimer.singleShot(500,self.close);return
            if self.perception:self.perception.stop(wait=False)
            self.e.exit_requested=True;self.camera_worker.requestInterruption()
            # An inference request can finish in the background; don't destroy a running QThread.
            if self.language_worker and self.language_worker.isRunning():
                self.footer.setText('等待当前文字解析结束后关闭…');event.ignore();QTimer.singleShot(500,self.close);return
            if not self.camera_worker.wait(3500):
                event.ignore();QTimer.singleShot(500,self.close);return
            event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv)
    from PyQt5.QtCore import QSharedMemory
    from PyQt5.QtWidgets import QMessageBox
    lock=QSharedMemory('FixedMapMissionDroneA20261009')
    if not lock.create(1):
        QMessageBox.information(None,'界面已运行','请按 Alt+Tab 选择“固定地图单机绕障接近 · 人工任务界面”。');sys.exit(0)
    try:ensure_scene()
    except Exception as error:
        save(OUT/'launcher_failure.json',dict(utc=utc(),error=str(error)))
        QMessageBox.critical(None,'固定地图启动失败',str(error));sys.exit(2)
    app.setFont(__import__('PyQt5.QtGui',fromlist=['QFont']).QFont('Microsoft YaHei',10));window=Dashboard();window.show();sys.exit(app.exec_())
