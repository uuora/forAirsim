"""PyQt5 evidence dashboard with saved-frame replay and Chinese plots."""
from __future__ import annotations

import csv
from pathlib import Path
from PyQt5.QtCore import Qt,QThread,pyqtSignal,QTimer,QUrl
from PyQt5.QtGui import QPixmap,QPainter,QPen,QColor,QFont,QDesktopServices
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,
    QComboBox,QSplitter,QTabWidget,QSlider,QFileDialog,QFrame,QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib import font_manager

from t1_qt_evidence import (ROOT,latest_analysis,load_campaign,load_frames,
    validate_frame_image)

STATUS_LABELS={"CORRECT":"正确目标","NO_OUTPUT":"无目标输出","MISSING":"未保存帧","UNSCORED":"中断流水未评分",
               "UNKNOWN_OUTPUT_STATE":"采集中断／输出未知","UNKNOWN_MEDIA":"媒体记录不完整"}
PHASE_LABELS={"visible":"目标可见","hidden":"完全遮挡","return":"返回阶段",
              "glint":"小白斑干扰"}
SEMANTIC_LABELS={
    "SOURCE_TO_CURRENT_CHAIN_ACCEPTED":"复核已接纳",
    "SEMANTIC_RESPONSE_NOT_READY":"等待后台复核",
    "NO_VALID_SEMANTIC_EVIDENCE":"暂无有效复核",
    "CHAIN_NO_UNIQUE_MATCH":"没有唯一候选链接",
    "SOURCE_CHAIN_AGE_EXPIRED":"原证据已经过期",
    "SEMANTIC_BACKEND_ERROR":"复核未获得有效结果",
    "CURRENT_CHAIN_VALID":"当前证据链有效",
}
MOTION_LABELS={
    "FRESH_BOUND_CURRENT_TARGET_AND_XY_ENVELOPE":"当前目标证据有效，运动约束允许",
    "NO_CURRENT_DESIGNATED_TARGET":"当前没有指定目标，停止目标动作",
    "INSUFFICIENT_CURRENT_FRAME_BUDGET_FOR_LEASE_AND_ACTUAL_CLEAR":"本帧剩余时间不足，停止目标动作",
    "CURRENT_FRAME_EXPIRED":"当前图像已过期，停止目标动作",
    "WAITING_FOR_STRICT_STOP":"等待采样停稳",
}


class LoadThread(QThread):
    result=pyqtSignal(object)
    failed=pyqtSignal(str)

    def __init__(self,loader,parent=None):
        super().__init__(parent);self.loader=loader

    def run(self):
        try:self.result.emit(self.loader())
        except (ValueError,KeyError,TypeError,OSError) as exc:
            # Only project-generated diagnostics; no raw external response.
            self.failed.emit(str(exc) if isinstance(exc,ValueError) else "保存记录不完整或无法读取。")


class FrameImage(QLabel):
    def __init__(self,parent=None):
        super().__init__(parent);self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(330,245);self.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        self.setObjectName("replayImage");self.original=None;self.annotated=None
        self.show_message("读取已保存的实验图像")

    def show_message(self,text):
        self.original=self.annotated=None;self.clear();self.setText(text)

    def set_frame(self,frame):
        self.show_message("正在核对图像")
        try:path=validate_frame_image(frame)
        except ValueError as exc:self.show_message(str(exc));return False
        pix=QPixmap(str(path))
        if pix.isNull():self.show_message("保存图像无法解码");return False
        self.original=pix;self.annotated=QPixmap(pix)
        painter=QPainter(self.annotated)
        painter.setFont(QFont("Microsoft YaHei UI",10))
        for i,box in enumerate(frame.candidates,1):
            if len(box)!=4:continue
            x1,y1,x2,y2=map(int,box)
            painter.setPen(QPen(QColor("#79aab4"),2));painter.drawRect(x1,y1,x2-x1,y2-y1)
            painter.drawText(x1+3,max(16,y1-4),f"候选{i}")
        if frame.selected:
            x1,y1,x2,y2=map(int,frame.selected)
            painter.setPen(QPen(QColor("#16a085"),3));painter.drawRect(x1,y1,x2-x1,y2-y1)
            painter.drawText(x1+3,max(16,y1-20),"当前关联目标")
        painter.end();self._fit();return True

    def _fit(self):
        if self.annotated is not None:
            self.setPixmap(self.annotated.scaled(self.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))

    def resizeEvent(self,event):
        super().resizeEvent(event);self._fit()


class ResearchPlots(QTabWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.canvases=[]
        path=Path("C:/Windows/Fonts/msyh.ttc")
        self.font=font_manager.FontProperties(fname=str(path)) if path.is_file() else font_manager.FontProperties(family="Microsoft YaHei")
        for title in ("XY轨迹","身份与位移","完成时序"):
            fig=Figure(figsize=(5.8,4.0),tight_layout=True)
            canvas=FigureCanvasQTAgg(fig);self.canvases.append(canvas);self.addTab(canvas,title)
        self.currentChanged.connect(lambda _:self.canvases[self.currentIndex()].draw_idle())
        self.cursor_lines=[]

    def decorate(self,ax):
        ax.grid(alpha=.14)
        for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
            t.set_fontproperties(self.font);t.set_fontsize(9)

    def set_run(self,run):
        self.cursor_lines=[]
        fig=self.canvases[0].figure;fig.clear();ax=fig.add_subplot(111)
        pts=run.state_points
        if pts:
            ax.plot([p["x_mm"] for p in pts],[p["y_mm"] for p in pts],c="#377f92",lw=1.4)
            ax.scatter([pts[0]["x_mm"]],[pts[0]["y_mm"]],s=27,c="#89949a",label="起点")
            ax.scatter([pts[-1]["x_mm"]],[pts[-1]["y_mm"]],s=30,c="#158775",label="终点")
            self.position_cursor,=ax.plot([],[],"o",color="#d1833f",markersize=6)
            ax.legend(prop=self.font,loc="best")
        else:
            self.position_cursor=None
            ax.text(.5,.5,"无已审计轨迹数据",ha="center",va="center",transform=ax.transAxes,fontproperties=self.font)
        ax.set(xlabel="相对起点X（毫米）",ylabel="相对起点Y（毫米）",title="自身状态轨迹 · 保留实际坐标方向")
        ax.set_aspect("equal",adjustable="datalim");self.decorate(ax)
        fig=self.canvases[1].figure;fig.clear();axes=fig.subplots(2,1,sharex=True)
        for status,label,value,color in [("CORRECT","正确",1,"#158775"),("NO_OUTPUT","无输出",0,"#87959d")]:
            fs=[p for p in run.identity_points if p["status"]==status]
            if fs:axes[0].scatter([p["time_s"] for p in fs],[value]*len(fs),s=8,c=color,label=label)
        other=[p for p in run.identity_points if p["status"] not in ("CORRECT","NO_OUTPUT")]
        if other:axes[0].scatter([p["time_s"] for p in other],[-1]*len(other),s=13,c="#bb5850",label="错选/待核查")
        if not run.identity_points:
            axes[0].text(.5,.5,"没有已保存的主帧评分",ha="center",va="center",transform=axes[0].transAxes,fontproperties=self.font)
        axes[0].set(yticks=[-1,0,1],yticklabels=["待核查","无输出","正确"],ylim=(-1.3,1.4),ylabel="身份")
        if run.identity_points:axes[0].legend(prop=self.font,loc="lower right")
        if pts:axes[1].plot([p["time_s"] for p in pts],[p["net_mm"] for p in pts],c="#377f92",lw=1.3)
        if run.minimum_mm is not None:
            axes[1].axhline(run.minimum_mm,color="#b27f42",ls="--",lw=1,label=f"{run.minimum_mm:g}毫米判据")
            axes[1].legend(prop=self.font,loc="upper left")
        axes[1].set(xlabel="相对主阶段开始（秒）",ylabel="XY距离（毫米）")
        for ax in axes:
            self.cursor_lines.append(ax.axvline(0,c="#d1833f",alpha=.65,lw=1))
            self.decorate(ax)
        fig=self.canvases[2].figure;fig.clear();ax=fig.add_subplot(111)
        for key,label,color in [("frame_processing","帧处理完成","#158775"),("historical_full","完整完成（含停稳）","#b47b44")]:
            rows=[p for p in run.timing_points if p["recorded_timing"][key]["after_due_s"] is not None]
            if rows:ax.plot([p["slot_index"] for p in rows],[p["recorded_timing"][key]["after_due_s"] for p in rows],c=color,lw=1,label=label)
        ax.axhline(.4,c="#929b9e",ls="--",label="原0.4秒截止点")
        ax.set(xlabel="主测量帧序号",ylabel="从本帧计划起点计时（秒）",title="完成时钟分别显示")
        ax.legend(prop=self.font,loc="upper right");self.decorate(ax)
        for canvas in self.canvases:canvas.draw_idle()

    def set_frame(self,frame,run):
        for line in self.cursor_lines:line.set_xdata([frame.time_s,frame.time_s])
        if self.position_cursor is not None and run.state_points:
            point=min(run.state_points,key=lambda p:abs(p["time_s"]-frame.time_s))
            self.position_cursor.set_data([point["x_mm"]],[point["y_mm"]])
        self.canvases[self.currentIndex()].draw_idle()


class ExperimentDashboard(QWidget):
    def __init__(self,root=ROOT,parent=None):
        super().__init__(parent);self.root=root;self.campaign=None;self.run=None
        self.frames=[];self.worker=None;self.last_error=None;self.busy=False
        self.setObjectName("experimentDashboard");self.timer=QTimer(self)
        self.timer.setInterval(400);self.timer.timeout.connect(self.next_frame)
        self._build()

    def _build(self):
        outer=QVBoxLayout(self);outer.setContentsMargins(18,16,18,12);outer.setSpacing(12)
        header=QHBoxLayout()
        heading=QVBoxLayout();title=QLabel("单机感知轨迹 · 实验回放");title.setObjectName("researchTitle")
        heading.addWidget(title);heading.addWidget(QLabel("指定目标：带白色横条的红色气球"))
        header.addLayout(heading);header.addStretch()
        self.latest_btn=QPushButton("加载最新结果");self.latest_btn.clicked.connect(self.load_latest)
        self.open_btn=QPushButton("打开分析报告…");self.open_btn.clicked.connect(self.open_campaign)
        header.addWidget(self.latest_btn);header.addWidget(self.open_btn);outer.addLayout(header)
        self.source_label=QLabel("等待加载实验记录");self.source_label.setTextFormat(Qt.PlainText)
        self.source_label.setObjectName("muted");outer.addWidget(self.source_label)
        row=QHBoxLayout();row.addWidget(QLabel("飞行会话"))
        self.session_combo=QComboBox();self.session_combo.setMinimumWidth(260)
        self.session_combo.currentIndexChanged.connect(self.select_run);row.addWidget(self.session_combo,1)
        self.evidence_label=QLabel("受控开发验证 · 正式四方法比较待完成")
        self.evidence_label.setObjectName("evidenceBadge");row.addWidget(self.evidence_label);outer.addLayout(row)
        cards=QHBoxLayout();self.metrics={}
        for key,title in [("correct","正确主帧"),("no_output","无目标输出"),("other","错选／待核查"),("net","净XY位移")]:
            card=QFrame();card.setObjectName("metricCard");layout=QVBoxLayout(card)
            label=QLabel(title);label.setObjectName("muted");layout.addWidget(label)
            value=QLabel("—");value.setObjectName("metricValue");layout.addWidget(value)
            cards.addWidget(card);self.metrics[key]=value
        outer.addLayout(cards)
        splitter=QSplitter(Qt.Horizontal)
        image_panel=QWidget();image_layout=QVBoxLayout(image_panel);image_layout.setContentsMargins(0,0,10,0)
        self.image=FrameImage();image_layout.addWidget(self.image,1)
        self.frame_label=QLabel("保存帧");self.frame_label.setTextFormat(Qt.PlainText);image_layout.addWidget(self.frame_label)
        self.frame_detail=QLabel("回放来源、身份和运动决策会随帧更新。")
        self.frame_detail.setTextFormat(Qt.PlainText);self.frame_detail.setWordWrap(True)
        self.frame_detail.setObjectName("muted");image_layout.addWidget(self.frame_detail)
        controls=QHBoxLayout()
        self.play_btn=QPushButton("播放");self.play_btn.clicked.connect(self.toggle_play)
        self.previous_btn=QPushButton("上一帧");self.previous_btn.clicked.connect(lambda:self.slider.setValue(self.slider.value()-1))
        self.next_btn=QPushButton("下一帧");self.next_btn.clicked.connect(self.next_frame)
        for btn in [self.previous_btn,self.play_btn,self.next_btn]:controls.addWidget(btn)
        image_layout.addLayout(controls)
        self.slider=QSlider(Qt.Horizontal);self.slider.valueChanged.connect(self.show_frame)
        image_layout.addWidget(self.slider);splitter.addWidget(image_panel)
        self.plots=ResearchPlots();splitter.addWidget(self.plots)
        splitter.setStretchFactor(0,1);splitter.setStretchFactor(1,1);splitter.setSizes([580,620])
        outer.addWidget(splitter,1)
        bottom=QHBoxLayout();self.status_label=QLabel("点击“加载最新结果”读取保存记录。")
        self.status_label.setTextFormat(Qt.PlainText);self.status_label.setWordWrap(True)
        bottom.addWidget(self.status_label,1)
        self.export_image_btn=QPushButton("导出当前图像");self.export_image_btn.clicked.connect(self.export_image)
        self.export_csv_btn=QPushButton("导出帧指标");self.export_csv_btn.clicked.connect(self.export_csv)
        self.folder_btn=QPushButton("打开会话目录");self.folder_btn.clicked.connect(self.open_folder)
        for btn in [self.export_image_btn,self.export_csv_btn,self.folder_btn]:bottom.addWidget(btn)
        outer.addLayout(bottom);self._controls()

    def _controls(self):
        self.latest_btn.setEnabled(not self.busy);self.open_btn.setEnabled(not self.busy)
        self.session_combo.setEnabled(not self.busy and self.campaign is not None)
        for obj in [self.slider,self.play_btn,self.previous_btn,self.next_btn,self.export_csv_btn]:
            obj.setEnabled(not self.busy and bool(self.frames))
        self.export_image_btn.setEnabled(not self.busy and self.image.annotated is not None)
        self.folder_btn.setEnabled(not self.busy and self.run is not None)

    def stop_play(self):
        self.timer.stop();self.play_btn.setText("播放")

    def start_job(self,loader,receiver,message):
        if self.busy:return False
        self.stop_play();self.busy=True;self.last_error=None;self.status_label.setText(message);self._controls()
        self.worker=LoadThread(loader,self)
        self.worker.result.connect(receiver);self.worker.failed.connect(self.failed)
        # Base QThread.finished, unlike result, guarantees the worker has exited.
        self.worker.finished.connect(self.job_finished);self.worker.start();return True

    def job_finished(self):
        self.busy=False;self._controls()
        if self.campaign is not None and not self.frames and self.last_error is None:
            QTimer.singleShot(0,lambda:self.select_run(self.session_combo.currentIndex()))

    def failed(self,message):
        self.last_error=message;self.status_label.setText(message)

    def load_latest(self):
        return self.start_job(lambda:load_campaign(latest_analysis(self.root),self.root),self.set_campaign,"正在读取最新分析记录…")

    def open_campaign(self):
        path,_=QFileDialog.getOpenFileName(self,"打开任务一分析报告",str(self.root/"logs"),"分析报告 (*.json)")
        if path:self.load_analysis(path)

    def load_analysis(self,path):
        return self.start_job(lambda:load_campaign(path,self.root),self.set_campaign,"正在读取分析报告…")

    def set_campaign(self,campaign):
        self.campaign=campaign;self.run=None;self.frames=[];self.image.show_message("请选择保存会话")
        self.session_combo.blockSignals(True);self.session_combo.clear()
        for run in campaign.runs:self.session_combo.addItem(run.label,run.session_id)
        self.session_combo.setCurrentIndex(0);self.session_combo.blockSignals(False)
        self.source_label.setText(f"{campaign.path.parent.name} · {len(campaign.runs)}个会话 · 已保存分析")
        self.source_label.setToolTip(str(campaign.path))
        self.status_label.setText("分析记录已读取，正在准备会话回放。")

    def select_run(self,index):
        if self.busy or self.campaign is None or not 0<=index<len(self.campaign.runs):return
        self.run=self.campaign.runs[index];self.frames=[];self.image.show_message("核对会话报告与图像来源…")
        run=self.run;c=run.counts
        self.metrics["correct"].setText(f'{c.get("CORRECT",0)} / {run.measured}')
        self.metrics["no_output"].setText(str(c.get("NO_OUTPUT",0)))
        self.metrics["other"].setText(str(sum(n for k,n in c.items() if k not in ("CORRECT","NO_OUTPUT"))))
        self.metrics["net"].setText("未知" if run.net_mm is None else f"{run.net_mm:.2f} mm")
        self.plots.set_run(run)
        self.start_job(lambda:load_frames(run,self.root),self.set_frames,"正在核对会话报告…")

    def set_frames(self,frames):
        self.frames=frames;self.slider.blockSignals(True);self.slider.setRange(0,max(0,len(frames)-1))
        self.slider.setValue(0);self.slider.blockSignals(False)
        self.show_frame(0)
        run=self.run
        late=run.source["timing_summary"]["historical_full"]["counts"].get("LATE",0)
        evidence="通过" if run.evidence_status.startswith("PASS") else "未通过／待核查"
        scale="未核实" if run.net_mm is None else "通过" if run.scale_status.startswith("PASS") else "未达到／待核查"
        incomplete=run.counts.get("UNKNOWN_MEDIA",0)
        media_notice=f"（完整媒体缺失{incomplete}帧）" if incomplete else ""
        self.status_label.setText(f"采集主帧 {run.measured}/{run.planned}{media_notice} · 初始化另计{run.acquisition}帧 · 证据核查{evidence} · 尺度{scale} · 完整完成晚到{late}帧")
        if run.source.get("incomplete_record_notice"):
            self.status_label.setText(self.status_label.text()+" · "+run.source["incomplete_record_notice"])
        dispatch=run.source.get("feedback_dispatch_audit")
        if dispatch:
            state="通过" if dispatch.get("status","").startswith("PASS") else "未通过／证据不完整"
            self.status_label.setText(self.status_label.text()+
                f"\n反馈时效核查{state} · 过期拒绝{dispatch.get('refused_attempts',0)}次 · 新状态恢复{dispatch.get('recovered_ticks',0)}次 · 时效阈值120毫秒")
        issues=run.source.get("base_evidence",{}).get("issues",[])
        if any(issue.get("issue")=="TARGET_EXPLICIT_CLEAR_AFTER_OBSERVATION_DEADLINE" for issue in issues):
            supplement=run.source.get("verified_clear_supplement",{})
            rows=supplement.get("rows",[])
            if (supplement.get("status")=="PASS_INTERRUPTED_TARGET_CLEAR_RECONSTRUCTION"
                    and rows and all(r.get("actual_first_clear_within_original_deadline") is True for r in rows)):
                margin=min(r["actual_first_clear_margin_s"] for r in rows)*1000
                self.status_label.setText(self.status_label.text()+
                    f"\n原清零晚到提示保留；补充核查：首次实际清零提前{margin:.0f}毫秒，收尾窗口另计。")
            else:
                self.status_label.setText(self.status_label.text()+"\n未通过原因：目标指令实际清零超过源观测截止时间。")
        process=run.source.get("process_control_audit")
        if process:
            state="通过" if process.get("status")=="PASS_PROCESS_CONTROL_EVIDENCE" else "未通过／停止证据不完整"
            replay=run.source.get("process_trace_replay_audit",{}).get("status")=="PASS_PROCESS_TRACE_REPLAY"
            self.status_label.setText(self.status_label.text()+
                f"\n控制进程核查{state} · 原执行记录回放{'一致' if replay else '待核查'}")
        if run.measured<run.planned or incomplete:
            self.status_label.setText(self.status_label.text()+
                "\n未完整保存的主槽保留在计划分母中，缺失端点仍显示未知。")

    def show_frame(self,index):
        if not 0<=index<len(self.frames):return
        frame=self.frames[index];self.image.set_frame(frame)
        identity=STATUS_LABELS.get(frame.identity,"错选／待核查")
        phase=PHASE_LABELS.get(frame.phase,frame.phase)
        clock_label="" if frame.actual_time else "计划时刻 "
        self.frame_label.setText(f"{index+1}/{len(self.frames)} · {phase} · {clock_label}{frame.time_s:.2f}秒 · {identity}")
        self.frame_label.setToolTip(frame.frame_id)
        semantic=SEMANTIC_LABELS.get(frame.semantic_reason,"查看已保存复核记录" if frame.semantic_reason else "无回执")
        motion=MOTION_LABELS.get(frame.motion_reason,"查看已保存运动判据" if frame.motion_reason else "无派发记录")
        self.frame_detail.setText(f"复核：{semantic}\n运动：{motion}")
        self.frame_detail.setToolTip(f"{frame.semantic_reason}\n{frame.motion_reason}")
        self.plots.set_frame(frame,self.run);self._controls()

    def toggle_play(self):
        if self.timer.isActive():self.stop_play()
        elif self.frames:
            if self.slider.value()==len(self.frames)-1:self.slider.setValue(0)
            self.timer.start();self.play_btn.setText("暂停")

    def next_frame(self):
        if self.slider.value()>=len(self.frames)-1:self.stop_play()
        else:self.slider.setValue(self.slider.value()+1)

    def export_image(self):
        if self.image.annotated is None:return
        path,_=QFileDialog.getSaveFileName(self,"导出回放图像",str(self.root/"logs"/f"{self.frames[self.slider.value()].frame_id}_view.png"),"PNG图像 (*.png)")
        if path:
            ok=self.image.annotated.save(path,"PNG")
            self.status_label.setText("当前帧图像已导出。" if ok else "图像导出失败。")

    def export_csv(self):
        if not self.frames:return
        path,_=QFileDialog.getSaveFileName(self,"导出帧指标",str(self.root/"logs"/f"{self.run.session_id}_frames.csv"),"CSV指标 (*.csv)")
        if not path:return
        try:
            with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.writer(stream);writer.writerow(["主帧序号","相对时间_秒","独立身份评分","计划槽位状态"])
                for frame in self.frames:
                    writer.writerow([frame.slot,frame.time_s,STATUS_LABELS.get(frame.identity,"其他"),"已采集" if frame.slot_status=="CAPTURED" else "未完整采集"])
            self.status_label.setText("帧指标已导出。")
        except OSError:self.status_label.setText("导出失败，请选择可写入的位置。")

    def open_folder(self):
        if self.run:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.run.directory)))

    def has_running_job(self):
        return self.worker is not None and self.worker.isRunning()
