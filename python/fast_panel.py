"""Qt5 control and timestamped playback for the forAirsim baseline and TOPPQuad."""
import argparse
import hashlib
from bisect import bisect_right
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from PyQt5.QtCore import Qt,QTimer,QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (QApplication,QComboBox,QHBoxLayout,QLabel,QLineEdit,QMainWindow,
                             QPushButton,QSlider,QVBoxLayout,QWidget)

from dual_target_control.prepared_route_live import PreparedRouteRuntime
from dual_target_control.route_view import RouteFlightView
from dual_target_control.ui import STATUS_LABELS
from dual_target_control.simulator import ROOT
from prepared_route_planner import prepared_spec


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def configuration(policy,geometry,arm,start=None,points=None,*,sweep=None):
    preset=read(ROOT/"configs/fast_route.json")
    limits=preset["command_limits"]
    task=dict(points=preset["waypoints"] if points is None else points,start=preset["start"] if start is None else start,
              policy=policy,seed=preset["seed"],pass_speed=preset["pass_speed_mps"],dwell_s=preset["dwell_s"],
              position_error_limit_m=preset["position_error_limit_m"],
              tracking=preset["tracking"],retiming=preset["retiming"],
              time_scale=preset["time_scale"],max_speed=limits["max_speed_mps"],max_slew=limits["max_slew_mps2"])
    if sweep:
        task["retiming"] = dict(task["retiming"], continuous_profile="fixed",
                                fractions=[sweep["fraction"]], frozen_source=sweep["source"])
    spec=prepared_spec(**task)
    cfg=read(ROOT/"configs/dual_target_control.json")
    cfg["control"].update(max_speed=task["max_speed"],max_acceleration=task["max_slew"])
    cfg["airsim"]["initial_height"]=task["start"][2]
    method=f"{geometry}_{arm}"
    cfg["preparation"]=dict(task=task,method=method,protocol_revision=15)
    cfg["curve_experiment"]=dict(plan=dict(group_id=method,route=spec,planning_pending=True),
                                  time_scale=task["time_scale"],auto_show_scene=True,execution_id=uuid.uuid4().hex)
    return cfg


def combo(items):
    box=QComboBox()
    for label,value in items:
        box.addItem(label,value)
    return box


class Panel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AirSim 轨迹实验 · forAirsim 快基线与 TOPPQuad")
        self.resize(1060,760)
        self.cfg=configuration("stop","hermite","ordinary")
        self.runtime=PreparedRouteRuntime(self.cfg)
        self.previous_dir=None
        self.pending=None
        self.suite=None
        self.replay=None
        self.feedback=""
        self.closing=False
        self.verification_exit=0
        widget=QWidget();self.setCentralWidget(widget);layout=QVBoxLayout(widget)
        row=QHBoxLayout();layout.addLayout(row)
        self.policy=combo([("逐点停稳","stop"),("连续过点","pass")])
        self.geometry=combo([("Hermite","hermite"),("Bézier","bezier")])
        self.arm=combo([("普通基线","ordinary"),("TOPPQuad","toppquad")])
        for title,box in (("任务",self.policy),("曲线",self.geometry),("计时",self.arm)):
            row.addWidget(QLabel(title));row.addWidget(box)
        row.addStretch()
        self.launch=QPushButton("启动 / 连接 AirSim");row.addWidget(self.launch)
        self.launch.clicked.connect(self.launch_sim)
        show=QPushButton("显示 UE 场景");row.addWidget(show);show.clicked.connect(self.runtime.show_scene)
        task=self.cfg["preparation"]["task"]; edit=QHBoxLayout();layout.addLayout(edit)
        self.start=QLineEdit(json.dumps(task["start"]));self.start.setMaximumWidth(150)
        self.points=QLineEdit(json.dumps(task["points"]))
        edit.addWidget(QLabel("准备点 [X,Y,高度]"));edit.addWidget(self.start)
        edit.addWidget(QLabel("有序航点（米）"));edit.addWidget(self.points)
        self.start_button=QPushButton("开始所选飞行");edit.addWidget(self.start_button)
        self.stop_button=QPushButton("停止并降落");edit.addWidget(self.stop_button)
        self.start_button.clicked.connect(self.fly);self.stop_button.clicked.connect(self.stop)
        self.view=RouteFlightView(self.cfg);layout.addWidget(self.view,1)
        self.status=QLabel("待启动");self.status.setWordWrap(True);layout.addWidget(self.status)
        history=QHBoxLayout();layout.addLayout(history)
        self.records=QComboBox();self.records.setMinimumContentsLength(26)
        history.addWidget(QLabel("本项目飞行记录"));history.addWidget(self.records,1)
        self.play=QPushButton("回放");self.play.clicked.connect(self.toggle_replay);history.addWidget(self.play)
        self.slider=QSlider(Qt.Horizontal);self.slider.setEnabled(False);self.slider.sliderMoved.connect(self.seek);history.addWidget(self.slider,1)
        folder=QPushButton("打开数据");folder.clicked.connect(self.open_record);history.addWidget(folder)
        self.comparison=QPushButton("查看批次对比");history.addWidget(self.comparison)
        self.comparison.clicked.connect(self.open_comparison)
        self.records.currentIndexChanged.connect(self.clear_replay)
        self.refresh_records()
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)

    def refresh_records(self):
        self.records.clear()
        for p in sorted((ROOT/"logs").glob("dual_target_airsim_*/summary.json"),reverse=True):
            try:
                r=read(p); task=r["config"]["preparation"]["task"]
            except (OSError,KeyError,ValueError):
                continue
            accepted=r.get("metrics") or {}
            result="通过" if r.get("outcome")=="COMPLETED" and accepted.get("task_success") else "未通过"
            group="逐点停稳" if task["policy"]=="stop" else "连续过点"
            geometry,arm=r["method"].rsplit("_",1)
            plan=r["config"].get("curve_experiment", {}).get("plan", {})
            method=("普通基线" if arm=="ordinary" else
                    "TOPPQuad 分段" if plan.get("retiming_profile") else "TOPPQuad 固定比例")
            stamp=p.parent.name.removeprefix("dual_target_airsim_")
            self.records.addItem(f"{stamp}  {group}  {geometry.title()} / {method}  {result}",str(p.parent))

    def clear_replay(self):
        self.replay=None
        self.feedback=""
        self.play.setText("回放")
        self.slider.setEnabled(False)

    def open_record(self):
        path=self.records.currentData() or str(ROOT/"logs")
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def open_comparison(self):
        page=max((ROOT/"reports").glob("verification_*/index.html"),default=None)
        if page:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(page)))
        else:
            self.feedback="当前还没有已完成的对比批次"

    def stop(self):
        if self.suite is not None:self.suite["remaining"].clear()
        self.runtime.stop()

    def launch_sim(self):
        self.clear_replay()
        self.runtime.start_simulator()

    def fly(self):
        snap=self.runtime.snapshot()
        if self.pending or snap["running"] or not snap["connected"]:return
        try:
            cfg=configuration(self.policy.currentData(),self.geometry.currentData(),self.arm.currentData(),
                              json.loads(self.start.text()),json.loads(self.points.text()),
                              sweep=self.suite.get("current_sweep") if self.suite else None)
        except (ValueError,TypeError) as exc:
            self.feedback=str(exc)
            return
        self.clear_replay()
        self.cfg=cfg;self.previous_dir=snap.get("run_dir")
        self.pending=cfg["curve_experiment"]["execution_id"]
        self.runtime.configure(cfg);self.runtime.start()

    def toggle_replay(self):
        if self.pending or self.runtime.snapshot()["running"]:return
        if self.replay is not None:
            self.replay["playing"]=not self.replay["playing"]
            if self.slider.value()>=self.slider.maximum():self.slider.setValue(0)
            self.seek(self.slider.value());self.play.setText("暂停" if self.replay["playing"] else "继续");return
        directory=self.records.currentData()
        if not directory:return
        p=Path(directory);states=p/"live_states.jsonl"
        if not states.is_file():
            self.feedback="这次记录没有执行样本"
            return
        try:
            with states.open(encoding="utf8") as f: rows=[json.loads(line)["state"] for line in f if line.strip()]
            cfg=read(p/"config.json")
        except (OSError,KeyError,ValueError) as exc:
            self.feedback=str(exc)
            return
        if not rows:
            self.feedback="这次记录没有执行样本"
            return
        self.feedback=""
        self.replay=dict(rows=rows,times=[r["reference_elapsed_s"] for r in rows],cfg=cfg,playing=True,origin=time.monotonic(),
                         path=[r["reference_position_zup_m"] for r in rows],trail=[r["position_zup_m"] for r in rows])
        self.slider.setRange(0,len(rows)-1);self.slider.setValue(0);self.slider.setEnabled(True);self.play.setText("暂停")

    def seek(self,i):
        if self.replay:self.replay["origin"]=time.monotonic()-self.replay["times"][i]

    def tick(self):
        snap=self.runtime.snapshot(); running=snap["running"]
        if self.pending and snap.get("execution_id")==self.pending:
            directory=Path(snap.get("run_dir") or "."); ended=directory/"summary.json"
            if not running and snap["status"] in ("STOPPED","ERROR") and ended.is_file() and str(directory)!=self.previous_dir:
                report=read(ended)
                if report["config"]["curve_experiment"]["execution_id"]==self.pending:
                    self.pending=None;self.refresh_records()
                    if self.suite is not None:self.record_trial(directory,report)
            elif not running and snap["status"]=="ERROR" and (str(directory)==self.previous_dir or not ended.is_file()):
                self.pending=None
                if self.suite is not None:self.abort_suite(snap["message"])
        busy=bool(self.pending or running or self.suite is not None)
        for control in (self.geometry,self.policy,self.arm,self.start,self.points,self.launch,self.play,self.records):
            control.setEnabled(not busy)
        self.start_button.setEnabled(not busy and snap["connected"])
        if self.replay is not None:
            rep=self.replay
            if rep["playing"]:
                i=max(0,bisect_right(rep["times"],time.monotonic()-rep["origin"])-1)
                self.slider.setValue(i)
                if i==len(rep["times"])-1:rep["playing"]=False;self.play.setText("重播")
            i=self.slider.value();s=rep["rows"][i]
            self.view.set_snapshot(dict(config=rep["cfg"],connected=True,position=s["position_zup_m"],yaw=s["yaw_rad"],
                                        velocity=s["velocity_zup_mps"],trail=rep["trail"][:i+1],planned_path=rep["path"]))
            self.status.setText(f"实测回放　{rep['times'][i]:.2f} s　速度 {math.hypot(*s['velocity_zup_mps']):.2f} m/s")
        else:
            self.view.set_snapshot(snap)
            self.status.setText(self.feedback or f"{STATUS_LABELS.get(snap['status'].lower(),snap['status'])}　{snap['message']}　速度 {math.hypot(*snap['velocity']):.2f} m/s")
        if self.suite is not None and not self.pending and not running:
            if snap["status"]=="ERROR" and not (snap["connected"] and self.suite["results"] and self.suite["results"][-1]["safe_landing"]):
                self.abort_suite(snap["message"])
            elif snap["connected"] and snap["status"] in ("READY","STOPPED","ERROR"):self.next_trial()

    def begin_suite(self,jobs,*,sweep_source=None):
        if any(job[2] == "toppquad-fixed" for job in jobs):
            self.arm.addItem("TOPPQuad 固定比例（对照）", "toppquad-fixed")
        prefix = "retiming_sweep_" if sweep_source else "verification_"
        dest=ROOT/"reports"/(prefix+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        dest.mkdir(parents=True)
        self.suite=dict(remaining=jobs.copy(),results=[],dir=dest,expected=len(jobs))
        if sweep_source:
            self.suite.update(sweep_source=sweep_source, requested_fractions=[job[3] for job in jobs])
        self.launch_sim()

    def next_trial(self):
        s=self.suite
        if not s["remaining"]:
            self.finish_suite();return
        policy,geom,arm,*fraction=s["remaining"].pop(0)
        if fraction:
            s["current_sweep"] = dict(fraction=fraction[0], source=s["sweep_source"])
        for box,val in ((self.policy,policy),(self.geometry,geom),(self.arm,arm)):box.setCurrentIndex(box.findData(val))
        self.fly()

    def record_trial(self,directory,r):
        s=self.suite;c=read(directory/"cleanup.json") if (directory/"cleanup.json").exists() else {}
        safe=all(c.get(k) for k in ("landed","disarmed","api_control_released"))
        s["results"].append(dict(method=r["method"],policy=r["config"]["preparation"]["task"]["policy"],outcome=r["outcome"],
                                 passed=bool(r["outcome"]=="COMPLETED" and (r.get("metrics") or {}).get("task_success") and safe and r.get("sources_unchanged")),safe_landing=safe,
                                 samples=r["samples"],run=str(directory),metrics=r.get("metrics"),error=r.get("error")))
        if not safe:self.abort_suite("未确认落地，已结束批次")
        elif s.get("current_sweep"):
            s["results"][-1]["requested_fraction"] = s["current_sweep"]["fraction"]
            if not s["results"][-1]["passed"]:
                self.abort_suite("当前档位未通过，停止提高计时比例")
            else:
                self.save_suite()
        else:self.save_suite()

    def save_suite(self):
        s=self.suite
        (s["dir"]/"results.json").write_text(json.dumps({k:v for k,v in s.items() if k!="dir"},ensure_ascii=False,indent=2),encoding="utf8")

    def abort_suite(self,message):
        self.suite["error"]=message;self.finish_suite()

    def finish_suite(self):
        self.verification_exit=int(len(self.suite["results"])!=self.suite["expected"] or
                                   not all(r["passed"] for r in self.suite["results"]))
        self.save_suite()
        from flight_comparison import render
        verified=render(self.suite["dir"])
        if verified["passed"]!=verified["requested"]:
            self.verification_exit=1
        if sys.stdout:print(self.suite["dir"],flush=True)
        self.suite=None
        QTimer.singleShot(300,self.close)

    def closeEvent(self,event):
        if not self.closing:
            self.closing=True;self.runtime.shutdown()
        if not self.runtime.wait_closed(0):
            event.ignore();QTimer.singleShot(250,self.close);return
        self.timer.stop();event.accept()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    action=parser.add_mutually_exclusive_group()
    action.add_argument("--verify-flights",nargs="+",metavar="POLICY/GEOMETRY/ARM")
    action.add_argument("--replay",type=Path,metavar="RUN_DIRECTORY")
    action.add_argument("--retiming-sweep",nargs="+",type=float,metavar="FRACTION")
    parser.add_argument("--retiming-source",type=Path,metavar="LIVE_MANIFEST")
    parser.add_argument("--waypoints", type=json.loads, metavar="JSON", help="Ordered waypoints for --verify-flights")
    args=parser.parse_args()
    if bool(args.retiming_sweep) != bool(args.retiming_source):
        parser.error("--retiming-sweep and --retiming-source are required together")
    if args.retiming_sweep and (not all(0 < f <= 1 for f in args.retiming_sweep)
                                 or args.retiming_sweep != sorted(set(args.retiming_sweep))):
        parser.error("Retiming fractions must be strictly increasing and in (0, 1]")
    if args.check:os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    app=QApplication(sys.argv[:1]);panel=Panel();panel.show()
    if args.check:
        assert not panel.runtime.snapshot()["connected"] and panel.arm.count()==2
        QTimer.singleShot(100,panel.close)
    if args.verify_flights:
        jobs=[s.split("/") for s in args.verify_flights]
        if any(len(j)!=3 or j[0] not in ("stop","pass") or j[1] not in ("hermite","bezier")
               or j[2] not in ("ordinary","toppquad","toppquad-fixed")
               or (j[2]=="toppquad-fixed" and j[0]!="pass") for j in jobs):parser.error("Invalid flight selection")
        if args.waypoints is not None:
            try:
                for policy in {job[0] for job in jobs}:
                    configuration(policy, "hermite", "ordinary", points=args.waypoints)
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
            panel.points.setText(json.dumps(args.waypoints))
        QTimer.singleShot(500,lambda:panel.begin_suite(jobs))
    elif args.waypoints is not None:
        parser.error("--waypoints requires --verify-flights")
    if args.retiming_sweep:
        path = args.retiming_source.resolve()
        raw = path.read_bytes()
        plan = json.loads(raw)["plan"]
        if plan["group_id"] not in ("hermite_toppquad", "bezier_toppquad") or plan["route"]["policy"] != "pass":
            parser.error("Retiming sweep requires one completed continuous TOPPQuad flight manifest")
        source = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())
        jobs = [("pass",plan["representation"],"toppquad",fraction) for fraction in args.retiming_sweep]
        QTimer.singleShot(500,lambda:panel.begin_suite(jobs,sweep_source=source))
    if args.replay:
        index=panel.records.findData(str(args.replay.resolve()))
        if index<0:
            panel.runtime.shutdown()
            panel.runtime.wait_closed(5)
            parser.error("所选目录不在本项目飞行记录中")
        panel.records.setCurrentIndex(index)
        QTimer.singleShot(150,panel.toggle_replay)
    code=app.exec_()
    return code or panel.verification_exit


if __name__=="__main__":
    raise SystemExit(main())
