"""Select one AirSim curve experiment, then compare explicitly selected records.

The panel does not launch a simulation on startup. The user chooses a method
and clicks Run. Comparisons only consume selected historical run/group pairs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANS = ROOT / "reports/control_targets_fixed_matched_20261030_v3/airsim_replay_plan.json"
DEFAULT_CONFIG = ROOT / "configs/curve_abom_airsim_control_targets_v12.json"
TITLES = {
    "bezier_ordinary": "Bézier 未优化参考",
    "hermite_ordinary": "Hermite 未优化参考",
    "bezier_abom": "Bézier + ABOM",
    "hermite_abom": "Hermite + ABOM",
    "bezier_abom_improved_candidate": "Bézier + 改进候选",
    "hermite_abom_improved_candidate": "Hermite + 改进候选",
}
STATE_PATH = ROOT / ".runtime/curve_airsim_panel/state.json"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def curve_order_note(plan):
    """Describe the declared construction, without classifying by method ID.

    Older ordinary references can have nonzero deformation. A quintic baseline
    can also be stored in a degree-six/seven representation by elevation.
    """
    family=plan.get('family');parameters=plan.get('parameters')
    expected={'single_bump':4,'paired_bump':7}.get(family)
    if not expected or not isinstance(parameters,list) or len(parameters)!=expected:return '次数未核验'
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in parameters):return '次数未核验'
    stored=6 if family=='single_bump' else 7
    if all(x==0 for x in parameters[:-1]):return f'五次端状态基线，按{stored}次表示'
    if family=='paired_bump' and any(x!=0 for x in parameters[3:6]):return '至多七次端状态保持曲线'
    return f'至多六次端状态保持曲线，按{stored}次表示'


def catalogue(path):
    document = load_json(path)
    plans = document.get("plans", [])
    ids = [item.get("group_id") for item in plans]
    if document.get("schema_version") != 1 or len(ids) != 6 or set(ids) != set(TITLES):
        raise ValueError("请选择含有当前六种方法的完整轨迹方案文件。")
    return plans


def replay_directory(path):
    path = Path(path).resolve()
    if (path / "replay/manifest.json").is_file():
        return path / "replay"
    if (path / "manifest.json").is_file():
        return path
    raise ValueError(f"没有找到实验记录：{path}")


def history_records(paths):
    """Read each actual/selected slot once without mutating source evidence."""
    records, seen, errors = [], set(), []
    for path in paths:
        try:
            replay = replay_directory(path)
            if str(replay) in seen:
                continue
            seen.add(str(replay))
            manifest = load_json(replay / "manifest.json")
            if manifest.get("probe_only"):
                continue
            plans = manifest.get("plans_document", {}).get("plans", [])
            plan_by_id = {p["group_id"]: p for p in plans}
            selected = manifest.get("selected_group_ids", list(plan_by_id))
            reports = load_json(replay / "rows.json") if (replay / "rows.json").is_file() else []
            report_by_id = {r["group_id"]: r for r in reports}
            for group_id in selected:
                report = report_by_id.get(group_id, {})
                metrics = report.get("metrics", {})
                status = report.get("status", "NOT_RUN")
                if not report and manifest.get('status')=='RUNNING':
                    status='RUNNING' if (replay/group_id).is_dir() else 'PENDING'
                passed = metrics.get("control_error_gates_passed")
                records.append(dict(run_path=str(replay), run_name=replay.parent.name,
                    group_id=group_id, title=TITLES.get(group_id, group_id),
                    time=report.get("started_at_utc", manifest.get("started_at_utc", "")),
                    status=status, control_passed=passed,
                    rmse=metrics.get("position_rmse_m"), peak=metrics.get("max_position_error_m"),
                    terminal=metrics.get("terminal_position_error_m"),
                    seed=plan_by_id.get(group_id, {}).get("seed"),
                    scene_id=manifest.get('config',{}).get('target_scene',manifest.get('config',{}).get('sphere_scene',{})).get('scene_id','旧版空地图'),
                    task_success=metrics.get('task_success'),clearance=metrics.get('measured_chord_min_envelope_clearance_m')))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(f"{path}: {exc}")
    records.sort(key=lambda r:(r["time"], r["group_id"]), reverse=True)
    return records, errors


def exact_selections(records):
    grouped = defaultdict(list)
    for record in records:
        group_id = record["group_id"]
        if group_id not in TITLES:
            raise ValueError("未知方法不能加入当前对比。")
        path = str(replay_directory(record["run_path"]))
        if group_id not in grouped[path]:
            grouped[path].append(group_id)
    return [dict(run_path=path, group_ids=ids) for path, ids in grouped.items()]


def launch_command(group_id, plans, config, output, *, show_window=True, shell=None):
    if group_id not in TITLES:
        raise ValueError("请选择一个有效的方法。")
    if group_id not in {p["group_id"] for p in catalogue(plans)}:
        raise ValueError("选择的方法不在轨迹方案中。")
    if not Path(config).is_file():
        raise ValueError("没有找到实验配置文件。")
    shell = shell or shutil.which("pwsh") or shutil.which("powershell.exe")
    if not shell:
        raise ValueError("没有找到 PowerShell。")
    command = [str(shell), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts/run-airsim-curve-replay-pilot.ps1"),
               "-Plans", str(Path(plans).resolve()), "-Config", str(Path(config).resolve()),
               "-OutputDir", str(Path(output).resolve()), "-Groups", group_id]
    if show_window:
        command.append("-ShowWindow")
    return command


def number(value):
    return "—" if value is None else f"{float(value):.4f}"


class ExperimentPanel(ttk.Frame):
    def __init__(self, master, *, root=ROOT):
        super().__init__(master, padding=18)
        self.pack(fill="both", expand=True)
        self.workspace = Path(root)
        self.messages = queue.Queue()
        self.busy = False
        self.rows = {}
        self.checked = set()
        self.extra_paths = []
        self.last_comparison = None
        self.continuous_remaining=0
        self.campaign=None
        self.planning_client=None
        self.generated_goal_text=None
        self.generated_goal_exact=None
        self.plans_path = tk.StringVar(value=str(DEFAULT_PLANS))
        self.config_path = tk.StringVar(value=str(DEFAULT_CONFIG))
        self.method = tk.StringVar(value=TITLES["hermite_ordinary"])
        self.show_window = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="请选择本次运行的方法；打开面板不会启动仿真。")
        self.preview = tk.StringVar()
        self.goal_value=tk.StringVar(value='8, 0, 5')
        self.seed_value=tk.StringVar(value='1101')
        self.repeat_value=tk.StringVar(value='3')
        self._read_state()
        self._build()
        self.update_preview()
        self.refresh_history()
        self.after(150, self._drain)

    def _read_state(self):
        try:
            state = load_json(STATE_PATH)
            self.extra_paths = state.get("imported_run_paths", [])
        except (OSError, ValueError, TypeError):
            pass

    def _save_state(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(dict(imported_run_paths=self.extra_paths), ensure_ascii=False, indent=2), encoding="utf-8")

    def _build(self):
        ttk.Label(self, text="AirSim 轨迹实验", style="Title.TLabel").pack(anchor="w")
        ttk.Label(self, text="纯控制 · 在简化模型中生成参考，再选择一种交 AirSim 执行；勾选历史记录对比。", style="Subtitle.TLabel").pack(anchor="w", pady=(4,16))
        settings = ttk.LabelFrame(self, text="本次运行", padding=12)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="方法").grid(row=0,column=0,sticky="w",padx=(0,12))
        self.method_box = ttk.Combobox(settings, textvariable=self.method, state="readonly", values=list(TITLES.values()), width=34)
        self.method_box.grid(row=0,column=1,sticky="w")
        self.method_box.bind("<<ComboboxSelected>>", lambda event:self.update_preview())
        ttk.Checkbutton(settings,text="显示 AirSim 窗口",variable=self.show_window).grid(row=0,column=2,padx=14,sticky="w")
        self.run_button = ttk.Button(settings,text="运行所选方法",style="Accent.TButton",command=self.start_run)
        self.run_button.grid(row=0,column=3,sticky="e")
        ttk.Label(settings,text="轨迹方案").grid(row=1,column=0,sticky="w",pady=(10,0))
        ttk.Entry(settings,textvariable=self.plans_path).grid(row=1,column=1,columnspan=2,sticky="ew",pady=(10,0))
        ttk.Button(settings,text="选择文件…",command=self.pick_plans).grid(row=1,column=3,padx=(10,0),pady=(10,0))
        ttk.Label(settings,textvariable=self.preview,style="Subtitle.TLabel",wraplength=1100).grid(row=2,column=0,columnspan=4,sticky="w",pady=(8,0))
        generator=ttk.Frame(settings);generator.grid(row=3,column=0,columnspan=4,sticky='w',pady=(8,0))
        ttk.Label(generator,text='目标 x,y,高度').pack(side='left')
        ttk.Entry(generator,textvariable=self.goal_value,width=25).pack(side='left',padx=5)
        ttk.Label(generator,text='种子').pack(side='left')
        ttk.Entry(generator,textvariable=self.seed_value,width=7).pack(side='left',padx=5)
        ttk.Button(generator,text='固定目标生成方案',command=lambda:self.generate_targets(False)).pack(side='left',padx=4)
        ttk.Button(generator,text='生成随机目标',command=lambda:self.generate_targets(True)).pack(side='left',padx=4)
        ttk.Spinbox(generator,from_=1,to=30,textvariable=self.repeat_value,width=3).pack(side='left',padx=4)
        ttk.Button(generator,text='连续测所选方法',command=self.start_continuous).pack(side='left',padx=4)
        ttk.Button(generator,text='本次后停止',command=self.stop_continuous).pack(side='left',padx=4)
        ttk.Label(self,textvariable=self.status,wraplength=1100).pack(anchor="w",pady=(12,10))
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x")
        ttk.Label(toolbar,text="历史结果",style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar,text="刷新",command=self.refresh_history).pack(side="right",padx=(8,0))
        ttk.Button(toolbar,text="导入结果文件夹…",command=self.import_run).pack(side="right")
        ttk.Label(self,text="勾选左侧要比较的记录，也可按 Ctrl / Shift 多选；勾选项优先。重复运行分别保留。",style="Subtitle.TLabel").pack(anchor="w",pady=(5,7))
        history = ttk.Frame(self)
        history.pack(fill="both",expand=True)
        columns=("check","method","scene","run","status","rmse","peak","terminal","gate")
        self.tree=ttk.Treeview(history,columns=columns,show="headings",selectmode="extended",height=5)
        headings=("对比","方法","场景","运行批次","执行状态","RMSE (m)","峰值误差 (m)","终端位置 (m)","误差判据")
        for col,label,width in zip(columns,headings,(45,170,120,210,90,85,90,90,85)):
            self.tree.heading(col,text=label)
            self.tree.column(col,width=width,minwidth=45 if col=="check" else 75,anchor="w" if col in ("method","run") else "center",stretch=col!="check")
        self.tree.bind("<Button-1>",self.toggle_check,add=True)
        scrollbar=ttk.Scrollbar(history,orient="vertical",command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left",fill="both",expand=True)
        scrollbar.pack(side="right",fill="y")
        self.tree.tag_configure("control_fail",foreground="#9B423A")
        self.tree.tag_configure("not_run",foreground="#7A8692")
        actions=ttk.Frame(self)
        actions.pack(fill="x",pady=(10,9))
        self.compare_button=ttk.Button(actions,text="比较选中结果",style="Accent.TButton",command=self.start_comparison)
        self.compare_button.pack(side="left")
        ttk.Button(actions,text="清除勾选",command=self.clear_checks).pack(side="left",padx=(8,0))
        ttk.Button(actions,text="打开所选结果",command=self.open_selected).pack(side="left",padx=8)
        ttk.Button(actions,text="重跑选中记录",command=self.rerun_selected).pack(side="left",padx=(0,8))
        self.open_compare_button=ttk.Button(actions,text="打开最近对比",command=self.open_comparison,state="disabled")
        self.open_compare_button.pack(side="left")
        ttk.Label(actions,text="流程完成与控制误差通过分别记录。",style="Subtitle.TLabel").pack(side="right")
        self.log=tk.Text(self,height=3,font=("Consolas",9),wrap="word",background="#F4F7F9",relief="flat",padx=8,pady=7)
        self.log.pack(fill="x")

    def selected_group(self):
        return next(key for key,value in TITLES.items() if value==self.method.get())

    def update_preview(self):
        try:
            plan=next(p for p in catalogue(self.plans_path.get()) if p["group_id"]==self.selected_group())
            cfg=load_json(self.config_path.get());scene=cfg.get('target_scene',cfg.get('sphere_scene',{})).get('scene_id','空地图开发验证')
            goal=plan.get('boundary',{}).get('p1')
            goal_text='' if goal is None else ' · 目标('+','.join(f'{x:.2f}' for x in goal)+')m'
            gates=cfg.get('control_error_gates',{})
            criteria=('控制通过需同时满足：峰值≤'+str(gates.get('sampled_peak_position_m','未设'))+'m；终端位置≤'
                +str(gates.get('terminal_position_m','未设'))+'m；终端速度≤'+str(gates.get('terminal_velocity_mps','未设'))+'m/s。')
            self.preview.set(f"{scene} · 协议{cfg.get('protocol_revision')} · {self.method.get()} · 种子{plan['seed']} · T={plan['parameters'][-1]:.3f}s"+goal_text+'\n'+curve_order_note(plan)+'；'+criteria)
        except Exception as exc:
            self.preview.set(str(exc))

    def pick_plans(self):
        if self.busy:return
        path=filedialog.askopenfilename(title="选择完整轨迹方案",filetypes=[("JSON文件","*.json")])
        if path:
            try: catalogue(path)
            except Exception as exc:
                messagebox.showerror("方案不可用",str(exc)); return
            self.plans_path.set(path)
            paired=Path(path).parent/'airsim_config.json'
            if paired.is_file():self.config_path.set(str(paired))
            elif (Path(path).parent/'case.json').is_file():
                cfg=load_json(DEFAULT_CONFIG);cfg['target_scene']=load_json(Path(path).parent/'case.json')
                derived=ROOT/'.runtime/curve_airsim_panel'/f'case_config_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
                derived.parent.mkdir(parents=True,exist_ok=True);derived.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
                self.config_path.set(str(derived))
            self.update_preview()

    def refresh_history(self):
        selected={(self.rows[i]["run_path"],self.rows[i]["group_id"]) for i in self.tree.selection() if i in self.rows}
        candidates=list(self.extra_paths)
        reports=self.workspace/"reports"
        if reports.is_dir():
            candidates.extend(str(path) for path in reports.iterdir() if path.is_dir() and (path/"replay/manifest.json").is_file())
        records,errors=history_records(candidates)
        self.checked.intersection_update((record["run_path"],record["group_id"]) for record in records)
        self.tree.delete(*self.tree.get_children())
        self.rows={}
        for i,record in enumerate(records):
            key=f"row{i}"
            self.rows[key]=record
            status={"COMPLETED":"流程完成","FAILED":"执行异常","NOT_RUN":"未运行","RUNNING":"运行中","PENDING":"待执行"}.get(record["status"],record["status"])
            gate="通过" if record["control_passed"] is True else "未通过" if record["control_passed"] is False else "—"
            tags=("not_run",) if record["status"]=="NOT_RUN" else ("control_fail",) if gate=="未通过" or record["status"]=="FAILED" else ()
            checked="☑" if (record["run_path"],record["group_id"]) in self.checked else "☐"
            self.tree.insert("","end",iid=key,values=(checked,record["title"],record['scene_id'],record["run_name"],status,
                number(record["rmse"]),number(record["peak"]),number(record["terminal"]),gate),tags=tags)
            if (record["run_path"],record["group_id"]) in selected:
                self.tree.selection_add(key)
        if errors:
            self._log("部分记录未载入："+"; ".join(errors[:3]))

    def toggle_check(self,event):
        if self.tree.identify_column(event.x)!="#1": return
        row_id=self.tree.identify_row(event.y)
        if row_id not in self.rows: return
        row=self.rows[row_id]
        pair=(row["run_path"],row["group_id"])
        if pair in self.checked: self.checked.remove(pair)
        else: self.checked.add(pair)
        self.tree.set(row_id,"check","☑" if pair in self.checked else "☐")
        return "break"

    def clear_checks(self):
        self.checked.clear()
        for row_id in self.rows: self.tree.set(row_id,"check","☐")
        if self.tree.selection(): self.tree.selection_remove(*self.tree.selection())

    def import_run(self):
        path=filedialog.askdirectory(title="选择实验结果目录（或其中的replay目录）")
        if not path: return
        try: replay=replay_directory(path)
        except Exception as exc:
            messagebox.showerror("没有找到记录",str(exc)); return
        if str(replay) not in self.extra_paths:
            self.extra_paths.append(str(replay)); self._save_state()
        self.refresh_history()

    def _log(self,text):
        self.log.insert("end",str(text)+"\n")
        if int(self.log.index("end-1c").split(".")[0])>350:
            self.log.delete("1.0","100.0")
        self.log.see("end")

    def _set_busy(self,value):
        self.busy=value
        for button in (self.run_button,self.compare_button):
            button.configure(state="disabled" if value else "normal")

    def generate_targets(self,random_target,*,run_after=False):
        if self.busy:return
        try:
            seed=int(self.seed_value.get())
            if not 0<=seed<2**31:raise ValueError('种子须为0到2147483647之间的整数。')
            goal=None if random_target else [float(x.strip()) for x in self.goal_value.get().replace('，',',').split(',')]
            if not random_target and self.goal_value.get()==self.generated_goal_text:goal=self.generated_goal_exact
            from control_target_experiments import make_case
            make_case(seed,goal)
        except Exception as exc:
            messagebox.showerror('目标参数错误',str(exc));return
        output=self.workspace/'reports'/f'control_target_plans_{datetime.now():%Y%m%d_%H%M%S_%f}'
        self._set_busy(True)
        self.status.set('正在简化模型中生成六组参考方案；此步骤不启动AirSim飞行。')
        self._log(f'方案生成，种子 {seed}，目录 {output}')
        if self.campaign and self.campaign['status']=='RUNNING':
            self.campaign['active_planning_path']=str(output);self._save_campaign()
        def worker():
            try:
                from control_target_planner_client import PlannerClient
                if self.planning_client is None:self.planning_client=PlannerClient()
                receipt=self.planning_client.generate(output,seed,goal)
                cfg=load_json(DEFAULT_CONFIG);cfg['target_scene']=load_json(output/'case.json')
                (output/'airsim_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
                self.messages.put(('plans_done',dict(output=str(output),run_after=run_after,receipt=receipt)))
            except Exception as exc:self.messages.put(('error',str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def start_continuous(self):
        if self.busy:return
        try:
            count=int(self.repeat_value.get());seed=int(self.seed_value.get())
            if not 1<=count<=30:raise ValueError('连续测试次数须为1至30。')
            if not 0<=seed<=2**31-count:raise ValueError('种子范围无效。')
        except Exception as exc:messagebox.showerror('连续测试参数错误',str(exc));return
        self.continuous_remaining=count
        self.campaign=dict(method=self.selected_group(),status='RUNNING',seeds=list(range(seed,seed+count)),runs=[],
            scope='Independent rest-to-rest episodes with reset; not continuous rolling flight')
        self.campaign_path=self.workspace/'reports'/f'control_campaign_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
        self._save_campaign()
        self._continue_campaign()

    def _continue_campaign(self):
        # A queued Tk callback may run after Stop was clicked. Recheck the
        # frozen campaign rather than starting work from editable widgets.
        if self.busy or not self.continuous_remaining or not self.campaign or self.campaign['status']!='RUNNING':return
        index=len(self.campaign['runs'])
        if index>=len(self.campaign['seeds']):return
        self.method.set(TITLES[self.campaign['method']])
        self.seed_value.set(str(self.campaign['seeds'][index]))
        self.generate_targets(True,run_after=True)

    def _save_campaign(self):
        if self.campaign:
            self.campaign_path.parent.mkdir(parents=True,exist_ok=True)
            self.campaign_path.write_text(json.dumps(self.campaign,ensure_ascii=False,indent=2),encoding='utf-8')

    def stop_continuous(self):
        if not self.campaign or self.campaign['status'] not in ('RUNNING','STOP_AFTER_CURRENT'):
            self._log('当前没有连续测试任务。')
            return
        self.continuous_remaining=0
        self.campaign['status']='STOP_AFTER_CURRENT' if self.busy else 'STOPPED'
        self._save_campaign()
        self._log('已取消后续自动测试；当前任务继续保存结果。')

    def close_panel(self):
        self.continuous_remaining=0
        if self.campaign and self.campaign['status'] in ('RUNNING','STOP_AFTER_CURRENT'):
            self.campaign['status']='UI_CLOSED_CURRENT_TASK_MAY_CONTINUE'
            self._save_campaign()
        if self.planning_client is not None:self.planning_client.close()
        self.master.destroy()

    def start_run(self):
        if self.busy: return
        output=self.workspace/"reports"/f"airsim_selected_{datetime.now():%Y%m%d_%H%M%S_%f}"
        try:
            group_id=self.selected_group()
            command=launch_command(group_id,self.plans_path.get(),self.config_path.get(),output,show_window=self.show_window.get())
        except Exception as exc:
            messagebox.showerror("不能启动",str(exc)); return
        self._set_busy(True)
        self.status.set(f"正在运行：{TITLES[group_id]}。其他方法不会自动启动。")
        self._log("本次方法："+group_id)
        self._log("结果目录："+str(output))
        if self.campaign and self.campaign['status']=='RUNNING':
            self.campaign['active_run_path']=str(output);self._save_campaign()
        def worker():
            env=os.environ.copy(); env.update(PYTHONUTF8="1",PYTHONIOENCODING="utf-8",PYTHONPATH=str(ROOT/"python"))
            try:
                # The child owns a file handle, so closing the panel cannot
                # break its stdout pipe while the isolated experiment finishes.
                log_path=output.with_suffix(".launcher.log")
                log_path.parent.mkdir(parents=True,exist_ok=True)
                self.messages.put(("log","启动器日志："+str(log_path)))
                with log_path.open("wb") as log_writer:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=log_writer,stderr=subprocess.STDOUT,
                        env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
                with log_path.open("r",encoding="utf-8",errors="replace") as log_reader:
                    while True:
                        line=log_reader.readline()
                        if line:
                            self.messages.put(("log",line.rstrip()))
                        elif process.poll() is not None:
                            break
                        else:
                            time.sleep(.15)
                code=process.wait()
                self.messages.put(("run_done",dict(code=code,output=str(output),group_id=group_id)))
            except Exception as exc:
                self.messages.put(("error",str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def start_comparison(self):
        if self.busy: return
        records=([row for row in self.rows.values() if (row["run_path"],row["group_id"]) in self.checked]
                 if self.checked else [self.rows[i] for i in self.tree.selection() if i in self.rows])
        if len(records)<2:
            messagebox.showinfo("选择对比记录","请勾选至少两条结果，也可按 Ctrl 或 Shift 多选。")
            return
        try: selections=exact_selections(records)
        except Exception as exc:
            messagebox.showerror("不能比较",str(exc)); return
        output=self.workspace/"reports"/f"airsim_comparison_{datetime.now():%Y%m%d_%H%M%S_%f}"
        self._set_busy(True)
        self.status.set(f"正在比较选中的 {len(records)} 条记录；原始结果保持在各自批次目录。")
        def worker():
            try:
                from compare_airsim_curve_runs import build_comparison
                build_comparison([],output,selections=selections)
                self.messages.put(("compare_done",str(output)))
            except Exception as exc:
                self.messages.put(("error",str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def _drain(self):
        try:
            while True:
                kind,value=self.messages.get_nowait()
                if kind=="log": self._log(value)
                elif kind=="run_done":
                    self._set_busy(False); self.refresh_history()
                    self.status.set(("已结束所选方法，请查看误差判据并选择对比记录。" if value["code"]==0 else "本次运行异常，失败记录已保留；请查看日志。"))
                    self._log("运行结束，退出码："+str(value["code"]))
                    if self.campaign and self.campaign['status'] in ('RUNNING','STOP_AFTER_CURRENT'):
                        self.campaign['runs'].append(value);self.continuous_remaining=max(0,self.continuous_remaining-1)
                        if value['code']!=0:self.continuous_remaining=0;self.campaign['status']='FAILED_STOPPED'
                        elif not self.continuous_remaining:self.campaign['status']='COMPLETED' if self.campaign['status']=='RUNNING' else 'STOPPED'
                        self._save_campaign()
                        if self.continuous_remaining:
                            self.after(250,self._continue_campaign)
                    for key,row in self.rows.items():
                        if Path(row["run_path"]).parent==Path(value["output"]):
                            self.tree.selection_set(key); self.tree.see(key)
                elif kind=='plans_done':
                    output=Path(value['output']);self.plans_path.set(str(output/'airsim_replay_plan.json'));self.config_path.set(str(output/'airsim_config.json'))
                    self.generated_goal_exact=load_json(output/'case.json')['boundary']['p1']
                    self.generated_goal_text=', '.join(f'{x:.5f}' for x in self.generated_goal_exact)
                    self.goal_value.set(self.generated_goal_text)
                    self._set_busy(False);self.update_preview()
                    self.status.set('六组参考方案已生成；请选择一种运行，也可固定本方案反复测试。')
                    self._log('方案：'+str(output/'airsim_replay_plan.json'))
                    receipt=value.get('receipt')
                    if receipt:self._log(f"规划进程{'复用导入环境' if receipt['runtime_warm'] else '冷启动'}，完整规划墙钟 {receipt['parent_total_planning_wall_s']:.3f}s；网络权重和目标评价均重新计算。")
                    if value['run_after'] and self.continuous_remaining:
                        if self.campaign:self.method.set(TITLES[self.campaign['method']]);self.update_preview()
                        self.start_run()
                    elif value['run_after'] and self.campaign:
                        self.campaign['status']='STOPPED_BEFORE_FLIGHT';self._save_campaign()
                elif kind=="compare_done":
                    self._set_busy(False)
                    self.last_comparison=Path(value)
                    self.open_compare_button.configure(state="normal")
                    self.status.set("对比已生成：包含指标表、统计说明、轨迹和误差图。")
                    self._log("对比报告："+str(self.last_comparison/"comparison.md"))
                    self.open_comparison()
                elif kind=="error":
                    self.continuous_remaining=0
                    if self.campaign:self.campaign['status']='FAILED_STOPPED';self.campaign['error']=value;self._save_campaign()
                    self._set_busy(False)
                    self.status.set("操作未完成，请查看原因；实验条件不同的结果不能混合比较。")
                    self._log(value)
                    messagebox.showerror("操作未完成",value)
        except queue.Empty:
            pass
        self.after(150,self._drain)

    def open_selected(self):
        rows=[self.rows[i] for i in self.tree.selection() if i in self.rows]
        if rows: os.startfile(rows[0]["run_path"])

    def rerun_selected(self):
        if self.busy:return
        rows=[self.rows[i] for i in self.tree.selection() if i in self.rows]
        if not rows:rows=[r for r in self.rows.values() if (r['run_path'],r['group_id']) in self.checked]
        if len(rows)!=1:
            messagebox.showinfo('选择重跑记录','请只选择或勾选一条要重跑的记录。');return
        try:
            folder=Path(rows[0]['run_path']);manifest=load_json(folder/'manifest.json')
            cfg=next(folder/s['snapshot'] for s in manifest['sources'] if s['path'].endswith('.json') and load_json(folder/s['snapshot'])==manifest['config'])
            plans=next(s['path'] for s in manifest['sources'] if s['path'].endswith('.json') and load_json(folder/s['snapshot'])==manifest['plans_document'])
            if load_json(plans)!=manifest['plans_document']:raise ValueError('原始完整方案文件已改变，不能按原条件重跑。')
            self.config_path.set(str(cfg));self.plans_path.set(plans);self.method.set(TITLES[rows[0]['group_id']]);self.update_preview()
            self.start_run()
        except Exception as exc:messagebox.showerror('不能重跑',str(exc))

    def open_comparison(self):
        if self.last_comparison and self.last_comparison.is_dir():
            os.startfile(str(self.last_comparison))


def create_panel(*,withdraw=False):
    window=tk.Tk()
    window.title("AirSim 轨迹实验 - 选择与对比")
    width=min(1220,max(980,window.winfo_screenwidth()-80))
    height=min(800,max(650,window.winfo_screenheight()-120))
    window.geometry(f"{width}x{height}")
    window.minsize(min(1050,width),min(680,height))
    if withdraw: window.withdraw()
    style=ttk.Style(window)
    style.theme_use("clam")
    style.configure(".",font=("Microsoft YaHei UI",10))
    style.configure("Title.TLabel",font=("Microsoft YaHei UI",19,"bold"),foreground="#17394D")
    style.configure("Section.TLabel",font=("Microsoft YaHei UI",12,"bold"))
    style.configure("Subtitle.TLabel",font=("Microsoft YaHei UI",9),foreground="#5B6E7B")
    style.configure("Accent.TButton",background="#176E88",foreground="white",padding=(12,7))
    style.map("Accent.TButton",background=[("active","#145C72"),("disabled","#829AA4")])
    style.configure("Treeview",rowheight=28,font=("Microsoft YaHei UI",9))
    style.configure("Treeview.Heading",font=("Microsoft YaHei UI",9,"bold"))
    panel=ExperimentPanel(window)
    window.protocol('WM_DELETE_WINDOW',panel.close_panel)
    return window,panel


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true",help="Build widgets without showing a window or launching AirSim")
    args=parser.parse_args()
    window,panel=create_panel(withdraw=args.self_test)
    if args.self_test:
        window.update_idletasks()
        assert str(panel.method_box["state"])=="readonly"
        assert str(panel.tree["selectmode"])=="extended"
        assert not panel.busy and len(panel.rows)>=0
        print(json.dumps(dict(status="PASS",catalogue_size=len(TITLES),history_rows=len(panel.rows),simulation_launched=False)))
        window.destroy()
    else:
        window.mainloop()
