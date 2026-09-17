"""Tkinter workbench for local language planning, AirSim and dataset review."""
import argparse
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from console_backend import ConsoleBackend, ROOT, read_snapshot, snapshot_id
from language_dataset import DatasetStore
from language_plan import preview_steps

LABELS = {
    'visual_alignment': 'A视觉对准 / 观察', 'visual_range': 'A接近至预设距离',
    'clarify': '需要补充任务信息', 'unsupported': '当前能力不支持',
}
STEPS = {
    'takeoff_both': '两机起飞', 'staging_B_standby': '集结，B待命',
    'A_fixed_observation_point': 'A到固定观察点', 'A_visual_alignment': 'A视觉对准',
    'A_approach_to_configured_range': 'A接近至预设距离停止',
    'return_both': '两机返回', 'land_both': '两机降落并释放控制',
}

def label_plan(label):
    kind = next(key for key, value in LABELS.items() if value == label)
    if kind in ('visual_alignment', 'visual_range'):
        return dict(disposition='ACCEPT', routine=kind, vehicle='DroneA', reason='OK')
    return dict(disposition='CLARIFY' if kind == 'clarify' else 'UNSUPPORTED', routine='none', vehicle='none',
                reason='MISSING_INFORMATION' if kind == 'clarify' else 'CAPABILITY_UNAVAILABLE')

def plan_label(plan):
    if plan.get('disposition') == 'ACCEPT':
        return LABELS.get(plan.get('routine'), '')
    return LABELS['clarify' if plan.get('disposition') == 'CLARIFY' else 'unsupported']

class ExperimentConsole:
    def __init__(self, root):
        self.root = root
        self.root.title('AirSim 研究工作台 · 小模型 / 任务 / 数据')
        self.root.geometry('1320x870')
        self.root.minsize(1120, 800)
        self.root.configure(bg='#f1f5f7')
        self.queue = queue.Queue()
        self.backend = ConsoleBackend(lambda line: self.queue.put(('log', line)))
        self.store = DatasetStore(ROOT/'datasets/language_tasks_v0')
        self.busy = False
        self.sampling = False
        self.pause_pending = False
        self.last_plan = None
        self.approved = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value='就绪 · 本地CPU小模型 / 固定Blocks场景')
        self.photo_refs = {}
        self._style()
        self._build()
        self._refresh_rows()
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(100, self._drain)

    def _style(self):
        self.root.option_add('*Font', ('Microsoft YaHei UI', 10))
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', font=('Microsoft YaHei UI', 10), background='#f1f5f7', foreground='#182f40')
        s.configure('TFrame', background='#f1f5f7')
        s.configure('TButton', padding=(12, 8), background='#e2eaee', borderwidth=0)
        s.configure('Accent.TButton', background='#137d83', foreground='white')
        s.map('Accent.TButton', background=[('active', '#0f6268'), ('disabled', '#bfcece')])
        s.configure('Stop.TButton', background='#b7403c', foreground='white')
        s.configure('TNotebook.Tab', padding=(22, 10))
        s.configure('Treeview', rowheight=33, background='white', fieldbackground='white')
        s.configure('Treeview.Heading', font=('Microsoft YaHei UI', 10, 'bold'))
        s.configure('TLabelframe.Label', font=('Microsoft YaHei UI', 10, 'bold'))

    def _build(self):
        header = tk.Frame(self.root, bg='#142d40', height=80)
        header.pack(fill='x')
        tk.Label(header, text='AirSim 研究工作台', fg='white', bg='#142d40',
                 font=('Microsoft YaHei UI', 21, 'bold')).pack(anchor='w', padx=24, pady=(12, 0))
        tk.Label(header, text='自然语言计划  /  仿真执行  /  数据审核', fg='#a8c4d6', bg='#142d40').pack(anchor='w', padx=25, pady=(0, 12))
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill='both', expand=True, padx=18, pady=(14, 8))
        control = ttk.Frame(self.tabs, padding=14)
        data = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(control, text='任务控制')
        self.tabs.add(data, text='数据审核')
        self._control_tab(control)
        self._data_tab(data)
        footer = ttk.Frame(self.root)
        footer.pack(fill='x', padx=20, pady=(0, 10))
        ttk.Label(footer, textvariable=self.status).pack(side='left')
        ttk.Button(footer, text='打开本次日志', command=lambda: os.startfile(self.backend.folder)).pack(side='right')

    def _control_tab(self, parent):
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill='x', pady=(0, 12))
        self.operation_buttons = []
        for label, callback in [('启动 Blocks', lambda: self.job(self.backend.start_simulator)),
                                ('准备固定气球', lambda: self.flight_job(self.backend.prepare_scene)),
                                ('回收并降落', lambda: self.flight_job(self.backend.recover))]:
            button = ttk.Button(toolbar, text=label, command=callback)
            button.pack(side='left', padx=(0, 8))
            self.operation_buttons.append(button)
        self.refresh_button = ttk.Button(toolbar, text='读取双机画面 / 状态', command=self.refresh_snapshot)
        self.refresh_button.pack(side='left')
        self.pause_button = ttk.Button(toolbar, text='暂停仿真', style='Stop.TButton', command=self.pause)
        self.pause_button.pack(side='right')
        columns = ttk.Frame(parent)
        columns.pack(fill='both', expand=True)
        columns.columnconfigure(0, weight=4)
        columns.columnconfigure(1, weight=6)
        columns.rowconfigure(0, weight=1)
        left = ttk.Frame(columns, padding=(0, 0, 16, 0))
        right = ttk.Frame(columns)
        left.grid(row=0, column=0, sticky='nsew')
        right.grid(row=0, column=1, sticky='nsew')
        ttk.Label(left, text='01  输入任务', font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w')
        ttk.Label(left, text='支持：固定红气球的观察或距离停止；两机起降，B待命。', wraplength=440).pack(anchor='w', pady=(4, 7))
        self.instruction = tk.Text(left, height=4, width=47, wrap='word', relief='flat', padx=12, pady=10)
        self.instruction.pack(fill='x')
        self.instruction.insert('1.0', '让A对准中心红气球，然后返回降落。')
        self.instruction.edit_modified(False)
        self.instruction.bind('<<Modified>>', self.input_changed)
        actions = ttk.Frame(left)
        actions.pack(fill='x', pady=10)
        self.plan_button = ttk.Button(actions, text='用小模型解析', style='Accent.TButton', command=self.request_plan)
        self.plan_button.pack(side='left')
        ttk.Button(actions, text='存为待审核样本', command=self.capture_candidate).pack(side='left', padx=8)
        ttk.Label(left, text='02  核对计划', font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w', pady=(10, 6))
        self.plan_text = tk.Text(left, height=13, width=47, wrap='word', relief='flat', padx=12, pady=10)
        self.plan_text.pack(fill='both', expand=True)
        self._text(self.plan_text, '等待模型解析。\n\n模型：Qwen3-0.6B Q8_0\n部署：本机CPU；尚未微调\n\n模型可能遗漏附加条件。\n请对照原始指令核对完整动作。')
        self.confirm_check = ttk.Checkbutton(left, text='我已核对当前指令与完整计划', variable=self.approved, command=self._buttons)
        self.confirm_check.pack(anchor='w', pady=(12, 8))
        self.execute_button = ttk.Button(left, text='在 AirSim 执行当前计划', style='Accent.TButton', command=self.execute)
        self.execute_button.pack(fill='x')
        ttk.Label(left, text='计划经校验后调用已有脚本；暂不支持未知区域搜索。', wraplength=440).pack(anchor='w', pady=7)
        ttk.Label(right, text='双机观察', font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w')
        ttk.Label(right, text='按需读取相机快照，时间戳显示在下方。').pack(anchor='w', pady=(3, 8))
        cameras = ttk.Frame(right)
        cameras.pack(fill='x')
        self.cameras = {}
        for col, name in enumerate(('DroneA', 'DroneB')):
            cameras.columnconfigure(col, weight=1)
            panel = ttk.Frame(cameras, padding=(0, 0, 6 if col == 0 else 0, 0))
            panel.grid(row=0, column=col, sticky='nsew')
            ttk.Label(panel, text=name+' · front_center').pack(anchor='w')
            canvas = tk.Canvas(panel, width=300, height=185, bg='#203c50', highlightthickness=0)
            canvas.pack(fill='x', pady=5)
            canvas.create_text(150, 92, text='等待读取 AirSim 画面', fill='#9bb5c5', tags='placeholder')
            self.cameras[name] = canvas
        self.telemetry = tk.StringVar(value='尚未连接 · 点击“启动 Blocks”，然后读取画面。')
        ttk.Label(right, textvariable=self.telemetry, wraplength=610, justify='left').pack(fill='x', pady=(4, 12))
        ttk.Label(right, text='运行日志', font=('Microsoft YaHei UI', 13, 'bold')).pack(anchor='w')
        self.log_text = tk.Text(right, height=15, width=65, bg='#142d40', fg='#d4e6ed', insertbackground='white',
                                wrap='word', relief='flat', padx=12, pady=10, font=('Consolas', 10))
        self.log_text.pack(fill='both', expand=True, pady=(6, 0))
        self.log_text.configure(state='disabled')
        self._buttons()

    def _data_tab(self, parent):
        ttk.Label(parent, text='把真实指令和模型错误，变成可追溯的训练样本',
                  font=('Microsoft YaHei UI', 14, 'bold')).pack(anchor='w')
        self.data_summary = tk.StringVar()
        ttk.Label(parent, textvariable=self.data_summary).pack(anchor='w', pady=(6, 12))
        panel = ttk.Frame(parent)
        panel.pack(fill='both', expand=True)
        panel.columnconfigure(0, weight=6)
        panel.columnconfigure(1, weight=4)
        panel.rowconfigure(0, weight=1)
        listing = ttk.Frame(panel)
        listing.grid(row=0, column=0, sticky='nsew', padx=(0, 16))
        self.tree = ttk.Treeview(listing, columns=('split', 'status', 'text'), show='headings', selectmode='browse')
        for field, label, width in [('split', '分组', 110), ('status', '审核', 90), ('text', '原始指令', 370)]:
            self.tree.heading(field, text=label)
            self.tree.column(field, width=width, minwidth=65)
        scroll = ttk.Scrollbar(listing, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        self.tree.pack(fill='both', expand=True)
        self.tree.bind('<<TreeviewSelect>>', self.select_row)
        editor = ttk.Frame(panel)
        editor.grid(row=0, column=1, sticky='nsew')
        ttk.Label(editor, text='原始指令 / 来源', font=('Microsoft YaHei UI', 12, 'bold')).pack(anchor='w')
        self.row_text = tk.Text(editor, height=7, width=43, wrap='word', relief='flat', padx=10, pady=10)
        self.row_text.pack(fill='x', pady=7)
        self.row_text.configure(state='disabled')
        ttk.Label(editor, text='人工确认的正确结果').pack(anchor='w')
        self.label_choice = ttk.Combobox(editor, values=list(LABELS.values()), state='readonly')
        self.label_choice.pack(fill='x', pady=(5, 12))
        ttk.Label(editor, text='审核人（必填）').pack(anchor='w')
        self.reviewer = ttk.Entry(editor)
        self.reviewer.pack(fill='x', pady=(5, 12))
        ttk.Label(editor, text='修改原因 / 备注').pack(anchor='w')
        self.note = tk.Text(editor, height=4, width=43, wrap='word', relief='flat', padx=8, pady=8)
        self.note.pack(fill='x', pady=5)
        buttons = ttk.Frame(editor)
        buttons.pack(fill='x', pady=10)
        ttk.Button(buttons, text='保存审核通过', style='Accent.TButton', command=lambda: self.save_review('approved')).pack(side='left')
        ttk.Button(buttons, text='标为不采用', command=lambda: self.save_review('rejected')).pack(side='left', padx=8)
        ttk.Button(editor, text='导出已审核训练数据', command=self.export_data).pack(fill='x', pady=10)
        ttk.Label(editor, text='仅导出已审核的 development 样本。\nsmoke_eval 留在诊断集，不进入训练。\n模型输出和AI候选标签不会自动通过审核。', wraplength=400).pack(anchor='w', pady=6)

    @staticmethod
    def _text(widget, value):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', value)
        widget.configure(state='disabled')

    def _log(self, text):
        self.log_text.configure(state='normal')
        self.log_text.insert('end', datetime.now().strftime('%H:%M:%S')+'  '+text+'\n')
        if int(self.log_text.index('end-1c').split('.')[0]) > 1000:
            self.log_text.delete('1.0', '200.0')
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    def job(self, callback, on_success=None):
        if self.busy or self.pause_pending:
            return
        self.busy = True
        self.status.set('处理中 · 详细进度见运行日志')
        self._buttons()
        def work():
            try:
                result = callback()
                self.queue.put(('done', result, on_success))
            except Exception as exc:
                self.queue.put(('error', str(exc)))
        threading.Thread(target=work, daemon=False).start()

    def flight_job(self, callback):
        epoch = self.backend.request_epoch()
        self.job(lambda: callback(request_epoch=epoch))

    def _buttons(self):
        state = 'disabled' if self.busy or self.pause_pending else 'normal'
        for button in self.operation_buttons + [self.plan_button]:
            button.configure(state=state)
        record = self.last_plan
        good = (record and record.get('schema_valid') and record.get('plan', {}).get('disposition') == 'ACCEPT'
                and record['instruction'] == self.instruction.get('1.0', 'end').strip())
        self.execute_button.configure(state='normal' if good and self.approved.get() and state == 'normal' else 'disabled')
        self.confirm_check.configure(state='normal' if good and state == 'normal' else 'disabled')

    def input_changed(self, _event=None):
        if self.instruction.edit_modified():
            self.approved.set(False)
            self.instruction.edit_modified(False)
            self._buttons()

    def request_plan(self):
        instruction = self.instruction.get('1.0', 'end').strip()
        if not instruction:
            return
        self.approved.set(False)
        self.last_plan = None
        self.job(lambda: self.backend.plan(instruction), self.show_plan)

    def show_plan(self, record):
        self.last_plan = record
        raw = record.get('raw_output') or '(无输出)'
        if record.get('schema_valid'):
            plan = record['plan']
            lines = [plan_label(plan), f"模型耗时 {record['latency_s']:.2f} s", '']
            lines += [f'{i+1}. {STEPS.get(step, step)}' for i, step in enumerate(preview_steps(plan))]
            if not preview_steps(plan):
                lines.append('当前结果不产生飞行动作。')
            lines += ['', '模型原始输出：', raw]
            value = '\n'.join(lines)
        else:
            value = '计划未通过校验，执行已禁用。\n'+record.get('validation_error', '')+'\n\n原始输出：\n'+raw
        if record['instruction'] != self.instruction.get('1.0', 'end').strip():
            value = '原指令已被修改，请重新解析。\n\n'+value
        self._text(self.plan_text, value)
        self._buttons()

    def execute(self):
        if not self.approved.get() or self.last_plan is None:
            return
        instruction = self.instruction.get('1.0', 'end').strip()
        record = json.loads(json.dumps(self.last_plan))
        token = snapshot_id(instruction, record['plan'])
        epoch = self.backend.request_epoch()
        self.approved.set(False)
        self.job(lambda: self.backend.execute(instruction, record, token, request_epoch=epoch))

    def pause(self):
        if self.pause_pending:
            return
        self.pause_pending = True
        self.backend.cancel_pending()
        self.pause_button.configure(state='disabled')
        self._buttons()
        def work():
            try:
                self.queue.put(('pause_done', self.backend.pause()))
            except Exception as exc:
                self.queue.put(('pause_done', '暂停未确认：'+str(exc)))
        threading.Thread(target=work, daemon=False).start()

    def refresh_snapshot(self):
        if self.sampling:
            return
        self.sampling = True
        self.refresh_button.configure(state='disabled')
        def work():
            try:
                self.queue.put(('snapshot', read_snapshot()))
            except Exception as exc:
                self.queue.put(('snapshot_error', str(exc)))
        threading.Thread(target=work, daemon=False).start()

    def show_snapshot(self, data):
        lines = [data['sampled_at']+' · '+('已暂停' if data['paused'] else '仿真运行中')]
        for name, item in data['vehicles'].items():
            position = ', '.join(f'{v:.2f}' for v in item['position'])
            lines.append(f"{name}  NED [{position}] m  |  {item['speed']:.2f} m/s  |  API {'已接管' if item['api_control'] else '释放'}")
            raw = item.get('image')
            canvas = self.cameras[name]
            canvas.delete('all')
            if raw:
                try:
                    photo = tk.PhotoImage(data=base64.b64encode(raw).decode('ascii'))
                    factor = max(1, (photo.width()+299)//300, (photo.height()+184)//185)
                    photo = photo.subsample(factor)
                    self.photo_refs[name] = photo
                    canvas.create_image(150, 92, image=photo)
                except tk.TclError:
                    canvas.create_text(150, 92, text='图像解码失败', fill='white')
            else:
                canvas.create_text(150, 92, text='本次相机未返回图像', fill='white')
        self.telemetry.set('\n'.join(lines))

    def capture_candidate(self):
        text = self.instruction.get('1.0', 'end').strip()
        plan = (self.last_plan.get('plan') if self.last_plan and self.last_plan['instruction'] == text else None)
        try:
            self.store.collect_candidate(text, plan, source='gui_user_instruction')
            self._refresh_rows()
            self._log('已保存待审核样本；尚未纳入训练。')
        except Exception as exc:
            messagebox.showerror('无法保存', str(exc), parent=self.root)

    def _refresh_rows(self):
        self.rows = {row['id']: row for row in self.store.rows()}
        selected = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        approved = 0
        for row_id, row in self.rows.items():
            review = self.store.review_for(row)
            decision = review['decision'] if review else 'pending'
            approved += decision == 'approved'
            text = next(m['content'] for m in row['messages'] if m['role'] == 'user')
            self.tree.insert('', 'end', iid=row_id, values=(row['split'], {'pending':'待审核','approved':'通过','rejected':'不采用'}[decision], text))
        self.data_summary.set(f'共 {len(self.rows)} 条候选  /  已审核通过 {approved} 条  ·  开发冒烟集已被查看，不是论文盲测集')
        if selected and selected[0] in self.rows:
            self.tree.selection_set(selected[0])

    def select_row(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            return
        row = self.rows[selected[0]]
        text = next(m['content'] for m in row['messages'] if m['role'] == 'user')
        review = self.store.review_for(row)
        plan = review.get('expected_plan') if review else None
        if not plan:
            answers = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
            plan = json.loads(answers[-1]) if answers else None
        self.label_choice.set(plan_label(plan) if plan else '')
        self._text(self.row_text, text+'\n\n来源：'+row['source']+'\n分组：'+row['split']+'\nID：'+row['id'])
        self.note.delete('1.0', 'end')
        if review:
            self.note.insert('1.0', review.get('note', ''))

    def save_review(self, decision):
        selected = self.tree.selection()
        if not selected:
            return
        try:
            label = self.label_choice.get()
            plan = label_plan(label) if label else None
            self.store.review(self.rows[selected[0]], decision, self.reviewer.get(), plan, self.note.get('1.0','end').strip())
            self._refresh_rows()
            self._log('审核记录已追加；原始候选保持不变。')
        except Exception as exc:
            messagebox.showerror('无法审核', str(exc), parent=self.root)

    def export_data(self):
        folder = ROOT/'datasets/language_tasks_v0/exports'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        try:
            manifest = self.store.export_training(folder)
            messagebox.showinfo('导出完成', '仅导出已审核的development样本。\n'+str(folder)+'\n\n'+json.dumps(manifest, ensure_ascii=False, indent=2), parent=self.root)
        except Exception as exc:
            messagebox.showerror('导出失败', str(exc), parent=self.root)

    def _drain(self):
        try:
            for _ in range(100):
                item = self.queue.get_nowait()
                kind = item[0]
                if kind == 'log':
                    self._log(item[1])
                elif kind in ('done', 'error'):
                    self.busy = False
                    if kind == 'done' and item[2]:
                        item[2](item[1])
                        self.status.set('模型已返回 · 请核对指令与计划')
                    else:
                        self._log(str(item[1]))
                        self.status.set(('操作结束 · ' if kind == 'done' else '操作未完成 · ')+str(item[1])[:80])
                    self._buttons()
                elif kind in ('snapshot', 'snapshot_error'):
                    self.sampling = False
                    self.refresh_button.configure(state='normal')
                    if kind == 'snapshot':
                        self.show_snapshot(item[1])
                    else:
                        self.telemetry.set('读取失败 · '+item[1])
                        self._log(item[1])
                elif kind == 'pause_done':
                    self.pause_pending = False
                    self.pause_button.configure(state='normal')
                    self.status.set(item[1])
                    self._log(item[1])
                    self._buttons()
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def close(self):
        if self.busy or self.pause_pending or self.sampling:
            messagebox.showinfo('仍有操作进行中', '请等待操作结束；飞行任务可以先暂停仿真，再检查或回收。', parent=self.root)
            return
        self.root.destroy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Build widgets and exit without model inference or flight')
    args = parser.parse_args()
    root = tk.Tk()
    app = ExperimentConsole(root)
    if args.smoke_test:
        root.update()
        assert str(app.execute_button['state']) == 'disabled'
        assert len(app.tree.get_children()) >= 26
        assert not app.backend._process
        root.destroy()
        print('GUI smoke PASS: both tabs built; execution disabled; no RPC or flight')
    else:
        root.mainloop()

if __name__ == '__main__':
    main()
