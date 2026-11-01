"""Native Qt workspace layout and an embedded, zoomable result gallery."""
import json,shutil
from pathlib import Path
from PyQt5.QtCore import Qt,QUrl,QSettings,QTimer,pyqtSignal
from PyQt5.QtGui import QPixmap,QDesktopServices
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QFormLayout,QScrollArea,QFrame,QSplitter,QTabWidget,QComboBox,QGroupBox,QFileDialog,QSizePolicy)
from .experiment_library_ui import DataLibrary

STYLE='''
QMainWindow { background:#eef2f5; } QWidget { color:#243d4f; font-size:12px; }
QLabel#workspaceTitle { font-size:24px; font-weight:700; color:#173447; }
QLabel#muted { color:#718393; font-size:11px; }
QFrame#workspaceCard { background:white; border:1px solid #dce5ea; border-radius:9px; }
QFrame#transport { background:white; border:1px solid #d3e0e8; border-radius:9px; }
QTabWidget::pane { border:0; background:transparent; }
QTabBar::tab { background:transparent; padding:12px 25px; color:#718393; border-bottom:3px solid transparent; font-size:13px; }
QTabBar::tab:selected { color:#126c79; border-bottom:3px solid #147d86; font-weight:600; }
QTabBar::tab:hover { background:#e3ecef; }
QPushButton { min-height:25px; border:1px solid #d1dfe6; border-radius:5px; background:white; padding:5px 10px; color:#34566b; }
QPushButton:hover { background:#edf5f6; border-color:#91b7bc; } QPushButton:disabled { color:#a0adb5; background:#f3f6f8; border-color:#e0e7eb; }
QPushButton#primary { background:#137c84; color:white; border-color:#137c84; font-weight:600; } QPushButton#primary:hover { background:#106b72; }
QPushButton#primary:disabled { background:#a7c7c9; border-color:#a7c7c9; color:#f0f6f6; }
QPushButton#stop { color:#b14645; border-color:#e6c6c4; background:#fffafa; }
QComboBox,QDoubleSpinBox,QSpinBox,QLineEdit { border:1px solid #cedde5; border-radius:5px; background:white; padding:5px; min-height:23px; selection-background-color:#147d86; }
QTableView { background:white; alternate-background-color:#f5f8fa; border:1px solid #dce5ea; border-radius:5px; selection-background-color:#dfedf0; selection-color:#173d4b; }
QHeaderView::section { background:#eaf0f4; color:#597284; font-weight:600; padding:9px 5px; border:0; border-bottom:1px solid #d8e2e8; }
QPlainTextEdit { background:white; border:1px solid #d5e1e8; border-radius:5px; padding:7px; }
QGroupBox { border:1px solid #dce5ea; border-radius:7px; margin-top:12px; padding:12px 8px 7px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 5px; }
QScrollArea { border:0; background:transparent; }
QScrollBar:vertical { width:8px; background:#edf2f5; } QScrollBar::handle:vertical { background:#b5c8d3; border-radius:4px; min-height:25px; }
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical { height:0; }
'''

class ResultGallery(QWidget):
    close_requested=pyqtSignal()
    def __init__(self):
        super().__init__();self.page=None;self.folder=None;self.figures=[];self.original=QPixmap();self.zoom=1.;self.scale_mode='fit'
        layout=QVBoxLayout(self);layout.setContentsMargins(0,12,0,0)
        row=QHBoxLayout();self.group=QComboBox();self.group.setMinimumWidth(180);self.choice=QComboBox();self.choice.setMinimumWidth(250)
        row.addWidget(QLabel('条件组'));row.addWidget(self.group);row.addWidget(self.choice,1)
        self.fit=QComboBox()
        for title,value in [('适合窗口','fit'),('适合宽度','width'),('原始尺寸','actual')]:self.fit.addItem(title,value)
        self.fit.activated.connect(lambda _:self.scale(self.fit.currentData()));row.addWidget(self.fit)
        minus=QPushButton('−');minus.clicked.connect(lambda:self.scale(.8));row.addWidget(minus);plus=QPushButton('＋');plus.clicked.connect(lambda:self.scale(1.25));row.addWidget(plus)
        export=QPushButton('导出当前图');export.clicked.connect(self.export);row.addWidget(export);web=QPushButton('完整数据页');web.clicked.connect(self.open_page);row.addWidget(web);layout.addLayout(row)
        self.scroll=QScrollArea();self.scroll.setWidgetResizable(False);self.scroll.setAlignment(Qt.AlignHCenter|Qt.AlignTop);self.image=QLabel('在“数据管理”勾选飞行记录，生成图表。');self.image.setAlignment(Qt.AlignTop|Qt.AlignHCenter);self.image.setStyleSheet('background:white;padding:12px;');self.scroll.setWidget(self.image);layout.addWidget(self.scroll,1)
        self.caption=QLabel('固定颜色对应方法；不同规则和条件分别比较。图来自原始日志，不做平滑或改变指标。');self.caption.setWordWrap(True);self.caption.setObjectName('muted')
        footer=QHBoxLayout();footer.addWidget(self.caption,1);close=QPushButton('关闭当前结果');close.clicked.connect(self.close_requested.emit);footer.addWidget(close);layout.addLayout(footer)
        self.group.currentIndexChanged.connect(self.set_group);self.choice.currentIndexChanged.connect(self.set_figure)
    def load(self,page):
        self.page=Path(page);parent=self.page.parent;folders=[parent] if (parent/'figure_index.json').is_file() else [p.parent for p in sorted(parent.glob('*/figure_index.json'))]
        self.group.blockSignals(True);self.group.clear()
        for folder in folders:self.group.addItem(('逐点停稳' if folder.name.startswith('stop') else '连续过点' if folder.name.startswith('pass') else folder.name),str(folder))
        self.group.blockSignals(False)
        if folders:self.set_group()
        else:self.image.setText('该结果为批量清单或旧图页，请点击“完整数据页”查看。');self.image.adjustSize();self.choice.clear()
    def set_group(self):
        if not self.group.currentData():return
        self.folder=Path(self.group.currentData());self.figures=json.loads((self.folder/'figure_index.json').read_text(encoding='utf-8'))
        self.choice.blockSignals(True);self.choice.clear()
        for figure in self.figures:self.choice.addItem(figure['title'])
        self.choice.blockSignals(False);self.set_figure()
    def set_figure(self):
        index=self.choice.currentIndex()
        if not 0<=index<len(self.figures):return
        self.original=QPixmap(str(self.folder/self.figures[index]['png']));self.scale(self.fit.currentData())
        self.caption.setText(self.figures[index]['title']+' · '+str(self.folder)+' · 可导出PNG、SVG、PDF；原始数值不变。')
    def scale(self,factor):
        if self.original.isNull():return
        self.scale_mode=factor if isinstance(factor,str) else 'manual'
        width=max(200,self.scroll.viewport().width()-30)/self.original.width();height=max(150,self.scroll.viewport().height()-30)/self.original.height()
        value=min(width,height) if factor=='fit' else width if factor=='width' else 1. if factor=='actual' else self.zoom*factor
        self.zoom=max(.05,min(3.,value))
        size=self.original.size()*self.zoom;self.image.setPixmap(self.original.scaled(size,Qt.KeepAspectRatio,Qt.SmoothTransformation));self.image.resize(size.width()+24,size.height()+24)
    def showEvent(self,event):
        super().showEvent(event)
        if self.scale_mode in ('fit','width'):QTimer.singleShot(0,lambda mode=self.scale_mode:self.scale(mode))
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.scale_mode in ('fit','width'):QTimer.singleShot(0,lambda mode=self.scale_mode:self.scale(mode))
    def export(self):
        index=self.choice.currentIndex()
        if not self.folder or not 0<=index<len(self.figures):return
        name=self.figures[index]['name'];path,_=QFileDialog.getSaveFileName(self,'导出当前图',str(self.folder/(name+'.png')),'PNG高清图 (*.png);;SVG矢量图 (*.svg);;PDF矢量图 (*.pdf)')
        if path:
            target=Path(path);source=self.folder/(name+target.suffix.lower())
            if source.is_file() and source.resolve()!=target.resolve():shutil.copy2(source,target);self.caption.setText('已导出：'+str(target))
    def open_page(self):
        if self.page:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.page)))

class RouteWorkspace(QWidget):
    def __init__(self,window,root):
        super().__init__();self.owner=window;self.root=Path(root);old=window.takeCentralWidget();self.parking=old;old.setParent(self);old.hide()
        self.buttons={b.text():b for b in old.findChildren(QPushButton)}
        window.setCentralWidget(self);window.setStyleSheet(window.styleSheet()+STYLE)
        layout=QVBoxLayout(self);layout.setContentsMargins(22,15,22,12);layout.setSpacing(10)
        header=QHBoxLayout();titles=QVBoxLayout();title=QLabel('无人机控制实验室');title.setObjectName('workspaceTitle');titles.addWidget(title)
        hint=QLabel('AIRSIM  /  六种方法  /  两组多点任务');hint.setObjectName('muted');titles.addWidget(hint);header.addLayout(titles);header.addStretch()
        header.addWidget(window.launch_button);header.addWidget(window.connect_button);header.addWidget(window.scene_button);header.addWidget(window.badge);layout.addLayout(header)
        self.tabs=QTabWidget();layout.addWidget(self.tabs,1)
        control=QWidget();body=QHBoxLayout(control);body.setContentsMargins(0,10,0,0);body.setSpacing(14)
        settings=QWidget();settings.setMinimumWidth(278);settings.setMaximumWidth(315);settings_layout=QVBoxLayout(settings);settings_layout.setContentsMargins(0,0,0,0);settings_layout.setSpacing(9)
        def card(title):
            frame=QFrame();frame.setObjectName('workspaceCard');frame.setAttribute(Qt.WA_StyledBackground,True);frame.setStyleSheet('QFrame#workspaceCard{background:white;border:1px solid #dce5ea;border-radius:8px;}');box=QVBoxLayout(frame);box.setContentsMargins(14,12,14,12);label=QLabel(title);label.setObjectName('sectionTitle');box.addWidget(label);settings_layout.addWidget(frame);return box
        window.method.setMinimumContentsLength(12);window.method.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        method=card('01  选择实验');form=QFormLayout();form.setSpacing(8);form.addRow('过点规则',window.policy);form.addRow('曲线 / 方法',window.method);form.addRow('随机种子',window.seed);method.addLayout(form)
        route=card('02  共同航点');route.addWidget(window.goal);window.goal.setMaximumWidth(16777215);route.addWidget(self.buttons['编辑多点路线']);window.path_label.setObjectName('muted');route.addWidget(window.path_label)
        row=QHBoxLayout();row.addWidget(self.buttons['生成当前组路线']);row.addWidget(self.buttons['随机多点路线']);route.addLayout(row);route.addWidget(self.buttons['加载已有方案'])
        params=card('03  执行参数');form=QFormLayout();form.setSpacing(7)
        for label,widget in [('速度上限 m/s',window.inputs['max_speed']),('指令变化率 m/s²',window.acceleration),('参考时间倍率',window.time_scale),('过点速度 m/s',window.pass_speed),('停留时间 s',window.dwell)]:form.addRow(label,widget)
        params.addLayout(form)
        advanced=QGroupBox('场地边界（展开修改）');advanced.setCheckable(True);advanced.setChecked(False);adv=QWidget();form=QFormLayout(adv)
        for label,lo,hi in [('X范围','x_min','x_max'),('Y范围','y_min','y_max'),('高度范围','height_min','height_max')]:
            row=QHBoxLayout();row.addWidget(window.inputs[lo]);row.addWidget(QLabel('至'));row.addWidget(window.inputs[hi]);form.addRow(label,row)
        form.addRow(window.apply_button);adv.hide();advanced.toggled.connect(adv.setVisible);box=QVBoxLayout(advanced);box.addWidget(adv);settings_layout.addWidget(advanced);settings_layout.addStretch()
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);scroll.setWidget(settings);scroll.setMinimumWidth(295);scroll.setMaximumWidth(330);body.addWidget(scroll)
        self.settings_scroll=scroll
        view=QWidget();vl=QVBoxLayout(view);vl.setContentsMargins(0,0,0,0);vl.setSpacing(10)
        tiles=QHBoxLayout()
        for key in ('speed','altitude','distance','hits','yaw_error_deg'):tiles.addWidget(window.metrics[key])
        vl.addLayout(tiles);window.view.setMinimumSize(350,180);frame=QFrame();frame.setObjectName('workspaceCard');fl=QVBoxLayout(frame);fl.setContentsMargins(0,0,0,0);fl.addWidget(window.view);vl.addWidget(frame,1)
        status=QHBoxLayout();status.addWidget(window.position_label);status.addStretch()
        for title,plane in [('俯视等比例','xy'),('正视等比例','xz'),('三维','3d')]:
            button=QPushButton(title);button.clicked.connect(lambda _,p=plane:window.view.set_plane(p));status.addWidget(button)
        expand=QPushButton('展开视图');expand.setCheckable(True)
        def expand_view(checked):
            scroll.setVisible(not checked)
            for key in ('speed','altitude','distance','hits','yaw_error_deg'):window.metrics[key].setVisible(not checked)
            expand.setText('恢复布局' if checked else '展开视图')
        expand.toggled.connect(expand_view);status.addWidget(expand)
        vl.addLayout(status)
        window.source.setObjectName('muted');window.dynamics_status.setObjectName('muted');vl.addWidget(window.source);vl.addWidget(window.dynamics_status);vl.addWidget(window.scene_status);body.addWidget(view,1)
        self.tabs.addTab(control,'实验控制');self.library=DataLibrary(window,root);self.tabs.addTab(self.library,'数据管理');self.gallery=ResultGallery();self.gallery.close_requested.connect(self.close_results);self.tabs.addTab(self.gallery,'图表对比')
        transport=QFrame();transport.setObjectName('transport');bar=QVBoxLayout(transport);bar.setContentsMargins(14,10,14,10);bar.setSpacing(5);row=QHBoxLayout()
        for button in (window.start_button,window.pause_button,window.stop_button):row.addWidget(button)
        row.addSpacing(10);row.addWidget(self.buttons['勾选批量运行']);row.addWidget(self.buttons['本次后停止批量']);row.addStretch();window.data_button.setText('原始日志');row.addWidget(window.data_button);row.addWidget(window.figure_button);row.addWidget(window.batch_results);row.addWidget(self.buttons['数据怎么读']);bar.addLayout(row)
        window.batch_label.setObjectName('muted');bar.addWidget(window.batch_label);layout.addWidget(transport)
        window.message.setMinimumHeight(18);window.message.setMaximumHeight(38);layout.addWidget(window.message)
        window.footer.setWordWrap(True);window.footer.setMaximumHeight(34);window.footer.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred);window.footer.setObjectName('muted');layout.addWidget(window.footer)
        self.settings=QSettings('forAirsim','ControlExperimentWorkspace');geometry=self.settings.value('geometry')
        window.setMinimumSize(980,680)
        if geometry:window.restoreGeometry(geometry)
        else:
            from PyQt5.QtWidgets import QApplication
            screen=QApplication.primaryScreen().availableGeometry();window.resize(max(980,min(1400,screen.width()-30)),max(680,min(920,screen.height()-45)))
        from PyQt5.QtWidgets import QApplication
        available=QApplication.primaryScreen().availableGeometry();window.resize(min(window.width(),max(980,available.width()-30)),min(window.height(),max(680,available.height()-45)))
        default=self.root/'reports/control_view_protocol15_final/index.html'
        if not default.is_file():default=self.root/'reports/route_dashboard_v2/index.html'
        if default.is_file():self.gallery.load(default)
    def show_results(self,page):
        self.owner.last_results=Path(page);self.gallery.load(page);self.tabs.setCurrentWidget(self.gallery);self.owner._set_message('结果已在“图表对比”打开；可切换实验组、缩放或导出。')
    def close_results(self):
        self.owner.last_results=None;self.gallery.page=None;self.gallery.folder=None;self.gallery.figures=[];self.gallery.original=QPixmap();self.gallery.group.clear();self.gallery.choice.clear();self.gallery.image.clear();self.gallery.image.setText('结果已关闭，可返回数据管理选择记录。');self.gallery.image.adjustSize();self.tabs.setCurrentWidget(self.library)
    def refresh(self):
        state=self.owner.runtime.snapshot()
        if state.get('run_dir'):
            self.owner.footer.setText('本次数据：'+state['run_dir']+'  |  底部“原始日志”打开目录；图表在“图表对比”页查看。')
            self.owner.footer.setToolTip(state['run_dir'])
        else:self.owner.footer.setText('启动仿真 → 选择方法 → 开始飞行；底部“原始日志”打开本次目录，数据管理可检索历史记录。')
        self.library.set_operable(not self.owner.locked())
        if self.library.busy:self.owner.start_button.setEnabled(False)
        if not self.library.busy and not self.owner.locked():
            paths=[]
            if state.get('result_ready') and state.get('run_dir'):paths.append(state['run_dir'])
            if self.owner.campaign and not self.owner.campaign.active:
                paths.extend(s['run_dir'] for s in self.owner.campaign.slots if s.get('run_dir'));paths.append(str(self.owner.campaign.folder))
            paths.extend(str(p.parent) for p in self.owner.packages.values())
            if self.owner.last_results:paths.append(str(self.owner.last_results.parent))
            known={r['path'] for r in self.library.model.rows};missing=[p for p in dict.fromkeys(paths) if str(Path(p).resolve()) not in known and Path(p).is_dir()]
            if missing:self.library.job('自动归类新记录',lambda:self.library.store.index_paths(missing))
    def save(self):self.settings.setValue('geometry',self.owner.saveGeometry())
