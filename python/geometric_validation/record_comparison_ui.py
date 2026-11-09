"""Idle-only selection and background rendering of existing flight records."""
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QUrl
from PyQt5.QtGui import QDesktopServices, QColor
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QStyle, QTableWidget, QTableWidgetItem,
    QVBoxLayout)

from .record_comparison import ROOT, discover_records, generate_comparison, group_records
from .trajectory_view import LABELS


class DataJob(QThread):
    def __init__(self, work, parent):
        super().__init__(parent);self.work=work;self.result=None;self.error=None

    def run(self):
        try:self.result=self.work()
        except Exception as exc:self.error=str(exc)


class RecordComparisonDialog(QDialog):
    def __init__(self, parent=None, environment_id=None, root=ROOT, scanner=None, generator=None):
        super().__init__(parent)
        self.root=Path(root);self.scanner=scanner or discover_records;self.generator=generator or generate_comparison
        self.records=[];self.checked=set();self.extra=[];self.worker=None;self.last_page=None
        self._close_requested=False;self._filter_environment=environment_id
        self.setWindowTitle('选择记录生成对比');self.resize(1240,650)
        self.setMinimumSize(760,480)
        layout=QVBoxLayout(self)
        toolbar=QHBoxLayout()
        self.environment=QComboBox();self.environment.addItem('全部环境',None)
        self.environment.currentIndexChanged.connect(self.populate)
        toolbar.addWidget(self.environment)
        self.search=QLineEdit();self.search.setPlaceholderText('搜索记录 / 路径');self.search.textChanged.connect(self.populate)
        toolbar.addWidget(self.search,1)
        self.refresh_button=QPushButton('刷新');self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.refresh_button.clicked.connect(self.scan);toolbar.addWidget(self.refresh_button)
        self.add_button=QPushButton('添加记录目录');self.add_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.add_button.clicked.connect(self.add_record);toolbar.addWidget(self.add_button)
        self.clear_button=QPushButton('清空勾选');self.clear_button.clicked.connect(self.clear_selection);toolbar.addWidget(self.clear_button)
        layout.addLayout(toolbar)
        self.table=QTableWidget(0,11)
        self.table.setHorizontalHeaderLabels(['选择','记录时间','环境','方案','完成','跟踪','过点','审计','收尾','条件','记录'])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers);self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True);self.table.verticalHeader().hide()
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(10,QHeaderView.Stretch)
        self.table.itemChanged.connect(self.selection_changed)
        self.table.currentCellChanged.connect(self.show_path)
        layout.addWidget(self.table,1)
        self.path_label=QLabel();self.path_label.setWordWrap(True);self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        self.status=QLabel('正在读取记录…');self.status.setWordWrap(True);self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        actions=QHBoxLayout()
        self.open_after=QCheckBox('生成后打开');self.open_after.setChecked(True);actions.addWidget(self.open_after)
        actions.addStretch()
        self.open_button=QPushButton('打开上次生成');self.open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.open_button.setEnabled(False);self.open_button.clicked.connect(self.open_page);actions.addWidget(self.open_button)
        self.generate_button=QPushButton('生成对比页');self.generate_button.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.generate_button.setEnabled(False);self.generate_button.clicked.connect(self.generate);actions.addWidget(self.generate_button)
        close=QPushButton('关闭');close.clicked.connect(self.reject);actions.addWidget(close)
        layout.addLayout(actions)
        self.scan()

    @property
    def busy(self):
        return self.worker is not None

    def _job(self,work,kind):
        if self.busy:return
        self._kind=kind;self.worker=DataJob(work,self)
        for widget in (self.environment,self.search,self.refresh_button,self.add_button,self.clear_button,
                       self.table,self.generate_button,self.open_button):widget.setEnabled(False)
        self.worker.finished.connect(self._finished);self.worker.start()

    def scan(self):
        if self.busy:return
        self._close_requested=False
        self.status.setText('正在读取记录…')
        extra=list(self.extra);self._job(lambda:self.scanner(self.root,extra),'scan')

    def _finished(self):
        worker=self.worker;self.worker=None
        result,error=worker.result,worker.error;worker.deleteLater()
        for widget in (self.environment,self.search,self.refresh_button,self.add_button,self.clear_button,self.table):widget.setEnabled(True)
        if not error and self._kind=='scan':
            self.records=result
            self.checked.intersection_update(r['path'] for r in self.records if r['eligible'])
            selected=self.environment.currentData() or self._filter_environment;self._filter_environment=None
            self.environment.blockSignals(True);self.environment.clear();self.environment.addItem('全部环境',None)
            labels={r['environment_id']:r['environment'] for r in self.records if r['environment_id']}
            for key,label in sorted(labels.items()):self.environment.addItem(label,key)
            index=self.environment.findData(selected);self.environment.setCurrentIndex(max(0,index));self.environment.blockSignals(False)
            self.populate()
        else:
            self.validate_selection()
        if error:self.status.setText('处理失败：'+error)
        if not error and self._kind=='generate':
            self.last_page=Path(result);self.status.setText('已生成：'+str(self.last_page))
            if self.open_after.isChecked() and not self._close_requested:self.open_page()
        self.open_button.setEnabled(self.last_page is not None)
        if self._close_requested:super().reject()

    def populate(self,*_):
        self.table.blockSignals(True);self.table.setRowCount(0)
        environment=self.environment.currentData();query=self.search.text().strip().casefold()
        visible=[r for r in self.records if (environment is None or r['environment_id']==environment)
            and (not query or query in (r['path']+' '+r['environment']+' '+r['method']).casefold())]
        def verdict(value):return '通过' if value is True else '未通过' if value is False else '未知'
        for record in visible:
            row=self.table.rowCount();self.table.insertRow(row)
            choose=QTableWidgetItem();choose.setData(Qt.UserRole,record['path'])
            choose.setFlags(Qt.ItemIsEnabled|Qt.ItemIsUserCheckable if record['eligible'] else Qt.NoItemFlags)
            choose.setCheckState(Qt.Checked if record['path'] in self.checked else Qt.Unchecked)
            choose.setToolTip(record['reason'] or record['path']);self.table.setItem(row,0,choose)
            values=[record['time'],record['environment'],LABELS.get(record['method'],record['method'])]
            values += [verdict(record[k]) for k in ('normal_completion','tracking_passed','passage_passed','audit_passed','safe_cleanup')]
            values += [record['fingerprint'][:8] or '不可用',record['id']]
            for column,value in enumerate(values,1):
                item=QTableWidgetItem(value);item.setToolTip(record['reason'] or record['path'])
                if value=='未通过':item.setForeground(QColor('#ad3444'))
                elif not record['eligible']:item.setForeground(QColor('#76818b'))
                self.table.setItem(row,column,item)
        self.table.blockSignals(False);self.validate_selection()

    def selection_changed(self,item):
        if item.column()!=0:return
        path=item.data(Qt.UserRole)
        if item.checkState()==Qt.Checked:self.checked.add(path)
        else:self.checked.discard(path)
        self.validate_selection()

    def selected_records(self):
        return [r for r in self.records if r['path'] in self.checked]

    def validate_selection(self):
        self.generate_button.setEnabled(False)
        try:
            groups=group_records(self.selected_records())
            self.status.setText(f'{len(self.checked)} 条已选 · {len(groups)} 个独立条件组')
            self.generate_button.setEnabled(not self.busy)
        except ValueError as exc:
            self.status.setText(str(exc))

    def show_path(self,row,*_):
        item=self.table.item(row,0)
        self.path_label.setText(item.data(Qt.UserRole) if item else '')

    def clear_selection(self):
        self.checked.clear();self.populate()

    def add_record(self):
        directory=QFileDialog.getExistingDirectory(self,'选择已结束的飞行记录目录',str(self.root/'logs'))
        if directory:
            self.extra.append(directory);self.scan()

    def generate(self):
        if self.busy:return
        records=self.selected_records()
        try:group_records(records)
        except ValueError as exc:self.status.setText(str(exc));return
        paths=[r['path'] for r in records]
        self.status.setText('正在核验所选记录并生成图表…')
        self._job(lambda:self.generator(paths),'generate')

    def open_page(self):
        if self.last_page and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.last_page.resolve()))):
            self.status.setText('已生成，请打开：'+str(self.last_page))

    def reject(self):
        if self.busy:
            self._close_requested=True;self.status.setText('等待数据任务结束后关闭。');return
        super().reject()

    def closeEvent(self,event):
        if self.busy:
            self._close_requested=True;self.status.setText('等待数据任务结束后关闭。');event.ignore()
        else:super().closeEvent(event)
