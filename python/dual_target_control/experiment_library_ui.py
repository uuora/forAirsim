"""Reusable native Qt data library; filesystem mutations use ExperimentStore."""
import json,threading
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import Qt,QAbstractTableModel,QModelIndex,QSortFilterProxyModel,QObject,pyqtSignal,QUrl
from PyQt5.QtGui import QDesktopServices,QColor
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QLineEdit,QComboBox,QPushButton,QTableView,QHeaderView,QAbstractItemView,QSplitter,QPlainTextEdit,QCheckBox,QFormLayout,QDialog,QDialogButtonBox)
from .experiment_store import ExperimentStore,KINDS,STATUS
from .curve_results import TITLES


def human_size(value):
    for unit in ('B','KB','MB','GB','TB'):
        if value<1024:return f'{value:.1f} {unit}'
        value/=1024

class Signals(QObject):
    done=pyqtSignal(str,object)
    error=pyqtSignal(str)

class RecordModel(QAbstractTableModel):
    headers=['选择','时间 / 目录','类别','实验组 / 协议','方法','结果','大小','标签','同源图页']
    def __init__(self):super().__init__();self.rows=[];self.checked=set()
    def rowCount(self,parent=QModelIndex()):return 0 if parent.isValid() else len(self.rows)
    def columnCount(self,parent=QModelIndex()):return len(self.headers)
    def headerData(self,section,orientation,role):return self.headers[section] if orientation==Qt.Horizontal and role==Qt.DisplayRole else None
    def data(self,index,role=Qt.DisplayRole):
        if not index.isValid():return None
        row=self.rows[index.row()];col=index.column()
        if role==Qt.CheckStateRole and col==0:return Qt.Checked if row['id'] in self.checked else Qt.Unchecked
        if role==Qt.ToolTipRole:return row['path']+'\n'+row.get('error','')
        if role==Qt.ForegroundRole and col==5:return QColor('#14745d' if row['status']=='PASS' else '#b34845' if row['status'] in ('FAILED','CONTROL_FAIL') else '#5b7080')
        if role!=Qt.DisplayRole:return None
        return ['',datetime.fromtimestamp(row['modified']).strftime('%m-%d %H:%M')+'  '+row.get('display_name',row['name']),KINDS.get(row['kind'],row['kind']),{'stop':'逐点停稳','pass':'连续过点','single':'单目标','multi':'分组结果'}.get(row['policy'],row['policy'])+' / '+str(row['protocol']),TITLES.get(row['method'],row['method']),STATUS.get(row['status'],'已归档' if row['status']=='ARCHIVED' else row['status']),human_size(row['size_bytes']),('★ ' if row.get('pinned') else '')+row.get('tags',''),row.get('same_input_views') or '—'][col]
    def flags(self,index):return super().flags(index)|(Qt.ItemIsUserCheckable if index.column()==0 else Qt.NoItemFlags)
    def setData(self,index,value,role):
        if role==Qt.CheckStateRole and index.column()==0:
            key=self.rows[index.row()]['id']
            if value==Qt.Checked:self.checked.add(key)
            else:self.checked.discard(key)
            self.dataChanged.emit(index,index,[role]);return True
        return False
    def replace(self,rows):
        self.beginResetModel();self.rows=rows;self.checked.intersection_update(r['id'] for r in rows);self.endResetModel()

class RecordFilter(QSortFilterProxyModel):
    def __init__(self):super().__init__();self.query='';self.filters={};self.show_history=False
    def filterAcceptsRow(self,row,parent):
        item=self.sourceModel().rows[row]
        if not self.show_history and item.get('collection','current')!='current':return False
        text=json.dumps(item,ensure_ascii=False)+' '+TITLES.get(item.get('method'),item.get('method',''))+' '+' '.join(TITLES.get(m,m or '') for m in item.get('methods',[]))
        return all(not value or str(item.get(key,''))==str(value) for key,value in self.filters.items()) and (not self.query or self.query in text.lower())

class DataLibrary(QWidget):
    def __init__(self,owner,root):
        super().__init__();self.owner=owner;self.store=ExperimentStore(root);self.busy=False;self.focused=None;self.signals=Signals(self)
        self.signals.done.connect(self.finished);self.signals.error.connect(self.failed)
        layout=QVBoxLayout(self);layout.setContentsMargins(0,10,0,0);layout.setSpacing(12)
        title=QLabel('实验记录库');title.setObjectName('sectionTitle');layout.addWidget(title)
        layout.addWidget(QLabel('默认只显示当前控制实验。旧版、开发核验和其他任务已收进历史；原始及失败记录保留。'))
        filters=QHBoxLayout();self.search=QLineEdit();self.search.setPlaceholderText('搜索目录、方法、种子、标签或备注…');self.search.setClearButtonEnabled(True);filters.addWidget(self.search,2)
        self.filters={}
        for key,choices in [('kind',[('', '全部类别'),*[(k,v) for k,v in KINDS.items()]]),('policy',[('','全部实验组'),('stop','逐点停稳'),('pass','连续过点'),('online','连续在线'),('single','单目标')]),('status',[('','全部结果'),*[(k,v) for k,v in STATUS.items()],('ARCHIVED','已归档')]),('protocol',[('','全部协议'),('16','协议16·连续在线'),('15','协议15·准备后规划'),('14','协议14'),('13','协议13')])]:
            box=QComboBox()
            for code,text in choices:box.addItem(text,code)
            filters.addWidget(box);box.currentIndexChanged.connect(self.filter_changed);self.filters[key]=box
        self.history=QCheckBox('显示历史 / 开发资料');filters.addWidget(self.history);layout.addLayout(filters)
        tools=QHBoxLayout();self.refresh_button=QPushButton('刷新索引');self.refresh_button.clicked.connect(self.scan);tools.addWidget(self.refresh_button)
        check=QPushButton('勾选当前筛选');check.clicked.connect(self.check_visible);tools.addWidget(check)
        clear=QPushButton('清空勾选');clear.clicked.connect(self.clear_checked);tools.addWidget(clear)
        self.compare_button=QPushButton('生成所选对比');self.compare_button.setObjectName('primary');self.compare_button.clicked.connect(self.compare);tools.addWidget(self.compare_button)
        self.history_button=QPushButton('收起到历史');self.history_button.clicked.connect(lambda:self.change_collection('history'));tools.addWidget(self.history_button)
        self.current_button=QPushButton('恢复到当前');self.current_button.clicked.connect(lambda:self.change_collection('current'));tools.addWidget(self.current_button)
        self.archive_button=QPushButton('预览归档旧图表');self.archive_button.clicked.connect(self.preview);tools.addWidget(self.archive_button)
        self.restore_button=QPushButton('恢复所选归档');self.restore_button.clicked.connect(self.restore);tools.addWidget(self.restore_button)
        tools.addStretch();layout.addLayout(tools)
        splitter=QSplitter();self.model=RecordModel();self.proxy=RecordFilter();self.proxy.setSourceModel(self.model)
        self.table=QTableView();self.table.setModel(self.proxy);self.table.setAlternatingRowColors(True);self.table.setSelectionBehavior(QAbstractItemView.SelectRows);self.table.setSelectionMode(QAbstractItemView.SingleSelection);self.table.setShowGrid(False);self.table.verticalHeader().hide();self.table.verticalHeader().setDefaultSectionSize(36)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents);self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch);self.table.horizontalHeader().setSectionResizeMode(7,QHeaderView.Stretch)
        self.table.clicked.connect(self.focus);self.table.doubleClicked.connect(lambda _:self.view_record());splitter.addWidget(self.table)
        details=QWidget();form=QVBoxLayout(details);form.setContentsMargins(10,0,0,0)
        self.detail=QPlainTextEdit();self.detail.setReadOnly(True);self.detail.setPlaceholderText('点击一条记录查看路径、判定及错误原因。');form.addWidget(self.detail,1)
        self.tags=QLineEdit();self.tags.setPlaceholderText('例如：开发集,导师演示,待复核');self.note=QPlainTextEdit();self.note.setPlaceholderText('备注保存在索引库，不修改原始实验。');self.note.setMaximumHeight(95)
        self.pin=QCheckBox('固定保留（禁止归档）');form.addWidget(QLabel('标签'));form.addWidget(self.tags);form.addWidget(self.note);form.addWidget(self.pin)
        save=QPushButton('保存标签与备注');save.clicked.connect(self.save_note);form.addWidget(save)
        open_row=QHBoxLayout();open_folder=QPushButton('打开目录');open_folder.clicked.connect(self.open_folder);open_row.addWidget(open_folder);view=QPushButton('查看结果');view.clicked.connect(self.view_record);open_row.addWidget(view);form.addLayout(open_row)
        splitter.addWidget(details);splitter.setSizes([850,310]);layout.addWidget(splitter,1)
        self.message=QLabel();self.message.setWordWrap(True);layout.addWidget(self.message)
        self.search.textChanged.connect(self.filter_changed);self.model.dataChanged.connect(lambda *_:self.update_count())
        self.history.toggled.connect(self.history_toggled)
        self.model.replace(self.store.records());self.filter_changed()
    def history_toggled(self,value):
        if not value:
            for key,code in [('protocol','13'),('status','ARCHIVED')]:
                box=self.filters[key]
                if box.currentData()==code:box.blockSignals(True);box.setCurrentIndex(0);box.blockSignals(False)
        self.filter_changed()
    def filter_changed(self):
        sender=self.sender()
        if (sender==self.filters.get('protocol') and sender.currentData()=='13') or (sender==self.filters.get('status') and sender.currentData()=='ARCHIVED'):self.history.setChecked(True)
        self.proxy.show_history=self.history.isChecked()
        if not self.proxy.show_history:self.model.checked.intersection_update(r['id'] for r in self.model.rows if r.get('collection','current')=='current')
        self.proxy.query=self.search.text().strip().lower();self.proxy.filters={key:box.currentData() for key,box in self.filters.items()};self.proxy.invalidateFilter();self.update_count()
        for button in (self.current_button,self.archive_button,self.restore_button):button.setVisible(self.history.isChecked())
    def update_count(self):
        current=sum(r.get('collection','current')=='current' for r in self.model.rows)
        self.message.setText(f'当前资料 {current} 项 · 历史已收起 {len(self.model.rows)-current} 项 · 正在显示 {self.proxy.rowCount()} 项 · 已勾选 {len(self.model.checked)} 项。')
    def check_visible(self):
        for i in range(self.proxy.rowCount()):self.model.setData(self.proxy.mapToSource(self.proxy.index(i,0)),Qt.Checked,Qt.CheckStateRole)
        self.update_count()
    def clear_checked(self):
        self.model.checked.clear();self.model.layoutChanged.emit();self.update_count()
    def selected(self):return [row for row in self.model.rows if row['id'] in self.model.checked]
    def change_collection(self,value):
        selected=self.selected()
        if not selected and self.focused:selected=[self.focused]
        if not selected:self.message.setText('先勾选要整理的记录。');return
        self.job('整理历史',lambda:self.store.set_visibility([r['id'] for r in selected],value))
    def guard(self):
        if self.busy:return False
        if self.owner.locked():self.message.setText('请在本次飞行/批量结束后执行扫描、归档或生成图表；已有索引仍可浏览。');return False
        return True
    def job(self,kind,work):
        if not self.guard():return
        self.busy=True;self.set_operable(False);self.message.setText('处理中：'+kind+'，原始日志保持原位。')
        def worker():
            try:self.signals.done.emit(kind,work())
            except Exception as exc:
                try:self.signals.error.emit(str(exc))
                except RuntimeError:pass
        threading.Thread(target=worker,daemon=True).start()
    def failed(self,error):self.busy=False;self.set_operable(True);self.message.setText('未完成：'+error)
    def finished(self,kind,value):
        self.busy=False;self.model.replace(self.store.records());self.set_operable(True);self.update_count()
        if kind=='整理历史':self.clear_checked();self.filter_changed()
        if kind=='图表':self.owner.workspace.show_results(Path(value))
        elif kind=='归档':self.message.setText(f'已校验并归档 {len(value)} 个旧图表目录。原始日志保留，可在“已归档”结果中恢复。')
        elif kind=='恢复':self.message.setText('已按校验值恢复至：'+str(value))
    def set_operable(self,enabled):
        enabled=enabled and not self.busy
        for button in (self.refresh_button,self.compare_button,self.history_button,self.current_button,self.archive_button,self.restore_button):button.setEnabled(enabled)
    def scan(self):self.job('扫描分类',self.store.scan)
    def focus(self,index):
        row=self.model.rows[self.proxy.mapToSource(index).row()];self.focused=row
        self.tags.setText(row.get('tags',''));self.note.setPlainText(row.get('note',''));self.pin.setChecked(row.get('pinned',False))
        lines=[row['id'],'','类别：'+KINDS.get(row['kind'],row['kind']),'结果：'+STATUS.get(row['status'],row['status']),f"协议：{row['protocol']}   组：{row['policy']}",'方法：'+TITLES.get(row['method'],row['method']),f"种子：{row['seed']}   文件：{row['file_count']}   大小：{human_size(row['size_bytes'])}",'同原始输入的图表数：'+str(row.get('same_input_views',0)),'条件指纹：'+row.get('condition',''),'','错误/证据缺口：'+(row.get('error') or '索引未发现；不等于完成独立审计'),'','完整目录：'+row['path']]
        self.detail.setPlainText('\n'.join(lines))
    def save_note(self):
        if not self.focused:return
        if self.busy:self.message.setText('数据任务完成后再保存标签或固定保留标记。');return
        try:self.store.annotate(self.focused['id'],tags=self.tags.text(),note=self.note.toPlainText(),pinned=self.pin.isChecked());self.store.export();self.model.replace(self.store.records());self.message.setText('标签、备注及固定保留标记已保存。')
        except Exception as exc:self.failed(str(exc))
    def open_folder(self):
        if self.focused:
            path=Path(self.focused['path']);QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))
    def view_record(self):
        if not self.focused:return
        row=self.focused;folder=Path(row['path'])
        if row.get('archive_id'):self.message.setText('此项已归档；选择“恢复所选归档”后查看。');return
        if (folder/'online_result.html').is_file():QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder/'online_result.html')));return
        if row['kind']=='flight':self.plot([folder]);return
        if (folder/'figure_index.json').is_file() or any(folder.glob('*/figure_index.json')):self.owner.workspace.show_results(folder/'index.html');return
        if (folder/'index.html').is_file():QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder/'index.html')));return
        if (folder/'campaign.json').is_file():
            from .curve_results import render_campaign
            self.job('图表',lambda:render_campaign(folder/'campaign.json',self.new_output()));return
        self.open_folder()
    def new_output(self):return self.store.root/'reports'/('control_view_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    def compare(self):
        selected=self.selected()
        if len(selected)<2:self.message.setText('请勾选至少两条曲线飞行记录。');return
        if any(r['kind']!='flight' for r in selected):self.message.setText('对比只接受“曲线飞行”记录；其他类别可单独查看或归档。');return
        self.plot([Path(r['path']) for r in selected])
    def plot(self,folders):
        from .curve_results import render_selection
        self.job('图表',lambda:render_selection(folders,self.new_output()))
    def protected(self):
        state=self.owner.runtime.snapshot();paths=[state.get('run_dir')]
        if self.owner.last_results:paths.append(str(self.owner.last_results.parent))
        return paths
    def preview(self):
        if not self.guard():return
        try:preview=self.store.preview_archive([r['id'] for r in self.selected()],self.protected())
        except Exception as exc:self.message.setText(str(exc));return
        if not preview['entries']:self.message.setText('先筛选“派生图表”，勾选要归档的旧目录。');return
        dialog=QDialog(self);dialog.setWindowTitle('清理预览 · 可恢复归档');dialog.resize(720,450);layout=QVBoxLayout(dialog)
        layout.addWidget(QLabel('只归档下列可再生成图表。先压缩并逐文件SHA-256校验，再清理原图表目录。\n原始飞行、失败日志、冻结方案和未知文件不清理；压缩包可恢复，不保证明显释放空间。'))
        text=QPlainTextEdit();text.setReadOnly(True);text.setPlainText('\n'.join(f"{e['original']}\n  {len(e['files'])}个文件 / {human_size(sum(f['size'] for f in e['files']))}" for e in preview['entries']));layout.addWidget(text)
        buttons=QDialogButtonBox(QDialogButtonBox.Cancel);run=buttons.addButton('执行可恢复归档',QDialogButtonBox.AcceptRole);buttons.rejected.connect(dialog.reject);run.clicked.connect(dialog.accept);layout.addWidget(buttons)
        if dialog.exec_()==QDialog.Accepted:self.job('归档',lambda:self.store.archive(preview,self.protected()))
    def restore(self):
        selected=self.selected()
        if not selected and self.focused:selected=[self.focused]
        if len(selected)!=1 or not selected[0].get('archive_id'):self.message.setText('请选择一条“已归档”记录进行恢复。');return
        self.job('恢复',lambda:self.store.restore(selected[0]['archive_id']))
