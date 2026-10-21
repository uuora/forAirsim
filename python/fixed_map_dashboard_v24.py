"""v24 task window: observe the persistent driver; never own flight RPCs."""
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSharedMemory, QRectF
from PyQt5.QtGui import QColor, QPainter, QPixmap, QFont, QPen
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QPlainTextEdit, QGroupBox,
                            QSplitter, QMessageBox, QSpinBox)

from fixed_map_dashboard_session_v24 import V24Session

DEFAULT_INSTRUCTION = '让单机识别并接近带白色横条的红色气球，然后返航落地'
STAGES = {'CHECKING': '起飞前检查', 'READY': '地面就绪', 'TAKEOFF': '起飞',
          'APPROACH': '反馈接近', 'ARRIVAL_HOVER': '持续悬停验证',
          'ARRIVED': '悬停已验证', 'RETURN': '返航', 'LANDING': '落地',
          'STOP_HOVER': '停止悬停', 'ABORTED': '中止悬停',
          'FAILED': '检查失败', 'FAILED_HOVER': '失败后悬停'}
PHASES = {'OBSERVATION': '飞到观察点', 'CONFIRM': '确认当前目标',
          'APPROACH': '视觉接近与悬停', 'RETURN': '返航',
          'LANDED_RELEASED': '已落地并释放控制', 'STORAGE_SAFE_RETURN': '安全返航'}
ERRORS = {'TEXT_MODEL_MEMORY_HEADROOM_BELOW_2GIB': '文字模型启动被拒绝：可用物理内存或系统提交内存不足 2 GiB。',
          'MODEL_LOAD_MEMORY_HEADROOM_BELOW_3GIB': '视觉模型启动被拒绝：可用物理内存或系统提交内存不足 3 GiB。',
          'SCENE_ENGINE_LIFETIME_CHANGED': 'UE 已更换进程，需要先同步地面场景。',
          'OPERATOR_STOP_BEFORE_TAKEOFF': '已取消，尚未请求起飞。',
          'OPERATOR_STOP_BEFORE_MODEL_LOAD_NO_TAKEOFF': '已取消，尚未加载模型或请求起飞。'}


class Operation(QThread):
    completed = pyqtSignal(str, object, str)

    def __init__(self, name, operation):
        super().__init__()
        self.name, self.operation = name, operation

    def run(self):
        try:
            self.completed.emit(self.name, self.operation(), '')
        except Exception as ex:
            self.completed.emit(self.name, None, str(ex))


class ImageCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap()
        self.candidates = []
        self.selected_id = None
        self.message = '等待任务采集的实际机载图像'
        self.setMinimumSize(400, 250)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor('#101820'))
        if self.pixmap.isNull():
            painter.setPen(QColor('#d0d6dc'))
            painter.drawText(self.rect(), Qt.AlignCenter, self.message)
        else:
            scaled = self.pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ox, oy = (self.width()-scaled.width())//2, (self.height()-scaled.height())//2
            painter.drawPixmap(ox, oy, scaled)
            sx, sy = scaled.width()/self.pixmap.width(), scaled.height()/self.pixmap.height()
            painter.setFont(QFont('Microsoft YaHei', 11, QFont.Bold))
            for index, candidate in enumerate(self.candidates):
                box = candidate.get('bbox_xyxy_pixels')
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = box
                selected = candidate.get('candidate_id') == self.selected_id
                color = QColor('#39ef9a' if selected else '#ffc44d')
                rect = QRectF(ox+x1*sx, oy+y1*sy, (x2-x1)*sx, (y2-y1)*sy)
                painter.setPen(QPen(color, 3))
                painter.drawRect(rect)
                text = ('已关联目标' if selected else f'候选 {index+1}') + f" · 分数 {candidate.get('detector_score', 0):.2f}"
                width = painter.fontMetrics().horizontalAdvance(text)+16
                label = QRectF(max(ox, min(rect.left(), ox+scaled.width()-width)), max(oy, rect.top()-29), width, 27)
                painter.fillRect(label, QColor('#101820'))
                painter.setPen(color)
                painter.drawText(label.adjusted(8, 0, -4, 0), Qt.AlignVCenter, text)


class Dashboard(QWidget):
    def __init__(self, session=None, bootstrap_on_open=True):
        super().__init__()
        self.session = session or V24Session()
        self.worker = None
        self.busy = None
        self.closing = False
        self.stop_pending = False
        self.cancel_start = threading.Event()
        self.last_image = None
        self.ui_error = ''
        self.setWindowTitle('固定地图实验 · v24 视觉控制闭环')
        self.resize(1280, 860)
        layout = QVBoxLayout(self)
        title = QLabel('DroneA · v24 视觉控制闭环')
        title.setStyleSheet('font-size:23px;font-weight:600;padding:6px')
        layout.addWidget(title)
        description = QLabel('自然语言任务 → RGB / 深度确认 → 反馈接近 → 持续悬停 → 返航落地\n'
                             '固定高度 3.1 m · 指令速度上限 0.65 m/s · DroneB 地面待命')
        description.setWordWrap(True)
        layout.addWidget(description)
        self.engine_status = QLabel('正在连接后台 UE…')
        self.status = QLabel('尚未启动飞行任务')
        self.status.setStyleSheet('font-size:17px;padding:8px;background:#edf2f6')
        layout.addWidget(self.engine_status)
        layout.addWidget(self.status)
        task_box = QGroupBox('输入任务并启动一次完整闭环')
        task_layout = QVBoxLayout(task_box)
        self.instruction = QPlainTextEdit(DEFAULT_INSTRUCTION)
        self.instruction.setMaximumHeight(76)
        task_layout.addWidget(self.instruction)
        row = QHBoxLayout()
        self.start_button = QPushButton('启动 v24 闭环')
        self.stop_button = QPushButton('停止并返航落地')
        self.stop_button.setStyleSheet('background:#a33939;color:white;font-weight:600')
        self.connect_button = QPushButton('重新连接地面场景')
        self.sync_button = QPushButton('同步地面气球场景')
        self.random_button = QPushButton('随机气球布局（地面）')
        for button in (self.start_button, self.stop_button, self.connect_button, self.sync_button):
            button.setMinimumHeight(40)
            row.addWidget(button)
        task_layout.addLayout(row)
        random_row=QHBoxLayout()
        random_row.addWidget(QLabel('布局种子'))
        self.random_seed=QSpinBox()
        self.random_seed.setRange(0,2147483647)
        self.random_seed.setValue(24101008)
        random_row.addWidget(self.random_seed)
        random_row.addWidget(self.random_button)
        random_row.addStretch()
        task_layout.addLayout(random_row)
        hint = QLabel('支持黄色气球、普通红色气球、白条红气球；多个匹配目标会拒识。随机按钮只在地面改变气球位置。\n'
                      '可输入“预览黄色气球的接近路径”做地面感知预览；观察准备仍沿已知地图路线。\n'
                      '本按钮执行完整任务：完成后自动返航；观察点30秒未确认目标也会自动返航。后台 UE 持续渲染。')
        hint.setWordWrap(True)
        task_layout.addWidget(hint)
        layout.addWidget(task_box)
        panels = QSplitter(Qt.Horizontal)
        image_box = QGroupBox('实际机载图像 · 来自 v24 同一采集进程')
        image_layout = QVBoxLayout(image_box)
        self.image = ImageCanvas()
        self.image_info = QLabel('尚无在线图像')
        self.image_info.setWordWrap(True)
        image_layout.addWidget(self.image, 1)
        self.detection_info = QLabel('YOLO：尚未开始采集')
        self.detection_info.setWordWrap(True)
        self.detection_info.setStyleSheet('font-size:16px;font-weight:600;padding:8px;background:#fff0cf;color:#533700')
        image_layout.addWidget(self.detection_info)
        image_layout.addWidget(self.image_info)
        panels.addWidget(image_box)
        details_box = QGroupBox('实际状态与任务结果')
        details_layout = QVBoxLayout(details_box)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMinimumWidth(410)
        details_layout.addWidget(self.details)
        panels.addWidget(details_box)
        panels.setSizes([750, 450])
        layout.addWidget(panels, 1)
        self.output_label = QLabel('记录目录：' + str(self.session.output))
        self.output_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)
        footer = QLabel('历史v24固定布局闭环3/3通过；本次新增随机布局与颜色目标的完整飞行验证待完成。\n'
                        '本窗口使用独立控制进程。关闭时请求返航，等待任务结束；UE 保留在后台。实验记录和临时文件写 D 盘。')
        footer.setWordWrap(True)
        layout.addWidget(footer)
        self.start_button.clicked.connect(self.start_trial)
        self.stop_button.clicked.connect(self.stop_trial)
        self.connect_button.clicked.connect(lambda: self.begin('bootstrap', self.session.bootstrap))
        self.sync_button.clicked.connect(lambda: self.begin('sync', self.session.refresh_scene))
        self.random_button.clicked.connect(self.randomize_scene)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)
        self.refresh()
        if bootstrap_on_open:
            QTimer.singleShot(0, lambda: self.begin('bootstrap', self.session.bootstrap))

    def begin(self, name, operation):
        if self.busy or (self.worker is not None and self.worker.isRunning()):
            return
        self.busy = name
        self.ui_error = ''
        self.worker = Operation(name, operation)
        self.worker.completed.connect(self.operation_completed)
        self.worker.finished.connect(self.refresh)
        self.worker.start()
        self.refresh()

    def randomize_scene(self):
        seed=self.random_seed.value()
        self.begin('random', lambda: self.session.refresh_scene(randomize=True,seed=seed))

    def operation_completed(self, name, result, error):
        self.busy = None
        if error:
            self.ui_error = error
        if self.stop_pending:
            self.stop_pending = False
            self.stop_trial()
        self.refresh()
        QTimer.singleShot(350, self.save_snapshot)
        if self.closing:
            QTimer.singleShot(500, self.close)

    def save_snapshot(self):
        if self.isVisible():
            self.grab().save(str(self.session.output / 'dashboard_live.png'))

    def start_trial(self):
        text = self.instruction.toPlainText().strip()
        if not text:
            self.ui_error = '请输入任务指令。'
            self.refresh()
            return
        self.last_image = None
        self.image.pixmap = QPixmap()
        self.image.candidates = []
        self.detection_info.setText('YOLO：等待模型预热和首帧检测')
        self.image.update()
        self.cancel_start.clear()
        self.begin('start', lambda: self.session.start_trial(text, cancel_event=self.cancel_start))

    def stop_trial(self):
        if self.busy == 'start':
            self.stop_pending = True
            self.cancel_start.set()
        try:
            self.session.request_stop()
        except Exception as ex:
            self.ui_error = str(ex)
        self.refresh()

    def refresh(self):
        snapshot = self.session.poll()
        running = bool(snapshot.get('running'))
        bootstrap = snapshot.get('bootstrap') or {}
        ready = bootstrap.get('status', bootstrap.get('phase')) == 'READY'
        engine = snapshot.get('engine_info') or bootstrap.get('engine') or {}
        self.engine_status.setText(('后台 UE 已连接 · PID ' + str(engine.get('pid', '—')))
                                   if engine else '后台 UE 尚未就绪')
        final = snapshot.get('final') or {}
        error = self.ui_error or snapshot.get('error') or bootstrap.get('error') or final.get('issue') or ''
        mission = snapshot.get('mission') or {}
        stage = snapshot.get('stage') or 'CHECKING'
        if self.busy:
            self.status.setText({'bootstrap': '正在核对后台 UE 与新鲜地面状态…',
                                 'sync': '正在地面同步气球场景…',
                                 'random': '正在地面随机摆放气球并核验位置…',
                                 'start': '正在冻结本次输入和核对地面状态…'}.get(self.busy, '处理中…'))
        elif running:
            phase = PHASES.get(mission.get('phase'), mission.get('phase') or '解析与模型准备')
            self.status.setText('任务进程 PID ' + str(snapshot.get('pid')) + ' · ' + phase + ' / ' + STAGES.get(stage, stage))
        elif error:
            self.status.setText(ERRORS.get(error, '任务未通过：' + error))
        elif final:
            outcome = (final.get('mission') or {}).get('outcome')
            self.status.setText('闭环完成 · 已落地释放控制' if final.get('perception_guidance_hover_passed')
                                else ('地面预览已结束 · 未起飞' if outcome == 'GROUND_PREVIEW_COMPLETED_NO_TAKEOFF' else
                                      '目标确认超时 · 已自动返航落地' if outcome == 'FAILED_CONFIRM_DEADLINE'
                                      else '本次任务已结束 · 请查看结果'))
        elif ready:
            self.status.setText('地面就绪 · 输入任务后点击“启动 v24 闭环”')
        else:
            self.status.setText('需要同步地面气球场景' if bootstrap.get('status') == 'READY_SCENE_SYNC_REQUIRED'
                                else '等待地面检查通过')
        blocked = running or bool(self.busy) or (self.worker is not None and self.worker.isRunning())
        self.start_button.setEnabled(ready and not blocked)
        self.stop_button.setEnabled(running or self.busy == 'start')
        self.connect_button.setEnabled(not blocked)
        self.sync_button.setEnabled(not blocked and bool(bootstrap.get('ground', {}).get('passed')))
        self.random_button.setEnabled(not blocked and bool(bootstrap.get('ground', {}).get('passed')))
        self.random_seed.setEnabled(not blocked)
        self.instruction.setReadOnly(blocked)
        state = snapshot.get('state') or {}
        latest = snapshot.get('latest') or {}
        lines = ['任务阶段：' + str(mission.get('phase', 'IDLE')),
                 '控制阶段：' + STAGES.get(stage, stage)]
        if state:
            lines += ['实测位置 XYZ/m：' + str(state.get('position_xyz_m')),
                      '实测速度/(m/s)：' + str(state.get('speed_m_s')),
                      '目标误差/m：' + str(state.get('goal_error_m')),
                      '计算包络净空/m：' + str(state.get('envelope_clearance_calculated_m')),
                      'DroneB 地面：' + str(state.get('DroneB_landed'))]
        if latest:
            estimate = latest.get('position_estimate') or {}
            candidates = (latest.get('semantic_frame') or {}).get('candidates') or []
            tracking = latest.get('temporal_identity') or {}
            description=((tracking.get('attribute_audit') or {}).get('description') or
                         ((tracking.get('attribute_audit') or {}).get('attribute_details') or {}).get('description') or '')
            outcome = '定位可用' if latest.get('valid') else ('尚未确认指定目标' if candidates else '未检出气球')
            self.detection_info.setText(f'目标：{description or "见任务描述"} · YOLO：{len(candidates)} 个候选 · {outcome}\n黄色框：检测候选；绿色框：已关联目标。检测分数不代表身份正确率。')
            lines += ['源帧：' + str(latest.get('frame_id')),
                      'YOLO候选数量：' + str(len(candidates)),
                      '原记录有效：' + str(latest.get('valid')),
                      '可见表面 XYZ/m：' + str(estimate.get('surface_world_xyz_m')),
                      '拒识：' + ', '.join(latest.get('rejection_reasons') or [])]
            self.show_image(latest, running and snapshot.get('publication_current_read', True))
        if final:
            outcome = (final.get('mission') or {}).get('outcome')
            reasons = {'FAILED_CONFIRM_DEADLINE': '观察点等待30秒仍未确认目标，自动返航收尾',
                       'APPROACH_HOVER_PASSED': '接近与悬停完成，按完整任务流程自动返航',
                       'FAILED_OBSERVATION_DEADLINE': '观察准备超时，自动返航收尾',
                       'FAILED_APPROACH_DEADLINE': '视觉接近超时，自动返航收尾',
                       'GROUND_PREVIEW_COMPLETED_NO_TAKEOFF': '地面预览结束，未请求起飞或返航'}
            if outcome:
                lines += ['', '任务结果：' + str(outcome), '返航原因：' + reasons.get(outcome, str(outcome))]
            lines += ['', '本次原始结果：', '请求过起飞：' + str(final.get('flight_start_requested')),
                      '感知接近悬停通过：' + str(final.get('perception_guidance_hover_passed')),
                      '最终地面核验通过：' + str(final.get('ground_verification_passed')),
                      '感知清理：' + str((final.get('sensor_cleanup') or {}).get('status', '未启动')),
                      '数据保留清理：' + str((final.get('raw_retention') or {}).get('status', '尚无回执'))]
        if error:
            lines += ['', ERRORS.get(error, error)]
        self.details.setPlainText('\n'.join(lines))
        self.output_label.setText('记录目录：' + str(snapshot.get('trial_output') or self.session.output))
        if not running and self.last_image:
            self.image_info.setText('本次任务最后保存帧 · 当前未采集，不能作为实时控制证据')
        if self.closing and not blocked:
            QTimer.singleShot(500, self.close)

    def show_image(self, latest, running):
        capture = latest.get('capture') or {}
        # Draw in widget pixels so labels stay readable after image scaling.
        path = capture.get('rgb_path') or latest.get('overlay_path')
        if not path:
            return
        path = Path(path).resolve()
        if not path.is_relative_to(self.session.output.resolve()):
            return
        stamp = (latest.get('frame_id'), str(path))
        if stamp != self.last_image and path.is_file():
            image = QPixmap(str(path))
            if not image.isNull():
                self.image.pixmap = image
                self.image.candidates = (latest.get('semantic_frame') or {}).get('candidates') or []
                self.image.selected_id = (latest.get('temporal_identity') or {}).get('selected_candidate_id')
                self.image.update()
                self.last_image = stamp
        received = capture.get('received_monotonic_s')
        age = time.monotonic() - received if isinstance(received, (int, float)) else None
        fresh = running and age is not None and 0 <= age <= .5
        self.image_info.setText(str(latest.get('frame_id')) + (' · 新鲜源帧' if fresh else ' · 已过期 / 最后保存帧')
                                + (f' · 帧龄 {age:.2f} s' if age is not None else ''))

    def closeEvent(self, event):
        snapshot = self.session.poll()
        if snapshot.get('running') or self.busy:
            self.closing = True
            if self.busy == 'start':
                self.stop_pending = True
                self.cancel_start.set()
                self.stop_trial()
            elif snapshot.get('running'):
                self.stop_trial()
            self.status.setText('已请求安全返航；等待落地、释放控制和任务进程结束后关闭。')
            event.ignore()
            return
        if self.worker and self.worker.isRunning():
            event.ignore()
            QTimer.singleShot(200, self.close)
            return
        self.timer.stop()
        self.session.close()
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/fixed_map_perception_yolo_gpu640_v24.json'))
    args = parser.parse_args()
    # No NumPy/mission imports before the same frozen v24 math budget.
    from fixed_map_cpu_runtime import configure_math_runtime
    configure_math_runtime(json.loads(Path(args.config).read_text(encoding='utf-8-sig')))
    session = V24Session()
    if Path(args.config).resolve() != session.config_path.resolve():
        raise ValueError('V24_WINDOW_REQUIRES_FROZEN_CONFIG')
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 10))
    lock = QSharedMemory('FixedMapMissionDroneA20261009')
    if not lock.create(1):
        QMessageBox.information(None, '界面已运行', '请用 Alt+Tab 切换到已有固定地图实验窗口。')
        return 0
    window = Dashboard(session)
    window.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
