"""Independent geometric-validation panel; rendering never calls AirSim.

The existing map uses [X, Y, height] while this protocol records NED.  This
adapter is the only coordinate conversion for the display. The body and rotor
planes follow the measured quaternion; the icon remains an enlarged schematic.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices, QFont, QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QPlainTextEdit, QPushButton, QStyle, QVBoxLayout, QWidget,
)

from dual_target_control.route_view import RouteFlightView


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "geometric_planners_v1.json"
CASES = (("hover", "1 · 悬停"), ("line", "2 · 直线"),
         ("turn", "3 · 平滑转弯"), ("waypoints", "4 · 三维多航点"))


def height_vector(ned: Any) -> list[float]:
    """Convert for painting only; never change the runtime/state log."""
    values = np.asarray(ned, dtype=float).reshape(3)
    if not np.isfinite(values).all():
        raise ValueError("Display vector must contain three finite NED values")
    return [float(values[0]), float(values[1]), -float(values[2])]


def measured_yaw(quaternion_wxyz: Any) -> float:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    length = float(np.linalg.norm([w, x, y, z]))
    if not math.isfinite(length) or length < 1e-12:
        raise ValueError("Invalid measured quaternion")
    w, x, y, z = w / length, x / length, y / length, z / length
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class LatestTelemetry:
    """One-slot mailbox. A contended display drops a frame, never waits.

    The producer only stores a shallow dictionary of immutable/vector payloads;
    conversion, trails and painting happen on the Qt timer.  Full evidence is
    persisted by the runtime independently of this lossy observation channel.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = None
        self.metadata = {}
        self.dropped = 0

    def publish(self, item: dict) -> None:
        if not self.lock.acquire(blocking=False):
            self.dropped += 1
            return
        try:
            # Runtime owns no later mutation of these per-frame values.
            self.latest = dict(item)
            for key in ("planned_path_ned", "task_points_ned"):
                if key in item:
                    self.metadata[key] = item[key]
                elif key in self.metadata:
                    self.latest[key] = self.metadata[key]
        finally:
            self.lock.release()

    def take(self):
        with self.lock:
            result, self.latest = self.latest, None
        return result


class MapAdapter:
    """Pure display state. Decimated samples are never used for evaluation."""
    def __init__(self, config: dict, case: str):
        self.trail = deque(maxlen=1200)
        self.reference = deque(maxlen=1200)
        self.set_task(config, case)

    def set_task(self, config: dict, case: str):
        from .reference import build_reference
        reference = build_reference(config, case)
        self.trail.clear()
        self.reference.clear()
        self.start = height_vector(reference.start)
        self.points = [height_vector(point) for point in reference.points[1:]]
        if not self.points:
            self.points = [list(self.start)]
        self.markers = [dict(id="S", kind="start", position=self.start, state="fixed")]
        self.markers += [dict(id=f"W{i+1}", kind="via", position=point, state="pending")
                         for i, point in enumerate(self.points[:-1])]
        self.markers.append(dict(id="E", kind="goal", position=self.points[-1], state="pending"))
        self.contact = set()
        self.view_config = {"preparation": {"task": {"start": self.start, "points": self.points}}}
        # Pure preview; the runtime separately decodes the frozen execution plan.
        self.path = [height_vector(point) for point in reference.sample_path(160)]
        self._path_input = None

    def update_metadata(self, frame: dict):
        path = frame.get("planned_path_ned")
        if path is not None and path is not self._path_input:
            self.path = [height_vector(point) for point in path]
            self._path_input = path

    def idle_snapshot(self):
        return dict(config=self.view_config, connected=False, trail=[], planned_path=self.path,
                    route_markers=self.markers)

    def adapt(self, frame: dict):
        position = height_vector(frame["position"])
        velocity = height_vector(frame.get("velocity", [0, 0, 0]))
        yaw = measured_yaw(frame["quaternion_wxyz"])
        from .core import quaternion_to_matrix, matrix_to_euler
        rotation = quaternion_to_matrix(frame["quaternion_wxyz"])
        flip = np.diag([1., 1., -1.])
        self.trail.append(position)
        reference = frame.get("reference_position")
        if reference is not None:
            self.reference.append(height_vector(reference))
        self.update_metadata(frame)
        # Proximity is a display observation, not the protocol's waypoint test.
        for i, marker in enumerate(self.markers[1:], 1):
            distance = float(np.linalg.norm(np.asarray(position) - marker["position"]))
            if distance <= .35:
                self.contact.add(i)
            elif i in self.contact and marker["kind"] == "via":
                marker["state"] = "visited"
        pending = next((m["position"] for m in self.markers[1:]
                        if m["state"] != "visited"), self.points[-1])
        return dict(config=self.view_config, connected=True, position=position, velocity=velocity,
                    yaw=yaw, trail=list(self.trail), planned_path=self.path or list(self.reference),
                    route_markers=self.markers, targets=[pending],
                    body_rotation_height=(flip @ rotation @ flip).tolist(),
                    attitude_rpy_rad=list(matrix_to_euler(rotation)))


class GeometricControlWindow(QMainWindow):
    """Starts only the new runtime, and only on an explicit start click."""
    def __init__(self, config_path: Path = DEFAULT_CONFIG, runner=None, viewport_factory=None):
        super().__init__()
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        from .environments import load_catalog, load_environment_config
        self.environments=load_catalog()['environments']
        selected=next((e for e in self.environments if (ROOT/e['config']).resolve()==self.config_path),None)
        if selected is None:
            selected=dict(id='custom',label='当前独立配置',config=str(self.config_path))
            self.environments=[selected]
        else:
            self.config=load_environment_config(selected['id'])
        self._environment_id=selected['id']
        self._runner = runner
        self._viewport_factory = viewport_factory
        self.viewport = None
        self._scene_visible = False
        self._thread = None
        self._stop_event = threading.Event()
        self._result = None
        self._error = None
        self._closing = False
        self.run_dir = None
        self.mailbox = LatestTelemetry()
        self.adapter = MapAdapter(self.config, next(iter(self.config['cases'])))
        self.setWindowTitle("三种轨迹规划 · 几何控制 · AirSim")
        self.resize(1240, 880)
        self.setMinimumSize(800, 650)
        self.setStyleSheet("QMainWindow{background:#edf2f7;} QLabel{color:#24384b;} "
                           "QPushButton{padding:8px 14px;} QComboBox{padding:6px;} "
                           "QFrame#card{background:white;border:1px solid #cdd8e1;border-radius:7px;}")
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel("轨迹规划 · 统一几何控制")
        title.setStyleSheet("font-size:20px;font-weight:600;")
        layout.addWidget(title)
        subtitle = QLabel("三维连续过点" if 'comparison' in self.config else "控制基线验证")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        environment_row=QHBoxLayout()
        environment_row.addWidget(QLabel('实验环境'))
        self.environment_combo=QComboBox()
        self.environment_combo.setMinimumWidth(220)
        for entry in self.environments:
            self.environment_combo.addItem(entry['label'],entry['id'])
            self.environment_combo.setItemData(self.environment_combo.count()-1,entry.get('focus',''),Qt.ToolTipRole)
        self.environment_combo.setCurrentIndex(self.environment_combo.findData(self._environment_id))
        self.environment_combo.setEnabled(len(self.environments)>1)
        self.environment_combo.currentIndexChanged.connect(self._change_environment)
        environment_row.addWidget(self.environment_combo)
        self.environment_info=QLabel()
        self.environment_info.setWordWrap(True)
        environment_row.addWidget(self.environment_info,1)
        layout.addLayout(environment_row)
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("轨迹方案" if 'comparison' in self.config else "验证级别"))
        self.case_combo = QComboBox()
        for key, spec in self.config['cases'].items():
            self.case_combo.addItem(spec.get('label', dict(CASES).get(key, key)), key)
        self.case_combo.currentIndexChanged.connect(self._change_case)
        task_row.addWidget(self.case_combo)
        self.task_summary = QLabel()
        self.task_summary.setWordWrap(True)
        task_row.addWidget(self.task_summary, 1)
        self.threshold_label=QLabel()
        self.threshold_label.setToolTip('位置是相对同步参考的三维欧氏误差；姿态是相对R_reference的SO(3)总旋转角误差。')
        task_row.addWidget(self.threshold_label)
        layout.addLayout(task_row)
        self.launch_checkbox = QCheckBox("启动独立AirSim（端口未运行时）")
        self.launch_checkbox.setChecked(False)
        self.launch_checkbox.setToolTip("仅在明确点击“开始所选验证”后生效；窗口打开与勾选不启动进程。")
        scene_row = QHBoxLayout()
        scene_row.addWidget(self.launch_checkbox)
        scene_row.addStretch()
        self.scene_button = QPushButton("显示 UE 视口")
        self.scene_button.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.scene_button.setMinimumWidth(145)
        self.scene_button.setToolTip("显示或隐藏当前 AirSim 原生视口；不触发起飞。")
        self.scene_button.clicked.connect(self.toggle_scene)
        scene_row.addWidget(self.scene_button)
        layout.addLayout(scene_row)
        self.scene_status = QLabel()
        self.scene_status.setWordWrap(True)
        layout.addWidget(self.scene_status)
        frame = QFrame()
        frame.setObjectName("card")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(1, 1, 1, 1)
        self.view = RouteFlightView(self.adapter.view_config)
        frame_layout.addWidget(self.view, 1)
        tools = QHBoxLayout()
        for text, plane in (("三维", "3d"), ("XY 等比例", "xy"), ("X / 高度等比例", "xz")):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, mode=plane: self.view.set_plane(mode))
            tools.addWidget(button)
        tools.addStretch()
        reset = QPushButton("重置视角")
        reset.clicked.connect(self.view.reset_view)
        tools.addWidget(reset)
        frame_layout.addLayout(tools)
        layout.addWidget(frame, 1)
        self.attitude_label = QLabel("实测姿态   横滚 —°   俯仰 —°   偏航 —°")
        layout.addWidget(self.attitude_label)
        self.status = QLabel("空闲 · 尚未连接 AirSim")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        self.error_details = QPlainTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setMaximumHeight(120)
        self.error_details.hide()
        layout.addWidget(self.error_details)
        actions = QHBoxLayout()
        self.start_button = QPushButton("开始所选验证")
        self.start_button.clicked.connect(self.start)
        actions.addWidget(self.start_button)
        self.stop_button = QPushButton("停止并安全降落")
        self.stop_button.clicked.connect(self.stop)
        self.stop_button.setEnabled(False)
        actions.addWidget(self.stop_button)
        actions.addStretch()
        self.open_button = QPushButton("打开本次记录")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self.open_run)
        actions.addWidget(self.open_button)
        self.compare_button = QPushButton("三种实测对比")
        self.compare_button.clicked.connect(self.open_comparison)
        actions.addWidget(self.compare_button)
        layout.addLayout(actions)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self._change_case()

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def _change_environment(self,*_):
        if self.active or self._closing:
            self.environment_combo.blockSignals(True)
            self.environment_combo.setCurrentIndex(self.environment_combo.findData(self._environment_id))
            self.environment_combo.blockSignals(False)
            return
        environment_id=self.environment_combo.currentData()
        entry=next(e for e in self.environments if e['id']==environment_id)
        try:
            path=(ROOT/entry['config']).resolve()
            from .environments import load_environment_config
            cfg=load_environment_config(environment_id) if environment_id!='custom' else json.loads(path.read_text(encoding='utf-8-sig'))
            old_case=self.case_combo.currentData()
            case=old_case if old_case in cfg['cases'] else next(iter(cfg['cases']))
            from .reference import build_reference
            build_reference(cfg,case)
        except (OSError,ValueError,KeyError) as exc:
            self.environment_combo.blockSignals(True)
            self.environment_combo.setCurrentIndex(self.environment_combo.findData(self._environment_id))
            self.environment_combo.blockSignals(False)
            self.error_details.setPlainText(str(exc));self.error_details.show()
            self.status.setText('环境配置不可用。')
            return
        self.config_path,self.config=path,cfg
        self._environment_id=environment_id
        self.case_combo.blockSignals(True)
        self.case_combo.clear()
        for key,spec in cfg['cases'].items():
            self.case_combo.addItem(spec.get('label',dict(CASES).get(key,key)),key)
        self.case_combo.setCurrentIndex(self.case_combo.findData(case))
        self.case_combo.blockSignals(False)
        self.mailbox=LatestTelemetry();self.run_dir=None
        self.open_button.setEnabled(False)
        self.error_details.clear();self.error_details.hide()
        self._change_case()
        self.view.reset_view()
        self.status.setText('空闲 · '+entry['label'])

    def _change_case(self, *_):
        if self.active:
            return
        case = self.case_combo.currentData()
        self.adapter.set_task(self.config, case)
        self.view.config = self.adapter.view_config
        self.view.set_snapshot(self.adapter.idle_snapshot())
        spec = self.config["cases"][case]
        durations = spec.get("segment_durations", [])
        self.task_summary.setText(f"{len(self.adapter.points)} 个任务点 · 参考时长 {sum(durations):.3f} s")
        limits=self.config.get('limits')
        self.threshold_label.setText((f"误差：位置≤{limits['position_error_m']:g}m · "
            f"姿态≤{math.degrees(limits['attitude_error_rad']):g}°" if limits else ''))
        from .environments import task_summary
        summary=task_summary(self.config)
        self.environment_info.setText((f"航点折线 {summary['polyline_length_m']:.1f} m · "
            f"航点高度 {summary['altitude_min_m']:g}–{summary['altitude_max_m']:g} m · "
            f"中途点 {summary['intermediate_points']}" if summary else ''))
        self.attitude_label.setText("实测姿态   横滚 —°   俯仰 —°   偏航 —°")
        if self.viewport is not None:
            self.viewport.update_task(self.adapter.view_config)
            self.viewport.update_snapshot(self.view.snapshot)

    def toggle_scene(self):
        if self._closing:
            return
        try:
            if self.viewport is None:
                factory = self._viewport_factory
                if factory is None:
                    from .viewport import GeometricViewport
                    factory = GeometricViewport
                self.viewport = factory(self.config, self.adapter.view_config)
                self.viewport.update_snapshot(self.view.snapshot)
            self.scene_button.setEnabled(False)
            self.scene_status.setText("正在显示 UE 视口…" if not self._scene_visible else "正在隐藏 UE 视口…")
            self.viewport.request(not self._scene_visible)
        except Exception as exc:
            self.scene_button.setEnabled(True)
            self.scene_status.setText(f"UE 视口不可用：{exc}")

    def start(self):
        if self.active or self._closing:
            return
        self._change_case()
        self._stop_event = threading.Event()
        self._result, self._error = None, None
        self.mailbox = LatestTelemetry()
        self.run_dir = None
        self.error_details.clear()
        self.error_details.hide()
        case = self.case_combo.currentData()
        config_path=self.config_path
        launch = self.launch_checkbox.isChecked()
        self.environment_combo.setEnabled(False)
        self.case_combo.setEnabled(False)
        self.launch_checkbox.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_button.setEnabled(False)
        self.status.setText("正在执行接口/运行条件核验；通过后才起飞。")

        def work():
            try:
                runner = self._runner
                if runner is None:
                    from .runtime import run
                    runner = run
                self._result = runner(config_path, case, output_root=None,
                                      telemetry=self.mailbox.publish, stop_event=self._stop_event,
                                      launch=launch)
            except BaseException as exc:
                self._error = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=work, name="geometric-validation-runtime", daemon=False)
        self._thread.start()

    def stop(self):
        if self.active:
            self._stop_event.set()
            self.stop_button.setEnabled(False)
            self.status.setText("已请求停止；等待独立运行时保存记录并完成安全收尾。")

    def refresh(self):
        frame = self.mailbox.take()
        if frame is not None:
            try:
                # Both conversion and painting are on the Qt thread.
                self.adapter.update_metadata(frame)
                if frame.get("position") is not None:
                    snapshot = self.adapter.adapt(frame)
                    self.view.set_snapshot(snapshot)
                    roll, pitch, yaw = np.degrees(snapshot["attitude_rpy_rad"])
                    self.attitude_label.setText(f"实测姿态   横滚 {roll:+.2f}°   俯仰 {pitch:+.2f}°   偏航 {yaw:+.2f}°")
                    if self.viewport is not None:
                        self.viewport.update_snapshot(snapshot)
                if not self._stop_event.is_set():
                    t = frame.get("t")
                    label = f" · 参考时间 {float(t):.2f} s" if t is not None else ""
                    self.status.setText(f"{frame.get('phase', '运行中')}{label} · 显示为抽样观察")
            except (ValueError, TypeError, KeyError) as exc:
                self.status.setText(f"显示数据不可用：{exc}；控制与完整日志独立。")
        if self._thread is not None and not self.active:
            self._thread.join(timeout=0)
            self._thread = None
            result = self._result
            if isinstance(result, dict):
                self.run_dir = result.get("run_dir")
                report = result.get("report", {})
                state = report.get("outcome", report.get("status", "请查看原始报告"))
                def verdict(value):
                    return "通过" if value is True else "未通过" if value is False else "未知"
                self.status.setText(f"终态：{state} · 正常完成：{verdict(report.get('normal_completion'))} · "
                                    f"控制约束：{verdict(report.get('control_constraints_passed'))} · "
                                    f"数据审计：{verdict(report.get('data_audit_passed'))} · "
                                    f"降落确认：{verdict(report.get('cleanup', {}).get('landed'))}。完整依据见本次记录。")
                if 'waypoint_passage_passed' in report:
                    self.status.setText(self.status.text()+f"\n连续过点：{verdict(report['waypoint_passage_passed'])}")
                if report.get("error"):
                    self.error_details.setPlainText(str(report["error"]))
                    self.error_details.show()
                    self.status.setText(self.status.text() + "\n失败原因：" + str(report["error"]).splitlines()[0])
            elif result is not None:
                self.run_dir = str(result)
                self.status.setText("运行结束；请分别查看完成状态、控制约束和独立审计。")
            if self._error:
                self.status.setText(f"运行异常：{self._error}。请检查失败记录及降落回执。")
                self.error_details.setPlainText(self._error)
                self.error_details.show()
            self.open_button.setEnabled(bool(self.run_dir))
            self.environment_combo.setEnabled(len(self.environments)>1 and not self._closing)
            self.case_combo.setEnabled(True)
            self.launch_checkbox.setEnabled(True)
            self.start_button.setEnabled(not self._closing)
            self.stop_button.setEnabled(False)
            if self._closing:
                self.close()
        if self.viewport is not None:
            update = self.viewport.take_update()
            if update:
                self._scene_visible = update.get("scene_visible", self._scene_visible)
                self.scene_status.setText(update.get("scene_message", ""))
                self.scene_button.setText("隐藏 UE 视口" if self._scene_visible else "显示 UE 视口")
                if update.get("scene_status") in ("VISIBLE", "HIDDEN", "ERROR"):
                    self.scene_button.setEnabled(not self._closing)
            if self._closing and not self.active and self.viewport.close(timeout=0):
                self.close()

    def open_run(self):
        if self.run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.run_dir).resolve())))

    def open_comparison(self):
        from .environments import matching_comparison
        page=matching_comparison(self.config)
        if page:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(page.resolve())))
        else:
            self.status.setText("当前环境与条件尚无批量实测对比记录。")

    def closeEvent(self, event):
        if self.active:
            self._closing = True
            self.start_button.setEnabled(False)
            self.stop()
            event.ignore()
            return
        if self.viewport is not None and not self.viewport.close(timeout=0):
            self._closing = True
            self.start_button.setEnabled(False)
            self.scene_button.setEnabled(False)
            event.ignore()
            return
        self.timer.stop()
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent geometric-control Qt validation (opens idle)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--environment", help="Task preset ID; selecting does not connect or fly")
    parser.add_argument("--check", action="store_true", help="Offscreen Qt smoke check without AirSim imports/RPC")
    parser.add_argument("--preview", type=Path, help="Save a Qt UI preview only; implies --check")
    args = parser.parse_args(argv)
    if args.environment:
        from .environments import config_for
        args.config=config_for(args.environment)
    checking = args.check or args.preview is not None
    if checking:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    if not QFontDatabase().families():
        font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
        if font_path.exists():
            QFontDatabase.addApplicationFont(str(font_path))
    app.setFont(QFont("Microsoft YaHei", 9))
    window = GeometricControlWindow(args.config)
    window.show()
    if checking:
        app.processEvents()
        initial_environment=window.environment_combo.currentIndex()
        for env_index in range(window.environment_combo.count()):
            window.environment_combo.setCurrentIndex(env_index)
            for index in range(window.case_combo.count()):
                window.case_combo.setCurrentIndex(index)
                for plane in ("3d", "xy", "xz"):
                    window.view.set_plane(plane)
                    app.processEvents()
        window.environment_combo.setCurrentIndex(initial_environment)
        window.case_combo.setCurrentIndex(0)
        window.view.set_plane("3d")
        app.processEvents()
        if args.preview:
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.preview.resolve())):
                raise RuntimeError("Could not save Qt preview")
        assert not window.active
        assert not window.launch_checkbox.isChecked()
        assert "geometric_validation.runtime" not in sys.modules
        assert "airsim" not in sys.modules
        print(json.dumps({"qt_check": "PASS", "runtime_imported": False,
                          "airsim_imported": False, "flight_started": False,
                          "preview_is_airsim_camera": False}, ensure_ascii=False))
        window.close()
        return 0
    return app.exec_()
