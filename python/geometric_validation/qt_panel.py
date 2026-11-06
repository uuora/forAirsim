"""Independent geometric-validation panel; rendering never calls AirSim.

The existing map uses [X, Y, height] while this protocol records NED.  This
adapter is the only coordinate conversion for the display.  The icon shows
measured yaw only; full attitude is evaluated from timestamped quaternions.
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
    QPushButton, QVBoxLayout, QWidget,
)

from dual_target_control.route_view import RouteFlightView


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "geometric_control_validation.json"
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
        # New reference generator only. This plotting preview is discarded for
        # execution; runtime regenerates and times its own frozen reference.
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
                    route_markers=self.markers, targets=[pending])


class GeometricControlWindow(QMainWindow):
    """Starts only the new runtime, and only on an explicit start click."""
    def __init__(self, config_path: Path = DEFAULT_CONFIG, runner=None):
        super().__init__()
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        self._runner = runner
        self._thread = None
        self._stop_event = threading.Event()
        self._result = None
        self._error = None
        self._closing = False
        self.run_dir = None
        self.mailbox = LatestTelemetry()
        self.adapter = MapAdapter(self.config, "hover")
        self.setWindowTitle("几何控制验证 · 独立协议 · AirSim")
        self.resize(1240, 880)
        self.setMinimumSize(800, 650)
        self.setStyleSheet("QMainWindow{background:#edf2f7;} QLabel{color:#24384b;} "
                           "QPushButton{padding:8px 14px;} QComboBox{padding:6px;} "
                           "QFrame#card{background:white;border:1px solid #cdd8e1;border-radius:7px;}")
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 14)
        title = QLabel("几何轨迹跟踪 · 控制基线验证")
        title.setStyleSheet("font-size:20px;font-weight:600;")
        layout.addWidget(title)
        subtitle = QLabel("独立配置与日志；先验证控制器，再开展轨迹方法比较。打开窗口和切换任务不连接、不起飞。")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        task_row = QHBoxLayout()
        task_row.addWidget(QLabel("验证级别"))
        self.case_combo = QComboBox()
        for key, text in CASES:
            if key in self.config["cases"]:
                self.case_combo.addItem(text, key)
        self.case_combo.currentIndexChanged.connect(self._change_case)
        task_row.addWidget(self.case_combo)
        self.task_summary = QLabel()
        self.task_summary.setWordWrap(True)
        task_row.addWidget(self.task_summary, 1)
        layout.addLayout(task_row)
        self.launch_checkbox = QCheckBox("启动独立AirSim（端口未运行时）")
        self.launch_checkbox.setChecked(False)
        self.launch_checkbox.setToolTip("仅在明确点击“开始所选验证”后生效；窗口打开与勾选不启动进程。")
        layout.addWidget(self.launch_checkbox)
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
        limits = QLabel("图标仅显示实测偏航，不显示横滚/俯仰。姿态误差依据日志中的带时间戳实际/参考姿态按 SO(3) 计算。\n"
                        "参考绿色，实测紫红；地图 H = −NED z。参考预览仅供绘图，执行参考会独立生成和计时。\n"
                        "地图接近/经过标记不是控制通过判据；显示抽样不参与评价。")
        limits.setWordWrap(True)
        layout.addWidget(limits)
        self.status = QLabel("空闲 · 尚未连接 AirSim")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
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
        layout.addLayout(actions)
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self._change_case()

    @property
    def active(self):
        return self._thread is not None and self._thread.is_alive()

    def _change_case(self, *_):
        if self.active:
            return
        case = self.case_combo.currentData()
        self.adapter.set_task(self.config, case)
        self.view.config = self.adapter.view_config
        self.view.set_snapshot(self.adapter.idle_snapshot())
        spec = self.config["cases"][case]
        durations = spec.get("segment_durations", [])
        self.task_summary.setText(f"{len(self.adapter.points)} 个任务点 · 预声明参考时长 {sum(durations):g} s · 参数来自独立配置")

    def start(self):
        if self.active or self._closing:
            return
        self._change_case()
        self._stop_event = threading.Event()
        self._result, self._error = None, None
        self.mailbox = LatestTelemetry()
        self.run_dir = None
        case = self.case_combo.currentData()
        launch = self.launch_checkbox.isChecked()
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
                self._result = runner(self.config_path, case, output_root=None,
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
                    self.view.set_snapshot(self.adapter.adapt(frame))
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
            elif result is not None:
                self.run_dir = str(result)
                self.status.setText("运行结束；请分别查看完成状态、控制约束和独立审计。")
            if self._error:
                self.status.setText(f"运行异常：{self._error}。请检查失败记录及降落回执。")
            self.open_button.setEnabled(bool(self.run_dir))
            self.case_combo.setEnabled(True)
            self.launch_checkbox.setEnabled(True)
            self.start_button.setEnabled(not self._closing)
            self.stop_button.setEnabled(False)
            if self._closing:
                self.close()

    def open_run(self):
        if self.run_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.run_dir).resolve())))

    def closeEvent(self, event):
        if self.active:
            self._closing = True
            self.start_button.setEnabled(False)
            self.stop()
            event.ignore()
            return
        self.timer.stop()
        event.accept()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent geometric-control Qt validation (opens idle)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="Offscreen Qt smoke check without AirSim imports/RPC")
    parser.add_argument("--preview", type=Path, help="Save a Qt UI preview only; implies --check")
    args = parser.parse_args(argv)
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
        for index in range(window.case_combo.count()):
            window.case_combo.setCurrentIndex(index)
            for plane in ("3d", "xy", "xz"):
                window.view.set_plane(plane)
                app.processEvents()
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
