"""Lightweight three-dimensional monitor for dual-target flight control.

The view projects world-space XYZ points with a movable orthographic camera.
It deliberately uses QPainter rather than another rendering engine: all flight
and AirSim work stays in ControlRuntime, while the GUI polls a safe snapshot.
Opening the monitor never connects, arms, or takes off automatically.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, QUrl
from PyQt5.QtGui import (
    QColor, QDesktopServices, QFont, QFontDatabase, QLinearGradient, QPainter,
    QPen, QPolygonF,
)
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLayout, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "dual_target_control.json"
RED = QColor("#e6555f")
BLUE = QColor("#3486ea")
GREEN = QColor("#1c9e75")
INK = QColor("#23344e")
STATUS_LABELS = {
    "idle": "待启动", "disconnected": "未连接", "connected": "已连接",
    "connecting": "正在连接", "launching": "正在启动仿真",
    "simulator_starting": "正在启动仿真", "simulator_started": "仿真已启动",
    "ready": "准备就绪", "starting": "正在起飞", "taking_off": "正在起飞",
    "takeoff": "正在起飞",
    "running": "飞行中", "paused": "已暂停", "pausing": "正在暂停",
    "stopping": "正在降落", "landing": "正在降落", "stopped": "已停止",
    "landed": "已降落", "error": "运行异常", "shutdown": "已关闭",
    "closed": "已关闭", "smoke_test": "界面预览",
}


def _control(config: dict) -> dict:
    return config.get("control", config)


def _bounds(config: dict) -> tuple[float, float, float, float, float, float]:
    values = _control(config)
    return (
        float(values.get("x_min", -20.0)), float(values.get("x_max", 20.0)),
        float(values.get("y_min", -20.0)), float(values.get("y_max", 20.0)),
        float(values.get("height_min", values.get("h_min", values.get("z_min", 2.0)))),
        float(values.get("height_max", values.get("h_max", values.get("z_max", 10.0)))),
    )


def _vec(value: Any, fallback=(0.0, 0.0, 0.0)) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float).reshape(3)
        if np.isfinite(result).all():
            return result
    except (TypeError, ValueError):
        pass
    return np.asarray(fallback, dtype=float)


class FlightView(QWidget):
    """Interactive real-XYZ projection, with no OpenGL or image polling."""

    MAX_DISPLAY_TRAIL = 600
    _angles = np.linspace(0.0, 2.0 * math.pi, 41)
    _unit_circles = np.zeros((3, 41, 3))
    for _index, _plane in enumerate(((0, 1), (0, 2), (1, 2))):
        _unit_circles[_index, :, _plane[0]] = np.cos(_angles)
        _unit_circles[_index, :, _plane[1]] = np.sin(_angles)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self.snapshot: dict = {}
        self.setMinimumSize(520, 430)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._mouse = None
        self._button = None
        self._camera_signature = None
        self._label_font = QFont("Microsoft YaHei", 9)
        self.reset_view()

    def reset_view(self):
        self.azimuth = math.radians(-57)
        self.elevation = math.radians(29)
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.update()

    def set_snapshot(self, snapshot: dict):
        self.snapshot = snapshot
        self.config = snapshot.get("config") or self.config
        self.update()

    def _camera(self):
        bounds = _bounds(self.config)
        signature = (bounds, self.width(), self.height(), self.azimuth,
                     self.elevation, self.zoom, float(self.pan[0]), float(self.pan[1]))
        if signature == self._camera_signature:
            return
        self._camera_signature = signature
        az, el = self.azimuth, self.elevation
        self.right = np.array([-math.sin(az), math.cos(az), 0.0])
        self.up = np.array([-math.sin(el) * math.cos(az),
                            -math.sin(el) * math.sin(az), math.cos(el)])
        self.depth = np.cross(self.right, self.up)
        xmin, xmax, ymin, ymax, hmin, hmax = bounds
        self.center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, hmax / 2])
        self.corners = np.array([[x, y, h] for h in (0.0, hmax)
                                 for y in (ymin, ymax) for x in (xmin, xmax)])
        centered = self.corners - self.center
        sx = max(float(np.ptp(centered @ self.right)), 1.0)
        sy = max(float(np.ptp(centered @ self.up)), 1.0)
        self.scale = min((self.width() - 155) / sx,
                         (self.height() - 190) / sy) * self.zoom
        self.screen_center = np.array([self.width() / 2,
                                       self.height() / 2 + 6]) + self.pan
        self.projection_matrix = np.column_stack((self.right, -self.up)) * self.scale
        self.projection_offset = self.screen_center - self.center @ self.projection_matrix

    @staticmethod
    def display_points(points, max_points=None):
        """Decimate only drawing input; leave runtime telemetry untouched."""
        if points is None or len(points) == 0:
            return np.empty((0, 3))
        if max_points is not None and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = points[indices] if isinstance(points, np.ndarray) else [points[i] for i in indices]
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            return np.empty((0, 3))
        return values

    def project_points(self, points):
        return points @ self.projection_matrix + self.projection_offset

    @staticmethod
    def _polygon(projected):
        return QPolygonF([QPointF(x, y) for x, y in projected.tolist()])

    def project(self, point) -> QPointF:
        value = _vec(point) @ self.projection_matrix + self.projection_offset
        return QPointF(float(value[0]), float(value[1]))

    @staticmethod
    def _pen(color, width=1.0, style=Qt.SolidLine):
        pen = QPen(color, width, style)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def _polyline(self, painter: QPainter, points, color, width=1.0,
                  style=Qt.SolidLine, max_points=None):
        values = self.display_points(points, max_points)
        if len(values) < 2:
            return
        painter.setPen(self._pen(color, width, style))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(self._polygon(self.project_points(values)))

    def _label(self, painter, point: QPointF, text: str, color=INK,
               offset=(10, -10), background=False):
        painter.setFont(self._label_font)
        width = painter.fontMetrics().horizontalAdvance(text)
        rect = QRectF(point.x() + offset[0], point.y() + offset[1] - 15,
                      width + 16, 24)
        if rect.right() > self.width() - 14:
            rect.moveLeft(max(14, point.x() - width - abs(offset[0]) - 16))
        if rect.left() < 14:
            rect.moveLeft(14)
        if background:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255, 235))
            painter.drawRoundedRect(rect, 5, 5)
        painter.setPen(color)
        painter.drawText(rect.adjusted(8, 0, -3, 0), Qt.AlignVCenter, text)

    def _draw_box(self, painter):
        xmin, xmax, ymin, ymax, hmin, hmax = _bounds(self.config)
        floor = [[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]]
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#eaf0f6"))
        painter.drawPolygon(QPolygonF([self.project(v) for v in floor]))
        for x in np.linspace(xmin, xmax, 9):
            self._polyline(painter, [[x, ymin, 0], [x, ymax, 0]], QColor("#d8e1ec"))
        for y in np.linspace(ymin, ymax, 9):
            self._polyline(painter, [[xmin, y, 0], [xmax, y, 0]], QColor("#d8e1ec"))
        # All twelve edges belong to a physical 3D volume, not a fixed diagram.
        for a, b in ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
                     (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
            self._polyline(painter, [self.corners[a], self.corners[b]],
                           QColor("#bac8d9"), 1.15)
        # Dashed lower flight plane makes the minimum altitude visible.
        if hmin > 0.01:
            self._polyline(painter, [[xmin, ymin, hmin], [xmax, ymin, hmin],
                                    [xmax, ymax, hmin], [xmin, ymax, hmin],
                                    [xmin, ymin, hmin]], QColor("#c4d0dd"),
                           0.85, Qt.DashLine)
        origin = [xmin, ymin, 0]
        axes = (([xmax, ymin, 0], "X / 米", "#607e9f"),
                ([xmin, ymax, 0], "Y / 米", "#607e9f"),
                ([xmin, ymin, hmax], "高度 / 米", "#607e9f"))
        for end, text, color in axes:
            self._polyline(painter, [origin, end], QColor(color), 1.8)
            self._label(painter, self.project(end), text, QColor(color), (6, -3))
        self._label(painter, self.project(origin),
                    f"{xmin:g}, {ymin:g}, 0", QColor("#8594a6"), (0, 27))
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor("#75859a"))
        painter.drawText(QRectF(25, 61, self.width() - 50, 24),
                         f"X  {xmin:g} ~ {xmax:g} m     Y  {ymin:g} ~ {ymax:g} m"
                         f"     高度  {hmin:g} ~ {hmax:g} m")

    def _target(self, painter, point, index):
        color = RED if index == 0 else BLUE
        xyz = _vec(point)
        projected = self.project(xyz)
        drop = xyz.copy(); drop[2] = 0
        faint = QColor(color); faint.setAlpha(95)
        self._polyline(painter, [drop, xyz], faint, 0.9, Qt.DotLine)
        bottom = self.project(drop)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(self._pen(faint, 1.1))
        painter.drawEllipse(bottom, 4, 2.7)
        radius = float(_control(self.config).get("arrival_radius", 0.5))
        shell = QColor(color); shell.setAlpha(52)
        circles = self._unit_circles * radius + xyz
        projected_circles = self.project_points(circles.reshape(-1, 3)).reshape(3, 41, 2)
        painter.setPen(self._pen(shell, 1.4)); painter.setBrush(Qt.NoBrush)
        for circle in projected_circles:
            painter.drawPolyline(self._polygon(circle))
        halo = QColor(color); halo.setAlpha(22)
        painter.setPen(Qt.NoPen)
        painter.setBrush(halo); painter.drawEllipse(projected, 15, 15)
        painter.setBrush(color); painter.drawEllipse(projected, 6, 6)
        painter.setBrush(QColor("#ffffff")); painter.drawEllipse(projected, 2, 2)
        name = "01  当前目标" if index == 0 else "02  下一目标"
        self._label(painter, projected, name, color, (14, -9), True)
        self._label(painter, projected,
                    f"({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) m",
                    QColor("#7a8a9e"), (14, 15), True)

    def _drone(self, painter):
        position = _vec(self.snapshot.get("position"))
        yaw = float(self.snapshot.get("yaw", 0.0) or 0.0)
        nose = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        # Keep the vehicle legible at wide bounds; only its icon is enlarged.
        arm = max(0.65, 9.0 / max(self.scale, 0.1))
        rotors = [position + arm * (i * nose + j * side)
                  for i, j in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
        self._polyline(painter, [rotors[0], rotors[1]], INK, 3.2)
        self._polyline(painter, [rotors[2], rotors[3]], INK, 3.2)
        for rotor in rotors:
            projected = self.project(rotor)
            painter.setPen(self._pen(INK, 1.7))
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.drawEllipse(projected, 5.5, 3.5)
        projected = self.project(position)
        painter.setPen(self._pen(QColor("#ffffff"), 1.4))
        painter.setBrush(INK); painter.drawEllipse(projected, 5.6, 5.6)
        end = position + nose * arm * 3.2
        self._polyline(painter, [position, end], QColor("#ec9840"), 2.5)
        self._polyline(painter, [end - nose * arm * 0.7 + side * arm * 0.4,
                                end, end - nose * arm * 0.7 - side * arm * 0.4],
                       QColor("#ec9840"), 2.5)
        self._label(painter, projected, "无人机", INK, (-53, -22), True)

    def _legend(self, painter):
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 230))
        rect = QRectF(22, self.height() - 91, min(510, self.width() - 44), 62)
        painter.drawRoundedRect(rect, 9, 9)
        painter.setFont(QFont("Microsoft YaHei", 9))
        entries = [(RED, "当前目标"), (BLUE, "下一目标"),
                   (GREEN, "规划轨迹"), (QColor("#79899b"), "实际轨迹")]
        x = 38
        for color, text in entries:
            painter.setPen(self._pen(color, 2.5)); painter.drawLine(x, self.height() - 69,
                                                                  x + 18, self.height() - 69)
            painter.setPen(QColor("#586b82"))
            painter.drawText(QPointF(x + 25, self.height() - 65), text)
            x += 119
        painter.setPen(QColor("#8493a6"))
        painter.drawText(QPointF(38, self.height() - 43),
                         "左键旋转   ·   右键平移   ·   滚轮缩放   ·   双击复位")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#f7fafe"))
        gradient.setColorAt(1, QColor("#edf3f9"))
        painter.fillRect(self.rect(), gradient)
        self._camera()
        painter.setFont(QFont("Microsoft YaHei", 11, QFont.DemiBold))
        painter.setPen(INK)
        painter.drawText(QRectF(25, 23, self.width() - 50, 27), "三维飞行场地")
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor("#7a8ca3"))
        painter.drawText(QRectF(self.width() - 200, 25, 175, 24), Qt.AlignRight,
                         "坐标 / 米 · 高度向上")
        self._draw_box(painter)
        self._polyline(painter, self.snapshot.get("trail", []),
                       QColor("#77889b"), 2.2, max_points=self.MAX_DISPLAY_TRAIL)
        self._polyline(painter, self.snapshot.get("planned_path", []), GREEN, 2.2)
        targets = self.snapshot.get("targets") or []
        # Painter's depth sort prevents arbitrary target-array occlusion.
        ordered = sorted(enumerate(targets[:2]),
                         key=lambda entry: float(np.dot(_vec(entry[1]), self.depth)))
        for index, target in ordered:
            self._target(painter, target, index)
        if self.snapshot.get("position") is not None:
            self._drone(painter)
        else:
            painter.setFont(QFont("Microsoft YaHei", 11))
            painter.setPen(QColor("#8797ab"))
            painter.drawText(QRectF(0, self.height() * .5 - 12, self.width(), 25),
                             Qt.AlignCenter, "连接仿真后显示无人机与目标")
        self._legend(painter)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._mouse = event.pos(); self._button = event.button()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._mouse is not None:
            delta = event.pos() - self._mouse
            if self._button == Qt.LeftButton:
                self.azimuth += math.radians(delta.x() * .5)
                self.elevation = max(math.radians(5), min(math.radians(85),
                                     self.elevation + math.radians(delta.y() * .4)))
            else:
                self.pan += np.array([delta.x(), delta.y()])
            self._mouse = event.pos(); self.update()

    def mouseReleaseEvent(self, event):
        self._mouse = None; self._button = None; self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def wheelEvent(self, event):
        self.zoom = min(4.0, max(.4, self.zoom * math.exp(event.angleDelta().y() / 1200)))
        self.update(); event.accept()


class MetricTile(QFrame):
    def __init__(self, title, unit, parent=None):
        super().__init__(parent)
        self.setObjectName("metric")
        self.setMinimumHeight(66)
        layout = QVBoxLayout(self); layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)
        caption = QLabel(title); caption.setObjectName("caption")
        self.value = QLabel("—"); self.value.setObjectName("metricValue")
        self.unit = unit
        layout.addWidget(caption); layout.addWidget(self.value)

    def set_value(self, value, digits=1):
        try:
            numeric = float(value)
            self.value.setText(f"{numeric:.{digits}f} {self.unit}" if math.isfinite(numeric) else "—")
        except (TypeError, ValueError):
            self.value.setText("—")


STYLES = """
QMainWindow { background: #f4f7fb; }
QWidget { color: #23344e; font-family: 'Microsoft YaHei'; font-size: 12px; }
QFrame#panel { background: white; border: 1px solid #e0e7ef; border-radius: 12px; }
QLabel#heading { font-size: 23px; font-weight: 600; }
QLabel#subtitle, QLabel#caption { color: #7b8a9d; }
QLabel#sectionTitle { font-weight: 600; font-size: 13px; }
QLabel#badge { background: #e8eef7; color: #47678b; padding: 8px 13px; border-radius: 13px; }
QLabel#message { color: #73839a; font-size: 11px; }
QFrame#metric { background: #f5f8fc; border: 0; border-radius: 8px; }
QLabel#metricValue { font-size: 19px; font-weight: 600; }
QPushButton { border: 1px solid #d8e1ed; border-radius: 6px; padding: 8px 10px;
              background: white; color: #3d5878; }
QPushButton:hover { border-color: #7397c3; background: #f3f7fd; }
QPushButton:disabled { color: #a6b1be; background: #f7f9fc; border-color: #e5eaf1; }
QPushButton#primary { background: #275c97; border-color: #275c97; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #326da9; }
QPushButton#primary:disabled { background: #a7bfd9; border-color: #a7bfd9; color: #eff4fa; }
QPushButton#stop { border-color: #ead2d5; color: #b54c57; }
QDoubleSpinBox, QComboBox { background: #fff; border: 1px solid #d8e1ed;
                         border-radius: 5px; padding: 5px 7px; }
QDoubleSpinBox:disabled, QComboBox:disabled { background: #f6f8fb; color: #a1aebb; }
"""


class ControlWindow(QMainWindow):
    """No GUI operation calls RPC directly; the runtime owns all flight I/O."""

    def __init__(self, runtime, config: dict, config_path: Path = DEFAULT_CONFIG,
                 runtime_factory: Callable | None = None,
                 overview_stream_factory: Callable | None = None):
        super().__init__()
        self.runtime = runtime
        self.runtime_factory = runtime_factory
        self.overview_stream_factory = overview_stream_factory
        self.overview_window = None
        self._overview_pending = False
        self._scene_pending = False
        self._scene_hide_requested = False
        self.config = copy.deepcopy(config)
        self.config_path = Path(config_path)
        self._last_snapshot: dict = {}
        self._closing = False
        self._pending_mode = None
        self._local_message = ""
        self._local_until = 0.0
        # The Windows offscreen plugin has no system font database by default.
        # Loading an installed font lets deterministic previews render Chinese.
        if not QFontDatabase().families():
            font_file = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
            if font_file.is_file():
                QFontDatabase.addApplicationFont(str(font_file))
        self.setWindowTitle("无人机双目标控制 · 三维工作台")
        self.resize(1360, 970)
        self.setMinimumSize(1100, 800)
        self.setStyleSheet(STYLES)
        container = QWidget(); self.setCentralWidget(container)
        outer = QVBoxLayout(container); outer.setContentsMargins(25, 19, 25, 18)
        outer.setSpacing(17)
        header = QHBoxLayout()
        titles = QVBoxLayout(); titles.setSpacing(4)
        title = QLabel("无人机双目标控制"); title.setObjectName("heading")
        subtitle = QLabel("三维有界场地  /  双目标滚动规划  /  位置与朝向同步跟踪")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title); titles.addWidget(subtitle)
        header.addLayout(titles); header.addStretch()
        self.badge = QLabel("待启动"); self.badge.setObjectName("badge")
        header.addWidget(self.badge); outer.addLayout(header)
        body = QHBoxLayout(); body.setSpacing(18)
        view_panel = QFrame(); view_panel.setObjectName("panel")
        view_layout = QVBoxLayout(view_panel); view_layout.setContentsMargins(1, 1, 1, 1)
        self.view = FlightView(self.config); view_layout.addWidget(self.view)
        body.addWidget(view_panel, 1)
        panel = QFrame(); panel.setObjectName("panel"); panel.setMinimumWidth(310)
        layout = QVBoxLayout(panel); layout.setContentsMargins(18, 16, 18, 15)
        layout.setSpacing(7)
        layout.setSizeConstraint(QLayout.SetMinimumSize)
        mode_row = QHBoxLayout(); mode_row.addWidget(self._section("运行模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("AirSim 仿真", "airsim")
        self.mode_combo.addItem("离线控制演示", "demo")
        initial = str(self.runtime.snapshot().get("mode", "airsim"))
        self.mode_combo.setCurrentIndex(1 if initial == "demo" else 0)
        mode_row.addWidget(self.mode_combo, 1); layout.addLayout(mode_row)
        self.source = QLabel(); self.source.setObjectName("message")
        self.source.setWordWrap(True); layout.addWidget(self.source)
        self.dynamics_status = QLabel("动力参数：连接后核验")
        self.dynamics_status.setObjectName("message")
        self.dynamics_status.setWordWrap(True); layout.addWidget(self.dynamics_status)
        startup = QHBoxLayout()
        self.launch_button = QPushButton("启动仿真")
        self.connect_button = QPushButton("连接 AirSim")
        startup.addWidget(self.launch_button); startup.addWidget(self.connect_button)
        layout.addLayout(startup)
        self.launch_button.clicked.connect(lambda: self._call("start_simulator"))
        self.connect_button.clicked.connect(lambda: self._call("connect"))
        self.scene_button = QPushButton("查看 UE4 场景")
        self.scene_button.setToolTip(
            "显示或隐藏同一 AirSim 仿真的 UE4 场景，仅用于观察；不会起飞、重启或暂停。"
            "场景内左拖旋转、右拖平移、滚轮缩放，双击无人机聚焦；左上可切跟随或整体场景。")
        self.scene_button.clicked.connect(self._toggle_scene)
        layout.addWidget(self.scene_button)
        self.scene_status = QLabel("")
        self.scene_status.setObjectName("message"); self.scene_status.setWordWrap(True)
        self.scene_status.setVisible(False); layout.addWidget(self.scene_status)
        controls = QHBoxLayout()
        self.start_button = QPushButton("开始飞行"); self.start_button.setObjectName("primary")
        self.pause_button = QPushButton("暂停")
        controls.addWidget(self.start_button, 1); controls.addWidget(self.pause_button, 1)
        layout.addLayout(controls)
        self.stop_button = QPushButton("停止并降落"); self.stop_button.setObjectName("stop")
        layout.addWidget(self.stop_button)
        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause_or_resume)
        self.stop_button.clicked.connect(lambda: self._call("stop"))
        layout.addWidget(self._divider())
        layout.addWidget(self._section("实时状态"))
        metrics = QGridLayout(); metrics.setSpacing(7)
        self.metrics = {}
        for index, (key, name, unit) in enumerate((
                ("speed", "当前速度", "m/s"), ("altitude", "飞行高度", "m"),
                ("distance", "距当前目标", "m"), ("hits", "已到达", "个"),
                ("yaw_error_deg", "朝向误差", "°"), ("planning_ms", "规划耗时", "ms"))):
            tile = MetricTile(name, unit); self.metrics[key] = tile
            metrics.addWidget(tile, index // 2, index % 2)
        layout.addLayout(metrics)
        self.position_label = QLabel("位置  —     RPC  —")
        self.position_label.setObjectName("message"); layout.addWidget(self.position_label)
        layout.addWidget(self._divider())
        layout.addWidget(self._section("场地与速度"))
        self.inputs = {}
        form = QGridLayout(); form.setHorizontalSpacing(5); form.setVerticalSpacing(7)
        bounds = _bounds(self.config)
        names = (("X / 米", "x_min", "x_max", bounds[0:2]),
                 ("Y / 米", "y_min", "y_max", bounds[2:4]),
                 ("高度 / 米", "height_min", "height_max", bounds[4:6]))
        for row, (label, lower, upper, values) in enumerate(names):
            form.addWidget(QLabel(label), row, 0)
            for column, (key, value) in enumerate(zip((lower, upper), values), 1):
                spin = self._spin(value, 0 if "height" in key else -5000, 5000)
                self.inputs[key] = spin; form.addWidget(spin, row, column)
        form.addWidget(QLabel("速度上限"), 3, 0)
        spin = self._spin(_control(self.config).get("max_speed", 4.0), .1, 100)
        spin.setSuffix(" m/s"); self.inputs["max_speed"] = spin
        form.addWidget(spin, 3, 1, 1, 2)
        for row, (label, key) in enumerate((("点间距下限", "target_distance_min"),
                                          ("点间距上限", "target_distance_max")), 4):
            spin = self._spin(_control(self.config).get(key, 0.0), 0, 5000)
            spin.setSuffix(" m")
            spin.setToolTip("相邻随机航路点的三维距离；上下限均为 0 时采用全场随机生成。")
            self.inputs[key] = spin
            form.addWidget(QLabel(label), row, 0); form.addWidget(spin, row, 1, 1, 2)
        layout.addLayout(form)
        self.apply_button = QPushButton("应用参数")
        self.apply_button.clicked.connect(self._apply)
        layout.addWidget(self.apply_button)
        self.message = QLabel("打开窗口不会自动起飞。")
        self.message.setObjectName("message"); self.message.setWordWrap(True)
        self.message.setMinimumHeight(33); layout.addWidget(self.message)
        hint = QLabel("机头偏航跟踪 · 仿真云台指向目标")
        hint.setObjectName("message"); layout.addWidget(hint)
        layout.addStretch(1)
        bottom = QHBoxLayout()
        reset_button = QPushButton("复位视图"); reset_button.clicked.connect(self.view.reset_view)
        config_button = QPushButton("配置文件")
        config_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(self.config_path.resolve()))))
        bottom.addWidget(reset_button); bottom.addWidget(config_button); layout.addLayout(bottom)
        sidebar = QScrollArea(); sidebar.setFixedWidth(332)
        sidebar.setFrameShape(QFrame.NoFrame); sidebar.setWidgetResizable(True)
        sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar.setStyleSheet("QScrollArea { border: 0; background: transparent; }"
                              "QScrollBar:vertical { width: 7px; background: #edf2f8; }"
                              "QScrollBar::handle:vertical { background: #bfccdd; min-height: 30px; }"
                              "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }")
        sidebar.setWidget(panel); body.addWidget(sidebar); outer.addLayout(body, 1)
        footer = QHBoxLayout()
        self.footer = QLabel("直接读取目标坐标 · 红点为当前目标，蓝点为下一目标")
        self.footer.setObjectName("caption"); footer.addWidget(self.footer)
        footer.addStretch()
        self.overview_button = QPushButton("俯视图（可选）")
        self.overview_button.setFlat(True)
        self.overview_button.setToolTip("同一 AirSim 的可选俯视相机，仅用于补充观察。")
        self.overview_button.clicked.connect(self._open_overview)
        footer.addWidget(self.overview_button)
        self.logs_button = QPushButton("运行记录")
        self.logs_button.setFlat(True); self.logs_button.clicked.connect(self._open_logs)
        footer.addWidget(self.logs_button); outer.addLayout(footer)
        self.mode_combo.currentIndexChanged.connect(self._change_mode)
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh)
        self.timer.start(50); self.refresh()
        self.lifecycle_timer = QTimer(self)
        self.lifecycle_timer.timeout.connect(self._finish_lifecycle)

    @staticmethod
    def _section(text):
        label = QLabel(text); label.setObjectName("sectionTitle"); return label

    @staticmethod
    def _divider():
        result = QFrame(); result.setFrameShape(QFrame.HLine)
        result.setStyleSheet("background: #eaf0f6; border: 0; max-height: 1px;")
        return result

    @staticmethod
    def _spin(value, minimum, maximum):
        result = QDoubleSpinBox(); result.setRange(minimum, maximum)
        result.setDecimals(1); result.setSingleStep(.5); result.setValue(float(value))
        result.setButtonSymbols(QDoubleSpinBox.NoButtons); result.setMinimumWidth(65)
        return result

    def _call(self, name):
        try:
            getattr(self.runtime, name)()
        except Exception as error:
            self._set_message(f"{name}：{error}")

    def _set_message(self, text):
        self._local_message = str(text)
        self._local_until = time.monotonic() + 5.0
        self.message.setText(self._local_message)

    def _start(self):
        if self._apply():
            self._call("start")
            self._local_until = 0.0

    def _pause_or_resume(self):
        self._call("resume" if str(self._last_snapshot.get("status", "")).lower() == "paused" else "pause")

    def _apply(self):
        config = copy.deepcopy(self.config)
        control = config.setdefault("control", {})
        changes = {key: spin.value() for key, spin in self.inputs.items()}
        if any(changes[lower] >= changes[upper] for lower, upper in
               (("x_min", "x_max"), ("y_min", "y_max"), ("height_min", "height_max"))):
            self._set_message("范围无效：每个下限必须小于上限。")
            return False
        # Preserve the runtime's canonical altitude spelling when present.
        for key, value in changes.items():
            aliases = {"height_min": ("height_min", "h_min", "z_min"),
                       "height_max": ("height_max", "h_max", "z_max")}
            if key in aliases:
                key = next((candidate for candidate in aliases[key] if candidate in control), key)
            control[key] = value
        try:
            # Validation is pure local work; rejected parameters must not enqueue
            # a flight behind a failed asynchronous configure command.
            from .core import ControlConfig
            from .trajectory_profile import load_trajectory_profile
            ControlConfig.from_dict(control)
            load_trajectory_profile(config)
            self.runtime.configure(config)
            self.config = config
            self._set_message("参数已提交；空闲时生效。")
            return True
        except Exception as error:
            self._set_message(str(error)); return False

    def _change_mode(self):
        if not self.runtime_factory:
            return
        try:
            self._close_overview()
            self._overview_pending = False
            self._hide_scene_for_lifecycle()
            self._pending_mode = self.mode_combo.currentData()
            self.runtime.shutdown()
            self.lifecycle_timer.start(50)
            self._set_message("正在结束当前运行，随后切换模式。")
            self.mode_combo.setEnabled(False)
        except Exception as error:
            self._pending_mode = None
            self._set_message(str(error))

    def _finish_lifecycle(self):
        ready = self.runtime.wait_closed(0) if hasattr(self.runtime, "wait_closed") else True
        if not ready:
            return
        self.lifecycle_timer.stop()
        if self._closing:
            self.timer.stop()
            self.close()
        elif self._pending_mode is not None:
            mode = self._pending_mode
            self._pending_mode = None
            try:
                self.runtime = self.runtime_factory(copy.deepcopy(self.config), mode)
                self._scene_hide_requested = False
                self._set_message("模式已切换，点击开始飞行启动控制。")
                self.refresh()
            except Exception as error:
                self._set_message(str(error))

    def _open_logs(self):
        directory = self._last_snapshot.get("run_dir")
        if directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(directory).resolve())))

    def request_overview(self):
        """Open after the user connects AirSim; this request never connects it."""
        self._overview_pending = True
        self.refresh()

    def request_scene(self):
        """Request observation once the user has connected; never connect or fly."""
        self._scene_pending = True
        self.refresh()

    def _toggle_scene(self):
        snapshot = self.runtime.snapshot()
        if (str(snapshot.get("mode", "")).lower() != "airsim"
                or not snapshot.get("connected", False)
                or self._closing or self._pending_mode is not None):
            self._set_message("连接 AirSim 后可查看 UE4 场景。")
            return
        self._scene_pending = False
        self._call("hide_scene" if snapshot.get("scene_visible", False) else "show_scene")

    def _hide_scene_for_lifecycle(self):
        self._scene_pending = False
        if not self._scene_hide_requested:
            self._scene_hide_requested = True
            self._call("hide_scene")

    def _open_overview(self):
        snapshot = self.runtime.snapshot()
        if (str(snapshot.get("mode", "")).lower() != "airsim"
                or not snapshot.get("connected", False)
                or self._closing or self._pending_mode is not None):
            self._set_message("连接 AirSim 后可打开可选俯视图。")
            return
        if self.overview_window is not None:
            self.overview_window.show()
            self.overview_window.raise_()
            self.overview_window.activateWindow()
            return
        try:
            from .overview_ui import OverviewWindow
            config = copy.deepcopy(snapshot.get("config") or self.config)
            self.overview_window = OverviewWindow(
                config, self.runtime, stream_factory=self.overview_stream_factory, parent=self)
            self.overview_window.closed.connect(self._overview_closed)
            self.overview_window.show()
        except Exception as error:
            self.overview_window = None
            self._set_message(f"俯视图连接失败：{error}")

    def _overview_closed(self):
        window = self.overview_window
        self.overview_window = None
        if window is not None:
            window.deleteLater()

    def _close_overview(self):
        if self.overview_window is not None:
            self.overview_window.close()

    def refresh(self):
        snapshot = self.runtime.snapshot()
        self._last_snapshot = snapshot
        self.view.set_snapshot(snapshot)
        status = str(snapshot.get("status", "idle")).lower()
        mode = str(snapshot.get("mode", self.mode_combo.currentData()))
        running = bool(snapshot.get("running", False))
        paused = status == "paused"
        landing_failure = status == "error" and running
        transition = status in {"connecting", "launching", "simulator_starting", "starting",
                                "taking_off", "takeoff", "stopping", "landing", "pausing"}
        connected = bool(snapshot.get("connected", False))
        operable = not self._closing and self._pending_mode is None
        can_view_map = mode == "airsim" and connected and operable
        self.scene_button.setEnabled(can_view_map)
        self.scene_button.setText("隐藏 UE4 场景" if snapshot.get("scene_visible", False)
                                  else "查看 UE4 场景")
        scene_message = str(snapshot.get("scene_message") or "")
        self.scene_status.setText(scene_message)
        self.scene_status.setVisible(bool(scene_message))
        if self._scene_pending and can_view_map:
            self._scene_pending = False
            self._call("show_scene")
        self.overview_button.setEnabled(can_view_map)
        if not can_view_map:
            self._close_overview()
        elif self._overview_pending:
            self._overview_pending = False
            self._open_overview()
        idle = not running and not paused and not transition and operable
        self.badge.setText(STATUS_LABELS["error"] if landing_failure else
                           "结束后关闭" if self._closing else STATUS_LABELS.get(status, status))
        self.source.setText("AirSim 状态反馈 · 控制指令通过 RPC 发送" if mode == "airsim"
                            else "离线动力学演示 · 数据来自简化模型")
        dynamics = snapshot.get("dynamics") or {}
        if mode == "demo":
            self.dynamics_status.setText("动力参数：离线简化模型")
        elif dynamics.get("verified"):
            self.dynamics_status.setText(
                f"模型质量 {dynamics['mass_kg']:.2f} kg · 标称推重比 {dynamics['nominal_twr']:.2f}（已核验）")
        else:
            self.dynamics_status.setText("动力参数：连接后核验")
        self.launch_button.setEnabled(mode == "airsim" and idle)
        self.connect_button.setEnabled(mode == "airsim" and idle)
        self.start_button.setEnabled(idle and (connected or mode == "demo"))
        self.pause_button.setEnabled(status in {"running", "paused"} and not transition and operable)
        self.pause_button.setText("继续飞行" if paused else "暂停")
        self.stop_button.setEnabled((running or paused or status in {"starting", "taking_off", "takeoff"})
                                    and status not in {"stopping", "landing"}
                                    and (operable or landing_failure))
        self.stop_button.setText("重试降落" if landing_failure else "停止并降落")
        self.mode_combo.setEnabled(idle and self.runtime_factory is not None)
        self.apply_button.setEnabled(idle)
        for spin in self.inputs.values():
            spin.setEnabled(idle)
        position = snapshot.get("position")
        velocity = snapshot.get("velocity")
        self.metrics["speed"].set_value(float(np.linalg.norm(_vec(velocity))) if velocity is not None else None)
        self.metrics["altitude"].set_value(_vec(position)[2] if position is not None else None)
        for key in ("distance", "hits", "yaw_error_deg", "planning_ms"):
            self.metrics[key].set_value(snapshot.get(key), 0 if key == "hits" else 1)
        if position is not None:
            x, y, h = _vec(position)
            rpc = snapshot.get("rpc_ms")
            rpc_text = f"{float(rpc):.1f} ms" if rpc is not None else "—"
            self.position_label.setText(f"位置  ({x:.1f}, {y:.1f}, {h:.1f}) m   ·   RPC  {rpc_text}")
        else:
            self.position_label.setText("位置  —   ·   RPC  —")
        message = snapshot.get("message")
        if landing_failure:
            self.message.setText(str(message or "降落未完成，请点击重试降落。"))
        elif self._closing:
            self.message.setText("正在结束控制并降落，完成后关闭窗口。")
        elif time.monotonic() < self._local_until:
            self.message.setText(self._local_message)
        elif message:
            self.message.setText(str(message))
        self.logs_button.setEnabled(bool(snapshot.get("run_dir")))
        self.footer.setText("离线演示：用于检查规划与交互；飞行性能须在 AirSim 中实测。"
                            if mode == "demo" else
                            "直接读取目标坐标 · 红点为当前目标，蓝点为下一目标")

    def closeEvent(self, event):
        self._overview_pending = False
        self._close_overview()
        self._hide_scene_for_lifecycle()
        ready = self.runtime.wait_closed(0) if hasattr(self.runtime, "wait_closed") else True
        if ready:
            self.timer.stop()
            self.lifecycle_timer.stop()
            if not self._closing:
                self.runtime.shutdown()
            event.accept()
        else:
            event.ignore()
            if not self._closing:
                self._closing = True
                self._pending_mode = None
                self.runtime.shutdown()
                self.lifecycle_timer.start(50)
                self.refresh()


class _PreviewRuntime:
    """Deterministic preview exclusively for --smoke-test, never a flight client."""
    def __init__(self, config):
        self.config = config
        ts = np.linspace(0, 1, 120)
        trail = [[-13 + 16*t, -11 + 10*t, 3 + 2*t + .5*math.sin(t*4)] for t in ts]
        p = np.asarray(trail[-1]); target = np.array([11, 7, 8])
        next_target = np.array([-9, 11, 5])
        path = []
        for t in ts:
            path.append(((1-t)*p + t*target +
                         np.array([3.5*math.sin(math.pi*t), -2*math.sin(math.pi*t), 0])).tolist())
        for t in ts[1:]:
            path.append(((1-t)*target + t*next_target +
                         np.array([0, 6*math.sin(math.pi*t), .9*math.sin(math.pi*t)])).tolist())
        self.data = dict(status="smoke_test", mode="demo", position=p.tolist(),
                         velocity=[3.2, 1.6, .5], yaw=.65,
                         targets=[target.tolist(), next_target.tolist()], planned_path=path,
                         trail=trail, hits=12, distance=float(np.linalg.norm(target-p)),
                         yaw_error_deg=4.3, planning_ms=1.7, rpc_ms=None,
                         message="界面预览，数值为示例；未连接仿真。",
                         run_dir="", connected=False, running=False, config=config,
                         scene_visible=False, scene_message="")

    def snapshot(self): return copy.deepcopy(self.data)
    def configure(self, config): self.config = config; self.data["config"] = config
    def shutdown(self): pass
    def start(self): pass
    def pause(self): pass
    def resume(self): pass
    def stop(self): pass
    def start_simulator(self): pass
    def connect(self): pass
    def show_scene(self): self.data["scene_visible"] = True
    def hide_scene(self): self.data["scene_visible"] = False


def main(argv=None):
    parser = argparse.ArgumentParser(description="无人机双目标三维控制工作台")
    parser.add_argument("--mode", choices=("airsim", "demo"), default="airsim")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--scene", action="store_true",
                        help="连接 AirSim 后显示同一仿真的 UE4 场景，不自动连接或起飞")
    parser.add_argument("--overview", action="store_true",
                        help="在连接 AirSim 后打开可选俯视相机，不自动连接或起飞")
    parser.add_argument("--launch", action="store_true",
                        help="打开窗口后启动并连接仿真，不自动起飞")
    parser.add_argument("--smoke-test", action="store_true",
                        help="离屏绘制示例状态、保存截图并退出，不连接 AirSim")
    args = parser.parse_args(argv)
    if args.scene and (args.mode != "airsim" or args.smoke_test):
        parser.error("--scene 需要 AirSim 模式，且不能与 --smoke-test 一起使用。")
    if args.overview and (args.mode != "airsim" or args.smoke_test):
        parser.error("--overview 需要 AirSim 模式，且不能与 --smoke-test 一起使用。")
    if args.launch and (args.mode != "airsim" or args.smoke_test):
        parser.error("--launch 需要 AirSim 模式，且不能与 --smoke-test 一起使用。")
    if args.smoke_test:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv[:1]); app.setFont(QFont("Microsoft YaHei", 9))
    if args.config.is_file():
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    elif args.smoke_test:
        config = {"control": dict(x_min=-20, x_max=20, y_min=-20, y_max=20,
                                  height_min=2, height_max=10, arrival_radius=.5, max_speed=4)}
    else:
        parser.error(f"找不到配置文件：{args.config}")
    factory = None
    if args.smoke_test:
        runtime = _PreviewRuntime(config)
    else:
        # Import only when opening an operational monitor; preview stays independent.
        from .runtime import ControlRuntime
        factory = lambda cfg, mode: ControlRuntime(cfg, mode=mode)
        runtime = factory(config, args.mode)
    window = ControlWindow(runtime, config, args.config, factory)
    window.show()
    if args.launch:
        QTimer.singleShot(0, runtime.start_simulator)
    if args.scene:
        window.request_scene()
    if args.overview:
        window.request_overview()
    if args.screenshot or args.smoke_test:
        screenshot = args.screenshot or ROOT / "reports" / "dual_target_ui_preview.png"
        def capture():
            window.refresh()
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(screenshot)):
                print(f"截图保存失败：{screenshot}", file=sys.stderr)
                app.exit(1)
                return
            print(str(screenshot.resolve()))
            if args.smoke_test:
                window.close(); app.quit()
        QTimer.singleShot(300, capture)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
