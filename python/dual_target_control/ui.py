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
import subprocess
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
ACTUAL_TRAIL = QColor("#d51acb")
INK = QColor("#23344e")
STATUS_LABELS = {
    "idle": "待启动", "disconnected": "未连接", "connected": "已连接",
    "connecting": "正在连接", "launching": "正在启动仿真",
    "simulator_starting": "正在启动仿真", "simulator_started": "仿真已启动",
    "ready": "准备就绪", "starting": "正在起飞", "taking_off": "正在起飞",
    "takeoff": "正在起飞",
    "configuring": "正在应用任务",
    "moving_to_start": "前往准备点", "planning": "悬停规划中", "settling": "执行前收敛",
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
        self.view_plane = '3d'
        self.azimuth = math.radians(-57)
        self.elevation = math.radians(29)
        self.zoom = 1.0
        self.pan = np.zeros(2)
        self.update()

    def set_plane(self,plane):
        if plane=='3d':self.reset_view();return
        self.view_plane=plane
        self.azimuth=math.radians(-90)
        self.elevation=math.pi/2 if plane=='xy' else 0.
        self.zoom=1.;self.pan=np.zeros(2);self.update()

    @staticmethod
    def _grid_step(span):
        raw=max(span/8.,.01);base=10**math.floor(math.log10(raw))
        return next(v*base for v in (1,2,5,10) if v*base>=raw)

    def set_snapshot(self, snapshot: dict):
        self.snapshot = snapshot
        self.config = snapshot.get("config") or self.config
        self.update()

    def _camera(self):
        bounds = self._display_bounds()
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
        self.scale = max(.01,min((self.width() - 110) / sx,
                         max(40,self.height() - 155) / sy)) * self.zoom
        self.screen_center = np.array([self.width() / 2,
                                       self.height() / 2 + 6]) + self.pan
        self.projection_matrix = np.column_stack((self.right, -self.up)) * self.scale
        self.projection_offset = self.screen_center - self.center @ self.projection_matrix

    def _display_bounds(self):
        task=self.config.get('preparation',{}).get('task')
        if not task:return _bounds(self.config)
        points=np.array([task['start'],*task['points']],dtype=float)
        low=points.min(axis=0);high=points.max(axis=0)
        return (float(low[0]-3),float(high[0]+3),float(low[1]-3),float(high[1]+3),
                0.,float(max(8,high[2]+3)))

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
        xmin, xmax, ymin, ymax, hmin, hmax = self._display_bounds()
        floor = [[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]]
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#eaf0f6"))
        painter.drawPolygon(QPolygonF([self.project(v) for v in floor]))
        step=self._grid_step(max(xmax-xmin,ymax-ymin))
        for x in np.arange(math.ceil(xmin/step)*step,xmax+1e-8,step):
            self._polyline(painter, [[x, ymin, 0], [x, ymax, 0]], QColor("#d8e1ec"))
            if self.view_plane in ('xy','xz'):self._label(painter,self.project([x,ymin,0]),f'{x:g}',QColor('#73869a'),(-6,10))
        for y in np.arange(math.ceil(ymin/step)*step,ymax+1e-8,step):
            self._polyline(painter, [[xmin, y, 0], [xmax, y, 0]], QColor("#d8e1ec"))
            if self.view_plane=='xy':self._label(painter,self.project([xmin,y,0]),f'{y:g}',QColor('#73869a'),(-35,-4))
        if self.view_plane=='xz':
            hstep=self._grid_step(hmax)
            for h in np.arange(0,hmax+1e-8,hstep):
                self._polyline(painter,[[xmin,ymin,h],[xmax,ymin,h]],QColor('#d8e1ec'))
                self._label(painter,self.project([xmin,ymin,h]),f'{h:g}',QColor('#73869a'),(-35,-4))
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
            if np.linalg.norm(np.array([self.project(end).x()-self.project(origin).x(),self.project(end).y()-self.project(origin).y()]))<3:continue
            self._polyline(painter, [origin, end], QColor(color), 1.8)
            self._label(painter, self.project(end), text, QColor(color), (6, -3))
        if self.view_plane=='3d':self._label(painter, self.project(origin),
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

    def _drone(self, painter, arm_pixels=None, show_label=True, ink_color=None, body_rotation=None):
        ink = INK if ink_color is None else QColor(ink_color)
        position = _vec(self.snapshot.get("position"))
        yaw = float(self.snapshot.get("yaw", 0.0) or 0.0)
        nose = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        if body_rotation is not None:
            rotation = np.asarray(body_rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise ValueError('Invalid display body rotation')
            nose, side = rotation[:, 0], rotation[:, 1]
        # Keep the vehicle legible at wide bounds; only its icon is enlarged.
        arm = max(0.65, 9.0 / max(self.scale, 0.1)) if arm_pixels is None else float(arm_pixels)/max(self.scale,.01)
        rotors = [position + arm * (i * nose + j * side)
                  for i, j in ((1, 1), (-1, -1), (1, -1), (-1, 1))]
        self._polyline(painter, [rotors[0], rotors[1]], ink, 3.2)
        self._polyline(painter, [rotors[2], rotors[3]], ink, 3.2)
        for rotor in rotors:
            projected = self.project(rotor)
            painter.setPen(self._pen(ink, 1.7))
            painter.setBrush(QColor(255, 255, 255, 230))
            if body_rotation is None:
                painter.drawEllipse(projected, 5.5, 3.5)
            else:
                angles = np.linspace(0, 2 * math.pi, 25)
                ring = rotor + .44 * arm * (np.cos(angles)[:, None] * nose + np.sin(angles)[:, None] * side)
                painter.drawPolygon(self._polygon(self.project_points(ring)))
        projected = self.project(position)
        painter.setPen(self._pen(QColor("#ffffff"), 1.4))
        painter.setBrush(ink)
        if body_rotation is None:
            painter.drawEllipse(projected, 5.6, 5.6)
        else:
            body = [position + .32 * arm * (i * nose + j * side)
                    for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            painter.drawPolygon(self._polygon(self.project_points(body)))
        end = position + nose * arm * 3.2
        self._polyline(painter, [position, end], QColor("#ec9840"), 2.5)
        self._polyline(painter, [end - nose * arm * 0.7 + side * arm * 0.4,
                                end, end - nose * arm * 0.7 - side * arm * 0.4],
                       QColor("#ec9840"), 2.5)
        if show_label:self._label(painter, projected, "无人机", ink, (-53, -22), True)

    def _legend(self, painter):
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 230))
        rect = QRectF(16, self.height() - 58, min(510, self.width() - 32), 49)
        painter.drawRoundedRect(rect, 9, 9)
        painter.setFont(QFont("Microsoft YaHei", 9))
        entries = [(RED, "当前目标"), (BLUE, "下一目标"),
                   (GREEN, "参考轨迹"), (ACTUAL_TRAIL, "实际轨迹")]
        x = 38
        for color, text in entries:
            painter.setPen(self._pen(color, 2.5)); painter.drawLine(x, self.height() - 42,
                                                                  x + 18, self.height() - 42)
            painter.setPen(QColor("#586b82"))
            painter.drawText(QPointF(x + 25, self.height() - 38), text)
            x += min(119,(self.width()-65)/4)
        painter.setPen(QColor("#8493a6"))
        painter.drawText(QPointF(30, self.height() - 18),
                         "左键旋转   ·   右键平移   ·   滚轮缩放   ·   双击复位")

    def _ruler(self,painter):
        raw=65/max(self.scale,.01);base=10**math.floor(math.log10(raw))
        metres=max(v*base for v in (1,2,5,10) if v*base<=raw)
        length=metres*self.scale
        painter.setPen(self._pen(INK,1.6))
        y=self.height()-75
        painter.drawLine(QPointF(30,y),QPointF(30+length,y))
        for x in (30,30+length):painter.drawLine(QPointF(x,y-4),QPointF(x,y+4))
        painter.setFont(self._label_font)
        suffix='等比例 XY' if self.view_plane=='xy' else '等比例 X/高度' if self.view_plane=='xz' else '正交投影，轴有视角缩短'
        painter.drawText(QPointF(30,y-9),f'{metres:g} m · {suffix}')

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
        painter.drawText(QRectF(25, 15, self.width() - 50, 27), "任务取景（安全边界见参数）" if 'preparation' in self.config else "三维飞行场地")
        painter.setFont(QFont("Microsoft YaHei", 9))
        painter.setPen(QColor("#7a8ca3"))
        painter.drawText(QRectF(self.width() - 200, 25, 175, 24), Qt.AlignRight,
                         "坐标 / 米 · 高度向上")
        self._draw_box(painter)
        self._polyline(painter, self.snapshot.get("planned_path", []), GREEN, 2.2)
        self._polyline(painter, self.snapshot.get("trail", []),
                       QColor('#ffffff'), 5., max_points=self.MAX_DISPLAY_TRAIL)
        self._polyline(painter, self.snapshot.get("trail", []),
                       ACTUAL_TRAIL, 3.1, max_points=self.MAX_DISPLAY_TRAIL)
        preparation=self.snapshot.get('preparation_point',self.config.get('preparation',{}).get('task',{}).get('start'))
        if preparation is not None:
            point=self.project(preparation);painter.setPen(self._pen(QColor('#dd9221'),2));painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRectF(point.x()-6,point.y()-6,12,12));self._label(painter,point,'准备点',QColor('#a46b13'),(8,0),True)
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
        self._ruler(painter)
        painter.end()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._mouse = event.pos(); self._button = event.button()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._mouse is not None:
            delta = event.pos() - self._mouse
            if self._button == Qt.LeftButton:
                self.view_plane='3d'
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
