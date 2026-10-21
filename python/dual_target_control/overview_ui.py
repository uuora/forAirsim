"""Qt viewer for the real AirSim external camera, separate from flight control."""
from __future__ import annotations

import math
import threading
from typing import Callable

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt5.QtWidgets import (
    QHBoxLayout, QLabel, QMainWindow, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


class OverviewCanvas(QWidget):
    """Own a detached RGB image and fit it without stretching the map."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.overlays = {}
        self.setMinimumSize(520, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, frame):
        values = np.asarray(frame)
        if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
            raise ValueError("地图画面必须是 RGB 图像。")
        height, width = values.shape[:2]
        if height == 0 or width == 0:
            raise ValueError("地图画面为空。")
        values = np.ascontiguousarray(values)
        self.image = QImage(values.data, width, height, values.strides[0],
                            QImage.Format_RGB888).copy()
        self.update()

    def set_overlays(self, overlays):
        """Store telemetry coordinates without modifying captured image pixels."""
        self.overlays = {}
        if isinstance(overlays, dict):
            for key in ("targets", "planned", "trail", "drone", "nose"):
                try:
                    points = np.asarray(overlays.get(key, []), dtype=float)
                    if key in ("drone", "nose"):
                        valid = points.shape == (2,)
                    else:
                        valid = points.ndim == 2 and points.shape[1] == 2
                    if valid and np.isfinite(points).all():
                        self.overlays[key] = points.copy()
                except (TypeError, ValueError):
                    pass
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        return QRectF((self.width() - width) / 2, (self.height() - height) / 2, width, height)

    def project_overlay_points(self, points):
        rect = self.image_rect()
        if rect.isEmpty():
            return np.empty((0, 2))
        return (np.asarray(points) * (rect.width() / self.image.width())
                + np.array([rect.left(), rect.top()]))

    def _draw_overlays(self, painter, rect):
        painter.save()
        painter.setClipRect(rect)
        painter.setRenderHint(QPainter.Antialiasing)
        for key, color in (("trail", "#77889b"), ("planned", "#1c9e75")):
            points = self.overlays.get(key)
            if points is not None and len(points) > 1:
                projected = self.project_overlay_points(points)
                pen = QPen(QColor(color), 2.3)
                pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen); painter.setBrush(Qt.NoBrush)
                painter.drawPolyline(QPolygonF([QPointF(*p) for p in projected]))
        targets = self.overlays.get("targets")
        if targets is not None:
            for index, p in enumerate(self.project_overlay_points(targets[:2])):
                point = QPointF(*p)
                color = QColor("#e6555f" if index == 0 else "#3486ea")
                painter.setPen(QPen(QColor("white"), 5)); painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(point, 7, 7)
                painter.setPen(QPen(color, 2)); painter.drawEllipse(point, 7, 7)
                text = "01 当前目标" if index == 0 else "02 下一目标"
                self._label(painter, rect, point, text, color)
        drone = self.overlays.get("drone")
        if drone is not None:
            position = self.project_overlay_points(drone)
            point = QPointF(*position)
            painter.setPen(QPen(QColor("white"), 4)); painter.setBrush(QColor("#23344e"))
            painter.drawEllipse(point, 5, 5)
            painter.setPen(QPen(QColor("#23344e"), 2)); painter.drawEllipse(point, 5, 5)
            nose = self.overlays.get("nose")
            if nose is not None:
                direction = self.project_overlay_points(nose) - position
                magnitude = float(np.linalg.norm(direction))
                if magnitude > .01:
                    direction /= magnitude
                    normal = np.array([-direction[1], direction[0]])
                    end = position + direction * 30
                    painter.setPen(QPen(QColor("#ec9840"), 2.5))
                    painter.drawLine(point, QPointF(*end))
                    arrow = np.array([end - direction * 8 + normal * 5, end,
                                      end - direction * 8 - normal * 5])
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPolyline(QPolygonF([QPointF(*p) for p in arrow]))
            self._label(painter, rect, point, "无人机", QColor("#23344e"), below=True)
        painter.restore()

    @staticmethod
    def _label(painter, bounds, point, text, color, below=False):
        width = painter.fontMetrics().horizontalAdvance(text) + 12
        left = max(bounds.left() + 3, min(point.x() + 12, bounds.right() - width - 3))
        top = point.y() + 13 if below else point.y() - 30
        top = max(bounds.top() + 3, min(top, bounds.bottom() - 24 - 3))
        label = QRectF(left, top, width, 24)
        painter.setPen(Qt.NoPen); painter.setBrush(QColor(255, 255, 255, 225))
        painter.drawRoundedRect(label, 4, 4)
        painter.setPen(color)
        painter.drawText(label.adjusted(6, 0, -6, 0), Qt.AlignVCenter, text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#182231"))
        if self.image.isNull():
            painter.setPen(QColor("#ccd6e3"))
            painter.drawText(self.rect(), Qt.AlignCenter, "正在连接完整地图画面…")
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            rect = self.image_rect()
            painter.drawImage(rect, self.image)
            self._draw_overlays(painter, rect)
        painter.end()


class OverviewWindow(QMainWindow):
    """Only request visual telemetry; closing this window does not affect flight."""

    closed = pyqtSignal()
    STATUS_LABELS = {
        "idle": "待连接", "starting": "正在连接", "connecting": "正在连接",
        "running": "实时画面", "connected": "已连接", "ready": "实时画面",
        "streaming": "实时画面", "waiting": "等待连接", "unavailable": "未提供画面",
        "stopping": "正在关闭",
        "error": "画面异常", "stopped": "画面已关闭", "closed": "画面已关闭",
    }

    def __init__(self, config: dict, runtime, stream_factory: Callable | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window)
        self.setWindowTitle("完整地图 · AirSim 实时总览")
        self.resize(1180, 820)
        self._closed = False
        self._last_frame_number = None
        self._stop_thread = None
        if stream_factory is None:
            from .overview import OverviewStream
            stream_factory = OverviewStream
        self.stream = stream_factory(config, runtime)
        self.setStyleSheet("QMainWindow { background: #f4f7fb; }"
                           "QLabel { color: #23344e; font-family: 'Microsoft YaHei'; }"
                           "QPushButton { padding: 7px 15px; }")
        container = QWidget(); self.setCentralWidget(container)
        layout = QVBoxLayout(container); layout.setContentsMargins(18, 16, 18, 14)
        header = QHBoxLayout()
        heading = QLabel("AirSim 实际地图画面 · 与控制窗口同一无人机")
        heading.setStyleSheet("font-size: 17px; font-weight: 600;")
        header.addWidget(heading, 1)
        close_button = QPushButton("关闭画面")
        close_button.clicked.connect(self.close); header.addWidget(close_button)
        layout.addLayout(header)
        self.canvas = OverviewCanvas(); layout.addWidget(self.canvas, 1)
        self.message = QLabel("正在连接总览相机…")
        self.message.setWordWrap(True); layout.addWidget(self.message)
        self.metrics = QLabel("画面 RPC  —   ·   每帧耗时  —   ·   已接收  0 帧")
        self.metrics.setStyleSheet("color: #73839a;"); layout.addWidget(self.metrics)
        note = QLabel("UE 实时俯视画面 · 轨迹与位置标注\n"
                      "红色：当前目标  ·  蓝色：下一目标  ·  绿色：规划轨迹  ·  灰色：实际轨迹")
        note.setStyleSheet("color: #73839a;"); layout.addWidget(note)
        self.timer = QTimer(self); self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        # start() only launches the stream worker; AirSim calls stay in that worker.
        self.stream.start()
        self.timer.start()

    @staticmethod
    def _milliseconds(value):
        try:
            number = float(value)
            return f"{number:.1f} ms" if math.isfinite(number) else "—"
        except (TypeError, ValueError):
            return "—"

    def refresh(self):
        if self._closed:
            return
        try:
            snapshot = self.stream.snapshot()
            frames = snapshot.get("frames", 0)
            frame = snapshot.get("frame")
            if frame is not None and frames != self._last_frame_number:
                self.canvas.set_frame(frame)
                self._last_frame_number = frames
            self.canvas.set_overlays(snapshot.get("overlays", {}))
            status = str(snapshot.get("status", "connecting")).lower()
            title = self.STATUS_LABELS.get(status, status)
            message = str(snapshot.get("message") or "")
            self.message.setText(f"{title} · {message}" if message else title)
            self.metrics.setText(
                f"画面 RPC  {self._milliseconds(snapshot.get('rpc_ms'))}   ·   "
                f"每帧耗时  {self._milliseconds(snapshot.get('frame_ms'))}   ·   "
                f"已接收  {frames} 帧")
        except Exception as error:
            self.message.setText(f"地图画面异常：{error}")

    def closeEvent(self, event):
        if not self._closed:
            self._closed = True
            self.timer.stop()
            # An image RPC can be in flight. Never wait for it in the GUI thread,
            # and never call the flight runtime when merely closing the preview.
            self._stop_thread = threading.Thread(target=self.stream.stop,
                                                 name="overview-window-close", daemon=True)
            self._stop_thread.start()
            self.closed.emit()
        event.accept()
