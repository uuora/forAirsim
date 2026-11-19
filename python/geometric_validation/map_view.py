"""Orthographic Qt map of the planned and measured route in metres."""
from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPolygonF
from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from .route_scene import RouteScene


class RouteMapView(QWidget):
    plane_changed = pyqtSignal(str)
    GROUND_REFERENCE_M = -1.0

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setMinimumSize(430, 275)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.snapshot = {}
        self.height_scale = 1.0
        self.include_takeoff = False
        self._mouse = None
        self._button = None
        self._camera_signature = None
        self._task_revision = 0
        self.set_task(dict(config=config))
        self.reset_view()
        self.scene = RouteScene(self)

    def set_task(self, snapshot):
        # Freeze the framing geometry at task selection. Telemetry only paints.
        task = snapshot.get("config", {}).get("preparation", {}).get("task")
        waypoints = self.display_points([task["start"], *task["points"]] if task else [])
        path = self.display_points(snapshot.get("planned_path"))
        points = np.concatenate((waypoints, path))
        self._task_points = points if len(points) else np.zeros((1, 3))
        self._task_revision += 1
        self.set_snapshot(snapshot)

    def set_snapshot(self, snapshot):
        self.snapshot = snapshot
        self.update()

    def reset_view(self):
        self.set_plane("3d")

    def set_plane(self, plane):
        self.view_plane = plane
        if plane == "3d":
            self.azimuth = math.radians(-50)
            self.elevation = math.radians(38)
        else:
            self.azimuth = -math.pi / 2
            self.elevation = math.pi / 2 if plane == "xy" else 0.
        self.zoom = 1.
        self.pan = np.zeros(2)
        self.plane_changed.emit(plane)
        self.update()

    def set_height_scale(self, scale):
        self.height_scale = float(scale)
        self.update()

    def set_include_takeoff(self, enabled):
        self.include_takeoff = enabled
        self.update()

    @property
    def display_height_scale(self):
        return self.height_scale if self.view_plane == "3d" else 1.

    def plot_rect(self):
        return QRectF(3., 46., self.width() - 6., self.height() - 81.)

    def _display_bounds(self):
        low = self._task_points.min(axis=0)
        high = self._task_points.max(axis=0)
        span = high - low
        pad_xy = float(np.clip(.12 * max(span[:2]), .45, 1.6))
        half_xy = np.maximum(span[:2] / 2 + pad_xy, 1.05)
        mid_xy = (low[:2] + high[:2]) / 2
        x_min, y_min = mid_xy - half_xy
        x_max, y_max = mid_xy + half_xy
        floor = math.floor(2 * (low[2] - .55)) / 2
        if self.include_takeoff:
            floor = min(floor, self.GROUND_REFERENCE_M)
        ceiling = max(high[2] + .45, floor + 1.)
        return x_min, x_max, y_min, y_max, floor, ceiling

    def _camera(self):
        height_scale = self.display_height_scale
        signature = (self._task_revision, self.width(), self.height(), self.view_plane, self.azimuth,
                     self.elevation, height_scale, self.include_takeoff, self.zoom, *self.pan)
        if signature == self._camera_signature:
            return
        self._camera_signature = signature
        az, el = self.azimuth, self.elevation
        self.right = np.array([-math.sin(az), math.cos(az), 0.])
        self.up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)])
        basis = np.column_stack((self.right, -self.up))
        basis[2] *= height_scale

        x_min, x_max, y_min, y_max, floor, ceiling = self._display_bounds()
        base = np.array([[x, y, floor] for x in (x_min, x_max) for y in (y_min, y_max)])
        if self.view_plane == "xz":
            frame = np.vstack((base, [[x_min, y_min, ceiling], [x_max, y_min, ceiling]]))
        else:
            frame = np.vstack((base, base - [0., 0., .16], [[x_min, y_min, ceiling]]))
        projected = np.vstack((frame, self._task_points)) @ basis
        lower, upper = projected.min(axis=0), projected.max(axis=0)
        span = np.maximum(upper - lower, .1)
        area = self.plot_rect().adjusted(39., 24., -34., -27.)
        self.scale = min(area.width() / span[0], area.height() / span[1]) * self.zoom
        self.projection_matrix = basis * self.scale
        center = np.array([area.center().x(), area.center().y()]) + self.pan
        self.projection_offset = center - (lower + upper) * .5 * self.scale

    @staticmethod
    def _grid_step(span):
        raw = max(span / 7., .01)
        base = 10 ** math.floor(math.log10(raw))
        return next(value * base for value in (1, 2, 5, 10) if value * base >= raw)

    @staticmethod
    def display_points(points, max_points=None):
        if points is None or len(points) == 0:
            return np.empty((0, 3))
        if max_points is not None and len(points) > max_points:
            indices = np.linspace(0, len(points) - 1, max_points, dtype=int)
            points = [points[i] for i in indices]
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            return np.empty((0, 3))
        return values

    def project_points(self, points):
        return points @ self.projection_matrix + self.projection_offset

    def project(self, point):
        return QPointF(*self.project_points(np.asarray(point, dtype=float)))

    @staticmethod
    def _polygon(projected):
        return QPolygonF([QPointF(x, y) for x, y in projected])

    @staticmethod
    def _pen(color, width=1., style=Qt.SolidLine):
        pen = QPen(color, width, style)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        return pen

    def _polyline(self, painter, points, color, width=1., style=Qt.SolidLine):
        values = self.display_points(points)
        if len(values) < 2:
            return
        painter.setPen(self._pen(color, width, style))
        painter.setBrush(Qt.NoBrush)
        painter.drawPolyline(self._polygon(self.project_points(values)))

    def draw_drone(self, painter, ink, body_rotation):
        position = np.asarray(self.snapshot["position"], dtype=float)
        yaw = float(self.snapshot.get("yaw", 0.))
        nose = np.array([math.cos(yaw), math.sin(yaw), 0.])
        side = np.array([-math.sin(yaw), math.cos(yaw), 0.])
        if body_rotation is not None:
            rotation = np.asarray(body_rotation, dtype=float)
            if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
                raise ValueError("Invalid display body rotation")
            nose, side = rotation[:, 0], rotation[:, 1]
        # Retain the established quadrotor schematic and orange nose arrow.
        arm = 12. / max(self.scale, .01)
        corners = ((1, 1), (-1, -1), (1, -1), (-1, 1))
        rotors = [position + arm * (i * nose + j * side) for i, j in corners]
        self._polyline(painter, rotors[:2], ink, 3.2)
        self._polyline(painter, rotors[2:], ink, 3.2)
        angles = np.linspace(0., 2 * math.pi, 25)
        ring = .44 * arm * (np.cos(angles)[:, None] * nose + np.sin(angles)[:, None] * side)
        for rotor in rotors:
            painter.setPen(self._pen(ink, 1.7))
            painter.setBrush(QColor(255, 255, 255, 230))
            if body_rotation is None:
                painter.drawEllipse(self.project(rotor), 5.5, 3.5)
            else:
                painter.drawPolygon(self._polygon(self.project_points(rotor + ring)))
        painter.setPen(self._pen(QColor("#ffffff"), 1.4))
        painter.setBrush(ink)
        if body_rotation is None:
            painter.drawEllipse(self.project(position), 5.6, 5.6)
        else:
            body = np.array([position + .32 * arm * (i * nose + j * side)
                             for i, j in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
            painter.drawPolygon(self._polygon(self.project_points(body)))
        end = position + nose * arm * 3.2
        arrow = QColor("#ec9840")
        self._polyline(painter, [position, end], arrow, 2.5)
        self._polyline(painter, [end - nose * arm * .7 + side * arm * .4,
                                end, end - nose * arm * .7 - side * arm * .4], arrow, 2.5)

    def paintEvent(self, event):
        self.scene.paint()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.RightButton):
            self._mouse, self._button = event.pos(), event.button()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._mouse is not None:
            delta = event.pos() - self._mouse
            if self._button == Qt.LeftButton and self.view_plane == "3d":
                self.azimuth += math.radians(.5 * delta.x())
                self.elevation = float(np.clip(self.elevation + math.radians(.4 * delta.y()),
                                               math.radians(5), math.radians(85)))
            else:
                self.pan += [delta.x(), delta.y()]
            self._mouse = event.pos()
            self.update()
            return
        self._camera()
        position = self.snapshot.get("position")
        if position is not None and self.snapshot.get("connected"):
            point = self.project(position)
            if math.hypot(point.x() - event.x(), point.y() - event.y()) < 26:
                self.setToolTip("四旋翼图标按实测姿态绘制，橙色箭头为机头。图标为放大示意。")
                return
        for marker in self.snapshot.get("route_markers", ()):
            point = self.project(marker["position"])
            if math.hypot(point.x() - event.x(), point.y() - event.y()) < 16:
                coordinates = ", ".join(f"{value:g}" for value in marker["position"])
                self.setToolTip(f"{marker['id']}  ({coordinates}) m")
                return
        self.setToolTip("三维左拖旋转，右拖平移，滚轮缩放，双击重置视角。坐标网格为显示参考面。")

    def mouseReleaseEvent(self, event):
        self._mouse = self._button = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        self.set_plane(self.view_plane)

    def wheelEvent(self, event):
        self.zoom = float(np.clip(self.zoom * math.exp(event.angleDelta().y() / 1200), .4, 5.))
        self.update()
        event.accept()


class RouteMapWindow(QWidget):
    dock_requested = pyqtSignal()

    def __init__(self):
        super().__init__(None, Qt.Window)
        self.setWindowTitle("三维轨迹地图")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

    def closeEvent(self, event):
        self.dock_requested.emit()
        event.accept()
