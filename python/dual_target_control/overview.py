"""Actual AirSim external-camera images, on an independent RPC worker.

Only visualization APIs are used here. This module neither owns nor changes the
flight lifecycle, and it never shares the control thread's msgpack client.
Scene pixels come from UE; no schematic is substituted for a failed capture.
"""
from __future__ import annotations

from copy import deepcopy
import math
import threading
import time

import numpy as np

from .core import ControlConfig


class _Stopped(Exception):
    pass


def decode_scene_frame(response):
    """Decode this workspace's uncompressed AirSim Scene buffer to top-down RGB.

    The local RenderRequest.cpp serializes FColor.B, G, R in that order, despite
    the API documentation calling the three channels RGB. ReadSurfaceData's row
    order is retained; unlike old API examples, this build needs no vertical flip.
    """
    if getattr(response, 'message', ''):
        raise RuntimeError('AirSim 相机返回错误：' + str(response.message))
    width, height = int(response.width), int(response.height)
    if not 0 < width <= 4096 or not 0 < height <= 4096:
        raise ValueError('AirSim 返回了无效的地图图像尺寸。')
    buffer = bytes(response.image_data_uint8)
    if len(buffer) != width * height * 3:
        raise ValueError('AirSim 地图图像不是完整的三通道未压缩数据。')
    return np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)[..., ::-1].copy()


def _points(values, limit, cfg=None):
    """Reject malformed geometry and bound the size of every debug RPC."""
    if values is None:
        return []
    result = np.asarray(values, dtype=float)
    if result.size == 0:
        return []
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError('地图标记包含无效的三维坐标。')
    if cfg is not None and (np.any(result < cfg.lower - 1e-6) or
                            np.any(result > cfg.upper + 1e-6)):
        raise ValueError('目标或规划标记位于实验场地范围外。')
    if len(result) > limit:
        result = result[np.linspace(0, len(result) - 1, limit, dtype=int)]
    return result.tolist()


def marker_geometry(snapshot, cfg):
    """Use the same height-up runtime snapshot as the Qt 3D view."""
    targets = _points(snapshot.get('targets', []), 3, cfg)
    if len(targets) not in (0, 2):
        raise ValueError('地图标记需要当前目标和下一目标。')
    return dict(targets=targets,
                planned=_points(snapshot.get('planned_path', []), 300, cfg),
                trail=_points(snapshot.get('trail', []), 300))


def field_edges(cfg):
    corners = [[x, y, h] for h in (cfg.height_min, cfg.height_max)
               for y in (cfg.y_min, cfg.y_max) for x in (cfg.x_min, cfg.x_max)]
    pairs = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7),
             (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    return [corners[index] for pair in pairs for index in pair]


def project_overlays(snapshot, camera_pose, width, height, ortho_width):
    """Project telemetry into camera pixels, without altering captured pixels.

    AirSim camera orientation maps local +X forward/+Y right/+Z down to world
    NED. Orthographic capture uses horizontal width in metres and square pixels.
    The caller clips drawing to the image; off-image points remain meaningful.
    """
    cfg = ControlConfig.from_dict(snapshot.get('config', {}).get('control', {}))
    geometry = marker_geometry(snapshot, cfg)
    position = np.asarray(snapshot.get('position', []), dtype=float)
    yaw = float(snapshot.get('yaw', float('nan')))
    if position.shape != (3,) or not np.isfinite(position).all() or not math.isfinite(yaw):
        raise ValueError('无人机位置或朝向无法投影。')
    origin = np.array([camera_pose.position.x_val, camera_pose.position.y_val,
                       camera_pose.position.z_val], dtype=float)
    orientation = camera_pose.orientation
    quaternion = np.array([orientation.x_val, orientation.y_val,
                           orientation.z_val, orientation.w_val], dtype=float)
    norm = np.linalg.norm(quaternion)
    if (not np.isfinite(origin).all() or not np.isfinite(quaternion).all() or
            not math.isfinite(norm) or norm < 1e-12):
        raise ValueError('AirSim 地图相机位姿无效。')
    if width <= 0 or height <= 0 or not math.isfinite(ortho_width) or ortho_width <= 0:
        raise ValueError('地图相机投影尺寸无效。')
    x, y, z, w = quaternion / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]])

    def pixels(rows):
        if not rows:
            return []
        world = np.asarray(rows, dtype=float).copy()
        world[:, 2] *= -1  # Runtime height-up -> AirSim NED.
        local = (world - origin) @ rotation
        projected = np.column_stack((width / 2. + local[:, 1] * width / ortho_width,
                                     height / 2. + local[:, 2] * width / ortho_width))
        if not np.isfinite(projected).all():
            raise ValueError('地图投影返回非有限像素坐标。')
        return projected.tolist()

    nose = position + np.array([3. * math.cos(yaw), 3. * math.sin(yaw), 0.])
    result = {name: pixels(rows) for name, rows in geometry.items()}
    # Metadata uses the same names the UI uses for the two trajectory layers.
    result['drone'] = pixels([position.tolist()])[0]
    result['nose'] = pixels([nose.tolist()])[0]
    return result


class OverviewStream:
    """4 Hz overview stream; UI calls start/stop and reads published snapshots."""
    def __init__(self, config, runtime):
        self.config = deepcopy(config)
        self.runtime = runtime
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._snap = dict(status='IDLE', message='打开地图后连接 AirSim 外部俯视相机。',
                          frame=None, frames=0, frame_ms=0., rpc_ms=0., overlays={})
        options = self.config.get('overview', {})
        self._debug_markers = options.get('ue_debug_markers', False)
        if not isinstance(self._debug_markers, bool):
            raise ValueError('UE 调试标记开关必须为布尔值。')
        rate = float(options.get('fps', 4))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('地图刷新频率必须为正数。')
        self._interval = 1. / min(rate, 4.)
        self._camera = str(options.get('camera', 'control_overview'))
        if not self._camera:
            raise ValueError('地图相机名称不能为空。')
        self._ortho_width = float(options.get('map_width_m', 125.))
        if not math.isfinite(self._ortho_width) or self._ortho_width <= 0:
            raise ValueError('地图相机宽度必须为正数。')
        duration = float(options.get('marker_lifetime_s', .6))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('地图标记持续时间必须为正数。')
        self._duration = min(1., max(.6, duration))

    def _publish(self, **values):
        with self._lock:
            self._snap.update(values)

    def snapshot(self):
        with self._lock:
            result = dict(self._snap)
            # Qt can retain a snapshot independently of the next publication.
            if result['frame'] is not None:
                result['frame'] = result['frame'].copy()
            result['overlays'] = deepcopy(result['overlays'])
            return result

    def start(self):
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            state = self.runtime.snapshot()
            if state.get('mode', getattr(self.runtime, 'mode', None)) != 'airsim':
                self._publish(status='UNAVAILABLE', frame=None, overlays={},
                              message='离线演示没有 UE 地图画面，请切换到 AirSim 模式。')
                return False
            if not state.get('connected', False):
                self._publish(status='WAITING', frame=None, overlays={},
                              message='请先启动并连接 AirSim，再打开完整地图。')
                return False
            self._stop.clear()
            self._publish(status='CONNECTING', frame=None, overlays={},
                          frames=0, frame_ms=0., rpc_ms=0.,
                          message='正在连接独立的 AirSim 俯视相机。')
            self._thread = threading.Thread(target=self._work, name='dual-target-map', daemon=True)
            self._thread.start()
            return True

    def stop(self):
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(2.5)
            if thread is None or not thread.is_alive():
                self._publish(status='STOPPED', frame=None, overlays={},
                              message='完整地图画面已关闭。')
            else:
                self._publish(status='STOPPING', message='正在结束地图相机请求。')

    def _call(self, function, *args, **kwargs):
        if self._stop.is_set():
            raise _Stopped()
        return function(*args, **kwargs)

    def _draw(self, client, airsim, state):
        cfg = ControlConfig.from_dict(state.get('config', self.config).get('control', {}))
        geometry = marker_geometry(state, cfg)
        ned = lambda rows: [airsim.Vector3r(float(x), float(y), -float(h)) for x, y, h in rows]
        warnings = []
        targets = geometry['targets']
        # Real movable UE meshes are captured even when a SceneCapture component
        # omits the engine's debug lines. Empty queues hide the two unused meshes.
        for index, name in enumerate(('DC_TargetRed', 'DC_TargetBlue')):
            point = ned([targets[index]])[0] if targets else airsim.Vector3r(0., 0., 1000.)
            if not self._call(client.simSetObjectPose, name,
                              airsim.Pose(point, airsim.Quaternionr()), teleport=True):
                warnings.append(name + ' 实体未同步')
        # SceneCapture omits UE debug primitives in this build. The Qt overlay
        # uses telemetry projection instead, so avoid five wasted marker RPCs.
        # An explicit option retains these primitives for a main UE viewport.
        if not self._debug_markers:
            return '；'.join(warnings)
        shared = dict(duration=self._duration, is_persistent=False)
        for index, color in enumerate(([1., .05, .05, 1.], [.1, .3, 1., 1.])):
            if targets:
                self._call(client.simPlotPoints, ned([targets[index]]),
                           color_rgba=color, size=10., **shared)
        self._call(client.simPlotLineList, ned(field_edges(cfg)),
                   color_rgba=[1., .65, .15, 1.], thickness=2., **shared)
        for key, color in (('planned', [.12, .9, .3, 1.]),
                           ('trail', [.65, .65, .65, 1.])):
            if len(geometry[key]) >= 2:
                self._call(client.simPlotLineStrip, ned(geometry[key]),
                           color_rgba=color, thickness=3., **shared)
        return '；'.join(warnings)

    def _work(self):
        client = None
        try:
            import airsim
            options = self.config.get('airsim', {})
            timeout = float(options.get('rpc_timeout', 3.))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError('地图 RPC 超时设置必须为正数。')
            # Only one bounded call can still be running after stop is requested.
            client = airsim.MultirotorClient(ip=options.get('host', '127.0.0.1'),
                port=int(options.get('port', 41460)), timeout_value=min(timeout, 1.))
            if not self._call(client.ping):
                raise RuntimeError('AirSim 地图连接失败。')
            camera_pose = None
            camera_warning = ''
            try:
                camera_pose = self._call(client.simGetCameraInfo, self._camera,
                    vehicle_name=options.get('vehicle', 'DroneControl'), external=True).pose
            except _Stopped:
                raise
            except Exception as exc:
                camera_warning = '地图投影信息不可用：' + str(exc)
            while not self._stop.is_set():
                begin = time.perf_counter()
                state = self.runtime.snapshot()
                if not state.get('connected', False):
                    self._publish(status='WAITING', frame=None, overlays={},
                                  message='AirSim 连接已断开。')
                    self._stop.wait(self._interval)
                    continue
                marker_warning = ''
                try:
                    marker_warning = self._draw(client, airsim, state)
                except _Stopped:
                    raise
                except Exception as exc:
                    marker_warning = '地图标记未更新：' + str(exc)
                capture_begin = time.perf_counter()
                responses = self._call(client.simGetImages,
                    [airsim.ImageRequest(self._camera, airsim.ImageType.Scene,
                                         pixels_as_float=False, compress=False)],
                    vehicle_name=options.get('vehicle', 'DroneControl'), external=True)
                capture_ms = (time.perf_counter() - capture_begin) * 1000.
                if len(responses) != 1:
                    raise RuntimeError('AirSim 没有返回完整地图相机图像。')
                frame = decode_scene_frame(responses[0])
                if self._stop.is_set():
                    break
                overlays = {}
                projection_warning = camera_warning
                if camera_pose is not None:
                    try:
                        overlays = project_overlays(state, camera_pose, frame.shape[1],
                                                    frame.shape[0], self._ortho_width)
                    except Exception as exc:
                        projection_warning = '地图坐标投影未更新：' + str(exc)
                warnings = '；'.join(value for value in (marker_warning, projection_warning) if value)
                elapsed = (time.perf_counter() - begin) * 1000.
                self._publish(status='STREAMING', frame=frame,
                    frames=self._snap['frames'] + 1, frame_ms=elapsed, rpc_ms=capture_ms,
                    overlays=overlays,
                    message=('UE 实时俯视画面；轨迹和朝向为遥测投影标注。' +
                             (' ' + warnings if warnings else '')))
                self._stop.wait(max(0., self._interval - elapsed / 1000.))
        except _Stopped:
            pass
        except Exception as exc:
            self._publish(status='ERROR', frame=None, overlays={},
                          message='完整地图连接失败：' + str(exc))
        finally:
            if client is not None:
                try:
                    client.client.close()
                except Exception:
                    pass
            if self._stop.is_set():
                self._publish(status='STOPPED', frame=None, overlays={},
                              message='完整地图画面已关闭。')
