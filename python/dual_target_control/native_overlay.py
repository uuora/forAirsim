"""Draw geometric validation telemetry in the native Unreal viewport."""
import math

import numpy as np

from .core import ControlConfig


def _points(values, limit, bounds=None):
    if values is None:
        return []
    points = np.asarray(values, dtype=float)
    if not points.size:
        return []
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("地图标记包含无效的三维坐标。")
    if bounds is not None and (np.any(points < bounds.lower - 1e-6)
                               or np.any(points > bounds.upper + 1e-6)):
        raise ValueError("目标或规划标记位于实验场地范围外。")
    if len(points) > limit:
        points = points[np.linspace(0, len(points) - 1, limit, dtype=int)]
    return points.tolist()


def field_edges(bounds):
    corners = [[x, y, h] for h in (bounds.height_min, bounds.height_max)
               for y in (bounds.y_min, bounds.y_max) for x in (bounds.x_min, bounds.x_max)]
    pairs = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7),
             (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    return [corners[index] for pair in pairs for index in pair]


def glyph_lines(marker):
    point = np.asarray(marker["position"], dtype=float)
    if marker["kind"] == "start":
        radius = .24
        corners = point + [[-radius, -radius, 0], [radius, -radius, 0],
                           [radius, radius, 0], [-radius, radius, 0]]
        return corners[[0, 1, 1, 2, 2, 3, 3, 0]].tolist()
    top = point + [0, 0, .7]
    tip = top + [.5, 0, -.13]
    bottom = top + [0, 0, -.29]
    return [point, top, top, tip, tip, bottom, bottom, top]


class NativeOverlay:
    def __init__(self, config):
        duration = float(config.get("overview", {}).get("marker_lifetime_s", .6))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("地图标记持续时间必须为正数。")
        self._duration = min(1., max(.6, duration))

    def draw(self, client, airsim, state):
        bounds = ControlConfig.from_dict(state.get("config", {}).get("control", {}))
        targets = _points(state.get("targets"), 3, bounds)
        if len(targets) > 2:
            raise ValueError("地图标记最多接受当前目标和下一目标。")
        planned = _points(state.get("planned_path"), 300, bounds)
        trail = _points(state.get("trail"), 300)
        origin = np.asarray(state.get("marker_world_origin_ned", [0., 0., 0.]))

        def ned(rows):
            return [airsim.Vector3r(float(x + origin[0]), float(y + origin[1]), float(-h + origin[2]))
                    for x, y, h in rows]

        warnings = []
        for index, name in enumerate(("DC_TargetRed", "DC_TargetBlue")):
            point = ned([targets[index]])[0] if index < len(targets) else airsim.Vector3r(0., 0., 1000.)
            if not client.simSetObjectPose(name, airsim.Pose(point, airsim.Quaternionr()), teleport=True):
                warnings.append(name + " 实体未同步")
        shared = dict(duration=self._duration, is_persistent=False)
        for target, color in zip(targets, ([1., .05, .05, 1.], [.1, .3, 1., 1.])):
            client.simPlotPoints(ned([target]), color_rgba=color, size=10., **shared)
        client.simPlotLineList(ned(field_edges(bounds)), color_rgba=[1., .65, .15, 1.],
                               thickness=2., **shared)
        for points, color, width in ((planned, [.12, .9, .3, 1.], 2.5), (trail, [1., .05, .85, 1.], 6.)):
            if len(points) >= 2:
                client.simPlotLineStrip(ned(points), color_rgba=color, thickness=width, **shared)

        visited, labels, positions = [], [], []
        for marker in state.get("route_markers") or ():
            point = marker["position"]
            if marker["kind"] == "via":
                if marker["state"] != "visited":
                    continue
                visited.append(point)
            else:
                color = [.02, .65, .65, 1.] if marker["kind"] == "start" else [1., .55, .05, 1.]
                client.simPlotLineList(ned(glyph_lines(marker)), color_rgba=color, thickness=4., **shared)
            labels.append(marker["id"])
            positions.append([point[0], point[1], point[2] + .85])
        if visited:
            client.simPlotPoints(ned(visited), color_rgba=[.1, .45, 1., 1.], size=9., **shared)
        if labels:
            client.simPlotStrings(labels, ned(positions), scale=1.2,
                                  color_rgba=[.05, .12, .18, 1.], duration=self._duration)
        return "；".join(warnings)
