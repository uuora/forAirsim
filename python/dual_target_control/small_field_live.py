"""Real-time tracking acceptance for the small AirSim field."""
import json
from pathlib import Path
import time

import numpy as np

from .prepared_route_live import PreparedRouteRuntime
from .runtime import wrap_angle
from .simulator import ROOT
from .small_field import (BODY_PROTOCOL_REVISION, expected_dynamics, flight_metrics, inside_field,
                          reference_checks, target_distance, total_tilt_deg)


class SmallFieldRuntime(PreparedRouteRuntime):
    protocol_revision = BODY_PROTOCOL_REVISION
    _local_ground_origin_zup = np.zeros(3)
    additional_source_paths = PreparedRouteRuntime.additional_source_paths + [
        Path(__file__), Path(__file__).with_name("small_field.py"), ROOT / "configs/small_field.json",
        ROOT / "python/fast_panel.py", ROOT / "configs/dual_target_airsim_settings.json",
        ROOT / "scripts/unreal/build_drone_control_overview.py",
    ]

    @property
    def rules(self):
        return self.config["small_field"]

    def _connect(self):
        super()._connect()
        settings = json.loads(self._client.getSettingsString())
        vehicle = settings.get("Vehicles", {}).get(self.vehicle_name, {})
        if vehicle.get("VehicleType") != "SimpleFlight" or settings.get("Recording", {}).get("Enabled") is not False:
            raise RuntimeError("本实验要求 SimpleFlight 并关闭自动录制")
        if not expected_dynamics(self._dynamics):
            raise RuntimeError("AirSim 实际动力回执与小场地配置不符")

    def _calibrate_position_frame(self, state):
        floor = self._client.simGetObjectPose('DC_VisibleFloor').position
        thickness = self._client.simGetObjectScale('DC_VisibleFloor').z_val
        drone = self._client.simGetObjectPose(self.vehicle_name).position
        local = state.kinematics_estimated.position
        lower, upper = np.asarray(self.rules['bounds_zup_m'])
        ground_z = floor.z_val - thickness / 2
        # AirSim's global position is in metres, relative to the PlayerStart.
        ground_x, ground_y = (lower[:2] + upper[:2]) / 2
        self.ground_origin_global_ned = np.array([floor.x_val - ground_x, floor.y_val - ground_y, ground_z])
        local_to_global = np.array([drone.x_val - local.x_val, drone.y_val - local.y_val, drone.z_val - local.z_val])
        self._local_ground_origin_zup = (self.ground_origin_global_ned - local_to_global) * [1, 1, -1]
        if not np.isfinite(self._local_ground_origin_zup).all() or thickness <= 0:
            raise RuntimeError('无法读取小场地地面原点。')

    def _position_zup(self, ned_position):
        return super()._position_zup(ned_position) - self._local_ground_origin_zup

    def _start(self):
        if self._active or self._flight_owned:
            return
        if self.mode == 'airsim':
            if self._client is None:
                self._connect()
            state = self._client.getMultirotorState(vehicle_name=self.vehicle_name)
            p = self._position_zup(state.kinematics_estimated.position)
            if np.linalg.norm(p[:2] - self.rules['launch'][:2]) > self.rules['preparation']['position_m']:
                if (state.landed_state != self._airsim.LandedState.Landed
                        or self._client.isApiControlEnabled(vehicle_name=self.vehicle_name)
                        or self._process is None or self._process.poll() is not None):
                    raise RuntimeError('当前仿真无法在地面原点重新起飞。')
                self._publish('LAUNCHING', '已落地，正在重开场景以从原点起飞。', connected=False)
                if self._scene is not None:
                    if not self._scene.close(timeout=8):
                        raise RuntimeError('观察线程尚未关闭。')
                    self._scene = None
                self._process.terminate()
                self._process.wait(timeout=10)
                for client in (self._client, self._lifecycle):
                    client.client.close()
                self._client = self._lifecycle = self._process = None
                self._observer_initialized_pid = None
                super()._handle('simulator', None)
            self._wait_for_ground_start()
        super()._start()

    def _wait_for_ground_start(self):
        deadline = time.monotonic() + 5.
        limits = self.rules['preparation']
        consecutive = 0
        last_stamp = None
        while time.monotonic() < deadline and not self._stop_requested.is_set():
            p, v, _, _ = self._read_state()
            stamp = self._last_state.timestamp
            if stamp != last_stamp:
                collision = self._client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
                ready = (self._last_state.landed_state == self._airsim.LandedState.Landed
                         and collision.object_name == 'DC_CollisionFloor' and collision.normal.z_val < -.7
                         and np.linalg.norm(p - self._position_zup(collision.position)) < .3
                         and np.linalg.norm(p[:2] - self.rules['launch'][:2]) <= limits['position_m']
                         and np.linalg.norm(v) <= limits['speed_mps'])
                consecutive = consecutive + 1 if ready else 0
                if consecutive >= limits['samples']:
                    self._ground_start = dict(sim_timestamp_ns=int(stamp), position_zup_m=p.tolist(),
                                              velocity_zup_mps=v.tolist(), contact=collision.object_name)
                    return
                last_stamp = stamp
            time.sleep(1 / self.cfg.control_hz)
        raise RuntimeError('未确认无人机在地面原点静止。')

    def _begin_run(self):
        self.tracking_fault = None
        super()._begin_run()
        (self._run_dir / 'field_frame.json').write_text(json.dumps(dict(
            ground_origin_global_ned_m=self.ground_origin_global_ned.tolist(),
            ground_origin_local_zup_m=self._local_ground_origin_zup.tolist(),
            launch_zup_m=self.rules['launch'], reference_start_zup_m=self.rules['start'],
            ground_start=self._ground_start,
        ), indent=2), encoding='utf-8')

    def _takeoff_goal(self, height):
        x, y, _ = self.rules["start"]
        return np.array([x, y, height])

    def _decode_reference(self, plan, factor):
        curve = super()._decode_reference(plan, factor)
        if not reference_checks(curve)["passed"]:
            raise ValueError("冻结参考超出速度、加速度、边界或终速要求")
        return curve

    def _tracking_sample(self, p, v, yaw, stamp, observed):
        orientation = self._last_state.kinematics_estimated.orientation
        quaternion = [orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val]
        why = None
        if self.live_previous_stamp is not None and observed == 0:
            return None
        if observed < 0 or observed > self.rules["max_sample_gap_s"] + 1e-9:
            why = f"实测状态采样间隔异常 {observed:.6f} s"
        if not inside_field(p, self.rules):
            lower, upper = self.rules["bounds_zup_m"]
            why = f"实测位置 {p.tolist()} m 超出实验区 {lower} 至 {upper} m"
        if self.live_previous_stamp is None:
            limits = self.rules["preparation"]
            preparation_gap = (stamp - self.preparation_rows[-1]["sim_timestamp_ns"]) * 1e-9
            if (np.linalg.norm(p - self.rules["start"]) > limits["position_m"]
                    or np.linalg.norm(v) > limits["speed_mps"] or abs(wrap_angle(yaw)) > limits["yaw_rad"]):
                why = "参考起始样本未达到位置、速度和偏航准备容差"
            if not 0. <= preparation_gap <= self.rules["max_sample_gap_s"]:
                why = f"准备结束到参考起始的实测采样间隔异常 {preparation_gap:.6f} s"
                observed = preparation_gap
        try:
            tilt = total_tilt_deg(quaternion)
        except ValueError as exc:
            why = str(exc)
            quaternion = None
            tilt = None
        if tilt is not None and tilt > self.rules["max_tilt_deg"]:
            why = f"实测机体总倾角 {tilt:.3f}° 超限"
        if why:
            self.tracking_fault = dict(reason=why, sim_timestamp_ns=stamp, sample_gap_s=observed,
                                       position_zup_m=p.tolist(), velocity_zup_mps=v.tolist(), yaw_rad=yaw,
                                       orientation_ned_xyzw=quaternion, tilt_deg=tilt)
            (self._run_dir / "tracking_fault.json").write_text(
                json.dumps(self.tracking_fault, ensure_ascii=False, indent=2), encoding="utf-8")
            raise RuntimeError(why)
        return dict(orientation_ned_xyzw=quaternion, tilt_deg=tilt)

    def _advance_targets(self, t, p, v):
        points = self.live_curve.spec["waypoints"]
        if self.route_index >= len(points) - 1:
            return
        orientation = self.live_states[-1]["orientation_ned_xyzw"]
        distance = target_distance(p, points[self.route_index], self.rules, orientation)
        hit = distance <= self.rules["waypoint_radius_m"]
        if self.live_curve.spec["policy"] == "pass":
            hit = hit and np.linalg.norm(v) >= self.rules["minimum_pass_speed_mps"]
        if hit:
            self.route_index += 1
            self._publish(targets=self._target_queue(), hits=self.route_index)
            self._markers(True)
        self._publish(route_markers=self.display_milestones.update(
            t, p, self.route_index, self.live_curve.waypoint_times,
            confirmed_waypoint_distance_m=float(distance) if hit else None))

    def _finish_tracking(self, t, duration, p, v):
        if t + 1e-9 < duration + self.rules["terminal_hold_s"]:
            return
        metrics = self._compute_metrics(None)
        if not metrics["control_error_gates_passed"]:
            labels = dict(waypoints="有序航点", terminal_hold="终点连续停稳", tilt="总倾角",
                          spatial_bounds="空间边界", sample_clock="采样时钟")
            failed = [labels[key] for key, ok in metrics["acceptance_checks"].items() if not ok]
            self._failure = "未通过" + "、".join(failed)
            self._stop("FAILED")
            return
        self._publish(route_markers=self.display_milestones.update(
            t, p, self.route_index, self.live_curve.waypoint_times, goal_reached=True))
        self._complete_tracking()

    def _completion_hits(self):
        return len(self.live_curve.spec["waypoints"])

    def _compute_metrics(self, config):
        if not self.live_states:
            raise ValueError("暂无参考执行采样")
        return flight_metrics(self.live_states, self.live_curve, self.tracking_fault)
