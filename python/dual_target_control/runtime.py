"""One background owner for RPC/control; UI reads immutable published snapshots.

AirSim is the actual 6-DOF physics backend. Demo mode is explicitly a simplified
acceleration-limited point-mass model, and is never reported as AirSim validation.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import threading
import time
import numpy as np

from .core import ControlConfig, RollingController
from .trajectory_profile import load_trajectory_profile
from .simulator import ROOT, launch_simulator


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def to_ned(vector):
    return [float(vector[0]), float(vector[1]), -float(vector[2])]


def from_ned(vector):
    return np.array([vector.x_val, vector.y_val, -vector.z_val], dtype=float)


def json_value(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_value(v) for v in value]
    return value


class DemoVehicle:
    """Lagged point-mass dynamics for controller/UI development only."""
    def __init__(self, config):
        self.config = config
        self.position = np.array([0., 0., 5.])
        self.velocity = np.zeros(3)
        self.yaw = 0.

    def step(self, command, yaw_deg, dt):
        acceleration = (np.asarray(command) - self.velocity) / .18
        norm = np.linalg.norm(acceleration)
        if norm > self.config.max_acceleration:
            acceleration *= self.config.max_acceleration / norm
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        self.yaw = wrap_angle(self.yaw + np.clip(wrap_angle(math.radians(yaw_deg) - self.yaw),
                              -math.radians(self.config.max_yaw_rate_deg) * dt,
                              math.radians(self.config.max_yaw_rate_deg) * dt))


class ControlRuntime:
    def __init__(self, config, mode='airsim'):
        if mode not in ('airsim', 'demo'):
            raise ValueError('mode must be airsim or demo')
        self.config = deepcopy(config)
        self.mode = mode
        self.cfg = ControlConfig.from_dict(self.config.get('control', self.config))
        self.cfg.validate()
        parameters, self._trajectory_name = load_trajectory_profile(self.config)
        self.controller = RollingController(self.cfg, parameters)
        self._commands = queue.Queue()
        self._lock = threading.Lock()
        self._client = None
        self._lifecycle = None
        self._airsim = None
        self._process = None
        self._scene = None
        self._active = False
        self._paused = False
        self._flight_owned = False
        self._motors_disarmed = False
        self._close_requested = False
        self._shutdown_pending = False
        self._closed = threading.Event()
        self._stop_requested = threading.Event()
        self._demo = DemoVehicle(self.cfg)
        self._trail = deque(maxlen=3000)
        self._samples = deque(maxlen=200000)
        self._events = []
        self._previous_futures = deque(maxlen=32)
        self._last_camera = 0.
        self._last_sample = 0.
        self._last_state = None
        self._dynamics = {}
        self._timings = []
        self._planning_times = []
        self._start_time = None
        self._run_dir = None
        self._summary_written = False
        self._failure = None
        self._snap = dict(mode=mode, status='IDLE', connected=mode == 'demo', running=False,
                          position=[0., 0., 0.], velocity=[0., 0., 0.], yaw=0.,
                          targets=[], planned_path=[], trail=[], hits=0, distance=0.,
                          yaw_error_deg=0., planning_ms=0., rpc_ms=0., config=deepcopy(config),
                          message='离线演示使用简化动力学。' if mode == 'demo' else '先启动仿真或连接，再点击开始。',
                          run_dir='', camera_pitch_deg=0., elapsed=0., loop_ms=0.,
                          scene_visible=False, scene_status='HIDDEN', scene_message='',
                          trajectory_name=self._trajectory_name,
                          bezier_parameters=self.controller.parameters.to_dict())
        self._thread = threading.Thread(target=self._work, name='dual-target-control', daemon=False)
        self._thread.start()

    def _enqueue(self, command, value=None):
        if not self._closed.is_set():
            self._commands.put((command, value))

    def start_simulator(self): self._enqueue('simulator')
    def connect(self): self._enqueue('connect')
    def show_scene(self):
        # Viewing has a separate RPC owner, so a view failure cannot stop flight.
        if self.mode != 'airsim' or not self.snapshot().get('connected'):
            self._publish(scene_message='连接 AirSim 后才能查看 UE4 场景。')
            return
        if self._scene is None:
            from .native_scene import NativeScene
            self._scene = NativeScene(self.config, self)
        self._scene.request(True)

    def hide_scene(self):
        if self._scene is not None:
            self._scene.request(False)
    def start(self):
        self._stop_requested.clear()
        self._enqueue('start')
    def pause(self): self._enqueue('pause')
    def resume(self): self._enqueue('resume')
    def stop(self):
        self._stop_requested.set()
        self._enqueue('stop')
    def configure(self, config): self._enqueue('configure', deepcopy(config))
    def shutdown(self):
        if self._scene is not None:
            self._scene.close()
        self._stop_requested.set()
        self._enqueue('shutdown')

    def wait_closed(self, timeout=None):
        return self._closed.wait(timeout)

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snap)

    def _publish(self, status=None, message=None, **values):
        if status is not None: values['status'] = status
        if message is not None: values['message'] = message
        with self._lock:
            self._snap.update(values)

    @property
    def vehicle_name(self):
        return self.config.get('airsim', {}).get('vehicle', 'DroneControl')

    def _connect(self):
        if self.mode == 'demo':
            self._publish('READY', '离线控制演示就绪；未连接 AirSim。', connected=True)
            return
        self._publish('CONNECTING', '检查 AirSim 控制接口。')
        import airsim
        self._airsim = airsim
        options = self.config['airsim']
        self._client = airsim.MultirotorClient(ip=options['host'], port=int(options['port']),
                                              timeout_value=float(options.get('rpc_timeout', 3)))
        self._lifecycle = airsim.MultirotorClient(ip=options['host'], port=int(options['port']), timeout_value=60)
        if not self._client.ping():
            raise RuntimeError('AirSim ping failed')
        self._client.simRunConsoleCommand('t.IdleWhenNotForeground 0')
        self._client.simRunConsoleCommand('Slate.bAllowThrottling 0')
        vehicles = self._client.listVehicles()
        if vehicles != [self.vehicle_name]:
            raise RuntimeError('本任务需要独立的单无人机场景：' + self.vehicle_name + '；实际：' + str(vehicles))
        state = self._client.getMultirotorState(vehicle_name=self.vehicle_name)
        if state.landed_state != airsim.LandedState.Landed:
            raise RuntimeError('无人机已在空中；请先由原控制方落地，再开始新会话。')
        from .dynamics import read_live_dynamics
        self._dynamics = read_live_dynamics(self._client, self.config,
            self._process.pid if self._process is not None else None)
        self._publish(dynamics=deepcopy(self._dynamics))
        self._read_state()
        self._publish('READY', 'AirSim 已连接，点击开始起飞。', connected=True)

    def _read_state(self):
        if self.mode == 'demo':
            return self._demo.position.copy(), self._demo.velocity.copy(), self._demo.yaw, 0.
        begin = time.perf_counter()
        state = self._client.getMultirotorState(vehicle_name=self.vehicle_name)
        kin = state.kinematics_estimated
        position, velocity = from_ned(kin.position), from_ned(kin.linear_velocity)
        yaw = self._airsim.to_eularian_angles(kin.orientation)[2]
        if not np.isfinite(position).all() or not np.isfinite(velocity).all() or not math.isfinite(yaw):
            raise RuntimeError('AirSim 返回非有限状态。')
        self._last_state = state
        rpc = (time.perf_counter() - begin) * 1000
        self._publish(position=position.tolist(), velocity=velocity.tolist(), yaw=yaw, rpc_ms=rpc)
        return position, velocity, yaw, rpc

    def _begin_run(self):
        self._run_dir = ROOT / 'logs' / ('dual_target_' + self.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self._run_dir.mkdir(parents=True, exist_ok=False)
        (self._run_dir / 'config.json').write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding='utf-8')
        if self._dynamics:
            if self._dynamics.get('log_snapshot'):
                (self._run_dir / 'dynamics_startup.log').write_bytes(
                    Path(self._dynamics['log_snapshot']).read_bytes())
            (self._run_dir / 'dynamics.json').write_text(
                json.dumps(self._dynamics, ensure_ascii=False, indent=2), encoding='utf-8')
        self._samples.clear()
        self._trail.clear()
        self._events = []
        self._timings = []
        self._planning_times = []
        self._summary_written = False
        self._failure = None
        self._start_time = time.perf_counter()
        self._publish(run_dir=str(self._run_dir))

    def _start(self):
        if self._active or self._flight_owned:
            return
        self._connect()
        self._begin_run()
        if self.mode == 'demo':
            self._demo = DemoVehicle(self.cfg)
            self._demo.position = np.array([(self.cfg.x_min + self.cfg.x_max) / 2,
                (self.cfg.y_min + self.cfg.y_max) / 2, (self.cfg.height_min + self.cfg.height_max) / 2])
        else:
            c, a, name = self._lifecycle, self._airsim, self.vehicle_name
            initial = float(self.config['airsim'].get('initial_height', 5.))
            if not self.cfg.height_min + self.cfg.margin <= initial <= self.cfg.height_max - self.cfg.margin:
                raise ValueError('初始高度必须位于场地缓冲边界内。')
            c.enableApiControl(True, vehicle_name=name)
            if not c.armDisarm(True, vehicle_name=name):
                raise RuntimeError('AirSim 拒绝解锁。')
            self._motors_disarmed = False
            self._flight_owned = True
            self._publish('TAKEOFF', '正在起飞并进入指定高度。', running=True)
            self._rise(initial)
        position, velocity, yaw, _ = self._read_state()
        self._check_initial_position(position)
        self.controller.reset(position, velocity, yaw)
        (self._run_dir / 'start_state.json').write_text(json.dumps(json_value(dict(
            position=position, velocity=velocity, yaw=yaw,
            targets=self.controller.targets)), ensure_ascii=False, indent=2), encoding='utf-8')
        self._planning_times.append(self.controller.snapshot()['planning_ms'])
        self._active = True
        self._paused = False
        self._publish('RUNNING', '双目标滚动规划中。轨迹参数：' + self._trajectory_name, running=True)

    def _rise(self, height):
        # State-verified vertical approach avoids native moveOnPath's coarse
        # terminal tolerance/False acknowledgements on this installed build.
        begin = previous = time.perf_counter()
        last_command = np.zeros(3)
        goal = np.array([(self.cfg.x_min + self.cfg.x_max) / 2,
                         (self.cfg.y_min + self.cfg.y_max) / 2, height])
        rows = []
        settled = 0
        try:
            while time.perf_counter() - begin < 25:
                if self._stop_requested.is_set():
                    raise RuntimeError('起飞已取消。')
                tick = time.perf_counter()
                position, velocity, yaw, _ = self._read_state()
                error = goal - position
                dt = max(.005, min(.1, tick - previous))
                command = 1.5 * error
                command *= min(1., 2. / max(np.linalg.norm(command), 1e-9))
                change = command - last_command
                change *= min(1., self.cfg.max_acceleration * dt / max(np.linalg.norm(change), 1e-9))
                command = last_command + change
                self._send(command, math.degrees(yaw))
                rows.append(dict(t=tick - begin, position=position.tolist(), velocity=velocity.tolist(), command=command.tolist()))
                settled = settled + 1 if np.linalg.norm(error) < .2 and np.linalg.norm(velocity) < .3 else 0
                if settled >= 5: return
                previous, last_command = tick, command
                time.sleep(max(0., 1. / self.cfg.control_hz - (time.perf_counter() - tick)))
            raise TimeoutError('无人机未在 25 秒内进入初始高度。')
        finally:
            (self._run_dir / 'takeoff.json').write_text(json.dumps(rows), encoding='utf-8')

    def _check_initial_position(self, position):
        lo = np.array([self.cfg.x_min, self.cfg.y_min, self.cfg.height_min]) + self.cfg.margin
        hi = np.array([self.cfg.x_max, self.cfg.y_max, self.cfg.height_max]) - self.cfg.margin
        if np.any(position < lo - .1) or np.any(position > hi + .1):
            raise RuntimeError('当前无人机位置不在场地内部，修改场地范围后重试。')

    def _aim_camera(self, position, target, now):
        if self.mode != 'airsim' or not self.config['airsim'].get('aim_camera', True):
            return
        body_delta = np.asarray(target) - position
        if now - self._last_camera < (.01 if np.linalg.norm(body_delta) < 2. else .05):
            return
        if np.linalg.norm(body_delta) < max(.1, self.cfg.arrival_radius):
            return  # Preserve last valid look direction through target crossing.
        a = self._airsim
        body_orientation = self._last_state.kinematics_estimated.orientation
        offset = a.Vector3r(.1, 0., -.03)
        rotated = body_orientation * offset.to_Quaternionr() * body_orientation.inverse()
        delta = np.asarray(target) - (position + from_ned(rotated))
        yaw = math.atan2(delta[1], delta[0])
        pitch = math.atan2(delta[2], math.hypot(delta[0], delta[1]))
        desired_world = a.to_quaternion(pitch, 0., yaw)
        relative = body_orientation.inverse() * desired_world
        self._client.simSetCameraPose(self.config['airsim'].get('camera', 'front_center'),
            a.Pose(offset, relative), vehicle_name=self.vehicle_name)
        self._last_camera = now
        self._publish(camera_pitch_deg=math.degrees(pitch))

    def _send(self, velocity, yaw_deg):
        if self.mode == 'demo':
            return
        # Finite refresh duration; SimpleFlight subsequently enters timeout hover
        # after its additional 60 ms api_goal_timeout. It does not instantly stop.
        duration = float(self.config['airsim'].get('command_duration', .15))
        future = self._client.moveByVelocityAsync(*to_ned(velocity), duration,
             drivetrain=self._airsim.DrivetrainType.MaxDegreeOfFreedom,
             yaw_mode=self._airsim.YawMode(False, float(yaw_deg)), vehicle_name=self.vehicle_name)
        self._previous_futures.append(future)
        # The next synchronous state RPC pumps this thread's msgpack event loop.
        for f in list(self._previous_futures):
            if getattr(f, '_set_flag', False):
                f.get()  # Propagate RPC errors; False is expected for superseded tasks.
                self._previous_futures.remove(f)

    def _tick(self, dt):
        started = time.perf_counter()
        position, velocity, yaw, rpc = self._read_state()
        lo = np.array([self.cfg.x_min, self.cfg.y_min, self.cfg.height_min])
        hi = np.array([self.cfg.x_max, self.cfg.y_max, self.cfg.height_max])
        if np.any(position < lo - .1) or np.any(position > hi + .1):
            raise RuntimeError('实际轨迹越过场地边界，任务停止并落地。')
        if self._paused:
            command = np.zeros(3)
            yaw_deg = math.degrees(yaw)
            output = self.snapshot()
        else:
            self._aim_camera(position, self.controller.targets[0], started)
            output = self.controller.update(position, velocity, yaw, min(dt, .1))
            command = output['velocity_command']
            yaw_deg = output['yaw_command_deg']
        send_begin = time.perf_counter()
        self._send(command, yaw_deg)
        rpc += (time.perf_counter() - send_begin) * 1000 if self.mode == 'airsim' else 0.
        if self.mode == 'demo':
            self._demo.step(command, yaw_deg, min(dt, .1))
        self._trail.append(position.tolist())
        elapsed = started - self._start_time
        loop_ms = (time.perf_counter() - started) * 1000
        self._timings.append((loop_ms, rpc, dt * 1000))
        event = output.get('hit_event')
        if event:
            self._planning_times.append(output['planning_ms'])
            event = json_value(event)
            event['elapsed'] = elapsed
            event['speed'] = float(np.linalg.norm(velocity))
            self._events.append(event)
        values = dict(position=position.tolist(), velocity=velocity.tolist(), yaw=float(yaw),
                      trail=list(self._trail), rpc_ms=rpc, loop_ms=loop_ms, elapsed=elapsed)
        for key in ('targets', 'planned_path', 'hits', 'distance', 'yaw_error_deg', 'planning_ms', 'boundary_limited'):
            if key in output:
                v = output[key]
                values[key] = np.asarray(v).tolist() if isinstance(v, (np.ndarray, list)) else v
        self._publish(**values)
        self._samples.append(dict(t=elapsed, position=position.tolist(), velocity=velocity.tolist(),
            yaw=float(yaw), velocity_command=np.asarray(command).tolist(), yaw_command_deg=float(yaw_deg),
            hits=values.get('hits', 0), distance=values.get('distance', 0),
            yaw_error_deg=values.get('yaw_error_deg', 0), planning_ms=values.get('planning_ms', 0),
            rpc_ms=rpc, loop_ms=loop_ms, interval_ms=dt * 1000,
            targets=values.get('targets', []), boundary_limited=values.get('boundary_limited', False)))

    def _pause(self):
        if self._active:
            self._paused = True
            self._publish('PAUSED', '暂停推进，持续发送零速度保持位置。')

    def _resume(self):
        if self._active and self._paused:
            position, velocity, yaw, _ = self._read_state()
            self.controller.replan(position, velocity, yaw)
            self._planning_times.append(self.controller.snapshot()['planning_ms'])
            self._paused = False
            # Controller continues same target pair; no hits or random reset on pause.
            self._publish('RUNNING', '继续跟踪当前目标队列。')

    def _land(self):
        if self.mode != 'airsim' or not self._flight_owned:
            return
        self._publish('LANDING', '停止推进，正在落地。', running=True)
        c, name = self._client, self.vehicle_name
        c.cancelLastTask(vehicle_name=name)
        rows = []
        settled = 0
        begin = time.perf_counter()
        try:
            landing_timeout = max(35., (self.cfg.height_max + 2.) / .8 + 10.)
            while time.perf_counter() - begin < landing_timeout:
                tick = time.perf_counter()
                position, velocity, yaw, _ = self._read_state()
                collision = c.simGetCollisionInfo(vehicle_name=name)
                # FastPhysics clears has_collided once ground-lock stops repeat
                # collisions. Its retained collision point/normal still identifies
                # this floor; require current position and stationary readings.
                contact = (collision.object_name == 'DC_CollisionFloor'
                           and collision.normal.z_val < -.7
                           and np.linalg.norm(position - from_ned(collision.position)) < .3
                           and np.linalg.norm(velocity) < .12)
                settled = settled + 1 if contact else 0
                if not self._motors_disarmed:
                    self._send([0, 0, -.8], math.degrees(yaw))
                rows.append(dict(t=tick - begin, position=position.tolist(), velocity=velocity.tolist(),
                                 contact=bool(contact), collision_active=bool(collision.has_collided)))
                if settled >= 8:
                    break
                time.sleep(max(0., .04 - (time.perf_counter() - tick)))
            else:
                raise RuntimeError('未获得持续静止地面接触，保留控制权。')
        finally:
            if self._run_dir:
                (self._run_dir / 'landing.json').write_text(json.dumps(rows), encoding='utf-8')
        if not self._motors_disarmed:
            c.cancelLastTask(vehicle_name=name)
            if not c.armDisarm(False, vehicle_name=name):
                raise RuntimeError('已落地，但解除解锁未成功。')
            self._motors_disarmed = True
        # The firmware flag follows the disarm acknowledgement asynchronously.
        # Do not send more descent goals while waiting for that state transition.
        settled_until = time.perf_counter() + 3.
        while c.getMultirotorState(vehicle_name=name).landed_state != self._airsim.LandedState.Landed:
            if time.perf_counter() > settled_until:
                raise RuntimeError('解除解锁后未获得已落地状态。')
            time.sleep(.05)
        c.enableApiControl(False, vehicle_name=name)
        self._flight_owned = False
        if self._run_dir:
            (self._run_dir / 'cleanup.json').write_text(json.dumps(dict(landed=True, disarmed=True,
                     api_control_released=True, previous_failure=self._failure), ensure_ascii=False), encoding='utf-8')
        self._read_state()

    def _save(self, outcome):
        if not self._run_dir or self._summary_written:
            return
        with (self._run_dir / 'telemetry.jsonl').open('w', encoding='utf-8') as stream:
            for row in self._samples:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        times = np.asarray(self._timings, dtype=float)
        def stats(index):
            if len(times) == 0: return {}
            return dict(p50=float(np.percentile(times[:, index], 50)),
                        p95=float(np.percentile(times[:, index], 95)), maximum=float(times[:, index].max()))
        snap = self.snapshot()
        summary = dict(mode=self.mode, physics='AirSim SimpleFlight 6-DOF' if self.mode == 'airsim' else 'simplified point mass',
            outcome=outcome, error=self._failure, flight_owned=self._flight_owned,
            samples=len(self._samples), hits=snap['hits'], events=self._events,
            loop_compute_ms=stats(0), rpc_ms=stats(1), state_interval_ms=stats(2),
            requested_control_hz=self.cfg.control_hz, config=self.config,
            dynamics=self._dynamics,
            trajectory_name=self._trajectory_name,
            bezier_parameters=self.controller.parameters.to_dict(),
            planning_ms=dict(count=len(self._planning_times),
                             p95=float(np.percentile(self._planning_times, 95)) if self._planning_times else None,
                             maximum=max(self._planning_times) if self._planning_times else None),
            limitation='本版本为控制原型；相机使用仿真可转动云台，高速性能需分速度档实测。')
        (self._run_dir / 'summary.json').write_text(json.dumps(json_value(summary), ensure_ascii=False, indent=2), encoding='utf-8')
        self._summary_written = True

    def _stop(self, outcome='STOPPED'):
        self._active = False
        self._paused = False
        self._land()
        self._save('FAILED' if self._failure else outcome)
        if self._failure:
            self._publish('ERROR', self._failure, running=self._flight_owned)
        else:
            self._publish('STOPPED', '已停止并保存运行记录。', running=False)

    def _handle(self, command, value):
        if command == 'configure':
            if self._active or self._flight_owned:
                raise RuntimeError('停止并落地后才能修改参数。')
            cfg = ControlConfig.from_dict(value.get('control', value))
            cfg.validate()
            parameters, trajectory_name = load_trajectory_profile(value)
            if self._scene is not None:
                if not self._scene.close(timeout=5):
                    raise RuntimeError('场景观察线程仍在结束，当前参数未替换；稍后重试。')
                self._scene = None
            self.config, self.cfg = value, cfg
            self._trajectory_name = trajectory_name
            self.controller = RollingController(cfg, parameters)
            self._demo = DemoVehicle(cfg)
            self._publish(config=deepcopy(value), message='参数已应用。轨迹参数：' + trajectory_name,
                          trajectory_name=trajectory_name,
                          bezier_parameters=parameters.to_dict())
        elif command == 'simulator':
            if self.mode == 'demo': self._connect()
            elif self._active: raise RuntimeError('飞行中不能启动新仿真。')
            else:
                process = launch_simulator(self.config, self._publish)
                if process is not None:
                    self._process = process
                self._connect()
        elif command == 'connect':
            if self._active: return
            self._connect()
        elif command == 'start': self._start()
        elif command == 'pause': self._pause()
        elif command == 'resume': self._resume()
        elif command == 'stop':
            self._stop()
            if self._shutdown_pending and not self._flight_owned:
                self._close_requested = True
        elif command == 'shutdown':
            self._shutdown_pending = True
            self._stop('CLOSED')
            self._close_requested = not self._flight_owned

    def _work(self):
        timer_resolution = False
        if __import__('os').name == 'nt':
            import ctypes
            timer_resolution = ctypes.windll.winmm.timeBeginPeriod(1) == 0
        previous = time.perf_counter()
        idle_sample = previous
        deadline = previous
        try:
            while not self._close_requested:
                try:
                    command, value = self._commands.get(timeout=0.01 if not self._active else 0)
                    self._handle(command, value)
                    previous = time.perf_counter()
                    deadline = previous
                except queue.Empty:
                    pass
                except Exception as ex:
                    self._active = False
                    error = str(ex)
                    if command not in ('stop', 'shutdown'):
                        try:
                            self._land()
                        except Exception as landing:
                            error += '；落地失败：' + str(landing)
                    self._failure = error
                    self._save('FAILED')
                    self._publish('ERROR', error, running=self._flight_owned)
                if self._active:
                    now = time.perf_counter()
                    if now < deadline:
                        time.sleep(min(deadline - now, .005))
                        continue
                    dt = now - previous
                    previous = now
                    deadline = now + 1. / self.cfg.control_hz
                    try:
                        self._tick(max(0.001, dt))
                    except Exception as ex:
                        self._active = False
                        error = str(ex)
                        try: self._land()
                        except Exception as landing: error += '；落地失败：' + str(landing)
                        self._failure = error
                        self._save('FAILED')
                        self._publish('ERROR', error, running=self._flight_owned)
                elif (self.mode == 'airsim' and self._client is not None
                      and self.snapshot().get('connected')
                      and time.perf_counter() - idle_sample >= .5):
                    # Keep the ground/idle display current after spawn settling.
                    idle_sample = time.perf_counter()
                    try: self._read_state()
                    except Exception as error:
                        self._publish('ERROR', str(error), connected=False)
        finally:
            # No abrupt engine kill: a failed landing remains explicitly visible.
            if self._flight_owned:
                try: self._land()
                except Exception as ex: self._publish('ERROR', '关闭时落地失败：' + str(ex), running=True)
            try:
                self._save('CLOSED')
            finally:
                if self._scene is not None:
                    self._scene.close(timeout=15)
                if self._process is not None and not self._flight_owned and self._process.poll() is None:
                    self._process.terminate()  # Only this runtime's own idle game.
                    try: self._process.wait(timeout=10)
                    except __import__('subprocess').TimeoutExpired: pass
                if timer_resolution:
                    ctypes.windll.winmm.timeEndPeriod(1)
                self._closed.set()


def main():
    """Bounded unattended acceptance run (same runtime used by desktop)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('airsim', 'demo'), default='demo')
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--config', default=str(ROOT / 'configs/dual_target_control.json'))
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    runtime = ControlRuntime(config, args.mode)
    try:
        if args.launch: runtime.start_simulator()
        runtime.start()
        wait = time.monotonic() + (240 if args.launch else 60)
        while time.monotonic() < wait:
            snap = runtime.snapshot()
            if snap['status'] == 'ERROR': raise RuntimeError(snap['message'])
            if snap['status'] == 'RUNNING': break
            time.sleep(.1)
        else: raise TimeoutError('运行未开始')
        begin = time.monotonic()
        while time.monotonic() - begin < args.seconds:
            if runtime.snapshot()['status'] == 'ERROR': break
            time.sleep(.1)
    finally:
        runtime.stop()
        runtime.shutdown()
        if not runtime.wait_closed(65): raise TimeoutError('控制线程未结束')
    snap = runtime.snapshot()
    print(json.dumps({key: snap[key] for key in ('mode', 'status', 'hits', 'message', 'run_dir', 'running')}, ensure_ascii=False))
    if snap['status'] == 'ERROR': raise SystemExit(1)


if __name__ == '__main__': main()
