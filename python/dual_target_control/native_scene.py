"""On-demand native UE viewport, independent of the flight RPC owner.

Uses this Development build's reflected UE console functions. The engine log
acknowledges each call: simRunConsoleCommand's bool alone is not a receipt.
No images, flight commands, pause, reset, or simulator launch happen here.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import queue
import re
import threading
import time

from .overview import OverviewStream
from .core import ControlConfig
from .simulator import ROOT


class _ViewCancelled(Exception):
    pass


def view_commands(map_name, visible):
    if not map_name.startswith('/Game/') or any(c in map_name for c in "\r\n ;'\""):
        raise ValueError('UE 场景地图路径无效。')
    leaf = map_name.rsplit('/', 1)[-1]
    camera = f'{map_name}.{leaf}:PersistentLevel.ExternalCamera'
    component = camera + '.CameraComponent'
    world_context = f'{map_name}.{leaf}:PersistentLevel.CameraDirector'
    rendering = ('ke /Script/Engine.Default__GameplayStatics SetEnableWorldRendering '
                 + world_context)
    if visible:
        return [
            'ke /Script/AirSim.CameraDirector setMode 3',
            f'ke {component} SetVisibility true false',
            f'ke {component} Activate true',
            f'ke /Script/Engine.PlayerController SetViewTargetWithBlend {camera} 0 0 0 false',
            rendering + ' true',
            'ke /Script/Blocks.DroneControlObserver SetEnabled true',
        ]
    return [
        'ke /Script/Blocks.DroneControlObserver SetEnabled false',
        'ke /Script/AirSim.CameraDirector setMode 6',
        f'ke {component} Deactivate',
        f'ke {component} SetVisibility false false',
        rendering + ' false',
    ]


def reflected_command(client, command, log_path, timeout=1.):
    """Require an actual reflection success written after this invocation."""
    log_path = Path(log_path)
    offset = log_path.stat().st_size  # Missing log is an explicit unsupported build.
    if not client.simRunConsoleCommand(command):
        raise RuntimeError('UE 没有接收场景切换命令。')
    invoked = command.split(' ', 2)[2]
    deadline = time.monotonic() + timeout
    while True:
        with log_path.open('rb') as stream:
            stream.seek(offset)
            rows = stream.read().decode('utf-8', errors='replace').splitlines()
        for row in rows:
            match = re.search(r"Called '([^']*)'", row)
            if match and match[1].strip() == invoked:
                if '(call succeeded)' in row or '(1 succeeded)' in row:
                    return row
                raise RuntimeError('UE 场景函数执行失败：' + invoked)
        if time.monotonic() >= deadline:
            raise RuntimeError('UE 场景函数缺少执行回执：' + invoked)
        time.sleep(.025)


class NativeScene:
    def __init__(self, config, runtime, window_factory=None, client_factory=None):
        self.config = deepcopy(config)
        self.runtime = runtime
        self._window_factory = window_factory
        self._client_factory = client_factory
        self._queue = queue.Queue()
        self._thread = None
        self._lock = threading.Lock()
        self._closed = False
        self._stopping = threading.Event()
        self._visible = False
        self._needs_restore = False
        self._window = None
        self._client = None
        self._markers = None
        self._receipts = []
        self._log_path = ROOT / 'logs/dual_target_control_ue.log'

    def _publish(self, **values):
        self.runtime._publish(**values)

    def request(self, visible):
        with self._lock:
            if self._closed:
                return
            if not visible and self._thread is None:
                return
            if self._thread is None:
                self._thread = threading.Thread(target=self._work,
                    name='dual-target-native-scene', daemon=True)
                self._thread.start()
            self._queue.put('show' if visible else 'hide')

    def update_config(self,config):
        with self._lock:
            if self._closed:return
            self._queue.put(('config',deepcopy(config)))

    def sync_targets(self,payload):
        with self._lock:
            if self._closed:raise RuntimeError('标记观察线程已关闭')
            if self._thread is None:
                self._thread=threading.Thread(target=self._work,name='dual-target-native-scene',daemon=True);self._thread.start()
            self._queue.put(('targets',deepcopy(payload)))

    def _sync_targets(self,payload):
        import numpy as np
        import airsim
        if self._client is None:self._initialize()
        receipts=[]
        for i,name in enumerate(('DC_TargetRed','DC_TargetBlue')):
            point=payload['points'][i] if i<len(payload['points']) else None
            xyz=np.array([0.,0.,1000.]) if point is None else np.asarray([point[0],point[1],-point[2]])+payload['origin']
            if point is not None:
                if not self._client.simSetObjectScale(name,airsim.Vector3r(1.2,1.2,1.2)):raise RuntimeError('目标缩放失败：'+name)
            if not self._client.simSetObjectPose(name,airsim.Pose(airsim.Vector3r(*xyz),airsim.Quaternionr()),teleport=True):raise RuntimeError('目标放置失败：'+name)
            actual=self._client.simGetObjectPose(name).position;actual=[actual.x_val,actual.y_val,actual.z_val]
            if not np.allclose(actual,xyz,atol=.002):raise RuntimeError('目标位置读回不符：'+name)
            scale=self._client.simGetObjectScale(name);scale=[scale.x_val,scale.y_val,scale.z_val]
            if point is not None and not np.allclose(scale,[1.2]*3,atol=.001):raise RuntimeError('目标尺寸读回不符：'+name)
            receipts.append(dict(name=name,visible=point is not None,requested_ned=xyz.tolist(),observed_ned=actual,scale=scale,
                route_index=payload['route_index'],reference_time_s=payload['reference_time_s'],
                scope='Asynchronous display-thread pose/scale readback; no flight commands'))
        self.runtime._accept_target_receipts(payload,receipts)

    def _field_bounds_command(self):
        task=self.config.get('preparation',{}).get('task')
        if task:
            import numpy as np
            points=np.asarray([task['start'],*task['points']],dtype=float)
            low=points.min(axis=0)-[3.,3.,2.];high=points.max(axis=0)+[3.,3.,2.]
            low[2]=max(0.,low[2])
            values=[low[0],high[0],low[1],high[1],low[2],high[2]]
        else:
            bounds=ControlConfig.from_dict(self.config.get('control',self.config))
            values=[getattr(bounds,key) for key in ('x_min','x_max','y_min','y_max','height_min','height_max')]
        return 'ke /Script/Blocks.DroneControlObserver SetFieldBounds '+' '.join(str(float(v)) for v in values)

    def close(self, timeout=None):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._stopping.set()
                if self._thread is not None:
                    self._queue.put('close')
            thread = self._thread
        if timeout is not None and thread is not None:
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def _initialize(self):
        if self._client is not None:
            return
        import airsim
        from .native_window import Win32GameWindow
        options = self.config['airsim']
        client = (self._client_factory or airsim.MultirotorClient)(
            ip=options['host'], port=int(options['port']), timeout_value=2.)
        try:
            if not client.ping() or client.listVehicles() != [options['vehicle']]:
                raise RuntimeError('UE 场景不属于当前单无人机控制会话。')
            owned = getattr(self.runtime, '_process', None)
            window = (self._window_factory or Win32GameWindow)(self.config,
                preferred_pid=owned.pid if owned is not None else None)
        except Exception:
            try: client.client.close()
            except Exception: pass
            raise
        marker_config = deepcopy(self.config)
        marker_config.setdefault('overview', {})['ue_debug_markers'] = True
        self._markers = OverviewStream(marker_config, self.runtime)
        self._client, self._window = client, window
        self._airsim = airsim

    def _switch(self, visible):
        if visible and self._visible:
            # Repeated connect/configure/start requests preserve mouse framing.
            return
        if visible:
            state = self.runtime.snapshot()
            if state.get('mode') != 'airsim' or not state.get('connected'):
                raise RuntimeError('连接 AirSim 后才能查看 UE4 场景。')
            self._initialize()
            binding = self._window.show()  # Validates port/PID/map/settings before mutation.
            self._needs_restore = True
        elif self._client is None:
            return
        else:
            binding = None
        if not visible:
            errors = self._restore()
            if errors:
                raise RuntimeError('；'.join(errors))
            return
        try:
            self._receipts.append(reflected_command(self._client,
                self._field_bounds_command(), self._log_path))
            commands=view_commands(self.config['airsim']['map'],visible)
            route_view=bool(self.config.get('preparation'))
            if route_view:
                # Enable the initialized observer before enabling rendering, so
                # the external camera is not flashed between the two views.
                commands[-2],commands[-1]=commands[-1],commands[-2]
            for command in commands:
                if self._stopping.is_set():
                    raise _ViewCancelled()
                if route_view and command.endswith('SetEnabled true') and getattr(self.runtime,'_observer_initialized_pid',None)!=(binding or {}).get('pid'):
                    self._receipts.append(reflected_command(self._client,'ke /Script/Blocks.DroneControlObserver ResetOverview',self._log_path))
                self._receipts.append(reflected_command(self._client, command, self._log_path))
            if route_view:self.runtime._observer_initialized_pid=(binding or {}).get('pid')
            elif self.config.get('curve_experiment',{}).get('auto_show_scene'):
                self._receipts.append(reflected_command(self._client,
                    'ke /Script/Blocks.DroneControlObserver SetFollow true',self._log_path))
            if self._stopping.is_set():
                raise _ViewCancelled()
        except Exception:
            # A view error is isolated from the control lifecycle.
            self._restore()
            raise
        self._visible = visible
        self._publish(scene_visible=visible, scene_render_unknown=False,
            scene_status='VISIBLE' if visible else 'HIDDEN',
            scene_pid=(binding or {}).get('pid'), scene_receipts=self._receipts[-7:],
            scene_message=('UE4 鼠标观察：左拖旋转、右拖平移、滚轮缩放；双击无人机聚焦。'
                           if visible else 'UE4 画面已隐藏，主视口停止渲染；控制继续运行。'))

    def _restore(self):
        """Each rollback step runs even when another RPC fails; always hide."""
        errors = []
        for command in view_commands(self.config['airsim']['map'], False):
            try:
                self._receipts.append(reflected_command(self._client, command, self._log_path))
            except Exception as error:
                errors.append(str(error))
        binding = {}
        try:
            binding = self._window.hide()
            self._visible = False
        except Exception as error:
            # Keep the possible visible state for another cleanup attempt.
            self._visible = True
            errors.append(str(error))
        self._needs_restore = bool(errors)
        self._publish(scene_visible=self._visible, scene_render_unknown=bool(errors),
            scene_status='ERROR' if errors else 'HIDDEN', scene_pid=binding.get('pid'),
            scene_receipts=self._receipts[-9:],
            scene_message=('UE4 恢复失败：' + '；'.join(errors) if errors else
                           'UE4 画面已隐藏，主视口停止渲染；控制继续运行。'))
        return errors

    def _work(self):
        try:
            while not self._stopping.is_set():
                try:
                    command = self._queue.get(timeout=.25)
                except queue.Empty:
                    if self._visible:
                        try:
                            warning = self._markers._draw(self._client, self._airsim, self.runtime.snapshot())
                            self._publish(scene_message=(
                                'UE4 鼠标观察：左拖旋转、右拖平移、滚轮缩放；双击无人机聚焦。'
                                + (' ' + warning if warning else '')))
                        except Exception as error:
                            self._publish(scene_message='UE4 场景标记未更新：' + str(error))
                    continue
                if command == 'close' or self._stopping.is_set():
                    break
                try:
                    if isinstance(command,tuple) and command[0]=='config':
                        self.config=command[1]
                        if self._client is not None:
                            self._receipts.append(reflected_command(self._client,self._field_bounds_command(),self._log_path))
                    elif isinstance(command,tuple) and command[0]=='targets':
                        self._sync_targets(command[1])
                    elif isinstance(command, tuple) and command[0] == 'screenshot':
                        # Development verification/export of this native viewport.
                        destination = Path(command[1]).resolve()
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if self._visible:
                            self._client.simRunConsoleCommand(
                                'HighResShot 1200x800 filename="' + destination.as_posix() + '"')
                    else:
                        self._switch(command == 'show')
                except _ViewCancelled:
                    break
                except Exception as error:
                    self._publish(scene_visible=self._visible, scene_status='ERROR',
                                  scene_message='UE4 场景切换失败：' + str(error))
        finally:
            if self._visible or self._needs_restore:
                try:
                    self._switch(False)
                except Exception as error:
                    self._publish(scene_status='ERROR', scene_message='隐藏 UE4 场景失败：' + str(error))
                    try: self._window.hide()
                    except Exception: pass
            if self._client is not None:
                try: self._client.client.close()
                except Exception: pass
