"""User-requested native viewing on the existing independent observer worker."""
from copy import deepcopy
from pathlib import Path
import queue
import threading

import psutil

from dual_target_control.machine_config import apply_machine_config
from dual_target_control.native_scene import NativeScene


class _GeometricScene(NativeScene):
    def _initialize(self):
        super()._initialize()
        pid = self._window._process_pid()
        arguments = psutil.Process(pid).cmdline()
        logs = [arg.split('=', 1)[1] for arg in arguments if arg.lower().startswith('-abslog=')]
        if len(logs) != 1 or not Path(logs[0]).is_absolute() or not Path(logs[0]).is_file():
            raise RuntimeError('无法找到当前几何实验 UE 进程的启动日志。')
        self._log_path = Path(logs[0])
        self.runtime.set_connected()

    def _switch(self, visible):
        if visible:
            self._initialize()
        super()._switch(visible)


class GeometricViewport:
    def __init__(self, config, view_config, scene_factory=None):
        self.config = apply_machine_config(config)
        self.config.update(deepcopy(view_config))
        self._lock = threading.Lock()
        self._state = dict(mode='airsim', connected=False, config=self.config)
        self._updates = queue.SimpleQueue()
        self.scene = (scene_factory or _GeometricScene)(self.config, self)

    def set_connected(self):
        with self._lock:
            self._state['connected'] = True

    def update_snapshot(self, snapshot):
        with self._lock:
            self._state.update(snapshot)
            self._state['config'] = self.config

    def update_task(self, view_config):
        with self._lock:
            self.config.update(deepcopy(view_config))
        self.scene.update_config(self.config)

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _publish(self, **values):
        self._updates.put(values)

    def take_update(self):
        latest = {}
        while True:
            try:
                latest.update(self._updates.get_nowait())
            except queue.Empty:
                return latest

    def request(self, visible):
        self.scene.request(bool(visible))

    def close(self, timeout=0):
        return self.scene.close(timeout)
