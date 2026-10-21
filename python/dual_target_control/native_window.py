"""Show/hide the existing, positively identified AirSim game window.

This is a user-clicked application feature, not a simulator launcher. It never
closes a window, stops a process, sends keys, or issues flight commands. By
default the game window's system-menu Close command is disabled when shown, so
the Qt "hide UE scene" button can hide it without ending the same simulation.
Set ``native_window.disable_close`` to False to retain the native Close command.
"""
from __future__ import annotations

from copy import deepcopy
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import ipaddress
import ntpath
from pathlib import Path
import sys

import psutil

from .simulator import ROOT


SW_HIDE = 0
SW_RESTORE = 9
SC_CLOSE = 0xF060
MF_BYCOMMAND = 0x0000
MF_GRAYED = 0x0001
MF_DISABLED = 0x0002
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x0080


def _config_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return ntpath.normcase(ntpath.normpath(str(path.resolve())))


def _command_path(value):
    """A running process must declare absolute paths, never rely on its cwd."""
    value = str(value).strip('"')
    if not ntpath.isabs(value) or not ntpath.splitdrive(value)[0]:
        raise RuntimeError('UE 游戏命令行未使用完整的绝对路径。')
    return ntpath.normcase(ntpath.normpath(str(Path(value).resolve())))


def _local_host(host):
    host = str(host).strip().lower()
    if host == 'localhost':
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise RuntimeError('UE 场景窗口只支持本机回环地址。') from exc
    if not address.is_loopback:
        raise RuntimeError('远端 AirSim 没有可在本机显示的 UE 场景窗口。')
    return str(address)


def _binding_matches(host, address):
    try:
        address = ipaddress.ip_address(address)
    except ValueError:
        return False
    if host == 'localhost':
        return address.is_loopback or address.is_unspecified
    wanted = ipaddress.ip_address(host)
    return address == wanted or (address.is_unspecified and address.version == wanted.version)


@dataclass(frozen=True)
class _Target:
    pid: int
    hwnd: int


class _Win32Api:
    """Small Win32 boundary; instantiated lazily and replaced in unit tests."""
    def __init__(self):
        self.user32 = ctypes.WinDLL('user32', use_last_error=True)
        self.callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        declarations = {
            'EnumWindows': ([self.callback_type, wintypes.LPARAM], wintypes.BOOL),
            'GetWindowThreadProcessId': ([wintypes.HWND, ctypes.POINTER(wintypes.DWORD)], wintypes.DWORD),
            'GetClassNameW': ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            'GetWindow': ([wintypes.HWND, wintypes.UINT], wintypes.HWND),
            'GetWindowTextLengthW': ([wintypes.HWND], ctypes.c_int),
            'GetWindowTextW': ([wintypes.HWND, wintypes.LPWSTR, ctypes.c_int], ctypes.c_int),
            'IsWindow': ([wintypes.HWND], wintypes.BOOL),
            'IsWindowVisible': ([wintypes.HWND], wintypes.BOOL),
            'ShowWindow': ([wintypes.HWND, ctypes.c_int], wintypes.BOOL),
            'SetWindowPos': ([wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, wintypes.UINT], wintypes.BOOL),
            'SetForegroundWindow': ([wintypes.HWND], wintypes.BOOL),
            'GetSystemMenu': ([wintypes.HWND, wintypes.BOOL], wintypes.HMENU),
            'GetMenuState': ([wintypes.HMENU, wintypes.UINT, wintypes.UINT], wintypes.UINT),
            'EnableMenuItem': ([wintypes.HMENU, wintypes.UINT, wintypes.UINT], wintypes.UINT),
            'DrawMenuBar': ([wintypes.HWND], wintypes.BOOL),
        }
        for name, (arguments, result) in declarations.items():
            function = getattr(self.user32, name)
            function.argtypes, function.restype = arguments, result
        long_function = 'GetWindowLongPtrW' if ctypes.sizeof(ctypes.c_void_p) == 8 else 'GetWindowLongW'
        self._get_window_long = getattr(self.user32, long_function)
        self._get_window_long.argtypes = [wintypes.HWND, ctypes.c_int]
        self._get_window_long.restype = ctypes.c_ssize_t

    def window_pid(self, hwnd):
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def window_class(self, hwnd):
        value = ctypes.create_unicode_buffer(256)
        if not self.user32.GetClassNameW(hwnd, value, len(value)):
            return ''
        return value.value

    def window_owner(self, hwnd):
        return int(self.user32.GetWindow(hwnd, GW_OWNER) or 0)

    def is_tool_window(self, hwnd):
        return bool(self._get_window_long(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW)

    def window_title(self, hwnd):
        length = self.user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ''
        value = ctypes.create_unicode_buffer(min(length + 1, 4096))
        if not self.user32.GetWindowTextW(hwnd, value, len(value)):
            return ''
        return value.value

    def windows(self):
        result = []

        @self.callback_type
        def callback(hwnd, _):
            result.append((int(hwnd), self.window_pid(hwnd), self.window_class(hwnd)))
            return True

        if not self.user32.EnumWindows(callback, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        return result

    def is_window(self, hwnd):
        return bool(self.user32.IsWindow(hwnd))

    def is_visible(self, hwnd):
        return bool(self.user32.IsWindowVisible(hwnd))

    def disable_close(self, hwnd):
        menu = self.user32.GetSystemMenu(hwnd, False)
        if not menu:
            raise RuntimeError('无法禁用 UE 场景窗口的关闭命令。')
        state = self.user32.GetMenuState(menu, SC_CLOSE, MF_BYCOMMAND)
        # Another Qt window/session may already have disabled or removed Close
        # on this same game HWND. Re-showing that window must remain idempotent.
        if state == 0xFFFFFFFF or state & (MF_GRAYED | MF_DISABLED):
            return True
        result = self.user32.EnableMenuItem(menu, SC_CLOSE, MF_BYCOMMAND | MF_GRAYED)
        if result == 0xFFFFFFFF:
            # UE can also rebuild its menu between the query and update.
            state = self.user32.GetMenuState(menu, SC_CLOSE, MF_BYCOMMAND)
            if state == 0xFFFFFFFF or state & (MF_GRAYED | MF_DISABLED):
                return True
            raise RuntimeError('无法禁用 UE 场景窗口的关闭命令。')
        self.user32.DrawMenuBar(hwnd)
        return True

    def resize(self, hwnd, width, height):
        if not self.user32.SetWindowPos(hwnd, 0, 0, 0, width, height,
                                       SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE):
            raise ctypes.WinError(ctypes.get_last_error())

    def show(self, hwnd):
        self.user32.ShowWindow(hwnd, SW_RESTORE)

    def hide(self, hwnd):
        self.user32.ShowWindow(hwnd, SW_HIDE)

    def foreground(self, hwnd):
        # Windows can decline focus stealing while still showing the correct
        # window. The button's success does not depend on foreground permission.
        return bool(self.user32.SetForegroundWindow(hwnd))


class Win32GameWindow:
    """Operate only the unique UE -game process owning this task's RPC port.

    ``show()`` and ``hide()`` return ``{'pid': int, 'hwnd': int}``. Every operation
    resolves ownership again; ``preferred_pid`` is an additional constraint, not
    authority to select an unrelated UE process. A failed visibility lookup is
    False, while a requested show/hide raises an explicit diagnostic.
    """
    def __init__(self, config, preferred_pid=None):
        self.config = deepcopy(config)
        self.preferred_pid = int(preferred_pid) if preferred_pid is not None else None
        self._api = None
        self._disable_close = self.config.get('native_window', {}).get('disable_close', True)
        if not isinstance(self._disable_close, bool):
            raise ValueError('UE 关闭命令开关必须为布尔值。')

    def _process_pid(self):
        options = self.config.get('airsim', {})
        host = _local_host(options.get('host', '127.0.0.1'))
        port = int(options.get('port', 41460))
        if not 1 <= port <= 65535:
            raise RuntimeError('AirSim RPC 端口无效。')
        try:
            listening = [connection for connection in psutil.net_connections(kind='tcp')
                         if connection.status == psutil.CONN_LISTEN and connection.laddr
                         and connection.laddr.port == port
                         and _binding_matches(host, connection.laddr.ip)]
        except (psutil.Error, OSError) as exc:
            raise RuntimeError('无法核验本任务 AirSim 控制端口的进程。') from exc
        owners = {connection.pid for connection in listening}
        if None in owners or len(owners) != 1:
            raise RuntimeError('本任务 AirSim 控制端口没有唯一可核验的进程。')
        pid = int(next(iter(owners)))
        if self.preferred_pid is not None and pid != self.preferred_pid:
            raise RuntimeError('AirSim 控制端口所属进程与本次启动的 UE 进程不一致。')
        try:
            process = psutil.Process(pid)
            executable, name, arguments = process.exe(), process.name(), process.cmdline()
        except (psutil.Error, OSError) as exc:
            raise RuntimeError('无法读取本任务 UE 游戏进程的身份。') from exc
        if name.casefold() != 'ue4editor.exe' or not arguments:
            raise RuntimeError('AirSim 端口所属进程不是本任务 UE4 游戏窗口。')
        if arguments.count('-game') != 1:
            raise RuntimeError('AirSim 端口所属进程没有唯一的 -game 参数。')
        expected_executable = _config_path(options['editor'])
        projects = [arg for arg in arguments[1:] if str(arg).casefold().endswith('.uproject')]
        maps = [arg for arg in arguments[1:] if str(arg).startswith('/Game/')]
        settings = [arg.split('=', 1)[1] for arg in arguments[1:]
                    if str(arg).casefold().startswith('-settings=')]
        expected_map = str(options['map'])
        if not expected_map.startswith('/Game/'):
            raise RuntimeError('配置中的 UE 地图必须是完整的 /Game 资源路径。')
        if (_command_path(executable) != expected_executable or
                _command_path(arguments[0]) != expected_executable or
                len(projects) != 1 or _command_path(projects[0]) != _config_path(options['project']) or
                maps != [expected_map] or len(settings) != 1 or
                _command_path(settings[0]) != _config_path(options['settings'])):
            raise RuntimeError('AirSim 端口所属 UE 进程的项目、地图或 settings 与本任务配置不一致。')
        return pid

    def _resolve(self):
        if sys.platform != 'win32':
            raise RuntimeError('显示本机 UE 场景窗口仅支持 Windows。')
        pid = self._process_pid()
        if self._api is None:
            self._api = _Win32Api()
        windows = [hwnd for hwnd, owner, class_name in self._api.windows()
                   if owner == pid and class_name == 'UnrealWindow'
                   and self._is_game_window(hwnd, pid)]
        if len(windows) != 1:
            raise RuntimeError('本任务 UE 进程没有唯一的 Unreal 游戏窗口。')
        target = _Target(pid=pid, hwnd=int(windows[0]))
        self._check_window(target)
        return target

    def _check_window(self, target):
        if not self._is_game_window(target.hwnd, target.pid):
            raise RuntimeError('已核验的 UE 游戏窗口已失效，请重新连接。')

    def _is_game_window(self, hwnd, pid):
        # UE's screenshot notifications can also use the UnrealWindow class.
        # The main game viewport is an unowned, titled, non-tool top-level
        # window. Avoid checking caption/border styles: fullscreen lacks them.
        return (self._api.is_window(hwnd) and self._api.window_pid(hwnd) == pid
                and self._api.window_class(hwnd) == 'UnrealWindow'
                and self._api.window_owner(hwnd) == 0
                and not self._api.is_tool_window(hwnd)
                and bool(self._api.window_title(hwnd).strip()))

    def show(self):
        target = self._resolve()
        if self._disable_close:
            self._check_window(target)
            self._api.disable_close(target.hwnd)
        self._check_window(target)
        self._api.resize(target.hwnd, 1200, 800)
        self._check_window(target)
        self._api.show(target.hwnd)
        self._check_window(target)
        self._api.foreground(target.hwnd)
        return dict(pid=target.pid, hwnd=target.hwnd)

    def hide(self):
        target = self._resolve()
        self._check_window(target)
        self._api.hide(target.hwnd)
        return dict(pid=target.pid, hwnd=target.hwnd)

    def is_visible(self):
        try:
            target = self._resolve()
            return bool(self._api.is_visible(target.hwnd))
        except (RuntimeError, psutil.Error, OSError, KeyError, ValueError):
            return False
