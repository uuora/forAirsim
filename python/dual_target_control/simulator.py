"""Isolated local simulator launcher; never stops an existing UE process."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def workspace_path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def port_open(host, port):
    try:
        with socket.create_connection((host, int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def hidden_process_options():
    if os.name != 'nt':
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = subprocess.SW_HIDE
    return {'startupinfo': startup, 'creationflags': subprocess.CREATE_NO_WINDOW}


def ensure_mouse_observer(config, map_file, runtime_dir, environment, publish):
    if config['airsim']['map'] != '/Game/DroneControlOverview/Maps/DroneControlOverview':
        return
    report = ROOT / 'logs/dual_target_mouse_observer_install.json'
    digest = hashlib.sha256(map_file.read_bytes()).hexdigest()
    try:
        receipt = json.loads(report.read_text(encoding='utf-8'))
        if receipt.get('status') == 'PASS' and receipt.get('version') == 1 and receipt.get('map_sha256') == digest:
            return
    except (OSError, ValueError):
        pass
    publish('BUILDING', '正在为本任务地图安装鼠标观察组件。')
    editor = workspace_path(config['airsim']['editor']).with_name('UE4Editor-Cmd.exe')
    project = workspace_path(config['airsim']['project'])
    result = subprocess.run([
        str(editor), str(project),
        '-ExecutePythonScript=' + str(ROOT / 'scripts/unreal/install_drone_control_observer.py'),
        '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities', '-unattended',
        '-NullRHI', '-nosplash', '-abslog=' + str(ROOT / 'logs/dual_target_mouse_observer_install_ue.log'),
        '-UserDir=' + str(runtime_dir / 'observer_installer')
    ], cwd=ROOT, env=environment, timeout=180, stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT, **hidden_process_options())
    receipt = json.loads(report.read_text(encoding='utf-8')) if report.exists() else {}
    if result.returncode != 0 or receipt.get('status') != 'PASS' or receipt.get('map_sha256') != hashlib.sha256(map_file.read_bytes()).hexdigest():
        raise RuntimeError('鼠标观察组件安装失败，请检查 Blocks 模块编译与安装日志。')


def launch_simulator(config, publish):
    a = config['airsim']
    if port_open(a['host'], a['port']):
        publish('CONNECTING', '端口已在运行，检查指定无人机。')
        return None
    editor = workspace_path(a['editor'])
    project = workspace_path(a['project'])
    settings = workspace_path(a['settings'])
    map_file = project.parent / 'Content' / (a['map'].removeprefix('/Game/') + '.umap')
    if not editor.exists() or not project.exists() or not settings.exists():
        raise FileNotFoundError('找不到 UE、Blocks 项目或本任务 settings 配置。')
    runtime_dir = ROOT / '.runtime/dual_target_control'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment['UE-LocalDataCachePath'] = str(ROOT / '.unreal-cache')
    if not map_file.exists():
        overview = a['map'] == '/Game/DroneControlOverview/Maps/DroneControlOverview'
        publish('BUILDING', '首次启动：创建独立的轻量控制场景。')
        builder = ROOT / ('scripts/unreal/build_drone_control_overview.py' if overview else
                          'scripts/unreal/build_drone_control_empty.py')
        cmd_editor = editor.with_name('UE4Editor-Cmd.exe')
        result = subprocess.run([
            str(cmd_editor), str(project), '-ExecutePythonScript=' + str(builder),
            '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities', '-unattended',
            '-NullRHI', '-nosplash', '-abslog=' + str(ROOT / 'logs/dual_target_control_build_ue.log'),
            '-UserDir=' + str(runtime_dir / 'builder')
        ], cwd=ROOT, env=environment, timeout=180, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, **hidden_process_options())
        report = ROOT / ('logs/dual_target_control_overview_build.json' if overview else
                         'logs/dual_target_control_build.json')
        if result.returncode != 0 or not report.exists() or json.loads(report.read_text()).get('status') != 'PASS':
            raise RuntimeError('最小场景创建失败，查看 logs/dual_target_control_build_ue.log。')
    ensure_mouse_observer(config, map_file, runtime_dir, environment, publish)
    publish('LAUNCHING', '正在启动后台 AirSim；连接后可点击“查看 UE4 场景”。')
    # Keep RHI available, as AirSim camera pose APIs remain usable. NoDisplay freezes main view.
    process = subprocess.Popen([
        str(editor), str(project), a['map'], '-game', '-windowed', '-ResX=1200', '-ResY=800',
        '-nosplash', '-NoSound', '-NoVSync', '-settings=' + str(settings),
        '-abslog=' + str(ROOT / 'logs/dual_target_control_ue.log'),
        '-UserDir=' + str(runtime_dir / 'game'),
        '-ExecCmds=t.MaxFPS 120,t.IdleWhenNotForeground 0,Slate.bAllowThrottling 0,r.DepthOfFieldQuality 0,r.BloomQuality 0'
    ], cwd=ROOT, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        **hidden_process_options())
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError('AirSim 启动退出，查看 logs/dual_target_control_ue.log。')
        if port_open(a['host'], a['port']):
            return process
        time.sleep(0.2)
    raise TimeoutError('AirSim 在 60 秒内未开放控制端口。')
