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
OVERVIEW_MAP = '/Game/DroneControlOverview/Maps/DroneControlOverview'
SMALL_FIELD_MAP = '/Game/DroneControlSmallField/Maps/DroneControlSmallField'


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
    if config['airsim']['map'] != OVERVIEW_MAP:
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


def ensure_small_field_visuals(config, project, editor, runtime_dir, environment, publish):
    visual = config.get('ue_visualization')
    if visual is None:
        return
    recipe = ROOT / '.runtime/small_field_visuals.json'
    if recipe.is_file():
        receipt = json.loads(recipe.read_text(encoding='utf-8'))
        assets = [slot['scaled'] for slots in receipt['components'].values() for slot in slots]
        if receipt['scale'] == visual['drone_scale'] and all(
            (project.parent / 'Content' / (asset.removeprefix('/Game/').split('.')[0] + '.uasset')).is_file()
            for asset in assets
        ):
            return
    publish('BUILDING', '正在准备小场地的无人机显示材质。')
    result = subprocess.run([
        str(editor.with_name('UE4Editor-Cmd.exe')), str(project),
        '-ExecutePythonScript=' + str(ROOT / 'scripts/unreal/build_small_field_visuals.py'),
        '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities', '-unattended',
        '-NullRHI', '-nosplash', '-abslog=' + str(ROOT / 'logs/small_field_visuals_ue.log'),
        '-UserDir=' + str(runtime_dir / 'visual_builder'),
    ], cwd=ROOT, env=environment, timeout=120, stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT, **hidden_process_options())
    if result.returncode or not recipe.is_file() or json.loads(recipe.read_bytes())['scale'] != visual['drone_scale']:
        raise RuntimeError('无人机显示材质生成失败，请查看 logs/small_field_visuals_ue.log。')


def small_field_map_matches(report, map_file, builder, bounds, padding):
    try:
        built = json.loads(report.read_bytes())
        lower, upper = bounds
        geometry = dict(x_min=lower[0], x_max=upper[0], y_min=lower[1], y_max=upper[1],
                        grid_spacing_m=1.0, floor_padding_m=padding)
        return (built['status'] == 'PASS' and built['map'] == SMALL_FIELD_MAP
                and built['geometry'] == geometry
                and built['map_sha256'] == hashlib.sha256(map_file.read_bytes()).hexdigest()
                and built['builder_sha256'] == hashlib.sha256(builder.read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError):
        return False


def launch_simulator(config, publish):
    a = config['airsim']
    if port_open(a['host'], a['port']):
        if a.get('require_fresh'):
            raise RuntimeError('独立试验端口已被占用，未连接已有模拟器。')
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
    small_field = a['map'] == SMALL_FIELD_MAP
    overview = a['map'] == OVERVIEW_MAP or small_field
    builder = ROOT / ('scripts/unreal/build_drone_control_overview.py' if overview else
                      'scripts/unreal/build_drone_control_empty.py')
    report = ROOT / 'logs' / ('small_field_map_build.json' if small_field else
                            'dual_target_control_overview_build.json' if overview else
                            'dual_target_control_build.json')
    if small_field:
        bounds = config['small_field']['bounds_zup_m']
        padding = config['ue_visualization']['floor_padding_m']
        needs_build = not small_field_map_matches(report, map_file, builder, bounds, padding)
    else:
        needs_build = not map_file.exists()
    if needs_build:
        publish('BUILDING', '正在生成小场地地图。' if small_field else '首次启动：创建独立的轻量控制场景。')
        cmd_editor = editor.with_name('UE4Editor-Cmd.exe')
        build_environment = dict(environment, AIRSIM_BUILD_SMALL_FIELD_MAP='1' if small_field else '0')
        result = subprocess.run([
            str(cmd_editor), str(project), '-ExecutePythonScript=' + str(builder),
            '-EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities', '-unattended',
            '-NullRHI', '-nosplash', '-abslog=' + str(ROOT / 'logs/dual_target_control_build_ue.log'),
            '-UserDir=' + str(runtime_dir / 'builder')
        ], cwd=ROOT, env=build_environment, timeout=180, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT, **hidden_process_options())
        ready = (small_field_map_matches(report, map_file, builder, bounds, padding) if small_field else
                 report.exists() and json.loads(report.read_text()).get('status') == 'PASS')
        if result.returncode != 0 or not ready:
            raise RuntimeError('最小场景创建失败，查看 logs/dual_target_control_build_ue.log。')
    ensure_mouse_observer(config, map_file, runtime_dir, environment, publish)
    ensure_small_field_visuals(config, project, editor, runtime_dir, environment, publish)
    if a.get('require_fresh') and port_open(a['host'], a['port']):
        raise RuntimeError('独立试验端口在场景准备期间被占用。')
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
