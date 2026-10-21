"""Verify the opted-in physics profile against the live UE startup receipt."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from .simulator import ROOT, workspace_path


def requested_profile(config):
    options = config.get('airsim', {})
    if not options.get('settings'):
        return None
    settings = json.loads(workspace_path(options['settings']).read_text(encoding='utf-8-sig'))
    return settings.get('Vehicles', {}).get(options.get('vehicle', 'DroneControl'), {}).get('DynamicsProfile')


def validate_receipt(profile, loaded_profile, receipt, vehicle):
    if loaded_profile != profile:
        raise RuntimeError('正在运行的 AirSim 动力配置与当前设置不同，请结束仿真后重新启动。')
    if (profile.get('Schema') != 1 or profile.get('Name') != 'dual_target_v1'
            or receipt.get('schema') != 1 or receipt.get('profile') != profile['Name']
            or receipt.get('vehicle') != vehicle or receipt.get('model') != 'GenericQuadX'):
        raise RuntimeError('AirSim 动力参数回执的机型、版本或无人机不一致。')
    names = ('mass_kg', 'max_thrust_per_rotor_n', 'max_torque_per_rotor_nm',
             'total_max_thrust_n', 'nominal_twr', 'nominal_gravity_mps2',
             'nominal_hover_throttle', 'min_velocity_throttle', 'max_rpm')
    for name in names:
        value = receipt.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError('AirSim 动力回执包含无效参数：' + name)
    inertia = receipt.get('inertia_diag_kgm2', [])
    if (not isinstance(inertia, list) or len(inertia) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in inertia)):
        raise RuntimeError('AirSim 动力回执的转动惯量无效。')
    if receipt.get('rotor_count') != 4:
        raise RuntimeError('AirSim 动力回执不是四旋翼。')
    checks = (
        (receipt['mass_kg'], profile['MassKg']),
        (receipt['nominal_twr'], profile['ThrustToWeightRatio']),
        (receipt['total_max_thrust_n'], 4 * receipt['max_thrust_per_rotor_n']),
        (receipt['nominal_twr'], receipt['total_max_thrust_n']
         / (receipt['mass_kg'] * receipt['nominal_gravity_mps2'])),
        (receipt['nominal_hover_throttle'], 1 / receipt['nominal_twr']),
    )
    if not all(math.isclose(actual, expected, rel_tol=2e-5, abs_tol=1e-5)
               for actual, expected in checks):
        raise RuntimeError('AirSim 实际初始化的质量或推重比与请求值不一致。')
    if not receipt['min_velocity_throttle'] < receipt['nominal_hover_throttle'] < 1:
        raise RuntimeError('当前动力配置的悬停推力不在速度控制范围内。')
    return dict(receipt, verified=True,
                verification='Live UE physics initialization receipt; nominal parameters, not a thrust calibration')


def read_live_dynamics(client, config, preferred_pid=None):
    profile = requested_profile(config)
    if profile is None:
        return {}
    loaded = json.loads(client.getSettingsString())
    vehicle = config['airsim'].get('vehicle', 'DroneControl')
    loaded_profile = loaded.get('Vehicles', {}).get(vehicle, {}).get('DynamicsProfile')
    if loaded_profile != profile:
        raise RuntimeError('AirSim 尚未加载新的动力配置，请结束仿真后重新启动。')
    # Resolve the existing task process without touching a native window.
    from .native_window import Win32GameWindow
    import psutil
    pid = Win32GameWindow(config, preferred_pid)._process_pid()
    process = psutil.Process(pid)
    logs = [argument.split('=', 1)[1] for argument in process.cmdline()
            if argument.casefold().startswith('-abslog=')]
    expected_log = (ROOT / 'logs/dual_target_control_ue.log').resolve()
    if len(logs) != 1 or Path(logs[0]).resolve() != expected_log:
        raise RuntimeError('无法核验本次 AirSim 动力回执日志。')
    raw = expected_log.read_bytes()
    text = raw.decode('utf-8-sig', errors='replace')
    first_line = text.splitlines()[0] if text else ''
    try:
        opened = datetime.strptime(first_line, 'Log file open, %m/%d/%y %H:%M:%S').timestamp()
    except ValueError as error:
        raise RuntimeError('AirSim 动力日志没有有效的启动时间。') from error
    if abs(opened - process.create_time()) > 30:
        raise RuntimeError('AirSim 动力回执来自旧进程，当前仿真尚未核验。')
    receipts = []
    for line in text.splitlines():
        if 'DTC_DYNAMICS ' in line:
            try:
                data = json.loads(line.split('DTC_DYNAMICS ', 1)[1])
            except ValueError as error:
                raise RuntimeError('AirSim 动力初始化回执不完整。') from error
            if data.get('vehicle') == vehicle:
                receipts.append(data)
    if not receipts:
        raise RuntimeError('当前 AirSim 插件未提供动力回执，请先编译新的动力配置模块。')
    verified = validate_receipt(profile, loaded_profile, receipts[-1], vehicle)
    digest = hashlib.sha256(raw).hexdigest()
    archive = ROOT / '.runtime/dual_target_control/physics_receipts' / f'{pid}_{digest}.log'
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(raw)
    verified.update(pid=pid, log=str(expected_log), log_sha256=digest, log_snapshot=str(archive))
    return verified
