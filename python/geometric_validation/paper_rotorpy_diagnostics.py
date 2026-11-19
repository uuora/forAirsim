"""Diagnose paper-comparison failures from frozen RotorPy evidence."""
from argparse import ArgumentParser
from collections import Counter
import csv
from itertools import groupby
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .rotorpy_results import load
from .runtime import write_json


ROTOR_LIMIT_N = .14375
FORCE_TOL_N = 1.e-6
SPEED_TOL_MPS = .01
METHOD_LABELS = {'bezier': 'Bézier', 'hermite': 'Hermite', 'bspline': 'B 样条'}
GATES = {
    'reference_speed': '参考限速',
    'reference_rotor': '参考电机推力',
    'waypoint': '实际过点',
    'position': '位置误差',
    'attitude': '姿态误差',
    'tilt': '实际倾角',
    'terminal': '终点停稳',
}
STATUS_LABELS = {
    'completed': '正常停稳',
    'terminal_timeout': '终点超时',
    'ground': '低于仿真地面高度',
    'drift': '位置误差超过 20 m',
    'nonfinite': '非有限状态',
    'unresolved_stop': '停止原因未复现',
    'planning_or_logging_error': '规划或记录失败',
}
WAYPOINT_LABELS = {
    'hit': '按序通过',
    'blocked': '前序未通过',
    'no_window': '时间窗内无采样',
    'outside_radius': '未进入半径',
    'too_slow': '球内速度不足',
}


def columns(data, names):
    return np.column_stack([data[name] for name in names])


def first_timestamp(t, mask):
    found = np.flatnonzero(mask)
    return float(t[found[0]]) if len(found) else None


def stop_reason(status, finite, up, error):
    if status != 'nonfinite_or_flyaway_or_ground':
        return status if status in ('completed', 'terminal_timeout') else 'planning_or_logging_error'
    if not finite[-1]:
        return 'nonfinite'
    if up[-1] < 0.:
        return 'ground'
    if error[-1] > 20.:
        return 'drift'
    return 'unresolved_stop'


def rotor_excursion(thrust):
    return max(0., float(np.maximum(-thrust, thrust - ROTOR_LIMIT_N).max()))


def mixer_diagnostics(folder):
    """Hold the body wrench fixed and swap only the actuator allocation matrix."""
    arm = .043
    yaw_moment_per_newton = 7.8e-10 / 2.3e-8
    plus = np.array([
        [1, 1, 1, 1], [0, arm, 0, -arm], [-arm, 0, arm, 0],
        [yaw_moment_per_newton, -yaw_moment_per_newton, yaw_moment_per_newton, -yaw_moment_per_newton],
    ])
    diagonal = arm / np.sqrt(2)
    x = np.array([
        [1, 1, 1, 1], [diagonal, -diagonal, -diagonal, diagonal], [-diagonal, -diagonal, diagonal, diagonal],
        plus[3],
    ])
    raw = np.asarray(load(folder/'raw_solution.json')['raw_rotor_thrust_n'])
    rebuilt = np.loadtxt(folder/'reference_metrics.csv', delimiter=',', skiprows=1, usecols=(5, 6, 7, 8))
    return dict(
        raw_nodes_with_rotorpy_mixer_n=rotor_excursion(np.linalg.solve(x, (raw @ plus.T).T).T),
        rebuilt_with_upstream_mixer_n=rotor_excursion(np.linalg.solve(plus, (rebuilt @ x.T).T).T),
    )


def passage_diagnostics(row, design, points, t, position, speed, finite):
    rule = design['passage']
    states = []
    previous = -np.inf
    for point, gate in zip(points[1:-1], row['passages'], strict=True):
        if previous == np.inf:
            if gate['passed']:
                raise ValueError('Point passed despite an earlier missed point')
            states.append('blocked')
            continue
        window = finite & (t > previous) & (abs(t - gate['reference_time_s']) <= rule['reference_time_window_s'] + 1.e-10)
        close = window & (np.linalg.norm(position - point, axis=1) <= rule['position_tolerance_m'] + 1.e-8)
        hit = np.flatnonzero(close & (speed >= rule['min_actual_speed_mps'] - 1.e-8))
        if len(hit):
            states.append('hit')
            previous = t[hit[0]]
        else:
            states.append('too_slow' if close.any() else 'outside_radius' if window.any() else 'no_window')
            previous = np.inf
        if gate['passed'] != bool(len(hit)):
            raise ValueError('Waypoint diagnostic did not reproduce saved passage')
    return states


def diagnose_run(folder, row, design, points):
    details = dict(route=row['route'], method=row['method'], arm=row['arm'], passed=row['passed'])
    if not row.get('samples'):
        return dict(details, status='planning_or_logging_error', failed_gates=[], waypoint_states=[])
    data = np.atleast_1d(np.genfromtxt(folder/'simulation.csv.gz', delimiter=',', names=True, encoding='utf-8'))
    if len(data) != row['samples']:
        raise ValueError(f'Sample count mismatch at {folder}')
    finite = np.logical_and.reduce([np.isfinite(data[name]) for name in data.dtype.names])
    xyz = columns(data, [f'position_{axis}_m' for axis in 'enu'])
    target = columns(data, [f'reference_{axis}_m' for axis in 'enu'])
    error = np.linalg.norm(xyz - target, axis=1)
    t = data['time_s']
    if not finite.any():
        return dict(details, status=stop_reason(row['status'], finite, xyz[:, 2], error),
                    failed_gates=['terminal'], waypoint_states=[])
    q = Rotation.from_quat(columns(data, ['q' + axis for axis in 'xyzw'])[finite])
    ref_q = Rotation.from_quat(columns(data, ['reference_q' + axis for axis in 'xyzw'])[finite])
    attitude = np.degrees((ref_q.inv() * q).magnitude())
    tilt = np.degrees(np.arccos(np.clip(q.as_matrix()[:, 2, 2], -1, 1)))
    reference_tilt = np.degrees(np.arccos(np.clip(ref_q.as_matrix()[:, 2, 2], -1, 1)))
    speed = np.linalg.norm(columns(data, [f'velocity_{axis}_mps' for axis in 'enu']), axis=1)
    command = columns(data, [f'cmd_f{number}_n' for number in range(1, 5)])[finite]
    command_outside = np.any((command < -FORCE_TOL_N) | (command > ROTOR_LIMIT_N + FORCE_TOL_N), axis=1)
    limits = design['limits']
    within = {
        'reference_speed': row['reference_speed_peak_mps'] <= design['vmax_mps'] + SPEED_TOL_MPS,
        'reference_rotor': (row['reference_thrust_min_n'] >= -FORCE_TOL_N and
                            row['reference_thrust_max_n'] <= ROTOR_LIMIT_N + FORCE_TOL_N),
        'position': error[finite] <= limits['position_error_m'] + 1.e-8,
        'attitude': attitude <= np.degrees(limits['attitude_error_rad']) + 1.e-7,
        'tilt': tilt <= np.degrees(limits['safety_tilt_rad']) + 1.e-7,
    }
    waypoint_states = passage_diagnostics(row, design, points, t, xyz, speed, finite)
    gates = {key: bool(np.all(value)) for key, value in within.items()}
    after_reference = t[finite] + 1.e-10 >= row['reference_s']
    terminal = np.column_stack((error[finite] <= limits['terminal_position_m'] + 1.e-8,
                                speed[finite] <= limits['terminal_speed_mps'] + 1.e-8,
                                attitude <= np.degrees(limits['terminal_attitude_rad']) + 1.e-7))
    good = np.zeros(len(t), dtype=bool)
    good[finite] = after_reference & terminal.all(axis=1)
    longest = max((len(list(group)) for ok, group in groupby(good) if ok), default=0)
    hold = max(0, longest - 1) / design['rate_hz']
    gates.update(waypoint=all(status == 'hit' for status in waypoint_states),
                 terminal=hold >= design['terminal_hold_s'] - 1.e-10)
    if gates['terminal'] != (row['status'] == 'completed') or all(gates.values()) != row['passed']:
        raise ValueError(f'Gate mismatch at {folder}')
    raw_good = (row.get('raw_thrust_min_n') is not None and
                row['raw_thrust_min_n'] >= -FORCE_TOL_N and row['raw_thrust_max_n'] <= ROTOR_LIMIT_N + FORCE_TOL_N)
    details.update(
        status=stop_reason(row['status'], finite, xyz[:, 2], error),
        failed_gates=[key for key in GATES if not gates[key]],
        waypoint_states=waypoint_states, command_outside=bool(command_outside.any()),
        raw_node_rotor_feasible=bool(raw_good), interpolation=row.get('interpolation'),
        tilt_reference_peak_deg=float(reference_tilt.max()), speed_reference_peak_mps=row['reference_speed_peak_mps'],
        force_reference_min_n=row['reference_thrust_min_n'], force_reference_max_n=row['reference_thrust_max_n'],
        force_collective_peak_n=row['collective_peak_n'], alpha=row.get('alpha'),
        position_peak_m=float(error[finite].max()), attitude_peak_deg=float(attitude.max()), tilt_peak_deg=float(tilt.max()),
        first_position_over_s=first_timestamp(t[finite], ~within['position']),
        first_attitude_over_s=first_timestamp(t[finite], ~within['attitude']),
        first_tilt_over_s=first_timestamp(t[finite], ~within['tilt']),
        first_command_over_s=first_timestamp(t[finite], command_outside),
        terminal_longest_hold_s=hold,
        terminal_last_bad=[key for key, ok in zip(('position', 'speed', 'attitude'), terminal[-1]) if not ok],
    )
    if (details['arm'] == 'toppquad' and raw_good and
            design.get('rotorpy_commit') == 'f69a2a4e15788c834dfe58fe95dd55a017fd7bd3' and
            design.get('toppquad_commit') == '3db181cb0c76baaa1669f859f16a83060c2ee7a0'):
        details.update(mixer_diagnostics(folder))
    return details


def aggregate(rows, design):
    groups = []
    for method in design['methods']:
        for arm in design['arms']:
            group = [r for r in rows if r['method'] == method and r['arm'] == arm]
            counts = dict(method=method, arm=arm, attempts=len(group), passed=sum(r['passed'] for r in group),
                          failures=dict(Counter(key for r in group for key in r['failed_gates'])),
                          statuses=dict(Counter(r['status'] for r in group)),
                          waypoint_states=dict(Counter(state for r in group for state in r['waypoint_states'])),
                          timeout_last_bad=dict(Counter(key for r in group if r['status'] == 'terminal_timeout'
                                                        for key in r['terminal_last_bad'])),
                          command_outside=sum(r.get('command_outside', False) for r in group),
                          raw_node_rotor_feasible=sum(r.get('raw_node_rotor_feasible', False) for r in group),
                          interpolation=dict(Counter(r['interpolation'] for r in group if r.get('interpolation'))))
            for key in ('speed_reference_peak_mps', 'force_collective_peak_n', 'force_reference_max_n',
                        'position_peak_m', 'attitude_peak_deg', 'tilt_peak_deg', 'tilt_reference_peak_deg', 'alpha'):
                values = [r[key] for r in group if r.get(key) is not None]
                if values:
                    counts[key] = dict(zip(('min', 'median', 'max'), map(float, np.quantile(values, [0, .5, 1]))))
            groups.append(counts)
    return groups


def write_report(root, design, rows, groups):
    labels = dict(zip(design['arms'], design['labels']))
    by_arm = {arm: [g for g in groups if g['arm'] == arm] for arm in design['arms']}
    flat = by_arm['flat']
    alpha = by_arm['flat_alpha']
    toppquad = by_arm['toppquad']

    def count(items, key, gate):
        return sum(g[key].get(gate, 0) for g in items)

    def size(items):
        return sum(g['attempts'] for g in items)

    def med(group, key):
        values = [f'{g[key]["median"]:.2f}' if key in g else '—' for g in group]
        return ' / '.join(values)

    speed_over = [r['speed_reference_peak_mps'] - design['vmax_mps'] for r in rows
                  if r['arm'] == 'velocity' and 'reference_speed' in r['failed_gates']]
    speed_evidence = (f"仅限速 TOPP 中 {len(speed_over)} 项重建峰速越界，"
                      f"相对 {design['vmax_mps']:g} m/s 超出 {min(speed_over):.3f}–{max(speed_over):.3f} m/s。"
                      if speed_over else '仅限速 TOPP 无重建峰速告警。')
    lines = [
        '# RotorPy 失败原因分析', '',
        f"从种子 {design['seed']} 的 {len(rows)} 项记录重算位置、姿态、倾角、终点保持和有序过点，"
        '并核查已保存的参考限速和推力峰值。每个判据独立计数，一次运行可同时失败多项。'
        '参考推力来自规划参考采样，指令超限来自 RotorPy 控制器的限幅前记录。', '',
        '## 原始平坦性轨迹', '',
        f"- 按距离 ÷ {design['nominal_mps']:g} m/s 分段，同时要求端点静止并保持平滑，"
        f"因此段内峰值可远超标称速度。Bézier / Hermite / B 样条的参考峰速中位数为 {med(flat, 'speed_reference_peak_mps')} m/s，"
        f"{count(flat, 'failures', 'reference_speed')}/{size(flat)} 项超过 {design['vmax_mps']:g} m/s 加 {SPEED_TOL_MPS:g} m/s 容差。"
        '以首段为例，给定时间内的弦长平均速度已经等于标称值，且起步速度为零。'
        '即使两点间走直线，也需要在途中高于标称速度才能按时到达。',
        f"- {count(flat, 'failures', 'reference_rotor')}/{size(flat)} 项参考电机推力越界，"
        f"{sum(g['command_outside'] for g in flat)}/{size(flat)} 项闭环电机指令超限。"
        f"各构造最大总推力需求的中位数为 {med(flat, 'force_collective_peak_n')} N，"
        f"高于四电机合计上限 {ROTOR_LIMIT_N*4:g} N。"
        '快速加减速带来的高推力和姿态需求，与电机饱和及跟踪丢失一致。',
        f"- 原始组正常停稳 {count(flat, 'statuses', 'completed')} 项，终点超时 "
        f"{count(flat, 'statuses', 'terminal_timeout')} 项，地面高度保护触发 {count(flat, 'statuses', 'ground')} 项，"
        f"20 m 跟踪误差保护触发 {count(flat, 'statuses', 'drift')} 项，"
        f"非有限状态 {count(flat, 'statuses', 'nonfinite')} 项。地面高度保护按 ENU 的 U < 0 判定。", '',
        '## α-scaling 与松弛 TOPP', '',
        f"- 平坦性 α 组有 {size(alpha)-count(alpha, 'failures', 'reference_rotor')}/{size(alpha)} 项参考电机推力合规，"
        f"{count(alpha, 'statuses', 'completed')}/{size(alpha)} 项停稳。"
        f"参考速度、姿态误差、实际倾角分别超限 {count(alpha, 'failures', 'reference_speed')}、"
        f"{count(alpha, 'failures', 'attitude')}、{count(alpha, 'failures', 'tilt')} 项。"
        '本批 α 按参考电机推力选倍率。严格限速和倾角需要独立检查。',
        f'- 速度和总推力 TOPP 约束的是离散节点，插值参考再按稠密采样验收。{speed_evidence}'
        'Bézier 和 Hermite 采用相同端状态，几何等价。'
        '总推力对照约束与仅限速对照的通过数相同，下表列出两者的限速、实际倾角与姿态误差失败。', '',
        '## TOPPQuad', '',
        f"- {sum(g['raw_node_rotor_feasible'] for g in toppquad)}/{size(toppquad)} 项原始节点逐电机推力合规，"
        f"重建参考有 {count(toppquad, 'failures', 'reference_rotor')}/{size(toppquad)} 项推力超限。"
        f"普通三次插值 {count(toppquad, 'interpolation', 'cubic')} 项，对数三次回退 "
        f"{count(toppquad, 'interpolation', 'log_cubic_floor_1e-6')} 项，两类都出现参考推力超限。"
        '当前执行层重新插值平方路径速率并固定偏航，使用 RotorPy 的 X 形电机分配重建推力。'
        '节点间插值、姿态处理和电机坐标的影响尚未单独隔离。',
        f"- {count(toppquad, 'statuses', 'completed')}/{size(toppquad)} 项终点停稳，"
        f"姿态误差与实际倾角分别有 {count(toppquad, 'failures', 'attitude')}、"
        f"{count(toppquad, 'failures', 'tilt')} 项超限。"
        '本批倾角门槛用于运行验收，超过门槛后继续记录直至满足既定停止条件。', '',
        '## 逐组失败计数', '',
        '| 路径 | 处理方式 | 通过 | ' + ' | '.join(GATES.values()) + ' | 指令超限航路 |',
        '|---|---|---:|' + '---:|' * (len(GATES) + 1),
    ]
    for g in groups:
        failures = ' | '.join(str(g['failures'].get(key, 0)) for key in GATES)
        lines.append(f"| {METHOD_LABELS[g['method']]} | {labels[g['arm']]} | {g['passed']}/{g['attempts']} | "
                     f"{failures} | {g['command_outside']} |")
    counts = Counter(state for row in rows for state in row['waypoint_states'])
    timeouts = [r for r in rows if r['status'] == 'terminal_timeout']
    timeout_last = Counter(key for row in timeouts for key in row['terminal_last_bad'])
    attempts = [r for r in rows if r['arm'] == 'toppquad' and 'raw_nodes_with_rotorpy_mixer_n' in r]
    if attempts:
        swapped = sum(r['raw_nodes_with_rotorpy_mixer_n'] > FORCE_TOL_N for r in attempts)
        rebuilt = sum(r['rebuilt_with_upstream_mixer_n'] > FORCE_TOL_N for r in attempts)
        high_total = sum(r['force_collective_peak_n'] > 4 * ROTOR_LIMIT_N + FORCE_TOL_N for r in attempts)
        lines.extend([
            '', '## TOPPQuad 电机分配诊断', '',
            '在保存的节点与参考采样上固定总推力和机体力矩，分别换用另一套电机分配矩阵。',
            f"对 {len(attempts)} 个原始节点解换用 RotorPy X 形分配，有 {swapped} 项至少一台电机越界。"
            f"对重建后的参考换回上游分配，还有 {rebuilt} 项越界。"
            f"其中 {high_total} 项的参考总推力峰值超过 0.575 N，任何电机分配都无法在这一推力上限内满足该采样点。",
            '这是保持机体力矩分量固定的离线重分配，未重新求解或闭环执行。', '',
        ])
    lines.extend([
        '', '## 过点与终点诊断', '',
        '逐航点按首个满足半径、速度和时间窗的采样做有序验收。'
        f"未进半径 {counts['outside_radius']} 个，进入半径但速度不足 {counts['too_slow']} 个，"
        f"窗口内无采样 {counts['no_window']} 个，前序失败后未继续验收 {counts['blocked']} 个。"
        '这些计数的单位是航点。',
        f"{len(timeouts)} 项终点超时中，末帧位置、速度和姿态分别还有 {timeout_last['position']}、"
        f"{timeout_last['speed']}、{timeout_last['attitude']} 项越界。"
        '各项可重叠，且末帧合格仍需验证连续保持时长。最长保持时间也保存在逐次 CSV。', '',
        '## 改进顺序', '',
        '1. 保留原始无约束组供论文比较。新增面向可跟踪性的平坦性对照，统一复核参考速度、倾角和逐电机推力，'
        '再用同一闭环门槛确认姿态和实际过点。',
        '2. 对松弛 TOPP，在重建后校核峰速并预留余量，同时把倾角约束接到规划层。'
        '可要求竖直比推力为正，并使水平比推力不超过竖直比推力乘以允许倾角的正切。',
        '3. 对 TOPPQuad，先对齐求解和执行的电机坐标及偏航，再对同一路线比较原始节点、节点间连续重建和闭环响应。'
        '使用相同门槛复跑后，再判断是否具备进入 AirSim 的条件。', '',
        '逐次详情见 failure_diagnostics.csv。first_*_over_s 是 50 Hz 原始数据中首次越界时刻。'
        '指令超限计数供诊断，原任务分数按原判据重算。', '',
        '方法依据 [TOPPQuad 原论文 §IV–VI](https://arxiv.org/abs/2309.11637) 与 '
        '[固定版本 RotorPy 控制器和动力学](https://github.com/spencerfolk/rotorpy/tree/f69a2a4e15788c834dfe58fe95dd55a017fd7bd3/rotorpy)。', '',
        'Codex 重算已保存数据并编写诊断。', '',
    ])
    (root/'failure_analysis.md').write_text('\n'.join(lines), encoding='utf-8')


def diagnose(root):
    design = load(root/'design.json')
    rows = []
    for task in design['routes']:
        points = np.asarray(task['points_ned'])[:, [1, 0, 2]] * [1., 1., -1.]
        for method in design['methods']:
            for arm in design['arms']:
                folder = root/task['id']/method/arm
                rows.append(diagnose_run(folder, load(folder/'result.json'), design, points))
    groups = aggregate(rows, design)
    keys = sorted({key for row in rows for key in row})
    with (root/'failure_diagnostics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows({k: '|'.join(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    write_report(root, design, rows, groups)
    run_keys = ('status', 'failed_gates', 'waypoint_states', 'first_position_over_s', 'first_attitude_over_s',
                'first_tilt_over_s', 'first_command_over_s')
    runs = {'/'.join(row[key] for key in ('route','method','arm')): {key:row.get(key) for key in run_keys} for row in rows}
    write_json(root/'failure_diagnostics.json', dict(groups=groups, gates=GATES, statuses=STATUS_LABELS,
                                                    waypoint_states=WAYPOINT_LABELS, runs=runs))
    print(f'Diagnosed {len(rows)} RotorPy attempts', flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('campaign', type=Path)
    diagnose(parser.parse_args().campaign)
