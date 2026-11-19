"""Build paired timing tables and independently recheck saved simulation records."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
from statistics import median
import sys

import numpy as np

from .paper_topp import SpeedReference
from .paper_rotorpy_diagnostics import diagnose
from .rotorpy_results import audit_samples, load
from .runtime import ROOT, write_json


def copy_dashboard_assets(root):
    assets = ROOT/'python/geometric_validation/assets'
    shutil.copyfile(assets/'paper_rotorpy_dashboard.html', root/'index.html')
    for name in ('paper_rotorpy_dashboard.mjs', 'paper_rotorpy_pose.mjs',
                 'paper_rotorpy_attitude.mjs', 'plotly-gl3d-3.1.0.min.js'):
        shutil.copyfile(assets/name, root/name)


def describe(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def write_conclusion(root, design, groups, paired, rows, verified, discrepancies):
    labels = dict(zip(design['arms'], design['labels']))
    methods = {'bezier': 'Bézier', 'hermite': 'Hermite', 'bspline': 'B 样条'}
    n = len(design['routes'])
    lines = [
        '# RotorPy 随机航路对照实验', '',
        f"种子 {design['seed']} 生成 {n} 条独立随机航路，每条由随机起点、两个中间点和随机终点组成。"
        f"三种路径构造各跑七组，共 {len(rows)} 项。数据核验覆盖 {verified} 条闭环记录，发现 {discrepancies} 项差异。", '',
        '## 实验设计', '',
        '在 NED 中采样水平坐标 ±8 m、高度 6–10 m，连续点距离 5–11 m，同路线任意点至少相距 3 m。'
        '每个构造内共享路径几何。原平坦性组按距离除以 5 m/s 分段，速度与总推力 TOPP 对平方路径速率做离散约束优化。'
        '三种 α 组分别用各自原组的时间轴统一降速，以稠密扫描的逐电机参考推力达到 [0, 0.14375] N 为目标。', '',
        f"上游 TOPPQuad 使用 1 m/s 的等几何初值、{design['nodes']} 个节点及 5 m/s 的速度限制。执行层对平方路径速率做三次插值，"
        '遇到非正插值时对具有正下限的速率取对数再插值，按固定偏航重新计算姿态和推力。'
        '原始节点推力、插值后的参考推力和 RotorPy 控制器限幅前指令推力分别入档。稠密扫描是采样验收。', '',
        '所有组用同一 RotorPy Crazyflie 2.0 和 SE(3) 控制器，50 Hz，零风，状态真值。'
        '实际过点用 0.30 m 半径、0.50 m/s 最低速度和 ±1 s 时间窗。'
        '位置峰值 ≤0.50 m、姿态跟踪峰值 ≤10°、实际倾角 ≤35°。末端要求位置 ≤0.15 m、速度 ≤0.15 m/s、姿态 ≤5°，'
        '连续保持 1 s，参考结束后最多 3 s。完成时间含终端保持，无额外统一时长上限。'
        '全任务通过同时要求正常完成、闭环门槛、采样参考电机推力和参考速度。参考速度验收允许 0.01 m/s 数值容差。', '',
        '各判据的失效计数与改进顺序见[失败原因分析](failure_analysis.md)。', '',
        '## 各组结果', '',
        '| 路径 | 方法 | 全任务通过 | 正常完成 | 参考电机合规 | 参考时间中位数 s | 完成时间中位数 s | 位置 RMSE 中位数 m |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for g in groups:
        values = [describe(g['median_' + key]) for key in ('reference_s', 'complete_s', 'position_rmse_m')]
        lines.append(f"| {methods[g['method']]} | {labels[g['arm']]} | {g['passed']}/{g['attempts']} | {g['completed']} | "
                     f"{g['rotor_feasible']} | {' | '.join(values)} |")
    lines.extend(['', '## TOPPQuad 与 α 组配对差', '',
                  '差值为 TOPPQuad 减比较组。报告双方正常完成与双方全部通过两种配对分母。', '',
                  '| 路径 | 比较组 | 配对子集 | n | 参考时间差 s | 完成时间差 s | RMSE 差 m |',
                  '|---|---|---|---:|---:|---:|---:|'])
    for row in paired:
        values = [describe(row[key]) for key in ('reference_s','complete_s','position_rmse_m')]
        lines.append(f"| {methods[row['method']]} | {labels[row['comparator']]} | {row['cohort']} | {row['n']} | {' | '.join(values)} |")
    lines.extend(['', '## 结论', ''])
    for method in design['methods']:
        subset = [r for r in rows if r['method'] == method and r['arm'] == 'toppquad']
        raw = [r for r in subset if r.get('raw_duration_s') is not None]
        raw_good = sum(r['raw_thrust_min_n'] >= -1.e-6 and r['raw_thrust_max_n'] <= .14375+1.e-6 for r in raw)
        rotor = sum(r.get('reference_rotor_feasible', False) for r in subset)
        ok = sum(r['passed'] for r in subset)
        lines.append(f"- {methods[method]} 的 TOPPQuad 上游有 {len(raw)}/{n} 项得到原始解，"
                     f"其中 {raw_good}/{len(raw)} 项节点推力合规。插值参考推力合规 {rotor}/{n}，全任务通过 {ok}/{n}。")
    comparable = [r for r in paired if r['cohort'] == '双方通过' and r['n']]
    if comparable:
        lines.append('- 同时通过全部门槛的时间配对见上表，按各构造和比较组的实际样本数解释。')
    else:
        lines.append('- 双方通过子集为空。双方完成子集的时间差只描述完成仿真的样本，尚不能建立满足同等门槛时的提速结论。')
    lines.append('- 上游优化与跟踪参考在节点离散、姿态处理及电机分配上均有差异。'
                 '上游机臂分配与 RotorPy Crazyflie 的 X 形分配不同，当前数据无法单独量化各项差异的影响。'
                 'AirSim 阶段前应先对齐重建和电机分配。')
    lines.extend(['', '## 原始离散规划时间与 α 组参考时间', '',
                  '下表仅比较上游节点推力合规的 TOPPQuad 原始离散时间与参考推力合规的 α 组时间。两者使用不同阶段的可行性检查，供定位导出前后的变化。', '',
                  '| 路径 | 比较组 | 参考配对数 | TOPPQuad 离散时间减 α 参考时间中位数 s |',
                  '|---|---|---:|---:|'])
    for row in paired:
        if row['cohort'] == '规划配对':
            lines.append(f"| {methods[row['method']]} | {labels[row['comparator']]} | {row['n']} | {describe(row['raw_reference_delta_s'])} |")
    lines.extend(['- Bézier 和 Hermite 使用等价端点条件，二者结果体现实现和求解敏感性。当前数据为 RotorPy 模拟结果，AirSim 留待下一阶段。', '',
                  '参考 [TOPPQuad 原论文 §IV–VI](https://arxiv.org/abs/2309.11637) 与 '
                  '[RotorPy 扩展实验](https://arxiv.org/abs/2506.13915)。', '',
                  'Codex 实施实验、整理原始记录并生成统计和核验回执。'])
    (root/'conclusion.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def csv_write(path, rows):
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (list, dict))})
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def summarize(root):
    design = load(root/'design.json')
    rows, passage, pairs, groups, failures = [], [], [], [], []
    paired_to = ('flat_alpha', 'velocity_alpha', 'collective_alpha')
    sample_count = 0
    for task in design['routes']:
        for method in design['methods']:
            indexed = {}
            for arm in design['arms']:
                directory = Path(task['id'])/method/arm
                folder = root/directory
                row = load(folder/'result.json')
                row['directory'] = directory.as_posix()
                indexed[arm] = row
                rows.append(row)
                passage.extend(dict(route=row['route'], method=method, arm=arm, **p) for p in row.get('passages', []))
                try:
                    if row.get('samples'):
                        frozen = load(folder/'reference.json')
                        cfg = dict(design, hz=design['rate_hz'])
                        # This auditor independently recomputes the execution-only gate.
                        raw = dict(row, accepted=row['gate_passed'], plan_issues=[])
                        audit_samples(folder, raw, cfg, frozen)
                        table = np.loadtxt(folder/'reference_metrics.csv', delimiter=',', skiprows=1)
                        reference = SpeedReference(frozen['base'], frozen['source_nodes_s'], frozen['squared_path_rate'], frozen['alpha'])
                        np.testing.assert_allclose(reference.times[1:], np.cumsum(frozen['segment_durations']), atol=1.e-9, rtol=0.)
                        probes = table[np.linspace(0, len(table)-1, 30, dtype=int)]
                        values = reference.sample_batch(np.clip(probes[:, 0], 0., reference.duration))
                        np.testing.assert_allclose(values[:, 0, [1, 0, 2]] * [1, 1, -1], probes[:, 1:4], atol=5.e-7, rtol=0.)
                        peak, minimum = table[:, 5:].max(), table[:, 5:].min()
                        np.testing.assert_allclose([peak, minimum, table[:, 4].max()],
                                                   [row['reference_thrust_max_n'], row['reference_thrust_min_n'],
                                                    row['reference_speed_peak_mps']], atol=1.e-8, rtol=0.)
                        assert row['passed'] == (row['gate_passed'] and row['reference_rotor_feasible'] and row['reference_speed_passed'])
                        sample_count += 1
                except (AssertionError, ValueError, KeyError, OSError) as error:
                    failures.append(dict(directory=row['directory'], error=repr(error)))
            final = indexed['toppquad']
            for comparator in paired_to:
                prior = indexed[comparator]
                pair = dict(route=task['id'], method=method, comparator=comparator,
                            baseline_passed=prior['passed'], toppquad_passed=final['passed'])
                for key in ('reference_s','complete_s','position_rmse_m','position_peak_m','attitude_peak_deg'):
                    if prior.get(key) is not None and final.get(key) is not None:
                        pair[key] = final[key] - prior[key]
                if (prior.get('reference_rotor_feasible') and final.get('raw_duration_s') is not None
                        and final['raw_thrust_min_n'] >= -1.e-6 and final['raw_thrust_max_n'] <= .14375+1.e-6):
                    pair['raw_reference_delta_s'] = final['raw_duration_s'] - prior['reference_s']
                pairs.append(pair)
    def med(records, key):
        values = [row[key] for row in records if row.get(key) is not None]
        return median(values) if values else None
    for method in design['methods']:
        for arm in design['arms']:
            cohort = [r for r in rows if r['method'] == method and r['arm'] == arm]
            group = dict(method=method, arm=arm, attempts=len(cohort), passed=sum(r['passed'] for r in cohort),
                         completed=sum(r['status'] == 'completed' for r in cohort),
                         tracked=sum(bool(r.get('gate_passed')) for r in cohort),
                         waypoint_passed=sum(bool(r.get('waypoint_passed')) for r in cohort),
                         reference_speed_passed=sum(bool(r.get('reference_speed_passed')) for r in cohort),
                         rotor_feasible=sum(bool(r.get('reference_rotor_feasible')) for r in cohort),
                         raw_solved=sum(r.get('raw_duration_s') is not None for r in cohort),
                         traces=sum(r.get('samples', 0) > 0 for r in cohort),
                         errors=dict(Counter(r['status'] for r in cohort)))
            for key in ('reference_s','complete_s','position_rmse_m','position_peak_m','attitude_peak_deg',
                        'reference_thrust_min_n','reference_thrust_max_n','reference_rotor_violation_n','thrust_violation_peak_n'):
                group['median_'+key] = med(cohort,key)
            groups.append(group)
    paired_summary = []
    for method in design['methods']:
        for comparator in paired_to:
            match = [r for r in pairs if r['method']==method and r['comparator']==comparator]
            for cohort, subset in (('双方完成',[r for r in match if 'complete_s' in r]),
                                   ('双方通过',[r for r in match if r['baseline_passed'] and r['toppquad_passed']])):
                paired_summary.append(dict(method=method, comparator=comparator, cohort=cohort, n=len(subset),
                                           **{key:med(subset,key) for key in ('reference_s','complete_s','position_rmse_m','position_peak_m','attitude_peak_deg')}))
            planning = [r for r in match if 'raw_reference_delta_s' in r]
            paired_summary.append(dict(method=method, comparator=comparator, cohort='规划配对', n=len(planning),
                                       raw_reference_delta_s=med(planning, 'raw_reference_delta_s'),
                                       reference_s=None, complete_s=None, position_rmse_m=None,
                                       position_peak_m=None, attitude_peak_deg=None))
    csv_write(root/'runs.csv', rows)
    csv_write(root/'waypoints.csv', passage)
    csv_write(root/'paired.csv', pairs)
    write_json(root/'audit.json', dict(verified_samples=sample_count, failures=failures))
    write_conclusion(root, design, groups, paired_summary, rows, sample_count, len(failures))
    if not failures:
        diagnose(root)
    write_json(root/'summary.json', dict(groups=groups, paired=paired_summary))
    print(f'Audit verified {sample_count} simulation records, discrepancies {len(failures)}', flush=True)
    return not failures


if __name__ == '__main__':
    raise SystemExit(not summarize(Path(sys.argv[1])))
