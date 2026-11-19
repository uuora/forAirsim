"""Summarize RotorPy batches using same-route comparisons."""
from collections import Counter
from statistics import median

from .paper_rotorpy_diagnostics import GATES, STATUS_LABELS, WAYPOINT_LABELS
from .paper_rotorpy_report import csv_write, describe
from .runtime import write_json


METRICS = ('reference_s', 'complete_s', 'position_rmse_m', 'position_peak_m', 'attitude_peak_deg')
GEOMETRY = {'bezier': 'Bézier', 'hermite': 'Hermite', 'bspline': 'B 样条'}


def median_or_none(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return median(values) if values else None


def paired_rows(design, indexed):
    pairs = []
    for method in design['methods']:
        for task in design['routes']:
            route = task['id']
            quad = indexed[route, method, 'toppquad']
            for comparator in design['arms'][:-1]:
                base = indexed[route, method, comparator]
                pair = dict(route=route, method=method, comparator=comparator,
                            motor_feasible=quad['reference_rotor_feasible'] and base['reference_rotor_feasible'],
                            both_completed=quad['status'] == base['status'] == 'completed',
                            both_passed=quad['passed'] and base['passed'])
                pair['reference_feasible'] = (pair['motor_feasible'] and quad['reference_speed_passed']
                                              and base['reference_speed_passed'])
                for key in METRICS:
                    if quad.get(key) is not None and base.get(key) is not None:
                        pair[key] = quad[key] - base[key]
                if pair['motor_feasible'] and base['reference_s'] > 0.:
                    pair['reference_reduction_pct'] = -100. * pair['reference_s'] / base['reference_s']
                pairs.append(pair)
    return pairs


def summarize(output, design, rows, details, *, refresh_audit=True):
    key = lambda row: (row['route'], row['method'], row['arm'])
    rows.sort(key=key)
    indexed = {key(row): row for row in rows}
    expected = len(design['routes']) * len(design['methods']) * len(design['arms'])
    if len(indexed) != len(rows) or len(rows) != expected:
        raise ValueError('The campaign record set is incomplete')
    groups, paired_summary = [], []
    pairs = paired_rows(design, indexed)
    gate_labels = dict(GATES, heading='机头跟踪', envelope='参考规划余量', planning='规划失败',
                       audit_or_execution='执行或记录核验异常')
    labels = dict(zip(design['arms'], design['labels']))
    independent = design.get('independent_random', False)
    has_alpha = any(arm.endswith('_alpha') for arm in design['arms'])
    batch_label = '新随机批次' if independent else f"种子 {design['seed']}"
    report_counts = ['正常完成', '参考电机合规', '参考限速达标']
    if not independent:
        report_counts.insert(0, '全任务通过')
    headline_columns = ['构造', '方法', *report_counts, '参考时间中位数 s', '完成时间中位数 s', '位置 RMSE 中位数 m']
    provenance = ('全部方案在本轮规划和执行。保存冻结航点、首飞实测及规划失败回执。' if independent else
                  f"六个对照组新跑 {expected * 6 // 7} 项。TOPPQuad 连续重优化复用 {expected // 7} 项首飞时序，逐条重算过点、姿态、终点和机头判据。")
    lines = [
        '# RotorPy 计时方法对照 · 统一航向', '',
        f"{batch_label}，{len(design['routes'])} 条随机航路 × 三种固定路径构造 × {len(design['arms'])} 种计时方法，共 {expected} 项。",
        provenance,
        '所有组在同一路线、同一构造下使用相同路径几何、航向函数、RotorPy Crazyflie 和 SE(3) 跟踪器。',
        ('三种 α 方法以统一时间倍率使逐电机参考推力进入 [0, 0.14375] N。' if has_alpha else
         '主基线约束参考速度、加速度、倾角及电机余量。另两种 TOPP 按论文设置分别限制速度及速度＋总推力。'),
        'TOPPQuad 使用连续重优化及规划余量。各组统一比较首次闭环。', '',
        '## 共同验收参考值', '',
        '参考限速 5 m/s（数值容差 0.01 m/s），逐电机参考推力合规；实际过点半径 0.30 m、最低速度 0.50 m/s、时间窗 ±1 s；'
        '全程位置误差 ≤0.50 m、姿态误差 ≤10°、实际倾角 ≤35°、机头误差 ≤10°；'
        '终点位置 ≤0.15 m、速度 ≤0.15 m/s、姿态误差 ≤5°并连续保持 1 秒，额外稳定预算 3 秒。', '',
        '## 动力学可行性与用时', '',
        '| ' + ' | '.join(headline_columns) + ' |',
        '|' + '|'.join(['---', '---', *['---:']*(len(headline_columns)-2)]) + '|',
    ]
    for method in design['methods']:
        for arm in design['arms']:
            subset = [row for row in rows if row['method'] == method and row['arm'] == arm]
            failed = Counter(gate for detail in details if detail['method'] == method and detail['arm'] == arm
                             for gate in detail['failed_gates'])
            group = dict(method=method, arm=arm, attempts=len(subset),
                         completed=sum(r['status'] == 'completed' for r in subset),
                         rotor_feasible=sum(r['reference_rotor_feasible'] for r in subset),
                         speed_feasible=sum(r['reference_speed_passed'] for r in subset), failures=dict(failed),
                         command_outside=sum(r.get('thrust_violation_peak_n', 0.) > 1.e-6 for r in subset))
            if not independent:
                group['passed'] = sum(row['passed'] for row in subset)
            groups.append(group)
            medians = [describe(median_or_none(subset, field)) for field in ('reference_s', 'complete_s', 'position_rmse_m')]
            counts = [group[field] for field in ('completed', 'rotor_feasible', 'speed_feasible')]
            if not independent:
                counts.insert(0, group['passed'])
            ratio = [f'{count}/{len(subset)}' for count in counts]
            lines.append('| ' + ' | '.join([GEOMETRY[method], labels[arm], *ratio, *medians]) + ' |')
    lines.extend(['', '## 同路线配对', '',
                  '差值为 TOPPQuad 减去比较组。参考飞行时间同时报告论文的电机合规口径和附加参考限速的口径。'
                  + ('闭环完成时间和误差取双方正常完成的路线，各项越限单独统计。' if independent else
                     '完成与任务通过配对用于闭环完成时间和误差。')
                  + '参考时间缩短率为 (对照参考时间 − TOPPQuad 参考时间) / 对照参考时间。', '',
                  '| 构造 | 对照 | 配对子集 | n | 参考时间差 s | 完成时间差 s | 位置 RMSE 差 m |',
                  '|---|---|---|---:|---:|---:|---:|'])
    cohorts = [('motor_feasible', '双方参考推力合规'), ('reference_feasible', '双方参考限速与推力合规'),
               ('both_completed', '双方正常完成')]
    if not independent:
        cohorts.append(('both_passed', '双方任务通过'))
    for method in design['methods']:
        for arm in design['arms'][:-1]:
            match = [r for r in pairs if r['method'] == method and r['comparator'] == arm]
            for flag, cohort in cohorts:
                subset = [row for row in match if row[flag]]
                fields = ('reference_s',) if flag in ('motor_feasible', 'reference_feasible') else METRICS
                result = dict(method=method, comparator=arm, cohort=cohort, n=len(subset),
                              **{metric: median_or_none(subset, metric) for metric in fields})
                paired_summary.append(result)
                medians = [describe(result.get(metric)) for metric in ('reference_s', 'complete_s', 'position_rmse_m')]
                lines.append(f"| {GEOMETRY[method]} | {labels[arm]} | {cohort} | {len(subset)} | "
                             + ' | '.join(medians) + ' |')
    quad = [r for r in rows if r['arm'] == 'toppquad']
    if independent:
        selected_gates = ('reference_speed', 'waypoint', 'position', 'attitude', 'tilt', 'heading', 'terminal')
        lines.extend(['', '## 逐项验收参考数据', '',
                      '下表为单项超限次数。同一工况可以触发多项，各项用本组全部记录作分母。', '',
                      '| 构造 | 方法 | 样本数 | ' + ' | '.join(gate_labels[g] for g in selected_gates) + ' |',
                      '|---|---|' + '---:|'*(len(selected_gates)+1)])
        for group in groups:
            counts = [str(group['failures'].get(gate, 0)) for gate in selected_gates]
            lines.append('| ' + ' | '.join([GEOMETRY[group['method']], labels[group['arm']],
                                            str(group['attempts']), *counts]) + ' |')
    lines.extend(['', '## 结论', '',
                  f"- TOPPQuad 的参考电机推力合规 {sum(r['reference_rotor_feasible'] for r in quad)}/{len(quad)}。"
                  + (f"闭环正常完成 {sum(r['status'] == 'completed' for r in quad)}/{len(quad)}。"
                     if independent else f"统一首飞门槛下通过 {sum(r['passed'] for r in quad)}/{len(quad)}。")])
    for arm in (arm for arm in design['arms'][:-1] if not has_alpha or arm.endswith('_alpha')):
        selected = [p for p in pairs if p['comparator'] == arm and p['motor_feasible']]
        count = sum(pair['reference_s'] < 0. for pair in selected)
        if selected:
            mid = median_or_none(selected, 'reference_s')
            lines.append(f"- 对 {labels[arm]}，双方参考电机推力合规的 {len(selected)} 对中，TOPPQuad 有 {count} 对参考时间更短。"
                         f"配对时间差中位数为 {mid:+.3f} 秒。")
        if independent:
            completed = [p for p in pairs if p['comparator'] == arm and p['both_completed']]
            if completed:
                time = median_or_none(completed, 'complete_s')
                error = 1000. * median_or_none(completed, 'position_rmse_m')
                lines.append(f"  双方正常完成 {len(completed)} 对，完成时间差 {time:+.3f} 秒、位置 RMSE 差 {error:+.2f} 毫米。")
    lines.extend(['', '逐电机推力、参考飞行时间及配对变化沿用论文第 V 节的比较口径。'
                  + ('本批同时报告闭环完成情况、配对跟踪误差和独立验收指标。' if independent else
                     '本批同时报告闭环任务通过率与配对跟踪误差。')
                  + ' [TOPPQuad 原论文第 V 节](https://arxiv.org/html/2309.11637v2#S5)', '',
                  'Codex 执行仿真、整理和复核实验记录。', ''])
    published_runs = [{k:v for k,v in r.items() if k not in ('passed','gate_passed')} for r in rows] if independent else rows
    published_pairs = [{k:v for k,v in p.items() if k != 'both_passed'} for p in pairs] if independent else pairs
    csv_write(output/'runs.csv', published_runs)
    csv_write(output/'waypoints.csv', [dict(route=r['route'], method=r['method'], arm=r['arm'], **p)
                                      for r in rows for p in r.get('passages', [])])
    csv_write(output/'paired.csv', published_pairs)
    write_json(output/'failure_diagnostics.json', dict(groups=groups, gates=gate_labels,
               statuses=dict(STATUS_LABELS, planning_rejected='规划失败', error='执行或核验异常'),
               waypoint_states=WAYPOINT_LABELS, runs={'/'.join(key(row)): row for row in details}))
    if refresh_audit:
        write_json(output/'audit.json', dict(expected=expected, unique_records=len(indexed),
                   verified_closed_loop=sum(bool(r.get('samples')) and r['status'] != 'error' for r in rows),
                   planning_rejected=sum(r['status'] == 'planning_rejected' for r in rows),
                   failures=[{'route': r['route'], 'method': r['method'], 'arm': r['arm'], 'error': r['error']}
                             for r in rows if r['status'] == 'error']))
    (output/'conclusion.md').write_text('\n'.join(lines), encoding='utf-8')
    write_json(output/'summary.json', dict(groups=groups, paired=paired_summary))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    from .rotorpy_results import load

    parser = ArgumentParser(description='Refresh comparison figures and tables from saved RotorPy records')
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output
    design = load(output/'design.json')
    folders = [output/task['id']/method/arm for task in design['routes']
               for method in design['methods'] for arm in design['arms']]
    summarize(output, design, [load(folder/'result.json') for folder in folders],
              [load(folder/'diagnostics.json') for folder in folders], refresh_audit=False)
