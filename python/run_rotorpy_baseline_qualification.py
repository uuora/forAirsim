"""Replay the fixed paper geometries with conservative, independently checked timing."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess

from run_rotorpy_waypoint_comparison import PLANNER_CONFIG, ROOT, ROTORPY_COMMIT, WORKER, enu, init_worker, simulate

import numpy as np

from geometric_validation.paper_rotorpy_diagnostics import aggregate, diagnose_run, GATES, STATUS_LABELS, WAYPOINT_LABELS
from geometric_validation.paper_rotorpy_report import copy_dashboard_assets, csv_write
from geometric_validation.paper_topp import SpeedReference, dense_progress, flat_thrust, path_values
from geometric_validation.planners import norm_peak
from geometric_validation.rotorpy_results import audit_samples, load
from geometric_validation.runtime import digest, write_json
from geometric_validation.toppquad import UPSTREAM_COMMIT


ARM = 'flat_stable'
POLICY = dict(speed_cap_mps=4.2, acceleration_cap_mps2=4.9, reference_tilt_deg=30.,
              motor_margin_n=.01, retry_multiplier=1.03, max_margin_steps=30,
              closed_loop_tilt_retries=3)


def qualify_reference(frozen, policy, controller, yaw, rotor_limit, min_alpha=1.):
    """Scale one common clock and verify reference conditions before flight."""
    base = frozen['base']
    nodes = frozen['source_nodes_s']
    rates = frozen['squared_path_rate']
    if frozen['alpha'] != 1. or not np.allclose(rates, 1., atol=0., rtol=0.):
        raise ValueError('Qualification requires the original flatness timing')
    source = SpeedReference(base, nodes, rates)
    probes = dense_progress(source, subdivisions=24)
    original = enu(source.at_progress(probes))
    durations = base['segment_durations']
    coeff = base['coefficients']
    speed = norm_peak(np.asarray(coeff), durations, 1)
    acceleration = norm_peak(np.asarray(coeff), durations, 2)
    alpha = max(min_alpha, speed / policy['speed_cap_mps'], np.sqrt(acceleration / policy['acceleration_cap_mps2']))
    for step in range(policy['max_margin_steps'] + 1):
        flats = original / alpha**np.arange(5)[None, :, None]
        force = flat_thrust(flats, controller, yaw)
        specific_force = flats[:, 2] + [0., 0., controller.g]
        tilt = np.degrees(np.arctan2(np.linalg.norm(specific_force[:, :2], axis=1), specific_force[:, 2]))
        min_force, max_force, max_tilt = float(force.min()), float(force.max()), float(tilt.max())
        margin = policy['motor_margin_n']
        if margin <= min_force and max_force <= rotor_limit - margin and max_tilt <= policy['reference_tilt_deg']:
            sample_speed = np.linalg.norm(flats[:, 1], axis=1)
            return SpeedReference(base, nodes, rates, alpha), probes, flats, force, dict(
                step=step, reference_tilt_peak_deg=max_tilt, reference_speed_exact_mps=float(speed/alpha),
                reference_acceleration_exact_mps2=float(acceleration/alpha**2),
                reference_speed_peak_mps=float(sample_speed.max()), reference_thrust_min_n=min_force,
                reference_thrust_max_n=max_force, reference_rotor_violation_n=0.,
                collective_peak_n=float(force.sum(axis=1).max()),
            )
        alpha *= policy['retry_multiplier']
    raise ValueError('Time dilation exhausted the reference margin search')


def run_candidate(folder, prior, entry, method, design, min_alpha):
    folder.mkdir(parents=True, exist_ok=True)
    row = dict(route=entry['id'], method=method, arm=ARM, passed=False, status='error')
    try:
        source = load(prior/'reference.json')
        controller = WORKER['control'](WORKER['params'])
        cap = WORKER['params']['k_eta'] * WORKER['params']['rotor_speed_max']**2
        ref, probes, flats, force, stats = qualify_reference(source, design['reference_policy'], controller,
                                                            np.pi/2-source['yaw'], cap, min_alpha)
        reference = dict(source, alpha=ref.alpha, segment_durations=ref.segment_durations.tolist(),
                         planning=dict(source_sha256=digest(prior/'reference.json'), policy=design['reference_policy'],
                                       reference_checks=stats))
        write_json(folder/'reference.json', reference)
        ts = ref.time_at(probes) * ref.alpha
        order = np.argsort(ts, kind='stable')
        metrics = np.column_stack((ts, flats[:, 0], np.linalg.norm(flats[:, 1], axis=1), force))
        np.savetxt(folder/'reference_metrics.csv', metrics[order], delimiter=',', comments='',
                   header='time_s,p_e_m,p_n_m,p_u_m,speed_mps,f1_n,f2_n,f3_n,f4_n', fmt='%.10g')
        cfg = deepcopy(WORKER['template'])
        cfg.update(start=source['points'][0], yaw=source['yaw'], cases={method:dict(frozen_reference=reference)})
        row.update(**stats, alpha=ref.alpha, interpolation=ref.interpolation, reference_s=ref.duration,
                   reference_speed_passed=bool(stats['reference_speed_peak_mps'] <= design['vmax_mps']+.01),
                   reference_rotor_feasible=True)
        row.update(simulate(ref, cfg, method, folder, sampler=lambda r,t:enu(r.sample_batch(t))))
        row.update(status=row['sim_status'],
                   passed=bool(row['gate_passed'] and row['reference_speed_passed'] and row['reference_rotor_feasible']))
        audit_samples(folder, dict(row, accepted=row['gate_passed'], plan_issues=[]),
                      dict(design, hz=design['rate_hz']), reference)
        diagnostics = diagnose_run(folder, row, design, enu(entry['points_ned']))
    except (AssertionError, ValueError, OSError, KeyError, RuntimeError) as error:
        row.update(status='error', passed=False, error=f'{type(error).__name__}: {error}')
        diagnostics = dict(route=row['route'], method=method, arm=ARM, passed=False,
                           status='planning_or_logging_error', failed_gates=[], waypoint_states=[])
    write_json(folder/'result.json', row)
    return row, diagnostics


def evaluate(source_root, output, entry, method, design):
    relative = Path(entry['id']) / method
    prior = source_root / relative / 'flat'
    folder = output / relative / ARM
    row, diagnostics = run_candidate(folder, prior, entry, method, design, 1.)
    if diagnostics['failed_gates'] == ['tilt'] and design['reference_policy']['closed_loop_tilt_retries']:
        history = folder/'timing_attempts'
        history.mkdir()
        initial = history/'attempt_000'
        initial.mkdir()
        for record in list(folder.iterdir()):
            if record.is_file():
                record.replace(initial/record.name)
        for attempt in range(1, design['reference_policy']['closed_loop_tilt_retries']+1):
            min_alpha = row['alpha'] * design['reference_policy']['retry_multiplier']
            candidate = history/f'attempt_{attempt:03}'
            row, diagnostics = run_candidate(candidate, prior, entry, method, design, min_alpha)
            if diagnostics['failed_gates'] != ['tilt']:
                break
        for record in candidate.iterdir():
            if record.is_file():
                shutil.copyfile(record,folder/record.name)
        row['closed_loop_retiming_attempts'] = attempt
        write_json(folder/'result.json', row)
    return load(prior/'result.json'), row, diagnostics


def write_progress(output, observations):
    rows = [row for pair in observations for row in pair[:2]]
    temp = output/'progress.tmp'
    write_json(temp, rows)
    temp.replace(output/'progress.json')
    return rows


def path_geometry(source, design):
    """Measure deviation from each finite waypoint-to-waypoint chord in NED."""
    rows = []
    for entry in design['routes']:
        points = np.asarray(entry['points_ned'])
        chords = np.diff(points, axis=0)
        lengths = np.linalg.norm(chords, axis=1)
        cosine = np.sum(chords[:-1]*chords[1:], axis=1) / (lengths[:-1]*lengths[1:])
        turn = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))).max(initial=0.))
        for method in design['methods']:
            raw = load(source/entry['id']/method/'flat'/'reference.json')
            ref = SpeedReference(raw['base'], raw['source_nodes_s'], raw['squared_path_rate'])
            probes = np.sort(dense_progress(ref, subdivisions=24))
            curve = path_values(ref.path, probes)[:, 0]
            gaps = []
            for i, (left, right) in enumerate(zip(ref.path.times, ref.path.times[1:])):
                local = curve[(probes >= left) & (probes <= right)] - points[i]
                fraction = np.clip(local @ chords[i] / lengths[i]**2, 0., 1.)
                gaps.append(float(np.linalg.norm(local - fraction[:, None]*chords[i], axis=1).max()))
            arc = np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()
            rows.append(dict(route=entry['id'], method=method, maximum_turn_deg=turn,
                             maximum_chord_distance_m=max(gaps), arc_to_chords_ratio=float(arc/lengths.sum())))
    return rows


def summarize(output, observations, design):
    rows = write_progress(output, observations)
    csv_write(output/'runs.csv', rows)
    data = [item[2] for item in observations]
    fields = ('status','failed_gates','waypoint_states','first_position_over_s','first_attitude_over_s',
              'first_tilt_over_s','first_command_over_s')
    detailed = {'/'.join(row[k] for k in ('route','method','arm')):{key:row.get(key) for key in fields} for row in data}
    groups = aggregate(data, dict(design, arms=[ARM]))
    write_json(output/'failure_diagnostics.json', dict(groups=groups, gates=GATES, statuses=STATUS_LABELS,
                                                       waypoint_states=WAYPOINT_LABELS, runs=detailed))
    write_json(output/'summary.json', dict(groups=groups, paired=[]))
    policy = design['reference_policy']
    lines = ['# RotorPy 基线飞行核验', '',
             f"本批复用种子 {design['seed']} 的冻结航点与原始平坦性几何。"
             '各段共用一个时间倍率，系数、几何、闭环控制器和任务门槛一致。'
             f"参考先按 {policy['speed_cap_mps']:g} m/s 峰速与 {policy['acceleration_cap_mps2']:g} m/s² 峰值加速度分配时间，"
             f"再逐样本检查 {policy['reference_tilt_deg']:g}° 参考倾角和 {policy['motor_margin_n']:g} N 电机推力余量。"
             '速度与加速度用多项式驻点求极值，倾角与推力以分段稠密采样验收。', '',
             '| 构造 | 全任务通过 | 正常停稳 | 完成时间中位数 s | 位置峰值中位数 m | 姿态误差峰值最大值 ° | 实际倾角最大值 ° |',
             '|---|---:|---:|---:|---:|---:|---:|']
    pairs = []
    for group in groups:
        method = group['method']
        run = [row for row in rows if row['arm'] == ARM and row['method'] == method]
        done = [r['complete_s'] for r in run if r.get('complete_s') is not None]
        time = f'{median(done):.2f}' if done else '—'
        pos = group.get('position_peak_m', {}).get('median', float('nan'))
        att = group.get('attitude_peak_deg', {}).get('max', float('nan'))
        tilt = group.get('tilt_peak_deg', {}).get('max', float('nan'))
        lines.append(f"| {method} | {group['passed']}/{group['attempts']} | {group['statuses'].get('completed',0)} | "
                     f"{time} | {pos:.3f} | {att:.2f} | {tilt:.2f} |")
        for old, new, _ in observations:
            if new['method'] != method:
                continue
            pair = dict(route=new['route'], method=method, old_passed=old['passed'], stable_passed=new['passed'])
            for key in ('reference_s','complete_s','position_rmse_m','position_peak_m','attitude_peak_deg','tilt_peak_deg'):
                if old.get(key) is not None and new.get(key) is not None:
                    pair['delta_'+key] = new[key] - old[key]
            pairs.append(pair)
    failures = [row for row in data if not row['passed']]
    if failures:
        lines.extend(['','## 失败记录',''])
        for row in failures:
            gates = '、'.join(GATES[key] for key in row['failed_gates'])
            lines.append(f"- {row['route']} / {row['method']} · {gates or STATUS_LABELS[row['status']]}")
    first = next((pair for pair in observations if pair[1]['route'] == 'route_001' and pair[1]['method'] == 'bezier'), None)
    if first and all('position_peak_m' in row for row in first[:2]):
        old, new = first[:2]
        lines.extend(['','## 同一路线回放','',
                      f"route_001 的 Bézier 参考时长为 {old['reference_s']:.2f} → {new['reference_s']:.2f} s，"
                      f"位置跟踪峰值为 {old['position_peak_m']:.3f} → {new['position_peak_m']:.3f} m，"
                      f"实际倾角峰值为 {old['tilt_peak_deg']:.1f}° → {new['tilt_peak_deg']:.1f}°。"])
    geometry = path_geometry((output/design['source_path']).resolve(), design)
    csv_write(output/'path_geometry.csv', geometry)
    example = max(geometry, key=lambda row:row['maximum_chord_distance_m'])
    turns = {row['route']:row['maximum_turn_deg'] for row in geometry}
    over_90 = sum(turn > 90. for turn in turns.values())
    bspline_overshoot = sum(row['method'] == 'bspline' and row['maximum_chord_distance_m'] > 1. for row in geometry)
    lines.extend(['','## 参考几何','',
                  f"按相邻航段向量计算，{over_90}/{len(turns)} 条路线有转角超过 90°，最大约 {max(turns.values()):.1f}°。"
                  f"B 样条里有 {bspline_overshoot} 条参考曲线与对应航点线段的距离超过 1 m。"
                  f"偏离最大的 {example['route']} 约 {example['maximum_chord_distance_m']:.2f} m，"
                  f"曲线弧长约为航点连线总长的 {example['arc_to_chords_ratio']:.2f} 倍。"
                  '这些是冻结参考的稠密采样结果。重新计时只改善跟踪，绕行仍由原曲线几何决定。'])
    lines.extend(['', '## 证据范围', '',
                  '本批参数在同一批路线的对照试飞后选定，结论属于该批的 RotorPy 开发核验。'
                  '表格使用全部尝试作分母，原始平坦性记录从冻结批次引用。'
                  '实际完成时间在正常结束时记录，原组提前停止时的时长独立保存。'])
    if policy.get('closed_loop_tilt_retries'):
        retried = [row for row in rows if row.get('closed_loop_retiming_attempts')]
        lines.append(f"如果初次闭环仅倾角超限，按预先设定的倍率再减速，最多重试 {policy['closed_loop_tilt_retries']} 次。"
                     f"有 {len(retried)} 个任务触发重试，共执行 {sum(row['closed_loop_retiming_attempts'] for row in retried)} 次额外闭环。"
                     '逐次原始证据保存在对应航路的 timing_attempts。')
    lines.extend(['','数据与模型均来自 RotorPy。Codex 生成并独立重算新飞行记录。',''])
    (output/'conclusion.md').write_text('\n'.join(lines), encoding='utf-8')
    csv_write(output/'waypoints.csv', [dict(route=row['route'], method=row['method'], arm=row['arm'], **p)
                                      for row in rows for p in row.get('passages', [])])
    csv_write(output/'paired.csv', pairs)
    csv_write(output/'qualification_diagnostics.csv', [
        {key:'|'.join(value) if isinstance(value,list) else value for key,value in row.items()} for row in data])
    retried = [row for row in rows if row.get('closed_loop_retiming_attempts')]
    write_json(output/'audit.json', dict(verified_new_flights=sum(row['status']!='planning_or_logging_error' for row in data),
                                       attempts=len(data), passed=sum(row['passed'] for row in data),
                                       first_passed=sum(row['passed'] for row in data)-sum(row['passed'] for row in retried),
                                       additional_runs=sum(row['closed_loop_retiming_attempts'] for row in retried),
                                       errors=[r for r in failures if not r['failed_gates']]))


def collect_existing(output, source, design):
    pairs = []
    for entry in design['routes']:
        for method in design['methods']:
            relative = Path(entry['id'])/method
            folder = output/relative/ARM
            row = load(folder/'result.json')
            if row.get('samples') and row['status'] != 'error':
                audit_samples(folder, dict(row, accepted=row['gate_passed'], plan_issues=[]),
                              dict(design, hz=design['rate_hz']), load(folder/'reference.json'))
            pairs.append((load(source/relative/'flat'/'result.json'), row,
                          diagnose_run(folder, row, design, enu(entry['points_ned']))))
    return pairs


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--reference-speed-cap', type=float, default=POLICY['speed_cap_mps'])
    parser.add_argument('--reference-acceleration-cap', type=float, default=POLICY['acceleration_cap_mps2'])
    parser.add_argument('--summarize-existing', action='store_true')
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    baseline = load(source/'design.json')
    if args.summarize_existing:
        design = load(output/'design.json')
        if design['source_design_sha256'] != digest(source/'design.json'):
            raise ValueError('Original campaign design changed')
        summarize(output, collect_existing(output, source, design), design)
        return
    if args.jobs < 1 or not 0. < args.reference_speed_cap <= baseline['vmax_mps'] \
            or not np.isfinite(args.reference_acceleration_cap) or args.reference_acceleration_cap <= 0.:
        parser.error('Use positive jobs, a reference speed within the original cap, and positive finite acceleration')
    upstream = ROOT/'.runtime/toppquad_upstream'
    rotorpy = upstream/'rotorpy'
    for checkout, commit, key in ((rotorpy, ROTORPY_COMMIT, 'rotorpy_commit'), (upstream, UPSTREAM_COMMIT, 'toppquad_commit')):
        if baseline[key] != commit or subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'], text=True).strip() != commit:
            raise ValueError(f'Experiment revision mismatch: {checkout}')
    template = json.loads(PLANNER_CONFIG.read_text(encoding='utf-8-sig'))
    template['limits'] = baseline['limits']
    template['comparison']['passage'] = baseline['passage']
    template['execution']['terminal_hold_s'] = baseline['terminal_hold_s']
    source_path = Path(os.path.relpath(source, output)).as_posix()
    design = dict(baseline, experiment_type='baseline_qualification', arms=['flat',ARM],
                  default_arm=ARM, labels=['原始平坦性轨迹','稳健计时基线'], source_path=source_path,
                  source_design_sha256=digest(source/'design.json'),
                  reference_policy=dict(POLICY, speed_cap_mps=args.reference_speed_cap,
                                        acceleration_cap_mps2=args.reference_acceleration_cap),
                  qualification_source_sha256=digest(Path(__file__)))
    output.mkdir(parents=True)
    write_json(output/'design.json', design)
    copy_dashboard_assets(output)
    write_progress(output, [])
    observations = []
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker, initargs=(rotorpy,upstream,template)) as pool:
        jobs = [pool.submit(evaluate, source, output, entry, method, design)
                for entry in design['routes'] for method in design['methods']]
        for future in as_completed(jobs):
            observations.append(future.result())
            if len(observations) % 15 == 0 or len(observations) == len(jobs):
                write_progress(output, observations)
                passed = sum(pair[1]['passed'] for pair in observations)
                print(f'{len(observations)}/{len(jobs)} new flights, {passed} passed', flush=True)
    observations.sort(key=lambda pair:(pair[1]['route'], design['methods'].index(pair[1]['method'])))
    summarize(output, observations, design)


if __name__ == '__main__':
    main()
