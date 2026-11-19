"""Compare yaw-aware RotorPy TOPPQuad refinement with the qualified baseline."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
from statistics import median
from time import sleep

from run_rotorpy_waypoint_comparison import PLANNER_CONFIG, ROOT, WORKER, enu, init_worker, simulate

import numpy as np
from scipy.spatial.transform import Rotation

from geometric_validation.heading_toppquad import (RouteHeading, identity_reference, optimize_reference,
                                                    sample_heading, validate_reference)
from geometric_validation.mapped_reference import MappedReference
from geometric_validation.paper_rotorpy_diagnostics import diagnose_run, GATES, STATUS_LABELS, WAYPOINT_LABELS
from geometric_validation.paper_rotorpy_report import copy_dashboard_assets, csv_write
from geometric_validation.rotorpy_results import audit_samples, load
from geometric_validation.runtime import digest, write_json


ARMS = ('heading_baseline', 'heading_toppquad')
LABELS = ('稳健计时＋航向跟踪', 'TOPPQuad 连续重优化＋航向跟踪')
EXAMPLE_ROUTES = ('route_001', 'route_012', 'route_031', 'route_036', 'route_079', 'route_088')
POLICY = dict(heading_rate_radps=np.pi, reference_waypoint_speed_mps=.50, heading_error_limit_deg=10.)
RETRY_DILATIONS = (1.05, 1.10, 1.16)
RETRYABLE_GATES = {'heading', 'attitude', 'tilt', 'envelope'}


def measure_heading(folder, reference, heading, design):
    data = np.genfromtxt(folder/'simulation.csv.gz', names=True, delimiter=',')
    def heading_of(prefix):
        quat = np.column_stack([data[prefix+'q'+axis] for axis in 'xyzw'])
        body_x = Rotation.from_quat(quat).as_matrix()[:, :2, 0]
        return np.arctan2(body_x[:, 1], body_x[:, 0])
    desired, yaw_rate = sample_heading(reference, heading, data['time_s'])
    actual = heading_of('')
    nominal = heading_of('reference_')
    wrap = lambda angle: np.arctan2(np.sin(angle), np.cos(angle))
    error = np.degrees(wrap(actual - desired))
    nominal_error = np.degrees(wrap(actual - nominal))
    index = int(np.argmax(np.abs(error)))
    t = data['time_s']
    motion = (t <= reference.duration) & (np.hypot(data['ref_ve_mps'],data['ref_vn_mps']) >= .5)
    velocity_heading = np.arctan2(data['ref_vn_mps'],data['ref_ve_mps'])
    tangent_error = np.degrees(np.abs(wrap(actual[motion]-velocity_heading[motion])))
    np.savetxt(folder/'heading.csv', np.column_stack((t, np.degrees(desired), np.degrees(yaw_rate),
                                                       np.degrees(actual), error, nominal_error)),
               header='time_s,desired_yaw_deg,desired_yaw_rate_degps,actual_heading_deg,heading_error_deg,nominal_heading_error_deg',
               comments='', delimiter=',', fmt='%.10g')
    peak = float(np.abs(error).max())
    return dict(heading_peak_deg=peak, heading_peak_time_s=float(t[index]),
                heading_rmse_deg=float(np.sqrt(np.mean(error**2))),
                heading_passed=bool(peak <= design['heading_policy']['heading_error_limit_deg']),
                tangent_heading_p90_deg=float(np.percentile(tangent_error,90)) if len(tangent_error) else None,
                heading_turn_deg=float(np.ptp(np.degrees(desired))))


def execute(folder, arm, reference, heading, source, checks, design, row, controller, rotor_limit):
    _, angles, speed, force, check = validate_reference(reference, heading, controller, design['reference_policy'], rotor_limit)
    minimum = float(force.min())
    maximum = float(force.max())
    policy = design['reference_policy']
    envelope = (minimum >= policy['motor_margin_n']-1.e-6 and maximum <= rotor_limit-policy['motor_margin_n']+1.e-6
                and check['reference_tilt_peak_deg'] <= policy['reference_tilt_deg']+1.e-5
                and check['reference_heading_rate_peak_degps'] <= np.degrees(policy['heading_rate_radps'])+1.e-5
                and check['speed_cap_passed'] and check['monotonicity_passed'] and check['minimum_planning_margin'] >= -1.e-6)
    cfg = deepcopy(WORKER['template'])
    case = arm
    frozen = dict(format='bernstein_route_heading_v1', method=source['method'], yaw=source['yaw'],
                  points=source['points'], segment_durations=reference.segment_durations.tolist(), base=source,
                  clock_bernstein=reference.clocks.tolist(),
                  heading=dict(source_breaks=heading.source_times.tolist(), headings_enu_rad=heading.angles.tolist(),
                               interpolation='clamped_cubic_bearings'), planning=checks)
    cfg.update(start=source['points'][0], yaw=source['yaw'], cases={case:dict(frozen_reference=frozen)})
    write_json(folder/'reference.json', frozen)
    timeline = np.unique(np.r_[np.linspace(0.,reference.duration,800), reference.times])
    trajectory = enu(reference.sample_batch(timeline))
    yaw, yaw_rate = sample_heading(reference, heading, timeline)
    np.savetxt(folder/'reference_metrics.csv', np.column_stack((timeline,trajectory[:,0],
                                                                np.linalg.norm(trajectory[:,1],axis=1),
                                                                np.degrees(yaw),np.degrees(yaw_rate))),
               header='time_s,p_e_m,p_n_m,p_u_m,speed_mps,yaw_deg,yaw_rate_degps', comments='', delimiter=',', fmt='%.10g')
    np.savetxt(folder/'actuator_reference.csv', np.column_stack((np.repeat(np.arange(len(reference.times)-1), len(speed)//(len(reference.times)-1)),
                                                                  speed,np.degrees(angles[0]),force)),
               header='segment,reference_speed_mps,desired_yaw_deg,f1_n,f2_n,f3_n,f4_n',
               comments='', delimiter=',', fmt='%.10g')
    result = simulate(reference,cfg,case,folder,sampler=lambda r,t:enu(r.sample_batch(t)),
                      heading=lambda t:sample_heading(reference,heading,t))
    row.update(result, status=result['sim_status'])
    row.update(reference_s=reference.duration, reference_speed_peak_mps=float(speed.max()),
               reference_speed_passed=bool(speed.max() <= design['vmax_mps'] + .01),
               reference_tilt_peak_deg=check['reference_tilt_peak_deg'],
               reference_rotor_feasible=bool(minimum >= -1.e-6 and maximum <= rotor_limit+1.e-6),
               reference_thrust_min_n=minimum, reference_thrust_max_n=maximum,
               reference_rotor_violation_n=max(0.,-minimum,maximum-rotor_limit),
               collective_peak_n=float(force.sum(axis=1).max()), envelope_passed=bool(envelope),
               **measure_heading(folder,reference,heading,design))
    row['passed'] = bool(row['gate_passed'] and row['reference_speed_passed'] and row['reference_rotor_feasible'])
    audit_samples(folder,dict(row,accepted=row['gate_passed'],plan_issues=[]),dict(design,hz=design['rate_hz']),frozen)
    details = diagnose_run(folder,row,design,enu(source['points']))
    row['passed'] = bool(row['passed'] and envelope and row['heading_passed'])
    details['passed'] = row['passed']
    if not row['heading_passed']:
        details['failed_gates'].append('heading')
    if not envelope:
        details['failed_gates'].append('envelope')
    return details


def execute_with_retries(folder, arm, candidate, heading, source, receipt, design, row, controller, cap):
    first = execute(folder, arm, candidate, heading, source, receipt, design, row, controller, cap)
    row.update(first_passed=row['passed'], retries=0, dilation=1.)
    gates = set(first['failed_gates'])
    if arm != ARMS[1] or not gates or not gates <= RETRYABLE_GATES:
        return first
    history = folder/'timing_attempts'
    initial = history/'attempt_000'
    initial.mkdir(parents=True)
    for name in ('reference.json','reference_metrics.csv','actuator_reference.csv','simulation.csv.gz','heading.csv'):
        (folder/name).replace(initial/name)
    write_json(initial/'result.json',dict(row, failed_gates=first['failed_gates']))
    attempts = []
    for index, factor in enumerate(RETRY_DILATIONS,1):
        trial_folder = history/f'attempt_{index:03}'
        trial_folder.mkdir()
        trial_reference = MappedReference(None, candidate.segment_durations*factor, candidate.clocks, geometry=candidate)
        trial = {key:row[key] for key in ('route','method','arm','planning_status')}
        checks = dict(receipt, dilation=factor)
        detail = execute(trial_folder,arm,trial_reference,heading,source,checks,design,trial,controller,cap)
        write_json(trial_folder/'result.json',dict(trial,failed_gates=detail['failed_gates']))
        attempts.append(dict(index=index,dilation=factor,passed=trial['passed'],failed_gates=detail['failed_gates']))
        if trial['passed'] or not set(detail['failed_gates']) <= RETRYABLE_GATES:
            break
    for record in trial_folder.iterdir():
        if record.name != 'result.json':
            shutil.copyfile(record,folder/record.name)
    row.update(trial, first_passed=False, retries=len(attempts), dilation=factor, retry_results=attempts)
    return detail


def evaluate(source, output, entry, method, design):
    relative = Path(entry['id'])/method
    qualified = source/'baseline_verified_v2'/relative/'flat_stable'
    prior = load(qualified/'result.json')
    if not prior['passed']:
        raise ValueError(f'Unqualified baseline {relative}')
    frozen = load(qualified/'reference.json')
    base = identity_reference(frozen['base'], np.asarray(frozen['segment_durations']))
    heading = RouteHeading(base)
    controller = WORKER['control'](WORKER['params'])
    cap = WORKER['params']['k_eta']*WORKER['params']['rotor_speed_max']**2
    raw_path = source/relative/'toppquad'/'raw_solution.json'
    raw = load(raw_path) if raw_path.is_file() else None
    rows, receipts = [], []
    for arm in ARMS:
        folder = output/relative/arm
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(route=entry['id'],method=method,arm=arm,passed=False,status='error')
        detail = dict(status='planning_or_logging_error',failed_gates=[],waypoint_states=[])
        try:
            if arm == 'heading_baseline':
                candidate, receipt = base, dict(status='qualified_baseline_clock', source_reference_sha256=digest(qualified/'reference.json'))
            else:
                candidate, receipt = optimize_reference(base,heading,controller,design['reference_policy'],cap,raw)
                if raw is not None:
                    receipt['raw_solution_sha256'] = digest(raw_path)
            write_json(folder/'planning.json',receipt)
            row['planning_status'] = receipt['status']
            if candidate is None:
                row.update(status='planning_rejected',error=receipt['status'])
                detail.update(status='planning_rejected',failed_gates=['envelope'])
            else:
                detail = execute_with_retries(folder,arm,candidate,heading,frozen['base'],receipt,design,row,controller,cap)
        except (ValueError, RuntimeError, OSError, AssertionError) as error:
            row.update(passed=False,status='error',error=f'{type(error).__name__}: {error}')
        write_json(folder/'result.json',row)
        rows.append(row)
        receipts.append(detail)
    return rows, receipts


def write_progress(output, rows):
    pending = output/'progress.tmp'
    write_json(pending,rows)
    for attempt in range(12):
        try:
            pending.replace(output/'progress.json')
            return
        except PermissionError:
            if attempt == 11:
                raise
            # Windows may hold the previous JSON open briefly while HTTP serves it.
            sleep(.075)


def summarize(output, design, rows, receipts):
    rows.sort(key=lambda row:(row['route'],row['method'],row['arm']))
    csv_write(output/'runs.csv',rows)
    csv_write(output/'waypoints.csv',[dict(route=r['route'],method=r['method'],arm=r['arm'],**p)
                                      for r in rows for p in r.get('passages',[])])
    gate_labels = dict(GATES,heading='机头跟踪',envelope='规划余量')
    groups, pairs = [], []
    lines = ['# RotorPy 航向跟踪与 TOPPQuad 连续重优化','',
             '使用已核验的路径和时间作为基线。比较两种计时方式时共用航向参考、控制器和全部闭环门槛。'
             '机头采用水平路径切向方位的平滑插值，低速段跨样本过渡，起点和终点保持航向。'
             '保存的 TOPPQuad 离散解用于初值，SLSQP 在同一几何上调整分段时长和节点速率，'
             '用 RotorPy 电机分配和与航向一致的姿态逆动力学核对速度、加速度、倾角、航向速率和单电机推力余量。'
             '电机与姿态约束是节点间稠密采样核验。', '',
             '重优化参考若只在姿态、实际倾角、机头或规划余量上失败，按固定倍率重跑，保留全部尝试。', '',
             '| 构造 | 原时间＋新航向通过 | 重优化首次通过 | 重优化最终通过 | 双方通过 | 完成差中位数 s | 位置 RMSE 差中位数 mm |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method in design['methods']:
        method_rows = [r for r in rows if r['method']==method]
        indexed = {(r['route'],r['arm']):r for r in method_rows}
        jointly = []
        for entry in design['routes']:
            old, new = [indexed[entry['id'],arm] for arm in ARMS]
            record = dict(route=entry['id'],method=method,baseline_passed=old['passed'],toppquad_passed=new['passed'])
            for key in ('complete_s','reference_s','position_rmse_m','heading_rmse_deg','heading_peak_deg'):
                if old.get(key) is not None and new.get(key) is not None:
                    record['delta_'+key] = new[key]-old[key]
            pairs.append(record)
            if old['passed'] and new['passed']:
                jointly.append(record)
        for arm in ARMS:
            cohort = [r for r in method_rows if r['arm']==arm]
            failures = {key:sum(key in receipts[r['route'],method,arm]['failed_gates'] for r in cohort) for key in gate_labels}
            groups.append(dict(method=method,arm=arm,attempts=len(cohort),passed=sum(r['passed'] for r in cohort),failures=failures,
                               first_passed=sum(r.get('first_passed',False) for r in cohort),
                               retries=sum(r.get('retries',0) for r in cohort),
                               command_outside=sum(r.get('thrust_violation_peak_n',0.) > 1.e-6 for r in cohort)))
        old, new = groups[-2:]
        delta = f"{median(r['delta_complete_s'] for r in jointly):+.2f}" if jointly else '—'
        rmse = f"{1.e3*median(r['delta_position_rmse_m'] for r in jointly):+.2f}" if jointly else '—'
        lines.append(f"| {method} | {old['passed']}/{old['attempts']} | {new['first_passed']}/{new['attempts']} | "
                     f"{new['passed']}/{new['attempts']} | {len(jointly)} | {delta} | {rmse} |")
    csv_write(output/'paired.csv',pairs)
    fields = ('status','failed_gates','waypoint_states','first_position_over_s','first_attitude_over_s',
              'first_tilt_over_s','first_command_over_s')
    details = {'/'.join(key):{field:r.get(field) for field in fields} for key,r in receipts.items()}
    write_json(output/'failure_diagnostics.json',dict(groups=groups,gates=gate_labels,
               statuses=dict(STATUS_LABELS,planning_rejected='参考校验未通过'),waypoint_states=WAYPOINT_LABELS,runs=details))
    write_json(output/'summary.json',dict(groups=groups,paired=[]))
    total_ok = sum(row['passed'] for row in rows if row['arm']==ARMS[1])
    lines.extend(['',f"重优化通过 {total_ok}/{len(rows)//2} 项，共执行 {sum(r.get('retries',0) for r in rows)} 次额外闭环。"
                  '规划或闭环失败的尝试保留原始回执。'
                  '机头误差比较水平投影的机体 +X 与航向命令，全程门槛为 10°。'
                  '低速急转处平滑过渡，局部机头方向可与曲线速度方向不同。'
                  '仿真结果限于当前 RotorPy 航路和控制器。','',
                  'Codex 实现开发版本、运行仿真与核验。',''])
    (output/'conclusion.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(output/'audit.json',dict(attempts=len(rows), audited_closed_loop=sum('heading_peak_deg' in r for r in rows),
                                       errors=[r for r in rows if r['status']=='error']))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--routes',nargs='+',default=EXAMPLE_ROUTES)
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--jobs',type=int,default=6)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    original = load(source/'design.json')
    rotorpy = ROOT/'.runtime/toppquad_upstream/rotorpy'
    upstream = rotorpy.parent
    qualified = load(source/'baseline_verified_v2'/'design.json')
    policy = dict(qualified['reference_policy'],heading_rate_radps=POLICY['heading_rate_radps'],
                  reference_waypoint_speed_mps=POLICY['reference_waypoint_speed_mps'],source_nominal_mps=original['nominal_mps'])
    picked = [entry for entry in original['routes'] if args.all or entry['id'] in args.routes]
    if not picked or args.jobs < 1:
        parser.error('Choose at least one route and one worker')
    design = dict(original,routes=picked,arms=ARMS,labels=LABELS,default_arm=ARMS[0],experiment_type='heading_toppquad',
                  heading_policy=dict(POLICY,retry_dilations=RETRY_DILATIONS),reference_policy=policy,
                  design_parent_sha256=digest(source/'design.json'),
                  qualified_design_sha256=digest(source/'baseline_verified_v2'/'design.json'),
                  heading_code_sha256={p.name:digest(p) for p in (Path(__file__), ROOT/'python/geometric_validation/heading_toppquad.py',
                                                                 ROOT/'python/run_rotorpy_waypoint_comparison.py')})
    template = load(PLANNER_CONFIG)
    template['limits'] = design['limits']
    template['comparison']['passage'] = design['passage']
    template['execution']['terminal_hold_s'] = design['terminal_hold_s']
    output.mkdir(parents=True)
    write_json(output/'design.json',design)
    copy_dashboard_assets(output)
    write_progress(output, [])
    rows, receipts = [], {}
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=init_worker,initargs=(rotorpy,upstream,template)) as pool:
        jobs = [pool.submit(evaluate,source,output,entry,method,design) for entry in picked for method in design['methods']]
        for future in as_completed(jobs):
            record, detail = future.result()
            rows.extend(record)
            receipts.update({(r['route'],r['method'],r['arm']):info for r,info in zip(record,detail)})
            if len(rows)%12==0 or len(rows)==2*len(jobs):
                write_progress(output,rows)
                print(f'{len(rows)}/{len(jobs)*2} flight attempts recorded',flush=True)
    summarize(output,design,rows,receipts)


if __name__ == '__main__':
    main()
