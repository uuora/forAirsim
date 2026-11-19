"""Complete the seven-arm paper comparison with a common flown heading."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from geometric_validation.heading_toppquad import RouteHeading, enu, identity_reference, sample_heading
from geometric_validation.mapped_reference import MappedReference
from geometric_validation.paper_rotorpy_diagnostics import diagnose_run
from geometric_validation.paper_rotorpy_report import copy_dashboard_assets
from geometric_validation.paper_topp import ARMS, LABELS, SpeedReference, dense_progress, flat_thrust
from geometric_validation.rotorpy_results import audit_samples, load
from geometric_validation.runtime import digest, write_json
from run_rotorpy_heading_toppquad import write_progress
from run_rotorpy_waypoint_comparison import PLANNER_CONFIG, ROOT, WORKER, simulate


def init_replay_worker(rotorpy, template):
    sys.path.insert(0, str(rotorpy))
    from rotorpy.controllers.quadrotor_control import SE3Control
    from rotorpy.vehicles.crazyflie_params import quad_params
    from rotorpy.vehicles.multirotor import Multirotor

    WORKER.update(template=template, vehicle=Multirotor, control=SE3Control, params=quad_params)


def heading_at_progress(reference, heading, progress):
    h = reference.profile(progress)
    return heading.sample(progress, np.sqrt(h) / reference.alpha,
                          .5 * reference.profile(progress, 1) / reference.alpha**2)


def speed_heading(reference, heading, times):
    progress = reference.progress_at_time(times)
    yaw, rate, _ = heading_at_progress(reference, heading, progress)
    rate[(times < 0.) | (times >= reference.duration)] = 0.
    return yaw, rate


def uniform_motor_dilation(flats, angles, controller, cap):
    def feasible(factor):
        yaw, rate, acceleration = angles
        force = flat_thrust(flats / factor**np.arange(5)[None, :, None], controller,
                            yaw, rate / factor, acceleration / factor**2)
        return force.min() >= -1.e-8 and force.max() <= cap + 1.e-8

    if feasible(1.):
        return 1.
    low, high = 1., 1.1
    while not feasible(high):
        low, high = high, 1.1 * high
        if high > 32.:
            raise ValueError('No motor-feasible uniform dilation found')
    for _ in range(24):
        middle = (low + high) / 2.
        if feasible(middle):
            high = middle
        else:
            low = middle
    return high


def heading_evidence(folder, yaw, design, *, write):
    trace = np.atleast_1d(np.genfromtxt(folder/'simulation.csv.gz', names=True, delimiter=','))
    t = trace['time_s']
    command, rate = yaw(t)
    quaternion = np.column_stack([trace['q'+axis] for axis in 'xyzw'])
    body_x = Rotation.from_quat(quaternion).as_matrix()[:, :2, 0]
    actual = np.arctan2(body_x[:, 1], body_x[:, 0])
    difference = actual - command
    error = np.degrees(np.arctan2(np.sin(difference), np.cos(difference)))
    peak = int(np.argmax(np.abs(error)))
    data = np.column_stack((t, np.degrees(command), np.degrees(rate), np.degrees(actual), error))
    if write:
        np.savetxt(folder/'heading.csv', data, delimiter=',', fmt='%.10g', comments='',
                   header='time_s,desired_yaw_deg,desired_yaw_rate_degps,actual_heading_deg,heading_error_deg')
    else:
        saved = np.loadtxt(folder/'heading.csv', delimiter=',', skiprows=1, usecols=range(5))
        np.testing.assert_allclose(saved, data, atol=2.e-6, rtol=0.)
    return dict(heading_peak_deg=float(abs(error[peak])), heading_peak_time_s=float(t[peak]),
                heading_rmse_deg=float(np.sqrt(np.mean(error**2))),
                heading_passed=bool(abs(error[peak]) <= design['heading_policy']['heading_error_limit_deg']))


def finish(folder, row, design, frozen, heading_checks):
    row.update(heading_checks)
    audit_samples(folder, dict(row, accepted=row['gate_passed'], plan_issues=[]),
                  dict(design, hz=design['rate_hz']), frozen)
    row['passed'] = row['gate_passed'] and row['reference_rotor_feasible'] and row['reference_speed_passed']
    detail = diagnose_run(folder, row, design, enu(np.asarray(frozen['points'])))
    row['passed'] = row['passed'] and row['heading_passed']
    if not row['heading_passed']:
        detail['failed_gates'].append('heading')
    detail['passed'] = row['passed']
    return detail


def fly_paper_arm(folder, row, design, source, heading, controller, cap, provenance):
    base, nodes, h = (source[key] for key in ('base', 'source_nodes_s', 'squared_path_rate'))
    ref = SpeedReference(base, nodes, h)
    bounds = np.unique(np.r_[dense_progress(ref), heading.source_times,
                            np.nextafter(heading.source_times[1:], -np.inf)])
    flats = enu(ref.at_progress(bounds))
    angles = heading_at_progress(ref, heading, bounds)
    alpha = uniform_motor_dilation(flats, angles, controller, cap) if row['arm'].endswith('_alpha') else 1.
    if alpha > 1.:
        ref = SpeedReference(base, nodes, h, alpha)
        flats = enu(ref.at_progress(bounds))
        angles = heading_at_progress(ref, heading, bounds)
    force = flat_thrust(flats, controller, *angles)
    speed = np.linalg.norm(flats[:, 1], axis=1)
    times = ref.time_at(bounds) * alpha
    yaw_deg, rate_deg, _ = np.degrees(angles)
    metrics = np.column_stack((times, flats[:, 0], speed, yaw_deg, rate_deg, force))
    np.savetxt(folder/'reference_metrics.csv', metrics, delimiter=',', fmt='%.10g', comments='',
               header='time_s,p_e_m,p_n_m,p_u_m,speed_mps,yaw_deg,yaw_rate_degps,f1_n,f2_n,f3_n,f4_n')
    receipt = dict(status='source_topp_with_common_heading', alpha=alpha, **provenance)
    frozen = dict(source, format='spatial_rate_route_heading_v1', alpha=alpha, planning=receipt,
                  segment_durations=ref.segment_durations.tolist(),
                  heading=dict(source_breaks=heading.source_times.tolist(), headings_enu_rad=heading.angles.tolist(),
                               interpolation='clamped_cubic_bearings'))
    write_json(folder/'reference.json', frozen)
    write_json(folder/'planning.json', receipt)
    cfg = dict(WORKER['template'], start=source['points'][0], yaw=source['yaw'],
               cases={row['arm']: dict(frozen_reference=frozen)})
    yaw = lambda t: speed_heading(ref, heading, t)
    row.update(simulate(ref, cfg, row['arm'], folder, sampler=lambda r, t: enu(r.sample_batch(t)), heading=yaw))
    low, high, peak = float(force.min()), float(force.max()), float(speed.max())
    row.update(status=row['sim_status'], planning_status=receipt['status'], alpha=alpha, reference_s=ref.duration,
               reference_speed_peak_mps=peak, reference_speed_passed=peak <= design['vmax_mps'] + .01,
               reference_thrust_min_n=low, reference_thrust_max_n=high,
               reference_rotor_violation_n=max(0., -low, high-cap),
               reference_rotor_feasible=low >= -1.e-6 and high <= cap + 1.e-6,
               collective_peak_n=float(force.sum(axis=1).max()))
    return finish(folder, row, design, frozen, heading_evidence(folder, yaw, design, write=True))


def baseline(folder, row, design, original, heading, controller, cap):
    source = load(original)
    provenance = dict(source_reference_sha256=digest(original), original_fixed_heading_alpha=source['alpha'])
    return fly_paper_arm(folder, row, design, source, heading, controller, cap, provenance)


def restore_toppquad(folder, row, design, origin):
    final_row = load(origin/'result.json')
    first = origin/'timing_attempts'/'attempt_000' if final_row['retries'] else origin
    primary = load(first/'result.json')
    files = ('reference.json', 'reference_metrics.csv', 'actuator_reference.csv', 'simulation.csv.gz', 'heading.csv')
    for name in files:
        shutil.copyfile(first/name, folder/name)
    frozen = load(folder/'reference.json')
    ref = MappedReference(frozen['base'], frozen['segment_durations'], frozen['clock_bernstein'])
    heading = RouteHeading(ref)
    checks = heading_evidence(folder, lambda t: sample_heading(ref, heading, t), design, write=False)
    np.testing.assert_allclose(checks['heading_peak_deg'], primary['heading_peak_deg'], atol=1.e-6, rtol=0.)
    row.update(primary, arm='toppquad', final_retry_passed=final_row['passed'], retry_count=final_row['retries'])
    write_json(folder/'planning.json', dict(load(origin/'planning.json'), primary_result_sha256=digest(first/'result.json'),
                                          primary_directory=first.relative_to(ROOT).as_posix(),
                                          development_result_directory=origin.relative_to(ROOT).as_posix()))
    return finish(folder, row, design, frozen, checks), heading, frozen['base']


def evaluate(source, output, design, entry, method):
    relative = Path(entry['id'])/method
    origin = source/'heading_toppquad_full'/relative/'heading_toppquad'
    controller = WORKER['control'](WORKER['params'])
    cap = WORKER['params']['k_eta'] * WORKER['params']['rotor_speed_max']**2
    records, details = [], []
    for arm in ('toppquad', *ARMS[:-1]):
        folder = output/relative/arm
        folder.mkdir(parents=True, exist_ok=True)
        row = dict(route=entry['id'], method=method, arm=arm, status='error', passed=False)
        if arm == 'toppquad':
            detail, heading, geometry = restore_toppquad(folder, row, design, origin)
        else:
            original = source/relative/arm/'reference.json'
            if load(original)['base'] != geometry:
                raise ValueError(f'Source geometry differs at {relative}/{arm}')
            detail = baseline(folder, row, design, original, heading, controller, cap)
        write_json(folder/'diagnostics.json', detail)
        write_json(folder/'result.tmp', row)
        (folder/'result.tmp').replace(folder/'result.json')
        records.append(row)
        details.append(detail)
    return records, details


def read_completed(output, entry, method, arms=ARMS):
    folders = [output/entry['id']/method/arm for arm in arms]
    if all((folder/'result.json').is_file() and (folder/'diagnostics.json').is_file() for folder in folders):
        return [[load(folder/file) for folder in folders] for file in ('result.json', 'diagnostics.json')]
    return None


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routes', nargs='+')
    parser.add_argument('--jobs', type=int, default=8)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    original = load(source/'design.json')
    development = source/'heading_toppquad_full'/'design.json'
    design = load(development)
    for key in ('routes', 'methods', 'rate_hz', 'limits', 'passage', 'vmax_mps', 'terminal_hold_s',
                'rotorpy_commit', 'toppquad_commit'):
        if original[key] != design[key]:
            raise ValueError(f'Original and development campaigns differ in {key}')
    if args.routes:
        design['routes'] = [entry for entry in design['routes'] if entry['id'] in args.routes]
    if not design['routes'] or args.jobs < 1:
        parser.error('At least one matching route and one worker are required')
    design.update(arms=list(ARMS), labels=[*LABELS[:-1], 'TOPPQuad 连续重优化'], default_arm='toppquad',
                  experiment_type='paper_heading', original_design_sha256=digest(source/'design.json'),
                  heading_design_sha256=digest(development),
                  comparison_retry_policy='first_attempt_only',
                  current_source_sha256={p.name: digest(p) for p in (Path(__file__),
                      ROOT/'python/geometric_validation/paper_topp.py', ROOT/'python/run_rotorpy_waypoint_comparison.py',
                      ROOT/'python/geometric_validation/heading_toppquad.py')})
    template = load(PLANNER_CONFIG)
    template['limits'] = design['limits']
    template['comparison']['passage'] = design['passage']
    template['execution']['terminal_hold_s'] = design['terminal_hold_s']
    upstream = ROOT/'.runtime/toppquad_upstream'
    for path, field in ((upstream, 'toppquad_commit'), (upstream/'rotorpy', 'rotorpy_commit')):
        if subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() != design[field]:
            raise ValueError(f'Unexpected repository revision at {path}')
    output.mkdir(parents=True, exist_ok=True)
    design_path = output/'design.json'
    if design_path.exists():
        if load(design_path) != design:
            raise ValueError('Output already has a different experimental design')
    else:
        write_json(design_path, design)
    copy_dashboard_assets(output)
    rows, details = [], []
    pending = []
    for task in design['routes']:
        for method in design['methods']:
            completed = read_completed(output, task, method)
            if completed is None:
                pending.append((task, method))
            else:
                rows.extend(completed[0])
                details.extend(completed[1])
    write_progress(output, rows)
    total = len(design['routes']) * len(design['methods']) * len(ARMS)
    print(f'{len(rows)}/{total} records ready; running {len(pending)} route/construction groups', flush=True)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_replay_worker,
                             initargs=(upstream/'rotorpy', template)) as pool:
        jobs = [pool.submit(evaluate, source, output, design, task, method) for task, method in pending]
        for index, job in enumerate(as_completed(jobs), 1):
            records, diagnostics = job.result()
            rows.extend(records)
            details.extend(diagnostics)
            if index % 4 == 0 or index == len(jobs):
                write_progress(output, rows)
                print(f'{len(rows)}/{total} records / {perf_counter()-start:.0f}s', flush=True)
    write_json(output/'finished.json', dict(attempts=len(rows), new_simulations=len(rows)*6//7, reused=len(rows)//7,
                                          wall_s=perf_counter()-start))
    from geometric_validation.paper_heading_report import summarize
    summarize(output, design, rows, details)


if __name__ == '__main__':
    main()
