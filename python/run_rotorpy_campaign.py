"""Fly one frozen random batch with the stable baseline and three timing methods."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from geometric_validation.heading_toppquad import RouteHeading, enu, identity_reference, optimize_reference, validate_reference
from geometric_validation.paper_heading_report import summarize
from geometric_validation.paper_rotorpy_report import copy_dashboard_assets
from geometric_validation.paper_topp import optimize_speed, path_values, routes
from geometric_validation.planners import construct
from geometric_validation.reference import FrozenReference
from geometric_validation.rotorpy_results import load
from geometric_validation.runtime import digest, write_json
from geometric_validation.toppquad import UPSTREAM_COMMIT, load_solver, solve_path
from run_rotorpy_baseline_qualification import POLICY as STABLE_POLICY, qualify_reference
from run_rotorpy_heading_toppquad import POLICY as HEADING_POLICY, execute, write_progress
from run_rotorpy_paper_heading import fly_paper_arm, init_replay_worker, read_completed
from run_rotorpy_waypoint_comparison import METHODS, PLANNER_CONFIG, RATE, ROOT, ROTORPY_COMMIT, SETTLE_ALLOWANCE_S, WORKER


STABLE = 'heading_baseline'
ARMS = (STABLE, 'velocity', 'collective', 'toppquad')
LABELS = ('稳健计时＋航向跟踪', '速度 TOPP', '速度＋总推力 TOPP', 'TOPPQuad 连续重优化')
NOMINAL = 5.
NODES = 100
V_MAX = 5.


def init_campaign_worker(upstream, template, expected):
    init_replay_worker(upstream/'rotorpy', template)
    WORKER['solver'], _ = load_solver(upstream, checked_sha256=expected)


def frozen_geometry(entry, method):
    points = np.asarray(entry['points_ned'])
    durations = np.linalg.norm(np.diff(points, axis=0), axis=1) / NOMINAL
    source = dict(format='normalized_power_v1', method=method, points=points.tolist(), yaw=0.,
                  segment_durations=durations.tolist(), coefficients=construct(method, points, durations).tolist())
    return source, durations, np.linspace(0., durations.sum(), NODES)


def paper_reference(base, nodes, h):
    return dict(base=base, method=base['method'], points=base['points'], yaw=base['yaw'],
                source_nodes_s=nodes.tolist(), squared_path_rate=h.tolist(), alpha=1.)


def stable_reference(source, heading, design, controller, cap):
    policy = design['reference_policy']
    reference, _, _, _, qualification = qualify_reference(source, policy, controller, np.pi/2, cap)
    alpha = reference.alpha
    for _ in range(policy['max_margin_steps']+1):
        clock = identity_reference(source['base'], np.asarray(source['base']['segment_durations'])*alpha)
        _, _, _, _, check = validate_reference(clock, heading, controller, policy, cap)
        if check['minimum_planning_margin'] >= -1.e-6:
            return clock, dict(status='qualified_stable_clock', alpha=alpha, **qualification)
        alpha *= policy['retry_multiplier']
    raise ValueError('Stable reference failed common-heading qualification')


def raw_solution(folder, source, durations):
    slow = dict(source, segment_durations=(durations*NOMINAL).tolist())
    # Capture the pinned solver output alongside the unmodified node solution.
    with (folder/'solver.log').open('w', encoding='utf-8') as stream, redirect_stdout(stream):
        _, timestamps, seconds, solution = solve_path(WORKER['solver'], FrozenReference(slow), V_MAX, NODES)
    raw = dict(raw_times_s=timestamps.tolist(), raw_h=np.asarray(solution['h']).ravel().tolist(),
               raw_rotor_thrust_n=np.asarray(solution['T']).tolist(),
               raw_q_wxyz=np.asarray(solution['q']).tolist(), raw_body_rates=np.asarray(solution['w']).tolist(),
               solver_wall_s=seconds)
    write_json(folder/'raw_solution.json', raw)
    return raw


def publish(folder, row, detail):
    write_json(folder/'diagnostics.json', detail)
    write_json(folder/'result.tmp', row)
    (folder/'result.tmp').replace(folder/'result.json')


def evaluate(output, design, entry, method):
    base, durations, nodes = frozen_geometry(entry, method)
    original = paper_reference(base, nodes, np.ones(NODES))
    heading = RouteHeading(identity_reference(base, durations))
    controller = WORKER['control'](WORKER['params'])
    cap = WORKER['params']['k_eta'] * WORKER['params']['rotor_speed_max']**2
    path = enu(path_values(FrozenReference(base), nodes))
    rows, details = [], []
    steady = None
    for arm in design['arms']:
        folder = output/entry['id']/method/arm
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'result.json').is_file() and (folder/'diagnostics.json').is_file():
            if arm == STABLE and (folder/'reference.json').is_file():
                saved = load(folder/'reference.json')
                steady = identity_reference(base, np.asarray(saved['segment_durations']))
            rows.append(load(folder/'result.json'))
            details.append(load(folder/'diagnostics.json'))
            continue
        row = dict(route=entry['id'], method=method, arm=arm, status='planning_rejected', passed=False,
                   reference_rotor_feasible=False, reference_speed_passed=False)
        detail = dict(route=entry['id'], method=method, arm=arm, status='planning_rejected', passed=False,
                      failed_gates=['planning'], waypoint_states=[])
        try:
            if arm in (STABLE, 'toppquad'):
                if arm == STABLE:
                    steady, receipt = stable_reference(original, heading, design, controller, cap)
                    reference = steady
                else:
                    if steady is None:
                        raise ValueError('Stable reference unavailable for TOPPQuad initialization')
                    raw = raw_solution(folder, base, durations)
                    reference, receipt = optimize_reference(steady, heading, controller, design['reference_policy'], cap, raw)
                    receipt['raw_solution_sha256'] = digest(folder/'raw_solution.json')
                    receipt['stable_reference_sha256'] = digest(folder.parent/STABLE/'reference.json')
                write_json(folder/'planning.json', receipt)
                row['planning_status'] = receipt['status']
                if reference is None:
                    row['error'] = receipt['status']
                else:
                    detail = execute(folder, arm, reference, heading, base, receipt, design, row, controller, cap)
            else:
                max_collective = 4*cap/controller.mass if arm == 'collective' else None
                rate, iterations = optimize_speed(nodes, path, V_MAX, max_collective)
                detail = fly_paper_arm(folder, row, design, paper_reference(base, nodes, rate), heading,
                                       controller, cap, dict(solver_iterations=iterations))
        except (ArithmeticError, RuntimeError, ValueError, AssertionError, OSError) as error:
            if (folder/'reference.json').exists() or isinstance(error, (AssertionError, OSError)):
                row['status'] = detail['status'] = 'error'
                detail['failed_gates'] = ['audit_or_execution']
            row.update(passed=False, error=f'{type(error).__name__}: {error}')
        publish(folder, row, detail)
        rows.append(row)
        details.append(detail)
    return rows, details


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'reports/rotorpy_current')
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--pilot', action='store_true', help='Fly the first, 36th and 79th frozen routes')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    upstream = ROOT/'.runtime/toppquad_upstream'
    commit = subprocess.check_output(['git', '-C', str(upstream/'rotorpy'), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != ROTORPY_COMMIT:
        raise ValueError('RotorPy source differs from the pinned revision')
    _, solver_hash = load_solver(upstream)
    template = load(PLANNER_CONFIG)
    template['comparison']['passage']['position_tolerance_m'] = .3
    template['execution']['terminal_hold_s'] = 1.
    design_path = output/'design.json'
    source_files = [Path(__file__), *[ROOT/p for p in (
        'python/geometric_validation/heading_toppquad.py', 'python/geometric_validation/paper_topp.py',
        'python/run_rotorpy_paper_heading.py', 'python/run_rotorpy_baseline_qualification.py',
        'python/run_rotorpy_heading_toppquad.py', 'python/run_rotorpy_waypoint_comparison.py')]]
    config = dict(experiment_type='paper_heading', independent_random=True, methods=list(METHODS),
                  arms=list(ARMS), labels=list(LABELS),
                  default_arm=STABLE, nodes=NODES, nominal_mps=NOMINAL, vmax_mps=V_MAX, rate_hz=RATE,
                  limits=template['limits'], passage=template['comparison']['passage'], terminal_hold_s=1.,
                  settle_extra_s=SETTLE_ALLOWANCE_S, comparison_retry_policy='first_attempt_only',
                  rotorpy_commit=ROTORPY_COMMIT, toppquad_commit=UPSTREAM_COMMIT, upstream_source_sha256=solver_hash,
                  heading_policy=HEADING_POLICY, reference_policy=dict(STABLE_POLICY, **HEADING_POLICY, source_nominal_mps=NOMINAL),
                  source_sha256={p.name: digest(p) for p in source_files})
    if design_path.exists():
        design = load(design_path)
        if any(design.get(key) != value for key,value in config.items()):
            raise ValueError('Cannot resume a different frozen experiment')
    else:
        design = dict(config, routes=list(routes(None, 100)), created_utc=datetime.now(timezone.utc).isoformat())
        write_json(design_path, design)
    copy_dashboard_assets(output)
    tasks, rows, diagnostics = [], [], []
    for idx, entry in enumerate(design['routes'], 1):
        for method in design['methods']:
            existing = read_completed(output, entry, method, design['arms'])
            if existing:
                rows.extend(existing[0])
                diagnostics.extend(existing[1])
            elif not args.pilot or idx in (1, 36, 79):
                tasks.append((entry, method))
    write_progress(output, rows)
    total = len(design['routes'])*len(design['methods'])*len(design['arms'])
    print(f'{len(rows)}/{total} records; {len(tasks)} route/construction groups pending', flush=True)
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_campaign_worker,
                             initargs=(upstream, template, solver_hash)) as workers:
        futures = [workers.submit(evaluate, output, design, entry, method) for entry, method in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            outcomes, evidence = future.result()
            rows.extend(outcomes)
            diagnostics.extend(evidence)
            if i % 4 == 0 or i == len(tasks):
                try:
                    write_progress(output, rows)
                except PermissionError as error:
                    print(f'Progress display delayed: {error}', flush=True)
                print(f'{len(rows)}/{total} records · {perf_counter()-start:.0f}s', flush=True)
    if len(rows) == total:
        summarize(output, design, rows, diagnostics)
        write_json(output/'finished.json', dict(attempts=len(rows), new_simulations=sum(bool(r.get('samples')) for r in rows),
                                               wall_s=perf_counter()-start))


if __name__ == '__main__':
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        os.environ.setdefault(name, '1')
    main()
