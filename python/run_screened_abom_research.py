"""Paired, interleaved timing of exact rejection before expensive evaluation.

This is an offline geometry benchmark, not a UAV flight or tracking test.
Raw and screened runs share proposals; selection equality is audited, not assumed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import time
import numpy as np

from trajectory_research.curves import PolynomialSegment
from trajectory_research.curve_task import jerk_energy
from trajectory_research.parameterizations import curve_from_parameters, parameter_bounds, default_parameters
from trajectory_research.perturbation_task import evaluate_parameters, shape_rms
from trajectory_research.search import run_search

ROOT = Path(__file__).resolve().parents[1]


def quality_lower_bound(parameters, family):
    """Same quality arithmetic as oracle, minus an explicit numerical cushion.

    Score = Q if feasible and >= 1+Q otherwise. Thus Q is a lower bound
    irrespective of certificates. This bound does not establish feasibility.
    The cushion is an engineering tolerance, not interval arithmetic.
    """
    curve = curve_from_parameters(parameters, family=family)
    if curve.degree < 7:
        curve = PolynomialSegment(np.pad(curve.coefficients, ((0, 7-curve.degree), (0, 0))), curve.duration_s)
    energy = jerk_energy(curve)
    deformation = shape_rms(parameters, family=family)
    t = curve.duration_s
    return float(.9*t/(t+5.) + .05*energy/(energy+30.) + .05*deformation/(deformation+1.) - 1e-10)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def run(args):
    from trajectory_research.screened_search import run_screened_search
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), ROOT/'python/trajectory_research/screened_search.py',
               ROOT/'python/trajectory_research/search.py', ROOT/'python/trajectory_research/perturbation_task.py',
               ROOT/'python/trajectory_research/parameterizations.py', ROOT/'python/trajectory_research/curves.py',
               ROOT/'python/trajectory_research/curve_task.py', ROOT/'python/trajectory_research/constraints.py',
               ROOT/'python/trajectory_research/directional.py']
    if args.bound_mode == 'closed_form':
        sources.append(ROOT/'python/trajectory_research/quality_bound.py')
    snapshot = output/'source_snapshot'
    snapshot.mkdir()
    hashes = {}
    for source in sources:
        shutil.copy2(source, snapshot/source.name)
        hashes[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(output/'manifest.json', dict(protocol='paired-exact-screening-v1',
        args=vars(args), sources=hashes, python=platform.python_version(), platform=platform.platform(),
        scope='offline point-reference geometry; no controller, aircraft, wind, or flight validation',
        timer='warm Python process; includes optimizer initialization, lower bounds, objective and network work; excludes initial torch import and post-run audit',
        budgets='equal proposals, not equal full objective calls',
        pruning='LB >= worst retained elite; stable ties keep old parents; no feasibility imputed',
        novelty='engineering baseline; exact lower-bound elimination is not claimed as a new optimization principle'))
    # Import cost is outside both paired timings. Network construction stays inside.
    if any(m.startswith('abom') for m in args.methods):
        import torch
        torch.set_num_threads(1)
        torch.zeros(1).sum()
        warm_tick = time.perf_counter()
        run_search(lambda x: dict(score=float(x@x), feasible=True),
                   np.full(4,-1.),np.full(4,1.),method='abom',budget=32,upstream=args.upstream)
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        manifest['excluded_one_time_optimizer_warmup_s'] = time.perf_counter()-warm_tick
        write_json(output/'manifest.json',manifest)
    rows = []
    for family in args.families:
        lo, hi = parameter_bounds(family)
        for layout in args.layouts:
            for method in args.methods:
                for seed in args.seeds:
                    objective = lambda x: evaluate_parameters(x, family=family, layout=layout)
                    options = dict(method=method, seed=seed, population=16, budget=args.budget,
                                   anchor=default_parameters(family), upstream=args.upstream)
                    results = {}
                    for name in (('plain','screened') if seed % 2 else ('screened','plain')):
                        tick = time.perf_counter()
                        cpu = time.process_time()
                        if name == 'plain':
                            result = run_search(objective, lo, hi, **options)
                        else:
                            if args.bound_mode == 'closed_form':
                                from trajectory_research.quality_bound import quality_lower_bound as fast_bound
                                bound = lambda x: fast_bound(x, family=family)
                            else:
                                bound = lambda x: quality_lower_bound(x, family)
                            result = run_screened_search(objective, lo, hi,
                                lower_bound=bound, **options)
                        result['outer_wall_s'] = time.perf_counter()-tick
                        result['outer_cpu_s'] = time.process_time()-cpu
                        results[name] = result
                    plain, screened = results['plain'], results['screened']
                    assert plain['best_score'] == screened['best_score']
                    assert plain['best_parameters'] == screened['best_parameters']
                    assert len(plain['evaluations']) == len(screened['evaluations']) == args.budget
                    pruned = 0
                    for a, b in zip(plain['evaluations'], screened['evaluations']):
                        assert a['parameters'] == b['parameters'], 'candidate stream changed'
                        if b.get('status') == 'PRUNED':
                            pruned += 1
                            assert b['score'] is None and b['feasible'] is None
                            assert b['lower_bound'] <= a['score'] + 1e-14, 'invalid bound'
                        else:
                            assert a['score'] == b['score'] and a['feasible'] == b['feasible']
                    assert [h['best'] for h in plain['history']] == [h['best'] for h in screened['history']]
                    checked = objective(screened['best_parameters'])
                    assert checked['score'] == screened['best_score']
                    row = dict(family=family, layout=layout, method=method, seed=seed,
                        proposals=args.budget, full_evaluations_plain=args.budget,
                        full_evaluations_screened=args.budget-pruned, pruned=pruned,
                        best_score=plain['best_score'], feasible=checked['feasible'],
                        plain_wall_s=plain['outer_wall_s'], screened_wall_s=screened['outer_wall_s'],
                        plain_cpu_s=plain['outer_cpu_s'], screened_cpu_s=screened['outer_cpu_s'],
                        equivalent=True)
                    stem=f'{family}_{layout}_{method}_{seed}'
                    write_json(output/f'{stem}.json', dict(summary=row, **results))
                    rows.append(row)
                    write_json(output/'rows.json', rows)
                    print(json.dumps(row), flush=True)
    summary = []
    for family in args.families:
        for layout in args.layouts:
            for method in args.methods:
                group = [r for r in rows if (r['family'],r['layout'],r['method']) == (family,layout,method)]
                ratios = np.array([r['screened_wall_s']/r['plain_wall_s'] for r in group])
                rng = np.random.default_rng(20261029)
                boot = np.median(ratios[rng.integers(0,len(group),(5000,len(group)) )],axis=1)
                summary.append(dict(family=family,layout=layout,method=method,n=len(group),
                    median_wall_ratio=float(np.median(ratios)), exploratory_paired_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                    calls_saved_fraction=sum(r['pruned'] for r in group)/(len(group)*args.budget),
                    feasible=sum(r['feasible'] for r in group), all_equivalent=all(r['equivalent'] for r in group)))
    write_json(output/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget', type=int, default=256)
    parser.add_argument('--bound-mode', choices=['numerical','closed_form'], default='numerical')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(101,111)))
    parser.add_argument('--methods', nargs='+', default=['abom','abom_perturb','gaussian'])
    parser.add_argument('--families', nargs='+', default=['single_bump','paired_bump'])
    parser.add_argument('--layouts', nargs='+', default=['central','early_offset','two_obstacles'])
    parser.add_argument('--upstream', default=str(ROOT/'tmp/abom_deploy_probe/upstream'))
    run(parser.parse_args())
