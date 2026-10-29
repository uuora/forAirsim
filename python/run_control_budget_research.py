"""Exact sequential-case pruning pilot on the existing fixed controller proxy.

No rotor/attitude/wind model and no time-indexed Hermite tracking is introduced.
Case order learns a heuristic failure-per-second priority from observed cases.
It never decides feasibility: only mathematical score bounds can prune.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import time
import numpy as np
from dual_target_control.core import BezierParameters
from dual_target_control.trajectory_objective import evaluate_parameters, make_training_cases, make_validation_cases
from trajectory_research.search import run_search

ROOT = Path(__file__).resolve().parents[1]


def dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


class OrderedEvaluator:
    def __init__(self, adaptive):
        self.adaptive = adaptive
        self.cases = make_training_cases()
        n = len(self.cases)
        self.count = np.zeros(n)
        self.failures = np.zeros(n)
        self.seconds = np.zeros(n)
        self.ledger = []

    def __call__(self, parameters, cutoff):
        from trajectory_research.control_bounded import evaluate_control_bounded
        rate = (self.failures+1.)/(self.count+2.)
        cost = np.divide(self.seconds,self.count,out=np.ones_like(self.seconds),where=self.count>0)
        priority = rate/np.maximum(cost,1e-9)
        order = np.argsort(-priority,kind='stable').tolist() if self.adaptive else list(range(len(self.cases)))
        result = evaluate_control_bounded(parameters,cutoff=cutoff,cases=self.cases,case_order=order)
        for obs in result['case_observations']:
            i = obs['case_index']
            self.count[i] += 1
            self.failures[i] += not obs['passed']
            self.seconds[i] += obs['wall_s']
        self.ledger.append(dict(order=order,case_observations=result['case_observations'],
                               priority_before=priority.tolist(),completed_cases=result['completed_cases'],
                               status=result.get('status','EVALUATED')))
        return result


def run(args):
    from trajectory_research.screened_search import run_screened_search
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    snapshot=output/'source_snapshot'
    snapshot.mkdir()
    sources=[Path(__file__),ROOT/'python/trajectory_research/search.py',ROOT/'python/trajectory_research/screened_search.py',
             ROOT/'python/trajectory_research/control_bounded.py',ROOT/'python/dual_target_control/core.py',
             ROOT/'python/dual_target_control/trajectory_objective.py',ROOT/'python/validate_dual_target_turning.py']
    hashes={}
    for source in sources:
        shutil.copy2(source,snapshot/source.name)
        hashes[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    dump(output/'manifest.json',dict(args=vars(args),sources=hashes,platform=platform.platform(),
         protocol='fixed-controller-partial-case-pilot-v2-cached',scope='5D production planner parameters, fixed controller, first-order lagged point mass; no AirSim or real-flight validation',
         primary='same proposals and selected score; completed case rollouts and interleaved wall/CPU time',
         order_learning='empirical failure-frequency / observed average wall-time; censored heuristic, not calibrated probability',
         timing='warm Python process; all modes cache identical training cases; network construction included; validation and auditing excluded',
         validation='30 pre-existing regression/heldout conditions applied only after candidate selection; no selection based on them',
         novelty='exact bound pruning is standard; this pilot evaluates task-specific implementation, not priority claims'))
    import torch
    torch.set_num_threads(1)
    torch.zeros(1).sum()
    warm_tick=time.perf_counter()
    run_search(lambda x:dict(score=float(x@x),feasible=True),np.full(5,-1.),np.full(5,1.),
               method='abom',budget=32,upstream=args.upstream)
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    manifest['excluded_one_time_optimizer_warmup_s']=time.perf_counter()-warm_tick
    dump(output/'manifest.json',manifest)
    rows=[]
    lo,hi=BezierParameters.lower_bounds(),BezierParameters.upper_bounds()
    cached_cases=make_training_cases()
    objective=lambda x:evaluate_parameters(x,cases=cached_cases)
    # Warm the same complete controller rollout path before interleaved timings.
    objective(BezierParameters().to_vector())
    for method in args.methods:
        for seed in args.seeds:
            modes=['plain','fixed','adaptive']
            shift=seed%3
            order=modes[shift:]+modes[:shift]
            results={}
            for mode in order:
                scheduler=OrderedEvaluator(mode=='adaptive')
                options=dict(method=method,seed=seed,population=16,budget=args.budget,
                             anchor=BezierParameters().to_vector(),upstream=args.upstream)
                wall=time.perf_counter()
                cpu=time.process_time()
                if mode=='plain':
                    result=run_search(objective,lo,hi,**options)
                    cases=args.budget*len(cached_cases)
                else:
                    result=run_screened_search(objective,lo,hi,bounded_objective=scheduler,**options)
                    cases=sum(x['completed_cases'] for x in scheduler.ledger)
                result['outer_wall_s']=time.perf_counter()-wall
                result['outer_cpu_s']=time.process_time()-cpu
                result['completed_case_rollouts']=cases
                result['case_order_ledger']=scheduler.ledger
                results[mode]=result
            plain=results['plain']
            for mode in ['fixed','adaptive']:
                other=results[mode]
                assert plain['best_parameters']==other['best_parameters']
                assert plain['best_score']==other['best_score']
                assert [x['best'] for x in plain['history']]==[x['best'] for x in other['history']]
                for a,b in zip(plain['evaluations'],other['evaluations']):
                    assert a['parameters']==b['parameters'],'candidate sequence changed'
                    if b.get('status')=='PRUNED':
                        assert b['score'] is None and b['feasible'] is None
                        assert b['lower_bound']<=a['score']+1e-12
                    else:
                        assert a['score']==b['score'] and a['feasible']==b['feasible']
            # Exactly one validation of the shared incumbent per triple, not during search.
            validation=evaluate_parameters(plain['best_parameters'],cases=make_validation_cases())
            row=dict(method=method,seed=seed,budget=args.budget,best_score=plain['best_score'],
                     validation_feasible=validation['feasible'],validation_score=validation['score'],
                     equivalent=True,modes={k:dict(wall_s=v['outer_wall_s'],cpu_s=v['outer_cpu_s'],
                     case_rollouts=v['completed_case_rollouts']) for k,v in results.items()})
            dump(output/f'{method}_{seed}.json',dict(summary=row,validation=validation,**results))
            rows.append(row)
            dump(output/'rows.json',rows)
            print(json.dumps(row),flush=True)
    summary=[]
    for method in args.methods:
        group=[r for r in rows if r['method']==method]
        for mode in ['fixed','adaptive']:
            ratios=[r['modes'][mode]['wall_s']/r['modes']['plain']['wall_s'] for r in group]
            summary.append(dict(method=method,mode=mode,n=len(group),median_wall_ratio=float(np.median(ratios)),
                case_rollout_reduction=1-sum(r['modes'][mode]['case_rollouts'] for r in group)/sum(r['modes']['plain']['case_rollouts'] for r in group),
                validation_passed=sum(r['validation_feasible'] for r in group),all_equivalent=True))
    dump(output/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--seeds',nargs='+',type=int,default=[201,202,203,204])
    p.add_argument('--methods',nargs='+',default=['abom','abom_perturb','gaussian'])
    p.add_argument('--upstream',default=str(ROOT/'tmp/abom_deploy_probe/upstream'))
    run(p.parse_args())
