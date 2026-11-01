"""Paired same-capacity screening ablation; no AirSim connection or flights."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from control_target_experiments import make_case,objective,save,ROOT
from trajectory_research.control_target_bound import target_quality_lower_bound
from trajectory_research.screened_search import run_screened_search


def run(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'python/control_target_experiments.py',ROOT/'python/trajectory_research/control_target_bound.py',
        ROOT/'python/trajectory_research/screened_search.py',ROOT/'python/trajectory_research/timed_tracking.py']
    snapshot=output/'source_snapshot';snapshot.mkdir()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for p in sources:(snapshot/p.name).write_bytes(p.read_bytes())
    cases=[make_case(1101,[8,0,5]),make_case(1102),make_case(1105)]
    save(output/'precommit.json',dict(cases=cases,search_seeds=[13,31,47],population=20,proposals=80,
        modes=['paired perturbation ABOM','same adapter plus lower bound'],dropout=.95,
        scope='Paired equal-proposal diagnostic; not a time-budget study. First lazy Torch import is retained and flagged. No efficacy/novelty claim.'))
    pairs=[];first=True
    for case_index,case in enumerate(cases):
        boundary=case['boundary'];distance=np.linalg.norm(np.asarray(boundary['p1'])-boundary['p0'])
        anchor=[0,0,0,min(7.,max(5.5,distance/1.4+1))]
        for seed_index,seed in enumerate((13,31,47)):
            runs={};order=[False,True] if (case_index+seed_index)%2==0 else [True,False]
            for screened in order:
                callbacks=[]
                def evaluate(x):
                    result=objective(x,'hermite',boundary);callbacks.append(result);return result
                started=time.perf_counter()
                result=run_screened_search(evaluate,[-.8,-1.4,-.8,3],[.8,1.4,.8,7],method='abom_perturb',
                    seed=seed,population=20,budget=80,anchor=anchor,upstream=ROOT/'tmp/abom_deploy_probe/upstream',
                    lower_bound=(lambda x:target_quality_lower_bound(x,boundary)) if screened else None,
                    dropout=.95,injection_fraction=.5)
                elapsed=time.perf_counter()-started
                record=dict(case_index=case_index,seed=seed,screened=screened,wall_s=elapsed,
                    first_lazy_import_exposure=first,full_calls=len(callbacks),result=result)
                first=False;runs[screened]=record
                save(output/f'case{case_index}_seed{seed}_screen{int(screened)}.json',record)
                print(json.dumps(dict(case=case_index,seed=seed,screened=screened,calls=len(callbacks),wall_s=elapsed)),flush=True)
            a,b=runs[False]['result'],runs[True]['result']
            identical=np.array_equal([e['parameters'] for e in a['evaluations']],[e['parameters'] for e in b['evaluations']])
            if not identical or a['best_parameters']!=b['best_parameters'] or a['best_score']!=b['best_score']:
                raise AssertionError('screening changed the candidate stream or selected incumbent')
            pairs.append(dict(case_index=case_index,seed=seed,identical_stream=identical,
                proposals=80,baseline_full_calls=a['full_evaluations'],screened_full_calls=b['full_evaluations'],
                saved_full_calls=a['full_evaluations']-b['full_evaluations'],bound_calls=b['lower_bound_calls'],
                baseline_wall_s=runs[False]['wall_s'],screened_wall_s=runs[True]['wall_s'],
                cold_import_exposure=runs[False]['first_lazy_import_exposure'] or runs[True]['first_lazy_import_exposure']))
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=h for p,h in [(p,hashes[str(p)]) for p in sources]):
        raise RuntimeError('covered source changed during study')
    summary=dict(status='COMPLETE',pairs=pairs,source_sha256=hashes,all_streams_identical=True,
        total_baseline_full_calls=sum(p['baseline_full_calls'] for p in pairs),
        total_screened_full_calls=sum(p['screened_full_calls'] for p in pairs),
        airsim_calls=0,scope='Surrogate computation only; no claim of reduced AirSim calls or equal-wall-time superiority')
    save(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
