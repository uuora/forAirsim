"""Fresh-process soft-wall-budget comparison of the control surrogate oracle.

Process startup/import/initialization/bounds/training and full evaluation work
consume the deadline. Calls and library imports are non-preemptible: all
overruns are retained, while primary quality uses only feasible evaluations
completed by the deadline. This is not a hard real-time implementation.
"""
import argparse,hashlib,json,math,os,platform,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=('random','gaussian','coordinate','abom','abom_perturb','abom_perturb_screened')


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def worker(args):
    # This start was measured by the parent before creating this fresh process.
    started=args.start;deadline=started+args.budget
    import numpy as np
    from control_target_experiments import objective,PROXY_CONDITION,PROXY_CONTROLLER
    from dataclasses import asdict
    from trajectory_research.control_target_bound import target_quality_lower_bound
    from trajectory_research.screened_search import run_screened_search
    case=json.loads(args.case.read_text(encoding='utf-8'));boundary=case['boundary']
    lower=np.array([-.8,-1.4,-.8,3.]);upper=np.array([.8,1.4,.8,7.])
    distance=np.linalg.norm(np.asarray(boundary['p1'])-boundary['p0'])
    anchor=np.array([0.,0.,0.,min(7.,max(5.5,distance/1.4+1))])
    calls=[];bounds=[];status='COMPLETED_PROPOSAL_CAP';exception=None
    journal=getattr(args,'journal',None)
    def event(value):
        if journal:
            with Path(journal).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,allow_nan=False)+'\n');stream.flush()
    class DeadlineReached(Exception):pass
    def check():
        if time.perf_counter()>=deadline:raise DeadlineReached()
    def evaluate(x):
        check();index=len(calls)+1
        event(dict(kind='BEGIN',index=index,parameters=list(map(float,x)),host_s=time.perf_counter()-started))
        begin=time.perf_counter();result=objective(x,'hermite',boundary);end=time.perf_counter()
        record=dict(index=index,parameters=list(map(float,x)),score=result['score'],feasible=result['feasible'],
            start_s=begin-started,finish_s=end-started,callback_wall_s=end-begin,
            completed_by_deadline=end<=deadline,metrics=result['metrics'])
        event(dict(kind='COMPLETE',record=record))
        published=time.perf_counter()
        event(dict(kind='PUBLISHED',index=index,ready_s=published-started))
        calls.append(record)
        return result
    def bound(x):
        check();tick=time.perf_counter();value=target_quality_lower_bound(x,boundary)
        bounds.append(dict(parameters=list(map(float,x)),bound=value,wall_s=time.perf_counter()-tick))
        return value
    imports_and_setup_s=time.perf_counter()-started
    try:
        check()
        if args.method=='coordinate':
            rng=np.random.default_rng(args.seed);points=rng.uniform(-1,1,(20,4));points[0]=2*(anchor-lower)/(upper-lower)-1
            evaluated=[]
            for point in points:
                x=lower+(point+1)*.5*(upper-lower);evaluated.append((evaluate(x)['score'],point.copy()))
            best=min(evaluated,key=lambda r:r[0]);step=.3
            while True:
                old=best[0]
                for axis in range(4):
                    for direction in (-1,1):
                        p=best[1].copy();p[axis]=np.clip(p[axis]+direction*step,-1,1)
                        x=lower+(p+1)*.5*(upper-lower);score=evaluate(x)['score']
                        if score<best[0]:best=(score,p)
                if best[0]>=old:step=max(.005,step*.6)
        else:
            run_screened_search(evaluate,lower,upper,method='abom_perturb' if args.method=='abom_perturb_screened' else args.method,
                seed=args.seed,population=20,budget=200000,anchor=anchor,dropout=.95,injection_fraction=.5,
                upstream=ROOT/'tmp/abom_deploy_probe/upstream',lower_bound=bound if args.method=='abom_perturb_screened' else None)
    except DeadlineReached:status='SOFT_DEADLINE_REACHED'
    except Exception as exc:status='FAILED';exception=f'{type(exc).__name__}: {exc}'
    eligible=[r for r in calls if r['completed_by_deadline'] and r['feasible']]
    best=min(eligible,key=lambda r:r['score']) if eligible else None
    finished=time.perf_counter();evaluation_wall=sum(r['callback_wall_s'] for r in calls)
    result=dict(status=status,error=exception,method=args.method,seed=args.seed,budget_s=args.budget,case=case,
        primary_best_feasible_score=None if best is None else best['score'],primary_best_parameters=None if best is None else best['parameters'],
        primary_has_feasible=best is not None,full_oracle_calls=len(calls),complete_calls_by_deadline=sum(r['completed_by_deadline'] for r in calls),
        completed_after_deadline=sum(not r['completed_by_deadline'] for r in calls),
        imports_and_setup_s=imports_and_setup_s,planner_elapsed_before_publish_s=finished-started,
        overrun_s=max(0.,finished-deadline),full_oracle_wall_s=evaluation_wall,
        nonoracle_overhead_s=max(0.,finished-started-evaluation_wall),lower_bound_calls=len(bounds),
        lower_bound_wall_s=sum(r['wall_s'] for r in bounds),calls=calls,bounds=bounds,
        controller_model=dict(condition=asdict(PROXY_CONDITION),extensions=PROXY_CONTROLLER),
        airsim_calls=0,scope='Fresh-process non-preemptive soft deadline; finite-prefix anytime quality, not hard real-time or physical-flight evidence')
    save(args.output,result)
    return 0 if status!='FAILED' else 1


def run(args):
    # Require a shared host monotonic clock before passing deadlines across processes.
    before=time.perf_counter()
    probe=subprocess.run([sys.executable,'-c','import time;print(time.perf_counter())'],capture_output=True,text=True,check=True)
    after=time.perf_counter();child=float(probe.stdout)
    if not before<=child<=after:raise RuntimeError('host perf_counter is not shared across processes')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'python/control_target_experiments.py',ROOT/'python/trajectory_research/timed_tracking.py',
        ROOT/'python/trajectory_research/screened_search.py',ROOT/'python/trajectory_research/search.py',
        ROOT/'python/trajectory_research/control_target_bound.py',ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for p in sources:(snapshot/p.name).write_bytes(p.read_bytes())
    import random
    trials=[dict(method=m,budget=b,seed=s,case=str(c.resolve())) for c in args.cases for b in args.budgets for s in args.seeds for m in METHODS]
    random.Random(20261030).shuffle(trials)
    save(args.output/'precommit.json',dict(trials=trials,common_initial_population=20,
        clock_cross_process_check=dict(before=before,child=child,after=after),source_sha256=hashes,
        python=sys.executable,platform=platform.platform(),cpu_count=os.cpu_count(),
        scope='Primary quality from feasible full calls completed by deadline including process startup, imports, initialization and optimization. All overruns/failures retained. No AirSim calls.'))
    rows=[]
    for i,trial in enumerate(trials):
        output=args.output/f'trial_{i:03d}.json';log=args.output/f'trial_{i:03d}.log'
        started=time.perf_counter()
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--method',trial['method'],'--seed',str(trial['seed']),
            '--budget',str(trial['budget']),'--start',str(started),'--case',trial['case'],'--output',str(output)]
        with log.open('wb') as stream:
            process=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        host_elapsed=time.perf_counter()-started
        if output.is_file():
            record=json.loads(output.read_text(encoding='utf-8'))
            row={k:v for k,v in record.items() if k not in ('calls','bounds','case','controller_model')}
            row.update(trial=trial,index=i,host_total_wall_s=host_elapsed,worker_returncode=process.returncode)
        else:row=dict(trial=trial,index=i,status='WORKER_FAILED_NO_RESULT',host_total_wall_s=host_elapsed,worker_returncode=process.returncode)
        rows.append(row);save(args.output/'rows.json',rows)
        print(json.dumps(dict(index=i,method=trial['method'],budget=trial['budget'],seed=trial['seed'],
            status=row['status'],score=row.get('primary_best_feasible_score'),calls=row.get('full_oracle_calls'),overrun_s=row.get('overrun_s'))),flush=True)
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)] for p in sources):raise RuntimeError('source changed during benchmark')
    save(args.output/'completion.json',dict(status='COMPLETE',trials=len(rows),source_sha256=hashes,airsim_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--cases',type=Path,nargs='+',default=[ROOT/'reports/control_targets_fixed_matched_20261030_v3/case.json'])
    p.add_argument('--budgets',type=float,nargs='+',default=[2.,5.]);p.add_argument('--seeds',type=int,nargs='+',default=[13,31,47])
    p.add_argument('--method',choices=METHODS);p.add_argument('--budget',type=float);p.add_argument('--start',type=float)
    p.add_argument('--seed',type=int);p.add_argument('--case',type=Path)
    p.add_argument('--journal',type=Path)
    a=p.parse_args();sys.exit(worker(a)) if a.worker else run(a)
