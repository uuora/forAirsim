"""Parent-watchdog wall-budget research with durable complete incumbents.

Only the owned optimizer worker may be killed; no simulator is connected.
Every begin/complete/published event is preserved. The parent can return a
previously published feasible reference when a worker stalls in library code.
OS scheduling, process cleanup and publication still preclude hard real time.
"""
import argparse,hashlib,json,math,os,platform,random,subprocess,sys,time
from pathlib import Path
from run_control_wall_budget_research import METHODS,ROOT,save,worker


def read_incumbents(journal,budget):
    data=Path(journal).read_text(encoding='utf-8') if Path(journal).is_file() else ''
    raw=data.splitlines(keepends=True);events=[];truncated=False
    for i,line in enumerate(raw):
        if not line.endswith('\n'):
            if i!=len(raw)-1:raise ValueError('nonterminal incomplete journal line')
            truncated=True;break
        events.append(json.loads(line))
    begins={};complete={};published={}
    for event in events:
        if event['kind']=='BEGIN':
            if event['index'] in begins:raise ValueError('duplicate oracle begin')
            begins[event['index']]=event
        elif event['kind']=='COMPLETE':
            r=event['record'];index=r['index']
            if index not in begins or index in complete or r['parameters']!=begins[index]['parameters']:
                raise ValueError('complete call not bound to unique begin')
            if not math.isfinite(r['score']) or not isinstance(r['feasible'],bool):raise ValueError('invalid complete score')
            complete[index]=r
        elif event['kind']=='PUBLISHED':
            index=event['index']
            if index not in complete or index in published or event['ready_s']<complete[index]['finish_s']:
                raise ValueError('invalid incumbent publication order')
            published[index]=event['ready_s']
        else:raise ValueError('unknown journal event')
    eligible=[r for i,r in complete.items() if r['feasible'] and i in published and published[i]<=budget]
    best=min(eligible,key=lambda r:r['score']) if eligible else None
    return dict(best=best,started_calls=len(begins),completed_calls=len(complete),
        started_without_complete=len(begins)-len(complete),published_calls=len(published),
        published_by_deadline=sum(v<=budget for v in published.values()),
        complete_after_deadline=sum(r['finish_s']>budget for r in complete.values()),
        incomplete_trailing_line=truncated,events=events)


def enforce(command,*,started,budget,journal,log):
    with Path(log).open('wb') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        cancelled=False
        try:process.wait(timeout=max(.001,started+budget-time.perf_counter()))
        except subprocess.TimeoutExpired:
            cancelled=True;process.kill();process.wait(timeout=10)
    # Recover the exact complete/published prefix even after an interrupted write.
    receipt=read_incumbents(journal,budget)
    published=time.perf_counter();best=receipt.pop('best')
    return dict(status='PARENT_DEADLINE_CANCELLED' if cancelled else 'WORKER_EXITED',
        worker_returncode=process.returncode,primary_has_feasible=best is not None,
        primary_best_feasible_score=None if best is None else best['score'],
        primary_best_parameters=None if best is None else best['parameters'],
        host_return_wall_s=published-started,parent_overrun_s=max(0.,published-started-budget),
        process_cancelled=cancelled,**receipt)


def run(args):
    before=time.perf_counter();p=subprocess.run([sys.executable,'-c','import time;print(time.perf_counter())'],capture_output=True,text=True,check=True)
    after=time.perf_counter();value=float(p.stdout)
    if not before<=value<=after:raise RuntimeError('cross-process monotonic clock unavailable')
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),ROOT/'python/run_control_wall_budget_research.py',ROOT/'python/control_target_experiments.py',
        ROOT/'python/trajectory_research/timed_tracking.py',ROOT/'python/trajectory_research/control_target_bound.py',
        ROOT/'python/trajectory_research/screened_search.py',ROOT/'python/trajectory_research/search.py',ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (args.output/'source_snapshot').mkdir()
    for p in sources:(args.output/'source_snapshot'/p.name).write_bytes(p.read_bytes())
    trials=[dict(method=m,budget=b,seed=s,case=str(c.resolve())) for c in args.cases for b in args.budgets for s in args.seeds for m in args.methods]
    random.Random(20261031).shuffle(trials)
    save(args.output/'precommit.json',dict(trials=trials,clock_shared=True,source_sha256=hashes,
        python=sys.executable,cpu_count=os.cpu_count(),platform=platform.platform(),
        primary='Feasible complete objective published by deadline; parent recovers durable prefix after cancelling owned worker',
        boundary='Fresh-process model-only watchdog study, all work and startup included; measured OS overrun retained, not a hard real-time guarantee'))
    rows=[]
    for index,trial in enumerate(trials):
        journal=args.output/f'trial_{index:03}.jsonl';result=args.output/f'trial_{index:03}_worker.json'
        started=time.perf_counter()
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--method',trial['method'],'--seed',str(trial['seed']),
            '--budget',str(trial['budget']),'--start',str(started),'--case',trial['case'],'--output',str(result),'--journal',str(journal)]
        receipt=enforce(command,started=started,budget=trial['budget'],journal=journal,log=args.output/f'trial_{index:03}.log')
        record=dict(index=index,trial=trial,**receipt)
        save(args.output/f'trial_{index:03}.json',record)
        rows.append({k:v for k,v in record.items() if k!='events'});save(args.output/'rows.json',rows)
        print(json.dumps(dict(index=index,method=trial['method'],budget=trial['budget'],seed=trial['seed'],
            score=receipt['primary_best_feasible_score'],started=receipt['started_calls'],complete=receipt['completed_calls'],
            overrun_s=receipt['parent_overrun_s'],status=receipt['status'])),flush=True)
    if any(hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p)] for p in sources):raise RuntimeError('source changed during watchdog study')
    save(args.output/'completion.json',dict(status='COMPLETE',trials=len(rows),source_sha256=hashes,airsim_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--cases',type=Path,nargs='+',default=[ROOT/'reports/control_targets_fixed_matched_20261030_v3/case.json'])
    p.add_argument('--budgets',type=float,nargs='+',default=[2.,5.]);p.add_argument('--seeds',type=int,nargs='+',default=[13,31,47])
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--method',choices=METHODS);p.add_argument('--budget',type=float);p.add_argument('--start',type=float)
    p.add_argument('--seed',type=int);p.add_argument('--case',type=Path);p.add_argument('--journal',type=Path)
    a=p.parse_args();sys.exit(worker(a)) if a.worker else run(a)
