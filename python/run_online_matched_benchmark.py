"""Same frozen requests: objective-call and wall-budget search comparison.

Offline only. Run after AirSim campaigns to avoid control-thread interference.
"""
import argparse,hashlib,json,random,time,csv
from copy import deepcopy
from pathlib import Path
from dual_target_control.online_results import read_lines
from dual_target_control.online_curve import ReferenceTimeline
from dual_target_control.online_study import ARMS,prepare_request
from dual_target_control.online_worker_client import PlannerClient
from dual_target_control.reproduction_paths import workspace_record_path,leaf_name


def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def frozen_requests(source):
    source=Path(source);windows=json.loads((source/'online_reference_windows.json').read_text(encoding='utf-8'));requests=[]
    for e in read_lines(source/'online_events.jsonl'):
        if e['stage']!='SEARCH_DISPATCHED':continue
        r=e['request'];timeline=ReferenceTimeline(windows[0]['request'],windows[0]['parameters'])
        for w in windows[1:]:
            if w['epoch']<=r['epoch']:timeline.add(w['start_s'],w['request'],w['parameters'],w['source'],now=w['start_s']-.1)
        active=timeline.at(r['handoff_s'])[1];local,reason=prepare_request(r,active,r['handoff_s'],ARMS['PGL'])
        if reason:continue
        requests.append(dict(source_job=r['job_id'],source_target=r['target_index'],request=local))
        if len(requests)==4:break
    if not requests:raise ValueError('没有可在同一时间/局部域比较的请求')
    return requests

def run(source,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);requests=frozen_requests(source)
    root=Path(__file__).resolve().parents[1];source=Path(source).resolve()
    manifest=json.loads((source/'online_manifest.json').read_text(encoding='utf-8'))
    for path,digest in manifest['source_sha256'].items():
        archived=source/'source_snapshot'/leaf_name(path)
        if hashlib.sha256(archived.read_bytes()).hexdigest()!=digest:raise ValueError('旧来源快照不完整')
        if leaf_name(path) in ('online_curve.py','curves.py','parameterizations.py','route_curve.py','curve_task.py') and hashlib.sha256(workspace_record_path(path).read_bytes()).hexdigest()!=digest:raise ValueError('旧请求解码器与当前不同')
    paths=[workspace_record_path(p) for p in manifest['source_sha256']]+[Path(__file__),root/'python/dual_target_control/online_study.py']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    snapshot=output/'source_snapshot';snapshot.mkdir()
    for p in paths:(snapshot/p.name).write_bytes(p.read_bytes())
    jobs=[dict(request_index=i,seed=s,arm=a,budget=b,cap=100000,mode='wall') for i in range(len(requests)) for s in (31,32,33,34) for b in (.2,.4,.8) for a in ('AGL','PGL','RGL')]
    jobs += [dict(request_index=i,seed=s,arm=a,budget=3.,cap=32,mode='candidate') for i in range(len(requests)) for s in (31,32,33,34) for a in ('AGL','PGL','RGL')]
    random.Random(1101).shuffle(jobs)
    design=dict(source_run=str(source),requests=requests,jobs=jobs,execution_source_sha256=hashes,
        purpose='Matched objective problems, identical time/shape bounds and anchor; method proposal only differs; no AirSim, no tuning',
        source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('online_manifest.json','online_events.jsonl','online_reference_windows.json')},
        scope='Old1201 development requests, new optimizer seeds; not new AirSim evidence; common worker startup separately charged')
    save(output/'design_frozen.json',design);save(output/'status.json',dict(status='RUNNING',complete=0,total=len(jobs)))
    client=PlannerClient(output);results=[];hello=[];failure=None
    try:
        timeout=time.perf_counter()+40
        while not client.ready and time.perf_counter()<timeout:hello+=client.poll();time.sleep(.02)
        if not client.ready:raise RuntimeError('工作进程未就绪')
        hello+=client.poll();save(output/'startup.json',hello)
        for i,job in enumerate(jobs):
            r=deepcopy(requests[job['request_index']]['request']);settings=ARMS[job['arm']]
            r.update(seed=job['seed'],method=settings['method'],job_id=i,epoch=1,population=8,max_candidates=job['cap'],worker_journal_file=str(output/'evaluations.jsonl'))
            r['search_settings']=dict(paired_scale=.15,paired_fraction=.5,dropout=.95,learning_rate=.001,proposal_mode=settings.get('proposal_mode','network'))
            start=time.perf_counter();r['deadline_host_s']=start+job['budget'];client.submit(r);reply=None;timeout=start+15
            while time.perf_counter()<timeout:
                for x in client.poll():
                    if x.get('job_id')==i:reply=x
                if reply:break
                time.sleep(.002)
            if reply is None:reply=dict(type='error',error='基准调用超时',job_id=i)
            result=reply.get('result');row=dict(**job,job_id=i,reply=reply,dispatch_host_s=start,receive_wall_s=time.perf_counter()-start)
            row.update(status='COMPLETE' if result else 'FAILED',received_by_deadline=row['receive_wall_s']<=job['budget'],usable_result=False)
            if result:
                feasible=[v for v in result['ledger'] if v['feasible']]
                row.update(best_feasible_score=min((v['score'] for v in feasible),default=None),anchor_score=result['ledger'][0]['score'] if result['ledger'] else None,
                    selected_feasible=result['validation']['feasible'],search_calls=result['counts']['search_calls'],validation_calls=result['counts']['validation_calls'],wall_s=result['wall_s'],deadline_met=result['deadline_met'])
                row['usable_result']=bool(row['received_by_deadline'] and result['deadline_met'] and result['validation']['feasible'])
                row['usable_selected_score']=result['validation']['score'] if row['usable_result'] else None
                row['equal_candidate_count_met']=result['counts']['search_calls']==32 if job['mode']=='candidate' else None
            results.append(row)
            with (output/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            save(output/'status.json',dict(status='RUNNING',complete=len(results),total=len(jobs)))
            if (i+1)%12==0:print('MATCHED',i+1,'/',len(jobs),flush=True)
            if not result:failure=reply;break
    except Exception as ex:failure=dict(error=repr(ex))
    finally:
        try:client.close()
        except Exception as ex:failure=dict(error=repr(ex),prior=failure)
    unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    save(output/'status.json',dict(status='ABORTED' if failure else 'COMPLETE',error=failure,complete=len(results),total=len(jobs),not_run=jobs[len(results):],sources_unchanged=unchanged))
    fields=['job_id','request_index','seed','arm','mode','budget','cap','status','usable_selected_score','usable_result','received_by_deadline','equal_candidate_count_met','best_feasible_score','anchor_score','selected_feasible','search_calls','validation_calls','wall_s','receive_wall_s','deadline_met']
    with (output/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    save(output/'provenance.json',dict(design_sha256=hashlib.sha256((output/'design_frozen.json').read_bytes()).hexdigest(),execution_source_sha256=hashes,sources_unchanged=unchanged))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(run(a.source,a.output))
