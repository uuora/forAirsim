"""Explicit six-arm research pilot, never invoked by opening the GUI."""
import argparse,hashlib,json,time,random,sys
from pathlib import Path
from dual_target_control.online_study import ARMS,ResearchRuntime,config_for


def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def run_trial(config,output):
    output.mkdir();runtime=ResearchRuntime(config);record=dict(status='STARTING',arm=config['online']['research']['arm'],config=config,run_dir=None)
    try:
        runtime.start_simulator();deadline=time.perf_counter()+180;started=False;last=None
        while time.perf_counter()<deadline:
            snap=runtime.snapshot();status=snap['status']
            if last!=status:print(record['arm'],status,snap.get('message'),flush=True);last=status
            record['run_dir']=snap.get('run_dir')
            if status=='ERROR':raise RuntimeError(snap['message'])
            if status=='READY' and not started:runtime.start();started=True;deadline=time.perf_counter()+config['online']['duration_s']+150
            if started and status=='STOPPED' and snap.get('result_ready'):
                record.update(status='COMPLETE',metrics=snap['online_metrics']);break
            time.sleep(.1)
        else:raise TimeoutError('消融试验超时')
    except Exception as ex:record.update(status='FAILED',error=str(ex),run_dir=runtime.snapshot().get('run_dir'))
    finally:
        runtime.stop();runtime.shutdown();closed=runtime.wait_closed(90)
        record['closed_safely']=bool(closed and not runtime._flight_owned)
        if record['run_dir']:
            path=Path(record['run_dir'])/'online_metrics.json'
            if path.exists():record['metrics']=json.loads(path.read_text(encoding='utf-8'))
        save(output/'trial.json',record)
    return record


def run(output,seed=1301,duration=25.,arms=None,budget=.8):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    selected=list(ARMS) if arms is None else list(arms)
    if len(set(selected))!=len(selected) or any(a not in ARMS for a in selected):raise ValueError('未知/重复消融组')
    order=selected.copy();random.Random(461).shuffle(order)
    design=dict(study='progress_locality_v1',created_utc_ns=time.time_ns(),scenario_seed=seed,optimizer_seed_rule='scenario seed + target index',
        purpose='Exploratory mechanism pilot; one seed/one trial per arm; no significance or superiority inference',
        duration_s=duration,wall_budget_s=budget,validation_reserve_s=.10,candidate_cap=100000,population=8,order_seed=461,execution_order=order,
        arms={a:config_for(a,seed,duration,budget)['online']['research'] for a in selected},
        primary='Completed target count in fixed mission time, with failed/aborted trials retained',
        secondary=['synchronized RMSE/peak/gaps','new absolute target-arrival postponement','complete proxy calls per reached target','job latency p50/p95/max/deadline misses','source of adopted proposal','worker startup and phase times'],
        treatment_fields=['method','online.research'],fixed='Same target stream, dynamics, controller, limits, gates, architecture, population, per-job wall budget',
        stop_rules='Stop current trial and land on control error; abort remaining trials if landing/closure cannot be confirmed; no retuning after seeing seed1301',
        not_claimed=['AirSim objective search','novelty priority','general effectiveness','statistical significance'])
    save(output/'design_frozen.json',design);digest=hashlib.sha256((output/'design_frozen.json').read_bytes()).hexdigest()
    campaign=dict(design_sha256=digest,status='RUNNING',trials=[],not_run=order.copy());save(output/'campaign.json',campaign)
    for i,arm in enumerate(order):
        record=run_trial(config_for(arm,seed,duration,budget),output/f'{i+1:02d}_{arm}')
        campaign['trials'].append(record);campaign['not_run'].remove(arm);save(output/'campaign.json',campaign)
        if not record['closed_safely']:campaign['status']='ABORTED_UNCONFIRMED_LANDING';break
    else:campaign['status']='COMPLETE_WITH_ALL_OUTCOMES'
    save(output/'campaign.json',campaign);return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--airsim',action='store_true',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1301);parser.add_argument('--duration',type=float,default=25.);parser.add_argument('--arms',nargs='+',choices=list(ARMS));parser.add_argument('--budget',type=float,default=.8)
    a=parser.parse_args();run(a.output,a.seed,a.duration,a.arms,a.budget)
