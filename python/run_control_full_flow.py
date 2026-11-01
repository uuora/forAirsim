"""Explicit complete development workflow. No flights without --airsim.

Preserves each failure; aborts dispatch if landing cannot be confirmed.
The frozen design is saved before the first simulator call.
"""
import argparse,hashlib,json,random,time
from pathlib import Path
from copy import deepcopy
from dual_target_control.online_study import ARMS,ResearchRuntime,config_for


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def fixed_config(policy,method):
    from prepared_route_panel import prepared_config,task_template,task_from_spec
    from prepared_route_planner import prepared_spec
    cfg=prepared_config()
    spec=prepared_spec([[8,0,5],[12,6,7],[6,10,6]],policy,1201,start=[-3,1,5])
    cfg['preparation'].update(task=task_from_spec(spec),method=method)
    cfg['airsim']['initial_height']=5.
    cfg['curve_experiment'].update(plan=task_template(spec,method),time_scale=spec['time_scale'])
    return cfg


def trial(task,output):
    from dual_target_control.prepared_route_live import PreparedRouteRuntime
    output.mkdir();cfg=task['config'];online=task['family']=='online'
    runtime=ResearchRuntime(cfg) if online else PreparedRouteRuntime(cfg)
    record=dict(task_id=task['id'],family=task['family'],arm=task.get('arm'),policy=task.get('policy'),method=task['method'],status='STARTING',run_dir=None,events=[])
    started=False;last=None
    try:
        runtime.start_simulator();deadline=time.perf_counter()+180
        while time.perf_counter()<deadline:
            snap=runtime.snapshot();status=snap['status'];record['run_dir']=snap.get('run_dir')
            if status!=last:
                event=dict(status=status,message=snap.get('message'),host_s=time.perf_counter())
                record['events'].append(event);print(task['id'],status,snap.get('message'),flush=True);last=status;save(output/'trial.json',record)
            if status=='ERROR':raise RuntimeError(snap.get('message'))
            if status=='READY' and not started:
                record['idle_no_flight_no_reference']=not snap.get('running') and not snap.get('planned_path')
                if not record['idle_no_flight_no_reference']:raise RuntimeError('空闲状态出现飞行/参考')
                runtime.start();started=True;deadline=time.perf_counter()+360
            if started and status=='STOPPED' and snap.get('result_ready'):
                record['status']='COMPLETE';break
            time.sleep(.1)
        else:raise TimeoutError('单次流程未在时限内结束')
    except Exception as ex:
        record.update(status='FAILED',error=str(ex),run_dir=runtime.snapshot().get('run_dir'))
    finally:
        runtime.stop();runtime.shutdown();closed=runtime.wait_closed(90)
        record['closed_safely']=bool(closed and not runtime._flight_owned)
        if record['run_dir']:
            folder=Path(record['run_dir'])
            for name in ('summary.json','online_metrics.json'):
                if (folder/name).exists():record[name.removesuffix('.json')]=json.loads((folder/name).read_text(encoding='utf-8'))
        save(output/'trial.json',record)
    return record


def make_design(sections):
    from control_target_experiments import GROUPS
    tasks=[]
    if 'fixed' in sections:
        selections=[(p,m['id'] if isinstance(m,dict) else m) for p in ('stop','pass') for m in GROUPS]
        random.Random(1101).shuffle(selections)
        for policy,method in selections:
            tasks.append(dict(id=f'fixed_{policy}_{method}',family='fixed',policy=policy,method=method,config=fixed_config(policy,method)))
    if 'online' in sections:
        # Three paired scenario/search blocks; three blocks are exploratory,
        # not a powered efficacy test. No tuning on these outcomes.
        for scene,optimizer in [(1401,21),(1402,22),(1403,23)]:
            arms=list(ARMS);random.Random(1101+scene).shuffle(arms)
            for arm in arms:
                cfg=config_for(arm,scene,40.,.8);cfg['online']['optimizer_seed']=optimizer
                cfg['online']['study_id']='progress_locality_repeated_v2'
                tasks.append(dict(id=f'online_{scene}_{optimizer}_{arm}',family='online',arm=arm,method=cfg['online']['method'],scene_seed=scene,optimizer_seed=optimizer,config=cfg))
    return dict(study='control_full_flow_20261101_v1',created_utc_ns=time.time_ns(),sections=sections,tasks=tasks,
        treatment_fields=['method','online.research'],online_blocks='scenario1401/optimizer21,1402/22,1403/23; paired within block; one flight per arm per block',
        primary=dict(fixed='Completion and control gates, then tracking quality and model-planning wall time separately',online='Targets reached in 40s; tracking gates; paired task/resource differences'),
        conditions='Original03 map and real-time lifecycle; no obstacles/perception; common fixed feedback, limits and thresholds; ordinary/ABOM/paired B/H fixed-route arms; eight Hermite online mechanisms',
        interpretation='Development replication, not proof of general superiority, high agility or statistical significance; B/H same physical family',
        stop_rules='Preserve failed trials; continue only after confirmed landing/closure; source changes require new campaign; no outcome-driven retuning',
        scope='Model-only objective search; AirSim executes the full closed-loop mission and performs zero objective-search calls')


def run(output,sections):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    design=make_design(sections);save(output/'design_frozen.json',design)
    campaign=dict(design_sha256=hashlib.sha256((output/'design_frozen.json').read_bytes()).hexdigest(),status='RUNNING',trials=[],not_run=[t['id'] for t in design['tasks']])
    save(output/'campaign.json',campaign)
    for task in design['tasks']:
        result=trial(task,output/task['id']);campaign['trials'].append(result);campaign['not_run'].remove(task['id'])
        save(output/'campaign.json',campaign)
        if not result['closed_safely']:
            campaign['status']='ABORTED_UNCONFIRMED_LANDING';save(output/'campaign.json',campaign);return output
    campaign['status']='COMPLETE_WITH_ALL_OUTCOMES';save(output/'campaign.json',campaign);return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--airsim',action='store_true',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sections',nargs='+',choices=['fixed','online'],default=['fixed','online'])
    args=p.parse_args();print(run(args.output,args.sections))
