"""Explicit selected-group development validation; one original03 simulator.

There is no default group list. This CLI is separate from the interactive
single-method UI; it never silently expands a user selection to six methods.
"""
import argparse,copy,json,time
from pathlib import Path
from run_curve_live_panel import live_config
from dual_target_control.curve_live import LiveCurveRuntime
from curve_airsim_experiment_panel import TITLES,catalogue


def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def run(args):
    if len(set(args.groups))!=len(args.groups):raise ValueError('重复方法')
    cfg=live_config();cfg['curve_experiment']['auto_show_scene']=True
    plans={p['group_id']:p for p in catalogue(cfg['curve_experiment']['plans_file'])}
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'precommit.json',dict(groups=args.groups,base_config=cfg,
        scope='Explicit development coverage; single persistent original03 simulator, protocol13; not a superiority experiment'))
    runtime=LiveCurveRuntime(cfg);results=[]
    try:
        runtime.start_simulator();deadline=time.monotonic()+180
        while runtime.snapshot()['status']!='READY':
            snap=runtime.snapshot()
            if snap['status']=='ERROR':raise RuntimeError(snap['message'])
            if time.monotonic()>deadline:raise TimeoutError('环境启动超时')
            time.sleep(.1)
        for group in args.groups:
            current=copy.deepcopy(cfg);current['curve_experiment']['plan']=plans[group]
            old_path=runtime.snapshot().get('run_dir');runtime.configure(current);runtime.start()
            deadline=time.monotonic()+150;started=False
            while True:
                snap=runtime.snapshot();started=started or (snap.get('run_dir')!=old_path and snap.get('running'))
                if snap['status']=='ERROR':raise RuntimeError(snap['message'])
                if started and snap['status']=='STOPPED':break
                if time.monotonic()>deadline:raise TimeoutError('单组执行超时：'+group)
                time.sleep(.1)
            row=dict(method=group,run_dir=snap['run_dir'],metrics=snap.get('control_metrics'))
            results.append(row);save(args.output/'rows.json',results)
            print(json.dumps(dict(method=group,run_dir=snap['run_dir'],control_passed=(row['metrics'] or {}).get('control_error_gates_passed')),ensure_ascii=False),flush=True)
        save(args.output/'completion.json',dict(status='COMPLETE',executed=len(results),groups=args.groups))
    except Exception as exc:
        save(args.output/'failure.json',dict(error=str(exc),snapshot=runtime.snapshot(),completed=len(results),uncompleted=args.groups[len(results):]))
        raise
    finally:
        runtime.stop();runtime.shutdown()
        if not runtime.wait_closed(75):raise RuntimeError('保留现场：运行线程尚未完成落地/退出')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',choices=list(TITLES),required=True)
    p.add_argument('--output',type=Path,required=True);run(p.parse_args())
