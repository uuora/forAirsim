"""Explicit real AirSim protocol15 smoke; never called by opening the panel."""
import argparse,json,time
from pathlib import Path
from prepared_route_panel import prepared_config,task_template,task_from_spec
from prepared_route_planner import prepared_spec
from dual_target_control.prepared_route_live import PreparedRouteRuntime
from dual_target_control.curve_audit import audit_run


def run(args):
    cfg=prepared_config();spec=prepared_spec([[8,0,5],[12,6,7],[6,10,6]],args.policy,start=args.start)
    cfg['preparation'].update(task=task_from_spec(spec),method=args.method)
    cfg['airsim']['initial_height']=spec['start'][2]
    cfg['curve_experiment'].update(plan=task_template(spec,args.method),time_scale=spec['time_scale'])
    runtime=PreparedRouteRuntime(cfg);output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    try:
        runtime.start_simulator();deadline=time.monotonic()+180
        while runtime.snapshot()['status']!='READY':
            snap=runtime.snapshot()
            if snap['status']=='ERROR':raise RuntimeError(snap['message'])
            if time.monotonic()>deadline:raise TimeoutError('环境未就绪')
            time.sleep(.1)
        snap=runtime.snapshot();assert not snap['running'] and not snap.get('planned_path')
        (output/'idle.json').write_text(json.dumps(snap,ensure_ascii=False,indent=2),encoding='utf-8')
        print('IDLE_NO_REFERENCE_NO_FLIGHT',flush=True);runtime.start();deadline=time.monotonic()+300;last=None
        while True:
            snap=runtime.snapshot()
            if snap['status']!=last:print(snap['status'],snap.get('message'),flush=True);last=snap['status']
            if snap['status'] in ('ERROR','STOPPED') and snap.get('run_dir'):
                folder=Path(snap['run_dir']);(output/'run.json').write_text(json.dumps(dict(run_dir=str(folder),snapshot=snap),ensure_ascii=False,indent=2),encoding='utf-8')
                receipt=audit_run(folder);(output/'audit.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(receipt,ensure_ascii=False),flush=True)
                if snap['status']=='ERROR':raise RuntimeError(snap['message'])
                return
            if time.monotonic()>deadline:raise TimeoutError('准备/规划/执行未在300秒内结束')
            time.sleep(.05)
    finally:
        runtime.stop();runtime.shutdown()
        if not runtime.wait_closed(90):raise RuntimeError('落地/关闭未完成，保留现场')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--method',required=True);parser.add_argument('--policy',choices=['stop','pass'],default='stop')
    parser.add_argument('--start',type=float,nargs=3,default=[-3.,1.,5.]);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
