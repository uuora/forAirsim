"""Explicit single-method AirSim validation of the original03 live framework."""
import argparse,json,time
from pathlib import Path
from run_curve_live_panel import live_config
from curve_airsim_experiment_panel import TITLES,catalogue
from dual_target_control.curve_live import LiveCurveRuntime


def run(args):
    cfg=live_config(args.plans) if args.plans else live_config()
    cfg['curve_experiment']['plan']=next(p for p in catalogue(cfg['curve_experiment']['plans_file']) if p['group_id']==args.group)
    cfg['curve_experiment'].update(auto_show_scene=args.show_scene,time_scale=args.time_scale)
    cfg['control']['max_speed']=args.max_speed
    cfg['control']['max_acceleration']=args.max_acceleration
    cfg['airsim']['initial_height']=cfg['curve_experiment']['plan']['boundary']['p0'][2]
    runtime=LiveCurveRuntime(cfg)
    try:
        runtime.start_simulator();deadline=time.monotonic()+180
        while runtime.snapshot()['status']!='READY':
            snap=runtime.snapshot()
            if snap['status']=='ERROR':raise RuntimeError(snap['message'])
            if time.monotonic()>deadline:raise TimeoutError('03实时环境未就绪')
            time.sleep(.1)
        runtime.start();deadline=time.monotonic()+130;observed_run=False
        while True:
            snap=runtime.snapshot();observed_run=observed_run or snap.get('running',False)
            if snap['status']=='ERROR':raise RuntimeError(snap['message'])
            if observed_run and snap['status']=='STOPPED':break
            if time.monotonic()>deadline:raise TimeoutError('实时单组验证超时')
            time.sleep(.1)
        print(json.dumps(dict(status=snap['status'],run_dir=snap['run_dir'],metrics=snap.get('control_metrics'),
            mode=snap['mode'],protocol_revision=13),ensure_ascii=False))
    finally:
        runtime.stop();runtime.shutdown()
        if not runtime.wait_closed(70):raise RuntimeError('落地或控制线程未结束；保留现场')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True,choices=list(TITLES));p.add_argument('--plans',type=Path)
    p.add_argument('--time-scale',type=float,default=1.);p.add_argument('--max-speed',type=float,default=4.)
    p.add_argument('--max-acceleration',type=float,default=4.)
    p.add_argument('--show-scene',action='store_true');run(p.parse_args())
