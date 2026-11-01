"""Explicit selected methods/policies through the same product batch scheduler."""
import argparse,json,time
from pathlib import Path
from dual_target_control.curve_campaign import CurveCampaign
from dual_target_control.route_live import RouteLiveRuntime
from route_experiment_panel import route_live_config,PRESET,ROOT
from curve_airsim_experiment_panel import TITLES


def run(args):
    cfg=route_live_config();packages={k:ROOT/v for k,v in json.loads(PRESET.read_text(encoding='utf-8'))['packages'].items()}
    campaign=CurveCampaign(cfg,packages,[(p,m) for p in args.policies for m in args.methods],args.repeats,args.output)
    runtime=RouteLiveRuntime(cfg)
    try:
        runtime.start_simulator();deadline=time.monotonic()+180
        while runtime.snapshot()['status']!='READY':
            snap=runtime.snapshot()
            if snap['status']=='ERROR':raise RuntimeError(snap['message'])
            if time.monotonic()>deadline:raise TimeoutError('环境未就绪')
            time.sleep(.1)
        print(json.dumps(dict(stage='PREVIEW_NO_FLIGHT',seconds=args.preview_seconds)),flush=True)
        time.sleep(args.preview_seconds)
        while campaign.active:
            before=campaign.index;campaign.tick(runtime)
            if campaign.index!=before:
                slot=campaign.slots[campaign.index-1]
                print(json.dumps({k:slot.get(k) for k in ('policy','method','status','run_dir')},ensure_ascii=False),flush=True)
            time.sleep(.05)
        if campaign.phase!='COMPLETED':raise RuntimeError('批量未全部结束：'+campaign.phase)
    finally:
        runtime.stop();runtime.shutdown()
        if not runtime.wait_closed(75):raise RuntimeError('落地/关闭未完成，保留现场')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',required=True,choices=list(TITLES))
    p.add_argument('--policies',nargs='+',required=True,choices=['stop','pass']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=1);p.add_argument('--preview-seconds',type=float,default=2)
    run(p.parse_args())
