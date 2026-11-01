"""Explicit development verification. --airsim performs a real flight."""
import argparse,json,time
from pathlib import Path
from online_curve_panel import online_config
from dual_target_control.online_live import OnlineRuntime


def run(output,method,duration,seed):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=OnlineRuntime(online_config(seed,duration,method));state=dict(method=method,requested_duration_s=duration,seed=seed,phase='LAUNCH',events=[])
    def save():(output/'verification.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    try:
        runtime.start_simulator();deadline=time.perf_counter()+180;started=False;last=None
        while time.perf_counter()<deadline:
            snap=runtime.snapshot();status=snap['status']
            if status!=last:
                state['events'].append(dict(status=status,message=snap.get('message'),host_s=time.perf_counter()));print(status,snap.get('message'),flush=True);last=status;save()
            if status=='ERROR':raise RuntimeError(snap['message'])
            if status=='READY' and not started:runtime.start();started=True;deadline=time.perf_counter()+duration+150
            if started and status=='STOPPED' and snap.get('result_ready'):
                state.update(phase='COMPLETE',run_dir=snap['run_dir'],metrics=snap.get('online_metrics'));save();return state
            time.sleep(.1)
        raise TimeoutError('真实在线核验超时')
    except Exception as ex:
        state.update(phase='FAILED',error=str(ex),run_dir=runtime.snapshot().get('run_dir'));save();raise
    finally:
        runtime.stop();runtime.shutdown()
        if not runtime.wait_closed(90):raise RuntimeError('运行线程未完成降落/关闭')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--airsim',action='store_true',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--method',default='hermite_abom');parser.add_argument('--duration',type=float,default=25.);parser.add_argument('--seed',type=int,default=1201)
    args=parser.parse_args();run(args.output,args.method,args.duration,args.seed)
