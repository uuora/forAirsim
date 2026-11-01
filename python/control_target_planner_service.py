"""Sequential planning service; reuse imports, never learned weights or scores.

Each request runs the unchanged seeded six-group generator from a fresh model
initialization. No AirSim API or flight is used. EOF/quit exits after the current
request has saved its evidence, so closing the panel cannot break its log pipe.
"""
import contextlib,hashlib,json,sys,time
from pathlib import Path
from control_target_experiments import generate,make_case,ROOT

SOURCE_PATHS=[Path(__file__),ROOT/'python/control_target_experiments.py',
    *[ROOT/'python/trajectory_research'/name for name in ('parameterizations.py','curves.py','timed_tracking.py','curve_task.py','screened_search.py','search.py')],
    ROOT/'python/route_control_experiments.py',ROOT/'python/dual_target_control/route_curve.py',
    ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py']
FROZEN={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE_PATHS}


def validate_request(request):
    if not isinstance(request,dict) or set(request)!={'action','output','seed','goal'} or request['action']!='generate':
        raise ValueError('closed generate request schema required')
    if isinstance(request['seed'],bool) or not isinstance(request['seed'],int) or not 0<=request['seed']<2**31:
        raise ValueError('seed must be an integer in [0,2**31)')
    output=Path(request['output']).resolve();root=(ROOT/'reports').resolve()
    if root not in output.parents:raise ValueError('planning output must be a new child of workspace/reports')
    case=make_case(request['seed'],request['goal'])
    return output,case


def respond(value):
    try:print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
    except BrokenPipeError:return False
    return True


def validate_route_request(request):
    keys={'action','output','seed','points','policy','pass_speed','dwell_s','time_scale','max_speed','max_slew'}
    if not isinstance(request,dict) or set(request)!=keys or request['action']!='generate_route':raise ValueError('closed route request schema required')
    from dual_target_control.route_curve import route_spec
    output,_=validate_request(dict(action='generate',output=request['output'],seed=request['seed'],goal=[8,0,5]))
    route_spec(request['points'],request['policy'],request['seed'],pass_speed=request['pass_speed'],dwell_s=request['dwell_s'],time_scale=request['time_scale'],max_speed=request['max_speed'],max_slew=request['max_slew'])
    return output


def main():
    if not respond(dict(status='READY',source_sha256=FROZEN)):return
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request=={'action':'quit'}:break
            if any(hashlib.sha256(p.read_bytes()).hexdigest()!=FROZEN[str(p)] for p in SOURCE_PATHS):
                if not respond(dict(status='STALE_SOURCE',error='Planning source changed; restart service before a new request')):break
                continue
            is_route=isinstance(request,dict) and request.get('action')=='generate_route'
            if is_route:output=validate_route_request(request)
            else:output,case=validate_request(request)
            warm='torch' in sys.modules
            log=output.with_suffix('.planning.log');log.parent.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter()
            with log.open('w',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream):
                if is_route:
                    from route_control_experiments import generate_route
                    plan=generate_route(output,request['points'],request['policy'],request['seed'],
                        pass_speed=request['pass_speed'],dwell_s=request['dwell_s'],time_scale=request['time_scale'],max_speed=request['max_speed'],max_slew=request['max_slew'])
                else:plan=generate(output,case)
            receipt=dict(status='PASS',output=str(output),plan=str(plan),request_generation_wall_s=time.perf_counter()-started,
                runtime_warm=warm,source_sha256=FROZEN,
                reuse_scope='Imported runtime modules only. No network-weight warm start, no cached objective measurements; all 326 complete model calls are repeated.')
            (output/'planning_service_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
            if not respond(receipt):break
        except Exception as exc:
            if not respond(dict(status='ERROR',error=f'{type(exc).__name__}: {exc}')):break


if __name__=='__main__':main()
