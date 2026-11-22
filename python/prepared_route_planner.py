"""Protocol15 task declaration and one selected-method, model-only planner.

The controller submits each job only after measured arrival at the start.
"""
import hashlib,json,os,sys,time,traceback
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from dual_target_control.route_curve import route_spec


def prepared_spec(points,policy='stop',seed=1201,*,start=(0.,0.,5.),position_error_limit_m=None,tracking=None,retiming=None,**settings):
    # Obtain the unchanged rule/scalar schema independently of task geometry.
    spec=route_spec([[8.,0.,5.]],policy,seed,**settings)
    p=np.asarray(points,dtype=float);s=np.asarray(start,dtype=float)
    if s.shape!=(3,) or not np.isfinite(s).all():raise ValueError('准备点必须是三个有限坐标')
    if p.ndim!=2 or p.shape[1]!=3 or not 1<=len(p)<=10 or not np.isfinite(p).all():raise ValueError('请输入1至10个有限三维航点')
    chain=np.vstack([s,p]);delta=np.diff(chain,axis=0)
    if np.any(chain<[-45,-45,3]) or np.any(chain>[45,45,23]):raise ValueError('准备点和航点须在x/y±45、高度3至23米内')
    if np.any(np.linalg.norm(delta,axis=1)<2):raise ValueError('准备点和相邻航点至少相隔2米')
    velocities=np.zeros_like(p)
    if policy=='pass':
        for i in range(len(p)-1):
            a=delta[i]/np.linalg.norm(delta[i]);b=delta[i+1]/np.linalg.norm(delta[i+1])
            if a@b<-.85:raise ValueError('连续过点暂不接受近原地掉头，请增大转弯空间')
            velocities[i]=(a+b)/np.linalg.norm(a+b)*spec['pass_speed_mps']
    spec.update(start=s.tolist(),waypoints=p.tolist(),waypoint_velocities=velocities.tolist())
    if position_error_limit_m is not None:
        if not 0 < position_error_limit_m <= .5:
            raise ValueError('位置误差上限须在 (0, 0.5] 米内')
        spec.update(waypoint_radius_m=position_error_limit_m, peak_position_limit_m=position_error_limit_m)
    if tracking is not None:
        for name, minimum in (('acceleration_feedforward_s', 0.), ('proxy_lag_s', 1.e-3), ('proxy_delay_s', 0.)):
            values = np.asarray(tracking[name], dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < minimum):
                raise ValueError('无效的跟踪参数：' + name)
        spec['tracking'] = tracking
    if retiming is not None:
        if not all(0 < retiming[key] <= 1 for key in ('nominal_speed_fraction', 'nominal_slew_fraction')):
            raise ValueError('名义速度与变化率占比须在 (0, 1] 内')
        if retiming.get('continuous_waypoint_speed', 'fixed') not in ('fixed', 'optimize'):
            raise ValueError('连续过点速度模式无效')
        fractions = retiming['fractions']
        if not fractions or any(not 0 < value <= 1 for value in fractions) or fractions != sorted(set(fractions), reverse=True):
            raise ValueError('计时系数须在 (0, 1] 内并按严格降序排列')
        spec['retiming'] = retiming
    return spec


def generate_selected(output, request):
    from fast_reference import make_plan
    return make_plan(output, prepared_spec(**request['task']), request['method'], request['dynamics_model'])


def warm_common(method):
    import fast_reference
    return dict(objective_calls=0, scope='Imports only; planning starts after arrival.')


def serve(method,source_manifest):
    """Persistent JSON-line transport; no request means no search."""
    from dual_target_control.prepared_worker_client import binding,source_digest
    sources=json.loads(Path(source_manifest).read_text(encoding='utf-8'));digest=source_digest(sources)
    def verify_sources():
        if any(not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected for path,expected in sources.items()):
            raise RuntimeError('规划工作进程源码与本次冻结条件不一致')
    # Keep native solver banners and DLL output out of the JSON reply pipe.
    replies = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8')
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    def emit(row):print(json.dumps(row,ensure_ascii=False,allow_nan=False),file=replies,flush=True)
    try:
        verify_sources()
        with redirect_stdout(sys.stderr):startup=warm_common(method)
        verify_sources();emit(dict(type='ready',method=method,source_sha256=digest,startup=startup))
    except Exception:
        emit(dict(type='error',error=traceback.format_exc(),source_sha256=digest));return
    for line in sys.stdin:
        job={};started=time.perf_counter()
        try:
            job=json.loads(line)
            if job.get('source_sha256')!=digest or job.get('request_sha256')!=binding(job['request'],job['output']):raise ValueError('规划任务请求或源码绑定不一致')
            if job['request'].get('method')!=method:raise ValueError('规划请求方法与预热进程不一致')
            if job['submitted_host_s']<job['arrived_host_s']:raise ValueError('准备点到达前不得提交搜索')
            verify_sources()
            with redirect_stdout(sys.stderr):path=generate_selected(Path(job['output']),job['request'])
            verify_sources()
            emit(dict(type='result',job_id=job['job_id'],request_sha256=job['request_sha256'],source_sha256=digest,
                plan_path=str(Path(path).resolve()),timings=dict(worker_started_host_s=started,worker_completed_host_s=time.perf_counter(),
                    worker_request_wall_s=time.perf_counter()-started)))
        except Exception:
            emit(dict(type='error',job_id=job.get('job_id'),request_sha256=job.get('request_sha256'),source_sha256=digest,
                error=traceback.format_exc(),timings=dict(worker_started_host_s=started,worker_terminal_host_s=time.perf_counter(),
                    worker_request_wall_s=time.perf_counter()-started)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--serve',action='store_true');parser.add_argument('--method');parser.add_argument('--source-manifest',type=Path)
    args=parser.parse_args()
    if args.serve:
        if args.request or args.output or not args.method or not args.source_manifest:parser.error('--serve requires --method and --source-manifest only')
        serve(args.method,args.source_manifest)
    else:
        if not args.request or not args.output:parser.error('--request and --output are required for a one-shot plan')
        request=json.loads(args.request.read_text(encoding='utf-8'));print(generate_selected(args.output,request),flush=True)
