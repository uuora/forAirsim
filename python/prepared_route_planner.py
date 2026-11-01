"""Protocol15 task declaration and one selected-method, model-only planner.

Launched after measured arrival at the preparation point. No AirSim imports or
black-box AirSim evaluations. The worker has fresh network weights each run.
"""
import json
from pathlib import Path
import numpy as np
from dual_target_control.route_curve import route_spec


def prepared_spec(points,policy='stop',seed=1201,*,start=(0.,0.,5.),**settings):
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
    return spec


def generate_selected(output,request):
    from route_control_experiments import generate_route
    spec=prepared_spec(**request['task'])
    return generate_route(output,spec['waypoints'],spec['policy'],spec['seed'],
        selected_groups=[request['method']],spec_override=spec,budget=80,population=20)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();request=json.loads(args.request.read_text(encoding='utf-8'))
    print(generate_selected(args.output,request),flush=True)
