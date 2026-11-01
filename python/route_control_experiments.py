"""Joint multi-point model search for two separately declared policies."""
from dataclasses import asdict
import hashlib,json,time
from pathlib import Path
import numpy as np
from dual_target_control.route_curve import route_spec,route_boundaries,decode_route,route_metrics
from trajectory_research.curve_task import subdivided_derivative_norm_bound
from trajectory_research.screened_search import run_screened_search
from control_target_experiments import GROUPS,ROOT,save,sha


def plan_from_vector(vector,spec,representation):
    boundaries=route_boundaries(spec);x=np.asarray(vector).reshape(len(boundaries),4)
    return dict(group_id=representation+'_ordinary',representation=representation,family='route_single_bump',seed=spec['seed'],route=spec,
        boundary=boundaries[0],parameters=x[0].tolist(),segments=[dict(parameters=row.tolist(),boundary=b) for row,b in zip(x,boundaries)])


def evaluate(vector,spec,representation):
    plan=plan_from_vector(vector,spec,representation);curve=decode_route(plan)
    times=np.unique(np.r_[np.arange(0,curve.duration_s,.02),curve.ends,curve.duration_s]);q,qv,qa=[curve.sample(times,derivative=d) for d in (0,1,2)]
    p=q[0].copy();v=qv[0].copy();previous=qv[0].copy();states=[];commands=[];tau=.9
    cap=spec['command_limits']['max_speed_mps'];slew=spec['command_limits']['max_slew_mps2']
    for i,t in enumerate(times):
        states.append(dict(reference_elapsed_s=float(t),position_zup_m=p.tolist(),velocity_zup_mps=v.tolist()))
        if i==len(times)-1:break
        dt=float(times[i+1]-t);raw=qv[i]+2*(q[i]-p)+.8*(qv[i]-v)+.9*qa[i]
        speed=raw*min(1.,cap/max(np.linalg.norm(raw),1e-300));change=speed-previous
        limited=previous+change*min(1.,slew*dt/max(np.linalg.norm(change),1e-300))
        commands.append(dict(speed_clipped=bool(np.linalg.norm(raw)>cap),slew_clipped=bool(np.linalg.norm(change)>slew*dt)))
        e=np.exp(-dt/tau);p=p+limited*dt+(v-limited)*tau*(1-e);v=limited+(v-limited)*e;previous=limited
    metrics=route_metrics(curve,states,commands)
    derivative_bounds=[max(subdivided_derivative_norm_bound(s,d,depth=2) for s in curve.segments) for d in (1,2,3)]
    excess=sum(max(0.,v/bound-1) for v,bound in zip(derivative_bounds,(cap,slew,30.)))
    excess+=max(0.,metrics['max_position_error_m']/.5-1)+max(0.,metrics['terminal_position_error_m']/.25-1)+max(0.,metrics['terminal_velocity_error_mps']/.5-1)
    excess+=sum(0. if r['passed'] else 1. for r in metrics['waypoint_results'])
    feasible=metrics['control_error_gates_passed'] and excess<=0
    return dict(parameters=list(map(float,vector)),score=float(metrics['quality_term'] if feasible else 1+excess),feasible=bool(feasible),
        metrics=metrics,derivative_bounds=derivative_bounds,scope='Isotropic lagged point mass, no AirSim calls; ranking transfer unproven')


def generate_route(output,points,policy,seed=1201,*,pass_speed=1.2,dwell_s=1.,time_scale=1.25,max_speed=6.,max_slew=6.,budget=80,population=20,selected_groups=None,spec_override=None):
    started=time.perf_counter();spec=spec_override if spec_override is not None else route_spec(points,policy,seed,pass_speed=pass_speed,dwell_s=dwell_s,time_scale=time_scale,max_speed=max_speed,max_slew=max_slew)
    groups=list(GROUPS if selected_groups is None else selected_groups)
    if not groups or len(set(groups))!=len(groups) or any(g not in GROUPS for g in groups):raise ValueError('未知或重复的方法')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'groups').mkdir();snapshot=output/'source_snapshot';snapshot.mkdir()
    sources=[Path(__file__),ROOT/'python/dual_target_control/route_curve.py',ROOT/'python/trajectory_research/screened_search.py',
        ROOT/'python/trajectory_research/search.py',ROOT/'python/trajectory_research/curves.py',ROOT/'python/trajectory_research/parameterizations.py',
        ROOT/'python/trajectory_research/curve_task.py',ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py']
    if spec_override is not None:sources.append(ROOT/'python/prepared_route_planner.py')
    hashes={str(p):sha(p) for p in sources}
    for p in sources:(snapshot/p.name).write_bytes(p.read_bytes())
    boundaries=route_boundaries(spec);n=len(boundaries)
    lower=np.tile([-.8,-1.4,-.8,3.],n);upper=np.tile([.8,1.4,.8,7.],n)
    anchor=np.array([[0,0,0,min(7.,max(3.,np.linalg.norm(np.asarray(b['p1'])-b['p0'])/1.6+1.))] for b in boundaries]).reshape(-1)
    save(output/'route.json',spec);save(output/'config.json',dict(route=spec,lower=lower.tolist(),upper=upper.tolist(),population=population,budget=budget,
        controller=dict(position_gain=2.,velocity_error_gain=.8,acceleration_feedforward_s=.9,max_speed=float(max_speed),max_slew=float(max_slew),lag_s=.9),
        objective='Whole-route model rollout including waypoint rules; time scaling included during search',source_sha256=hashes))
    plans=[];total_calls=0
    for group in groups:
        representation=group.split('_')[0];method=None if group.endswith('ordinary') else 'abom_perturb' if group.endswith('candidate') else 'abom'
        calls=[]
        def objective(x):
            tick=time.perf_counter();value=evaluate(x,spec,representation);calls.append(dict(parameters=value['parameters'],score=value['score'],feasible=value['feasible'],wall_s=time.perf_counter()-tick));return value
        tick=time.perf_counter();search=None
        if method:search=run_screened_search(objective,lower,upper,method=method,seed=seed,population=population,budget=budget,
            anchor=anchor,upstream=ROOT/'tmp/abom_deploy_probe/upstream',dropout=.95,injection_fraction=.5)
        search_wall=time.perf_counter()-tick if method else 0.
        vector=search['best_parameters'] if search else anchor
        tick=time.perf_counter();validation=evaluate(vector,spec,representation);validation_wall=time.perf_counter()-tick
        plan=plan_from_vector(vector,spec,representation);plan['group_id']=group
        plan['selection_provenance']=dict(result_json=f'groups/{group}.json',method=method,model_feasible=validation['feasible'],
            scope='Pinned network adapter, joint route search; generic paired perturbation candidate, not original-paper replication')
        counts=dict(search_objective_calls=len(calls),search_proposals=len(calls),selected_validation_calls=1,total_objective_calls=len(calls)+1);total_calls+=len(calls)+1
        timings=dict(search_wall_s=search_wall,selected_validation_wall_s=validation_wall,search_plus_selected_validation_wall_s=search_wall+validation_wall,
            neural_generation_s=search.get('neural_generation_s',0) if search else 0.,neural_update_s=search.get('neural_update_s',0) if search else 0.)
        save(output/f'groups/{group}.json',dict(**plan,counts=counts,timings=timings,selected_validation=validation,search_call_ledger=calls,search=search))
        plans.append(plan);print(json.dumps(dict(policy=policy,group=group,feasible=validation['feasible'],calls=len(calls),seconds=search_wall+validation_wall)),flush=True)
    save(output/'airsim_replay_plan.json',dict(schema_version=1,plans=plans))
    if any(sha(p)!=h for p,h in hashes.items()):raise RuntimeError('路线规划过程中源码变化')
    save(output/'planning_summary.json',dict(full_surrogate_calls=total_calls,airsim_calls=0,planning_total_wall_s=time.perf_counter()-started))
    save(output/'completion.json',dict(status='COMPLETE',groups=len(groups),artifact_sha256={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))
    return output/'airsim_replay_plan.json'


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--policy',choices=['stop','pass'],required=True)
    parser.add_argument('--points',required=True,help='JSON array');parser.add_argument('--seed',type=int,default=1201);parser.add_argument('--budget',type=int,default=80)
    a=parser.parse_args();print(generate_route(a.output,json.loads(a.points),a.policy,a.seed,budget=a.budget))
