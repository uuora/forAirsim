"""Reproducible obstacle-free target cases and six frozen reference plans.

Optimization uses a lagged point-mass surrogate. AirSim is a separate execution
validation, not the search oracle. Bezier/Hermite share the physical curve space.
"""
import argparse
from dataclasses import asdict
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from trajectory_research.parameterizations import BoundaryStates,curve_from_parameters
from trajectory_research.curve_task import jerk_energy,subdivided_derivative_norm_bound
from trajectory_research.timed_tracking import TrackingCondition,run_timed_tracking
from trajectory_research.screened_search import run_screened_search

ROOT=Path(__file__).resolve().parents[1]
PROXY_CONDITION=TrackingCondition(velocity_lag_s=.9)
PROXY_CONTROLLER=dict(velocity_error_gain=.8,acceleration_feedforward_s=.9)
GROUPS=[f'{representation}_{kind}' for kind in ('ordinary','abom','abom_improved_candidate') for representation in ('bezier','hermite')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def make_case(seed=1101,goal=None):
    rng=np.random.default_rng(seed)
    generated=goal is None
    if goal is None:
        distance=float(rng.uniform(6,10));angle=float(rng.uniform(-.55,.55))
        goal=[distance*math.cos(angle),distance*math.sin(angle),float(rng.uniform(4,8))]
    boundary=asdict(BoundaryStates(p0=(0,0,5),v0=(0,0,0),p1=tuple(goal),v1=(0,0,0)))
    if not np.isfinite(goal).all() or len(goal)!=3 or not (2<=goal[0]<=12 and abs(goal[1])<=6 and 2<=goal[2]<=10):
        raise ValueError('target must satisfy x in [2,12], y in [-6,6], height in [2,10] metres')
    return dict(schema_version=1,scene_id=f'control_targets_seed_{seed}',seed=int(seed),boundary=boundary,
        obstacles=[],purpose='Pure control, rest-to-rest target tracking on the exact 03 Overview map',
        generation='seeded target' if generated else 'explicit target')


def objective(parameters,representation,boundary):
    curve=curve_from_parameters(parameters,family='single_bump',representation=representation,boundary=BoundaryStates(**boundary))
    rollout=run_timed_tracking(curve,PROXY_CONDITION,**PROXY_CONTROLLER)
    m=rollout['metrics']
    limits=[subdivided_derivative_norm_bound(curve,d,depth=3) for d in (1,2,3)]
    errors=[m['max_position_error_m']/.5,m['terminal_position_error_m']/.25,m['terminal_velocity_error_mps']/.5]
    ratios=[v/cap for v,cap in zip(limits,(4.,4.,20.))]+errors
    feasible=all(v<=1 for v in ratios)
    energy=float(jerk_energy(curve));T=curve.duration_s;rmse=m['position_rmse_m']
    quality=.7*T/(T+5)+.1*energy/(energy+30)+.2*rmse/(rmse+.25)
    score=quality if feasible else 1.+sum(max(0,x-1) for x in ratios)
    return dict(parameters=list(map(float,parameters)),score=float(score),feasible=bool(feasible),
        metrics=m,derivative_upper_bounds=limits,scope='First-order lagged point mass; no AirSim calls, body, motor, wind or sensing model')


def generate(output,case,*,budget=80,population=20):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'groups').mkdir()
    if budget<population or population<4:raise ValueError('budget must cover the initial population')
    boundary=case['boundary'];distance=np.linalg.norm(np.asarray(boundary['p1'])-boundary['p0'])
    anchor=[0.,0.,0.,float(min(7.,max(5.5,distance/1.4+1.)))]
    lower=[-.8,-1.4,-.8,3.];upper=[.8,1.4,.8,7.]
    case_sha=hashlib.sha256(json.dumps(case,sort_keys=True).encode()).hexdigest()
    save(output/'case.json',case)
    cfg=dict(kind='PURE_CONTROL_SURROGATE_SEARCH',case=case,seed=case['seed'],budget=budget,population=population,
        lower=lower,upper=upper,anchor=anchor,dropout=.95,initialization='uniform',injection_fraction=.5,
        model_condition=asdict(PROXY_CONDITION),model_controller_extensions=PROXY_CONTROLLER,
        model_revision='matched_velocity_outer_loop_v1',
        proxy_identification_scope='Rounded effective horizontal velocity-API lag from development trace; isotropic first-order approximation, not physical actuator identification',
        derivative_limits=[4,4,20],
        budget_scope='Equal proposal count; wall-time superiority requires a separate time-budget study',
        basis_scope='same degree-six polynomial space, identical physical parameter box and endpoint p/v/a')
    save(output/'config.json',cfg)
    plans=[];started=time.perf_counter()
    snapshot=output/'source_snapshot';snapshot.mkdir()
    sources=[Path(__file__),ROOT/'python/trajectory_research/screened_search.py',ROOT/'python/trajectory_research/search.py',
        ROOT/'python/trajectory_research/parameterizations.py',ROOT/'python/trajectory_research/curves.py',
        ROOT/'python/trajectory_research/timed_tracking.py',ROOT/'python/trajectory_research/curve_task.py',
        ROOT/'tmp/abom_deploy_probe/upstream/abom_net.py']
    source_hashes={str(p):sha(p) for p in sources}
    for i,p in enumerate(sources):(snapshot/f'{i:02}_{p.name}').write_bytes(p.read_bytes())
    for group in GROUPS:
        representation=group.split('_')[0];method=None if group.endswith('_ordinary') else 'abom_perturb' if group.endswith('_candidate') else 'abom'
        calls=[];search_start=time.perf_counter()
        def evaluate(x):
            begin=time.perf_counter();value=objective(x,representation,boundary)
            calls.append(dict(parameters=value['parameters'],score=value['score'],feasible=value['feasible'],wall_s=time.perf_counter()-begin))
            return value
        search=None
        if method:
            search=run_screened_search(evaluate,lower,upper,method=method,seed=case['seed'],population=population,
                budget=budget,anchor=anchor,upstream=ROOT/'tmp/abom_deploy_probe/upstream',dropout=.95,injection_fraction=.5)
        search_wall=time.perf_counter()-search_start if method else 0.
        parameters=search['best_parameters'] if search else anchor
        validation_start=time.perf_counter();selected=objective(parameters,representation,boundary);validation_wall=time.perf_counter()-validation_start
        provenance=dict(kind='stable_retained_incumbent' if method else 'prescribed_anchor',method=method,
            result_json=f'groups/{group}.json',family='single_bump',layout='NO_OBSTACLES',case_sha256=case_sha,
            model_feasible=selected['feasible'],model_score=selected['score'],model_revision='matched_velocity_outer_loop_v1',
            replication_scope='Pinned-network adapter; generic paired perturbation candidate, not full original-paper replication')
        counts=dict(search_proposals=len(calls),search_objective_calls=len(calls),selected_validation_calls=1,total_objective_calls=len(calls)+1)
        timings=dict(search_wall_s=search_wall,selected_validation_wall_s=validation_wall,
            search_plus_selected_validation_wall_s=search_wall+validation_wall,
            neural_generation_s=search['neural_generation_s'] if search else 0.,
            neural_update_s=search['neural_update_s'] if search else 0.)
        plan=dict(group_id=group,representation=representation,family='single_bump',parameters=parameters,
            seed=case['seed'],boundary=boundary,selection_provenance=provenance)
        record=dict(**plan,counts=counts,timings=timings,selected_validation=selected,search=search,search_call_ledger=calls)
        save(output/provenance['result_json'],record);plans.append(plan)
        print(json.dumps(dict(group_id=group,score=selected['score'],model_feasible=selected['feasible'],calls=len(calls),planning_wall_s=search_wall+validation_wall)),flush=True)
    save(output/'airsim_replay_plan.json',dict(schema_version=1,plans=plans))
    if any(sha(p)!=h for p,h in source_hashes.items()):raise RuntimeError('source changed while planning')
    save(output/'planning_summary.json',dict(planning_total_wall_s=time.perf_counter()-started,source_sha256=source_hashes,
        full_surrogate_calls=sum(json.loads((output/f'groups/{g}.json').read_text())['counts']['total_objective_calls'] for g in GROUPS),
        airsim_calls=0,case_sha256=case_sha,ai_assistance=True))
    hashes={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}
    save(output/'completion.json',dict(status='COMPLETE',groups=6,artifact_sha256=hashes))
    return output/'airsim_replay_plan.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=1101)
    parser.add_argument('--goal',type=float,nargs=3);parser.add_argument('--case',type=Path);parser.add_argument('--budget',type=int,default=80)
    args=parser.parse_args();case=json.loads(args.case.read_text(encoding='utf-8')) if args.case else make_case(args.seed,args.goal)
    print(generate(args.output,case,budget=args.budget))
