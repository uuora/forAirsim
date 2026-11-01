"""Persistent isolated protocol16 planner. stdout is JSON lines only."""
import json,sys,time,traceback
from pathlib import Path
import numpy as np
from dual_target_control.online_curve import evaluate
from trajectory_research.search import load_upstream_network,paired_perturbation,PerturbationState,UPSTREAM_COMMIT

ROOT=Path(__file__).resolve().parents[1]


def search(request,torch,network):
    started=time.perf_counter();deadline=request['deadline_host_s'];reserve=.10
    rng=np.random.default_rng(request['seed']);pop=int(request.get('population',8));maximum=int(request.get('max_candidates',32));dim=8
    lower=np.asarray(request['lower']);upper=np.asarray(request['upper']);anchor=np.asarray(request['anchor']);records=[]
    timings=dict(initialization_s=0.,evaluation_s=0.,neural_generation_s=0.,neural_update_s=0.,validation_s=0.)
    settings=request.get('search_settings',{});dropout=float(settings.get('dropout',.95));fraction=float(settings.get('paired_fraction',.5));scale=float(settings.get('paired_scale',.15));lr=float(settings.get('learning_rate',1e-3))
    if pop<4 or pop%2 or not 0<=fraction<=1 or not 0<scale<=1 or not 0<=dropout<1 or lr<=0:raise ValueError('无效的搜索超参数')
    if np.any(upper<=lower) or np.any(anchor<lower) or np.any(anchor>upper):raise ValueError('锚点或搜索边界无效')
    proposal_mode=settings.get('proposal_mode','network')
    if proposal_mode not in ('network','gaussian'):raise ValueError('未知提案机制')
    gaussian=proposal_mode=='gaussian'
    best=None;model=None;method=request['method'];paired=method.endswith('candidate');state=PerturbationState(scale=scale)
    paired_count=int(pop*fraction)//2*2
    def journal(kind,**data):
        path=request.get('worker_journal_file')
        if path:
            with Path(path).open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(kind=kind,job_id=request['job_id'],host_s=time.perf_counter(),**data),allow_nan=False)+'\n')
    def validate(selected):
        tick=time.perf_counter();journal('VALIDATION_STARTED');value=evaluate(request,selected);timings['validation_s']=time.perf_counter()-tick
        journal('VALIDATION_COMPLETE',parameters=list(selected),value=value,wall_s=timings['validation_s']);return value
    def assess(x,source):
        nonlocal best
        tick=time.perf_counter();physical=lower+(x+1)*.5*(upper-lower);journal('EVALUATION_STARTED',call=len(records),parameters=physical.tolist());value=evaluate(request,physical)
        duration=time.perf_counter()-tick;timings['evaluation_s']+=duration
        records.append(dict(parameters=physical.tolist(),source=source,wall_s=duration,**value))
        journal('EVALUATION_COMPLETE',call=len(records)-1,record=records[-1])
        if value['feasible'] and (best is None or value['score']<best['score']):best=dict(parameters=physical.tolist(),**value)
        return value['score']
    # Even the baseline consumes validation time, and late replies are rejected.
    if method.endswith('ordinary'):
        validation=validate(anchor)
        return dict(parameters=anchor.tolist(),validation=validation,counts=dict(search_calls=0,validation_calls=1,airsim_search_calls=0),timings=timings,wall_s=time.perf_counter()-started,deadline_met=time.perf_counter()<=deadline,ledger=[])
    tick=time.perf_counter()
    generator=torch.Generator(device='cpu').manual_seed(request['seed'])
    if not gaussian:
      with torch.random.fork_rng(devices=[]):
        torch.manual_seed(request['seed']);model=network(1,dim,8,dropout,dropout,dim,'cpu',generator).double()
        for name,p in model.named_parameters():
            if 'weight' in name:torch.nn.init.xavier_uniform_(p)
            elif 'bias' in name:torch.nn.init.constant_(p,0)
      model.train()
    timings['initialization_s']=time.perf_counter()-tick
    parents=rng.uniform(-1,1,(pop,dim));parents[0]=2*(anchor-lower)/(upper-lower)-1;costs=[]
    for i,x in enumerate(parents):
        if time.perf_counter()>deadline-reserve:break
        costs.append(assess(x,'anchor' if i==0 else 'initial'))
    n=len(costs);parents=parents[:n];costs=np.asarray(costs)
    if n:order=np.argsort(costs,kind='stable');parents=parents[order];costs=costs[order]
    generation=0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(request['seed']+271828)
        while n==pop and len(records)<maximum and time.perf_counter()<deadline-reserve:
            tick=time.perf_counter()
            if gaussian:
                # Same elitist population and normalized local scale, independent
                # Gaussian proposals around uniformly sampled retained parents.
                children=np.clip(parents[rng.integers(0,pop,pop)]+rng.normal(0,scale,(pop,dim)),-1,1)
            else:
                output=model(torch.from_numpy(parents.copy()).unsqueeze(0),torch.arange(pop,dtype=torch.float64).view(1,pop,1),generation)
                children=output.detach().clamp(-1,1).squeeze(0).numpy().copy();timings['neural_generation_s']+=time.perf_counter()-tick
                if paired and paired_count:children[-paired_count:]=paired_perturbation(parents,paired_count,rng,state)
            child_costs=[];old_best=float(costs[0])
            for i,x in enumerate(children):
                if len(records)>=maximum or time.perf_counter()>deadline-reserve:break
                child_costs.append(assess(x,'gaussian' if gaussian else 'paired' if paired and paired_count and i>=pop-paired_count else 'network'))
            if not child_costs:break
            points=np.vstack([parents,children[:len(child_costs)]]);values=np.r_[costs,child_costs];order=np.argsort(values,kind='stable')[:pop];parents=points[order];costs=values[order]
            if not gaussian and len(child_costs)==pop and time.perf_counter()<deadline-reserve:
                tick=time.perf_counter();optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
                target=torch.from_numpy(parents[0].copy()).view(1,1,dim).repeat(1,pop,1)
                loss=torch.nn.functional.smooth_l1_loss(output,target,beta=1.,reduction='sum');optimizer.zero_grad();loss.backward();optimizer.step()
                timings['neural_update_s']+=time.perf_counter()-tick
            state.update(float(np.mean(np.asarray(child_costs)<old_best)));generation+=1
    selected=best['parameters'] if best is not None else anchor.tolist()
    validation=validate(selected)
    return dict(parameters=selected,validation=validation,counts=dict(search_calls=len(records),validation_calls=1,airsim_search_calls=0),timings=timings,wall_s=time.perf_counter()-started,deadline_met=time.perf_counter()<=deadline,ledger=records,upstream_commit=UPSTREAM_COMMIT,generations=generation,
        network_state='No network; same elitist population' if gaussian else 'Fresh per horizon; elite/weights retained within horizon; AdamW recreated each generation',identity='Gaussian local elitist search' if gaussian else 'Pinned local ABOM-net adapter, paired half-population candidate' if paired else 'Pinned local ABOM-net adapter')


def main():
    started=time.perf_counter()
    # Warm imports + first AdamW setup before flight jobs. Full cost is reported.
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    network=load_upstream_network(ROOT/'tmp/abom_deploy_probe/upstream')
    warm=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64));torch.optim.AdamW([warm],lr=1e-3)
    print(json.dumps(dict(type='ready',startup_wall_s=time.perf_counter()-started)),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line);result=search(request,torch,network)
            print(json.dumps(dict(type='result',job_id=request['job_id'],epoch=request['epoch'],handoff_s=request['handoff_s'],completed_host_s=time.perf_counter(),result=result),allow_nan=False),flush=True)
        except Exception:
            print(json.dumps(dict(type='error',job_id=locals().get('request',{}).get('job_id'),error=traceback.format_exc())),flush=True)


if __name__=='__main__':main()
